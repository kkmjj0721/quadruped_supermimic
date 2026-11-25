# Copyright (c) 2026-2027 zh
"""内容：提供统计归一化和只学习记录物理转移的动力学网络。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .compilation import compile_kernel
from .interfaces import TaskState


class RunningNorm(nn.Module):
    """按参考实现合并总体方差；前向固定统计，不隐式更新。"""

    def __init__(self, width: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))
        self.register_buffer("std", torch.full((width,), 1 + epsilon), persistent=False)
        self.register_buffer("inv_std", self.std.reciprocal(), persistent=False)

    @torch.no_grad()
    def refresh_scale(self):
        """统计更新或恢复后刷新派生尺度，不增加检查点的必需字段。"""
        self.std.copy_(self.var.sqrt() + self.epsilon)
        self.inv_std.copy_(self.std.reciprocal())

    def _apply(self, function, recurse=True):
        super()._apply(function, recurse=recurse)
        self.refresh_scale()
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.refresh_scale()

    @torch.no_grad()
    def update(self, values: Tensor, mask: Tensor | None = None):
        """合并本批总体统计，掩码全部关闭时保留已有统计。"""
        values = values.reshape(-1, self.mean.numel())
        if mask is None:
            variance, mean = torch.var_mean(values, dim=0, correction=0)
            count = self.count.new_full((), values.shape[0])
        else:
            count = mask.sum().to(self.count.dtype)
            safe = torch.where(mask[:, None], values, 0)
            mean = safe.sum(dim=0) / count.clamp_min(1)
            centered = torch.where(mask[:, None], values - mean, 0)
            variance = centered.square().sum(dim=0) / count.clamp_min(1)
        total = self.count + count
        difference = mean - self.mean
        updated_var = (self.var * self.count + variance * count + difference.square() * self.count * count / total) / total
        updated_mean = self.mean + difference * count / total
        active = count > 0
        self.var.copy_(torch.where(active, updated_var, self.var))
        self.mean.copy_(torch.where(active, updated_mean, self.mean))
        self.count.copy_(torch.where(active, total, self.count))
        self.refresh_scale()

    def forward(self, values: Tensor) -> Tensor:
        return ((values - self.mean) * self.inv_std).clamp(-10.0, 10.0)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * self.std + self.mean


class WorldModel(nn.Module):
    """任务决定残差坐标；网络只学习记录物理转移，不学习奖励或失败头。"""

    def __init__(self, task, hidden):
        super().__init__()
        self.task = task
        self.spec = task.spec
        layers, width = [], task.model_input_dim
        for hidden_width in hidden:
            layers.extend((nn.Linear(width, hidden_width), nn.SiLU()))
            width = hidden_width
        layers.append(nn.Linear(width, task.spec.delta_dim))
        self.net = nn.Sequential(*layers)
        self.input_norm = RunningNorm(task.model_input_dim)
        self.output_norm = RunningNorm(task.spec.delta_dim)
        self.use_bf16 = False
        self._supervised_step_kernel = self._supervised_step

    def configure_supervised_compile(self, *, enabled: bool, backend: str):
        """只编译可复用的单步预测与监督，外层保留自回归和随机数顺序。"""
        self._supervised_step_kernel = compile_kernel(self._supervised_step, backend) if enabled else self._supervised_step

    @torch.no_grad()
    def update_stats(self, state: TaskState, action: Tensor, target: TaskState, mask: Tensor | None = None):
        self.input_norm.update(self.task.model_input(state, action), mask)
        self.output_norm.update(self.task.model_target(state, target), mask)

    def forward(self, state: TaskState, action: Tensor) -> TaskState:
        return self.predict(state, action)[0]

    def predict(self, state: TaskState, action: Tensor) -> tuple[TaskState, Tensor]:
        """共享一次网络前向，监督同时取得推进状态与未施加输出链接的原始预测。"""
        normalized = self.input_norm(self.task.model_input(state, action))
        if self.use_bf16:
            # 仅网络参与混合精度；反归一化及任务状态推进保持 FP32。
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                network_delta = self.net(normalized)
            network_delta = network_delta.float()
        else:
            network_delta = self.net(normalized)
        delta = self.output_norm.unnormalize(network_delta)
        physical = self.task.integrate(state, action, delta)
        # 监督推进仍消费命令采样随机数；单步编译返回完整状态以保留这些副作用。
        return self.task.advance(state, action, physical), delta

    def loss(self, window) -> Tensor:
        return self._loss(window)

    def loss_scale(self) -> Tensor:
        """默认沿用输出统计；残差模型可独立定义物理状态误差尺度。"""
        return self.output_norm.std

    def supervised_loss(self, predicted, target, delta, scale) -> Tensor:
        """默认监督标准化物理误差；任务子类可定义其分类输出的损失。"""
        del delta
        error = self.task.model_error(predicted, target) / scale
        return error.square().mean(-1)

    def _supervised_step(self, current, action, target, scale):
        """返回完整推进状态和逐样本损失，使区域编译保留advance的随机数消费。"""
        predicted, delta = self.predict(current, action)
        loss = self.supervised_loss(predicted, target, delta, scale)
        return predicted, loss

    def _loss(self, window) -> Tensor:
        current = window.initial
        scale = self.loss_scale()
        total = current.physical.new_zeros(())
        for step in range(window.actions.shape[1]):
            target = window.targets.index((slice(None), step))
            action = window.actions[:, step]
            predicted, loss = self._supervised_step_kernel(current, action, target, scale)
            total = total + (loss * window.valid[:, step]).sum()
            # 本协议的记忆只含记录动作与外生计时，按日志恢复命令重采样；物理状态始终回喂预测。
            current = TaskState(predicted.physical, target.memory, target.command, target.context)
        return total / window.valid.sum().clamp_min(1)
