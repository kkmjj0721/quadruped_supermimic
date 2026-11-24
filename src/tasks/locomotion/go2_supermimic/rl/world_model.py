# Copyright (c) 2026-2027 zh
"""内容：分离Go2残差输出与物理增量统计，保留通用自回归监督和接触概率语义。"""

import torch
from torch.nn import functional as F

from agents.supermimic.world_model import RunningNorm, WorldModel

from ..mdp.reference_state import CONTACT_EVENTS, GENERALIZED_DELTA_DIM, MODEL_CONTACT, MODEL_EVENTS, MODEL_HISTORY_CONTACT


class Go2WorldModel(WorldModel):
    """BlinkMimic的残差／增量双统计；接触logit使用单位尺度。"""

    def __init__(self, task, hidden):
        super().__init__(task, hidden)
        # 与来源残差模型一致，初始输出为归一化目标均值。
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.increment_norm = RunningNorm(task.spec.delta_dim)
        # 来源BlinkMimic d264658 train.py ERR_FLOOR；顺序为p、旋转、v、w、q、qd。
        self.register_buffer("core_error_floor", torch.tensor(
            [3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3 + [3e-3] * 12 + [5e-2] * 12))

    @torch.no_grad()
    def prepare_statistics(self, state, action, target, mask=None):
        return (*super().prepare_statistics(state, action, target, mask),
                self.increment_norm.prepare_update(self.task.model_increment(state, target), mask))

    @torch.no_grad()
    def apply_statistics(self, statistics):
        super().apply_statistics(statistics)
        self.increment_norm.apply_update(statistics[2])
        active = statistics[1].count > 0
        for norm in (self.output_norm, self.increment_norm):
            for region in (MODEL_CONTACT, MODEL_HISTORY_CONTACT, MODEL_EVENTS):
                mean, variance = norm.mean[region], norm.var[region]
                mean.copy_(torch.where(active, 0, mean))
                variance.copy_(torch.where(active, 1, variance))

    def loss_scale(self):
        # 核心物理误差使用参考下限；任务辅助量继续采用原有标准差尺度。
        scale = self.increment_norm.var.sqrt() + 1e-6
        core = torch.maximum((self.increment_norm.var[:GENERALIZED_DELTA_DIM] + 1e-8).sqrt(), self.core_error_floor)
        return torch.cat((core, scale[GENERALIZED_DELTA_DIM:]), -1)

    def supervised_loss(self, current, predicted, target, recorded_current, delta, scale):
        """物理量保留原97维MSE，新增五项真实事件以未加类别权重的BCE监督。"""
        error = self.task.model_error(current, predicted, target, recorded_current)[:, :MODEL_EVENTS.start]
        regression = (error / scale[:MODEL_EVENTS.start]).square().mean(-1)
        events = F.binary_cross_entropy_with_logits(
            delta[:, MODEL_EVENTS], target.physical[:, CONTACT_EVENTS], reduction="none").mean(-1)
        return regression + events
