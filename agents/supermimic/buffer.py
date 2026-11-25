# Copyright (c) 2026-2027 zh
"""内容：保存真实转移及带噪历史，支持单帧存储重建，并构造终止掩码与稀有事件归档。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .interfaces import TaskState


@dataclass
class Seed:
    state: TaskState
    history: Tensor

    def index(self, index):
        """分批保留种子协议；带外感受状态的种子覆写此方法。"""
        return Seed(self.state.index(index), self.history[index])


@dataclass
class Window:
    """批次优先；目标取下一slot，终止当步的reset后状态仅供参考稀有事件判定。"""

    initial: TaskState
    actions: Tensor
    targets: TaskState
    valid: Tensor
    terminated: Tensor

    def index(self, index):
        return Window(self.initial.index(index), self.actions[index], self.targets.index(index),
                      self.valid[index], self.terminated[index])

    @staticmethod
    def concatenate(windows):
        initial = TaskState.from_flat(torch.cat([window.initial.data for window in windows]), windows[0].initial.widths)
        targets = TaskState.from_flat(torch.cat([window.targets.data for window in windows]), windows[0].targets.widths)
        return Window(initial, torch.cat([window.actions for window in windows]), targets,
                      torch.cat([window.valid for window in windows]), torch.cat([window.terminated for window in windows]))


class Replay:
    """按环境、时间顺序保存连续状态和半精度历史，可独立采样所有已采集时刻。

    frames模式要求逐控制步连续插入：新历史移动一帧，结束后重复新回合首帧。
    首次插入和clear后的首次插入均保留调用方传入的完整历史前缀。
    设备索引用于已就绪回放的图捕获；clear后调用方须重新填充再重放采样图。
    """

    def __init__(self, spec, num_envs: int, capacity: int, device, history_storage="full", *, device_sampling=False):
        self.spec, self.num_envs, self.capacity = spec, num_envs, capacity
        self.device = torch.device(device)
        self.history_storage = history_storage
        self.device_sampling = device_sampling
        self.size = self.clock = 0
        if device_sampling:
            # 图重放读取稳定地址的当前边界；主机计数仍供写入和就绪检查使用。
            self.size_device = torch.zeros((), dtype=torch.long, device=device)
            self.clock_device = torch.zeros_like(self.size_device)
        self.states = self._state_storage((num_envs, capacity))
        self.actions = torch.zeros(num_envs, capacity, spec.action_dim, device=device)
        if history_storage == "full":
            self.histories = torch.zeros(num_envs, capacity, spec.history_length, spec.observation_dim,
                                         device=device, dtype=torch.float16)
        else:
            self.frame_capacity = capacity + spec.history_length - 1
            self.observation_frames = torch.zeros(num_envs, self.frame_capacity, spec.observation_dim,
                                                  device=device, dtype=torch.float16)
            self.history_depth = torch.zeros(num_envs, capacity, device=device, dtype=torch.int32)
            self.next_history_depth = torch.full((num_envs,), spec.history_length, device=device, dtype=torch.int32)
            self.history_offsets = torch.arange(spec.history_length - 1, -1, -1, device=device)
        self.done = torch.zeros(num_envs, capacity, device=device, dtype=torch.bool)
        self.terminated = torch.zeros_like(self.done)

    def _state_storage(self, shape):
        widths = (self.spec.state_dim, self.spec.memory_dim, self.spec.command_dim, self.spec.context_dim)
        return TaskState.from_flat(torch.zeros(*shape, sum(widths), device=self.device), widths)

    def clear(self):
        self.size = self.clock = 0
        if self.device_sampling:
            self.size_device.zero_()
            self.clock_device.zero_()
        if self.history_storage == "frames":
            self.history_depth.zero_()
            self.next_history_depth.fill_(self.spec.history_length)

    @torch.no_grad()
    def add(self, state: TaskState, history: Tensor, action: Tensor, result):
        slot = self.clock % self.capacity
        self.states.data[:, slot].copy_(state.data)
        self.actions[:, slot].copy_(action)
        self.done[:, slot].copy_(result.terminated | result.truncated)
        self.terminated[:, slot].copy_(result.terminated)
        if self.history_storage == "full":
            self.histories[:, slot].copy_(history)
        else:
            if self.clock == 0:
                # 虚拟时刻-(H-1)..-1恰好对应观测环末尾的H-1个槽。
                self.observation_frames[:, self.capacity:].copy_(history[:, :-1])
            self.observation_frames[:, self.clock % self.frame_capacity].copy_(history[:, -1])
            self.history_depth[:, slot].copy_(self.next_history_depth)
            # 结束标志属于本次动作，只限制下一决策态的历史回退深度。
            self.next_history_depth.copy_(torch.where(
                self.done[:, slot], 1, (self.next_history_depth + 1).clamp_max(self.spec.history_length)))
        self.clock += 1
        self.size = min(self.size + 1, self.capacity)
        if self.device_sampling:
            self.clock_device.add_(1)
            self.size_device.add_(1).clamp_max_(self.capacity)

    def ready(self, horizon: int) -> bool:
        return self.size > horizon + 1

    def _sample_positions(self, batch, horizon, device_indices):
        rows = torch.randint(self.num_envs, (batch,), device=self.device)
        if device_indices:
            # 上界随每次重放变化；FP64量化保持近似均匀分布，随机流不同于randint。
            bound = self.size_device - horizon
            logical = (torch.rand(batch, dtype=torch.float64, device=self.device) * bound).long()
            logical = logical.clamp_max(bound - 1)
            absolute = self.clock_device - self.size_device + logical
        else:
            logical = (torch.randint(0, self.size - horizon, (batch,), device=self.device) if horizon
                       else torch.randint(self.size, (batch,), device=self.device))
            absolute = self.clock - self.size + logical
        return rows, absolute

    def _sample_indices(self, batch: int, horizon: int, *, device_indices=False):
        rows, absolute = self._sample_positions(batch, horizon, device_indices)
        return rows, absolute % self.capacity

    def sample_seeds(self, batch: int, *, device_indices=False) -> Seed:
        rows, absolute = self._sample_positions(batch, 0, device_indices)
        slots = absolute % self.capacity
        if self.history_storage == "full":
            history = self.histories[rows, slots].float()
        else:
            depth = self.history_depth[rows, slots]
            back = torch.minimum(self.history_offsets[None, :], depth[:, None] - 1)
            frame_slots = (absolute[:, None] - back) % self.frame_capacity
            history = self.observation_frames[rows[:, None], frame_slots].float()
        return Seed(self.states.index((rows, slots)), history)

    def sample_windows(self, batch: int, horizon: int, *, device_indices=False) -> Window:
        rows, starts = self.sample_window_indices(batch, horizon, device_indices=device_indices)
        return self.gather_windows(rows, starts, horizon)

    def sample_window_indices(self, batch: int, horizon: int, *, device_indices=False):
        """仅消费窗口采样随机数，延后决定是否读取回放数据。"""
        return self._sample_indices(batch, horizon, device_indices=device_indices)

    def gather_windows(self, rows: Tensor, starts: Tensor, horizon: int) -> Window:
        """按既定环境与时间索引读取窗口，不再消费随机数。"""
        slots = (starts[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        done = self.done[rows[:, None], slots]
        valid = (~done).cumprod(dim=1).bool()
        return Window(self.states.index((rows, starts)), self.actions[rows[:, None], slots],
                      self.states.index((rows[:, None], (slots + 1) % self.capacity)), valid,
                      self.terminated[rows[:, None], slots])


class WindowArchive:
    """固定形状候选按掩码精确写入；无效候选不读取或写入窗口数据。"""

    def __init__(self, capacity: int, device="cpu"):
        self.capacity = capacity
        self.size = torch.zeros((), dtype=torch.long, device=device)
        self.clock = torch.zeros_like(self.size)
        self.storage = None

    def clear(self):
        self.size.zero_()
        self.clock.zero_()

    @torch.no_grad()
    def add(self, windows: Window, keep: Tensor | None = None):
        candidates = windows.actions.shape[0]
        device = windows.actions.device
        if candidates == 0 or self.capacity == 0:
            return
        if self.storage is None:
            # 首批候选即分配，不读取有效数量；候选批次大小由调用方固定。
            def allocate(value):
                return torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=device)

            self.storage = Window(windows.initial.map(allocate), allocate(windows.actions), windows.targets.map(allocate),
                                  allocate(windows.valid), allocate(windows.terminated))
            self.size, self.clock = self.size.to(device), self.clock.to(device)
        keep = torch.ones(candidates, dtype=torch.bool, device=device) if keep is None else keep
        rank = keep.long().cumsum(0) - 1
        accepted = keep & (rank < self.capacity)
        indices = (self.clock + rank) % self.capacity
        if device.type == "cuda":
            from .archive_kernels import write_windows

            write_windows(self.storage, windows, accepted, indices)
        else:
            selected = indices[accepted]
            self.storage.initial.data[selected] = windows.initial.data[accepted]
            self.storage.targets.data[selected] = windows.targets.data[accepted]
            self.storage.actions[selected] = windows.actions[accepted]
            self.storage.valid[selected] = windows.valid[accepted]
            self.storage.terminated[selected] = windows.terminated[accepted]
        count = accepted.sum()
        self.clock.add_(count)
        self.size.add_(count).clamp_max_(self.capacity)

    def _sample_indices(self, batch):
        # 设备上界避免提取size；FP64随机数量化误差远小于FP32，随机流不同于randint。
        bound = self.size.clamp_min(1)
        indices = (torch.rand(batch, dtype=torch.float64, device=self.size.device) * bound).long()
        return indices.clamp_max(bound - 1)

    def _sample(self, batch):
        return self.storage.index(self._sample_indices(batch))

    def sample(self, batch: int) -> Window:
        return self._sample(batch)

    def sample_or_replay(self, replay: Replay, rows: Tensor, starts: Tensor, horizon: int, populated: Tensor) -> Window:
        """沿用写入前占用与随机数，仅执行实际选中的归档或回放读取。"""
        if self.storage is None:
            return replay.gather_windows(rows, starts, horizon)
        indices = self._sample_indices(rows.shape[0])
        if self.size.device.type == "cuda":
            from .archive_kernels import sample_archive_or_replay

            return sample_archive_or_replay(self.storage, replay, rows, starts, indices, horizon, populated)
        return self.storage.index(indices) if bool(populated) else replay.gather_windows(rows, starts, horizon)
