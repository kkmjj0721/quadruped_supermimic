# Copyright (c) 2026-2027 zh
"""
内容：
    定义高程图教师世界模型训练的状态、维度及真实转移协议。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlgorithmSpec:
    """维度之外，semantic_id必须标识坐标系、布局、动作及任务语义。"""

    state_dim: int
    delta_dim: int
    action_dim: int
    observation_dim: int
    privileged_dim: int
    context_dim: int
    memory_dim: int
    command_dim: int
    semantic_id: str
    heightmap_dim: int
    history_length: int = 10
    model_id: str = "supermimic_proof_teacher_v2"

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(init=False)
class TaskState:
    """四类状态共用一块存储，各公开字段为末轴视图。"""

    data: Tensor
    widths: tuple[int, int, int, int]

    def __init__(self, physical: Tensor, memory: Tensor, command: Tensor, context: Tensor):
        self.widths = (physical.shape[-1], memory.shape[-1], command.shape[-1], context.shape[-1])
        self.data = torch.cat((physical, memory, command, context), dim=-1)

    @classmethod
    def from_flat(cls, data: Tensor, widths: tuple[int, int, int, int]) -> TaskState:
        """复用已打包数据；调用方提供现行四段宽度契约。"""
        state = cls.__new__(cls)
        state.data, state.widths = data, widths
        return state

    @property
    def physical(self) -> Tensor:
        return self.data[..., :self.widths[0]]

    @property
    def memory(self) -> Tensor:
        return self.data[..., self.widths[0]:self.widths[0] + self.widths[1]]

    @property
    def command(self) -> Tensor:
        start = self.widths[0] + self.widths[1]
        return self.data[..., start:start + self.widths[2]]

    @property
    def context(self) -> Tensor:
        return self.data[..., self.widths[0] + self.widths[1] + self.widths[2]:]

    def map(self, function) -> TaskState:
        return TaskState.from_flat(function(self.data), self.widths)

    def index(self, index) -> TaskState:
        return self.map(lambda tensor: tensor[index])

    def detached_clone(self) -> TaskState:
        return self.map(lambda tensor: tensor.detach().clone())


@dataclass
class EnvStep:
    """adapter先复制final_state，再reset；next_state为下一决策态。"""

    final_state: TaskState
    next_state: TaskState
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    context_version: Tensor


def append_history(history: Tensor, observation: Tensor) -> Tensor:
    """无原地修改地推进历史，保留想象中的跨步梯度。

    真实回放要求连续插入：非结束环境沿用此推进，结束后下一决策历史
    重复同一个新回合带噪首帧；清空回放后的首次插入提供完整历史前缀。
    """
    return torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
