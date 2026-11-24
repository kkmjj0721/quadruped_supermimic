# Copyright (c) 2026-2027 zh
"""内容：提供PROOF教师的本体／高程双编码策略、速度估计及价值网络。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .world_model import RunningNorm


# ---------- 基础网络与观测归一化 ----------

class Mlp(nn.Module):
    """使用ELU隐藏层和线性输出层，编码器可在激活前加入LayerNorm。"""

    def __init__(self, dims, layer_norm=False):
        super().__init__()
        layers, width = [], dims[0]
        for hidden_width in dims[1:-1]:
            layers.append(nn.Linear(width, hidden_width))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_width))
            layers.append(nn.ELU())
            width = hidden_width
        layers.append(nn.Linear(width, dims[-1]))
        self.network = nn.Sequential(*layers)
        self.use_bf16 = False

    def initialize_head(self):
        """只对无LayerNorm的动作／价值主干应用作者的正交初始化。"""
        for layer in self.network[::2]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, values: Tensor) -> Tensor:
        if self.use_bf16:
            # 恢复FP32后再执行潜变量融合、mask与任务计算。
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                outputs = self.network(values)
            return outputs.float()
        return self.network(values)


class ObservationNorm(RunningNorm):
    """采用Z_RL的零初始计数及总体标准差加1e-2，不裁剪观测。"""

    def __init__(self, width: int):
        super().__init__(width)
        self.count.zero_()

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean) / (self.var.sqrt() + 1e-2)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * (self.var.sqrt() + 1e-2) + self.mean


# ---------- 教师策略 ----------

class ProofTeacherActor(nn.Module):
    """独立编码十帧本体与当前高程，在编码后按回合mask关闭外部感知。"""

    def __init__(self, spec, config, observation_term_dims):
        super().__init__()
        self.observation_dim = spec.observation_dim
        self.observation_term_dims = tuple(observation_term_dims)
        self.history_length = spec.history_length
        self.heightmap_dim = spec.heightmap_dim
        self.action_dim = spec.action_dim
        self.prop_latent_dim = config.prop_latent_dim
        self.height_latent_dim = config.height_latent_dim
        self.prop_encoder_hidden = tuple(config.prop_encoder_hidden)
        self.height_encoder_hidden = tuple(config.height_encoder_hidden)
        self.velocity_dim = 3
        self.history_order = "frame_major_oldest_first"
        self.encoder_history_order = "term_major_oldest_first"
        self.input_order = "proprioceptive_latent_normalized_current_frame_masked_height_latent"
        self.actor_input_dim = self.prop_latent_dim + self.observation_dim + self.height_latent_dim
        current_indices, offset = [], 0
        for width in self.observation_term_dims:
            current_indices.extend(range(offset + (self.history_length - 1) * width, offset + self.history_length * width))
            offset += self.history_length * width
        self.register_buffer("current_frame_indices", torch.tensor(current_indices, dtype=torch.long))
        self.normalizer = ObservationNorm(self.observation_dim * self.history_length)
        self.prop_encoder = Mlp([self.observation_dim * self.history_length, *self.prop_encoder_hidden,
                                 self.prop_latent_dim], layer_norm=True)
        self.height_encoder = Mlp([self.heightmap_dim, *self.height_encoder_hidden, self.height_latent_dim],
                                   layer_norm=True)
        self.actor = Mlp([self.actor_input_dim, *config.actor_hidden, self.action_dim])
        self.actor.initialize_head()

    def policy_normalizers(self):
        """导出与播放校验需要检查的归一化模块。"""
        return (self.normalizer,)

    def policy_parameters(self):
        """想象目标与真实速度监督共同更新完整教师策略。"""
        return self.parameters()

    def flatten_history(self, history: Tensor) -> Tensor:
        """外部逐帧历史转为作者逐观测项旧到新的450维编码布局。"""
        framed = history.reshape(-1, self.history_length, self.observation_dim)
        return torch.cat([term.flatten(1) for term in framed.split(self.observation_term_dims, dim=-1)], dim=-1)

    @torch.no_grad()
    def prepare_statistics(self, history: Tensor):
        """仅更新本体历史统计，高程与mask不作输入归一化。"""
        return (self.normalizer.prepare_update(self.flatten_history(history)),)

    @torch.no_grad()
    def apply_statistics(self, statistics):
        """提交本控制步已经统一校验的本体历史统计。"""
        self.normalizer.apply_update(statistics[0])

    @torch.no_grad()
    def update_statistics(self, history: Tensor):
        """独立入口即时校验；采集热路径通过准备／提交合并检查。"""
        self.normalizer.update(self.flatten_history(history))

    @torch.jit.export
    def encode(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """本体latent的前三维直接估计同一时刻机身xyz线速度。"""
        latent = self.prop_encoder(self.normalizer(self.flatten_history(history)))
        return latent, latent[:, :self.velocity_dim]

    def forward(self, history: Tensor, heightmap: Tensor, mask: Tensor) -> Tensor:
        """当前帧从同一归一化历史取出，高程latent经tanh后再乘mask。"""
        normalized = self.normalizer(self.flatten_history(history))
        latent = self.prop_encoder(normalized)
        current = normalized.index_select(-1, self.current_frame_indices)
        height = self.height_encoder(heightmap).tanh() * mask
        return self.actor(torch.cat((latent, current, height), -1))


# ---------- 装配与价值网络 ----------

def make_actor(spec, config, observation_term_dims):
    """装配唯一的PROOF高程图教师策略。"""
    return ProofTeacherActor(spec, config, observation_term_dims)


class Critic(nn.Module):
    def __init__(self, width: int, hidden):
        super().__init__()
        self.normalizer = ObservationNorm(width)
        self.net = Mlp([width, *hidden, 1])
        self.net.initialize_head()

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)
