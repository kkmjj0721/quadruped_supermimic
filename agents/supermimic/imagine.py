# Copyright (c) 2026-2027 zh
"""内容：编译可复用想象单步与完整返回递推，保留末端价值导数及固定目标。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .compilation import compile_kernel
from .interfaces import TaskState, append_history


@dataclass
class ImaginedData:
    objective: Tensor
    rewards: Tensor
    failures: Tensor
    features: Tensor
    survival: Tensor
    discounted_reward: Tensor
    bootstrap: Tensor
    unsupported_touchdowns: Tensor
    initial_velocity: Tensor


class Imagination:
    def __init__(self, task, actor, model, critic, config):
        self.task, self.actor, self.model, self.critic, self.config = task, actor, model, critic, config
        self.step = (compile_kernel(self._step, config.compile_backend)
                     if config.compile_imagination else self._step)
        self._return_kernel = (compile_kernel(self._lambda_returns, config.compile_backend)
                               if config.compile_imagination else self._lambda_returns)

    def _step(self, state: TaskState, history: Tensor):
        action, velocity = self.actor.action_and_velocity(
            history, self.task.actor_heightmap(state), self.task.actor_mask(state))
        predicted = self.model(state, action)
        failure = self.task.soft_termination(predicted)
        reward = self.task.reward(state, action, predicted, failure)
        with torch.no_grad():
            cut = self.task.unsupported_touchdown(state, predicted)
        frame = self.task.observe(predicted, noisy=True)
        return predicted, append_history(history, frame), reward, failure, self.task.privileged(predicted), cut, velocity

    def rollout(self, seed) -> ImaginedData:
        cfg = self.config
        state, history = seed.state, seed.history
        initial_feature = self.task.privileged(state).detach()
        rewards, failures, features, cuts = [], [], [], []
        for step in range(cfg.horizon):
            state, history, reward, failure, feature, cut, velocity = self.step(state, history)
            if step == 0:
                initial_velocity = velocity
            rewards.append(reward)
            failures.append(failure)
            features.append(feature)
            cuts.append(cut)
        features = torch.stack(features, dim=1)
        values = self.critic(features)
        # 末步或支撑不足的落脚步保留即时奖励，后续价值由当前落脚后的状态给出。
        discounted = rewards[-1]
        bootstrap = cfg.gamma * (1 - failures[-1]) * values[:, -1]
        for step in reversed(range(cfg.horizon - 1)):
            continuation = cfg.gamma * (1 - failures[step])
            discounted = rewards[step] + continuation * torch.where(cuts[step], 0., discounted)
            bootstrap = continuation * torch.where(cuts[step], values[:, step], bootstrap)
        failures = torch.stack(failures, dim=1).detach()
        critic_features = torch.cat((initial_feature[:, None], features.detach()), dim=1)
        return ImaginedData(discounted + bootstrap, torch.stack(rewards, dim=1).detach(), failures,
                            critic_features, (1 - failures).prod(dim=1), discounted, bootstrap,
                            torch.stack(cuts, dim=1), initial_velocity)

    @torch.no_grad()
    def lambda_returns(self, data: ImaginedData) -> Tensor:
        return self._return_kernel(data)

    @torch.no_grad()
    def _lambda_returns(self, data: ImaginedData) -> Tensor:
        """按 TGI 的逐步生存率生成价值目标；落脚截断只影响策略目标。"""
        values = self.critic(data.features)
        targets = torch.empty_like(data.rewards)
        tail = values[:, -1]
        for step in reversed(range(self.config.horizon)):
            continuation = 1 - data.failures[:, step]
            tail = data.rewards[:, step] + self.config.gamma * continuation * (
                (1 - self.config.lam) * values[:, step + 1] + self.config.lam * tail)
            targets[:, step] = tail
        return targets

