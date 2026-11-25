# Copyright (c) 2026-2027 zh
"""内容：按四步编译可微想象及完整返回递推，保留末端价值导数及固定目标。"""

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
    survivals: Tensor
    initial_velocity: Tensor


class Imagination:
    def __init__(self, task, actor, model, critic, config):
        self.task, self.actor, self.model, self.critic, self.config = task, actor, model, critic, config
        self._discounts = torch.tensor([config.gamma ** step for step in range(config.horizon)],
                                      device=next(model.parameters()).device)
        self._chunk_kernel = (compile_kernel(self._chunk, config.compile_backend)
                              if config.compile_imagination else self._chunk)
        self._return_kernel = (compile_kernel(self._lambda_returns, config.compile_backend)
                               if config.compile_imagination else self._lambda_returns)

    def _step(self, state: TaskState, history: Tensor):
        action, velocity = self.actor.action_and_velocity(
            history, self.task.actor_heightmap(state), self.task.actor_mask(state))
        with torch.no_grad():
            features = self.task.privileged(state)
        predicted = self.model(state, action)
        failure = self.task.soft_termination(predicted)
        reward = self.task.reward(state, action, predicted, failure)
        frame = self.task.observe(predicted, noisy=True)
        return predicted, append_history(history, frame), reward, failure, features, velocity

    def _chunk(self, state, history, survival, discounted, discounts):
        """固定四步在同一编译边界内推进，尾段按实际步数专门化。"""
        rewards, failures, features, survivals = [], [], [], []
        for step in range(discounts.shape[0]):
            survivals.append(survival)
            state, history, reward, failure, feature, velocity = self._step(state, history)
            if step == 0:
                initial_velocity = velocity
            discounted = discounted + discounts[step] * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            features.append(feature.detach())
        return (state, history, survival, discounted, torch.stack(rewards, dim=1),
                torch.stack(failures, dim=1), torch.stack(features, dim=1),
                torch.stack(survivals, dim=1).detach(), initial_velocity)

    def rollout(self, seed) -> ImaginedData:
        cfg = self.config
        state, history = seed.state, seed.history
        survival = torch.ones(history.shape[0], device=history.device)
        discounted = torch.zeros_like(survival)
        rewards, failures, features, survivals = [], [], [], []
        for start in range(0, cfg.horizon, 4):
            (state, history, survival, discounted, chunk_rewards, chunk_failures,
             chunk_features, chunk_survivals, velocity) = self._chunk_kernel(
                 state, history, survival, discounted, self._discounts[start:start + 4])
            if start == 0:
                initial_velocity = velocity
            rewards.append(chunk_rewards)
            failures.append(chunk_failures)
            features.append(chunk_features)
            survivals.append(chunk_survivals)
        endpoint = self.task.privileged(state)
        bootstrap = cfg.gamma ** cfg.horizon * survival * self.critic(endpoint)
        features.append(endpoint.detach()[:, None])
        survivals.append(survival.detach()[:, None])
        return ImaginedData(discounted + bootstrap, torch.cat(rewards, dim=1), torch.cat(failures, dim=1),
                            torch.cat(features, dim=1), survival, discounted, bootstrap,
                            torch.cat(survivals, dim=1), initial_velocity)

    @torch.no_grad()
    def lambda_returns(self, data: ImaginedData) -> Tensor:
        return self._return_kernel(data)

    @torch.no_grad()
    def _lambda_returns(self, data: ImaginedData) -> Tensor:
        """固定价值快照，按本步失败概率递推，到达结束步的奖励保留一次。"""
        values = self.critic(data.features)
        targets = torch.empty_like(data.rewards)
        tail = values[:, -1]
        for step in reversed(range(self.config.horizon)):
            continuation = 1 - data.failures[:, step]
            tail = data.rewards[:, step] + self.config.gamma * continuation * (
                (1 - self.config.lam) * values[:, step + 1] + self.config.lam * tail)
            targets[:, step] = tail
        return targets

