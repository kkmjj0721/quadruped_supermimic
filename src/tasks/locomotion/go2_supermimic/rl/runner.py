# Copyright (c) 2026-2027 zh
"""内容：装配 Go2 SuperMimic 训练器，持久化课程与任务配置快照。"""

from dataclasses import fields
import json
from pathlib import Path
import time

import torch

from agents.supermimic.config import ExecutionConfig
from agents.supermimic.runner import SuperMimicRunner
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.loading import algorithm_config
from .transition_adapter import make_go2_supermimic_env
from .world_model import Go2WorldModel


class Go2SuperMimicRunner(SuperMimicRunner):
    _driver_metrics = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_collection_mask = self.deterministic_collection_mask.clone()

    def _make_world_model(self):
        """Go2接触头使用固定logit尺度，其余世界模型计算沿用通用实现。"""
        return Go2WorldModel(self.task, self.config.wm_hidden)

    def _collection_step_metrics(self):
        metrics = {}
        for name, value in self.env.last_metrics.items():
            if name.startswith("reward/"):
                term = name.removeprefix("reward/")
                weight = self.task.reward_scales[term]
                metrics[f"task/reward_raw/{term}"] = value
                metrics[f"task/reward/{term}"] = self.env.step_dt * weight * value
                metrics[f"Reward_Rate/{term}"] = weight * value
            elif name.startswith("episode_reward/"):
                metrics["Episode_Reward/" + name.removeprefix("episode_reward/")] = value
            else:
                metrics[f"task/{name}"] = value
        if self.driver is not None or self.expert is not None:
            metrics.update(self._driver_metrics)
        return metrics

    def _metric_columns(self):
        """覆盖全部奖励与回合奖励列，保留课程迭代号和驱动计数的主机类型。"""
        columns = super()._metric_columns()
        scalar = torch.zeros((), device=self.device)
        for name in self.task.reward_scales:
            for prefix in ("task/reward_raw/", "task/reward/", "Reward_Rate/", "Episode_Reward/"):
                columns[prefix + name] = scalar
        columns.update(("task/" + name, scalar) for name in (
            "terrain_level", "tracking_linear_mse", "tracking_angular_mse", "action_rms",
            "exteroceptive_fraction", "base_height", "fast_yaw"))
        if self.task.cfg.terrain.mode == "parkour":
            columns.update((f"task/terrain_level/{name}", scalar) for name in self.env.terrain_family_rows)
        return columns

    def _override_actions(self, action, history, observation):
        """按参考实现从环境尾部切出专家与驱动区，其余环境保留当前策略动作。

        驱动区由预训练策略行走，专家区由历史MLP策略行走，两者都只贡献真实数据，
        不参与策略更新；占比随迭代线性衰减到零。
        """
        # 固定组不随命令改变，站立组在每个控制步重新计算。
        self.deterministic_collection_mask = self._fixed_collection_mask
        if self.task.cfg.runner.standing_zero_noise:
            command = self.current.command
            threshold = self.task.cfg.reward.command_threshold
            standing = (command[:, :2].square().sum(-1) <= threshold ** 2) & (command[:, 2].abs() <= threshold)
            self.deterministic_collection_mask = self._fixed_collection_mask | standing
        cfg = self.config
        if self.driver is None and self.expert is None:
            return action
        scale = self._driver_scale()
        num_envs = self.env.num_envs
        driver_count = int(scale * cfg.driver_frac * num_envs) if self.driver is not None else 0
        expert_count = int(scale * cfg.expert_frac * num_envs) if self.expert is not None else 0
        index = torch.arange(num_envs, device=self.device)
        driven = index >= num_envs - driver_count
        expert = (index >= num_envs - driver_count - expert_count) & ~driven
        if driven.any():
            # 当前任务以保留的HIM命令为目标，外部驱动只接管动作，不改变其余环境的命令。
            action = torch.where(driven[:, None], self.driver.act(observation), action)
        if expert.any():
            action = torch.where(expert[:, None], self.expert.act(
                observation, self.task.actor_heightmap(self.current), self.task.actor_mask(self.current)), action)
        if driven.any() or expert.any():
            self._driver_metrics = {"driver/env_count": float(driver_count), "driver/expert_env_count": float(expert_count)}
        return action

    def _reset_drivers(self, done):
        if self.driver is not None:
            self.driver.reset(done)
        if self.expert is not None:
            self.expert.reset(done)

    def _checkpoint_state(self):
        checkpoint = super()._checkpoint_state()
        checkpoint["go2_task"] = {"curriculum": self.env.curriculum_state_dict()}
        return checkpoint

    def save(self, path):
        started = time.perf_counter()
        super().save(path)
        self.timing.record_wall("save_total", time.perf_counter() - started)

    def load(self, path, *, continue_run=False, reset_critic=False):
        checkpoint = super().load(path, continue_run=continue_run, reset_critic=reset_critic)
        if continue_run:
            self.env.resume_progress(self.iteration)
        return checkpoint


def make_go2_supermimic_runner(cfg=None, *, device="cpu", log_dir=None, execution=None):
    """构造任务环境、可选驱动策略和任务快照。"""
    cfg = cfg if cfg is not None else Go2ReferenceCfg()
    config = algorithm_config(cfg)
    if execution is None:
        execution = ExecutionConfig(**{field.name: getattr(cfg.runner, field.name) for field in fields(ExecutionConfig)})
    torch.manual_seed(cfg.env.seed)
    env = make_go2_supermimic_env(cfg, device=device)
    runner = Go2SuperMimicRunner(env, config, log_dir=log_dir,
                                 control_enabled=cfg.runner.enable_control_updates, execution=execution,
                                 collection_groups=env.terrain_curriculum.family_ids)
    if cfg.runner.driver.policy:
        from .driver import MoeCtsDriver
        runner.driver = MoeCtsDriver(cfg.runner.driver.policy, env.num_envs, env.task.default_joint_pos,
                                     cfg.control.action_scales, runner.device)
    if cfg.runner.driver.expert_policy:
        from .driver import ActorDriver
        runner.expert = ActorDriver(cfg.runner.driver.expert_policy, runner.device)
    if log_dir is not None:
        snapshot = {"config": env.cfg.to_dict()}
        destination = Path(log_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "task_config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return runner
