# Copyright (c) 2026-2027 zh
"""
内容：
    独立格式化SuperMimic训练摘要，在调用方汇总边界写入TensorBoard标量。
"""

from __future__ import annotations

import math
from pathlib import Path
from datetime import timedelta


# 保留原始tag供已有分析工具读取，同时提供与PPO一致的常用面板分组。
_SCALAR_GROUPS = {
    "wm_loss": "Loss/world_model",
    "critic_loss": "Loss/value",
    "control_loss": "Loss/control",
    "velocity_loss": "Loss/velocity_estimation",
    "actor_loss": "Loss/actor_total",
    "velocity_rmse": "Metrics/velocity_estimation/rmse",
    "velocity_rmse_x": "Metrics/velocity_estimation/rmse_x",
    "velocity_rmse_y": "Metrics/velocity_estimation/rmse_y",
    "velocity_rmse_z": "Metrics/velocity_estimation/rmse_z",
    "world_model_learning_rate": "Loss/world_model_learning_rate",
    "critic_learning_rate": "Loss/value_learning_rate",
    "control_learning_rate": "Loss/control_learning_rate",
    "wm_grad_norm": "Grad/world_model",
    "control_grad_norm": "Grad/control",
    "performance/real_total_fps": "Perf/total_fps",
    "timing/collection_seconds": "Perf/collection_time",
    "timing/learning_seconds": "Perf/learning_time",
    "episode/return_mean": "Train/mean_reward",
    "episode/length_steps_mean": "Train/mean_episode_length",
}


def format_training_log(metrics: dict[str, int | float], *, num_envs: int, device) -> str:
    """按Go2 HIM的80列单块样式显示训练摘要和实时奖励率。"""
    def number(key: str, *, decimals: int = 4, count: bool = False) -> str:
        return f"{metrics[key]:.0f}" if count else f"{metrics[key]:.{decimals}f}"

    def duration(key: str) -> str:
        return str(timedelta(seconds=int(max(0, metrics[key]))))

    def row(label: str, value: str) -> str:
        return f"{label + ':':>40} {value}"

    progress = f"{metrics['iteration']}/{metrics['training_target_iteration']}"
    learning_time = metrics["timing/world_model_seconds"] + metrics["timing/control_seconds"]
    lines = ["#" * 80, f" Learning iteration {progress} ".center(80), "",
             row("Total steps", number("real_steps", count=True)),
             row("Steps per second", number("performance/real_total_fps", count=True)),
             row("Collection time", number("timing/collection_seconds", decimals=3) + "s"),
             row("Learning time", f"{learning_time:.3f}s")]
    if metrics["control_enabled"] == 0:
        lines.append(row("Training phase", "world-model only"))
    lines.append(row("Mean world model loss", number("wm_loss")))
    for label, key in (
        ("Mean value loss", "critic_loss"), ("Mean control loss", "control_loss"),
        ("Mean velocity estimation loss", "velocity_loss"), ("Mean total actor loss", "actor_loss"),
    ):
        if key in metrics:
            lines.append(row(label, number(key)))
    lines.append(row("Control learning rate", f"{metrics['control_learning_rate']:.2e}"))
    for label, key in (("Mean reward", "episode/return_mean"), ("Mean episode length", "episode/length_steps_mean")):
        if key in metrics:
            lines.append(row(label, number(key, decimals=2)))
    lines.extend([row("Metrics/failures", number("failures", count=True)),
                  row("Metrics/timeouts", number("timeouts", count=True))])
    if "task/terrain_level" in metrics:
        lines.append(row("Metrics/terrain_level", number("task/terrain_level")))
    for label, key in (("Metrics/linear_velocity_rmse", "task/tracking_linear_mse"),
                       ("Metrics/yaw_rate_rmse", "task/tracking_angular_mse")):
        if key in metrics:
            lines.append(row(label, f"{math.sqrt(max(0, metrics[key])):.4f}"))
    if "task/action_rms" in metrics:
        lines.append(row("Metrics/action_rms", number("task/action_rms")))
    if "velocity_rmse" in metrics:
        lines.append(row("Metrics/velocity_estimation_rmse", number("velocity_rmse")))
    for key in metrics:
        if key.startswith("Reward_Rate/"):
            lines.append(row(key, number(key)))
    lines.extend(["-" * 80, row("Iteration time", number("timing/iteration_seconds", decimals=2) + "s"),
                  row("Time elapsed", duration("timing/session_elapsed_seconds")),
                  row("ETA", duration("timing/eta_seconds")), ""])
    return "\n".join(lines)


class TrainingLogger:
    """每次汇总写入TensorBoard，横轴为窗口末轮；终端另按迭代间隔显示。"""

    def __init__(self, log_dir: str | Path | None, *, num_envs: int, device, log_interval: int = 1):
        self.num_envs = num_envs
        self.device = str(device)
        self.log_interval = log_interval
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._records = 0
        self._writer = None
        if self.log_dir is not None:
            # 窗口指标集中写入同一事件文件，tag按"/"分组成TensorBoard面板。
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(self.log_dir), flush_secs=10)

    def add_text(self, tag: str, text: str) -> None:
        """把一次性的运行来源信息写入同一事件文件，不新增运行目录文件。"""
        if self._writer is not None:
            self._writer.add_text(tag, text, 0)

    def record(self, metrics: dict[str, int | float], *, force: bool = False) -> None:
        iteration = metrics["iteration"]
        target = metrics["training_target_iteration"]
        self._records += 1
        if self._writer is not None:
            seconds = int(metrics["timing/session_elapsed_seconds"])
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, iteration)
                if name in _SCALAR_GROUPS:
                    self._writer.add_scalar(_SCALAR_GROUPS[name], value, iteration)
            for source, tag in (("task/tracking_linear_mse", "Metrics/tracking/linear_velocity_rmse"),
                                ("task/tracking_angular_mse", "Metrics/tracking/yaw_rate_rmse")):
                if source in metrics:
                    self._writer.add_scalar(tag, math.sqrt(max(0, metrics[source])), iteration)
            for source in ("episode/return_mean", "episode/length_steps_mean"):
                if source in metrics:
                    self._writer.add_scalar(_SCALAR_GROUPS[source] + "/time", metrics[source], seconds)
        if force or self._records == 1 or iteration % self.log_interval == 0 or iteration == target:
            print(format_training_log(metrics, num_envs=self.num_envs, device=self.device), flush=True)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
