# Copyright (c) 2026-2027 zh
"""
内容：
    恢复SuperMimic配置与策略，严格加载权重并离线导出部署产物。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from agents.supermimic.config import SuperMimicConfig
from agents.supermimic.interfaces import AlgorithmSpec
from agents.supermimic.export import export_policy
from agents.supermimic.policy import make_actor
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.loading import load_task_config
from ..mdp.reference_task import Go2ReferenceTaskModel


@dataclass(frozen=True)
class PlaybackSource:
    """独立读取的训练配置与CPU检查点；播放覆盖直接作用于本次配置。"""

    cfg: Go2ReferenceCfg
    checkpoint: dict


def load_playback_source(task_id: str, checkpoint_file: str | Path) -> PlaybackSource:
    """读取可信本项目检查点及同目录的完整任务快照，不创建仿真。"""
    path = Path(checkpoint_file).expanduser().resolve()
    cfg, _ = load_task_config(path.parent / "task_config.json")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return PlaybackSource(cfg, checkpoint)


def build_playback_policy(source: PlaybackSource, adapter) -> torch.nn.Module:
    """按检查点装配策略及归一化统计，并移到播放设备。"""
    policy, _ = checkpoint_actor(source.checkpoint)
    return policy.to(adapter.device)


def checkpoint_actor(checkpoint: dict):
    """只装配actor，以严格权重加载核对网络结构和维度。"""
    if checkpoint["format"] != "supermimic_proof_teacher_v2":
        raise ValueError("Incompatible SuperMimic velocity-policy checkpoint format")
    spec = AlgorithmSpec(**checkpoint["spec"])
    algorithm = SuperMimicConfig(**checkpoint["config"])
    actor = make_actor(spec, algorithm, Go2ReferenceTaskModel.observation_term_dims)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    return actor.eval().requires_grad_(False), algorithm


def export_checkpoint(checkpoint_path: str | Path, directory: str | Path, *, iteration: int | None = None):
    """读取检查点与任务快照，离线导出PT/ONNX/JSON三件套。"""
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor, algorithm = checkpoint_actor(checkpoint)
    cfg, _ = load_task_config(path.parent / "task_config.json")
    iteration = checkpoint["iteration"] if iteration is None else iteration
    from .deployment import deployment_metadata

    metadata = deployment_metadata(cfg, algorithm=algorithm)
    metadata["source_network_precision"] = "bf16" if checkpoint["execution"]["actor_bf16"] else "fp32"
    return export_policy(actor, Path(directory).expanduser().resolve(), iteration=iteration, metadata=metadata)
