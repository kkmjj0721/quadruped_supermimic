# Copyright (c) 2026-2027 zh
"""
内容：
    恢复SuperMimic配置与策略，核对任务语义并导出部署产物。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from agents.supermimic.config import SuperMimicConfig
from agents.supermimic.interfaces import AlgorithmSpec
from agents.supermimic.export import export_policy
from agents.supermimic.policy import make_actor
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.train_cfg import TASK_MODES
from ..config.loading import load_task_config, algorithm_config
from ..mdp.reference_task import Go2ReferenceTaskModel


@dataclass(frozen=True)
class PlaybackSource:
    """原训练配置与CPU检查点；播放覆盖应施加到独立的配置副本。"""

    cfg: Go2ReferenceCfg
    checkpoint: dict
    path: Path
    snapshot: dict


def load_playback_source(task_id: str, checkpoint_file: str | Path) -> PlaybackSource:
    """读取可信本项目检查点及同目录的完整任务快照，不创建仿真。"""
    path = Path(checkpoint_file).expanduser().resolve()
    cfg, snapshot = load_task_config(path.parent / "task_config.json")
    if "config" not in snapshot or not set(cfg.to_dict()).issubset(snapshot["config"]):
        raise ValueError("Playback requires a complete task_config.json snapshot")
    if cfg.terrain.mode != TASK_MODES[task_id]:
        raise ValueError("Playback task terrain mode differs from the saved configuration")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["format"] != "supermimic_proof_teacher_v2":
        raise ValueError("Incompatible SuperMimic checkpoint format")
    if checkpoint["spec_hash"] != snapshot["spec_hash"] or checkpoint["go2_task"]["spec_hash"] != snapshot["spec_hash"]:
        raise ValueError("Checkpoint and task snapshot metadata mismatch")
    return PlaybackSource(cfg, checkpoint, path, snapshot)


def build_playback_policy(source: PlaybackSource, adapter) -> torch.nn.Module:
    """核对原训练配置和地形语义，只装配策略及归一化统计。"""
    expected, actual = source.cfg.to_dict(), adapter.cfg.to_dict()
    for section, name in (("env", "num_envs"), ("noise", "add_noise"), ("terrain", "curriculum")):
        expected[section][name] = actual[section][name]
    expected["domain_randomization"]["enabled"] = actual["domain_randomization"]["enabled"]
    if json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True):
        raise ValueError("Playback configuration differs from the saved configuration")
    if source.snapshot["geometry_hash"] != adapter.surface.geometry_hash:
        raise ValueError("Playback terrain geometry differs from the saved task")
    if source.checkpoint["config"] != asdict(algorithm_config(source.cfg)):
        raise ValueError("SuperMimic checkpoint configuration mismatch")
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
    """读取检查点与任务快照，导出PT/ONNX/JSON三件套；有限性由导出入口检查。"""
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor, algorithm = checkpoint_actor(checkpoint)
    cfg, snapshot = load_task_config(path.parent / "task_config.json")
    if snapshot["spec_hash"] != checkpoint["spec_hash"]:
        raise ValueError("Checkpoint and task snapshot metadata mismatch")
    if (actor.observation_dim != cfg.env.num_observations or actor.history_length != cfg.env.num_obs_history
            or cfg.env.num_obs != actor.observation_dim * actor.history_length
            or actor.heightmap_dim != cfg.env.heightmap_dim):
        raise ValueError("Checkpoint observation layout differs from the saved task configuration")
    iteration = checkpoint["iteration"] if iteration is None else iteration
    from .deployment import deployment_metadata

    metadata = deployment_metadata(cfg, spec_hash=checkpoint["spec_hash"], algorithm=algorithm)
    metadata["source_network_precision"] = "bf16" if checkpoint["execution"]["actor_bf16"] else "fp32"
    return export_policy(actor, Path(directory).expanduser().resolve(), iteration=iteration, metadata=metadata)
