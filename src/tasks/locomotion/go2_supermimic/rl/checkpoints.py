# Copyright (c) 2026-2027 zh
"""
内容：
    恢复SuperMimic训练配置并校验播放检查点，严格装配策略和归一化统计。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from agents.supermimic.config import SuperMimicConfig
from agents.supermimic.export import export_policy
from agents.supermimic.interfaces import AlgorithmSpec
from agents.supermimic.policy import make_actor
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.loading import algorithm_config, load_task_config
from ..config.train_cfg import TASK_MODES
from ..mdp.reference_task import Go2ReferenceTaskModel, build_go2_task_spec


@dataclass(frozen=True)
class PlaybackSource:
    """原训练配置与CPU检查点；播放覆盖应施加到独立的配置副本。"""

    cfg: Go2ReferenceCfg
    checkpoint: dict
    spec: AlgorithmSpec
    geometry_hash: str


def _checkpoint_spec(checkpoint: dict) -> AlgorithmSpec:
    if checkpoint["format"] != "supermimic_proof_teacher_v2":
        raise ValueError("Incompatible SuperMimic checkpoint format")
    spec = AlgorithmSpec(**checkpoint["spec"])
    if spec.model_id != checkpoint["format"] or not spec.semantic_id.startswith("go2_proof_teacher_"):
        raise ValueError("Checkpoint does not describe a Go2 SuperMimic task")
    layout = Go2ReferenceCfg.env
    if (spec.observation_dim, spec.history_length, spec.action_dim, spec.heightmap_dim) != (
            layout.num_observations, layout.num_obs_history, layout.num_actions, layout.heightmap_dim):
        raise ValueError("Unsupported Go2 policy observation or action layout")
    if spec.hash != checkpoint["spec_hash"]:
        raise ValueError("Checkpoint specification hash mismatch")
    return spec


def _load_bundle(checkpoint_path):
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec = _checkpoint_spec(checkpoint)
    cfg, snapshot = load_task_config(path.parent / "task_config.json", require_snapshot=True)
    if snapshot["spec_hash"] != checkpoint["spec_hash"]:
        raise ValueError("Checkpoint and task snapshot metadata mismatch")
    return checkpoint, spec, cfg, snapshot


def load_playback_source(task_id: str, checkpoint_file: str | Path) -> PlaybackSource:
    """读取可信本项目检查点及同目录的完整任务快照，不创建仿真。"""
    checkpoint, spec, cfg, snapshot = _load_bundle(checkpoint_file)
    if cfg.terrain.mode != TASK_MODES[task_id]:
        raise ValueError("Playback task terrain mode differs from the saved configuration")
    if checkpoint["go2_task"]["spec_hash"] != checkpoint["spec_hash"]:
        raise ValueError("Checkpoint curriculum specification mismatch")
    return PlaybackSource(cfg, checkpoint, spec, snapshot["geometry_hash"])


def _make_checkpoint_actor(checkpoint, spec, algorithm, device="cpu"):
    actor = make_actor(spec, algorithm, Go2ReferenceTaskModel.observation_term_dims).to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    return actor.eval().requires_grad_(False)


def build_playback_policy(source: PlaybackSource, adapter) -> torch.nn.Module:
    """核对训练任务身份和当前几何，只装配播放策略。"""
    if source.geometry_hash != adapter.surface.geometry_hash:
        raise ValueError("Playback terrain geometry differs from the saved task")
    task = adapter.task
    cfg = deepcopy(adapter.cfg)
    cfg.env.num_envs = source.cfg.env.num_envs
    cfg.noise.add_noise = source.cfg.noise.add_noise
    cfg.terrain.curriculum = source.cfg.terrain.curriculum
    cfg.teacher.exteroceptive_dropout = source.cfg.teacher.exteroceptive_dropout
    cfg.domain_randomization.enabled = source.cfg.domain_randomization.enabled
    expected = build_go2_task_spec(cfg, source.geometry_hash, task.default_joint_pos, task.joint_limits,
                                   task.heightmap_dim)
    if source.spec != expected:
        raise ValueError("Checkpoint task spec does not match the saved configuration and current robot")
    algorithm = algorithm_config(source.cfg)
    if source.checkpoint["config"] != asdict(algorithm):
        raise ValueError("SuperMimic checkpoint configuration mismatch")
    return _make_checkpoint_actor(source.checkpoint, source.spec, algorithm, adapter.device)


def checkpoint_actor(checkpoint: dict):
    """只从外部检查点装配actor，不加载世界模型或critic。"""
    spec = _checkpoint_spec(checkpoint)
    algorithm = SuperMimicConfig(**checkpoint["config"])
    return _make_checkpoint_actor(checkpoint, spec, algorithm), algorithm


def export_checkpoint(checkpoint_path: str | Path, directory: str | Path, *, iteration: int | None = None):
    """读取检查点与同目录任务快照，导出PT/ONNX/JSON三件套。"""
    checkpoint, spec, cfg, _ = _load_bundle(checkpoint_path)
    actor_bf16 = checkpoint["execution"]["actor_bf16"]
    if type(actor_bf16) is not bool:
        raise ValueError("Checkpoint execution.actor_bf16 must be a boolean")
    algorithm = SuperMimicConfig(**checkpoint["config"])
    actor = _make_checkpoint_actor(checkpoint, spec, algorithm)
    iteration = checkpoint["iteration"] if iteration is None else iteration
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    from .deployment import deployment_metadata

    metadata = deployment_metadata(cfg, spec_hash=checkpoint["spec_hash"], algorithm=algorithm)
    # 离线重建的 actor 使用 FP32；训练来源精度由检查点执行配置保留。
    metadata["source_network_precision"] = "bf16" if actor_bf16 else "fp32"
    return export_policy(actor, Path(directory).expanduser().resolve(), iteration=iteration, metadata=metadata)
