# Copyright (c) 2026-2027 zh
"""
内容：
    统一读取训练／播放配置快照、覆盖声明字段并验证声明字段。
"""

from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path

from agents.supermimic.config import SuperMimicConfig
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg


_JSON_VALUE_TYPES = {float: (int, float), tuple: (tuple, list), list: (tuple, list), type(None): (type(None), int)}


def _merge_mapping(defaults: dict, values: dict) -> dict:
    result = deepcopy(defaults)
    for name, value in values.items():
        if name in result and isinstance(result[name], dict) and isinstance(value, dict):
            result[name] = _merge_mapping(result[name], value)
        else:
            result[name] = deepcopy(value)
    return result


def apply_settings(target, values: dict, prefix: str = "", *, replace_mappings: bool = False) -> None:
    """只覆盖已声明配置字段；字典参数允许局部覆盖和自定义键。"""
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section {prefix or 'root'} must be an object")
    for name, value in values.items():
        path = f"{prefix}.{name}" if prefix else name
        if name.startswith("_"):
            raise ValueError(f"Unknown task configuration field: {path}")
        current = getattr(target, name)
        if callable(current):
            raise ValueError(f"Task configuration field is not writable: {path}")
        if hasattr(current, "__dict__"):
            apply_settings(current, value, path, replace_mappings=replace_mappings)
            continue
        kind = type(current)
        allowed = _JSON_VALUE_TYPES[kind] if kind in _JSON_VALUE_TYPES else (kind,)
        if type(value) not in allowed:
            names = " or ".join(expected.__name__ for expected in allowed)
            raise ValueError(f"Task configuration field {path} must be {names}")
        if kind is dict:
            value = deepcopy(value) if replace_mappings else _merge_mapping(current, value)
        elif kind in (tuple, list):
            value = kind(deepcopy(value))
        setattr(target, name, value)


def load_task_config(path: Path, *, require_snapshot: bool = False) -> tuple[Go2ReferenceCfg, dict]:
    """仅在外部 JSON 入口检查并装配任务配置，返回同次读取的原始元数据。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(payload, allow_nan=False)
    if not isinstance(payload, dict):
        raise ValueError("Task configuration JSON must be an object")
    wrapped = "config" in payload
    settings = payload["config"] if wrapped else payload
    cfg = Go2ReferenceCfg()
    sections = set(cfg.to_dict())
    if require_snapshot and (not wrapped or not sections.issubset(settings)):
        raise ValueError(f"A complete task configuration snapshot is required: {path}")
    apply_settings(cfg, settings, replace_mappings=wrapped or sections.issubset(settings))
    validate_config(cfg)
    return cfg, payload


def validate_config(cfg) -> None:
    """在配置加载边界统一检查教师的观测布局与外部数值参数。"""
    if cfg.algorithm != "proof_teacher":
        raise ValueError("SuperMimic requires the proof_teacher algorithm")
    if (cfg.env.num_obs_history, cfg.env.num_observations, cfg.env.num_obs,
            cfg.env.num_critic_obs, cfg.env.num_privileged_obs, cfg.env.heightmap_dim) != (10, 45, 450, 285, 285, 162):
        raise ValueError("PROOF teacher dimensions must be history10, proprioception45, critic285 and heightmap162")
    if not cfg.commands.heading_command or cfg.commands.rel_heading_envs != 1.0 or cfg.commands.stages:
        raise ValueError("The Go2 teacher requires heading commands and the existing fixed velocity ranges")
    if cfg.control.decimation < 1:
        raise ValueError("Control decimation must be a positive integer")
    for name, value in (("episode_length_s", cfg.env.episode_length_s), ("timestep", cfg.sim.timestep),
                        ("tracking_linear_variance", cfg.reward.tracking_linear_variance),
                        ("tracking_angular_variance", cfg.reward.tracking_angular_variance),
                        ("fall_softness_deg", cfg.reward.fall_softness_deg)):
        if value <= 0:
            raise ValueError(f"Task parameter must be positive and finite: {name}")
    if not 0 < cfg.reward.fall_angle < math.pi:
        raise ValueError("Fall angle must be between zero and pi")
    if cfg.terrain.mode not in ("flat", "rough"):
        raise ValueError("Unsupported SuperMimic terrain mode")
    if tuple(cfg.terrain.scan_size_m) != (1.7, 0.8) or cfg.terrain.scan_resolution_m != 0.1:
        raise ValueError("PROOF teacher heightmap requires a 1.7x0.8m grid with 0.1m spacing")
    if not 0 <= cfg.teacher.exteroceptive_dropout <= 1:
        raise ValueError("Exteroceptive dropout must be a probability")
    command_ranges = (*cfg.commands.initial_ranges, cfg.commands.heading_range, cfg.commands.resampling_time)
    if len(cfg.commands.initial_ranges) != 3 or any(
        len(bounds) != 2 or any(type(value) not in (int, float) for value in bounds) or bounds[0] > bounds[1]
        for bounds in command_ranges
    ):
        raise ValueError("Command ranges must be finite and ordered pairs")
    for value in (cfg.commands.rel_standing_envs, cfg.commands.rel_forward_envs):
        if not 0 <= value <= 1:
            raise ValueError("Command mode probabilities must be in [0,1]")
    if not 0 < cfg.commands.resampling_time[0] <= cfg.commands.resampling_time[1]:
        raise ValueError("Command resampling times must be positive and ordered")


def algorithm_config(cfg):
    """在JSON覆盖后重新执行dataclass验证，把任务侧驱动配置并入算法预算。"""
    driver = cfg.runner.driver
    values = {field.name: getattr(cfg.algo, field.name) for field in fields(SuperMimicConfig)}
    values.update(driver_frac=driver.frac, expert_frac=driver.expert_frac,
                  driver_fade=tuple(driver.fade_iterations))
    return SuperMimicConfig(**values)
