# Copyright (c) 2026-2027 zh
"""
内容：
    统一读取训练／播放配置快照、覆盖声明字段并验证声明字段。
"""

from copy import deepcopy
from dataclasses import fields

from agents.supermimic.config import SuperMimicConfig
import json
from pathlib import Path


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
        if name.startswith("_") or not hasattr(target, name):
            raise ValueError(f"Unknown task configuration field: {path}")
        current = getattr(target, name)
        if callable(current):
            raise ValueError(f"Task configuration field is not writable: {path}")
        if hasattr(current, "__dict__"):
            apply_settings(current, value, path, replace_mappings=replace_mappings)
            continue
        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Task configuration field {path} must be an object")
            value = deepcopy(value) if replace_mappings else _merge_mapping(current, value)
        elif isinstance(current, (tuple, list)):
            value = type(current)(value)
        setattr(target, name, deepcopy(value))


def load_task_config(path: Path):
    """读取一次外部JSON并在此边界应用和校验配置，返回原始快照。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = payload["config"] if "config" in payload else payload
    cfg = configuration_for_settings(settings)
    complete = "config" in payload or set(cfg.to_dict()).issubset(settings)
    apply_settings(cfg, settings, replace_mappings=complete)
    validate_config(cfg)
    return cfg, payload


def configuration_for_settings(settings: dict):
    """配置I/O边界只接受当前PROOF教师，不迁移旧MoE快照。"""
    from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg
    from src.config.go2.go2_parkour_config import parkour_config

    if "algorithm" in settings and settings["algorithm"] != "proof_teacher":
        raise ValueError(f"Unsupported SuperMimic algorithm: {settings['algorithm']}")
    if "terrain" in settings and "mode" in settings["terrain"] and settings["terrain"]["mode"] == "parkour":
        return parkour_config(Go2ReferenceCfg())
    return Go2ReferenceCfg()


def validate_config(cfg) -> None:
    """只拒绝当前教师未实现、却可能静默改变任务的配置语义。"""
    if cfg.algorithm != "proof_teacher":
        raise ValueError("SuperMimic requires the proof_teacher algorithm")
    if not cfg.commands.heading_command or cfg.commands.rel_heading_envs != 1.0 or cfg.commands.stages:
        raise ValueError("The Go2 teacher requires heading commands and the existing fixed velocity ranges")
    if cfg.terrain.mode not in ("flat", "rough", "parkour"):
        raise ValueError("Unsupported SuperMimic terrain mode")
    if cfg.terrain.mode == "parkour":
        if not cfg.terrain.asset_path or not (Path(__file__).resolve().parents[5] / cfg.terrain.asset_path).is_file():
            raise ValueError(f"Parkour terrain asset does not exist: {cfg.terrain.asset_path}")
        if not cfg.commands.parkour:
            raise ValueError("Parkour requires the second-stage command configuration")
        if cfg.domain_randomization.joint_parameter_mode not in ("startup", "reset"):
            raise ValueError("Parkour joint randomization mode must be startup or reset")


def algorithm_config(cfg):
    """把任务侧驱动配置并入算法预算，并转换为算法配置。"""
    driver = cfg.runner.driver
    values = {field.name: getattr(cfg.algo, field.name) for field in fields(SuperMimicConfig)}
    values.update(driver_frac=driver.frac, expert_frac=driver.expert_frac,
                  driver_fade=tuple(driver.fade_iterations))
    return SuperMimicConfig(**values)
