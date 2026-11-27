# Copyright (c) 2026-2027 zh
"""
内容：
    统一读取训练／播放配置快照并覆盖声明字段。
"""

from dataclasses import fields

from agents.supermimic.config import SuperMimicConfig
import json
from pathlib import Path


def _merge_mapping(defaults: dict, values: dict) -> dict:
    for name, value in values.items():
        if name in defaults and isinstance(defaults[name], dict) and isinstance(value, dict):
            _merge_mapping(defaults[name], value)
        else:
            defaults[name] = value
    return defaults


def apply_settings(target, values: dict, *, replace_mappings: bool = False) -> None:
    """只覆盖已声明配置字段；字典参数允许局部覆盖和自定义键。"""
    for name, value in values.items():
        current = getattr(target, name)
        if isinstance(value, dict):
            if not isinstance(current, dict):
                apply_settings(current, value, replace_mappings=replace_mappings)
                continue
            value = value if replace_mappings else _merge_mapping(current, value)
        elif isinstance(current, (tuple, list)):
            value = type(current)(value)
        setattr(target, name, value)


def load_task_config(path: Path):
    """读取一次外部JSON并应用配置，返回共享已读取数据的快照。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = payload["config"] if "config" in payload else payload
    cfg = configuration_for_settings(settings)
    complete = "config" in payload or set(cfg.to_dict()).issubset(settings)
    apply_settings(cfg, settings, replace_mappings=complete)
    return cfg, payload


def configuration_for_settings(settings: dict):
    """先选择任务阶段的默认配方，再应用外部配置。"""
    from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg
    from src.config.go2.go2_parkour_config import parkour_config

    if "terrain" in settings and "mode" in settings["terrain"] and settings["terrain"]["mode"] == "parkour":
        return parkour_config(Go2ReferenceCfg())
    return Go2ReferenceCfg()


def algorithm_config(cfg):
    """把任务侧驱动配置并入算法预算，并转换为算法配置。"""
    driver = cfg.runner.driver
    values = {field.name: getattr(cfg.algo, field.name) for field in fields(SuperMimicConfig)}
    values.update(driver_frac=driver.frac, expert_frac=driver.expert_frac,
                  driver_fade=tuple(driver.fade_iterations))
    return SuperMimicConfig(**values)
