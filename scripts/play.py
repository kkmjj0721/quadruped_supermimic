# Copyright (c) 2026-2027 zh
"""
内容：
    播放 Go2 SuperMimic 检查点，或使用零动作、随机动作查看任务环境。

用法：
    python scripts/play.py Unitree-Go2-SuperMimic-Rough --checkpoint-file logs/supermimic/go2_supermimic/实验目录/model_100.pt
    python -m scripts.play Unitree-Go2-SuperMimic-Flat --agent zero --num-envs 1 --viewer viser
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

# 直接执行脚本时也从当前项目导入任务，不依赖外部 PYTHONPATH。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import configure_dependency_warnings

configure_dependency_warnings()

import mjlab
import torch
import tyro

from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
    """策略、设备、录像和查看器参数。"""

    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    """关闭所有终止条件，便于持续观察。"""


def _print_help(stream) -> None:
    """显示项目当前支持的播放任务。"""
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    print(f"usage: {sys.argv[0]} TASK [OPTIONS]", file=stream)
    print(f"Run '{sys.argv[0]} TASK --help' for playback options.", file=stream)
    print("Available tasks:", file=stream)
    for task in sorted(TASK_MODES):
        print(f"  {task}", file=stream)


def _validate_task(task_id: str) -> None:
    """拒绝项目未提供的播放任务。"""
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    if task_id not in TASK_MODES:
        raise ValueError(f"Unsupported playback task: {task_id}")


def _validate_config(cfg: PlayConfig) -> None:
    """在分配仿真资源前检查入口参数。"""
    if cfg.agent not in ("zero", "random", "trained"):
        raise ValueError(f"Unsupported agent mode: {cfg.agent}")
    if cfg.viewer not in ("auto", "native", "viser"):
        raise ValueError(f"Unsupported viewer backend: {cfg.viewer}")
    for name in ("num_envs", "video_height", "video_width"):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive.")
    if cfg.video and cfg.video_length <= 0:
        raise ValueError("video_length must be positive when recording video.")


def run_viewer(env, policy, cfg: PlayConfig, device: str) -> None:
    """按显示环境选择原生窗口或浏览器查看器，结束后由调用方关闭环境。"""
    viewer = cfg.viewer
    if viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "viser"
    print(f"[INFO] Playing with: device={device}, viewer={viewer}", flush=True)
    viewer_cls = {"native": NativeMujocoViewer, "viser": ViserPlayViewer}[viewer]
    viewer_cls(env, policy).run()


def _resolve_checkpoint(cfg: PlayConfig) -> Path | None:
    """解析训练策略需要的检查点；虚拟策略不读取文件。"""
    if cfg.agent != "trained":
        return None
    if cfg.checkpoint_file is None:
        raise ValueError("Trained playback requires --checkpoint-file.")
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    print(f"[INFO] Loading checkpoint: {checkpoint}", flush=True)
    return checkpoint


def run_play(task_id: str, cfg: PlayConfig) -> None:
    """装配 SuperMimic 播放环境和策略，并确保退出时关闭环境。"""
    _validate_task(task_id)
    _validate_config(cfg)
    checkpoint = _resolve_checkpoint(cfg)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.tasks.locomotion.go2_supermimic.rl.playback import make_playback_env

    env, policy = make_playback_env(
        task_id, checkpoint_file=checkpoint, agent=cfg.agent, num_envs=cfg.num_envs, device=device,
        video=cfg.video, video_length=cfg.video_length, video_height=cfg.video_height,
        video_width=cfg.video_width, no_terminations=cfg.no_terminations,
    )
    try:
        if cfg.no_terminations:
            print("[INFO] Terminations disabled.")
        run_viewer(env, policy, cfg, device)
    finally:
        env.close()


def main() -> None:
    """解析当前唯一支持的 SuperMimic 播放协议。"""
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return
    task_id = sys.argv[1]
    if task_id not in TASK_MODES:
        print(f"[ERROR] Unsupported playback task: {task_id}", file=sys.stderr)
        _print_help(sys.stderr)
        raise SystemExit(2)
    cfg = tyro.cli(PlayConfig, args=sys.argv[2:], prog=f"{sys.argv[0]} {task_id}", config=mjlab.TYRO_FLAGS)
    run_play(task_id, cfg)


if __name__ == "__main__":
    main()
