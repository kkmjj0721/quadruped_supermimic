# Copyright (c) 2026-2027 zh
"""
内容：
    启动 Go2 SuperMimic 训练，仅接受本项目保留的任务标识。

用法：
    python scripts/train.py Unitree-Go2-SuperMimic-Flat --device cuda:0 --num-envs 4096
"""

from __future__ import annotations

from pathlib import Path
import sys

# 直接执行脚本时也从当前项目导入任务。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import configure_dependency_warnings

configure_dependency_warnings()


def _print_help(stream) -> None:
    """显示项目当前支持的训练任务。"""
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    print(f"usage: {sys.argv[0]} <TASK> [OPTIONS]", file=stream)
    print(f"Run '{sys.argv[0]} <TASK> --help' for task-specific options.", file=stream)
    print("Available tasks:\n  " + "\n  ".join(sorted(TASK_MODES)), file=stream)


def main() -> None:
    """校验任务标识并调用 SuperMimic 训练入口。"""
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return
    task_id = sys.argv[1]
    if task_id not in TASK_MODES:
        print(f"[ERROR] Unsupported training task: {task_id}", file=sys.stderr)
        _print_help(sys.stderr)
        raise SystemExit(2)

    from src.tasks.locomotion.go2_supermimic.cli import train_main

    train_main(task_id, sys.argv[2:])


if __name__ == "__main__":
    main()
