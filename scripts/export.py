# Copyright (c) 2026-2027 zh
"""
内容：
    从训练检查点离线导出三输入教师策略的 TorchScript、ONNX 与部署元数据。

用法：
    python scripts/export.py logs/supermimic/实验目录/model_1000.pt exports/go2_teacher
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 直接执行脚本时也从当前项目导入任务。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import configure_dependency_warnings

configure_dependency_warnings()


def main() -> None:
    """读取检查点与同目录任务快照，按显式命令生成部署文件。"""
    parser = argparse.ArgumentParser(description="Export a SuperMimic teacher checkpoint for deployment.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()

    from src.tasks.locomotion.go2_supermimic.rl.checkpoints import export_checkpoint

    paths = export_checkpoint(args.checkpoint, args.directory, iteration=args.iteration)
    for kind, path in paths.items():
        print(f"[INFO] Exported {kind}: {path}", flush=True)


if __name__ == "__main__":
    main()
