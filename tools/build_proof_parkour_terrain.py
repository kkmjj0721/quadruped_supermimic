# Copyright (c) 2026-2027 zh
"""
内容：
    无 IsaacLab 依赖地构建 PROOF Parkour 碰撞地形、出生点、虚拟锐边和真值探针。
用法：
    .venv/bin/python tools/build_proof_parkour_terrain.py --seed 17
    .venv/bin/python tools/build_proof_parkour_terrain.py --seed 42 --num-rows 4 --num-cols 6 --output logs/parkour_4x6.npz
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.parkour_terrain.packing import pack
from tools.parkour_terrain.presets import parkour_terrain_cfg


def main():
    parser = argparse.ArgumentParser(description="Build a self-contained PROOF parkour terrain asset")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument("--num-cols", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parents[1] / f"src/assets/terrains/proof_parkour_{args.seed}.npz"
    )
    started = perf_counter()
    data = pack(args.seed, parkour_terrain_cfg(), args.num_rows, args.num_cols)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f"Built terrain asset: {output} ({perf_counter() - started:.3f} seconds)", flush=True)


if __name__ == "__main__":
    main()
