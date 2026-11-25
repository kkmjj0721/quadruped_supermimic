# Copyright (c) 2026-2027 zh
"""
内容：
    从 PROOF 与 LocoLab 官方生成器导出固定种子的二阶段地图、出生点和虚拟边缘。
    在作者环境安装 proof_parkour、locolab、isaaclab 和 embreex 后执行。
用法：
    python tools/build_proof_parkour_terrain.py --seed 17
"""

import argparse
from pathlib import Path
import random

import numpy as np
import trimesh


ASSETS = Path(__file__).resolve().parents[1] / "src/assets/terrains"
CROPS = {
    "gap": ((-3.9, 3.9), (-2., 2.), (-0.5, 0.5)),
    "climb": ((-3.9, 3.9), (-3.9, 3.9), (0.1, 1.)),
    "stairs_inv_high_straight": ((-3.9, 3.9), (-1.5, 1.5), (-0.1, 2.5)),
}


def upward_grid(mesh, family):
    """取矩形可行走面的原始顶点；不规则高度场按原始三角面向下投射到2厘米栅格。"""
    top = np.asarray(mesh.vertices)[np.unique(mesh.faces[mesh.face_normals[:, 2] > 0.01])]
    x, ix = np.unique(top[:, 0], return_inverse=True)
    y, iy = np.unique(top[:, 1], return_inverse=True)
    if family in ("slope", "slope_inv", "discrete"):
        # 少数陡沿经过 Isaac 的 slope_threshold 处理，多个顶点可能具有相同 XY。
        line = np.linspace(0., 8., 401)
        gx, gy = np.meshgrid(line, line)
        # 外边界共用的零高度节点直接从作者的地块边框取得。
        starts = np.column_stack((gx[1:-1, 1:-1].ravel(), gy[1:-1, 1:-1].ravel(),
                                  np.full((399 * 399,), top[:, 2].max() + 1.)))
        rays = np.tile((0., 0., -1.), (len(starts), 1))
        hits, ids, _ = mesh.ray.intersects_location(starts, rays, multiple_hits=False)
        assert len(ids) == len(starts), "Missing interior hit in source heightfield"
        inner = np.empty(len(starts))
        inner[ids] = hits[:, 2]
        return np.pad(inner.reshape(399, 399), 1)
    assert len(np.unique(iy * len(x) + ix)) == len(x) * len(y), "Incomplete rectangular source top"
    heights = np.empty((len(y), len(x)), dtype=np.float32)
    heights[iy, ix] = top[:, 2]
    return heights


def boundary_profile(asset, cell, axis, side):
    """读取与地块边界相接的完整顶面，各来源地形在最外圈关闭粗糙扰动。"""
    row, col = cell
    if not (0 <= row < 10 and 0 <= col < 20):
        return np.array([[0., 8., -1., 0.]])
    key = f"r{row}c{col}"
    strips = []
    for x0, y0, x1, y1, bottom, top, field_id in asset[key]:
        bounds = ((x0, x1), (y0, y1))
        if abs(bounds[axis][side] - side * 8) > 1e-6:
            continue
        if field_id >= 0:
            heights = asset[f"{key}f{int(field_id)}"]
            rim = np.take(heights, side * (heights.shape[1 - axis] - 1), axis=1 - axis)
            assert np.ptp(rim) < 1e-5, "Source mesh rim is not level"
            top = rim.mean()
        strips.append((*bounds[1 - axis], bottom, top))
    return np.array(strips)


def close_seams(asset):
    """将参考地形边界的未封闭高度差转换成窄实体，向高侧内缩 0.1mm。"""
    for row in range(10):
        for col in range(20):
            cell = (row, col)
            seams = []
            for axis in range(2):
                for side in range(2):
                    if side == 0 and cell[axis] > 0:
                        continue
                    neighbor = [row, col]
                    neighbor[axis] += 2 * side - 1
                    a = boundary_profile(asset, cell, axis, side)
                    b = boundary_profile(asset, neighbor, axis, 1 - side)
                    cuts = np.unique(np.clip(np.r_[a[:, :2].ravel(), b[:, :2].ravel()], 0, 8))
                    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                        if right - left < 1e-6:
                            continue
                        mid = (left + right) / 2
                        aa = a[(a[:, 0] <= mid) & (a[:, 1] >= mid)]
                        bb = b[(b[:, 0] <= mid) & (b[:, 1] >= mid)]
                        za, zb = aa[:, 3].max(), bb[:, 3].max()
                        if abs(za - zb) < 1e-5:
                            continue
                        low, high = sorted((za, zb))
                        # 已有实体侧壁覆盖的界面不另建碰撞体，仅补贯通缺口。
                        z = low
                        for floor, ceiling in sorted(np.vstack((aa, bb))[:, 2:].tolist()):
                            if floor <= z + 1e-6:
                                z = max(z, ceiling)
                        if z >= high - 1e-6:
                            continue
                        direction = (1 - 2 * side) * (1 if za > zb else -1)
                        boundary = side * 8
                        lo, hi = sorted((boundary, boundary + direction * 1e-4))
                        xy = (lo, left, hi, right) if axis == 0 else (left, lo, right, hi)
                        seams.append((*xy, z, high, -1))
            asset[f"r{row}c{col}s"] = np.array(seams, dtype=np.float64).reshape(-1, 7)


def pack(seed, terrain_cfg, detector, crop_region):
    """保留 LocoLab 的列分配、行内难度扰动、尺寸、表面和出生原点。"""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    names = list(terrain_cfg.sub_terrains)
    proportions = np.array([terrain_cfg.sub_terrains[name].proportion for name in names])
    columns = np.searchsorted(np.cumsum(proportions / proportions.sum()), np.arange(20) / 20 + 0.001)
    data = {"columns": np.array(names)[columns], "seed": np.array(seed)}
    # 固定在扫描栅格节点上，独立向作者原三角面投射测试射线；避开整分米的立边。
    nodes = np.arange(1, 400)
    probe_xy = np.random.default_rng(722).choice(nodes[nodes % 5 != 0], size=(32, 2)) * 0.02
    reference_z = np.empty((10, 20, len(probe_xy)))
    origins = np.empty((10, 20, 3))
    difficulties = np.empty((10, 20))
    edge_meshes = []
    for col, family_id in enumerate(columns):
        family = names[family_id]
        cfg = terrain_cfg.sub_terrains[family].copy()
        cfg.size = terrain_cfg.size
        cfg.horizontal_scale, cfg.vertical_scale = terrain_cfg.horizontal_scale, terrain_cfg.vertical_scale
        cfg.slope_threshold = terrain_cfg.slope_threshold
        for row in range(10):
            difficulty = (row + rng.uniform()) / 10
            difficulties[row, col] = difficulty
            meshes, origin = cfg.function(difficulty, cfg)
            origins[row, col] = origin
            whole = trimesh.util.concatenate(meshes)
            start = np.column_stack((probe_xy, np.full(len(probe_xy), 4.)))
            xyz, ray_ids, _ = whole.ray.intersects_location(start, np.tile((0., 0., -1.), (len(start), 1)),
                                                          multiple_hits=False)
            if len(ray_ids) != len(start):
                raise ValueError(f"Missing reference ray hit at row {row}, column {col}")
            reference_z[row, col, ray_ids] = xyz[:, 2]
            key = f"r{row}c{col}"
            pieces = []
            for mesh in meshes:
                lower, upper = mesh.bounds
                if np.any(upper[:2] == lower[:2]):
                    # 沟底的竖直挡面由内侧坑底箱体的侧壁表示。
                    continue
                heights = upward_grid(mesh, family)
                if not np.isfinite(heights).all() or heights.min() < lower[2] - 1e-5 or heights.max() > upper[2] + 1e-5:
                    raise ValueError(f"Surface grid exceeds the original mesh bounds at row {row}, column {col}")
                top = float(heights.max())
                bottom = float(lower[2])
                if lower[2] == upper[2]:
                    # 原始地面只有朝上的面；MuJoCo 实体地面须覆盖相邻深坑下方。
                    bottom = min(-4., top - 0.01)
                field_id = -1
                if np.ptp(heights) > 0:
                    field_id = len(pieces)
                    data[f"{key}f{field_id}"] = heights.astype(np.float32)
                pieces.append((*lower[:2], *upper[:2], bottom, top, field_id))
            data[key] = np.asarray(pieces)
            if family in CROPS:
                for part in meshes:
                    part.face_attributes.setdefault("exclude_virtual_obstacle", np.zeros(len(part.faces), dtype=bool))
                mesh = crop_region(trimesh.util.concatenate(meshes), CROPS[family], origin=origin)
                mesh.apply_translation((row * 8 - 40, col * 8 - 80, 0))
                edge_meshes.append(mesh)
        print(f"Packed column {col + 1}/20 ({family})", flush=True)
    close_seams(data)
    mesh = trimesh.util.concatenate(edge_meshes)
    mesh.merge_vertices()
    raw = detector._filter_edges_by_direction(detector._upper_surface_edges(mesh))
    cylinders = detector.process_edges(raw).reshape(-1, 2, 3)
    data.update(origins=origins, difficulties=difficulties, edges=cylinders, probe_xy=probe_xy, reference_z=reference_z)
    return data


def main():
    from locolab.utils.terrains import GreedyconcatEdgeCylinderCfg
    from locolab.utils.terrains.terrain_importer import TerrainImporter
    from proof_parkour.utils.terrains.terrains_cfg import PARKOUR_TERRAINS_CFG

    parser = argparse.ArgumentParser(description="Export a seeded PROOF parkour terrain asset")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    cfg = GreedyconcatEdgeCylinderCfg(cylinder_radius=0.05, min_points=2, max_edge_direction_z_ratio=0.2)
    data = pack(args.seed, PARKOUR_TERRAINS_CFG, cfg.class_type(cfg), TerrainImporter._select_mesh_region)
    ASSETS.mkdir(exist_ok=True)
    np.savez_compressed(ASSETS / f"proof_parkour_{args.seed}.npz", **data)


if __name__ == "__main__":
    main()
