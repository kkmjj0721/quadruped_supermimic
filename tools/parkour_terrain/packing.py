# Copyright (c) 2026-2027 zh
"""内容：将官方几何转换为盒体、原高度场和精确凸棱柱，补齐接缝并保存独立真值探针。"""

import random
from copy import deepcopy
from itertools import pairwise

import numpy as np
import trimesh

from .edges import detect_edges, select_mesh_region
from .presets import CROPS
from .primitives import convert_height_field_to_mesh


def boundary_profile(asset, cell, axis, side):
    """读取与地块边界相接的完整顶面，各来源地形在最外圈关闭粗糙扰动。"""
    row, col = cell
    if not (0 <= row < int(asset["num_rows"]) and 0 <= col < int(asset["num_cols"])):
        return np.array([[0.0, asset["patch_size"][1 - axis], -1.0, 0.0]])
    key = f"r{row}c{col}"
    strips = []
    for x0, y0, x1, y1, bottom, top, field_id in asset[key]:
        bounds = ((x0, x1), (y0, y1))
        if abs(bounds[axis][side] - side * asset["patch_size"][axis]) > 1e-06:
            continue
        if field_id >= 0:
            heights = asset[f"{key}f{int(field_id)}"]
            rim = np.take(heights, side * (heights.shape[1 - axis] - 1), axis=1 - axis)
            assert np.ptp(rim) < 1e-05, "Source mesh rim is not level"
            top = rim.mean()
        strips.append((*bounds[1 - axis], bottom, top))
    for triangle, bottom in zip(asset[key + "t"], asset[key + "b"], strict=True):
        rim = triangle[np.abs(triangle[:, axis] - side * asset["patch_size"][axis]) < 1e-6]
        if len(rim) < 2:
            continue
        assert np.ptp(rim[:, 2]) < 1e-5, "Source triangle rim is not level"
        strips.append((rim[:, 1 - axis].min(), rim[:, 1 - axis].max(), bottom, rim[:, 2].mean()))
    return np.array(strips)


def close_seams(asset):
    """将参考地形边界的未封闭高度差转换成窄实体，向高侧内缩 0.1mm。"""
    for row in range(int(asset["num_rows"])):
        for col in range(int(asset["num_cols"])):
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
                    cuts = np.unique(
                        np.clip(np.r_[a[:, :2].ravel(), b[:, :2].ravel()], 0, asset["patch_size"][1 - axis])
                    )
                    for left, right in pairwise(cuts):
                        if right - left < 1e-06:
                            continue
                        mid = (left + right) / 2
                        aa = a[(a[:, 0] <= mid) & (a[:, 1] >= mid)]
                        bb = b[(b[:, 0] <= mid) & (b[:, 1] >= mid)]
                        za, zb = (aa[:, 3].max(), bb[:, 3].max())
                        if abs(za - zb) < 1e-05:
                            continue
                        low, high = sorted((za, zb))
                        z = low
                        for floor, ceiling in sorted(np.vstack((aa, bb))[:, 2:].tolist()):
                            if floor <= z + 1e-06:
                                z = max(z, ceiling)
                        if z >= high - 1e-06:
                            continue
                        direction = (1 - 2 * side) * (1 if za > zb else -1)
                        boundary = side * asset["patch_size"][axis]
                        lo, hi = sorted((boundary, boundary + direction * 0.0001))
                        xy = (lo, left, hi, right) if axis == 0 else (left, lo, right, hi)
                        seams.append((*xy, z, high, -1))
            asset[f"r{row}c{col}s"] = np.array(seams, dtype=np.float64).reshape(-1, 7)


def regular_rectangles(mask):
    """将完整未移动单元无重叠地合并为矩形，不跨越任何移动顶点。"""
    remaining = mask.copy()
    rectangles = []
    for x in range(remaining.shape[0]):
        for y in range(remaining.shape[1]):
            if not remaining[x, y]:
                continue
            y1 = y
            while y1 < remaining.shape[1] and remaining[x, y1]:
                y1 += 1
            x1 = x + 1
            while x1 < remaining.shape[0] and remaining[x1, y:y1].all():
                x1 += 1
            remaining[x:x1, y:y1] = False
            rectangles.append((x, y, x1, y1))
    return rectangles


def exact_heightfield_parts(mesh, cfg):
    """规则原单元保留原高度场，移动顶点相邻单元保留原三角面。"""
    heights = mesh.metadata["source_height_field"]
    vertices, faces = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    regular, _ = convert_height_field_to_mesh(heights, cfg.horizontal_scale, cfg.vertical_scale, None)
    moved = np.any(vertices[:, :2] != regular[:, :2], axis=1).reshape(heights.shape)
    changed = moved[:-1, :-1] | moved[1:, :-1] | moved[:-1, 1:] | moved[1:, 1:]
    grid = vertices.reshape(*heights.shape, 3)
    bottom = float(vertices[:, 2].min()) - 0.01
    rectangles = []
    for x0, y0, x1, y1 in regular_rectangles(~changed):
        bounds = (*grid[x0, y0, :2], *grid[x1, y1, :2])
        rectangles.append((bounds, grid[x0 : x1 + 1, y0 : y1 + 1, 2].T.copy(), bottom))
    affected = faces.reshape(heights.shape[0] - 1, heights.shape[1] - 1, 2, 3)[changed].reshape(-1, 3)
    triangles = vertices[affected].astype(np.float64)
    area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2]
    # 竖直原面由相邻实体侧壁保留；它自身没有可挤出的 XY 面积。
    triangles = triangles[area != 0]
    return rectangles, triangles, np.full(len(triangles), bottom, dtype=np.float64)


def upward_grid(mesh):
    """读取未发生 XY 变形的矩形部件原始顶面节点。"""
    top = np.asarray(mesh.vertices)[np.unique(mesh.faces[mesh.face_normals[:, 2] > 0.01])]
    x, ix = np.unique(top[:, 0], return_inverse=True)
    y, iy = np.unique(top[:, 1], return_inverse=True)
    assert len(np.unique(iy * len(x) + ix)) == len(x) * len(y), "Incomplete rectangular source top"
    heights = np.empty((len(y), len(x)), dtype=np.float32)
    heights[iy, ix] = top[:, 2]
    return heights


def pack(seed, terrain_cfg, num_rows=10, num_cols=20):
    """保留官方列分配与逐列随机流，动态支持行列数及地块尺寸。"""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    names = list(terrain_cfg.sub_terrains)
    weights = np.asarray([terrain_cfg.sub_terrains[name].proportion for name in names])
    columns = np.searchsorted(np.cumsum(weights / weights.sum()), np.arange(num_cols) / num_cols + 0.001)
    data = {
        "format_version": np.array("proof_parkour_asset_v3"),
        "seed": np.array(seed),
        "num_rows": np.array(num_rows),
        "num_cols": np.array(num_cols),
        "patch_size": np.array(terrain_cfg.size, dtype=np.float64),
        "border_width": np.array(terrain_cfg.border_width, dtype=np.float64),
        "edge_radius": np.array(terrain_cfg.edge_radius, dtype=np.float64),
        "columns": np.asarray(names)[columns],
    }
    probe_rng = np.random.default_rng(722)
    # 官方基线沿用历史非整数分米探针；坐标不消费地形生成随机流。
    probe_nodes = [np.arange(1, round(extent / 0.02)) for extent in terrain_cfg.size]
    if terrain_cfg.size[0] == terrain_cfg.size[1]:
        nodes = probe_nodes[0]
        probe_xy = probe_rng.choice(nodes[nodes % 5 != 0], size=(32, 2)) * 0.02
    else:
        probe_xy = np.column_stack([probe_rng.choice(nodes[nodes % 5 != 0], size=32) * 0.02 for nodes in probe_nodes])
    reference_z = np.empty((num_rows, num_cols, len(probe_xy)))
    origins = np.empty((num_rows, num_cols, 3))
    difficulties = np.empty((num_rows, num_cols))
    edge_meshes = []
    half = np.asarray((num_rows, num_cols)) * terrain_cfg.size / 2
    for col, family_id in enumerate(columns):
        family = names[family_id]
        cfg = deepcopy(terrain_cfg.sub_terrains[family])
        cfg.size = terrain_cfg.size
        cfg.horizontal_scale, cfg.vertical_scale = terrain_cfg.horizontal_scale, terrain_cfg.vertical_scale
        cfg.slope_threshold = terrain_cfg.slope_threshold
        for row in range(num_rows):
            difficulty = (row + rng.uniform()) / num_rows
            difficulties[row, col] = difficulty
            meshes, origin = cfg.function(difficulty, cfg)
            origins[row, col] = origin
            whole = trimesh.util.concatenate(meshes)
            start = np.column_stack((probe_xy, np.full(len(probe_xy), whole.bounds[1, 2] + 1.0)))
            xyz, ray_ids, _ = whole.ray.intersects_location(
                start, np.tile((0.0, 0.0, -1.0), (len(start), 1)), multiple_hits=False
            )
            assert len(ray_ids) == len(start), f"Missing reference ray hit at row {row}, column {col}"
            reference_z[row, col, ray_ids] = xyz[:, 2]
            key = f"r{row}c{col}"
            pieces, top_triangles, bottom_planes = [], [], []
            for mesh in meshes:
                lower, upper = mesh.bounds
                if np.any(upper[:2] == lower[:2]):
                    continue
                if "source_height_field" in mesh.metadata:
                    rectangles, triangles, bottoms = exact_heightfield_parts(mesh, cfg)
                    top_triangles.append(triangles)
                    bottom_planes.append(bottoms)
                else:
                    heights = upward_grid(mesh)
                    bottom = float(lower[2])
                    if lower[2] == upper[2]:
                        bottom = min(-4.0, float(heights.max()) - 0.01)
                    rectangles = [((*lower[:2], *upper[:2]), heights, bottom)]
                for bounds, heights, bottom in rectangles:
                    top = float(heights.max())
                    field_id = -1
                    if np.ptp(heights) > 0:
                        field_id = len(pieces)
                        data[f"{key}f{field_id}"] = heights
                    pieces.append((*bounds, bottom, top, field_id))
            data[key] = np.asarray(pieces, dtype=np.float64).reshape(-1, 7)
            data[key + "t"] = np.concatenate(top_triangles) if top_triangles else np.empty((0, 3, 3), dtype=np.float64)
            data[key + "b"] = np.concatenate(bottom_planes) if bottom_planes else np.empty(0, dtype=np.float64)
            if family in CROPS:
                for part in meshes:
                    part.face_attributes["exclude_virtual_obstacle"] = np.zeros(len(part.faces), dtype=bool)
                mesh = select_mesh_region(trimesh.util.concatenate(meshes), CROPS[family], origin)
                mesh.apply_translation((row * terrain_cfg.size[0] - half[0], col * terrain_cfg.size[1] - half[1], 0))
                edge_meshes.append(mesh)
        print(f"Packed column {col + 1}/{num_cols} ({family})", flush=True)
    close_seams(data)
    if edge_meshes:
        mesh = trimesh.util.concatenate(edge_meshes)
        mesh.merge_vertices()
        edges = detect_edges(mesh)
    else:
        edges = np.empty((0, 2, 3), dtype=np.float32)
    data.update(origins=origins, difficulties=difficulties, edges=edges, probe_xy=probe_xy, reference_z=reference_z)
    return data
