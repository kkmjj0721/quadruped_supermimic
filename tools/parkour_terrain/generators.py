# Copyright (c) 2026-2027 zh
"""内容：保留官方随机流的七类离线地形生成器。

来源：LocoLab@37e567d，许可证见 LICENSE_LocoLab。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import trimesh

from .primitives import (
    height_field_to_mesh,
    make_border,
    make_plane,
    mesh_pyramid_stairs_base,
)
from .roughness import (
    _finalize_height_field,
    generate_roughness_height_field,
    sample_height_field,
    should_apply_roughness,
)

# ---------- 沟壑 ----------


def mesh_straight_gap_terrain(difficulty, cfg):
    gap_min, gap_max = cfg.gap_width_range
    depth_min, depth_max = cfg.gap_depth_range
    platform_min, platform_max = cfg.center_platform_width_range
    island_w_min, island_w_max = cfg.island_width_range
    offset_min, offset_max = cfg.island_y_offset_range
    height_min, height_max = cfg.island_height_offset_range
    bottom_width_range = cfg.gap_bottom_width_range
    if bottom_width_range is not None:
        bottom_min, bottom_max = bottom_width_range
    outer_depth_range = cfg.gap_outer_bottom_depth_range
    if outer_depth_range is not None:
        outer_min, outer_max = outer_depth_range
    gap_width = gap_min + difficulty * (gap_max - gap_min)
    if cfg.gap_depth_type == "difficulty":
        gap_depth = depth_min + difficulty * (depth_max - depth_min)
    elif cfg.gap_depth_type == "random":
        gap_depth = float(np.random.uniform(depth_min, depth_max))
    else:
        raise ValueError(
            f"cfg.gap_depth_type must be 'difficulty' or 'random'. Current value is `{cfg.gap_depth_type}`."
        )
    cx, cy = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    if bottom_width_range is None:
        bottom_y0, bottom_y1 = (0.0, float(cfg.size[1]))
    else:
        bottom_width = min(float(np.random.uniform(bottom_min, bottom_max)), float(cfg.size[1]))
        bottom_y0 = float(np.clip(cy - 0.5 * bottom_width, 0.0, cfg.size[1]))
        bottom_y1 = float(np.clip(cy + 0.5 * bottom_width, 0.0, cfg.size[1]))
    outer_depth = None
    if outer_depth_range is not None and bottom_y1 - bottom_y0 < float(cfg.size[1]) - 1e-06:
        outer_depth = gap_depth + float(np.random.uniform(outer_min, outer_max))
    center_platform_width = float(np.random.uniform(platform_min, platform_max))
    inner_left = max(cx - 0.5 * center_platform_width, 0.0)
    inner_right = min(cx + 0.5 * center_platform_width, cfg.size[0])

    def _pack_side(budget):
        budget = max(float(budget), 0.0)
        if budget + 1e-09 < gap_width:
            return (0, 0.0, [])
        remaining = budget - gap_width
        islands: list[tuple[float, float, float, float]] = []
        while True:
            room = remaining - gap_width
            if room + 1e-08 < island_w_min:
                break
            x_width = min(float(np.random.uniform(island_w_min, island_w_max)), room)
            islands.append(
                (
                    x_width,
                    float(np.random.uniform(island_w_min, island_w_max)),
                    float(np.random.uniform(offset_min, offset_max)),
                    float(np.random.uniform(height_min, height_max)),
                )
            )
            remaining -= x_width + gap_width
        return (len(islands) + 1, gap_width, islands)

    def _edge_landing_reserve():
        return 0.5 * float(np.random.uniform(island_w_min, island_w_max))

    left_n, left_gap, left_islands = _pack_side(max(inner_left - _edge_landing_reserve(), 0.0))
    right_n, right_gap, right_islands = _pack_side(max(cfg.size[0] - inner_right - _edge_landing_reserve(), 0.0))
    outer_left = inner_left - (left_n * left_gap + sum((width for width, *_ in left_islands)))
    outer_right = inner_right + (right_n * right_gap + sum((width for width, *_ in right_islands)))

    def make_floor(y0, y1, depth):
        if y1 - y0 <= 1e-06:
            return None
        plane = make_plane((cfg.size[0], y1 - y0), height=-depth, center_zero=False)
        if y0 > 0.0:
            plane.apply_translation((0.0, y0, 0.0))
        return plane

    def make_riser(y, z_lo, z_hi):
        if z_hi - z_lo <= 1e-06:
            return None
        vertices = np.array([[0.0, y, z_lo], [cfg.size[0], y, z_lo], [cfg.size[0], y, z_hi], [0.0, y, z_hi]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    floors = [make_floor(bottom_y0, bottom_y1, gap_depth)]
    if outer_depth is not None:
        floors.extend(
            (
                make_floor(0.0, bottom_y0, outer_depth),
                make_floor(bottom_y1, float(cfg.size[1]), outer_depth),
                make_riser(bottom_y0, -outer_depth, -gap_depth),
                make_riser(bottom_y1, -outer_depth, -gap_depth),
            )
        )
    floors = [mesh for mesh in floors if mesh is not None]
    meshes = list(floors)
    overhang_thickness = 0.05

    def add_span(x0, x1, y0, y1, z0, z1):
        if x1 - x0 <= 1e-06 or y1 - y0 <= 1e-06 or z1 - z0 <= 1e-06:
            return
        meshes.append(
            trimesh.creation.box(
                (x1 - x0, y1 - y0, z1 - z0),
                trimesh.transformations.translation_matrix(((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5)),
            )
        )

    def add_block(x0, x1, y_offset, y_width, top_z):
        x0 = float(np.clip(x0, 0.0, cfg.size[0]))
        x1 = float(np.clip(x1, 0.0, cfg.size[0]))
        if x1 - x0 <= 1e-06:
            return
        y0 = float(np.clip(cy + y_offset - 0.5 * y_width, 0.0, cfg.size[1]))
        y1 = float(np.clip(cy + y_offset + 0.5 * y_width, 0.0, cfg.size[1]))
        if y1 - y0 <= 1e-06:
            return
        floor_z = -gap_depth
        if top_z - floor_z <= 1e-06:
            return
        inside0 = max(y0, bottom_y0)
        inside1 = min(y1, bottom_y1)
        add_span(x0, x1, inside0, inside1, floor_z, top_z)
        if outer_depth is None:
            outer_z0 = top_z - min(overhang_thickness, top_z - floor_z)
        else:
            outer_z0 = -outer_depth
        if y0 < bottom_y0 - 1e-06:
            add_span(x0, x1, y0, min(y1, bottom_y0), outer_z0, top_z)
        if y1 > bottom_y1 + 1e-06:
            add_span(x0, x1, max(y0, bottom_y1), y1, outer_z0, top_z)

    add_block(0.0, outer_left, 0.0, float(np.random.uniform(island_w_min, island_w_max)), 0.0)
    add_block(inner_left, inner_right, 0.0, float(np.random.uniform(island_w_min, island_w_max)), 0.0)
    add_block(outer_right, cfg.size[0], 0.0, float(np.random.uniform(island_w_min, island_w_max)), 0.0)
    for inner, sign, gap_span, islands in (
        (inner_left, -1.0, left_gap, left_islands),
        (inner_right, 1.0, right_gap, right_islands),
    ):
        cursor = inner
        for x_width, y_width, y_offset, height in islands:
            cursor += sign * gap_span
            island_start = cursor
            cursor += sign * x_width
            lo, hi = (cursor, island_start) if sign < 0.0 else (island_start, cursor)
            add_block(lo, hi, y_offset, y_width, height)
    origin = np.array([cx, cy, 0.0])
    floor_count = len(floors)
    walkable, origin = apply_mesh_surface_details(meshes[floor_count:], origin, cfg, difficulty)
    border_width = float(cfg.border_width)
    border = []
    border_depth = gap_depth if outer_depth is None else outer_depth
    if border_width > 0.0 and border_depth > 1e-06:
        border = make_border(
            cfg.size,
            (cfg.size[0] - 2.0 * border_width, cfg.size[1] - 2.0 * border_width),
            border_depth,
            (cx, cy, -0.5 * border_depth),
        )
    return ([*meshes[:floor_count], *walkable, *border], origin)


# ---------- 攀爬与跨栏 ----------


def mesh_straight_climb_terrain(difficulty, cfg):
    length_min, length_max = cfg.wall_length_range
    platform_min, platform_max = cfg.center_platform_width_range
    spacing_min, spacing_max = cfg.spacing_range
    height_noise_min, height_noise_max = cfg.wall_height_noise_range
    edge_probability = float(cfg.apply_edge_half_wall)
    wall_width = _interpolate_decreasing_range(cfg.wall_width_range, difficulty)
    wall_height = _interpolate_range(cfg.wall_height_range, difficulty)
    wall_height += float(np.random.uniform(height_noise_min, height_noise_max))
    wall_length = min(float(np.random.uniform(length_min, length_max)), float(cfg.size[1]))
    center_platform_width = float(np.random.uniform(platform_min, platform_max))
    spacing = float(np.random.uniform(spacing_min, spacing_max))
    limit = float(cfg.size[0])

    def _outer_span(count):
        gaps = 2.0 * max(count - 1, 0) * spacing
        return center_platform_width + 2.0 * count * wall_width + gaps

    usable = limit - spacing
    denom = 2.0 * (wall_width + spacing)
    num_pairs = int(np.floor((usable - center_platform_width + 2.0 * spacing) / denom + 1e-09))
    while num_pairs >= 1 and _outer_span(num_pairs) > usable + 1e-06:
        num_pairs -= 1
    cx, cy = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    origin = np.array([cx, cy, 0.0])
    meshes = [make_plane(cfg.size, height=0.0, center_zero=False)]
    if wall_height == 0.0:
        return apply_mesh_surface_details(meshes, origin, cfg, difficulty)

    def add_wall(x, width):
        meshes.append(
            trimesh.creation.box(
                (width, wall_length, wall_height),
                trimesh.transformations.translation_matrix((x, cy, 0.5 * wall_height)),
            )
        )

    inset = 0.0
    outer_offset = 0.0
    for pair in range(num_pairs):
        if pair > 0:
            inset += spacing + wall_width
        inner = center_platform_width + 2.0 * inset
        outer_offset = 0.5 * inner + 0.5 * wall_width
        for x in (cx - outer_offset, cx + outer_offset):
            add_wall(x, wall_width)

    def place_edge_half_wall():
        if edge_probability <= 0.0:
            return False
        if edge_probability >= 1.0:
            return True
        return bool(np.random.random() < edge_probability)

    edge_width = 0.5 * wall_width
    clearance = cx - (outer_offset + 0.5 * wall_width)
    if clearance + 1e-06 >= spacing + edge_width:
        if place_edge_half_wall():
            add_wall(0.5 * edge_width, edge_width)
        if place_edge_half_wall():
            add_wall(cfg.size[0] - 0.5 * edge_width, edge_width)
    return apply_mesh_surface_details(meshes, origin, cfg, difficulty)


# ---------- 直行阶梯 ----------


def mesh_straight_stairs_terrain(difficulty, cfg):
    stair_height = cfg.stair_height_range[0] + difficulty * (cfg.stair_height_range[1] - cfg.stair_height_range[0])
    stair_width = _sample_stair_width(cfg.stair_width, cfg.stair_width_noise_range, cfg.stair_width_noise_step)
    border_width = float(cfg.border_width)
    usable_length = float(cfg.size[0] - 2.0 * border_width)
    stair_length = float(cfg.size[1] - 2.0 * border_width)
    center_target = float(np.random.uniform(cfg.center_platform_width_range[0], cfg.center_platform_width_range[1]))
    num_stairs = _sample_straight_stair_count(
        cfg, stair_width, center_target, available_length=usable_length, choose_max=True
    )
    center_platform_width = usable_length - 2 * num_stairs * stair_width
    meshes_list = []
    _add_straight_stair_border(meshes_list, cfg.size, usable_length, stair_length, border_width, stair_height)
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    for k in range(num_stairs):
        box_z = terrain_center[2] + (k + 1) * stair_height / 2.0
        box_offset = (center_platform_width + stair_width) / 2.0 + (num_stairs - k - 1) * stair_width
        box_height = (k + 1) * stair_height
        box_dims = (stair_width, stair_length, box_height)
        box_pos = (terrain_center[0] - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list += [box_right, box_left]
    box_dims = (center_platform_width, stair_length, num_stairs * stair_height)
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_stairs * stair_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    origin = np.array([terrain_center[0], terrain_center[1], num_stairs * stair_height])
    return (meshes_list, origin)


def mesh_inverted_straight_stairs_terrain(difficulty, cfg):
    stair_height = cfg.stair_height_range[0] + difficulty * (cfg.stair_height_range[1] - cfg.stair_height_range[0])
    stair_width = _sample_stair_width(cfg.stair_width, cfg.stair_width_noise_range, cfg.stair_width_noise_step)
    border_width, usable_length, stair_length = _sample_straight_stair_footprint(cfg)
    minimum_center = float(np.random.uniform(cfg.center_platform_width_range[0], cfg.center_platform_width_range[1]))
    num_stairs = _sample_straight_stair_count(
        cfg, stair_width, minimum_center, outer_width_scale=0.0, available_length=usable_length
    )
    center_platform_width = usable_length - 2 * (num_stairs - 1) * stair_width
    total_height = num_stairs * stair_height
    meshes_list = []
    _add_straight_stair_border(meshes_list, cfg.size, usable_length, stair_length, border_width, stair_height)
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    half_length = 0.5 * usable_length
    for k in range(num_stairs - 1):
        box_z = terrain_center[2] - stair_height / 2.0 - (k + 1) * stair_height
        box_offset = half_length - (k + 0.5) * stair_width
        box_dims = (stair_width, stair_length, stair_height)
        box_pos = (terrain_center[0] - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list += [box_right, box_left]
    box_dims = (usable_length, 0.2, total_height)
    box_pos_1 = (terrain_center[0], terrain_center[1] - stair_length / 2 - 0.1, terrain_center[2] - total_height / 2)
    box_pos_2 = (terrain_center[0], terrain_center[1] + stair_length / 2 + 0.1, terrain_center[2] - total_height / 2)
    box_side_1 = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_1))
    box_side_2 = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_2))
    meshes_list += [box_side_1, box_side_2]
    box_dims = (center_platform_width, stair_length, stair_height)
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - stair_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    origin = np.array([terrain_center[0], terrain_center[1], terrain_center[2] - total_height])
    return (meshes_list, origin)


# ---------- 粗糙顶面剖分 ----------


def _samples_including_ends(start, stop, max_edge):
    if abs(stop - start) <= max_edge:
        return np.array([start, stop], dtype=np.float64)
    count = int(np.ceil(abs(stop - start) / max_edge))
    return np.linspace(start, stop, count + 1, dtype=np.float64)


def _grid_faces(nx, ny, flip=False):
    if nx < 2 or ny < 2:
        return np.empty((0, 3), dtype=np.int64)
    i, j = np.mgrid[0 : nx - 1, 0 : ny - 1]
    a = i * ny + j
    b = (i + 1) * ny + j
    c = (i + 1) * ny + (j + 1)
    d = i * ny + (j + 1)
    if flip:
        return np.stack((a, d, c, a, c, b), axis=-1).reshape(-1, 3)
    return np.stack((a, b, c, a, c, d), axis=-1).reshape(-1, 3)


def _grid_z(xs, ys, z, flip=False):
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)), axis=1)
    return trimesh.Trimesh(vertices=verts, faces=_grid_faces(*xx.shape, flip=flip), process=False)


def _strip_const_y(xs, z0, z1, y, flip=False):
    xx, zz = np.meshgrid(xs, np.array([z0, z1], dtype=np.float64), indexing="ij")
    verts = np.stack((xx.ravel(), np.full(xx.size, y), zz.ravel()), axis=1)
    return trimesh.Trimesh(vertices=verts, faces=_grid_faces(*xx.shape, flip=flip), process=False)


def _strip_const_x(ys, z0, z1, x, flip=False):
    yy, zz = np.meshgrid(ys, np.array([z0, z1], dtype=np.float64), indexing="ij")
    verts = np.stack((np.full(yy.size, x), yy.ravel(), zz.ravel()), axis=1)
    return trimesh.Trimesh(vertices=verts, faces=_grid_faces(*yy.shape, flip=flip), process=False)


def _tessellate_for_height_noise(mesh, max_edge):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return mesh
    xs = np.unique(np.round(vertices[:, 0], 6))
    ys = np.unique(np.round(vertices[:, 1], 6))
    zs = np.unique(np.round(vertices[:, 2], 6))
    if len(xs) != 2 or len(ys) != 2:
        return mesh
    x0, x1 = (float(xs[0]), float(xs[1]))
    y0, y1 = (float(ys[0]), float(ys[1]))
    gx = _samples_including_ends(x0, x1, max_edge)
    gy = _samples_including_ends(y0, y1, max_edge)
    parts: list[trimesh.Trimesh] = []
    if len(zs) == 1:
        parts.append(_grid_z(gx, gy, float(zs[0])))
    elif len(zs) == 2:
        z0, z1 = (float(zs[0]), float(zs[1]))
        parts.append(_grid_z(gx, gy, z1))
        parts.append(_grid_z(np.array([x0, x1]), np.array([y0, y1]), z0, flip=True))
        parts.append(_strip_const_y(gx, z0, z1, y0, flip=False))
        parts.append(_strip_const_y(gx, z0, z1, y1, flip=True))
        parts.append(_strip_const_x(gy, z0, z1, x0, flip=True))
        parts.append(_strip_const_x(gy, z0, z1, x1, flip=False))
    else:
        return mesh
    return trimesh.util.concatenate(parts)


def _border_noise_weight(x, y, cfg):
    border = float(cfg.border_width)
    if border <= 0.0:
        border = float(cfg.horizontal_scale)
    weight = np.ones(np.shape(x), dtype=np.float64)
    weight[x < border] = 0.0
    weight[y < border] = 0.0
    weight[x > cfg.size[0] - border] = 0.0
    weight[y > cfg.size[1] - border] = 0.0
    return weight


def apply_mesh_roughness(meshes, origin, cfg, difficulty):
    if not should_apply_roughness(cfg):
        return (meshes, origin)
    noise_shape = (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1)
    noise_hf = generate_roughness_height_field(cfg, difficulty, shape=noise_shape)
    if not np.any(noise_hf):
        return (meshes, origin)
    noise_m = noise_hf.astype(np.float64) * cfg.vertical_scale
    rough_meshes = []
    for mesh in meshes:
        tessellated = _tessellate_for_height_noise(mesh, cfg.horizontal_scale)
        vertices = np.asarray(tessellated.vertices, dtype=np.float64).copy()
        dz = sample_height_field(noise_m, vertices[:, 0], vertices[:, 1], cfg.horizontal_scale)
        vertices[:, 2] += dz * _border_noise_weight(vertices[:, 0], vertices[:, 1], cfg)
        tessellated.vertices = vertices
        rough_meshes.append(tessellated)
    origin = np.asarray(origin, dtype=np.float64).copy()
    origin_weight = _border_noise_weight(origin[0:1], origin[1:2], cfg)[0]
    origin[2] += origin_weight * sample_height_field(noise_m, origin[0:1], origin[1:2], cfg.horizontal_scale)[0]
    return (rough_meshes, origin)


def _interpolate_range(value_range, difficulty):
    lower, upper = value_range
    return float(lower + np.clip(difficulty, 0.0, 1.0) * (upper - lower))


def _interpolate_decreasing_range(value_range, difficulty):
    lower, upper = value_range
    return float(upper - np.clip(difficulty, 0.0, 1.0) * (upper - lower))


# ---------- 阶梯参数采样 ----------


def _sample_stair_width(stair_width, noise_range, noise_step):
    noise_min, noise_max = noise_range
    noise_step_min = int(np.ceil(noise_min / noise_step))
    noise_step_max = int(np.floor(noise_max / noise_step))
    stair_width_noise = int(np.random.randint(noise_step_min, noise_step_max + 1)) * noise_step
    sampled = float(stair_width + stair_width_noise)
    return sampled


def _sample_straight_stair_footprint(cfg):
    border_width = float(cfg.border_width)
    usable_length = float(cfg.size[0] - 2.0 * border_width)
    max_stair_length = float(cfg.size[1] - 2.0 * border_width)
    stair_length = float(np.random.uniform(cfg.stair_length_range[0], cfg.stair_length_range[1]))
    return (border_width, usable_length, min(stair_length, max_stair_length))


def _add_straight_stair_border(meshes, size, usable_length, stair_length, border_width, stair_height):
    center_z = -0.5 * stair_height
    cx = 0.5 * size[0]
    cy = 0.5 * size[1]
    thickness_y = 0.5 * (size[1] - stair_length)
    if thickness_y > 1e-06:
        border_dims = (size[0], thickness_y, stair_height)
        for sign in (-1.0, 1.0):
            border_pos = (cx, cy + sign * (0.5 * stair_length + 0.5 * thickness_y), center_z)
            meshes.append(trimesh.creation.box(border_dims, trimesh.transformations.translation_matrix(border_pos)))
    if border_width > 1e-06:
        border_dims = (border_width, stair_length, stair_height)
        for sign in (-1.0, 1.0):
            border_pos = (cx + sign * (0.5 * usable_length + 0.5 * border_width), cy, center_z)
            meshes.append(trimesh.creation.box(border_dims, trimesh.transformations.translation_matrix(border_pos)))


def _sample_straight_stair_count(
    cfg, stair_width, center_platform_width, outer_width_scale=1.0, *, available_length, choose_max=False
):
    num_min, num_max = (int(cfg.num_stairs_range[0]), int(cfg.num_stairs_range[1]))
    terrain_length = float(available_length)
    if outer_width_scale == 1.0:
        max_stairs = int(np.floor((terrain_length - center_platform_width) / (2.0 * stair_width)))
    else:
        max_stairs = int(
            np.floor((terrain_length - center_platform_width) / (2.0 * stair_width) + (1.0 - outer_width_scale) + 1e-09)
        )
    num_min = min(num_min, max_stairs)
    num_max = min(num_max, max_stairs + 1)
    if num_min >= num_max or choose_max:
        return max_stairs if num_min >= num_max else num_max - 1
    return int(np.random.randint(num_min, num_max))


def apply_mesh_surface_details(meshes, origin, cfg, difficulty):
    return apply_mesh_roughness(meshes, origin, cfg, difficulty)


# ---------- 金字塔阶梯与高度场 ----------


def mesh_pyramid_stairs_terrain(difficulty, cfg):
    resolved = deepcopy(cfg)
    resolved.step_height_range = cfg.stair_height_range
    resolved.step_width = _sample_stair_width(cfg.stair_width, cfg.stair_width_noise_range, cfg.stair_width_noise_step)
    resolved.platform_width = cfg.center_platform_width
    meshes, origin = mesh_pyramid_stairs_base(difficulty, resolved)
    return apply_mesh_surface_details(meshes, origin, cfg, difficulty)


def _pack_height_field(height_field, cfg, height):
    origin = np.array(
        (
            (height_field.shape[0] - 1) * cfg.horizontal_scale / 2,
            (height_field.shape[1] - 1) * cfg.horizontal_scale / 2,
            float(height) * cfg.vertical_scale,
        )
    )
    return height_field, origin


@height_field_to_mesh
def pyramid_sloped_rough_terrain(difficulty, cfg):
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy
    center_platform_width = int(cfg.center_platform_width / cfg.horizontal_scale / 2)
    x_pf = width_pixels // 2 - center_platform_width
    y_pf = length_pixels // 2 - center_platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    hf_raw = _finalize_height_field(cfg, hf_raw, difficulty)
    return _pack_height_field(np.rint(hf_raw).astype(np.int16), cfg, _center_height(hf_raw))


def _center_height(height_field):
    return int(np.rint(height_field[height_field.shape[0] // 2, height_field.shape[1] // 2]))


@height_field_to_mesh
def discrete_obstacles_terrain(difficulty, cfg):
    obstacle_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obstacle_height = int(obstacle_height / cfg.vertical_scale)
    obstacle_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obstacle_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    center_platform_width = int(cfg.center_platform_width / cfg.horizontal_scale)
    obstacle_width_range = np.arange(obstacle_width_min, obstacle_width_max + 1, 4)
    obstacle_length_range = np.arange(obstacle_width_min, obstacle_width_max + 1, 4)
    if len(obstacle_width_range) == 0:
        obstacle_width_range = np.array([obstacle_width_min])
    if len(obstacle_length_range) == 0:
        obstacle_length_range = np.array([obstacle_width_min])
    height_range = np.array(
        [-obstacle_height, -obstacle_height // 2, obstacle_height // 2, obstacle_height], dtype=np.int16
    )
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    for _ in range(cfg.num_obstacles):
        width = int(np.random.choice(obstacle_width_range))
        length = int(np.random.choice(obstacle_length_range))
        width = min(width, width_pixels)
        length = min(length, length_pixels)
        x_start_range = np.arange(0, max(width_pixels - width, 0) + 1, 4)
        y_start_range = np.arange(0, max(length_pixels - length, 0) + 1, 4)
        x_start = int(np.random.choice(x_start_range))
        y_start = int(np.random.choice(y_start_range))
        hf_raw[x_start : x_start + width, y_start : y_start + length] = np.random.choice(height_range)
    center_platform_width = min(center_platform_width, width_pixels, length_pixels)
    x1 = (width_pixels - center_platform_width) // 2
    x2 = (width_pixels + center_platform_width) // 2
    y1 = (length_pixels - center_platform_width) // 2
    y2 = (length_pixels + center_platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    hf_raw = _finalize_height_field(cfg, hf_raw, difficulty)
    return _pack_height_field(np.rint(hf_raw).astype(np.int16), cfg, _center_height(hf_raw))
