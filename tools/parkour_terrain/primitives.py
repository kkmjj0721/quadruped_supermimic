# Copyright (c) 2026-2027 zh
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""内容：内化矩形几何原语、金字塔阶梯和保留陡沿的高度场转换。

来源：IsaacLab v2.3.0 terrains，LocoLab@37e567d preview；许可证见 LICENSE_LocoLab。
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps

import numpy as np
import scipy.spatial.transform as tf
import trimesh


def make_plane(size, height, center_zero=True):
    x0 = [size[0], size[1], height]
    x1 = [size[0], 0.0, height]
    x2 = [0.0, size[1], height]
    x3 = [0.0, 0.0, height]
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if center_zero:
        plane_mesh.apply_translation(-np.array([size[0] / 2.0, size[1] / 2.0, 0.0]))
    return plane_mesh


def make_border(size, inner_size, height, position):
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    box_dims = (size[0], thickness_y, height)
    box_pos = (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2])
    box_mesh_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    box_pos = (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2])
    box_mesh_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    box_dims = (thickness_x, inner_size[1], height)
    box_pos = (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2])
    box_mesh_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    box_pos = (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2])
    box_mesh_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    return [box_mesh_left, box_mesh_right, box_mesh_top, box_mesh_bottom]


def make_box(length, width, height, center, max_yx_angle=0, degrees=True):
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    euler_zyx = tf.Rotation.random().as_euler("zyx")
    if degrees:
        max_yx_angle = max_yx_angle / 180.0
    euler_zyx[1:] *= max_yx_angle
    transform[0:3, 0:3] = tf.Rotation.from_euler("zyx", euler_zyx).as_matrix()
    dims = (length, width, height)
    return trimesh.creation.box(dims, transform=transform)


def pyramid_stairs_terrain(difficulty, cfg):
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    meshes_list = []
    if cfg.border_width > 0.0 and (not cfg.holes):
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        meshes_list += make_borders
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        box_height = (k + 2) * step_height
        box_dims = (box_size[0], cfg.step_width, box_height)
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list += [box_top, box_bottom, box_right, box_left]
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])
    return (meshes_list, origin)


def inverted_pyramid_stairs_terrain(difficulty, cfg):
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height
    meshes_list = []
    if cfg.border_width > 0.0 and (not cfg.holes):
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        meshes_list += make_borders
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        box_height = total_height - (k + 1) * step_height
        box_dims = (box_size[0], cfg.step_width, box_height)
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list += [box_top, box_bottom, box_right, box_left]
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])
    return (meshes_list, origin)


def convert_height_field_to_mesh(height_field, horizontal_scale, vertical_scale, slope_threshold=None):
    num_rows, num_cols = height_field.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    hf = height_field.copy()
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3
    return (vertices, triangles)


def mesh_pyramid_stairs_base(difficulty, cfg):
    function = inverted_pyramid_stairs_terrain if cfg.inverted else pyramid_stairs_terrain
    return function(difficulty, cfg)


def height_field_to_mesh(function):
    @wraps(function)
    def wrapped(difficulty, cfg):
        width, length = (int(size / cfg.horizontal_scale) + 1 for size in cfg.size)
        border = int(cfg.border_width / cfg.horizontal_scale) + 1
        heights = np.zeros((width, length), dtype=np.int16)
        inner = deepcopy(cfg)
        inner.size = ((width - 2 * border) * cfg.horizontal_scale, (length - 2 * border) * cfg.horizontal_scale)
        samples, origin = function(difficulty, inner)
        heights[border:-border, border:-border] = samples
        origin[:2] += border * cfg.horizontal_scale
        vertices, triangles = convert_height_field_to_mesh(
            heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.metadata["source_height_field"] = heights
        return [mesh], origin

    return wrapped


def to_discrete_units(value, scale):
    return round(value / scale)
