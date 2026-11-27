# Copyright (c) 2026-2027 zh
"""内容：补全 Parkour 浏览器地形的高度场实体表面与统一米制纹理。"""

import mujoco
import numpy as np
import trimesh
from PIL import Image
from mjlab.viewer import ViserPlayViewer
from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh


def _heightfield_mesh(model, geom_id):
    """沿用上游顶面三角剖分，按碰撞高度场的 base 深度封闭侧壁和底面。"""
    mesh = create_primitive_mesh(model, geom_id)
    field_id = model.geom_dataid[geom_id]
    rows, cols = model.hfield_nrow[field_id], model.hfield_ncol[field_id]
    boundary = np.concatenate((
        np.arange(cols), np.arange(1, rows) * cols + cols - 1,
        (rows - 1) * cols + np.arange(cols - 2, -1, -1), np.arange(rows - 2, 0, -1) * cols,
    ))
    bottom = mesh.vertices[boundary].copy()
    bottom[:, 2] = -model.hfield_size[field_id, 3]
    lower = np.arange(len(boundary)) + len(mesh.vertices)
    following = np.roll(np.arange(len(boundary)), -1)
    center = len(mesh.vertices) + len(boundary)
    sides = np.concatenate((
        np.column_stack((boundary, lower, boundary[following])),
        np.column_stack((boundary[following], lower, lower[following])),
        np.column_stack((np.full(len(boundary), center), lower[following], lower)),
    ))
    return trimesh.Trimesh(
        vertices=np.vstack((mesh.vertices, bottom, [0., 0., bottom[0, 2]])),
        faces=np.vstack((mesh.faces, sides)), process=False,
    )


def _terrain_mesh(model, geom_ids):
    """显示网格只读取编译后的几何；同一地面材质按米铺设，避免每块重新拉伸。"""
    builders = {
        mujoco.mjtGeom.mjGEOM_BOX: create_primitive_mesh,
        mujoco.mjtGeom.mjGEOM_HFIELD: _heightfield_mesh,
        mujoco.mjtGeom.mjGEOM_MESH: mujoco_mesh_to_trimesh,
    }
    pieces = []
    for geom_id in geom_ids:
        mesh = builders[model.geom_type[geom_id]](model, geom_id)
        transform = np.eye(4)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        transform[:3, :3] = rotation.reshape(3, 3)
        transform[:3, 3] = model.geom_pos[geom_id]
        mesh.apply_transform(transform)
        # 统一在合并后设置材质，不继承上游盒体／棱柱的纯白色回退。
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        pieces.append(mesh)
    mesh = trimesh.util.concatenate(pieces)
    material_id = model.geom_matid[geom_ids[0]]
    texture_id = model.mat_texid[material_id, mujoco.mjtTextureRole.mjTEXROLE_RGB]
    width, height, channels = model.tex_width[texture_id], model.tex_height[texture_id], model.tex_nchannel[texture_id]
    address = model.tex_adr[texture_id]
    pixels = model.tex_data[address:address + width * height * channels].reshape(height, width, channels)
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=model.mat_rgba[material_id], baseColorTexture=Image.fromarray(np.flipud(pixels)),
        metallicFactor=0., roughnessFactor=1.,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.vertices[:, :2] * model.mat_texrepeat[material_id], material=material,
    )
    return mesh


class SuperMimicViserPlayViewer(ViserPlayViewer):
    """保留上游交互与分组控制，只替换 Parkour 的静态地形显示网格。"""

    def setup(self):
        super().setup()
        if self.env.unwrapped.adapter.cfg.terrain.mode != "parkour":
            return
        scene = self._scene
        model = scene.mj_model
        body_id = model.body("terrain").id
        terrain_ids = np.flatnonzero(model.geom_bodyid == body_id)
        for group_id in np.unique(model.geom_group[terrain_ids]):
            geom_ids = terrain_ids[model.geom_group[terrain_ids] == group_id]
            mesh = _terrain_mesh(model, geom_ids)
            keys = [key for key in scene._fixed_geom_handles if key[:2] == (body_id, group_id)]
            with scene.server.atomic():
                for key in keys:
                    scene._fixed_geom_handles.pop(key).remove()
                handle = scene.server.scene.add_mesh_trimesh(
                    f"/fixed_bodies/terrain/group{group_id}", mesh,
                    position=scene.mj_data.xpos[body_id], wxyz=scene.mj_data.xquat[body_id],
                    visible=scene.geom_groups_visible[group_id], cast_shadow=False, receive_shadow=0.2,
                )
                scene._fixed_geom_handles[(body_id, group_id, 0)] = handle
