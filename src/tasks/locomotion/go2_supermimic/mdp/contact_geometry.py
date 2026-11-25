# Copyright (c) 2026-2027 zh
"""内容：按 TGI 的局部地形查询，用 Go2 碰撞体计算球足支撑与机身净空。"""

import mujoco
import numpy as np
import torch

from src.assets.robots.go2.go2_reference import GO2_XML

from .reference_rotations import quat_to_mat
from .reference_state import FEET, HEIGHT, POSE
from .state_spec import FOOT_NAMES


class ContactGeometry:
    """几何常量来自训练资产；局部点在每一步依当前物理状态变到世界坐标。"""

    def __init__(self, surface, device):
        self.surface = surface
        model = mujoco.MjModel.from_xml_path(str(GO2_XML))
        box = model.geom("base_collision")
        capsule = model.geom("head_collision1")
        sphere = model.geom("head_collision2")
        radii = np.array([model.geom(f"{leg}_foot_collision").size[0] for leg in FOOT_NAMES])

        def tensor(array):
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        # 球足没有长轴；在水平投影圆盘中取中心、内圈及四个边缘点。
        coords = np.linspace(-1, 1, 5)
        disk = np.stack(np.meshgrid(coords, coords, indexing="ij"), -1).reshape(-1, 2)
        disk = disk[(disk**2).sum(-1) <= 1]
        self.foot_offsets = tensor(radii[:, None, None] * disk)
        self.foot_radii = tensor(radii)
        self.foot_center = len(disk) // 2

        # 身体的 7×5 地形扫描覆盖从后部箱体到头部的水平外形。
        tail = box.pos[0] - box.size[0]
        nose = max(capsule.pos[0] + capsule.size[0], sphere.pos[0] + sphere.size[0])
        half_width = max(box.size[1], capsule.size[0], sphere.size[0])
        self.base_grid = tensor(np.stack(np.meshgrid(np.linspace(tail, nose, 7), np.linspace(-half_width, half_width, 5),
                                                     indexing="ij"), -1).reshape(-1, 2))
        self.feature_dim = 1 + len(self.base_grid) + len(radii) * (1 + len(disk))

        # 两个箱面均取样，翻滚后仍包含最低角点。头部胶囊沿轴取两端与中点，
        # 与头部球体一起查询各自世界竖直方向的下半球。
        box_grid = np.stack(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 3), [-1, 1],
                                        indexing="ij"), -1).reshape(-1, 3)
        self.box_points = tensor(box.pos + box_grid * box.size)
        axial = np.array([[0, 0, z] for z in (-1, 0, 1)]) * capsule.size[1]
        self.head_centers = tensor(np.vstack((capsule.pos + axial, sphere.pos)))
        hemisphere = np.column_stack((disk, -np.sqrt(1 - (disk**2).sum(-1))))
        head_radii = np.array([capsule.size[0]] * 3 + [sphere.size[0]])
        self.head_offsets = tensor(head_radii[:, None, None] * hemisphere)

    @staticmethod
    def _world_xy(pose, offsets):
        """将航向系内的一批水平偏移变换到世界系。"""
        yaw = pose[:, 2]
        sine, cosine = yaw.sin(), yaw.cos()
        rotation = torch.stack((cosine, sine, -sine, cosine), -1).reshape(-1, 2, 2)
        return offsets @ rotation + pose[:, None, :2]

    def _foot_xy(self, pose, feet):
        local = feet[:, :, None, :2] + self.foot_offsets.to(feet)
        return self._world_xy(pose, local.flatten(1, 2)).reshape_as(local)

    def model_features(self, physical):
        """根部离地高度、机身下方地形起伏、四个球足的底部净空及足下相对中心的高差。"""
        pose = physical[:, POSE]
        feet = physical[:, FEET].reshape(-1, 4, 3)
        foot_xy = self._foot_xy(pose, feet)
        base_xy = self._world_xy(pose, self.base_grid.to(pose))
        points = torch.cat((pose[:, None, :2], base_xy, foot_xy.flatten(1, 2)), dim=1)
        heights = self.surface.height(points)
        origin, base, local = heights.split((1, base_xy.shape[1], foot_xy.shape[1] * foot_xy.shape[2]), dim=1)
        local = local.reshape_as(foot_xy[..., 0])
        center = local[..., self.foot_center]
        height = physical[:, HEIGHT]
        clearance = height + origin + feet[..., 2] - self.foot_radii.to(feet) - center
        relief = (local - center.unsqueeze(-1)).clamp(-1, 1)
        return torch.cat((height, (base - origin).clamp(-1, 1), clearance, relief.flatten(1)), -1)

    def unsupported(self, physical):
        """按 TGI 的 3 cm 高差与 1 cm 软化尺度，计算每足投影内缺乏支撑的比例。"""
        feet = physical[:, FEET].reshape(-1, 4, 3)
        xy = self._foot_xy(physical[:, POSE], feet)
        smooth = self.surface.height(xy)
        nodes = self.surface.nearest_height(xy)
        center = self.foot_center
        smooth_relief = (smooth - smooth[..., center, None]).clamp(-1, 1)
        node_relief = (nodes - nodes[..., center, None]).clamp(-1, 1)
        relief = smooth_relief + (node_relief - smooth_relief).detach()
        return torch.sigmoid((relief.abs() - .03) / .01).mean(-1)

    def body_clearance(self, physical):
        """机身箱体、头部胶囊和球体各采样点到其正下方地面的最小有符号高度。"""
        pose = physical[:, POSE]
        rotation = quat_to_mat(physical[:, -4:]).transpose(-1, -2)
        box = self.box_points.to(rotation) @ rotation
        centers = self.head_centers.to(rotation) @ rotation
        head = centers[:, :, None] + self.head_offsets.to(rotation)
        probes = torch.cat((box, head.flatten(1, 2)), dim=1)
        xy = torch.cat((pose[:, None, :2], pose[:, None, :2] + probes[..., :2]), dim=1)
        ground = self.surface.height(xy)
        gap = physical[:, HEIGHT] + ground[:, :1] + probes[..., 2] - ground[:, 1:]
        # 与 TGI 的坑沿查询一致，只将梯度传给本次决定边界的地形点。
        return gap.gather(1, gap.detach().argmin(1, keepdim=True)).squeeze(1)
