# Copyright (c) 2026-2027 zh
"""内容：依据真实 Go2 MJCF，以同一广义状态计算髋、膝、足位置和足速。

父子树递推移植自 BlinkMimic d264658 的 src/blinkmimic/kinematics.py，
并按本项目足端 site 增加刚体偏置的位置及速度运输。许可见 LICENSE_blinkmimic。
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from torch import nn

from src.assets.robots.go2.go2_reference import GO2_XML

from .reference_rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat
from .state_spec import FOOT_NAMES, JOINT_NAMES


class Go2Kinematics(nn.Module):
    """当前参考 Go2 的可微正运动学；几何常量从资产读取并注册为 buffer。"""

    def __init__(self, device="cpu"):
        super().__init__()
        model = mujoco.MjModel.from_xml_path(str(GO2_XML))
        joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        bodies, parents, joints, axes, offsets, rotations = [], [], [], [], [], []
        mj_to_local = {}
        for body in range(1, model.nbody):
            name = model.body(body).name
            joint = int(model.body_jntadr[body])
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                parents.append(-1)
                joints.append(-1)
            else:
                joint_name = model.joint(joint).name
                parent = int(model.body_parentid[body])
                parents.append(mj_to_local[parent])
                joints.append(joint_indices[joint_name])
            mj_to_local[body] = len(bodies)
            bodies.append(name)
            axes.append(model.jnt_axis[joint].copy())
            offsets.append(model.body_pos[body].copy())
            rotations.append(model.body_quat[body].copy())
        self.parents, self.joints = tuple(parents), tuple(joints)
        self.num_bodies = len(bodies)
        self.limb_indices = tuple(bodies.index(f"{leg}_{link}") for link in ("thigh", "calf") for leg in FOOT_NAMES)
        sites = [model.site(name).id for name in FOOT_NAMES]
        self.site_parents = tuple(mj_to_local[int(model.site_bodyid[site])] for site in sites)
        self.register_buffer("axes", torch.tensor(np.array(axes), dtype=torch.float32, device=device))
        self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.float32, device=device))
        self.register_buffer("rotations", quat_to_mat(torch.tensor(np.array(rotations), dtype=torch.float32, device=device)))
        self.register_buffer("site_offsets", torch.tensor(model.site_pos[sites].copy(), dtype=torch.float32, device=device))

    def foot_yaw(self, rotation, joints):
        """四腿按三级链并行恢复calf航向，各腿仍使用各自MJCF常量。"""
        axes = self.axes[1:].reshape(4, 3, 3)
        rotations = self.rotations[1:].reshape(4, 3, 3, 3)
        angles = joints.reshape(joints.shape[:-1] + (4, 3))
        feet = rotation.unsqueeze(-3)
        for level in range(3):
            offset = mm3(feet, rotations[:, level])
            feet = mm3(offset, axis_angle_to_mat(axes[:, level], angles[..., level]))
        return torch.atan2(feet[..., 1, 0], feet[..., 0, 0])

    def _calf_state(self, p, R, v, w, q, qd):
        """四腿并行恢复髋、膝和 calf 刚体的位置、姿态及速度，保留前置批次维度。

        p/R/v/w 是根部世界位置、旋转、线速度与角速度；q/qd 为 JOINT_NAMES
        顺序的真实关节角和速度。几何 buffer 与输入应在同一设备和精度。
        """
        axes = self.axes[1:].reshape(4, 3, 3)
        offsets = self.offsets[1:].reshape(4, 3, 3)
        rotations = self.rotations[1:].reshape(4, 3, 3, 3)
        angles = q.reshape(q.shape[:-1] + (4, 3))
        rates = qd.reshape(qd.shape[:-1] + (4, 3))
        position, rotation = p.unsqueeze(-2), R.unsqueeze(-3)
        velocity, angular = v.unsqueeze(-2), w.unsqueeze(-2)
        limbs = []
        for level in range(3):
            displacement = mv3(rotation, offsets[:, level])
            position = position + displacement
            rotation_offset = mm3(rotation, rotations[:, level])
            rotation = mm3(rotation_offset, axis_angle_to_mat(axes[:, level], angles[..., level]))
            omega = angular + mv3(rotation_offset, axes[:, level]) * rates[..., level, None]
            velocity = velocity + torch.cross(angular, displacement, dim=-1)
            angular = omega
            if level > 0:
                limbs.append(position)
        return limbs, position, rotation, velocity, angular

    def forward(self, p, R, v, w, q, qd):
        """四腿并行恢复髋、膝、足位置与足速。"""
        limbs, position, rotation, velocity, angular = self._calf_state(p, R, v, w, q, qd)
        displacement = mv3(rotation, self.site_offsets)
        feet = position + displacement
        foot_velocities = velocity + torch.cross(angular, displacement, dim=-1)
        return torch.cat((*limbs, feet), dim=-2), foot_velocities

    def foot_volume_points(self, p, R, v, w, q, qd, volume_offsets):
        """每足四点在 calf 系相对 site 偏移，刚体运输得到世界位置与速度。"""
        _, position, rotation, velocity, angular = self._calf_state(p, R, v, w, q, qd)
        displacement = mv3(rotation.unsqueeze(-3), self.site_offsets[:, None] + volume_offsets)
        points = position.unsqueeze(-2) + displacement
        velocities = velocity.unsqueeze(-2) + torch.cross(angular.unsqueeze(-2), displacement, dim=-1)
        return points.flatten(-3, -2), velocities.flatten(-3, -2)
