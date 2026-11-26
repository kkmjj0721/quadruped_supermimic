# Copyright (c) 2026-2027 zh
"""内容：声明 PROOF 二阶段地形族及乘法关节重置，族号随状态进入回放。"""

import torch
from mjlab.envs.mdp.events import resolve_env_ids


PARKOUR_FAMILIES = (
    "stairs_30", "stairs_30_inv", "slope", "slope_inv", "discrete",
    "stairs_30_straight", "stairs_30_inv_straight", "hurdle", "gap", "climb", "stairs_inv_high_straight",
)
SPECIAL_FAMILIES = ("gap", "climb", "hurdle", "stairs_inv_high_straight")
FORWARD_FAMILIES = (*SPECIAL_FAMILIES, "stairs_30_straight", "stairs_30_inv_straight")
EDGE_FAMILIES = ("gap", "climb", "stairs_inv_high_straight")


def reset_parkour_joints(env, env_ids, scale_range, asset_cfg):
    """沿用上游默认角乘均匀比例、软关节限位及零速度重置。"""
    env_ids = resolve_env_ids(env, env_ids)
    robot = env.scene[asset_cfg.name]
    position = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    position *= torch.empty_like(position).uniform_(*scale_range)
    limits = robot.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(position, torch.zeros_like(position), joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
