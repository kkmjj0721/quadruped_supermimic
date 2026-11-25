# Copyright (c) 2026-2027 zh
"""内容：按 proof-loco ae910b8 的地形顺序定义二阶段命令、高程及奖励的适用范围。"""

import torch
from mjlab.envs.mdp.events import resolve_env_ids


PARKOUR_FAMILIES = (
    "stairs_30", "stairs_30_inv", "slope", "slope_inv", "discrete",
    "stairs_30_straight", "stairs_30_inv_straight", "hurdle", "gap", "climb", "stairs_inv_high_straight",
)
SPECIAL_FAMILIES = ("gap", "climb", "hurdle", "stairs_inv_high_straight")
FORWARD_FAMILIES = (*SPECIAL_FAMILIES, "stairs_30_straight", "stairs_30_inv_straight")
EDGE_FAMILIES = ("gap", "climb", "stairs_inv_high_straight")


def reset_parkour_joints(env, env_ids, *, scale_range):
    """按作者二阶段 reset 比例逐关节采样，约束到资产软限位，以零速度写回选中行。"""
    env_ids = resolve_env_ids(env, env_ids)
    robot = env.scene["robot"]
    nominal = robot.data.default_joint_pos[env_ids]
    positions = nominal * torch.empty_like(nominal).uniform_(*scale_range)
    lower, upper = robot.data.soft_joint_pos_limits[env_ids].unbind(-1)
    robot.write_joint_state_to_sim(positions.clamp(lower, upper), torch.zeros_like(positions), env_ids=env_ids)
