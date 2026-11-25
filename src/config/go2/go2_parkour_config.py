# Copyright (c) 2026-2027 zh
"""内容：在独立配置副本上装配 PROOF 二阶段的地形、命令、重置与奖励。"""

from copy import deepcopy
from math import pi

from .go2_reference_supermimic_config import Go2ReferenceCfg


class ParkourRewardScales:
    """proof-loco ae910b8 的二阶段教师权重，每项在任务层乘一次控制周期。"""

    track_lin_vel_xy_exp = 1.5
    track_ang_vel_z_exp = 1.0
    lin_vel_z_l2 = -1.0
    ang_vel_xy_l2 = -0.05
    body_orientation_l2 = -0.1
    stand_still = -1.0
    joint_hip_deviation_l2 = -0.2
    joint_deviation_l2 = -2.5e-3
    joint_acc_l2 = -2.5e-7
    joint_torques_l2_soft_limit = -1e-4
    joint_power_l1 = -2e-5
    joint_pos_limits = -10.0
    action_rate_l2 = -0.01
    action_smoothness_l2 = -0.01
    undesired_contacts = -1.0
    feet_air_time = 0.5
    feet_air_time_variance = -0.1
    feet_stumble = -0.5
    feet_slide = -0.025
    is_alive = 0.1
    termination_penalty = -100.0
    stuck = -2.0
    cheat = -1.0
    volume_points_penetration = -2.0


def parkour_config(source: Go2ReferenceCfg) -> Go2ReferenceCfg:
    """切换到作者的二阶段任务，沿用已有 Go2 资产、仿真和 SuperMimic 网络配方。"""
    cfg = deepcopy(source)
    cfg.terrain.mode = "parkour"
    cfg.terrain.seed = 17
    cfg.terrain.core_rows = 10
    cfg.terrain.num_cols = 20
    cfg.terrain.patch_size_m = (8.0, 8.0)
    cfg.terrain.field_resolution_m = 0.02
    cfg.terrain.outer_flat_width_m = 20.0
    cfg.terrain.max_init_level = 0
    cfg.terrain.curriculum = True
    cfg.teacher.exteroceptive_dropout = 0.4
    cfg.commands.initial_ranges = ((-0.5, 1.1), (-0.5, 0.5), (-1.5, 1.5))
    cfg.commands.heading_range = (-pi, pi)
    cfg.commands.heading_control_stiffness = 0.8
    cfg.commands.rel_standing_envs = 0.0
    cfg.commands.rel_forward_envs = 0.3
    cfg.commands.resampling_time = (4.0, 6.0)
    cfg.reset.base_offset = {
        "pose_range": {"x": (-0.25, 0.25), "y": (-0.2, 0.2), "z": (0.0, 0.10), "yaw": (-0.3, 0.3)},
        "velocity_range": {},
    }
    cfg.reset.joint_scale = (0.7, 1.3)
    cfg.reward.scales = ParkourRewardScales()
    return cfg
