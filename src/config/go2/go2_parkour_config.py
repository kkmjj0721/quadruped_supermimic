# Copyright (c) 2026-2027 zh
"""内容：组合去除离散障碍的 PROOF 二阶段 Parkour 地形、命令、奖励与域随机化配置。"""

from copy import deepcopy
from math import pi

from .go2_reference_supermimic_config import Go2ReferenceCfg


class ParkourRewardScales:
    """严格采用参考项目的二阶段二十三项奖励权重。"""

    track_lin_vel_xy_exp = 1.5
    track_ang_vel_z_exp = 1.0
    lin_vel_z_l2 = -1.0
    ang_vel_xy_l2 = -0.05
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
    feet_slide = -2.5e-2
    is_alive = 0.1
    termination_penalty = -100.0
    stuck = -1.0
    cheat = -1.0
    volume_points_penetration = -2.0


def parkour_config(source: Go2ReferenceCfg) -> Go2ReferenceCfg:
    """保留来源算法和网络预算，完整替换二阶段任务配方。"""
    cfg = deepcopy(source)
    cfg.terrain.mode = "parkour"
    cfg.terrain.seed = 17
    cfg.terrain.asset_path = "src/assets/terrains/proof_parkour_17.npz"
    cfg.terrain.core_rows, cfg.terrain.num_cols = 10, 20
    cfg.terrain.patch_size_m = (8.0, 8.0)
    cfg.terrain.outer_flat_width_m = 20.0
    cfg.terrain.max_init_level = 0
    cfg.terrain.curriculum = True
    cfg.teacher.exteroceptive_dropout = 0.4
    cfg.commands.parkour = True
    cfg.commands.initial_ranges = ((-0.5, 1.1), (-0.5, 0.5), (-1.5, 1.5))
    cfg.commands.forward_ranges = ((0.7, 1.1), (0.0, 0.0), (-0.3, 0.3))
    cfg.commands.heading_range = (-pi, pi)
    cfg.commands.forward_heading_range = (-0.4, 0.4)
    cfg.commands.zero_velocity_threshold = 0.2
    cfg.commands.resampling_time = (4.0, 6.0)
    cfg.commands.heading_command = True
    cfg.commands.heading_control_stiffness = 0.8
    cfg.commands.rel_heading_envs = 1.0
    cfg.commands.rel_standing_envs = 0.0
    cfg.commands.rel_forward_envs = 0.3
    cfg.commands.zero_probability_start = cfg.commands.zero_probability_end = 0.0
    cfg.commands.stages = []
    cfg.reset.base_offset = {
        "pose_range": {"x": (-0.25, 0.25), "y": (-0.2, 0.2), "z": (0.0, 0.10), "yaw": (-0.3, 0.3)},
        "velocity_range": {},
    }
    cfg.reset.joint_scale_range = (0.7, 1.3)
    cfg.domain_randomization.enabled = True
    cfg.domain_randomization.parkour_enabled = True
    cfg.domain_randomization.friction_range = (0.2, 1.5)
    cfg.domain_randomization.com_range = {axis: (-0.05, 0.05) for axis in ("x", "y", "z")}
    cfg.domain_randomization.base_mass_range = (-1.5, 1.5)
    cfg.domain_randomization.limb_mass_scale_range = (0.8, 1.2)
    cfg.domain_randomization.pd_gain_range = (0.9, 1.1)
    cfg.domain_randomization.joint_friction_range = (0.0, 0.2)
    cfg.domain_randomization.joint_armature_range = (0.0, 0.01)
    cfg.domain_randomization.joint_parameter_mode = "startup"
    cfg.domain_randomization.push_enabled = True
    cfg.domain_randomization.push_interval_s = 10.0
    cfg.domain_randomization.push_velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    cfg.reward.scales = ParkourRewardScales()
    return cfg
