# Copyright (c) 2026-2027 zh
"""内容：从任务快照派生PROOF三输入教师的部署坐标、观测与控制语义。"""

from ..mdp.state_spec import JOINT_NAMES
from ..mdp.parkour import SPECIAL_FAMILIES


def deployment_metadata(cfg, *, algorithm) -> dict:
    """从任务配置生成离线部署所需的观测、地图及动作定义。"""
    control = cfg.control
    metadata = {
        "algorithm": "supermimic_proof_teacher",
        "inference_action": "deterministic_mean",
        "observation_dim": 45, "history_length": cfg.env.num_obs_history,
        "observation_layout": [
            {"name": "base_angular_velocity_body", "width": 3},
            {"name": "projected_gravity_body", "width": 3},
            {"name": "velocity_command_body_xy_body_yaw", "width": 3},
            {"name": "joint_position_offset_with_encoder_bias", "width": 12},
            {"name": "joint_velocity", "width": 12},
            {"name": "previous_unscaled_action", "width": 12},
        ],
        "joint_names": list(JOINT_NAMES), "action_joint_names": list(JOINT_NAMES),
        "default_joint_positions": [float(cfg.init_state.default_joint[name]) for name in JOINT_NAMES],
        "action_scales": [control.action_scales[name] for _ in range(4) for name in ("hip", "thigh", "calf")],
        "joint_target_formula": "default_joint_positions + actions * action_scales",
        "joint_target_coordinate": "encoder",
        "simulation_physical_target_formula": "default_joint_positions + actions * action_scales - encoder_bias",
        "control_dt_seconds": float(cfg.sim.timestep * cfg.control.decimation),
        "command_frame": "body_xy_body_yaw",
        "observation_semantics": "proprioceptive_history_current_heightmap_episode_mask",
        "heightmap": {"shape": [9, 18], "flatten_order": "y_outer_x_inner", "alignment": "base_yaw",
                      "size_m": list(cfg.terrain.scan_size_m), "resolution_m": cfg.terrain.scan_resolution_m,
                      "forward_offset_m": cfg.terrain.scan_forward_offset_m,
                      "height_offset_m": cfg.terrain.scan_height_offset_m,
                      "formula": "clip(base_z - terrain_z - height_offset_m, -10, 10)", "noise": "none"},
        "exteroceptive_mask": {"shape": [None, 1], "values": [0, 1], "sample": "episode_reset",
                               "drop_probability": cfg.teacher.exteroceptive_dropout,
                               "flat_family_mask": 1, "stage": "multiply_height_latent_after_tanh"},
        "actor_input_dim": algorithm.prop_latent_dim + 45 + algorithm.height_latent_dim,
        "actor_input_order": "proprioceptive_latent_normalized_current_frame_masked_height_latent",
        "velocity_estimation": {"dimension": 3, "latent_columns": [0, 1, 2], "frame": "body",
                                "time": "current", "units": "m/s", "components": ["vx", "vy", "vz"],
                                "supervision": "same_time_real_state_mse"},
        "reference_commit": "ae910b821d635f04226489df80bdd77d1d7b0b82",
    }
    if cfg.terrain.mode == "parkour":
        del metadata["exteroceptive_mask"]["flat_family_mask"]
        metadata["exteroceptive_mask"]["always_visible_families"] = list(SPECIAL_FAMILIES)
    return metadata
