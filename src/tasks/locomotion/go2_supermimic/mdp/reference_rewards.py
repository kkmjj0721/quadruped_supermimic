# Copyright (c) 2026-2027 zh
"""内容：复用PROOF普通地形奖励与main的站立关节速度、足端滑移项，返回奖励率。"""

import torch

from .reference_state import (
    ANGULAR_BODY, CALF_CONTACT_FORCE, COMPUTED_TORQUE, CONTACT_FOUND, FOOT_CONTACT_FORCE, FOOT_HISTORY_CONTACT,
    FOOT_VELOCITY_XY, JOINT_ACCELERATION, JOINT_OFFSET, JOINT_TORQUE, JOINT_VELOCITY,
)
from .state_spec import (
    ELAPSED_TIME, FIRST_CONTACT, LAST_AIR_TIME, LAST_CONTACT_TIME, PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION,
)


def reference_reward_terms(task, state, action, next_state, failure):
    """以当前决策命令评价步进后的状态；真实采集与想象沿用同一评分。"""
    cfg = task.cfg.reward
    physical = next_state.physical
    body_velocity = task.body_linear_velocity(next_state)
    angular_velocity = physical[:, ANGULAR_BODY]
    joint_offset = physical[:, JOINT_OFFSET]
    joint = joint_offset + task.default_joint_pos
    lower, upper = task.joint_limits.unbind(-1)
    moving = ((state.command[:, :2].square().sum(-1) > cfg.command_threshold ** 2)
              | (state.command[:, 2].abs() > cfg.command_threshold))
    command_magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = command_magnitude <= cfg.stand_still_command_threshold
    foot_force = physical[:, FOOT_CONTACT_FORCE].reshape(-1, 4, 3)
    foot_velocity = physical[:, FOOT_VELOCITY_XY].reshape(-1, 4, 2)
    foot_speed = foot_velocity.norm(dim=-1)
    foot_speed_squared = foot_velocity.square().sum(-1)
    # 来源先截断限幅前PD力矩，再对24Nm软阈值和28.5Nm硬阈值追加惩罚。
    torque = physical[:, COMPUTED_TORQUE].abs().clamp(max=cfg.torque_clip)
    torque_squared = torque.square()
    soft_excess = (torque - cfg.torque_soft_factor * cfg.torque_limit_nominal).clamp_min(0)
    at_hard_limit = torque >= cfg.torque_hard_factor * cfg.torque_limit_nominal
    torque_penalty = (torque_squared + cfg.torque_over_soft_scale * soft_excess.square()
                      + cfg.torque_over_hard_scale * torque_squared * at_hard_limit)
    last_air_time = next_state.memory[:, LAST_AIR_TIME]
    last_contact_time = next_state.memory[:, LAST_CONTACT_TIME]
    # LocoLab在新回合前两个控制步屏蔽二阶动作差分。
    smoothness_active = torch.round(next_state.memory[:, ELAPSED_TIME] / task.dt) >= 3
    failure = failure.float()
    return {
        "track_lin_vel_xy_exp": torch.exp(-(body_velocity[:, :2] - state.command[:, :2]).square().sum(-1)
                                         / cfg.tracking_linear_variance),
        "track_ang_vel_z_exp": torch.exp(-(angular_velocity[:, 2] - state.command[:, 2]).square()
                                        / cfg.tracking_angular_variance),
        "lin_vel_z_l2": body_velocity[:, 2].square(),
        "ang_vel_xy_l2": angular_velocity[:, :2].square().sum(-1),
        "stand_still": joint_offset.norm(dim=-1) * ~moving,
        "stand_still_joint_vel_l2": physical[:, JOINT_VELOCITY].square().sum(-1) * standing,
        "foot_slip": (foot_speed_squared * physical[:, CONTACT_FOUND]).sum(-1)
                     * (command_magnitude > cfg.foot_slip_command_threshold),
        "joint_hip_deviation_l2": joint_offset[:, ::3].square().sum(-1),
        "joint_deviation_l2": joint_offset.square().sum(-1),
        "joint_acc_l2": physical[:, JOINT_ACCELERATION].square().sum(-1),
        "joint_torques_l2_soft_limit": torque_penalty.sum(-1),
        "joint_power_l1": (physical[:, JOINT_TORQUE] * physical[:, JOINT_VELOCITY]).abs().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "action_rate_l2": (action - state.memory[:, PREVIOUS_ACTION]).square().sum(-1),
        "action_smoothness_l2": (action - 2 * state.memory[:, PREVIOUS_ACTION]
                                 + state.memory[:, PREVIOUS_PREVIOUS_ACTION]).square().sum(-1) * smoothness_active,
        "undesired_contacts": (physical[:, CALF_CONTACT_FORCE] > cfg.contact_force_threshold).sum(-1),
        "feet_air_time": ((last_air_time - cfg.feet_air_time_threshold)
                          * next_state.memory[:, FIRST_CONTACT]).sum(-1) * moving,
        # ae910b8的where在parkour分支返回零，普通地形实际保留此项；方差沿用默认无偏估计。
        "feet_air_time_variance": (last_air_time.clamp(max=cfg.contact_time_variance_clip).var(dim=-1)
                                   + last_contact_time.clamp(max=cfg.contact_time_variance_clip).var(dim=-1)),
        "feet_stumble": (foot_force[..., :2].norm(dim=-1)
                         > cfg.stumble_ratio * foot_force[..., 2].abs()).any(-1).float(),
        "feet_slide": (foot_speed * physical[:, FOOT_HISTORY_CONTACT]).sum(-1),
        "is_alive": 1 - failure,
        "termination_penalty": failure,
    }
