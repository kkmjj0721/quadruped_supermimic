# Copyright (c) 2026-2027 zh
"""内容：实现普通地形教师奖励与PROOF二阶段按地形分组的速度跟踪和足部边缘代价。"""

import math

import torch

from .reference_state import (
    ANGULAR_BODY, COMPUTED_TORQUE, FOOT_CONTACT, FOOT_HISTORY_CONTACT, GRAVITY,
    FOOT_VELOCITY_XY, JOINT_ACCELERATION, JOINT_OFFSET, JOINT_TORQUE, JOINT_VELOCITY, STUMBLE_EVENT,
    UNDESIRED_CONTACT_EVENT, normalized_gravity,
)
from .state_spec import (
    CONTEXT_TERRAIN_FAMILY, ELAPSED_TIME, FIRST_CONTACT, LAST_AIR_TIME, LAST_CONTACT_TIME,
    PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION,
)


def posture_reward(task, state, next_state):
    """复用main的分段关节容差及body平移／world偏航跟踪门控。"""
    cfg = task.cfg.reward
    magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = magnitude < cfg.posture.walking_threshold
    walking = (magnitude >= cfg.posture.walking_threshold) & (magnitude < cfg.posture.running_threshold)
    std = (task.posture_std[0] * standing[:, None] + task.posture_std[1] * walking[:, None]
           + task.posture_std[2] * (magnitude >= cfg.posture.running_threshold)[:, None])
    pose = torch.exp(-(next_state.physical[:, JOINT_OFFSET].square() / std.square()).mean(-1))
    if cfg.pose_tracking_gate:
        linear_velocity = task.body_linear_velocity(next_state)[:, :2]
        yaw_rate = -(normalized_gravity(next_state.physical) * next_state.physical[:, ANGULAR_BODY]).sum(-1)
        linear_target = state.command[:, :2].square().sum(-1)
        angular_target = state.command[:, 2].square()
        linear_error = (linear_velocity - state.command[:, :2]).square().sum(-1)
        angular_error = (yaw_rate - state.command[:, 2]).square()
        linear_active = state.command[:, :2].norm(dim=-1) > cfg.stand_still_command_threshold
        angular_active = state.command[:, 2].abs() > cfg.stand_still_command_threshold
        # main的未选中分支仍求值；零目标分母用1，保留原门控定义。
        linear_denominator = torch.where(linear_active, linear_target, torch.ones_like(linear_target))
        angular_denominator = torch.where(angular_active, angular_target, torch.ones_like(angular_target))
        linear_gate = torch.where(linear_active, (1 - linear_error / linear_denominator).clamp(0, 1),
                                  torch.exp(-linear_error / cfg.tracking_linear_variance))
        angular_gate = torch.where(angular_active, (1 - angular_error / angular_denominator).clamp(0, 1),
                                   torch.exp(-angular_error / cfg.tracking_angular_variance))
        gate = torch.where(magnitude <= cfg.stand_still_command_threshold, torch.ones_like(linear_gate),
                           linear_gate * angular_gate)
        pose = pose * gate
    return pose


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
    foot_velocity = physical[:, FOOT_VELOCITY_XY].reshape(-1, 4, 2)
    foot_speed = foot_velocity.norm(dim=-1)
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
    variance = (last_air_time.clamp(max=cfg.contact_time_variance_clip).var(dim=-1)
                + last_contact_time.clamp(max=cfg.contact_time_variance_clip).var(dim=-1))
    planar_velocity = body_velocity[:, :2]
    if task.parkour:
        families = state.context[:, CONTEXT_TERRAIN_FAMILY].long()
        special = task.special_families[families]
        vx = planar_velocity[:, 0]
        capped_vx = torch.minimum(vx, state.command[:, 0] + .1)
        planar_velocity = torch.stack((torch.where(special, capped_vx, vx), planar_velocity[:, 1]), -1)
        # 来源在专项地形关闭跨足腾空／支撑时长的方差惩罚。
        variance = torch.where(special, 0., variance)
        stand_still = joint_offset.norm(dim=-1) * ~moving
        points, velocity = task.kinematics.foot_volume_points(task.generalized_state(next_state), task.volume_offsets)
        depth = task.surface.edge_depth(points)
        # LocoLab的原式使用速度模长加1e-6，虚拟障碍仅对分配到对应地形族的环境启用。
        penetration = (depth * (velocity.norm(dim=-1) + 1e-6)).sum(-1)
        qw, qx, qy, qz = physical[:, task.root_quat].unbind(-1)
        forward_x = 1 - 2 * (qy.square() + qz.square())
        forward_y = 2 * (qx * qy + qw * qz)
        extras = {"cheat": (special & (forward_y.abs() > forward_x * math.tan(.8))).float(),
                  "volume_points_penetration": torch.where(task.edge_families[families], penetration, 0.)}
    else:
        command_magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
        standing = command_magnitude <= cfg.stand_still_command_threshold
        stand_still = joint_offset.square().sum(-1) * ~moving
        extras = {
            "pose": posture_reward(task, state, next_state),
            "stand_still_joint_vel_l2": physical[:, JOINT_VELOCITY].square().sum(-1) * standing,
            "foot_slip": (foot_velocity.square().sum(-1) * physical[:, FOOT_CONTACT]).sum(-1)
                         * (command_magnitude > cfg.foot_slip_command_threshold),
        }
    return {
        "track_lin_vel_xy_exp": torch.exp(-(planar_velocity - state.command[:, :2]).square().sum(-1)
                                         / cfg.tracking_linear_variance),
        "track_ang_vel_z_exp": torch.exp(-(angular_velocity[:, 2] - state.command[:, 2]).square()
                                        / cfg.tracking_angular_variance),
        "lin_vel_z_l2": body_velocity[:, 2].square(),
        "ang_vel_xy_l2": angular_velocity[:, :2].square().sum(-1),
        "body_orientation_l2": physical[:, GRAVITY][:, :2].square().sum(-1),
        "stuck": ((state.command[:, :2].norm(dim=-1) > cfg.stuck_velocity_threshold)
                  & (body_velocity[:, :2].norm(dim=-1) < cfg.stuck_velocity_threshold)).float(),
        "stand_still": stand_still,
        **extras,
        "joint_hip_deviation_l2": joint_offset[:, ::3].square().sum(-1),
        "joint_deviation_l2": joint_offset.square().sum(-1),
        "joint_acc_l2": physical[:, JOINT_ACCELERATION].square().sum(-1),
        "joint_torques_l2_soft_limit": torque_penalty.sum(-1),
        "joint_power_l1": (physical[:, JOINT_TORQUE] * physical[:, JOINT_VELOCITY]).abs().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "action_rate_l2": (action - state.memory[:, PREVIOUS_ACTION]).square().sum(-1),
        "action_smoothness_l2": (action - 2 * state.memory[:, PREVIOUS_ACTION]
                                 + state.memory[:, PREVIOUS_PREVIOUS_ACTION]).square().sum(-1) * smoothness_active,
        "undesired_contacts": physical[:, UNDESIRED_CONTACT_EVENT].sum(-1),
        "feet_air_time": ((last_air_time - cfg.feet_air_time_threshold)
                          * next_state.memory[:, FIRST_CONTACT]).sum(-1) * moving,
        "feet_air_time_variance": variance,
        "feet_stumble": physical[:, STUMBLE_EVENT].squeeze(-1),
        "feet_slide": (foot_speed * physical[:, FOOT_HISTORY_CONTACT]).sum(-1),
        "is_alive": 1 - failure,
        "termination_penalty": failure,
    }
