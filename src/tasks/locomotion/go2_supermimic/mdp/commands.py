# Copyright (c) 2026-2027 zh
"""内容：实现一阶段HIM与二阶段PROOF的速度和航向采样，真实与想象共享命令状态。"""

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from .state_spec import (
    COMMAND_OVERRIDE, COMMAND_TIMER, CONTEXT_RANGES, CONTEXT_TERRAIN_FAMILY, FORWARD, HEADING_TARGET, STANDING,
)


def sample_commands(task, context, yaw):
    """一阶段沿用HIM采样；二阶段按参考项目使用互斥命令模式及地形航向范围。"""
    cfg = task.cfg.commands
    bounds = context[:, CONTEXT_RANGES].reshape(-1, 3, 2)
    random = torch.rand_like(bounds[..., 0])
    command = bounds[..., 0] + random * (bounds[..., 1] - bounds[..., 0])
    if task.parkour:
        straight = task.forward_families[context[:, CONTEXT_TERRAIN_FAMILY].long()]
        heading_min = torch.where(straight, -.4, cfg.heading_range[0])
        heading_max = torch.where(straight, .4, cfg.heading_range[1])
        heading = heading_min + torch.rand_like(yaw) * (heading_max - heading_min)
        mode = torch.rand_like(yaw)
        standing = mode < cfg.rel_standing_envs
        forward = (mode >= cfg.rel_standing_envs) & (mode < cfg.rel_standing_envs + cfg.rel_forward_envs)
        # LocoLab先将低于阈值的平移整体清零，再施加仅x模式。
        planar = command[:, :2] * (command[:, :2].norm(dim=-1, keepdim=True) > .2)
        command = torch.stack((planar[:, 0], torch.where(forward, 0., planar[:, 1]), command[:, 2]), -1)
    else:
        heading = cfg.heading_range[0] + torch.rand_like(yaw) * (cfg.heading_range[1] - cfg.heading_range[0])
        standing = torch.rand_like(yaw) <= cfg.rel_standing_envs
        forward = torch.rand_like(yaw) <= cfg.rel_forward_envs
        command = torch.stack((torch.where(forward, command[:, 0].abs().clamp_min(0.3), command[:, 0]),
                               torch.where(forward, torch.zeros_like(yaw), command[:, 1]), command[:, 2]), -1)
    lower, upper = cfg.resampling_time
    timer = lower + (upper - lower) * torch.rand_like(yaw)
    return command, timer, heading, standing, forward


def heading_command(task, context, command, yaw, heading, standing, forward):
    """二阶段仅x模式关闭航向跟踪，其余按该地形的角速度范围限幅。"""
    cfg = task.cfg.commands
    error = wrap_to_pi(heading - yaw)
    angular = cfg.heading_control_stiffness * error
    if task.parkour:
        yaw_bounds = context[:, CONTEXT_RANGES].reshape(-1, 3, 2)[:, 2]
        angular = torch.where(forward, 0., angular.clamp(yaw_bounds[:, 0], yaw_bounds[:, 1]))
    else:
        angular = angular.clamp(*cfg.initial_ranges[2])
    result = torch.cat((command[:, :2], angular[:, None]), -1)
    return torch.where(standing[:, None].bool(), torch.zeros_like(result), result)


def advance_commands(task, state, physical, dt):
    """在终点yaw刷新下一决策命令；固定覆盖命令不受heading重写。"""
    memory = state.memory
    sampled, timer, heading, standing, forward = sample_commands(task, state.context, physical[:, 72])
    expired = memory[:, COMMAND_TIMER] - dt <= 0.0
    command = torch.where(expired[:, None], sampled, state.command)
    timer = torch.where(expired, timer, memory[:, COMMAND_TIMER] - dt)
    heading = torch.where(expired, heading, memory[:, HEADING_TARGET])
    standing = torch.where(expired, standing, memory[:, STANDING].bool())
    forward = torch.where(expired, forward, memory[:, FORWARD].bool())
    command = heading_command(task, state.context, command, physical[:, 72], heading, standing, forward)
    command = torch.where(memory[:, COMMAND_OVERRIDE, None].bool(), state.command, command)
    return command, timer, heading, standing, forward
