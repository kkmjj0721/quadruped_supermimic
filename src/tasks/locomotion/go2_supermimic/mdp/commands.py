# Copyright (c) 2026-2027 zh
"""内容：实现HIM速度采样与heading反馈，真实和想象使用同一命令状态。"""

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from .parkour import FORWARD_FAMILIES, PARKOUR_FAMILIES
from .state_spec import COMMAND_OVERRIDE, COMMAND_TIMER, CONTEXT_TERRAIN_FAMILY, FORWARD, HEADING_TARGET, STANDING


def sample_commands(cfg, context, yaw):
    """按阶段采样命令与航向，二阶段使用互斥仅 x 模式。"""
    bounds = context[:, 19:25].reshape(-1, 3, 2)
    random = torch.rand_like(bounds[..., 0])
    command = bounds[..., 0] + random * (bounds[..., 1] - bounds[..., 0])
    if cfg.parkour:
        heading_table = context.new_tensor([cfg.forward_heading_range if name in FORWARD_FAMILIES else cfg.heading_range
                                           for name in PARKOUR_FAMILIES])
        limits = heading_table[context[:, CONTEXT_TERRAIN_FAMILY].long()]
        heading = limits[:, 0] + torch.rand_like(yaw) * (limits[:, 1] - limits[:, 0])
        command = torch.cat((command[:, :2] * (command[:, :2].norm(dim=-1) > cfg.zero_velocity_threshold)[:, None],
                             command[:, 2:]), -1)
        mode = torch.rand_like(yaw)
        standing = mode < cfg.rel_standing_envs
        forward = (mode >= cfg.rel_standing_envs) & (mode < cfg.rel_standing_envs + cfg.rel_forward_envs)
        command = torch.where(forward[:, None], torch.cat((command[:, :1], torch.zeros_like(command[:, 1:])), -1), command)
    else:
        heading = cfg.heading_range[0] + torch.rand_like(yaw) * (cfg.heading_range[1] - cfg.heading_range[0])
        standing = torch.rand_like(yaw) <= cfg.rel_standing_envs
        forward = torch.rand_like(yaw) <= cfg.rel_forward_envs
        command = torch.stack((torch.where(forward, command[:, 0].abs().clamp_min(0.3), command[:, 0]),
                               torch.where(forward, torch.zeros_like(yaw), command[:, 1]), command[:, 2]), -1)
    lower, upper = cfg.resampling_time
    timer = lower + (upper - lower) * torch.rand_like(yaw)
    return command, timer, heading, standing, forward


def heading_command(cfg, command, yaw, heading, standing, forward, context):
    """普通阶段覆盖前进样本偏航；Parkour 仅 x 样本关闭航向反馈。"""
    error = wrap_to_pi(heading - yaw)
    angular = cfg.heading_control_stiffness * error
    if cfg.parkour:
        bounds = context[:, 19:25].reshape(-1, 3, 2)[:, 2]
        angular = angular.clamp(bounds[:, 0], bounds[:, 1])
        angular = torch.where(forward.bool(), torch.zeros_like(angular), angular)
    else:
        angular = angular.clamp(*cfg.initial_ranges[2])
    result = torch.cat((command[:, :2], angular[:, None]), -1)
    return torch.where(standing[:, None].bool(), torch.zeros_like(result), result)


def advance_commands(cfg, state, physical, dt):
    """在终点yaw刷新下一决策命令；固定覆盖命令不受heading重写。"""
    memory = state.memory
    sampled, timer, heading, standing, forward = sample_commands(cfg, state.context, physical[:, 72])
    expired = memory[:, COMMAND_TIMER] - dt <= 0.0
    command = torch.where(expired[:, None], sampled, state.command)
    timer = torch.where(expired, timer, memory[:, COMMAND_TIMER] - dt)
    heading = torch.where(expired, heading, memory[:, HEADING_TARGET])
    standing = torch.where(expired, standing, memory[:, STANDING].bool())
    forward = torch.where(expired, forward, memory[:, FORWARD].bool())
    command = heading_command(cfg, command, physical[:, 72], heading, standing, forward, state.context)
    command = torch.where(memory[:, COMMAND_OVERRIDE, None].bool(), state.command, command)
    return command, timer, heading, standing, forward
