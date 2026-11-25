# Copyright (c) 2026-2027 zh
"""内容：精确掩码写入连续归档，并在设备上选择归档或回放的窗口读取。"""

import triton
import triton.language as tl

from .interfaces import TaskState


@triton.jit
def _write_windows(initial, actions, targets, valid, terminated,
                   source_initial, source_actions, source_targets, source_valid, source_terminated,
                   accepted, indices,
                   INITIAL: tl.constexpr, ACTIONS: tl.constexpr, TARGETS: tl.constexpr,
                   STEPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    selected = tl.load(accepted + row)
    destination = tl.load(indices + row)
    columns = tl.arange(0, BLOCK)
    initial_mask = selected & (columns < INITIAL)
    action_mask = selected & (columns < ACTIONS)
    target_mask = selected & (columns < TARGETS)
    step_mask = selected & (columns < STEPS)
    initial_values = tl.load(source_initial + row * INITIAL + columns, initial_mask, other=0)
    action_values = tl.load(source_actions + row * ACTIONS + columns, action_mask, other=0)
    target_values = tl.load(source_targets + row * TARGETS + columns, target_mask, other=0)
    valid_values = tl.load(source_valid + row * STEPS + columns, step_mask, other=0)
    termination_values = tl.load(source_terminated + row * STEPS + columns, step_mask, other=0)
    tl.store(initial + destination * INITIAL + columns, initial_values, initial_mask)
    tl.store(actions + destination * ACTIONS + columns, action_values, action_mask)
    tl.store(targets + destination * TARGETS + columns, target_values, target_mask)
    tl.store(valid + destination * STEPS + columns, valid_values, step_mask)
    tl.store(terminated + destination * STEPS + columns, termination_values, step_mask)


def write_windows(destination, source, accepted, indices):
    """窗口均为批次优先连续存储；掩码与写入地址在设备上生成。"""
    initial = source.initial.data.shape[-1]
    actions = source.actions.shape[1] * source.actions.shape[2]
    targets = source.targets.data.shape[1] * source.targets.data.shape[2]
    steps = source.valid.shape[1]
    block = triton.next_power_of_2(max(initial, actions, targets, steps))
    _write_windows[(source.actions.shape[0],)](
        destination.initial.data, destination.actions, destination.targets.data, destination.valid, destination.terminated,
        source.initial.data, source.actions, source.targets.data, source.valid, source.terminated,
        accepted, indices, initial, actions, targets, steps, block)


@triton.jit
def _sample_archive_or_replay(initial, actions, targets, valid, terminated,
                             archive_initial, archive_actions, archive_targets, archive_valid, archive_terminated,
                             replay_initial, replay_actions, replay_targets, replay_done, replay_terminated,
                             rows, starts, indices, populated,
                             STATE: tl.constexpr, ACTION: tl.constexpr, STEPS: tl.constexpr,
                             CAPACITY: tl.constexpr, BLOCK: tl.constexpr, TIME_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    initial_mask = columns < STATE
    action_mask = columns < STEPS * ACTION
    target_mask = columns < STEPS * STATE
    if tl.load(populated):
        source = tl.load(indices + row)
        initial_values = tl.load(archive_initial + source * STATE + columns, initial_mask, other=0)
        action_values = tl.load(archive_actions + source * STEPS * ACTION + columns, action_mask, other=0)
        target_values = tl.load(archive_targets + source * STEPS * STATE + columns, target_mask, other=0)
        archive_step_mask = columns < STEPS
        archive_valid_values = tl.load(archive_valid + source * STEPS + columns, archive_step_mask, other=0)
        archive_termination_values = tl.load(archive_terminated + source * STEPS + columns, archive_step_mask, other=0)
        tl.store(valid + row * STEPS + columns, archive_valid_values, archive_step_mask)
        tl.store(terminated + row * STEPS + columns, archive_termination_values, archive_step_mask)
    else:
        environment = tl.load(rows + row)
        start = tl.load(starts + row)
        initial_values = tl.load(replay_initial + (environment * CAPACITY + start) * STATE + columns,
                                 initial_mask, other=0)
        action_slots = (start + columns // ACTION) % CAPACITY
        action_values = tl.load(replay_actions + (environment * CAPACITY + action_slots) * ACTION + columns % ACTION,
                                action_mask, other=0)
        target_slots = (start + columns // STATE) % CAPACITY
        target_values = tl.load(replay_targets + (environment * CAPACITY + target_slots) * STATE + columns % STATE,
                                target_mask, other=0)
        steps = tl.arange(0, TIME_BLOCK)
        replay_step_mask = steps < STEPS
        step_slots = (start + steps) % CAPACITY
        ended = tl.load(replay_done + environment * CAPACITY + step_slots, replay_step_mask, other=0)
        replay_termination_values = tl.load(replay_terminated + environment * CAPACITY + step_slots, replay_step_mask, other=0)
        # 首次结束当步仍有效，只屏蔽其后跨入新回合的转移。
        replay_valid_values = tl.cumsum(ended.to(tl.int32)) - ended.to(tl.int32) == 0
        tl.store(valid + row * STEPS + steps, replay_valid_values, replay_step_mask)
        tl.store(terminated + row * STEPS + steps, replay_termination_values, replay_step_mask)
    tl.store(initial + row * STATE + columns, initial_values, initial_mask)
    tl.store(actions + row * STEPS * ACTION + columns, action_values, action_mask)
    tl.store(targets + row * STEPS * STATE + columns, target_values, target_mask)


def sample_archive_or_replay(archive, replay, rows, starts, indices, horizon, populated):
    """条件只在设备核内求值，两条分支均只读取实际选中的完整窗口。"""
    from .buffer import Window

    batch, state_width, action_width = rows.shape[0], archive.initial.data.shape[-1], archive.actions.shape[-1]
    initial = archive.initial.data.new_empty(batch, state_width)
    actions = archive.actions.new_empty(batch, horizon, action_width)
    targets = archive.targets.data.new_empty(batch, horizon, state_width)
    valid = archive.valid.new_empty(batch, horizon)
    terminated = archive.terminated.new_empty(batch, horizon)
    block = triton.next_power_of_2(max(state_width, horizon * action_width, horizon * state_width))
    _sample_archive_or_replay[(batch,)](
        initial, actions, targets, valid, terminated,
        archive.initial.data, archive.actions, archive.targets.data, archive.valid, archive.terminated,
        replay.states.data, replay.actions, replay.final_states.data, replay.done, replay.terminated,
        rows, starts, indices, populated, state_width, action_width, horizon, replay.capacity,
        block, triton.next_power_of_2(horizon))
    return Window(TaskState.from_flat(initial, archive.initial.widths), actions,
                  TaskState.from_flat(targets, archive.targets.widths), valid, terminated)
