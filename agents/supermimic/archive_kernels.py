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
def _gather_windows(initial, actions, targets, valid, terminated, replay_states, replay_actions,
                    terminal_storage, terminal_indices, replay_terminated, rows, starts, archive,
                    STATE: tl.constexpr, ACTION: tl.constexpr, STEPS: tl.constexpr, SLOTS: tl.constexpr,
                    ROW_STRIDE: tl.constexpr, START_STRIDE: tl.constexpr, BLOCK: tl.constexpr, TIME_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    initial_mask = columns < STATE
    action_mask = columns < STEPS * ACTION
    target_mask = columns < STEPS * STATE
    if len(archive):
        archive_initial, archive_actions, archive_targets, archive_valid, archive_terminated, indices, populated = archive
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
            tl.store(initial + row * STATE + columns, initial_values, initial_mask)
            tl.store(actions + row * STEPS * ACTION + columns, action_values, action_mask)
            tl.store(targets + row * STEPS * STATE + columns, target_values, target_mask)
            return
    environment = tl.load(rows + row * ROW_STRIDE)
    start = tl.load(starts + row * START_STRIDE)
    initial_values = tl.load(replay_states + (environment * SLOTS + start) * STATE + columns,
                             initial_mask, other=0)
    action_slots = (start + columns // ACTION) % SLOTS
    action_values = tl.load(replay_actions + (environment * SLOTS + action_slots) * ACTION + columns % ACTION,
                            action_mask, other=0)
    target_slots = (start + columns // STATE) % SLOTS
    terminal = tl.load(terminal_indices + environment * SLOTS + target_slots, target_mask, other=-1)
    ordinary = tl.load(replay_states + (environment * SLOTS + (target_slots + 1) % SLOTS) * STATE + columns % STATE,
                       target_mask & (terminal < 0), other=0)
    terminal_states = tl.load(terminal_storage).to(replay_states.dtype)
    capacity = tl.load(terminal_storage + 1)
    final = tl.load(terminal_states + (terminal & (capacity - 1)) * STATE + columns % STATE,
                    target_mask & (terminal >= 0), other=0)
    target_values = tl.where(terminal >= 0, final, ordinary)
    steps = tl.arange(0, TIME_BLOCK)
    replay_step_mask = steps < STEPS
    step_slots = (start + steps) % SLOTS
    ended = (tl.load(terminal_indices + environment * SLOTS + step_slots, replay_step_mask, other=-1) >= 0).to(tl.int32)
    replay_termination_values = tl.load(replay_terminated + environment * SLOTS + step_slots, replay_step_mask, other=0)
    # 只检查此前各步是否已结束，当前步的终点仍可监督。
    replay_valid_values = tl.cumsum(ended) == ended
    tl.store(valid + row * STEPS + steps, replay_valid_values, replay_step_mask)
    tl.store(terminated + row * STEPS + steps, replay_termination_values, replay_step_mask)
    tl.store(initial + row * STATE + columns, initial_values, initial_mask)
    tl.store(actions + row * STEPS * ACTION + columns, action_values, action_mask)
    tl.store(targets + row * STEPS * STATE + columns, target_values, target_mask)


def gather_windows(replay, rows, starts, horizon, archive=()):
    """统一读取连续回放或设备选择的归档窗口；普通采样将归档分支静态消去。"""
    from .buffer import Window

    states = replay.states.data
    batch, state_width, action_width = rows.shape[0], states.shape[-1], replay.actions.shape[-1]
    initial = states.new_empty(batch, state_width)
    actions = replay.actions.new_empty(batch, horizon, action_width)
    targets = states.new_empty(batch, horizon, state_width)
    valid = replay.terminated.new_empty(batch, horizon)
    terminated = replay.terminated.new_empty(batch, horizon)
    block = triton.next_power_of_2(max(state_width, horizon * action_width, horizon * state_width))
    _gather_windows[(batch,)](
        initial, actions, targets, valid, terminated, states, replay.actions,
        replay.terminal_storage, replay.terminal_indices, replay.terminated, rows, starts, archive,
        state_width, action_width, horizon, replay.slot_count, rows.stride(0), starts.stride(0),
        block, triton.next_power_of_2(horizon))
    return Window(TaskState.from_flat(initial, replay.states.widths), actions,
                  TaskState.from_flat(targets, replay.states.widths), valid, terminated)
