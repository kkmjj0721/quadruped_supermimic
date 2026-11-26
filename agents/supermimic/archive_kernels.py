# Copyright (c) 2026-2027 zh
"""内容：对TE回放执行打包窗口读取、精确掩码入库及设备端来源选择。

结构参考本机quadruped_performance_427399a/candidate_safe的归档核，
按当前next-slot/结束当步无效协议重写；不使用旧终态池或ET布局。
"""

import triton
import triton.language as tl

from .interfaces import TaskState


@triton.jit
def _replay_values(states, actions, environment, start, selected,
                   STATE: tl.constexpr, ACTION: tl.constexpr, STEPS: tl.constexpr,
                   SLOTS: tl.constexpr, ENVS: tl.constexpr, BLOCK: tl.constexpr):
    columns = tl.arange(0, BLOCK)
    initial = tl.load(states + (start * ENVS + environment) * STATE + columns,
                      selected & (columns < STATE), other=0)
    action_slots = (start + columns // ACTION) % SLOTS
    action = tl.load(actions + (action_slots * ENVS + environment) * ACTION + columns % ACTION,
                     selected & (columns < STEPS * ACTION), other=0)
    target_slots = (start + columns // STATE + 1) % SLOTS
    targets = tl.load(states + (target_slots * ENVS + environment) * STATE + columns % STATE,
                      selected & (columns < STEPS * STATE), other=0)
    return initial, action, targets


@triton.jit
def _replay_flags(done, terminated, environment, start,
                  STEPS: tl.constexpr, SLOTS: tl.constexpr, ENVS: tl.constexpr, TIME_BLOCK: tl.constexpr):
    steps = tl.arange(0, TIME_BLOCK)
    slots = (start + steps) % SLOTS
    ended = tl.load(done + slots * ENVS + environment, steps < STEPS, other=0).to(tl.int32)
    terminal = tl.load(terminated + slots * ENVS + environment, steps < STEPS, other=0)
    # 与当前Replay一致：结束当步及其后均不参加监督。
    valid = tl.cumsum(ended) == 0
    return valid, terminal


@triton.jit
def _gather_windows(initial, actions, targets, valid, terminated,
                    states, replay_actions, replay_done, replay_terminated, rows, starts, archive,
                    STATE: tl.constexpr, ACTION: tl.constexpr, STEPS: tl.constexpr,
                    SLOTS: tl.constexpr, ENVS: tl.constexpr, ROW_STRIDE: tl.constexpr,
                    START_STRIDE: tl.constexpr, BLOCK: tl.constexpr, TIME_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    steps = tl.arange(0, TIME_BLOCK)
    if len(archive):
        archive_initial, archive_actions, archive_targets, archive_valid, archive_terminated, indices, populated = archive
        if tl.load(populated):
            source = tl.load(indices + row)
            initial_values = tl.load(archive_initial + source * STATE + columns, columns < STATE, other=0)
            action_values = tl.load(archive_actions + source * STEPS * ACTION + columns, columns < STEPS * ACTION, other=0)
            target_values = tl.load(archive_targets + source * STEPS * STATE + columns, columns < STEPS * STATE, other=0)
            valid_values = tl.load(archive_valid + source * STEPS + steps, steps < STEPS, other=0)
            terminal_values = tl.load(archive_terminated + source * STEPS + steps, steps < STEPS, other=0)
        else:
            environment = tl.load(rows + row * ROW_STRIDE)
            start = tl.load(starts + row * START_STRIDE)
            initial_values, action_values, target_values = _replay_values(
                states, replay_actions, environment, start, True, STATE, ACTION, STEPS, SLOTS, ENVS, BLOCK)
            valid_values, terminal_values = _replay_flags(
                replay_done, replay_terminated, environment, start, STEPS, SLOTS, ENVS, TIME_BLOCK)
    else:
        environment = tl.load(rows + row * ROW_STRIDE)
        start = tl.load(starts + row * START_STRIDE)
        initial_values, action_values, target_values = _replay_values(
            states, replay_actions, environment, start, True, STATE, ACTION, STEPS, SLOTS, ENVS, BLOCK)
        valid_values, terminal_values = _replay_flags(
            replay_done, replay_terminated, environment, start, STEPS, SLOTS, ENVS, TIME_BLOCK)
    tl.store(initial + row * STATE + columns, initial_values, columns < STATE)
    tl.store(actions + row * STEPS * ACTION + columns, action_values, columns < STEPS * ACTION)
    tl.store(targets + row * STEPS * STATE + columns, target_values, columns < STEPS * STATE)
    tl.store(valid + row * STEPS + steps, valid_values, steps < STEPS)
    tl.store(terminated + row * STEPS + steps, terminal_values, steps < STEPS)


def gather_windows(replay, rows, starts, horizon, archive=()):
    """读取当前TE协议窗口；归档是否已占用只在设备分支中决定。"""
    from .buffer import Window

    batch, state_width, action_width = rows.shape[0], replay.states.data.shape[-1], replay.actions.shape[-1]
    initial = replay.states.data.new_empty(batch, state_width)
    actions = replay.actions.new_empty(batch, horizon, action_width)
    targets = replay.states.data.new_empty(batch, horizon, state_width)
    valid = replay.done.new_empty(batch, horizon)
    terminated = replay.terminated.new_empty(batch, horizon)
    block = triton.next_power_of_2(max(state_width, horizon * action_width, horizon * state_width))
    _gather_windows[(batch,)](
        initial, actions, targets, valid, terminated, replay.states.data, replay.actions, replay.done,
        replay.terminated, rows, starts, archive, state_width, action_width, horizon,
        replay.capacity, replay.num_envs, rows.stride(0), starts.stride(0), block, triton.next_power_of_2(horizon))
    return Window(TaskState.from_flat(initial, replay.states.widths), actions,
                  TaskState.from_flat(targets, replay.states.widths), valid, terminated)


@triton.jit
def _write_from_replay(initial, actions, targets, valid, terminated,
                       states, replay_actions, replay_terminated, rows, starts, supplied_valid, accepted, indices,
                       STATE: tl.constexpr, ACTION: tl.constexpr, STEPS: tl.constexpr, SLOTS: tl.constexpr,
                       ENVS: tl.constexpr, ROW_STRIDE: tl.constexpr, START_STRIDE: tl.constexpr,
                       VALID_STRIDE: tl.constexpr, VALID_STEP_STRIDE: tl.constexpr,
                       BLOCK: tl.constexpr, TIME_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    selected = tl.load(accepted + row)
    destination = tl.load(indices + row)
    environment = tl.load(rows + row * ROW_STRIDE, selected, other=0)
    start = tl.load(starts + row * START_STRIDE, selected, other=0)
    columns = tl.arange(0, BLOCK)
    steps = tl.arange(0, TIME_BLOCK)
    initial_values, action_values, target_values = _replay_values(
        states, replay_actions, environment, start, selected, STATE, ACTION, STEPS, SLOTS, ENVS, BLOCK)
    step_mask = selected & (steps < STEPS)
    valid_values = tl.load(supplied_valid + row * VALID_STRIDE + steps * VALID_STEP_STRIDE, step_mask, other=0)
    slots = (start + steps) % SLOTS
    terminal_values = tl.load(replay_terminated + slots * ENVS + environment, step_mask, other=0)
    tl.store(initial + destination * STATE + columns, initial_values, selected & (columns < STATE))
    tl.store(actions + destination * STEPS * ACTION + columns, action_values, selected & (columns < STEPS * ACTION))
    tl.store(targets + destination * STEPS * STATE + columns, target_values, selected & (columns < STEPS * STATE))
    tl.store(valid + destination * STEPS + steps, valid_values, step_mask)
    tl.store(terminated + destination * STEPS + steps, terminal_values, step_mask)


def write_from_replay(destination, replay, rows, starts, horizon, valid, accepted, indices):
    """只为accepted行从Replay搬运全窗口；没有全候选中间Window。"""
    state_width, action_width = replay.states.data.shape[-1], replay.actions.shape[-1]
    block = triton.next_power_of_2(max(state_width, horizon * action_width, horizon * state_width))
    _write_from_replay[(rows.shape[0],)](
        destination.initial.data, destination.actions, destination.targets.data, destination.valid, destination.terminated,
        replay.states.data, replay.actions, replay.terminated, rows, starts, valid, accepted, indices,
        state_width, action_width, horizon, replay.capacity, replay.num_envs, rows.stride(0), starts.stride(0),
        valid.stride(0), valid.stride(1), block, triton.next_power_of_2(horizon))


@triton.jit
def _write_windows(initial, actions, targets, valid, terminated,
                   source_initial, source_actions, source_targets, source_valid, source_terminated, accepted, indices,
                   STATE: tl.constexpr, ACTION: tl.constexpr, ACTIONS: tl.constexpr,
                   TARGETS: tl.constexpr, STEPS: tl.constexpr,
                   INITIAL_STRIDE: tl.constexpr, INITIAL_COLUMN_STRIDE: tl.constexpr,
                   ACTION_STRIDE: tl.constexpr, ACTION_STEP_STRIDE: tl.constexpr, ACTION_COLUMN_STRIDE: tl.constexpr,
                   TARGET_STRIDE: tl.constexpr, TARGET_STEP_STRIDE: tl.constexpr, TARGET_COLUMN_STRIDE: tl.constexpr,
                   VALID_STRIDE: tl.constexpr, VALID_STEP_STRIDE: tl.constexpr,
                   TERMINAL_STRIDE: tl.constexpr, TERMINAL_STEP_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    selected = tl.load(accepted + row)
    destination = tl.load(indices + row)
    columns = tl.arange(0, BLOCK)
    initial_mask = selected & (columns < STATE)
    action_mask = selected & (columns < ACTIONS)
    target_mask = selected & (columns < TARGETS)
    step_mask = selected & (columns < STEPS)
    initial_values = tl.load(source_initial + row * INITIAL_STRIDE + columns * INITIAL_COLUMN_STRIDE,
                             initial_mask, other=0)
    action_values = tl.load(source_actions + row * ACTION_STRIDE + columns // ACTION * ACTION_STEP_STRIDE
                            + columns % ACTION * ACTION_COLUMN_STRIDE, action_mask, other=0)
    target_values = tl.load(source_targets + row * TARGET_STRIDE + columns // STATE * TARGET_STEP_STRIDE
                            + columns % STATE * TARGET_COLUMN_STRIDE, target_mask, other=0)
    valid_values = tl.load(source_valid + row * VALID_STRIDE + columns * VALID_STEP_STRIDE, step_mask, other=0)
    terminal_values = tl.load(source_terminated + row * TERMINAL_STRIDE + columns * TERMINAL_STEP_STRIDE,
                              step_mask, other=0)
    tl.store(initial + destination * STATE + columns, initial_values, initial_mask)
    tl.store(actions + destination * ACTIONS + columns, action_values, action_mask)
    tl.store(targets + destination * TARGETS + columns, target_values, target_mask)
    tl.store(valid + destination * STEPS + columns, valid_values, step_mask)
    tl.store(terminated + destination * STEPS + columns, terminal_values, step_mask)


def write_windows(destination, source, accepted, indices):
    """保留独立Window入库职责；按各源轴stride精确读取切片窗口。"""
    state_width = source.initial.data.shape[-1]
    actions = source.actions.shape[1] * source.actions.shape[2]
    targets = source.targets.data.shape[1] * source.targets.data.shape[2]
    steps = source.valid.shape[1]
    block = triton.next_power_of_2(max(state_width, actions, targets, steps))
    _write_windows[(source.actions.shape[0],)](
        destination.initial.data, destination.actions, destination.targets.data, destination.valid, destination.terminated,
        source.initial.data, source.actions, source.targets.data, source.valid, source.terminated, accepted, indices,
        state_width, source.actions.shape[-1], actions, targets, steps,
        *source.initial.data.stride(), *source.actions.stride(), *source.targets.data.stride(),
        *source.valid.stride(), *source.terminated.stride(), block)
