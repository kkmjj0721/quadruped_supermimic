# Copyright (c) 2026-2027 zh
"""
内容：
    单文件离线评估 Go2 策略的跟踪、急停、对角切换、通行和运动质量。
    借鉴 RoboGauge 的测试分类，输出本项目原始指标，不计算其归一化总分。
    仅统计首回合（包含失败步），动态测试分别报告各命令阶段。
    多检查点共用第一个检查点的任务配置；地形等级从零开始。

用法：
    python scripts/evaluate.py Unitree-Go2-SuperMimic-Flat \
        --checkpoints logs/experiment/model_1000.pt --output logs/evaluation.json
    python scripts/evaluate.py Unitree-Go2-SuperMimic-Parkour \
        --checkpoints logs/experiment/model_1000.pt logs/experiment/model_2000.pt \
        --suite traverse --levels 0 3 5 --seeds 0 1 2 --domain-randomization training \
        --output logs/parkour_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mjlab.utils.torch import configure_torch_backends

from agents.supermimic.interfaces import append_history
from src.config.go2.go2_parkour_config import parkour_config
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg
from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES
from src.tasks.locomotion.go2_supermimic.mdp.reference_state import (
    CONTACT_FOUND, FEET, GRAVITY, JOINT_OFFSET, JOINT_TORQUE, JOINT_VELOCITY,
)
from src.tasks.locomotion.go2_supermimic.mdp.state_spec import CONTEXT_ORIGIN
from src.tasks.locomotion.go2_supermimic.mdp.terrains import TerrainMap
from src.tasks.locomotion.go2_supermimic.rl.checkpoints import build_playback_policy, load_playback_source
from src.tasks.locomotion.go2_supermimic.rl.transition_adapter import Go2TransitionAdapter


# ---------- 测试协议 ----------

def scenarios(args):
    """阶段等长；固定命令覆盖站立、双向平移和双向转向。"""
    x, y, w = args.forward_speed, args.lateral_speed, args.yaw_rate
    suites = {
        "tracking": [(name, [command]) for name, command in (
            ("stand", (0., 0., 0.)), ("forward", (x, 0., 0.)), ("backward", (-x, 0., 0.)),
            ("left", (0., y, 0.)), ("right", (0., -y, 0.)),
            ("yaw_left", (0., 0., w)), ("yaw_right", (0., 0., -w)))],
        "stop": [("stop", [(x, 0., 0.), (0., 0., 0.)])],
        "switch": [("switch", [(x, y, w), (-x, -y, -w)])],
        "traverse": [("traverse", [(x, 0., 0.)])],
    }
    names = tuple(suites) if args.suite == "all" else (args.suite,)
    return [case for name in names for case in suites[name]]


def measurements(task, final, actions, previous_torque, previous_action, command):
    """实际力矩而非未限幅 PD 力矩；功率为各关节绝对机械功率之和。"""
    physical = final.physical
    velocity = task.tracking_velocity(final)
    torque = physical[:, JOINT_TORQUE]
    joint_pos = physical[:, JOINT_OFFSET] + task.default_joint_pos
    lower, upper = task.joint_limits.unbind(-1)
    excess = (lower - joint_pos).relu() + (joint_pos - upper).relu()
    gravity = physical[:, GRAVITY]
    return torch.stack((
        (velocity[:, :2] - command[:2]).square().sum(-1),
        (velocity[:, 2] - command[2]).square(),
        (torque * physical[:, JOINT_VELOCITY]).abs().sum(-1),
        excess.square().mean(-1), gravity[:, 1].abs(), gravity[:, 0].abs(),
        (torque - previous_torque).square().mean(-1),
        (actions - previous_action).square().mean(-1), physical[:, 0],
        velocity[:, 0], velocity[:, 1], velocity[:, 2],
    ), -1).double()


METRICS = (
    ("linear_velocity_rmse_mps", True), ("yaw_rate_rmse_radps", True),
    ("mean_absolute_mechanical_power_w", False), ("joint_limit_excess_rms_rad", True),
    ("mean_abs_gravity_y", False), ("mean_abs_gravity_x", False),
    ("torque_step_rms_nm", True), ("action_step_rms", True), ("mean_base_height_m", False),
    ("mean_vx_mps", False), ("mean_vy_mps", False), ("mean_yaw_rate_radps", False),
)


def metric_means(sums, count):
    """跨环境和时间先累计，再对平方量开根；无观测阶段由调用方省略。"""
    return {name: math.sqrt(total / count) if squared else total / count
            for (name, squared), total in zip(METRICS, sums, strict=True)}


# ---------- 首回合采样 ----------

@torch.no_grad()
def evaluate_case(env, actor, args, seed, level, name, commands):
    """结束行继续由底层环境维护，但不再参与本次试验的任何统计。"""
    task, device, size = env.task, env.device, env.num_envs
    phases = len(commands)
    steps = math.ceil(args.seconds / env.step_dt)
    command_table = torch.tensor(commands, device=device)
    env.set_play_commands(command_table[0])
    torch.manual_seed(seed)
    state = env.reset()
    origin = state.context[:, CONTEXT_ORIGIN].clone()
    start = task.world_position(state).clone()
    frame = task.observe(state, noisy=args.observation_noise)
    history = frame[:, None].repeat(1, task.spec.history_length, 1)
    previous_torque = state.physical[:, JOINT_TORQUE]
    previous_action = torch.zeros(size, task.spec.action_dim, device=device)
    active = torch.ones(size, dtype=torch.bool, device=device)
    fell, truncated, left_tile, pit, passed = (torch.zeros_like(active) for _ in range(5))
    counts = torch.zeros(phases, size, dtype=torch.long, device=device)
    sums = torch.zeros(phases, size, len(METRICS), dtype=torch.float64, device=device)
    forward = torch.zeros(size, device=device)
    support_steps = torch.zeros(size, dtype=torch.long, device=device)
    columns = env.terrain_curriculum.family_ids.tolist()
    families = [env.map.family_names[index] for index in columns]
    on_gap = torch.tensor([family == "gap" for family in families], device=device)
    phase = 0
    for step in range(steps):
        # 公共逐步种子避免此前结束行的数量改变下一步的随机数起点。
        torch.manual_seed(seed + step + 1)
        next_phase = step * phases // steps
        if next_phase != phase:
            phase = next_phase
            env.set_play_commands(command_table[phase])
            state = env.get_state()
            history[:, -1] = task.observe(state, noisy=args.observation_noise)
        actions = (actor(history, task.actor_heightmap(state),
                         task.actor_mask(state) * args.heightmap_visible)
                   if args.agent == "trained" else torch.zeros_like(previous_action))
        result = env.step(actions)
        final = result.final_state
        values = measurements(task, final, actions, previous_torque, previous_action, command_table[phase])
        # 索引有效行，避免失败后仿真的非有限量通过 0 * NaN 污染首回合。
        sums[phase, active] += values[active]
        counts[phase] += active
        position = task.world_position(final)
        offset = position - origin
        forward[active] = torch.maximum(forward[active], (position - start)[active, 0])
        contact = final.physical[:, CONTACT_FOUND].bool()
        foot_z = final.physical[:, FEET].reshape(size, 4, 3)[..., 2] + position[:, 2:3]
        pit |= active & on_gap & (contact & (foot_z < origin[:, 2:3] - .20)).any(-1)
        if not env.map.plane:
            outside = offset[:, 1].abs() >= env.map.size[1] / 2
            outside |= offset[:, 0] <= -env.map.size[0] / 2
            if name != "traverse":
                outside |= offset[:, 0] >= env.map.size[0] / 2
            left_tile |= active & outside
        fell |= active & result.terminated
        truncated |= active & result.truncated
        if name == "traverse":
            progress = (position - start)[:, 0] if env.map.plane else offset[:, 0]
            supported = active & (progress >= env.map.size[0] / 2) & contact.any(-1)
            support_steps = torch.where(supported, support_steps + 1, 0)
            passed |= (support_steps >= 5) & ~fell & ~truncated & ~left_tile & ~pit
        active &= ~(fell | truncated | left_tile | pit | passed)
        if not bool(active.any()):
            break
        state = result.next_state
        frame = task.observe(state, noisy=args.observation_noise)
        history = append_history(history, frame)
        previous_torque, previous_action = final.physical[:, JOINT_TORQUE], actions

    counts, sums = counts.cpu(), sums.cpu()
    records = []
    for row, family in enumerate(families):
        phase_records = [{"phase": index, "command": list(command), "samples": int(counts[index, row]),
                          "metric_sums": sums[index, row].tolist(),
                          "metrics": metric_means(sums[index, row].tolist(), int(counts[index, row]))}
                         for index, command in enumerate(commands) if int(counts[index, row]) > 0]
        records.append({
            "seed": seed, "level": level, "terrain": family, "terrain_column": columns[row],
            "scenario": name, "environment": row,
            "samples": int(counts[:, row].sum()), "seconds": int(counts[:, row].sum()) * env.step_dt,
            "fell": bool(fell[row]), "truncated": bool(truncated[row]), "left_tile": bool(left_tile[row]),
            "gap_floor_contact": bool(pit[row]), "passed": bool(passed[row]),
            "completed_horizon": bool(active[row]), "max_forward_displacement_m": float(forward[row]),
            "phases": phase_records,
        })
    return records


def summarize(records):
    """按场景／地形／难度跨种子汇总，保留各命令阶段及其实际样本数。"""
    groups = {}
    for record in records:
        key = (record["scenario"], record["terrain"], record["level"])
        groups.setdefault(key, []).append(record)
    results = []
    for (scenario, terrain, level), rows in groups.items():
        phases = {}
        for row in rows:
            for phase in row["phases"]:
                phases.setdefault(phase["phase"], []).append(phase)
        phase_results = []
        for index, samples in phases.items():
            count = sum(sample["samples"] for sample in samples)
            totals = [sum(sample["metric_sums"][i] for sample in samples) for i in range(len(METRICS))]
            phase_results.append({"phase": index, "command": samples[0]["command"], "samples": count,
                                  "episodes_observed": len(samples), "metrics": metric_means(totals, count)})
        results.append({
            "scenario": scenario, "terrain": terrain, "level": level, "episodes": len(rows),
            **{key + "_fraction": sum(row[key] for row in rows) / len(rows)
               for key in ("fell", "truncated", "left_tile", "gap_floor_contact", "completed_horizon")},
            "pass_fraction": sum(row["passed"] for row in rows) / len(rows) if scenario == "traverse" else None,
            "mean_first_episode_seconds": sum(row["seconds"] for row in rows) / len(rows),
            "mean_max_forward_displacement_m": sum(row["max_forward_displacement_m"] for row in rows) / len(rows),
            "phases": phase_results,
        })
    return results


# ---------- 命令行入口 ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Go2 checkpoints with first-episode physical metrics.")
    parser.add_argument("task", choices=TASK_MODES)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True,
                        help="Checkpoints to compare; the first supplies the shared task configuration.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--agent", choices=("trained", "zero"), default="trained")
    parser.add_argument("--suite", choices=("all", "tracking", "stop", "switch", "traverse"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--levels", nargs="+", type=int, default=[0], help="Zero-based terrain levels.")
    parser.add_argument("--families", nargs="+", default=[], help="Terrain family names; omitted means all.")
    parser.add_argument("--episodes-per-terrain", type=int, default=8,
                        help="Episodes per selected terrain column, seed, level and scenario.")
    parser.add_argument("--seconds", type=float, default=8., help="Total duration; dynamic phases split it equally.")
    parser.add_argument("--forward-speed", type=float, default=.8)
    parser.add_argument("--lateral-speed", type=float, default=.5)
    parser.add_argument("--yaw-rate", type=float, default=.8)
    parser.add_argument("--terrain-seed", type=int, default=17)
    parser.add_argument("--domain-randomization", choices=("nominal", "training"), default="nominal",
                        help="Nominal disables domain randomization and actuator delay; training keeps snapshot settings.")
    parser.add_argument("--observation-noise", action="store_true")
    parser.add_argument("--heightmap-visible", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_backends()
    checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    reference = load_playback_source(args.task, checkpoints[0])
    cfg = reference.cfg
    mode = TASK_MODES[args.task]
    if mode == "parkour" and cfg.terrain.mode != "parkour":
        cfg = parkour_config(cfg)
    elif mode != "parkour" and cfg.terrain.mode == "parkour":
        # 跨阶段评估使用目标任务的完整配方，不能仅切换地形名而保留 Parkour 奖励与命令。
        target = Go2ReferenceCfg()
        cfg.terrain, cfg.commands, cfg.reward = target.terrain, target.commands, target.reward
        cfg.reset, cfg.domain_randomization = target.reset, target.domain_randomization
    cfg.terrain.mode, cfg.terrain.seed, cfg.terrain.curriculum = mode, args.terrain_seed, False
    cfg.reset.base_offset["pose_range"]["yaw"] = (0., 0.)
    if args.domain_randomization == "nominal":
        cfg.domain_randomization.enabled = False
        cfg.control.delay_min_lag = cfg.control.delay_max_lag = 0
    terrain = TerrainMap(cfg.terrain)
    columns = {}
    for index, name in enumerate(terrain.family_names):
        columns.setdefault(name, []).append(index)
    family_ids = ([index for name in args.families for index in columns[name]]
                  if args.families else list(range(terrain.cols)))
    cfg.env.num_envs = len(family_ids) * args.episodes_per_terrain
    cases = scenarios(args)
    results = []
    for checkpoint in checkpoints:
        source = load_playback_source(args.task, checkpoint)
        records = []
        for seed in args.seeds:
            cfg.env.seed = seed
            for level in args.levels:
                for name, commands in cases:
                    # 独立环境使前一场景的早停不改变本场景的推扰和延迟时钟。
                    env = Go2TransitionAdapter(cfg, device=args.device, play=True)
                    try:
                        env.cfg.noise.add_noise = args.observation_noise
                        actor = build_playback_policy(source, env)
                        env.terrain_curriculum.family_ids.copy_(torch.tensor(family_ids, device=env.device)
                                                                .repeat_interleave(args.episodes_per_terrain))
                        env.terrain_curriculum.levels.fill_(level)
                        records.extend(evaluate_case(env, actor, args, seed, level, name, commands))
                        evaluation_config = env.cfg.to_dict()
                    finally:
                        env.close()
                    print(f"[INFO] Evaluated checkpoint={checkpoint.name} seed={seed} level={level} scenario={name}",
                          flush=True)
        summary = summarize(records)
        results.append({"checkpoint": str(checkpoint), "iteration": source.checkpoint["iteration"],
                        "summary": summary, "episodes": records})
        for row in summary:
            first_phase = row["phases"][0]["metrics"]
            print(f"[RESULT] checkpoint={checkpoint.name} scenario={row['scenario']} terrain={row['terrain']} "
                  f"level={row['level']} episodes={row['episodes']} falls={row['fell_fraction']:.3f} "
                  f"pass={row['pass_fraction']} seconds={row['mean_first_episode_seconds']:.3f} "
                  f"phase0_linear_rmse={first_phase['linear_velocity_rmse_mps']:.3f} "
                  f"phase0_power_w={first_phase['mean_absolute_mechanical_power_w']:.3f}", flush=True)
    output = {
        "protocol": "go2_first_episode_v1", "task": args.task, "configuration_checkpoint": str(checkpoints[0]),
        "arguments": {**vars(args), "checkpoints": [str(path) for path in checkpoints], "output": str(args.output)},
        "evaluation_config": evaluation_config, "metric_order": [name for name, _ in METRICS],
        "definitions": {
            "command_frame": "body_xy_body_yaw", "policy": "deterministic_fp32",
            "sampling": "First episode including failure step; no warmup exclusion; control-step physical metrics.",
            "traversal": "Reach original tile +x edge (flat: half-tile displacement), with 5 consecutive supported "
                         "control steps; no fall, truncation, side/rear exit or gap-floor contact below origin_z - 0.20m.",
            "dynamic_phases": "Equal duration command phases; missing phases after failure are omitted, not zero-filled.",
            "torque_step": "Per-joint RMS torque difference at control_dt, including initial-state to first-step difference.",
            "comparison": "Shared first-checkpoint task config, terrain seed, family assignment and per-step Torch seeds; "
                          "this does not establish exact equality of every trajectory-dependent random draw.",
            "cross_stage": "Changing between ordinary terrain and Parkour selects the target stage task recipe; "
                           "Parkour to Flat/Rough replaces terrain, commands, rewards, reset and DR with project defaults.",
            "score": "Raw physical metrics only; not the RoboGauge normalized benchmark score.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[INFO] Evaluation completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
