# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2 SuperMimic训练、参数续训、独立产物目录和周期检查点保存。
"""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml
from mjlab.utils.torch import configure_torch_backends

from agents.supermimic.config import ExecutionConfig, RUNNER_RUNTIME_FIELDS
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg
from src.runtime import configure_cuda_libdevice

from ..config.loading import load_task_config, algorithm_config
from ..config.train_cfg import SuperMimicTrainConfig, TASK_MODES
from .runner import make_go2_supermimic_runner


# ---------- 参数与任务配置 ----------

def _resolve_task_config(task_id: str, args: SuperMimicTrainConfig):
    resume = Path(args.resume).expanduser().resolve() if args.resume is not None else None
    source = Path(args.task_config).expanduser().resolve() if args.task_config is not None else None
    if source is None and resume is not None:
        source = resume.parent / "task_config.json"
    cfg = load_task_config(source)[0] if source is not None else Go2ReferenceCfg()
    cfg.terrain.mode = TASK_MODES[task_id]
    for field, value in (("num_envs", args.num_envs), ("seed", args.seed)):
        if value is not None:
            setattr(cfg.env, field, value)
    if args.model_only is not None or resume is None:
        enabled = not bool(args.model_only)
        cfg.runner.enable_control_updates = enabled
    for name in RUNNER_RUNTIME_FIELDS:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.runner, name, value)
    return cfg, resume, source


# ---------- 运行快照与训练装配 ----------

def _run_origin(task_id: str, args: SuperMimicTrainConfig, device: str, resume: Path | None, source: Path | None) -> dict:
    """收集命令、依赖版本与代码状态，供事件文本和params快照共同使用。"""
    root = Path(__file__).resolve().parents[5]
    workspace = {}
    for name, command in (("revision", ["git", "rev-parse", "HEAD"]),
                          ("status", ["git", "status", "--short"]),
                          ("diff", ["git", "diff", "HEAD", "--"])):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
            workspace[name] = (result.stdout if name == "diff" else result.stdout.strip()) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            workspace[name] = None
    return {"task_id": task_id, "started_at": datetime.now().astimezone().isoformat(),
            "arguments": asdict(args), "device": device, "resume_from": resume,
            "task_config_source": source, "command": list(sys.argv), "python": sys.version,
            "torch": torch.__version__,
            "triton_libdevice_path": os.environ.get("TRITON_LIBDEVICE_PATH"),
            "packages": {name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "tensorboard")},
            "workspace": workspace}


def launch_training(task_id: str, args: SuperMimicTrainConfig) -> Path:
    """调用既有runner训练至目标轮次，周期保存模型和ONNX并返回独立产物目录。"""
    entry_started = time.perf_counter()
    cfg, resume, source = _resolve_task_config(task_id, args)
    runtime = cfg.runner
    device = args.device
    if torch.device(device).type == "cuda":
        configure_cuda_libdevice()
    log_root = (Path(args.log_root).expanduser() / args.experiment_name).resolve()
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.run_name:
        name += f"_{args.run_name}"
    log_dir = log_root / name
    log_dir.mkdir(parents=True, exist_ok=False)
    configure_torch_backends()
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)
    print(f"[INFO] Training with: task={task_id}, device={device}, seed={cfg.env.seed}", flush=True)
    print(f"[INFO] Control updates enabled: {cfg.runner.enable_control_updates}", flush=True)
    execution = ExecutionConfig(**{field.name: getattr(runtime, field.name) for field in fields(ExecutionConfig)})
    runner = make_go2_supermimic_runner(cfg, device=device, log_dir=log_dir, execution=execution)
    try:
        timing, execution = runner.timing, runner.execution
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        timing.record_wall("initialization", time.perf_counter() - entry_started)
        print(f"[INFO] SuperMimic {cfg.algorithm} training: envs={cfg.env.num_envs}, "
              f"collect_avg={cfg.algo.collect_steps:g}, utd={1.0 / cfg.algo.collect_steps:g}, "
              f"wm_batch={cfg.algo.wm_batch}, imag_batch={cfg.algo.imag_batch}, "
              f"replay={cfg.algo.replay_capacity}, compile={cfg.algo.compile_imagination}, "
              f"imag_microbatch={execution.imag_microbatch}, critic_microbatch={execution.critic_microbatch}, "
              f"wm_compile={execution.wm_compile}, fused_adam={execution.fused_adam}, "
              f"cuda_graph={execution.cuda_graph}, detailed_timing={execution.detailed_timing}, "
              f"wm_bf16={execution.wm_bf16}, actor_bf16={execution.actor_bf16}, critic_bf16={execution.critic_bf16}, "
              f"timing_interval={execution.timing_interval}, metric_interval={execution.metric_interval}", flush=True)
        # 快照反映JSON与CLI覆盖后的实际配置，普通YAML类型便于直接查看和安全回读。
        origin = json.loads(json.dumps(_run_origin(task_id, args, device, resume, source), default=str))
        origin.update(seed=cfg.env.seed, num_envs=cfg.env.num_envs)
        params = log_dir / "params"
        params.mkdir()
        diff = origin["workspace"].pop("diff")
        origin["workspace"]["diff_file"] = "params/git.diff" if diff is not None else None
        if diff is not None:
            (params / "git.diff").write_text(diff, encoding="utf-8")
        settings = cfg.to_dict()
        agent = {"algorithm": settings.pop("algorithm"), "config": asdict(algorithm_config(cfg)),
                 "execution": asdict(execution), "runner": settings.pop("runner")}
        settings.pop("algo")
        for filename, values in (("env.yaml", settings), ("agent.yaml", agent), ("run.yaml", origin)):
            (params / filename).write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
        runner.logger.add_text("run/origin", "```json\n" + json.dumps(
            origin, indent=2) + "\n```")
        if resume is not None:
            print(f"[INFO] Loading model checkpoint from: {resume}", flush=True)
            restore_started = time.perf_counter()
            try:
                runner.load(resume, continue_run=args.continue_run, reset_critic=args.reset_critic)
            finally:
                timing.record_wall("restore", time.perf_counter() - restore_started)
        starting_steps, starting_iteration = runner.real_steps, runner.iteration
        starting_prefill_steps = runner.prefill_real_steps
        target_iteration = runtime.max_iterations
        if target_iteration <= runner.iteration:
            raise ValueError(f"Target iteration {target_iteration} must exceed checkpoint iteration {runner.iteration}")
        runner.configure_logging(target_iteration=target_iteration, log_interval=runtime.log_interval)
        while runner.iteration < target_iteration:
            remaining = target_iteration - runner.iteration
            until_checkpoint = runtime.save_interval - runner.iteration % runtime.save_interval
            learn_started = time.perf_counter()
            try:
                runner.learn(min(remaining, until_checkpoint))
            finally:
                timing.record_wall("learn_blocks", time.perf_counter() - learn_started)
            # 参考在塌缩护栏首次预警时跳过该保存点，不发布可疑策略及其ONNX。
            if runner._collapse_strikes == 0:
                checkpoint = log_dir / f"model_{runner.iteration}.pt"
                runner.save(checkpoint)
                print(f"[INFO] Saved checkpoint: {checkpoint}", flush=True)
    finally:
        try:
            runner.close()
        finally:
            runner.env.close()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    entry_seconds = time.perf_counter() - entry_started
    timing.record_wall("entry_total", entry_seconds)
    summary = timing.summary()
    added_steps = runner.real_steps - starting_steps
    prefill_steps = runner.prefill_real_steps - starting_prefill_steps
    training_steps = added_steps - prefill_steps
    summary["run"] = {
        "task_id": task_id, "seed": cfg.env.seed, "num_envs": cfg.env.num_envs,
        "initialized": True, "training_completed": True, "success": True, "error": None,
        "start_iteration": starting_iteration, "target_iteration": runtime.max_iterations,
        "final_iteration": runner.iteration,
        "new_real_steps_including_prefill": added_steps,
        "new_prefill_real_steps": prefill_steps, "new_training_real_steps": training_steps,
        "entry_real_transitions_per_second": added_steps / entry_seconds,
        "entry_training_transitions_per_second": training_steps / entry_seconds,
        "runner_internal_seconds": runner.elapsed,
    }
    summary["scope"].update({
        "entry_total": "launch_training entry through logger/environment close and final target-device synchronization; excludes this summary write",
        "learn_blocks": "Inclusive host wall of runner.learn calls; includes prefill, setup and per-iteration logging, excludes outer checkpoint save",
        "save_total": "Inclusive parent of checkpoint_prepare, onnx_export, checkpoint_publish and temporary-directory cleanup; do not sum parent and children",
        "process_wall": "Not measured here; the explicit training test launcher measures child-process wall externally",
        "transitions": "New steps subtract restored counters; entry_real_transitions_per_second includes prefill, entry_training_transitions_per_second excludes it",
    })
    (log_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return log_dir
