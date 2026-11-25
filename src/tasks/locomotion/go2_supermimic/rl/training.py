# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2 SuperMimic训练、参数续训、独立产物目录和周期检查点保存。
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
import yaml
from mjlab.utils.torch import configure_torch_backends

from agents.supermimic.config import ExecutionConfig, RUNNER_RUNTIME_FIELDS
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg
from src.runtime import configure_cuda_libdevice

from ..config.loading import load_task_config
from ..config.train_cfg import SuperMimicTrainConfig, TASK_MODES
from .runner import make_go2_supermimic_runner


# ---------- 参数与任务配置 ----------

def _validate_arguments(args: SuperMimicTrainConfig) -> None:
    if args.resume is None and (args.continue_run or args.reset_critic):
        raise ValueError("continue_run and reset_critic require a resume checkpoint")
    if re.fullmatch(r"cpu|cuda(?::\d+)?", args.device) is None:
        raise ValueError("SuperMimic device must be cpu or one CUDA device such as cuda:0")
    if not args.log_root.strip():
        raise ValueError("log_root must be a nonempty path")
    for name, value in (("experiment_name", args.experiment_name), ("run_name", args.run_name)):
        if value in (".", "..") or "/" in value or "\\" in value:
            raise ValueError(f"{name} must be a single directory component")
    if not args.experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Distributed SuperMimic training is not supported; WORLD_SIZE must be 1")


def _resolve_task_config(task_id: str, args: SuperMimicTrainConfig):
    resume = Path(args.resume).expanduser().resolve() if args.resume is not None else None
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {resume}")
    source = Path(args.task_config).expanduser().resolve() if args.task_config is not None else None
    if source is None and resume is not None:
        source = resume.parent / "task_config.json"
        if not source.is_file():
            raise FileNotFoundError(f"Resume requires {source} or an explicit --task-config")
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
    for name, value, minimum in (("max_iterations", cfg.runner.max_iterations, 1),
                                 ("save_interval", cfg.runner.save_interval, 1),
                                 ("log_interval", cfg.runner.log_interval, 1),
                                 ("num_envs", cfg.env.num_envs, 2), ("seed", cfg.env.seed, 0)):
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    return cfg, resume, source


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return requested
    device = torch.device(requested)
    index = device.index if device.index is not None else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA device is unavailable: cuda:{index}; use --device cpu for CPU")
    return f"cuda:{index}"


# ---------- 运行快照与训练装配 ----------

def _run_origin(task_id: str, args: SuperMimicTrainConfig, device: str, resume: Path | None, source: Path | None) -> dict:
    """记录本次训练的启动参数与运行环境。"""
    return {"task_id": task_id, "started_at": datetime.now().astimezone().isoformat(),
            "arguments": asdict(args), "device": device, "resume_from": resume,
            "task_config_source": source, "command": list(sys.argv), "python": sys.version,
            "torch": torch.__version__,
            "triton_libdevice_path": os.environ.get("TRITON_LIBDEVICE_PATH"),
            "packages": {name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "tensorboard")}}


def launch_training(task_id: str, args: SuperMimicTrainConfig) -> Path:
    """调用既有runner训练至目标轮次，周期保存模型和ONNX并返回独立产物目录。"""
    entry_started = time.perf_counter()
    _validate_arguments(args)
    cfg, resume, source = _resolve_task_config(task_id, args)
    runtime = cfg.runner
    device = _resolve_device(args.device)
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
    timing, execution = runner.timing, runner.execution

    def close_timed(name, close):
        started = time.perf_counter()
        close()
        timing.record_wall(name, time.perf_counter() - started)

    with ExitStack() as cleanup:
        cleanup.callback(close_timed, "environment_close", runner.env.close)
        cleanup.callback(close_timed, "logger_close", runner.close)
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
        settings = cfg.to_dict()
        agent = {"algorithm": settings.pop("algorithm"), "config": asdict(runner.config),
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
        "start_iteration": starting_iteration, "target_iteration": runtime.max_iterations, "final_iteration": runner.iteration,
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
