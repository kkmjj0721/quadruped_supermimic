# Copyright (c) 2026-2027 zh
"""内容：执行真实采集、世界模型拟合、策略更新、回放归档及参数续训。"""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import time

import torch

from .compilation import compile_kernel
from .interfaces import append_history
from .logging import TrainingLogger
from .buffer import Replay, Window, WindowArchive
from .config import ExecutionConfig, SuperMimicConfig
from .exploration import stratified_collection_mask
from .learner import Learner
from .metrics import EpisodeWindow, MetricWindow, ModelMetrics, read_metrics
from .policy import Critic, Mlp, make_actor
from .timing import PhaseTimer, TrainingTiming
from .world_model import WorldModel


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return deepcopy(value)


class SuperMimicRunner:
    """参考采集与优化循环；恢复区分重新计时的热启动与保留迭代号的继续训练。"""

    def __init__(self, env, config: SuperMimicConfig, log_dir=None, control_enabled=True, execution=None,
                 collection_groups=None):
        self.env, self.task, self.spec = env, env.task, env.task.spec
        self.device = torch.device(env.device)
        self.config = config
        self.execution = execution if execution is not None else ExecutionConfig()
        self.timing = TrainingTiming(self.device, enabled=self.execution.detailed_timing,
                                     interval=self.execution.timing_interval)
        cfg = self.config
        if collection_groups is None:
            collection_groups = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.deterministic_collection_mask = stratified_collection_mask(collection_groups, cfg.collect_det_frac)
        self.actor = make_actor(self.spec, cfg, self.task.observation_term_dims).to(self.device)
        self.policy = self.actor
        self.model = self._make_world_model().to(self.device)
        self.critic = Critic(self.spec.privileged_dim, cfg.critic_hidden).to(self.device)
        self.model.use_bf16 = self.execution.wm_bf16
        for module, enabled in ((self.actor, self.execution.actor_bf16), (self.critic, self.execution.critic_bf16)):
            for layer in module.modules():
                if isinstance(layer, Mlp):
                    layer.use_bf16 = enabled
        self.model.configure_supervised_compile(enabled=self.execution.wm_compile, backend=cfg.compile_backend)
        self.learner = self._make_learner()
        self.replay = Replay(self.spec, env.num_envs, cfg.replay_capacity, self.device,
                             history_storage=self.execution.replay_history_storage,
                             device_sampling=self.execution.cuda_graph)
        self.archive = WindowArchive(cfg.archive_size, self.device)
        self._configure_execution()
        self.control_enabled = bool(control_enabled)
        self.iteration = self.real_steps = self.imagined_steps = 0
        self.prefill_real_steps = 0
        self._collect_step_accumulator = 0.0
        self.current = self.history = None
        self.driver = self.expert = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.logger = TrainingLogger(self.log_dir, num_envs=env.num_envs, device=self.device)
        self.target_iteration = 0
        self.elapsed = 0.0
        self.session_iterations = 0
        self.episode_returns = torch.zeros(env.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.completed_episodes = EpisodeWindow(env.num_envs, self.device)
        self._collection_metric_validity = {}
        self._safe_policy = None
        self._collapse_strikes = 0
        self._guard_falls = 0.0
        self._guard_reset_steps = 0

    def _make_world_model(self):
        """默认使用通用状态模型；任务子类可指定其输出统计语义。"""
        return WorldModel(self.task, self.config.wm_hidden)

    def _make_learner(self):
        """初始化与恢复共用显式执行选择。"""
        learner_type = Learner
        if self.execution.cuda_graph:
            from .cuda_graph import CudaGraphLearner

            learner_type = CudaGraphLearner
        return learner_type(self.task, self.actor, self.model, self.critic, self.config, execution=self.execution)

    def _configure_execution(self):
        """采样与更新使用同一图，恢复时重建静态地址和捕获生命周期。"""
        self._model_training = None
        self._policy_training = {}
        if self.replay.device_sampling != self.execution.cuda_graph:
            self.replay = Replay(self.spec, self.env.num_envs, self.config.replay_capacity, self.device,
                                 history_storage=self.execution.replay_history_storage,
                                 device_sampling=self.execution.cuda_graph)
        if self.execution.cuda_graph:
            from .cuda_graph import CapturedUpdate

            self._model_training = CapturedUpdate(self._fit_models, self.device)
            self._policy_graph_generation = self.learner.graph_generation
            self._model_graph_generation = self.learner.model_graph_generation
        self._collection_heightmap = (compile_kernel(self.task.actor_heightmap, self.config.compile_backend)
                                      if self.execution.cuda_graph else self.task.actor_heightmap)
        self._collection_policy = (compile_kernel(self.actor, self.config.compile_backend)
                                   if self.execution.cuda_graph else self.actor)
        self._collection_update = (compile_kernel(self._advance_collection, self.config.compile_backend)
                                   if self.execution.cuda_graph else self._advance_collection)

    def _fit_models(self):
        """将整轮采样、全部世界模型更新与设备指标累计留在同一计算链。"""
        totals = ModelMetrics()
        for _ in range(self.config.wm_updates):
            window = self._model_batch(device_indices=self.execution.cuda_graph)
            totals.add(self.learner.fit_model(window))
        return totals.mean(self.config.wm_updates)

    def fit_model_updates(self):
        with self.timing.section("wm_sampling_update"):
            if self.execution.cuda_graph:
                if self._model_graph_generation != self.learner.model_graph_generation:
                    self._model_training.invalidate()
                    self._model_graph_generation = self.learner.model_graph_generation
                metrics = self._model_training()
                metrics = {**metrics, "cuda_graph/model_captured": float(self._model_training.graph is not None),
                           "cuda_graph/model_replays": self._model_training.replays}
                return metrics
            return self._fit_models()

    def improve_policy(self, *, freeze_actor):
        if self.execution.cuda_graph:
            from .cuda_graph import CapturedUpdate

            if self._policy_graph_generation != self.learner.graph_generation:
                for update in self._policy_training.values():
                    update.invalidate()
                self._policy_graph_generation = self.learner.graph_generation
            if freeze_actor not in self._policy_training:
                self._policy_training[freeze_actor] = CapturedUpdate(
                    lambda: self.learner.improve(self._sample_seed_device(), freeze_actor=freeze_actor), self.device)
            update = self._policy_training[freeze_actor]
            with self.timing.section("policy_sampling_update"):
                metrics = update()
            return {**metrics, "cuda_graph/actor_captured": float(update.graph is not None),
                    "cuda_graph/critic_captured": float(update.graph is not None),
                    "cuda_graph/actor_replays": update.replays, "cuda_graph/critic_replays": update.replays}
        with self.timing.section("seed_sampling"):
            seed = self.replay.sample_seeds(self.config.imag_batch)
        return self.learner.improve(seed, freeze_actor=freeze_actor, timing=self.timing)

    def _training_setup_pending(self, control_requested, freeze_actor):
        if not self.execution.cuda_graph:
            return False
        return (self._model_training.graph is None
                or self._model_graph_generation != self.learner.model_graph_generation or control_requested and (
            self._policy_graph_generation != self.learner.graph_generation
            or freeze_actor not in self._policy_training or self._policy_training[freeze_actor].graph is None))

    def configure_logging(self, *, target_iteration, log_interval=1):
        self.target_iteration = int(target_iteration)
        self.logger.log_interval = int(log_interval)

    def _collection_step_metrics(self):
        """任务子类提供设备上的原始／加权指标，算法不硬编码任务奖励。"""
        return {}

    def _metric_columns(self):
        """预注册算法全阶段设备指标；主机标量仍由原字典路径累计。"""
        scalar = torch.zeros((), device=self.device)
        integer = torch.zeros((), dtype=torch.int64, device=self.device)
        precise = torch.zeros((), dtype=torch.float64, device=self.device)
        columns = {name: scalar for name in (
            "real_reward", "failures", "timeouts", "episode/completed", "wm_loss", "wm_grad_norm", "wm_valid_steps",
            "control_loss", "velocity_loss", "actor_loss", "imagination/survival_h", "imagination/discounted_reward",
            "imagination/bootstrap", "imagination/reward", "velocity_rmse", "velocity_rmse_x", "velocity_rmse_y",
            "velocity_rmse_z", "control_grad_norm", "critic_loss")}
        columns.update((name, integer) for name in (
            "reset_steps", "world_model_updates_total", "control_updates_total", "critic_updates_total",
            "world_model_updates_delta", "control_updates_delta", "critic_updates_delta", "replay/archive_size",
            "episode/window_count"))
        columns.update((name, precise) for name in (
            "episode/return_mean", "episode/return_min", "episode/return_max", "episode/return_std",
            "episode/length_steps_mean", "episode/length_seconds_mean"))
        if self.execution.cuda_graph:
            columns.update((name, scalar) for name in (
                "world_model_learning_rate", "critic_learning_rate", "control_learning_rate"))
        return columns

    def _override_actions(self, action, history, observation):
        """任务子类可用外部策略接管部分环境的动作；算法只保留调用位置。"""
        return action

    def _driver_scale(self):
        """驱动占比按参考实现从起始值线性衰减到结束值。"""
        start, end = self.config.driver_fade
        if self.iteration < start:
            return 1.0
        if self.iteration >= end:
            return 0.0
        return (end - self.iteration) / (end - start)

    def _reset_drivers(self, done):
        """任务子类在回合结束时清理外部策略状态；默认没有外部策略。"""
        return None

    @torch.no_grad()
    def reset_collection(self):
        self.current = self.env.reset()
        self._collect_step_accumulator = 0.0
        frame = self.task.observe(self.current, noisy=True)
        self.history = frame[:, None].repeat(1, self.spec.history_length, 1)
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        self.completed_episodes.clear()
        self._collection_metric_validity = {}
        self.replay.clear()
        self.archive.clear()
        if self.driver is not None or self.expert is not None:
            self._reset_drivers(torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device))

    @torch.no_grad()
    def collect(self, steps):
        if self.current is None:
            self.reset_collection()
        cfg = self.config
        totals = defaultdict(int)
        metric_counts = defaultdict(int)
        reward_sum = torch.zeros((), device=self.device)
        failure_count = torch.zeros((), device=self.device)
        timeout_count = torch.zeros((), device=self.device)
        reset_steps = torch.zeros((), dtype=torch.long, device=self.device)
        for _ in range(steps):
            observation = self.history[:, -1]
            heightmap = self._collection_heightmap(self.current)
            mask = self.task.actor_mask(self.current)
            action = self._override_actions(self._collection_policy(self.history, heightmap, mask), self.history, observation)
            noise = cfg.action_noise * torch.randn_like(action)
            noise.masked_fill_(self.deterministic_collection_mask[:, None], 0)
            action = action + noise
            result = self.env.step(action)
            done = result.terminated | result.truncated
            with self.timing.section("collection_tensor_update"):
                history = self._collection_update(self.current, self.history, action, result)
            with self.timing.section("replay_history_write"):
                self.replay.add(self.current, self.history, action, result)
            reset_steps.add_(done.any())
            self.history = history
            if self.driver is not None or self.expert is not None:
                self._reset_drivers(done)
            self.current = result.next_state
            reward_sum += result.reward.mean()
            failure_count += result.terminated.sum()
            timeout_count += (result.truncated & ~result.terminated).sum()
            for name, value in self._collection_step_metrics().items():
                totals[name] += value
                metric_counts[name] += 1
            self.real_steps += self.env.num_envs
        # 回合指标只在结束步出现，按实际出现次数平均，与RSL-RL一致。
        metrics = {name: value / metric_counts[name] for name, value in totals.items()}
        metrics.update(real_reward=reward_sum / steps, failures=failure_count, timeouts=timeout_count,
                       reset_steps=reset_steps)
        metrics["episode/completed"] = failure_count + timeout_count
        episode_metrics, self._collection_metric_validity = self.completed_episodes.metrics(self.task.dt)
        metrics.update(episode_metrics)
        return metrics

    @torch.no_grad()
    def _advance_collection(self, state, history, action, result):
        """融合统计、回合记录与历史推进；保留终止掩码及原有随机数顺序。"""
        self.actor.update_statistics(history)
        done = result.terminated | result.truncated
        self.model.update_stats(state, action, result.final_state, mask=~done)
        self.episode_returns.add_(result.reward)
        self.episode_lengths.add_(1)
        self.completed_episodes.append(self.episode_returns, self.episode_lengths, done)
        self.episode_returns.masked_fill_(done, 0)
        self.episode_lengths.masked_fill_(done, 0)
        frame = self.task.observe(result.next_state, noisy=True)
        following = append_history(history, frame)
        return torch.where(done[:, None, None], frame[:, None].expand_as(following), following)

    def _sample_seed_device(self):
        return self.replay.sample_seeds(self.config.imag_batch, device_indices=True)

    def _model_batch(self, *, device_indices=False):
        cfg = self.config
        prioritized = int(cfg.wm_batch * cfg.wm_prioritized_frac)
        archived = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.capacity else 0
        uniform = cfg.wm_batch - prioritized - archived
        parts = [self.replay.sample_windows(uniform, cfg.wm_window, device_indices=device_indices)] if uniform else []
        populated = self.archive.size > 0
        if archived:
            archive_rows, archive_starts = self.replay.sample_window_indices(
                archived, cfg.wm_window, device_indices=device_indices)
        rows, starts = self.replay.sample_window_indices(
            4 * max(prioritized, 1), cfg.wm_window, device_indices=device_indices)
        features, valid = self.replay.window_features(rows, starts, cfg.wm_window, self.task.replay_feature_indices)
        scores, worthy = self.task.replay_selection_from_features(features, valid)
        keep = worthy & (torch.rand_like(scores) < cfg.archive_keep_prob)
        if self.archive.capacity:
            self.archive.add_from_replay(self.replay, rows, starts, cfg.wm_window, valid, keep)
        chosen = (scores + 1e-3 * torch.rand_like(scores)).topk(prioritized).indices
        if prioritized:
            parts.append(self.replay.gather_windows(rows[chosen], starts[chosen], cfg.wm_window))
        if archived:
            parts.append(self.archive.sample_or_replay(
                self.replay, archive_rows, archive_starts, cfg.wm_window, populated))
        return Window.concatenate(parts) if len(parts) > 1 else parts[0]

    def _collapse_guard(self, metrics):
        """模型被利用表现为真实摔倒激增：回滚策略并降低学习率。

        MjLab只在发生reset的step输出Episode_Termination；因此分母是发生过reset
        的step数（含纯超时step），不是窗口内所有step数，也不是环境分钟。
        """
        cfg = self.config
        self._guard_falls += metrics["failures"]
        self._guard_reset_steps += metrics["reset_steps"]
        if self.iteration % cfg.collapse_interval:
            return
        falls = self._guard_falls / max(self._guard_reset_steps, 1)
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        metrics["guard/falls_per_reset_step"] = falls
        if self.iteration <= self.learner.actor_frozen_until or falls <= cfg.collapse_falls:
            self._collapse_strikes = 0
            return
        self._collapse_strikes += 1
        if self._collapse_strikes < 2 or self._safe_policy is None:
            return
        for name in ("actor", "critic"):
            getattr(self, name).load_state_dict(self._safe_policy[name])
        self.learner.reset_policy_optimizers()
        self.learner.lr_multiplier *= .5
        self.learner.ramp_start = self.iteration
        self._collapse_strikes = 0
        metrics["guard/rollback"] = 1.0
        print(f"[WARN] Collapse guard: {falls:.1f} falls per reset step, "
              f"restored saved policy and value states; lr scale {self.learner.lr_multiplier:.3g}", flush=True)

    def learn(self, iterations):
        output = []
        metric_window = MetricWindow(self.device, compile_backend=self.config.compile_backend
                                     if self.execution.cuda_graph else None)
        metric_window.register_columns(self._metric_columns())
        phase_timers = []
        window_started = None
        previous_window = None
        target = max(self.target_iteration, self.iteration + iterations)
        # 预填使用整数批次且不消耗训练累加器；默认8步窗口仍先采10步。
        if self.replay.size <= self.config.wm_window + 1:
            prefill_started = time.perf_counter()
            prefill_previous_steps = self.real_steps
            prefill_falls = torch.zeros((), device=self.device)
            prefill_reset_steps = torch.zeros((), dtype=torch.long, device=self.device)
            while self.replay.size <= self.config.wm_window + 1:
                metrics = self.collect(int(self.config.collect_steps))
                prefill_falls += metrics["failures"]
                prefill_reset_steps += metrics["reset_steps"]
            prefill = read_metrics({"failures": prefill_falls, "reset_steps": prefill_reset_steps}, self.device)
            self._guard_falls += prefill["failures"]
            self._guard_reset_steps += prefill["reset_steps"]
            prefill_seconds = time.perf_counter() - prefill_started
            self.elapsed += prefill_seconds
            self.prefill_real_steps += self.real_steps - prefill_previous_steps
            self.timing.record_wall("prefill", prefill_seconds)
        for local_iteration in range(iterations):
            # 来源在真实采集前设置当前迭代号，驱动衰减与学习率使用同一编号。
            self.iteration += 1
            started = time.perf_counter()
            if window_started is None:
                window_started = started
            timer = PhaseTimer(self.device)
            previous_steps = self.real_steps
            previous_model_updates = self.learner.model_updates.clone()
            previous_actor_updates = self.learner.actor_updates.clone()
            previous_critic_updates = self.learner.critic_updates.clone()
            control_requested = self.control_enabled and self.iteration > self.learner.policy_start
            frozen = self.iteration <= self.learner.actor_frozen_until
            if self.execution.detailed_timing:
                phase = "actor_frozen" if control_requested and frozen else "training" if control_requested else "model_warmup"
                setup = self._training_setup_pending(control_requested, frozen)
                self.timing.begin_iteration(self.iteration, phase, setup=setup)
            # 累加器跨learn调用保留余量，默认依次采2、3步。
            self._collect_step_accumulator += self.config.collect_steps
            collect_steps = int(self._collect_step_accumulator)
            self._collect_step_accumulator -= collect_steps
            with self.timing.section("collection"):
                metrics = self.collect(collect_steps)
            metrics["collection/control_steps"] = collect_steps
            timer.mark("collection")
            self.learner.schedule(self.iteration)
            enabled = False
            metrics.update(self.fit_model_updates())
            timer.mark("model")
            if self.control_enabled and self.iteration > self.learner.policy_start:
                enabled = True
                policy_metrics = self.improve_policy(freeze_actor=frozen)
                metrics.update(policy_metrics)
                metrics["control/actor_frozen"] = float(frozen)
                self.imagined_steps += self.config.horizon * self.config.imag_batch
            timer.mark("control")
            delta = self.real_steps - previous_steps
            metrics.update(iteration=self.iteration, training_target_iteration=target, real_steps=self.real_steps,
                           real_steps_delta=delta, control_enabled=float(enabled), imagined_steps_total=self.imagined_steps,
                           world_model_updates_total=self.learner.model_updates, control_updates_total=self.learner.actor_updates,
                           critic_updates_total=self.learner.critic_updates,
                           world_model_updates_delta=self.learner.model_updates - previous_model_updates,
                           control_updates_delta=self.learner.actor_updates - previous_actor_updates,
                           critic_updates_delta=self.learner.critic_updates - previous_critic_updates,
                           world_model_learning_rate=self.learner.model_optimizer.param_groups[0]["lr"],
                           critic_learning_rate=self.learner.critic_optimizer.param_groups[0]["lr"],
                           control_learning_rate=self.learner.actor_optimizer.param_groups[0]["lr"],
                           **{"replay/steps_per_env": self.replay.size,
                              "replay/transitions": self.replay.size * self.env.num_envs,
                              "replay/fill_fraction": self.replay.size / self.replay.capacity,
                              "replay/archive_size": self.archive.size})
            metric_window.add(metrics, validity=self._collection_metric_validity)
            phase_timers.append((timer, enabled))
            flush = (self.session_iterations == 0 or self.iteration % self.execution.metric_interval == 0
                     or self.iteration % self.config.collapse_interval == 0 or local_iteration == iterations - 1)
            self.session_iterations += 1
            if not flush:
                self.timing.end_iteration(time.perf_counter() - started, 0.0, completed=False)
                continue
            # 展示数据仅在窗口边界集中回读，原护栏轮次也强制结束窗口。
            with self.timing.section("metric_read", cuda=False):
                aggregate, validity = metric_window.aggregate()
                metrics = read_metrics(aggregate, self.device, validity=validity)
            collection_time = sum(item.seconds("start", "collection") for item, _ in phase_timers)
            model_time = sum(item.seconds("collection", "model") for item, _ in phase_timers)
            control_time = sum(item.seconds("model", "control") for item, active in phase_timers if active)
            metrics.update(self.timing.device_metrics())
            if self.execution.metric_interval == 1:
                metrics.update(self.timing.previous_metrics())
            if previous_window is not None:
                metrics.update(previous_window)
            elapsed = time.perf_counter() - window_started
            count = len(metric_window)
            session_elapsed = self.elapsed + elapsed
            metrics.update({"timing/collection_seconds": collection_time / count,
                            "timing/world_model_seconds": model_time / count,
                            "timing/control_seconds": control_time / count,
                            "timing/learning_seconds": (model_time + control_time) / count,
                            "timing/iteration_seconds": elapsed / count,
                            "timing/window_seconds": elapsed,
                            "timing/window_collection_seconds": collection_time,
                            "timing/window_world_model_seconds": model_time,
                            "timing/window_control_seconds": control_time,
                            "timing/session_elapsed_seconds": session_elapsed,
                            "timing/eta_seconds": session_elapsed / self.session_iterations * (target - self.iteration),
                            "timing/cuda_events": float(self.device.type == "cuda"),
                            "performance/real_total_fps": metrics["real_steps_delta"] / elapsed,
                            "performance/real_collection_fps": metrics["real_steps_delta"] / collection_time})
            if metrics["episode/completed"]:
                metrics["episode/failure_fraction"] = metrics["failures"] / metrics["episode/completed"]
                metrics["episode/timeout_fraction"] = metrics["timeouts"] / metrics["episode/completed"]
            self._collapse_guard(metrics)
            logging_started = time.perf_counter()
            self.logger.record(metrics)
            finished = time.perf_counter()
            completed_wall = finished - window_started
            self.elapsed += completed_wall
            self.timing.record_wall("metrics_windows", completed_wall)
            self.timing.record_wall("metrics_logging", finished - logging_started)
            self.timing.end_iteration(finished - started, finished - logging_started)
            # 当前日志写完后才知道完整窗口时间，下一窗口记录原起止轮号；末窗保存在summary。
            previous_window = {"timing/previous_window_start_iteration": metrics["metrics/window_start_iteration"],
                               "timing/previous_window_end_iteration": self.iteration,
                               "timing/previous_window_wall_seconds": completed_wall,
                               "timing/previous_window_logging_seconds": finished - logging_started}
            output.append(metrics)
            metric_window.clear()
            phase_timers.clear()
            window_started = None
        return output

    def _remember_safe_policy(self):
        self._safe_policy = {name: _cpu_tree(getattr(self, name).state_dict()) for name in ("actor", "critic")}

    def close(self):
        """关闭事件文件写入器；环境由调用方负责释放。"""
        self.logger.close()

    def _checkpoint_state(self):
        """统一装配算法状态，任务子类在写盘前追加自己的元数据。"""
        return {"format": self.spec.model_id, "spec": asdict(self.spec),
                "config": asdict(self.config), "execution": asdict(self.execution), "control_enabled": self.control_enabled,
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "world_model": self.model.state_dict(),
                "learner": self.learner.state_dict(), "iteration": self.iteration, "real_steps": self.real_steps,
                "imagined_steps": self.imagined_steps}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), path)
        if self._collapse_strikes == 0:
            self._remember_safe_policy()

    def load(self, path, *, continue_run=False, reset_critic=False):
        """按参考热启动网络和同形状优化器；继续训练仅追加恢复累计日志计数。

        预热、回滚缩放、随机数及真实回放不从检查点恢复；策略形状不兼容时明确拒绝。
        网络结构变化时保留当前任务定义，重建相应网络并应用参考重预热排期。
        """
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data["format"] != self.spec.model_id:
            raise ValueError("SuperMimic checkpoint format or task specification mismatch")
        self.actor.load_state_dict(data["actor"], strict=True)
        same = {"actor": True}
        for name, checkpoint_key in (("critic", "critic"), ("model", "world_model")):
            state = data[checkpoint_key]
            current = getattr(self, name).state_dict()
            same[name] = set(current) == set(state) and all(current[key].shape == state[key].shape for key in current)
        saved_learner = data["learner"]
        counters = {name: data[name] if continue_run else 0 for name in ("iteration", "real_steps", "imagined_steps")}
        if same["model"]:
            self.model.load_state_dict(data["world_model"], strict=True)
        same["critic"] = same["critic"] and not reset_critic
        if same["critic"]:
            self.critic.load_state_dict(data["critic"], strict=True)
        self.learner = self._make_learner()
        for name in ("actor", "model", "critic"):
            if same[name]:
                self.learner.load_optimizer_state_dict(name, saved_learner[name])
            # 该累计计数仅用于本项目日志，不影响来源的学习率和预热排期。
            if continue_run and same[name]:
                setattr(self.learner, name + "_updates", saved_learner[name + "_updates"])
        if not same["model"]:
            self.learner.policy_start = max(self.learner.policy_start, self.config.wm_rewarmup_iterations)
            self.learner.actor_frozen_until = self.learner.policy_start + self.config.critic_warmup_iterations
            print(f"World model re-initialised (shape changed); policy updates from iteration "
                  f"{self.learner.policy_start}", flush=True)
        if not same["critic"]:
            self.learner.actor_frozen_until = max(self.learner.actor_frozen_until,
                                                 self.learner.policy_start + self.config.critic_warmup_iterations)
            print(f"Critic re-initialised; actor frozen until iteration {self.learner.actor_frozen_until}", flush=True)
        for name, value in counters.items():
            setattr(self, name, value)
        self.current = self.history = None
        # 与指定参考一致，恢复进程重新开始采集相位，物理与回放同时重置。
        self._collect_step_accumulator = 0.0
        self.replay.clear()
        self.archive.clear()
        self.elapsed = 0.0
        self._configure_execution()
        self.prefill_real_steps = 0
        self.session_iterations = 0
        self._collapse_strikes = 0
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        self._safe_policy = None
        print(f"Resumed from {path} (iteration {data['iteration']})", flush=True)
        if continue_run:
            print(f"Continuing at iteration {self.iteration}", flush=True)
        return data
