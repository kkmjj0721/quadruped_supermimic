# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2高程教师播放，维护本体历史、课程快照和查看器的完整／部分重置。
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import torch
from mjlab.managers.command_manager import NullCommandManager
from mjlab.utils.wrappers import VideoRecorder

from agents.supermimic.interfaces import append_history
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.train_cfg import TASK_MODES
from .checkpoints import build_playback_policy, load_playback_source
from .transition_adapter import Go2TransitionAdapter


class _PlaybackTransitionAdapter(Go2TransitionAdapter):
    """只在播放时重采样出生地形，沿用任务原有的原点与状态同步。"""

    def _set_origins(self, ids):
        if not self.map.plane:
            self.terrain_curriculum.family_ids[ids] = torch.randint(self.map.cols, (ids.numel(),), device=self.device)
            self.terrain_curriculum.levels[ids] = torch.randint(self.map.rows, (ids.numel(),), device=self.device)
        super()._set_origins(ids)


class _PlaybackHeightmapVisualizer:
    """把教师实际使用的高程采样网格显示为地面上的青色点。"""

    def __init__(self, adapter):
        self.adapter = adapter

    @torch.no_grad()
    def debug_vis(self, visualizer):
        task = self.adapter.task
        state = self.adapter.get_state()
        positions, yaws = task.world_position(state), task.yaw(state)
        x, y = task.scan_points.unbind(-1)
        for env_idx in visualizer.get_env_indices(self.adapter.num_envs):
            yaw = yaws[env_idx]
            xy = torch.stack((yaw.cos() * x - yaw.sin() * y,
                              yaw.sin() * x + yaw.cos() * y), -1) + positions[env_idx, :2]
            points = torch.cat((xy, task.surface_height(xy)[:, None]), -1).cpu().numpy()
            for point in points:
                visualizer.add_sphere(point, radius=0.5 * visualizer.meansize, color=(0., 1., 1., 1.))


class _PlaybackRewardManager:
    """向查看器提供所选环境的加权奖励率，不乘控制周期。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self.rates = {}

    def reset(self, state, env_ids):
        if not self.rates:
            actions = torch.zeros((self.adapter.num_envs, self.adapter.task.spec.action_dim), device=self.adapter.device)
            terms = self.adapter.task.reward_terms(state, actions, state, actions[:, 0])
            self.rates = {name: torch.zeros_like(value) for name, value in terms.items()}
        else:
            ids = slice(None) if env_ids is None else env_ids
            for values in self.rates.values():
                values[ids] = 0

    def update(self, state, actions, result):
        # 用真实转移的终态重算同一组纯张量评分，避免读取自动重置后的状态。
        terms = self.adapter.task.reward_terms(state, actions, result.final_state, result.terminated.float())
        self.rates = {name: getattr(self.adapter.cfg.reward.scales, name) * value for name, value in terms.items()}

    def get_active_iterable_terms(self, env_idx):
        return [(name, [values[env_idx].item()]) for name, values in self.rates.items()]

    def get_visualizable_terms(self):
        return []


class _PlaybackCommandManager(NullCommandManager):
    """在播放侧提供Viser速度面板，通过已有覆盖接口控制全部播放环境。"""

    def __init__(self, adapter):
        super().__init__()
        self.active_terms = ["twist"]
        self.adapter = adapter
        self._enabled = None
        self._sliders = []
        self._command = None

    def create_gui(self, server, get_env_idx, on_change=None, request_action=None):
        """沿用参考面板的三轴速度、可调范围与归零按钮，回调只修改界面状态。"""
        with server.gui.add_folder("速度控制"):
            self._enabled = server.gui.add_checkbox("启用手动速度（全部环境）", initial_value=False)
            axes = ("前进速度 vx (m/s)", "侧移速度 vy (m/s)", "转向速度 wz (rad/s)")
            for label, bounds in zip(axes, self.adapter.cfg.commands.initial_ranges, strict=True):
                limit = max(abs(bounds[0]), abs(bounds[1]))
                maximum = server.gui.add_slider(f"调节范围：{label}", min=0.1, max=10.0,
                                               step=0.1, initial_value=limit)
                slider = server.gui.add_slider(label, min=-limit, max=limit, step=0.05, initial_value=0.0)

                @maximum.on_update
                def _(_event, target=slider, source=maximum):
                    target.min, target.max = -source.value, source.value

                self._sliders.append(slider)
            zero = server.gui.add_button("速度归零")

            @zero.on_click
            def _(_event):
                for slider in self._sliders:
                    slider.value = 0.0

    def update(self):
        """在仿真线程应用界面输入；关闭手动控制后恢复任务原有命令逻辑。"""
        command = (tuple(slider.value for slider in self._sliders)
                   if self._enabled is not None and self._enabled.value else None)
        if command == self._command:
            return False
        self.adapter.set_play_commands(command)
        self._command = command
        return True

    def get_active_iterable_terms(self, env_idx):
        return [("twist", self.adapter.get_state().command[env_idx].cpu().tolist())]


class SuperMimicVideoRecorder(VideoRecorder):
    """显式转发教师策略观测，避免继承的原生manager读取绕过历史适配。"""

    def get_observations(self):
        return self._wrapped_env.get_observations()


class SuperMimicPlayEnv:
    """将任务转移转换为查看器协议；原生属性委托，重置始终经过任务adapter。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self.command_manager = _PlaybackCommandManager(adapter)
        self.reward_manager = _PlaybackRewardManager(adapter)
        adapter.raw_env.manager_visualizers["heightmap"] = _PlaybackHeightmapVisualizer(adapter)
        self._observation = None
        self._history = None

    def __getattr__(self, name):
        return getattr(self.adapter.raw_env, name)

    @property
    def unwrapped(self):
        # Viser的局部重置直接调用unwrapped.reset，不能暴露会绕过任务状态的raw_env。
        return self

    @torch.no_grad()
    def reset(self, *, env_ids=None):
        self._update_commands()
        state = self.adapter.reset(env_ids=env_ids)
        self.reward_manager.reset(state, env_ids)
        self.cfg.viewer.reward_bar_max_terms = len(self.reward_manager.rates)
        frame = self.adapter.task.observe(state, noisy=False)
        length = self.adapter.task.spec.history_length
        if env_ids is None or self._history is None:
            self._observation = frame
            self._history = frame.unsqueeze(1).repeat(1, length, 1)
        else:
            self._observation[env_ids] = frame[env_ids]
            self._history[env_ids] = frame[env_ids].unsqueeze(1).expand(-1, length, -1)
        return self.get_observations()

    def _update_commands(self):
        """同步当前命令与最新观测，局部重置前也更新未重置环境的决策帧。"""
        if self.command_manager.update() and self._history is not None:
            # 更新当前决策帧中的命令，保留旧历史帧，不额外推进历史时间。
            self._observation = self.adapter.task.observe(self.adapter.get_state(), noisy=False)
            self._history[:, -1] = self._observation

    def get_observations(self):
        if self._history is None:
            raise RuntimeError("Reset the SuperMimic playback environment before observing")
        self._update_commands()
        state = self.adapter.get_state()
        return {"observation": self._observation, "history": self._history,
                "heightmap": self.adapter.task.actor_heightmap(state),
                "exteroceptive_mask": self.adapter.task.actor_mask(state)}

    @torch.no_grad()
    def step(self, actions):
        self._update_commands()
        before = self.adapter.get_state()
        result = self.adapter.step(actions)
        self.reward_manager.update(before, actions, result)
        frame = self.adapter.task.observe(result.next_state, noisy=False)
        history = append_history(self._history, frame)
        done = result.terminated | result.truncated
        if done.any():
            history[done] = frame[done].unsqueeze(1).expand(-1, history.shape[1], -1)
        self._observation, self._history = frame, history
        return self.get_observations(), result.reward, result.terminated, result.truncated, {"final_state": result.final_state}

    def close(self):
        self.adapter.close()


def _restore_curriculum(adapter, checkpoint):
    """按保存的环境行循环映射地形，保留训练时命令阶段并冻结后续课程。"""
    saved = checkpoint["go2_task"]["curriculum"]
    terrain = saved["terrain"]
    families, levels = terrain["family_ids"], terrain["levels"]
    rows = torch.arange(adapter.num_envs, device=families.device) % families.numel()
    adapter.load_curriculum_state_dict({
        "command": saved["command"],
        "terrain": {**terrain, "family_ids": families[rows], "levels": levels[rows]},
    })


def make_playback(task_id, *, checkpoint_file=None, agent="trained", num_envs=None, device="cpu",
                  render_mode=None, video=False, video_length=200, video_height=None, video_width=None,
                  no_terminations=False):
    """装配并重置播放环境与策略，按需包装录像，不创建训练组件。"""
    trained = agent == "trained"
    sampler = None if trained else {"zero": torch.zeros, "random": torch.rand}[agent]
    checkpoint = Path(checkpoint_file).expanduser().resolve() if trained else None
    record_video = video and trained
    if video and not trained:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
    source = load_playback_source(task_id, checkpoint) if trained else None
    cfg = deepcopy(source.cfg) if source is not None else Go2ReferenceCfg()
    cfg.terrain.mode = TASK_MODES[task_id]
    cfg.env.num_envs = num_envs if num_envs is not None else 1
    torch.manual_seed(cfg.env.seed)
    adapter = _PlaybackTransitionAdapter(
        cfg, device=device, play=True, render_mode="rgb_array" if record_video else render_mode, video_height=video_height,
        video_width=video_width, no_terminations=no_terminations,
    )
    with ExitStack() as cleanup:
        cleanup.callback(adapter.close)
        model = build_playback_policy(source, adapter) if trained else None
        if trained:
            _restore_curriculum(adapter, source.checkpoint)
        env = SuperMimicPlayEnv(adapter)
        env.reset()
        if record_video:
            env = SuperMimicVideoRecorder(env, video_folder=checkpoint.parent / "videos" / "play",
                                          step_trigger=lambda step: step == 0,
                                          video_length=video_length, disable_logger=True)
            print("[INFO] Recording videos during play.")
        cleanup.pop_all()

    @torch.no_grad()
    def policy(observations):
        if trained:
            return model(observations["history"], observations["heightmap"], observations["exteroceptive_mask"])
        shape = (adapter.num_envs, adapter.task.spec.action_dim)
        values = sampler(shape, device=adapter.device)
        return values if agent == "zero" else 2 * values - 1

    return env, policy
