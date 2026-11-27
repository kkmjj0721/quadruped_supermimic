# Copyright (c) 2026-2027 zh
"""
内容：
    从真实Go2环境提取reset前转移，统一任务记忆、奖励、课程与部分reset时序。
"""

from __future__ import annotations

from types import MethodType

import torch
from mjlab.envs import ManagerBasedRlEnv

from agents.supermimic.compilation import compile_kernel
from agents.supermimic.interfaces import EnvStep

from .constant_graph import configure_reset_constant_graph
from .mjlab_step import step_manual_reset
from ..config.env_cfgs import resolve_go2_supermimic_cfg
from ..go_supermimic_env_cfg import make_go2_supermimic_env_cfg
from ..mdp.curriculums import CommandCurriculum, TerrainCurriculum
from ..mdp.parkour import PARKOUR_FAMILIES, SPECIAL_FAMILIES
from ..mdp.state_spec import (
    CONTEXT_ENCODER_BIAS, CONTEXT_FRICTION, CONTEXT_TERRAIN_FAMILY,
    CONTEXT_EXTEROCEPTIVE_MASK, CONTEXT_ORIGIN, CONTEXT_RANGES, CONTEXT_ZERO_PROBABILITY,
    COMMAND_OVERRIDE,
)
from ..mdp.reference_task import Go2ReferenceTaskModel
from ..mdp.reference_state import ANGULAR_BODY, CONTEXT_COM
from ..mdp.terrains import PlaneSurface, TerrainMap, TerrainSurface


class Go2TransitionAdapter:
    """连接原生仿真与CollectionEnv，任务规则由本目录的MDP实现。

    单步顺序为物理推进、复制终态、计算任务奖励／失败、更新真实课程、重置结束行。
    final_state保留重置前状态，next_state提供下一次决策输入。
    """

    # ---------- 初始化 ----------

    def __init__(self, cfg=None, *, device="cpu", play=False, render_mode=None, terrain_mode=None, num_envs=None,
                 video_height=None, video_width=None, no_terminations=False):
        self.cfg = resolve_go2_supermimic_cfg(cfg, play=play, terrain_mode=terrain_mode)
        if num_envs is not None:
            self.cfg.env.num_envs = num_envs
        if play:
            torch.manual_seed(self.cfg.env.seed)
        self.play = play
        self.no_terminations = no_terminations
        self.map = TerrainMap(self.cfg.terrain)
        env_cfg = make_go2_supermimic_env_cfg(self.cfg, terrain_map=self.map)
        if play:
            del env_cfg.terminations["time_out"]
        if no_terminations:
            env_cfg.terminations = {}
        if video_height is not None:
            env_cfg.viewer.height = video_height
        if video_width is not None:
            env_cfg.viewer.width = video_width
        self.raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        configure_reset_constant_graph(self.raw_env)
        if not self.play:
            self.raw_env.step = MethodType(step_manual_reset, self.raw_env)
        self.num_envs = self.raw_env.num_envs
        self.robot = self.raw_env.scene["robot"]
        self.device = self.robot.data.joint_pos.device
        terrain = self.raw_env.scene.terrain
        self.map.bind(terrain.terrain_origins)
        self.surface = (PlaneSurface(device=self.device) if self.cfg.terrain.mode == "flat" else
                        TerrainSurface.from_model(self.raw_env.sim.mj_model, self.device, terrain_map=self.map))
        self.task = Go2ReferenceTaskModel(self.cfg, self.robot.data.default_joint_pos[0].clone(),
                              self.robot.data.soft_joint_pos_limits[0].clone(), self.surface, device=self.device)
        self.task.bind_env(self.raw_env)
        self.foot_geom_id = self.raw_env.sim.mj_model.geom("robot/FL_foot_collision").id
        self.base_body_id = self.raw_env.sim.mj_model.body("robot/base").id
        self.nominal_body_ipos = torch.as_tensor(
            self.raw_env.sim.mj_model.body_ipos[self.base_body_id].copy(), device=self.device)
        self.command_curriculum = CommandCurriculum(self.cfg.commands, self.map.family_names, self.device)
        self.terrain_curriculum = TerrainCurriculum(
            self.map, self.num_envs, self.cfg.terrain.max_init_level, enabled=self.cfg.terrain.curriculum,
            episode_length_s=self.cfg.env.episode_length_s, device=self.device)
        if not self.map.plane:
            self.terrain_curriculum.family_ids.copy_(terrain.terrain_types)
        if self.cfg.terrain.mode == "parkour":
            self.column_family_ids = torch.tensor(
                [PARKOUR_FAMILIES.index(name) for name in self.map.family_names], device=self.device)
            eligible = [name not in SPECIAL_FAMILIES for name in self.map.family_names]
            env_families = [self.map.family_names[index] for index in self.terrain_curriculum.family_ids.cpu().tolist()]
            self.terrain_family_rows = {
                name: torch.tensor([index for index, family in enumerate(env_families) if family == name], device=self.device)
                for name in dict.fromkeys(env_families)
            }
        else:
            eligible = [name != "flat" for name in self.map.family_names]
            self.terrain_family_rows = {}
        self.mask_eligible = torch.tensor(eligible, device=self.device)
        self.terrain_half_size = torch.tensor(
            (self.map.rows * self.map.size[0] / 2 + 4., self.map.cols * self.map.size[1] / 2 + 4.), device=self.device)
        self.context_version = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._state = None
        self._play_commands = None
        self.last_metrics = {}
        self._episode_reward_sums = {name: torch.zeros(self.num_envs, device=self.device)
                                     for name in self.cfg.to_dict()["reward"]["scales"]}
        self._reward_names = tuple(self.task.reward_scales)
        compiled = not self.play and self.device.type == "cuda" and self.cfg.runner.cuda_graph
        self.task.configure_collection_compile(enabled=compiled, backend=self.cfg.algo.compile_backend)
        self._transition_kernel = (compile_kernel(self._transition_tensors, self.cfg.algo.compile_backend)
                                   if compiled else self._transition_tensors)

    @property
    def step_dt(self):
        return self.raw_env.step_dt

    def get_state(self):
        """提供当前决策态的独立副本，避免评估修改采集状态。"""
        return self._state.detached_clone()

    @torch.no_grad()
    def set_play_commands(self, commands=None):
        """播放时固定各环境的速度命令，None恢复原命令采样；不改变训练配置。"""
        if commands is None:
            self._play_commands = None
            if self._state is not None:
                self._state.memory[:, COMMAND_OVERRIDE] = 0
            return
        values = torch.as_tensor(commands, dtype=torch.float32, device=self.device)
        if values.shape == (3,):
            values = values.unsqueeze(0).expand(self.num_envs, -1)
        self._play_commands = values.clone()
        if self._state is not None:
            self._state.command.copy_(self._play_commands)
            self._state.memory[:, COMMAND_OVERRIDE] = 1

    # ---------- 回合重置 ----------

    def _set_origins(self, ids):
        if self.cfg.terrain.mode == "flat":
            return
        terrain = self.raw_env.scene.terrain
        self.terrain_curriculum.sample(ids)
        terrain.env_origins[ids] = self.terrain_curriculum.origins(ids)
        # 同步原生可视化元数据，课程计算仍只有当前adapter这一处。
        terrain.terrain_levels[ids] = self.terrain_curriculum.levels[ids] + self.map.row_offset
        terrain.terrain_types[ids] = self.terrain_curriculum.family_ids[ids] + self.map.rings

    def _context(self, ids):
        context = torch.zeros((ids.numel(), self.task.spec.context_dim), dtype=torch.float32, device=self.device)
        foot = self.foot_geom_id
        friction = self.raw_env.sim.model.geom_friction
        context[:, CONTEXT_FRICTION] = (friction[ids, foot, 0] if friction.ndim == 3 else friction[foot, 0])
        base = self.base_body_id
        ipos = self.raw_env.sim.model.body_ipos
        nominal = self.nominal_body_ipos
        context[:, CONTEXT_COM] = (ipos[ids, base] if ipos.ndim == 3 else ipos[base]) - nominal
        context[:, CONTEXT_ENCODER_BIAS] = self.robot.data.encoder_bias[ids]
        context[:, CONTEXT_ORIGIN] = self.raw_env.scene.env_origins[ids]
        bounds, probability = self.command_curriculum.parameters(self.terrain_curriculum.family_ids[ids])
        context[:, CONTEXT_RANGES] = bounds
        context[:, CONTEXT_ZERO_PROBABILITY] = probability.squeeze(-1)
        eligible = self.mask_eligible[self.terrain_curriculum.family_ids[ids]]
        enabled = torch.rand(ids.numel(), device=self.device) >= self.cfg.teacher.exteroceptive_dropout
        context[:, CONTEXT_EXTEROCEPTIVE_MASK] = (enabled | ~eligible).float()
        if self.cfg.terrain.mode == "parkour":
            context[:, CONTEXT_TERRAIN_FAMILY] = self.column_family_ids[self.terrain_curriculum.family_ids[ids]]
        return context

    def _reset_rows(self, ids):
        self._set_origins(ids)
        self.raw_env.reset(env_ids=ids)
        physical = self.task.read_physical(self.raw_env)
        context = self._context(ids)
        reset_state = self.task.initialize_state(physical[ids], context)
        if self._play_commands is not None:
            reset_state.command.copy_(self._play_commands[ids])
            reset_state.memory[:, COMMAND_OVERRIDE] = 1
        if self._state is None:
            self._state = reset_state.detached_clone()
        else:
            self._state.data[ids] = reset_state.data
        self.terrain_curriculum.reset(ids, self.robot.data.root_link_pos_w[ids, :2])
        self.context_version[ids] += 1
        for values in self._episode_reward_sums.values():
            values[ids] = 0

    @torch.no_grad()
    def reset(self, env_ids=None):
        """外部reset不计晋级，保留已加载的课程阶段与逻辑等级。"""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._reset_rows(ids)
        self.last_metrics = {}
        return self._state.detached_clone()

    # ---------- 真实转移 ----------

    def _transition_tensors(self, before, actions, physical, contact_times, raw_truncated,
                            play_commands, episode_sums):
        """融合完整转移评分和分项累计，不读取仿真或课程的Python步号。"""
        final = self.task.advance(before, actions, physical, contact_times=contact_times)
        if play_commands is not None:
            final.command.copy_(play_commands)
        terminated = (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) if self.no_terminations
                      else self.task.hard_termination(final))
        truncated = raw_truncated.clone()
        if not self.map.plane and not self.no_terminations:
            # 沿用核心地形外4米的超时定义，不增加失败惩罚。
            truncated |= (self.task.world_position(final)[:, :2].abs() > self.terrain_half_size).any(-1)
        terms = self.task.reward_terms(before, actions, final, terminated.float())
        dt = self.task.dt
        reward = dt * sum(self.task.reward_scales[name] * value for name, value in terms.items())
        for name, total in zip(self._reward_names, episode_sums, strict=True):
            total.add_(dt * self.task.reward_scales[name] * terms[name])
        velocity = self.task.tracking_velocity(final)
        linear_error = velocity[:, :2] - before.command[:, :2]
        angular_error = velocity[:, 2] - before.command[:, 2]
        metrics = {f"reward/{name}": value.float().mean() for name, value in terms.items()}
        metrics.update(tracking_linear_mse=linear_error.square().sum(-1).mean(),
                       tracking_angular_mse=angular_error.square().mean(),
                       action_rms=actions.square().mean().sqrt(),
                       exteroceptive_fraction=self.task.actor_mask(before).mean(),
                       base_height=self.task.physical_features(final)[:, 0].mean(),
                       fast_yaw=(final.physical[:, ANGULAR_BODY][:, 2].abs() > 3.0).float().mean())
        return final, reward, terminated, truncated, metrics

    @torch.no_grad()
    def step(self, actions):
        before = self._state
        versions = self.context_version.clone()
        _, _, _, raw_truncated, _ = self.raw_env.step(actions)
        physical = self.task.read_physical(self.raw_env)
        contact_sensor = self.raw_env.scene["feet_ground_contact"]
        sensor = contact_sensor.data
        contact_times = (sensor.current_air_time, sensor.current_contact_time, sensor.last_air_time,
                         contact_sensor.compute_first_contact(self.step_dt).float(), sensor.last_contact_time)
        episode_sums = tuple(self._episode_reward_sums[name] for name in self._reward_names)
        final, reward, terminated, truncated, metrics = self._transition_kernel(
            before, actions, physical, contact_times, raw_truncated, self._play_commands, episode_sums)
        if not self.play:
            self.command_curriculum.control_steps += 1
            self.terrain_curriculum.control_steps = self.raw_env.common_step_counter
        # reset只写新内部副本，final和调用方持有的旧next_state始终独立。
        self._state = final.detached_clone()
        done_ids = (terminated | truncated).nonzero().flatten()
        self.terrain_curriculum.update(self.raw_env, done_ids, before.command)
        self.terrain_curriculum.advance(before.command, self.robot.data.heading_w, self.step_dt)
        episode_metrics = {}
        if done_ids.numel():
            episode_metrics = {f"episode_reward/{name}": self._episode_reward_sums[name][done_ids].mean()
                               / self.cfg.env.episode_length_s for name in self._reward_names}
            self._reset_rows(done_ids)
        self.last_metrics = metrics
        self.last_metrics.update(episode_metrics)
        self.last_metrics.update(terrain_level=self.terrain_curriculum.levels.float().mean(),
                                 command_stage_iteration=float(self.command_curriculum.iteration))
        self.last_metrics.update({f"terrain_level/{name}": self.terrain_curriculum.levels[rows].float().mean()
                                  for name, rows in self.terrain_family_rows.items()})
        return EnvStep(final, self._state, reward, terminated, truncated, versions)

    # ---------- 课程持久化与资源释放 ----------

    def curriculum_state_dict(self):
        return {"command": self.command_curriculum.state_dict(), "terrain": self.terrain_curriculum.state_dict()}

    def load_curriculum_state_dict(self, state):
        self.command_curriculum.load_state_dict(state["command"])
        self.terrain_curriculum.load_state_dict(state["terrain"])
        self.raw_env.common_step_counter = self.command_curriculum.control_steps

    def resume_progress(self, iteration):
        """参考continue_run仅恢复公共进度，不伪装成恢复各环境物理状态。"""
        self.raw_env.common_step_counter = iteration
        self.command_curriculum.control_steps = iteration
        self.terrain_curriculum.control_steps = iteration

    def close(self):
        self.raw_env.close()


def make_go2_supermimic_env(cfg=None, *, device="cpu", play=False, render_mode=None,
                           video_height=None, video_width=None, no_terminations=False):
    """创建可直接交给SuperMimicRunner的真实Go2采集环境。"""
    return Go2TransitionAdapter(cfg, device=device, play=play, render_mode=render_mode,
                                video_height=video_height, video_width=video_width, no_terminations=no_terminations)
