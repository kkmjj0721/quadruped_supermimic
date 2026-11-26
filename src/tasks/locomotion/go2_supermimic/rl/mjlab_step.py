# Copyright (c) 2026-2027 zh
"""
内容：
    基于MjLab1.6.0原生step，手动reset路径只保留设备pending掩码更新。
    仅由Go2 adapter调用，其每步结尾处理全部raw_truncated；独立raw_env漏reset诊断不在此路径提供。
    物理积分、forward、事件、观测与自动reset顺序均沿用上游。
来源：
    mujocolab/mjlab v1.6.0，mjlab/envs/manager_based_rl_env.py::ManagerBasedRlEnv.step。
    上游Apache-2.0许可见同目录LICENSE_mjlab；安装包本身未修改。
"""
from __future__ import annotations

import torch
from mjlab.envs import types


def step_manual_reset(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """仅供Go2 adapter已保证本步reset的内部路径；保留上游物理与manager时序。"""
    self.extras['log'] = dict()
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=self.physics_dt)
        self.metrics_manager.compute_substep()
    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()
    if 'step' in self.event_manager.available_modes:
        self.event_manager.apply(mode='step', dt=self.step_dt)
    if 'interval' in self.event_manager.available_modes:
        self.event_manager.apply(mode='interval', dt=self.step_dt)
    if self.cfg.auto_reset:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
        self.recorder_manager.record_pre_reset(reset_env_ids)
        self._reset_idx(reset_env_ids)
        self.scene.write_data_to_sim()
    self.sim.forward()
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
        command_dt = self._command_dt
        command_dt.fill_(self.step_dt)
        command_dt[reset_env_ids] = 0.0
        self.command_manager.compute(dt=command_dt)
    else:
        self.command_manager.compute(dt=self.step_dt)
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
        self.recorder_manager.record_post_reset(reset_env_ids)
    elif not self.cfg.auto_reset:
        self._manual_reset_pending.logical_or_(self.reset_buf)
    self.recorder_manager.record_post_step()
    return (self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras)
