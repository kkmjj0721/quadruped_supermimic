# Copyright (c) 2026-2027 zh
"""内容：将PROOF官方教师的高程、本体观测和奖励接入Go2可微世界模型。"""

from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import Tensor

from agents.supermimic.config import RUNNER_RUNTIME_FIELDS
from agents.supermimic.interfaces import AlgorithmSpec, TaskState

from .reference_rewards import reference_reward_terms
from .commands import advance_commands, heading_command, sample_commands
from .reference_kinematics import Go2Kinematics
from .reference_rotations import mat_to_quat, mv3, quat_to_mat
from .structured_dynamics import ResidualDynamics
from .reference_state import (
    ANGULAR_BODY, AUXILIARY, BASE_CONTACT_FORCE, CONTACT_FOUND, CONTEXT_COM, FEET,
    FOOT_CONTACT_FORCE, GENERALIZED_DELTA_DIM, GRAVITY, HEIGHT, JOINT_ACCELERATION, JOINT_OFFSET,
    JOINT_TORQUE, JOINT_VELOCITY, MODEL_CONTACT, MODEL_HISTORY_CONTACT, PHYSICAL_DIM, POSE,
    ROBOT_STATE, VELOCITY_HEADING, body_planar_velocity, body_vertical_velocity, yaw_rotate_inverse,
)
from .state_spec import (
    AIR_TIME, COMMAND_OVERRIDE, COMMAND_TIMER, CONTACT, CONTACT_TIME, CONTEXT_DIM,
    CONTEXT_ENCODER_BIAS, CONTEXT_EXTEROCEPTIVE_MASK, ELAPSED_TIME, FIRST_CONTACT,
    FOOT_NAMES, FORWARD, HEADING_TARGET, LAST_AIR_TIME, LAST_CONTACT_TIME,
    MEMORY_DIM, PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION, STANDING,
)


class Go2ReferenceTaskModel:
    """真实采集与想象共用教师观测、残差状态、奖励和回合记忆。"""

    observation_term_dims = (3, 3, 3, 12, 12, 12)

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.surface = surface
        self.dt = cfg.sim.timestep * cfg.control.decimation
        self.physical_dim = PHYSICAL_DIM
        self.auxiliary = AUXILIARY
        self.root_quat = slice(PHYSICAL_DIM, PHYSICAL_DIM + 4)
        self.default_joint_pos = default_joint_pos.to(device=self.device, dtype=torch.float32).clone()
        self.joint_limits = joint_limits.to(device=self.device, dtype=torch.float32).clone()
        self.reward_scales = cfg.to_dict()["reward"]["scales"]
        self.dynamics = ResidualDynamics(self.dt, cfg.control.decimation)
        self.kinematics = Go2Kinematics(device=self.device)
        # PROOF ae910b8：18×9，yaw对齐，中心前移0.6米。
        size_x, size_y = cfg.terrain.scan_size_m
        resolution = cfg.terrain.scan_resolution_m
        xs = torch.arange(round(size_x / resolution) + 1, device=self.device) * resolution - size_x / 2
        ys = torch.arange(round(size_y / resolution) + 1, device=self.device) * resolution - size_y / 2
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        self.scan_points[:, 0] += cfg.terrain.scan_forward_offset_m
        foot_axis = torch.tensor((-.015, .015), device=self.device)
        self.foot_scan_points = torch.stack(torch.meshgrid(foot_axis, foot_axis, indexing="ij"), -1).reshape(-1, 2)
        self.heightmap_dim = self.scan_points.shape[0]
        self.model_input_dim = PHYSICAL_DIM - 3 + 12 + 12 + 16 + self.heightmap_dim
        noise = cfg.noise.noise_scales
        self.noise_amplitude = self.default_joint_pos.new_tensor(
            (noise.base_ang_vel,) * 3 + (noise.projected_gravity,) * 3 + (0.,) * 3
            + (noise.joint_pos,) * 12 + (noise.joint_vel,) * 12 + (0.,) * 12) * cfg.noise.noise_level
        payload = {"task": self._semantic_config(), "geometry": surface.geometry_hash,
                   "default_joint_pos": self.default_joint_pos.cpu().tolist(),
                   "joint_limits": self.joint_limits.cpu().tolist(),
                   "layout": "proof_teacher_physical142_quat4_memory54_context27_critic285_v2",
                   "contact_force_frame": "world_force_on_robot",
                   "dynamics": "blinkmimic_residual_fk_quaternion_v1"}
        self.semantic_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.spec = AlgorithmSpec(
            state_dim=PHYSICAL_DIM + 4, delta_dim=PHYSICAL_DIM - 45, action_dim=12,
            observation_dim=45, privileged_dim=cfg.env.num_critic_obs, context_dim=CONTEXT_DIM,
            memory_dim=MEMORY_DIM, command_dim=3, heightmap_dim=self.heightmap_dim,
            history_length=cfg.env.num_obs_history,
            semantic_id="go2_proof_teacher_ae910b8_reward_memory_residual_fk_v2:" + self.semantic_hash,
            model_id="supermimic_proof_teacher_v1")

    def _semantic_config(self) -> dict:
        """运行预算不改变任务语义；模型与观测结构由快照完整描述。"""
        config = self.cfg.to_dict()
        config["runner"] = {name: value for name, value in config["runner"].items() if name not in RUNNER_RUNTIME_FIELDS}
        for name in ("actor_optimizer", "wm_optimizer", "muon_weight_decay"):
            config["algo"].pop(name)
        return config

    # ---------- 真实物理状态 ----------

    def surface_height(self, xy: Tensor) -> Tensor:
        return self.surface.height_unchecked(xy)

    def bind_env(self, env):
        robot = env.scene["robot"]
        hip_ids, _ = robot.find_bodies(tuple(f"{leg}_thigh" for leg in FOOT_NAMES), preserve_order=True)
        knee_ids, _ = robot.find_bodies(tuple(f"{leg}_calf" for leg in FOOT_NAMES), preserve_order=True)
        foot_ids, _ = robot.find_sites(FOOT_NAMES, preserve_order=True)
        self.hip_ids = torch.tensor(hip_ids, dtype=torch.long, device=self.device)
        self.knee_ids = torch.tensor(knee_ids, dtype=torch.long, device=self.device)
        self.foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=self.device)
        # 每条calf在当前MJCF有两个独立碰撞体；足端几何不参与calf接触计数。
        names = env.scene["calf_ground_contact"].primary_names
        self.calf_geom_ids = torch.tensor(
            [[names.index(f"{leg}_calf_collision{index}") for index in (1, 2)] for leg in FOOT_NAMES],
            device=self.device)
        self.actuator_ids = robot.indexing.ctrl_ids
        model = env.sim.mj_model
        ids = self.actuator_ids.cpu().numpy()
        self.actuator_gain = torch.tensor(model.actuator_gainprm[ids, 0].copy(), dtype=torch.float32, device=self.device)
        self.actuator_bias = torch.tensor(model.actuator_biasprm[ids, :3].copy(), dtype=torch.float32, device=self.device)

    def read_physical(self, env) -> Tensor:
        data = env.scene["robot"].data
        sensor = env.scene["feet_ground_contact"].data
        position, yaw = data.root_link_pos_w, data.heading_w
        limbs = (data.body_link_pos_w[:, self.hip_ids], data.body_link_pos_w[:, self.knee_ids],
                 data.site_pos_w[:, self.foot_ids])
        relative = [yaw_rotate_inverse(points - position[:, None], yaw[:, None]).flatten(1) for points in limbs]
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        calf_history = env.scene["calf_ground_contact"].data.force_history
        calf_net = calf_history[:, self.calf_geom_ids].sum(dim=2)
        calf_force = calf_net.norm(dim=-1).amax(dim=2)
        base_force = env.scene["base_ground_contact"].data.force_history.norm(dim=-1).amax(dim=(1, 2))[:, None]
        sim = env.sim.data
        computed_torque = (self.actuator_gain * sim.ctrl[:, self.actuator_ids]
                           + self.actuator_bias[:, 0] + self.actuator_bias[:, 1] * sim.actuator_length[:, self.actuator_ids]
                           + self.actuator_bias[:, 2] * sim.actuator_velocity[:, self.actuator_ids])
        history_contact = (sensor.force_history.norm(dim=-1).amax(dim=2) > self.cfg.reward.contact_force_threshold).float()
        # 当前primary足端／secondary地面配置返回对地反作用力，取反后才是PROOF的机器人受力。
        return torch.cat((height, data.projected_gravity_b, yaw_rotate_inverse(data.root_link_lin_vel_w, yaw),
                          data.root_link_ang_vel_b, data.joint_pos - data.default_joint_pos, data.joint_vel, *relative,
                          position[:, :2], yaw[:, None], data.joint_acc,
                          data.site_lin_vel_w[:, self.foot_ids, :2].flatten(1), (sensor.found > 0).float(),
                          -sensor.force.flatten(1), data.actuator_force, calf_force, base_force, computed_torque,
                          history_contact, data.root_link_quat_w), -1)

    def world_position(self, state: TaskState) -> Tensor:
        xy = state.physical[:, POSE][:, :2]
        return torch.cat((xy, self.base_height(state)[:, None] + self.surface_height(xy)[:, None]), -1)

    def base_height(self, state: TaskState) -> Tensor:
        return state.physical[:, 0]

    def yaw(self, state: TaskState) -> Tensor:
        return state.physical[:, 72]

    def body_linear_velocity(self, state: TaskState) -> Tensor:
        return torch.cat((body_planar_velocity(state.physical), body_vertical_velocity(state.physical)[:, None]), -1)

    def tracking_velocity(self, state: TaskState) -> Tensor:
        return torch.cat((body_planar_velocity(state.physical), state.physical[:, ANGULAR_BODY][:, 2:3]), -1)

    # ---------- 高程与教师观测 ----------

    def actor_heightmap(self, state: TaskState) -> Tensor:
        position, yaw = self.world_position(state), self.yaw(state)
        x, y = self.scan_points.unbind(-1)
        points = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                              yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1) + position[:, None, :2]
        # 裁剪来自官方HeightmapObsCfg，不用于清洗非有限值。
        return (position[:, 2:3] - self.surface_height(points) - self.cfg.terrain.scan_height_offset_m).clamp(-10., 10.)

    @staticmethod
    def actor_mask(state: TaskState) -> Tensor:
        return state.context[:, CONTEXT_EXTEROCEPTIVE_MASK:CONTEXT_EXTEROCEPTIVE_MASK + 1]

    def physical_features(self, state: TaskState) -> Tensor:
        return torch.cat((state.physical[:, ROBOT_STATE], state.physical[:, self.auxiliary]), -1)

    @staticmethod
    def simulation_context(state: TaskState) -> Tensor:
        return torch.cat((state.context[:, :1], state.context[:, CONTEXT_COM],
                          state.context[:, CONTEXT_ENCODER_BIAS]), -1)

    def model_input(self, state: TaskState, action: Tensor) -> Tensor:
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), self.actor_heightmap(state)), -1)

    def supervised_model_input(self, state: TaskState, action: Tensor, recorded: TaskState) -> Tensor:
        """保留原监督协议：记录pose处的起伏半精度舍入，加预测机身高度。"""
        recorded_relief = (self.actor_heightmap(recorded) - recorded.physical[:, HEIGHT]).half().float()
        scan = recorded_relief + state.physical[:, HEIGHT]
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), scan), -1)

    def privileged(self, state: TaskState) -> Tensor:
        physical = state.physical
        memory = state.memory
        flags = (physical[:, FOOT_CONTACT_FORCE].reshape(-1, 4, 3) > 1.).any(-1).float() - .5
        features = torch.cat((self.body_linear_velocity(state).clamp(-100., 100.),
                              physical[:, ANGULAR_BODY].clamp(-100., 100.), physical[:, GRAVITY], state.command,
                              physical[:, JOINT_OFFSET].clamp(-100., 100.), physical[:, JOINT_VELOCITY].clamp(-100., 100.),
                              physical[:, JOINT_TORQUE].clamp(-100., 100.), physical[:, JOINT_ACCELERATION].clamp(-100., 100.),
                              state.memory[:, PREVIOUS_ACTION].clamp(-10., 10.), flags,
                              self.feet_height(physical).clamp(-10., 10.)), -1)
        # 沿用SuperMimic的末端价值地形梯度：保留高度导数，截断地图起伏导数。
        scan = physical[:, HEIGHT] + (self.actor_heightmap(state) - physical[:, HEIGHT]).detach()
        heading_error = memory[:, HEADING_TARGET] - self.yaw(state)
        # 对齐main的任务记忆，追加当前奖励的上一腾空／支撑时长与首次接触；前一步动作已在特权80维中。
        task_memory = torch.cat((memory[:, ELAPSED_TIME:COMMAND_TIMER + 1],
                                 heading_error.sin()[:, None], heading_error.cos()[:, None],
                                 memory[:, STANDING:FORWARD + 1], memory[:, CONTACT],
                                 memory[:, AIR_TIME], memory[:, CONTACT_TIME], memory[:, COMMAND_OVERRIDE, None],
                                 memory[:, PREVIOUS_PREVIOUS_ACTION], memory[:, LAST_AIR_TIME],
                                 memory[:, FIRST_CONTACT], memory[:, LAST_CONTACT_TIME]), -1)
        return torch.cat((features, scan, task_memory), -1)

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        physical = state.physical
        observation = torch.cat((physical[:, ANGULAR_BODY], physical[:, GRAVITY], state.command,
                                 physical[:, JOINT_OFFSET] + state.context[:, CONTEXT_ENCODER_BIAS],
                                 physical[:, JOINT_VELOCITY], state.memory[:, PREVIOUS_ACTION]), -1)
        if noisy and self.cfg.noise.add_noise:
            observation = observation + (2 * torch.rand_like(observation) - 1) * self.noise_amplitude
        # 对应源PropObsCfg逐项clip；重力与命令没有额外裁剪。
        return torch.cat((observation[:, :3].clamp(-100., 100.), observation[:, 3:9],
                          observation[:, 9:33].clamp(-100., 100.), observation[:, 33:45].clamp(-10., 10.)), -1)

    # ---------- 回合记忆与接触计时 ----------

    def initialize_state(self, physical: Tensor, context: Tensor) -> TaskState:
        command, timer, heading, standing, forward = sample_commands(self.cfg.commands, context, physical[:, 72])
        memory = physical.new_zeros((physical.shape[0], MEMORY_DIM))
        memory[:, COMMAND_TIMER], memory[:, HEADING_TARGET] = timer, heading
        memory[:, STANDING], memory[:, FORWARD] = standing, forward
        command = heading_command(self.cfg.commands, command, physical[:, 72], heading, standing)
        return TaskState(physical, memory, command, context)

    def feet_height(self, physical: Tensor) -> Tensor:
        """按官方每足0.03米网格的四条射线平均地面，足端沿用本项目site。"""
        feet = physical[:, FEET].reshape(-1, 4, 3)
        pose = physical[:, POSE]
        yaw = pose[:, 2, None]
        centers = torch.stack((yaw.cos() * feet[..., 0] - yaw.sin() * feet[..., 1],
                               yaw.sin() * feet[..., 0] + yaw.cos() * feet[..., 1]), -1) + pose[:, None, :2]
        rotation = quat_to_mat(physical[:, self.root_quat])
        foot_yaw = self.kinematics.foot_yaw(rotation, physical[:, JOINT_OFFSET] + self.default_joint_pos)[:, :, None]
        x, y = self.foot_scan_points.unbind(-1)
        offsets = torch.stack((foot_yaw.cos() * x - foot_yaw.sin() * y,
                               foot_yaw.sin() * x + foot_yaw.cos() * y), -1)
        xy = centers[:, :, None] + offsets
        ground = self.surface_height(xy).mean(-1)
        base_z = physical[:, HEIGHT] + self.surface_height(pose[:, :2])[:, None]
        return base_z + feet[..., 2] - ground

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor, *, contact_times=None) -> TaskState:
        clock = state.memory[:, ELAPSED_TIME] + self.dt
        command, timer, heading, standing, forward = advance_commands(self.cfg.commands, state, next_physical, self.dt)
        if contact_times is None:
            contact = next_physical[:, CONTACT_FOUND]
            previous = state.memory[:, CONTACT]
            # 保留原有最大持续耦合计时，不在概率恒定时重复制造落地事件。
            shared = torch.minimum(previous, contact)
            contact_stay = shared / previous.clamp_min(1e-6)
            air_stay = torch.minimum(1 - previous, 1 - contact) / (1 - previous).clamp_min(1e-6)
            air_time = air_stay * state.memory[:, AIR_TIME] + (1 - contact) * self.dt
            contact_time = contact_stay * state.memory[:, CONTACT_TIME] + contact * self.dt
            first_contact = contact - shared
            first_air = previous - shared
            last_air = first_contact * (state.memory[:, AIR_TIME] + self.dt) + (1 - first_contact) * state.memory[:, LAST_AIR_TIME]
            last_contact = first_air * (state.memory[:, CONTACT_TIME] + self.dt) + (1 - first_air) * state.memory[:, LAST_CONTACT_TIME]
        else:
            air_time, contact_time, last_air, first_contact, last_contact = contact_times
            contact = (contact_time > 0).to(next_physical.dtype)
        memory = torch.cat((action, clock[:, None], timer[:, None], heading[:, None],
                            standing.to(action.dtype)[:, None], forward.to(action.dtype)[:, None], contact,
                            air_time, contact_time, state.memory[:, COMMAND_OVERRIDE, None],
                            state.memory[:, PREVIOUS_ACTION], last_air, first_contact, last_contact), -1)
        return TaskState(next_physical, memory, command, state.context)

    # ---------- 广义状态、残差积分与正运动学 ----------

    def generalized_state(self, state: TaskState):
        """完整姿态来自四元数；局部速度在边界转换为参考实现的世界速度。"""
        physical = state.physical
        rotation = quat_to_mat(physical[:, self.root_quat])
        velocity = yaw_rotate_inverse(physical[:, VELOCITY_HEADING], -self.yaw(state))
        return (self.world_position(state), rotation, velocity, mv3(rotation, physical[:, ANGULAR_BODY]),
                physical[:, JOINT_OFFSET] + self.default_joint_pos, physical[:, JOINT_VELOCITY])

    def physical_state(self, generalized, auxiliary: Tensor) -> Tensor:
        """从单一广义状态生成任务视图；auxiliary为不含足速的独立瞬时量。"""
        position, rotation, velocity, angular, joints, joint_velocity = generalized
        quaternion = mat_to_quat(rotation)
        rotation = quat_to_mat(quaternion)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        # 以根部为原点做同一棵FK，避免绝对位置较大时相减损失相对足端精度。
        limbs, feet_velocity = self.kinematics.forward(
            torch.zeros_like(position), rotation, velocity, angular, joints, joint_velocity)
        relative = yaw_rotate_inverse(limbs, yaw[:, None]).flatten(1)
        return torch.cat((height, -rotation[:, 2, :], yaw_rotate_inverse(velocity, yaw),
                          mv3(rotation.transpose(-1, -2), angular), joints - self.default_joint_pos, joint_velocity,
                          relative, position[:, :2], yaw[:, None], auxiliary[:, :12],
                          feet_velocity[:, :, :2].flatten(1), auxiliary[:, 12:], quaternion), -1)

    def independent_auxiliary(self, state: TaskState) -> Tensor:
        """瞬时加速度、接触与力等保留原标签；足速由速度运动学生成。"""
        return torch.cat((state.physical[:, JOINT_ACCELERATION],
                          state.physical[:, CONTACT_FOUND.start:self.physical_dim]), -1)

    def model_target(self, state: TaskState, next_state: TaskState) -> Tensor:
        """沿用BlinkMimic的速度增量与位置残差目标，附加Go2独立辅助量。"""
        return torch.cat((self.dynamics.target(self.generalized_state(state), self.generalized_state(next_state)),
                          self.independent_auxiliary(next_state)), -1)

    def model_increment(self, state: TaskState, next_state: TaskState) -> Tensor:
        """普通物理增量只用于误差尺度统计，避免残差方差改变监督权重。"""
        return torch.cat((self.dynamics.increments(self.generalized_state(state), self.generalized_state(next_state)),
                          self.independent_auxiliary(next_state)), -1)

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        del action
        generalized = self.dynamics.integrate(self.generalized_state(state), delta[:, :GENERALIZED_DELTA_DIM])
        auxiliary = torch.cat((delta[:, GENERALIZED_DELTA_DIM:MODEL_CONTACT.start],
                               delta[:, MODEL_CONTACT].sigmoid(), delta[:, MODEL_CONTACT.stop:MODEL_HISTORY_CONTACT.start],
                               delta[:, MODEL_HISTORY_CONTACT].sigmoid()), -1)
        return self.physical_state(generalized, auxiliary)

    def model_error(self, current: TaskState, predicted: TaskState, target: TaskState,
                    recorded_current: TaskState) -> Tensor:
        """按目标状态坐标监督广义物理误差；接触使用预测概率残差。"""
        del current, recorded_current
        return torch.cat((self.dynamics.error(self.generalized_state(predicted), self.generalized_state(target)),
                          self.independent_auxiliary(predicted) - self.independent_auxiliary(target)), -1)

    # ---------- 奖励、失败与稀有窗口 ----------

    def reward_terms(self, state, action, next_state, failure):
        return reference_reward_terms(self, state, action, next_state, failure)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        return self.dt * sum(self.reward_scales[name] * value
                             for name, value in self.reward_terms(state, action, next_state, failure).items())

    def hard_termination(self, state: TaskState) -> Tensor:
        return ((state.physical[:, 3] > -math.cos(self.cfg.reward.fall_angle))
                | (state.physical[:, BASE_CONTACT_FORCE].squeeze(-1) > self.cfg.reward.base_contact_threshold))

    def soft_termination(self, state: TaskState) -> Tensor:
        # 倾角平滑沿用现有SuperMimic；新增base接触保持来源硬阈值，不私自另设软化尺度。
        tilt = torch.acos((-state.physical[:, 3]).clamp(-1 + 1e-4, 1 - 1e-4))
        orientation = torch.sigmoid((tilt - self.cfg.reward.fall_angle) / math.radians(self.cfg.reward.fall_softness_deg))
        base_contact = state.physical[:, BASE_CONTACT_FORCE].squeeze(-1) > self.cfg.reward.base_contact_threshold
        return torch.where(base_contact, torch.ones_like(orientation), orientation)

    def _replay_events(self, window):
        """严格沿用参考按state基准高度估计的稀有事件，不添加地形修正。"""
        physical = torch.cat((window.initial.physical[:, None], window.targets.physical), 1)
        batch, length = physical.shape[:2]
        flat = physical.flatten(0, 1)
        feet = flat[:, FEET].reshape(-1, 4, 3)
        height = flat[:, HEIGHT] + feet[..., 2]
        contact = (height < .035).reshape(batch, length, 4)
        flight = ~contact.any(-1)
        changes = (contact[:, 1:] != contact[:, :-1]).sum(-1)
        hard_motion = physical[..., VELOCITY_HEADING][..., 2].abs() > .6
        falls = ~window.valid[:, -1]
        return changes, flight, hard_motion, falls

    def replay_priority(self, window) -> Tensor:
        """严格按来源计算接触切换、飞行和窗口提前结束的分数。"""
        changes, flight, _, falls = self._replay_events(window)
        return changes.sum(-1).float() + flight.sum(-1).float() + 4.0 * falls.float()

    def archive_worthy(self, window) -> Tensor:
        """归档至少两帧飞行、明显垂向运动或真实跌倒。"""
        _, flight, hard_motion, falls = self._replay_events(window)
        return (flight.sum(-1) >= 2) | hard_motion.any(-1) | falls
