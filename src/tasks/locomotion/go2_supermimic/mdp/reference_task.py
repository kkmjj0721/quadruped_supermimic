# Copyright (c) 2026-2027 zh
"""内容：将PROOF官方教师的高程、本体观测和奖励接入Go2可微世界模型。"""

import math

import torch
from torch import Tensor

from agents.supermimic.compilation import compile_kernel
from agents.supermimic.interfaces import AlgorithmSpec, TaskState

from .reference_rewards import reference_reward_terms
from .parkour import EDGE_FAMILIES, PARKOUR_FAMILIES, SPECIAL_FAMILIES
from .commands import advance_commands, heading_command, sample_commands
from .reference_kinematics import Go2Kinematics
from .reference_rotations import mat_to_quat, mv3, quat_to_mat
from .structured_dynamics import ResidualDynamics
from .reference_state import (
    ANGULAR_BODY, AUXILIARY, BASE_CONTACT_FORCE, CONTACT_FOUND, CONTEXT_COM, FEET,
    FOOT_CONTACT_FORCE, GENERALIZED_DELTA_DIM, GRAVITY, HEIGHT, JOINT_ACCELERATION, JOINT_OFFSET,
    JOINT_TORQUE, JOINT_VELOCITY, MODEL_CONTACT, MODEL_EVENTS, MODEL_HISTORY_CONTACT, PHYSICAL_DIM, POSE,
    ROBOT_STATE, VELOCITY_HEADING, body_planar_velocity, body_vertical_velocity, yaw_rotate_inverse,
)
from .state_spec import (
    AIR_TIME, COMMAND_OVERRIDE, COMMAND_TIMER, CONTACT, CONTACT_TIME, CONTEXT_DIM,
    CONTEXT_ENCODER_BIAS, CONTEXT_EXTEROCEPTIVE_MASK, ELAPSED_TIME, FIRST_CONTACT,
    FOOT_NAMES, FORWARD, HEADING_TARGET, LAST_AIR_TIME, LAST_CONTACT_TIME,
    MEMORY_DIM, PARKOUR_CONTEXT_DIM, PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION, STANDING,
)


class Go2ReferenceTaskModel:
    """真实采集与想象共用教师观测、残差状态、奖励和回合记忆。"""

    observation_term_dims = (3, 3, 3, 12, 12, 12)

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.parkour = cfg.terrain.mode == "parkour"
        self.cfg = cfg
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
        if self.parkour:
            self.special_family = torch.tensor([name in SPECIAL_FAMILIES for name in PARKOUR_FAMILIES], device=self.device)
            self.edge_family = torch.tensor([name in EDGE_FAMILIES for name in PARKOUR_FAMILIES], device=self.device)
            axis = self.default_joint_pos.new_tensor((-.01, .01))
            yz = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(4, 2)
            self.volume_offsets = torch.cat((torch.zeros_like(yz[:, :1]), yz), -1)
        self._read_physical_kernel = self._read_physical_tensors
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
        self.replay_feature_indices = torch.tensor(
            (HEIGHT.start, *range(FEET.start, FEET.stop), VELOCITY_HEADING.start + 2),
            dtype=torch.long, device=self.device)
        self.model_input_dim = PHYSICAL_DIM - 3 + 12 + 12 + 16 + self.heightmap_dim
        noise = cfg.noise.noise_scales
        self.noise_amplitude = self.default_joint_pos.new_tensor(
            (noise.base_ang_vel,) * 3 + (noise.projected_gravity,) * 3 + (0.,) * 3
            + (noise.joint_pos,) * 12 + (noise.joint_vel,) * 12 + (0.,) * 12) * cfg.noise.noise_level
        self.spec = AlgorithmSpec(
            state_dim=PHYSICAL_DIM + 4, delta_dim=PHYSICAL_DIM - 45, action_dim=12,
            observation_dim=45, privileged_dim=cfg.env.num_critic_obs, context_dim=PARKOUR_CONTEXT_DIM if self.parkour else CONTEXT_DIM,
            memory_dim=MEMORY_DIM, command_dim=3, heightmap_dim=self.heightmap_dim,
            history_length=cfg.env.num_obs_history,
            semantic_id=("go2_proof_parkour_ae910b8_contact_events_residual_fk_v1" if self.parkour else
                         "go2_proof_teacher_ae910b8_contact_events_residual_fk_v3"),
            model_id="supermimic_proof_teacher_v2")

    # ---------- 真实物理状态 ----------

    def surface_height(self, xy: Tensor) -> Tensor:
        return self.surface.height(xy)

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
        gain, bias = env.sim.model.actuator_gainprm, env.sim.model.actuator_biasprm
        self.actuator_gain = gain[:, self.actuator_ids, 0] if gain.ndim == 3 else gain[self.actuator_ids, 0]
        self.actuator_bias = bias[:, self.actuator_ids, :3] if bias.ndim == 3 else bias[self.actuator_ids, :3]

    def configure_collection_compile(self, *, enabled, backend):
        """只编译纯张量快照核，MjLab惰性属性和步号在eager边界解析。"""
        self._read_physical_kernel = (compile_kernel(self._read_physical_tensors, backend)
                                      if enabled else self._read_physical_tensors)

    def read_physical(self, env) -> Tensor:
        """按当前仿真时序读取惰性数据，显式传递快照所需张量。"""
        data = env.scene["robot"].data
        sensor = env.scene["feet_ground_contact"].data
        position, yaw = data.root_link_pos_w, data.heading_w
        body_position, site_position = data.body_link_pos_w, data.site_pos_w
        calf_history = env.scene["calf_ground_contact"].data.force_history
        base_history = env.scene["base_ground_contact"].data.force_history
        sim = env.sim.data
        return self._read_physical_kernel(
            position, yaw, body_position, site_position,
            data.projected_gravity_b, data.root_link_lin_vel_w, data.root_link_ang_vel_b,
            data.joint_pos, data.default_joint_pos, data.joint_vel, data.joint_acc, data.site_lin_vel_w,
            sensor.found, sensor.force, sensor.force_history, data.actuator_force,
            calf_history, base_history, sim.ctrl[:], sim.actuator_length[:], sim.actuator_velocity[:],
            data.root_link_quat_w)

    def _read_physical_tensors(self, position, yaw, body_position, site_position, gravity,
                               linear_velocity, angular_velocity, joint_position, default_joint,
                               joint_velocity, joint_acceleration, site_velocity, found, foot_force,
                               foot_history, actuator_force, calf_history, base_history, control,
                               actuator_length, actuator_velocity, quaternion) -> Tensor:
        """纯张量坐标变换与拼接，输出独立于仿真存储的物理状态。"""
        limbs = (body_position[:, self.hip_ids], body_position[:, self.knee_ids], site_position[:, self.foot_ids])
        relative = [yaw_rotate_inverse(points - position[:, None], yaw[:, None]).flatten(1) for points in limbs]
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        calf_net = calf_history[:, self.calf_geom_ids].sum(dim=2)
        calf_force = calf_net.norm(dim=-1).amax(dim=2)
        base_force = base_history.norm(dim=-1).amax(dim=(1, 2))[:, None]
        computed_torque = (self.actuator_gain * control[:, self.actuator_ids]
                           + self.actuator_bias[..., 0] + self.actuator_bias[..., 1] * actuator_length[:, self.actuator_ids]
                           + self.actuator_bias[..., 2] * actuator_velocity[:, self.actuator_ids])
        history_contact = (foot_history.norm(dim=-1).amax(dim=2) > self.cfg.reward.contact_force_threshold).float()
        # 真实传感器返回地面受力，转为机器人受力；事件仍在此物理边界生成。
        foot_force = -foot_force
        stumble = (foot_force[..., :2].norm(dim=-1)
                   > self.cfg.reward.stumble_ratio * foot_force[..., 2].abs()).any(-1, keepdim=True).float()
        calf_contact = (calf_force > self.cfg.reward.contact_force_threshold).float()
        return torch.cat((height, gravity, yaw_rotate_inverse(linear_velocity, yaw), angular_velocity,
                          joint_position - default_joint, joint_velocity, *relative, position[:, :2], yaw[:, None],
                          joint_acceleration, site_velocity[:, self.foot_ids, :2].flatten(1), (found > 0).float(),
                          foot_force.flatten(1), actuator_force, calf_force, base_force, computed_torque,
                          history_contact, stumble, calf_contact, quaternion), -1)

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
        return self.model_input_from_heightmap(state, action, self.actor_heightmap(state))

    def model_input_from_heightmap(self, state: TaskState, action: Tensor, heightmap: Tensor) -> Tensor:
        """复用调用方在同一状态上计算的高程，保持原输入拼接顺序。"""
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), heightmap), -1)

    def supervised_model_input(self, state: TaskState, action: Tensor, recorded: TaskState) -> Tensor:
        """保留原监督协议：记录pose处的起伏半精度舍入，加预测机身高度。"""
        recorded_relief = (self.actor_heightmap(recorded) - recorded.physical[:, HEIGHT]).half().float()
        scan = recorded_relief + state.physical[:, HEIGHT]
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), scan), -1)

    def privileged(self, state: TaskState) -> Tensor:
        return self.privileged_from_heightmap(state, self.actor_heightmap(state))

    def privileged_from_heightmap(self, state: TaskState, heightmap: Tensor) -> Tensor:
        """复用高程数值，价值支路仍只保留机身高度导数。"""
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
        scan = physical[:, HEIGHT] + (heightmap - physical[:, HEIGHT]).detach()
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
        command = heading_command(self.cfg.commands, command, physical[:, 72], heading, standing, forward, context)
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
                               delta[:, MODEL_HISTORY_CONTACT].sigmoid(), delta[:, MODEL_EVENTS].sigmoid()), -1)
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
        """独立完整窗口入口复用与延迟读取相同的事件列定义。"""
        physical = torch.cat((window.initial.physical[:, None], window.targets.physical), 1)
        return self._events_from_features(physical[..., self.replay_feature_indices], window.valid)

    @staticmethod
    def _events_from_features(features, valid):
        feet = features[..., 1:13].reshape(*features.shape[:-1], 4, 3)
        height = features[..., :1] + feet[..., 2]
        contact = height < .035
        flight = ~contact.any(-1)
        changes = (contact[:, 1:] != contact[:, :-1]).sum(-1)
        hard_motion = features[..., 13].abs() > .6
        return changes, flight, hard_motion, ~valid[:, -1]

    @staticmethod
    def _selection_from_events(events):
        changes, flight, hard_motion, falls = events
        priority = changes.sum(-1).float() + flight.sum(-1).float() + 4.0 * falls.float()
        worthy = (flight.sum(-1) >= 2) | hard_motion.any(-1) | falls
        return priority, worthy

    def replay_selection(self, window):
        """完整窗口的显式事件接口；只求值一次接触、飞行和跌倒。"""
        return self._selection_from_events(self._replay_events(window))

    def replay_selection_from_features(self, features, valid):
        """候选只读取十四列后评分，不触及其余状态、记忆或动作。"""
        return self._selection_from_events(self._events_from_features(features, valid))
