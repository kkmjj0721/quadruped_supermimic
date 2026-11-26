# Copyright (c) 2026-2027 zh
"""内容：声明Go2的PROOF官方教师结构、普通地形奖励及SuperMimic训练配方。"""

from dataclasses import dataclass

from agents.supermimic.config import SuperMimicConfig
from src.config.base.base_config import BaseConfig


class Go2ReferenceCfg(BaseConfig):
    """PROOF普通地形教师与当前Go2仿真设置。"""

    algorithm = "proof_teacher"

    @dataclass
    class algo(SuperMimicConfig):
        """本机器人的实际训练配方；驱动策略参数在runner.driver中维护。"""

        # ---------- 采集与回放 ----------
        replay_capacity: int = 1000
        collect_steps: float = 2.5  # 每轮平均真实控制步，2／3步交替。
        wm_prioritized_frac: float = 0.25
        wm_archive_frac: float = 0.25
        archive_size: int = 100000
        archive_keep_prob: float = 0.02

        # ---------- 世界模型与想象 ----------
        wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
        wm_window: int = 8
        wm_batch: int = 2048
        wm_updates: int = 5
        imag_batch: int = 2048
        horizon: int = 32
        gamma: float = 0.99
        lam: float = 0.95

        # ---------- 策略与价值网络 ----------
        actor_hidden: tuple[int, ...] = (512, 256, 128)
        critic_hidden: tuple[int, ...] = (512, 256, 128)
        prop_encoder_hidden: tuple[int, ...] = (128, 64)
        height_encoder_hidden: tuple[int, ...] = (64, 32)
        prop_latent_dim: int = 32  # 前三维直接监督当前机身线速度。
        height_latent_dim: int = 32
        velocity_loss_coef: float = 1.0  # 真实同一时刻机身线速度的MSE监督权重。

        # ---------- 优化与排期 ----------
        learning_rate: float = 4e-4
        actor_lr_scale: float = 4.0
        actor_optimizer: str = "muon_adam"
        wm_optimizer: str = "muon_adam"
        muon_weight_decay: float = 0.1
        warmup_iterations: int = 0
        wm_rewarmup_iterations: int = 1500
        critic_warmup_iterations: int = 500
        actor_ramp_iterations: int = 1000
        lr_decay_iterations: int = 30000
        lr_final_frac: float = 0.03
        critic_epochs: int = 5
        critic_minibatches: int = 4
        grad_clip: float = 1.0
        grad_clip_element: float = 0.1

        # ---------- 探索噪声 ----------
        action_noise: float = 0.3  # 采用本项目消融配方；标准差为动作缩放前单位，论文表A.5默认0.1。
        collect_det_frac: float = 0.25  # 各地形按比例分配无额外动作探索噪声的环境。

        # ---------- 编译与回滚 ----------
        compile_imagination: bool = True  # 编译想象单步、完整lambda-return与critic损失。
        compile_backend: str = "inductor"
        collapse_falls: float = 3.0
        collapse_interval: int = 50

    class env:
        """固定任务协议的维度、环境规模及回合预算。"""

        num_envs = 4096
        num_actions = 12
        scene_extent = 2.0
        seed = 0
        episode_length_s = 20.0
        num_obs_history = 10
        history_size = num_obs_history
        num_observations = 45
        num_obs = num_observations * num_obs_history
        num_privileged_obs = 285  # 官方特权80＋高程162＋main式奖励／命令记忆43。
        num_critic_obs = 285
        heightmap_dim = 162

    class teacher:
        """官方普通地形回合级高程latent遮蔽；平地家族保持未遮蔽。"""

        exteroceptive_dropout = 0.4

    class sim:
        """平地和粗糙地形共用的仿真预算；物理步长以秒计。"""

        nconmax = None
        njmax = 300
        broadphase = "nxn"
        solver = "cg"
        timestep = 0.005
        iterations = 10
        ls_iterations = 20
        ccd_iterations = 50  # 凸碰撞迭代上限；EPA临时显存随该值线性增长。

    class reset:
        """回合重置的位姿与速度扰动；不改变真实课程分配的出生地块。"""

        base_offset = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05),
                           "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        }
        joint_offset = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

    class viewer:
        """Go2跟随相机；距离以米计，角度以度计。"""

        distance = 1.5
        elevation = -10.0
        azimuth = 90.0

    class noise:
        """只扰动新生成的本体帧，历史帧不重复加噪。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            """对称均匀噪声的半幅，单位与对应观测相同。"""

            base_ang_vel = 0.2  # rad/s。
            projected_gravity = 0.05
            joint_pos = 0.01  # rad。
            joint_vel = 1.5  # rad/s。

    class runner:
        """本机器人的训练目标、保存节奏与执行预算；命令行可显式覆盖。"""

        max_iterations = 20000
        save_interval = 10000
        log_interval = 1
        metric_interval = 10  # 设备端汇总展示指标，每10轮回读并写TensorBoard。
        imag_microbatch = 4096
        critic_microbatch = 32768
        wm_compile = True  # 编译完整八步监督及损失。
        fused_adam = True
        cuda_graph = True  # CUDA默认捕获模型、actor和critic更新；CPU使用原执行方式。
        detailed_timing = False  # 可显式开启图外采样／更新细分，原有阶段计时继续保留。
        timing_interval = 1  # 正整数；图准备及首次阶段强制采样，其余按此轮数间隔记录。
        replay_history_storage = "frames"  # 默认压缩带噪历史；full保留完整存储作显式对照。
        wm_bf16 = True  # 世界模型MLP默认使用CUDA BF16，监督与想象一致。
        actor_bf16 = True  # 策略编码器与动作MLP默认使用CUDA BF16，观测归一化保留FP32。
        critic_bf16 = True  # 价值MLP默认使用CUDA BF16，输出与返回递推仍为FP32。
        enable_control_updates = True
        standing_zero_noise = True  # 站立命令采集关闭额外动作噪声，保留固定无噪声组。

        class driver:
            """外部策略在真实采集中代替当前策略行动，只贡献数据而不参与更新。"""

            policy = ""
            expert_policy = ""
            # 与参考 driver_frac=0.25 一致；未设置 policy 时该占比不生效。
            frac = 0.25
            expert_frac = 0.0
            fade_iterations = (1_000_000_000, 1_000_000_000)

    class control:
        """内置位置执行器对齐 legged_wbc_mjlab Go2；动作比例按关节类型显式设置。"""

        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 40.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 2.0}
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}
        armature = {"hip": 0.01, "thigh": 0.01, "calf": 0.01}
        action_scales = {"hip": 0.25, "thigh": 0.5, "calf": 0.5}
        decimation = 4
        soft_joint_pos_limit_factor = 0.9
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class collision:
        foot_condim = 3
        body_condim = 1
        foot_priority = 1
        foot_friction = (1.0, 5e-3, 5e-4)
        solref = (0.01, 1.0)
        enable_self_collision = True

    class init_state:
        pos = [0.0, 0.0, 0.3]
        default_joint = {
            **{f"{leg}_hip_joint": 0.1 if leg in ("FL", "RL") else -0.1 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_thigh_joint": 0.9 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_calf_joint": -1.8 for leg in ("FL", "FR", "RL", "RR")},
        }

    class terrain:
        """Ri-5036 HIM地形追加正反金字塔斜坡；难度采用原有行走表现课程。"""

        mode = "flat"
        seed = None
        patch_size_m = (8.0, 8.0)
        core_rows = 6
        num_cols = 20  # 沿用参考配置；MjLab课程模式按十二种子地形实际生成十二列。
        outer_flat_width_m = 20.0
        max_init_level = 0
        curriculum = True
        field_resolution_m = 0.02
        field_half_m = 6.5
        field_cache_dir = "logs/heightfields"
        scan_size_m = (1.7, 0.8)
        scan_resolution_m = 0.1
        scan_forward_offset_m = 0.6
        scan_height_offset_m = 0.25

    class domain_randomization:
        """按指定Go2参考的MjLab基类随机化，不加入外力push。"""

        enabled = True
        friction_range = (0.3, 1.5)
        spin_friction_range = (1e-4, 2e-2)
        roll_friction_range = (1e-5, 5e-3)
        encoder_bias_range = (-0.015, 0.015)
        com_range = {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.03, 0.03)}

    class commands:
        """保留当前HIM采样与heading反馈，教师奖励评价body平移和body yaw。"""

        reference_steps_per_iteration = 1
        initial_ranges = ((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0))
        stages = []
        zero_probability_start = 0.1
        zero_probability_end = 0.1
        zero_probability_iterations = 1
        resampling_time = (4.0, 12.0)
        heading_command = True
        heading_range = (-3.14, 3.14)
        heading_control_stiffness = 0.5
        rel_heading_envs = 1.0
        rel_standing_envs = 0.1
        rel_forward_envs = 0.05

    class reward:
        """PROOF普通地形配方追加main的倾斜、站立关节速度与滑移项，各项乘一次控制周期。"""

        tracking_linear_variance = 0.25
        tracking_angular_variance = 0.25
        fall_angle = 1.5
        fall_softness_deg = 5.0  # 保留SuperMimic已有的想象倾角软失败。
        base_contact_threshold = 5.0
        command_threshold = 0.1
        stand_still_command_threshold = 0.1  # main：命令平移模长加偏航绝对值不超过阈值时启用。
        foot_slip_command_threshold = 0.1  # main：命令平移模长加偏航绝对值超过阈值时启用。
        contact_force_threshold = 1.0
        stumble_ratio = 4.0
        torque_limit_nominal = 30.0
        torque_soft_factor = 0.8
        torque_hard_factor = 0.95
        torque_over_soft_scale = 50.0
        torque_over_hard_scale = 25.0
        torque_clip = 50.0  # 来源奖励明确对限幅前PD力矩设置此上限。
        feet_air_time_threshold = 0.5
        contact_time_variance_clip = 0.5

        class scales:
            track_lin_vel_xy_exp = 1.5
            track_ang_vel_z_exp = 1.0
            lin_vel_z_l2 = -1.0
            ang_vel_xy_l2 = -0.05
            body_orientation_l2 = -0.1
            stand_still = -1.0
            stand_still_joint_vel_l2 = -0.1
            foot_slip = -0.1
            joint_hip_deviation_l2 = -0.2
            joint_deviation_l2 = -2.5e-3
            joint_acc_l2 = -2.5e-7
            joint_torques_l2_soft_limit = -1e-4
            joint_power_l1 = -2e-5
            joint_pos_limits = -10.0
            action_rate_l2 = -0.1
            action_smoothness_l2 = -0.01
            undesired_contacts = -1.0
            feet_air_time = 0.5
            feet_air_time_variance = -0.1
            feet_stumble = -1.0
            feet_slide = -2.5e-2
            is_alive = 0.1
            termination_penalty = -100.0
