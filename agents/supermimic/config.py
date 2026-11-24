# Copyright (c) 2026-2027 zh
"""内容：声明高程图教师策略的训练预算、预热排期和显式资源覆盖参数。"""

from dataclasses import dataclass


RUNNER_RUNTIME_FIELDS = ("max_iterations", "save_interval", "log_interval", "imag_microbatch",
                         "critic_microbatch", "wm_compile", "fused_adam", "cuda_graph",
                         "detailed_timing", "timing_interval", "replay_history_storage",
                         "wm_bf16", "actor_bf16", "critic_bf16", "metric_interval")


@dataclass
class ExecutionConfig:
    """声明分批和默认s101执行方式，不改变总训练批量或检查点语义。"""

    imag_microbatch: int = 4096
    critic_microbatch: int = 32768
    wm_compile: bool = True  # 完整自回归监督及损失。
    fused_adam: bool = True
    cuda_graph: bool = True  # CUDA捕获采样与更新，融合指标累计；CPU运行需显式关闭。
    detailed_timing: bool = False  # 图外记录阶段事件，原有计时指标保持原义。
    timing_interval: int = 1  # 详细计时的采样轮数间隔；首次使用和图准备轮强制记录。
    metric_interval: int = 1  # 设备指标汇总及TensorBoard写入间隔；护栏和learn返回强制完成窗口。
    replay_history_storage: str = "frames"  # 默认只保存实际带噪单帧；full保留完整历史作显式对照。
    wm_bf16: bool = False  # 仅世界模型MLP使用CUDA BF16；监督和想象共享该边界。
    actor_bf16: bool = False  # 策略MLP使用CUDA BF16；潜变量融合与观测归一化保留FP32。
    critic_bf16: bool = False  # 仅价值MLP使用CUDA BF16；输出和返回递推保留FP32。



@dataclass
class SuperMimicConfig:
    """共享算法字段与通用默认；各机器人在自己的algo段声明实际训练配方。"""

    replay_capacity: int = 1000
    # 每轮平均真实控制步；2.5对应UTD=0.4，循环累计后取整。
    collect_steps: float = 2.5
    warmup_iterations: int = 0
    wm_rewarmup_iterations: int = 1500
    critic_warmup_iterations: int = 500
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 10
    imag_batch: int = 4096
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    prop_encoder_hidden: tuple[int, ...] = (128, 64)
    height_encoder_hidden: tuple[int, ...] = (64, 32)
    prop_latent_dim: int = 32
    height_latent_dim: int = 32
    velocity_loss_coef: float = 1.0  # 同时刻机身系xyz线速度MSE，监督本体latent的前三维。
    learning_rate: float = 2e-4
    actor_lr_scale: float = 1.0
    actor_optimizer: str = "adam"  # muon_adam将策略全部Linear权重交给Muon，其余参数使用Adam。
    wm_optimizer: str = "adam"  # muon_adam仅将世界模型中间Linear权重交给Muon。
    muon_weight_decay: float = 0.1  # PartyMimic默认值，仅作用于Muon参数组。
    actor_ramp_iterations: int = 1000
    lr_decay_iterations: int = 30000
    lr_final_frac: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip: float = 1.0
    grad_clip_element: float = 0.1
    compile_imagination: bool = True  # 想象单步、完整lambda-return及critic损失。
    compile_backend: str = "inductor"
    # 论文表A.5：仅真实采集时逐步逐维独立添加固定高斯噪声，单位为动作缩放前单位。
    action_noise: float = 0.1
    collect_det_frac: float = 0.0  # 无额外动作探索噪声的环境比例；机器人配方显式选择。
    wm_prioritized_frac: float = 0.25
    wm_archive_frac: float = 0.25
    archive_size: int = 100000
    archive_keep_prob: float = 0.02
    collapse_falls: float = 3.0
    collapse_interval: int = 50
    driver_frac: float = 0.25
    expert_frac: float = 0.0
    driver_fade: tuple[int, int] = (1_000_000_000, 1_000_000_000)

    def __post_init__(self):
        for name in ("wm_hidden", "actor_hidden", "critic_hidden"):
            setattr(self, name, tuple(getattr(self, name)))
        self.driver_fade = tuple(self.driver_fade)
