# Copyright (c) 2026-2027 zh
"""内容：定义Go2关节顺序、任务记忆和真实上下文槽位；物理状态见reference_state。"""

JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
FOOT_NAMES = ("FL", "FR", "RL", "RR")
CONTACT_BODY_NAMES = tuple(f"{leg}_{part}" for leg in FOOT_NAMES for part in ("hip", "thigh", "calf")) + ("base",)
ACTION_DIM = len(JOINT_NAMES)
PREVIOUS_ACTION = slice(0, 12)
ELAPSED_TIME = 12
COMMAND_TIMER = 13
HEADING_TARGET = 14
STANDING = 15
FORWARD = 16
CONTACT = slice(17, 21)
AIR_TIME = slice(21, 25)
CONTACT_TIME = slice(25, 29)
COMMAND_OVERRIDE = 29
PREVIOUS_PREVIOUS_ACTION = slice(30, 42)
LAST_AIR_TIME = slice(42, 46)
FIRST_CONTACT = slice(46, 50)
LAST_CONTACT_TIME = slice(50, 54)
MEMORY_DIM = 54
CONTEXT_FRICTION = 0
CONTEXT_ENCODER_BIAS = slice(1, 13)
CONTEXT_ORIGIN = slice(16, 19)
CONTEXT_RANGES = slice(19, 25)
CONTEXT_ZERO_PROBABILITY = 25
CONTEXT_EXTEROCEPTIVE_MASK = 26
CONTEXT_DIM = 27
CONTEXT_TERRAIN_FAMILY = 27  # 二阶段回放与想象保留回合分配的地形族。
PARKOUR_CONTEXT_DIM = 28
