# Copyright (c) 2026-2027 zh
"""内容：字段扩展与startup事件结束后，为reset常量重算保留固定地址的CUDA图。"""

from types import MethodType

import mujoco_warp as mjwarp
import warp as wp
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.sim import _suspend_gc


def configure_reset_constant_graph(env):
    """只捕获实际激活reset事件声明的set_const_0，保留每次随机化与完整重算。"""
    sim = env.sim
    if sim.use_cuda_graph and any(
        ("recompute", RecomputeLevel.set_const_0) in env.event_manager.get_term_cfg(name).func.__dict__.items()
        for mode, names in env.event_manager.active_terms.items() if mode == "reset" for name in names
    ):
        with _suspend_gc(), wp.ScopedDevice(sim.wp_device):
            mjwarp.set_const_0(sim.wp_model, sim.wp_data)
            wp.synchronize_device(sim.wp_device)
            with wp.ScopedCapture() as capture:
                mjwarp.set_const_0(sim.wp_model, sim.wp_data)
        graphs = {RecomputeLevel.set_const_0: capture.graph}

        def recompute_constants(self, level):
            with wp.ScopedDevice(self.wp_device):
                wp.capture_launch(graphs[level])

        # 实例方法的闭包持有图；模型和数据地址沿用初始化完成后的sim字段。
        sim.recompute_constants = MethodType(recompute_constants, sim)
