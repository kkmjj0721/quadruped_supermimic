# Copyright (c) 2026-2027 zh
"""内容：统一完整监督、价值损失与返回递推的编译边界及随机数语义。"""

import torch


def compile_kernel(function, backend: str, *, fallback_random=True, emulate_precision_casts=True):
    """编译固定形状的完整函数，保持显式舍入及原生随机数消费。"""
    arguments = {"backend": backend, "fullgraph": True, "dynamic": False}
    if backend == "inductor":
        arguments["options"] = {"emulate_precision_casts": emulate_precision_casts, "fallback_random": fallback_random,
                                "triton.cudagraphs": False}
    return torch.compile(function, **arguments)
