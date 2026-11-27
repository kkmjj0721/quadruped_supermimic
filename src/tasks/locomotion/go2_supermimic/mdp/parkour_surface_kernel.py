# Copyright (c) 2026-2027 zh
"""
内容：
    Parkour 精确碰撞高度的 FP64 稀疏 CUDA 核，按查询块跳过末尾全空候选槽。
    保留原候选顺序、数值舍入与边界语义；梯度由 ParkourSurface.height 的栅格 STE 提供。
"""

import torch
import triton
from triton import language as tl


@triton.jit
def _collision_height_kernel(
    XY, HALF, SURFACE_IDS, SURFACE_COUNTS, BOUNDS, FLAT_TOP, FIELD_LAYOUT, FIELD_VALUES,
    TRIANGLE_IDS, TRIANGLE_COUNTS, TRIANGLE_ORIGIN, TRIANGLE_W0, TRIANGLE_W1, TRIANGLE_Z, OUTPUT,
    COUNT: tl.constexpr, SURFACE_CELL: tl.constexpr, TRIANGLE_CELL: tl.constexpr,
    SURFACE_X: tl.constexpr, SURFACE_Y: tl.constexpr, SURFACE_SLOTS: tl.constexpr,
    TRIANGLE_X: tl.constexpr, TRIANGLE_Y: tl.constexpr, TRIANGLE_SLOTS: tl.constexpr,
    ORIGIN_S0: tl.constexpr, ORIGIN_S1: tl.constexpr, Z_S0: tl.constexpr, Z_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    point_id = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = point_id < COUNT
    x = tl.load(XY + 2 * point_id, valid, other=0).to(tl.float64)
    y = tl.load(XY + 2 * point_id + 1, valid, other=0).to(tl.float64)
    hx = tl.load(HALF).to(tl.float64)
    hy = tl.load(HALF + 1).to(tl.float64)

    sx = tl.floor((x + hx) / SURFACE_CELL).to(tl.int64)
    sy = tl.floor((y + hy) / SURFACE_CELL).to(tl.int64)
    sx = tl.minimum(tl.maximum(sx, 0), SURFACE_X - 1)
    sy = tl.minimum(tl.maximum(sy, 0), SURFACE_Y - 1)
    surface_base = (sx * SURFACE_Y + sy) * SURFACE_SLOTS
    rectangle_height = tl.full((BLOCK,), -float("inf"), tl.float64)
    covered = tl.full((BLOCK,), False, tl.int1)
    counts = tl.load(SURFACE_COUNTS + sx * SURFACE_Y + sy, valid, other=0)
    last = tl.max(counts, 0).to(tl.int32)
    # 表中非零候选连续存放，只遍历当前查询块实际需要的前缀。
    for slot in range(last):
        ids = tl.load(SURFACE_IDS + surface_base + slot, valid, other=0).to(tl.int64)
        x0 = tl.load(BOUNDS + 4 * ids, valid, other=0)
        y0 = tl.load(BOUNDS + 4 * ids + 1, valid, other=0)
        x1 = tl.load(BOUNDS + 4 * ids + 2, valid, other=1)
        y1 = tl.load(BOUNDS + 4 * ids + 3, valid, other=1)
        inside = valid & (ids != 0) & (x >= x0) & (y >= y0) & (x <= x1) & (y <= y1)
        address = tl.load(FIELD_LAYOUT + 3 * ids, valid, other=0)
        rows = tl.load(FIELD_LAYOUT + 3 * ids + 1, valid, other=2)
        cols = tl.load(FIELD_LAYOUT + 3 * ids + 2, valid, other=2)
        # 这些裁剪逐项对应原矩形高度场坐标公式，没有新增数值清洗。
        u = tl.minimum(tl.maximum((x - x0) / (x1 - x0), 0., propagate_nan=tl.PropagateNan.ALL),
                       1., propagate_nan=tl.PropagateNan.ALL)
        v = tl.minimum(tl.maximum((y - y0) / (y1 - y0), 0., propagate_nan=tl.PropagateNan.ALL),
                       1., propagate_nan=tl.PropagateNan.ALL)
        gx, gy = u * (cols - 1), v * (rows - 1)
        ix = tl.minimum(gx.to(tl.int64), cols - 2)
        iy = tl.minimum(gy.to(tl.int64), rows - 2)
        fx, fy = gx - ix, gy - iy
        index = address + iy * cols + ix
        a = tl.load(FIELD_VALUES + index, valid, other=0)
        b = tl.load(FIELD_VALUES + index + 1, valid, other=0)
        c = tl.load(FIELD_VALUES + index + cols, valid, other=0)
        d = tl.load(FIELD_VALUES + index + cols + 1, valid, other=0)
        # 原实现先在 FP32 栅格上相减，再与 FP64 的 fx、fy 相乘。
        ba, db, ca, dc = b - a, d - b, c - a, d - c
        lower = a.to(tl.float64) + fx * ba.to(tl.float64) + fy * db.to(tl.float64)
        upper = a.to(tl.float64) + fy * ca.to(tl.float64) + fx * dc.to(tl.float64)
        interpolated = tl.where(fx >= fy, lower, upper)
        flat = tl.load(FLAT_TOP + ids, valid, other=0).to(tl.float64)
        candidate = tl.where(address == 0, flat, interpolated)
        rectangle_height = tl.maximum(rectangle_height, tl.where(inside, candidate, -float("inf")),
                                      propagate_nan=tl.PropagateNan.ALL)
        covered = covered | inside

    tx = tl.floor((x + hx) / TRIANGLE_CELL).to(tl.int64)
    ty = tl.floor((y + hy) / TRIANGLE_CELL).to(tl.int64)
    tx = tl.minimum(tl.maximum(tx, 0), TRIANGLE_X - 1)
    ty = tl.minimum(tl.maximum(ty, 0), TRIANGLE_Y - 1)
    triangle_base = (tx * TRIANGLE_Y + ty) * TRIANGLE_SLOTS
    triangle_height = tl.full((BLOCK,), -float("inf"), tl.float64)
    counts = tl.load(TRIANGLE_COUNTS + tx * TRIANGLE_Y + ty, valid, other=0)
    last = tl.max(counts, 0).to(tl.int32)
    for slot in range(last):
        ids = tl.load(TRIANGLE_IDS + triangle_base + slot, valid, other=0).to(tl.int64)
        ox = tl.load(TRIANGLE_ORIGIN + ids * ORIGIN_S0, valid, other=0)
        oy = tl.load(TRIANGLE_ORIGIN + ids * ORIGIN_S0 + ORIGIN_S1, valid, other=0)
        dx, dy = x - ox, y - oy
        w0x = tl.load(TRIANGLE_W0 + 2 * ids, valid, other=0)
        w0y = tl.load(TRIANGLE_W0 + 2 * ids + 1, valid, other=0)
        w1x = tl.load(TRIANGLE_W1 + 2 * ids, valid, other=0)
        w1y = tl.load(TRIANGLE_W1 + 2 * ids + 1, valid, other=0)
        w0 = dx * w0x + dy * w0y
        w1 = dx * w1x + dy * w1y
        w2 = 1. - w0 - w1
        inside = valid & (ids != 0) & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z0 = tl.load(TRIANGLE_Z + ids * Z_S0, valid, other=0)
        z1 = tl.load(TRIANGLE_Z + ids * Z_S0 + Z_S1, valid, other=0)
        z2 = tl.load(TRIANGLE_Z + ids * Z_S0 + 2 * Z_S1, valid, other=0)
        candidate = w0 * z0 + w1 * z1 + w2 * z2
        triangle_height = tl.maximum(triangle_height, tl.where(inside, candidate, -float("inf")),
                                     propagate_nan=tl.PropagateNan.ALL)

    height = tl.maximum(rectangle_height, triangle_height, propagate_nan=tl.PropagateNan.ALL)
    in_core = (x >= -hx) & (y >= -hy) & (x <= hx) & (y <= hy)
    result = tl.where(in_core, height, tl.where(covered, rectangle_height, 0.))
    tl.store(OUTPUT + point_id, result, valid)


@torch.library.custom_op("go2_supermimic::parkour_collision_height_fp64", mutates_args=())
def parkour_collision_height_fp64(
    xy: torch.Tensor, half: torch.Tensor, surface_ids: torch.Tensor, surface_counts: torch.Tensor, bounds: torch.Tensor,
    flat_top: torch.Tensor, field_layout: torch.Tensor, field_values: torch.Tensor,
    triangle_ids: torch.Tensor, triangle_counts: torch.Tensor, triangle_origin: torch.Tensor, triangle_w0: torch.Tensor,
    triangle_w1: torch.Tensor, triangle_z: torch.Tensor, surface_cell: float, triangle_cell: float,
) -> torch.Tensor:
    """独立算子保留输入形状，最后一个 program 仅屏蔽形状外的填充 lane。"""
    points = xy.contiguous().view(-1, 2)
    result = torch.empty(xy.shape[:-1], device=xy.device, dtype=xy.dtype)
    _collision_height_kernel[(triton.cdiv(points.shape[0], 128),)](
        points, half, surface_ids, surface_counts, bounds, flat_top, field_layout, field_values,
        triangle_ids, triangle_counts, triangle_origin, triangle_w0, triangle_w1, triangle_z, result,
        points.shape[0], surface_cell, triangle_cell,
        surface_ids.shape[0], surface_ids.shape[1], surface_ids.shape[2],
        triangle_ids.shape[0], triangle_ids.shape[1], triangle_ids.shape[2],
        triangle_origin.stride(0), triangle_origin.stride(1), triangle_z.stride(0), triangle_z.stride(1),
        128, num_warps=4, enable_fp_fusion=False,
    )
    return result


@parkour_collision_height_fp64.register_fake
def _parkour_collision_height_fake(
    xy, half, surface_ids, surface_counts, bounds, flat_top, field_layout, field_values,
    triangle_ids, triangle_counts, triangle_origin, triangle_w0, triangle_w1, triangle_z, surface_cell, triangle_cell,
):
    return xy.new_empty(xy.shape[:-1])

