# Copyright (c) 2026-2027 zh
"""
内容：
    独立插值梅花桩边长与净间隙，保留平地边框及平台出口等缝。
    箱体、坑底与平台对齐参考MjLab 1.6.0的BoxSteppingStonesTerrainCfg。
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp, darken_rgba


@dataclass(kw_only=True)
class SteppingStonesTerrainCfg(SubTerrainCfg):
    """按实际桩宽与净间隙生成中心对称网格，平台边长至少为配置值。"""

    stone_size_range: tuple[float, float] = (0.25, 0.35)
    stone_distance_range: tuple[float, float] = (0.10, 0.15)
    stone_height: float = 0.10
    stone_height_variation: float = 0.02
    floor_depth: float = 0.50
    platform_width: float = 2.5
    border_width: float = 0.25

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        side_min, side_max = self.stone_size_range
        gap_min, gap_max = self.stone_distance_range
        side = side_max + difficulty * (side_min - side_max)
        gap = gap_min + difficulty * (gap_max - gap_min)
        pitch = side + gap
        center = np.asarray(self.size) / 2
        inner_half = center - self.border_width
        grid_half_counts = np.floor(inner_half / pitch).astype(int)
        platform_steps = math.ceil((self.platform_width / 2 + side / 2 + gap) / pitch)
        platform_half = platform_steps * pitch - side / 2 - gap
        height_variation = self.stone_height_variation * difficulty
        green = (0.25, 0.80, 0.45)

        borders = make_border(
            body, self.size, tuple(2 * inner_half), self.stone_height + self.floor_depth,
            (*center, (self.stone_height - self.floor_depth) / 2),
        )
        geometries = [TerrainGeometry(geom=geom, color=darken_rgba(brand_ramp(green, 0.0), 0.85)) for geom in borders]
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=(*center, 0.05),
            pos=(*center, -self.floor_depth - 0.05),
        )
        geometries.append(TerrainGeometry(geom=floor, color=(0.1, 0.1, 0.1, 1.0)))
        platform = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(platform_half, platform_half, (self.stone_height + self.floor_depth) / 2),
            pos=(*center, (self.stone_height - self.floor_depth) / 2),
        )
        geometries.append(TerrainGeometry(geom=platform, color=brand_ramp(green, 0.5)))

        for ix in range(-grid_half_counts[0], grid_half_counts[0] + 1):
            for iy in range(-grid_half_counts[1], grid_half_counts[1] + 1):
                if abs(ix) < platform_steps and abs(iy) < platform_steps:
                    continue
                # 只在7.5米桩区边界裁切箱体，不重算内侧节距或清洗数值。
                xy = np.asarray((ix, iy)) * pitch
                lower = np.maximum(xy - side / 2, -inner_half)
                upper = np.minimum(xy + side / 2, inner_half)
                height = self.floor_depth + self.stone_height + rng.uniform(-height_variation, height_variation)
                stone = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX, size=(*(upper - lower) / 2, height / 2),
                    pos=(*(center + (lower + upper) / 2), -self.floor_depth + height / 2),
                )
                geometries.append(TerrainGeometry(geom=stone, color=brand_ramp(green, rng.uniform(0.4, 0.7))))
        return TerrainOutput(origin=np.array([*center, self.stone_height]), geometries=geometries)
