# Copyright (c) 2026-2027 zh
"""
内容：
    复用MjLab梅花桩生成器，将零边框布局的中央出生平台固定为指定尺寸。
"""

from dataclasses import dataclass

from mjlab.terrains import BoxSteppingStonesTerrainCfg


@dataclass(kw_only=True)
class SteppingStonesTerrainCfg(BoxSteppingStonesTerrainCfg):
    """8米地块采用0.4米节距，保留原生桩体并固定2.5米出生平台。"""

    border_width: float = 0.0

    def function(self, difficulty, spec, rng):
        output = super().function(difficulty, spec, rng)
        # MjLab 1.6.0零边框输出依次为坑底、中央平台及桩体。
        # 六档原生平台半宽1.35→1.275米，与1.25米平台覆盖相同的网格桩中心。
        output.geometries[1].geom.size[:2] = (self.platform_width / 2, self.platform_width / 2)
        return output
