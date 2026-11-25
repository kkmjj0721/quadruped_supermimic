# Copyright (c) 2026-2027 zh
"""内容：将 PROOF 固定种子的二阶段地块装配为 MuJoCo 碰撞体，并查询实际表面与虚拟锐边。"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from .terrains import TerrainSurface


ASSET_DIR = Path(__file__).resolve().parents[4] / "assets/terrains"


@lru_cache(maxsize=1)
def terrain_asset(seed):
    """按种子读取上游地块资产；训练与播放复用同一份布局。"""
    with np.load(ASSET_DIR / f"proof_parkour_{seed}.npz") as archive:
        return dict(archive)


@dataclass(kw_only=True)
class ProofColumnCfg(SubTerrainCfg):
    seed: int
    column: int

    def function(self, difficulty, spec, rng):
        # MjLab 遍历同列十行时传入 0/9 至 9/9；各行实际难度已由上游扰动决定。
        row = round(difficulty * 9)
        key = f"r{row}c{self.column}"
        asset = terrain_asset(self.seed)
        body = spec.body("terrain")
        geometries = []
        for x0, y0, x1, y1, bottom, top, field_id in np.vstack((asset[key], asset[key + "s"])):
            center = ((x0 + x1) / 2, (y0 + y1) / 2)
            half = ((x1 - x0) / 2, (y1 - y0) / 2)
            if field_id < 0:
                geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                     pos=(*center, (top + bottom) / 2), size=(*half, (top - bottom) / 2))
                geometries.append(TerrainGeometry(geom=geom))
            else:
                field_name = f"{key}f{int(field_id)}"
                heights = asset[field_name]
                low = float(heights.min())
                # MuJoCo 要求高度场底层厚度为正。单面地形在表面下方补 1cm 实体。
                field = spec.add_hfield(name=field_name, nrow=heights.shape[0], ncol=heights.shape[1],
                                        size=(*half, top - low, max(low - bottom, 0.01)),
                                        userdata=((heights - low) / (top - low)).ravel())
                geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field_name, pos=(*center, low))
                geometries.append(TerrainGeometry(geom=geom, hfield=field))
        return TerrainOutput(origin=asset["origins"][row, self.column], geometries=geometries)


def parkour_generator(cfg):
    """每列注册一个子配置，以保留作者按比例重复采样的二十条独立赛道。"""
    names = tuple(terrain_asset(cfg.seed)["columns"].tolist())
    generator = TerrainGeneratorCfg(
        seed=cfg.seed, curriculum=True, num_rows=10, num_cols=len(names), size=(8., 8.), border_width=20.,
        color_scheme="none", sub_terrains={
            f"{col}_{name}": ProofColumnCfg(seed=cfg.seed, column=col, proportion=1 / len(names))
            for col, name in enumerate(names)
        },
    )
    return generator, names


class ParkourSurface(TerrainSurface):
    """在编译好的 MuJoCo 拼块上重建扫描栅格；锐边查询使用同一资产世界坐标。"""

    @classmethod
    def from_model(cls, model, device, terrain_map):
        self = cls()
        self.device = torch.device(device)
        self.res = terrain_map.cfg.field_resolution_m
        half = np.asarray((terrain_map.rows, terrain_map.cols)) * np.asarray(terrain_map.size) / 2
        # 额外两格覆盖 z=0 边框；更远处为同一平面。
        self.origin = tuple(-half - 2 * self.res)
        xs, ys = [np.arange(round(2 * extent / self.res) + 5) * self.res + start
                  for extent, start in zip(half, self.origin, strict=True)]
        raster = np.full((len(ys), len(xs)), -np.inf, dtype=np.float32)
        # 静态无关节地形在世界系的位置由 MjData 的刚体变换决定。
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        # 播放任务规格须绑定到实际碰撞体、查询尺度及边缘。
        identity = hashlib.sha256(b"proof_parkour_collision_v1")
        identity.update(np.float64(self.res).tobytes())
        pieces, samples = [], [np.zeros(4, dtype=np.float32)]
        sample_offset = 4
        for geom_id in range(model.ngeom):
            body = model.body(model.geom_bodyid[geom_id]).name
            if body != "terrain" and not body.endswith("/terrain"):
                continue
            pos = data.geom_xpos[geom_id]
            box = model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
            field_id = model.geom_dataid[geom_id]
            size = model.geom_size[geom_id] if box else model.hfield_size[field_id]
            identity.update(np.int64(model.geom_type[geom_id]).tobytes())
            identity.update(pos.tobytes())
            identity.update(size.tobytes())
            x0, x1 = np.searchsorted(xs, [pos[0] - size[0] - 1e-10, pos[0] + size[0] + 1e-10])
            y0, y1 = np.searchsorted(ys, [pos[1] - size[1] - 1e-10, pos[1] + size[1] + 1e-10])
            region = raster[y0:y1, x0:x1]
            bounds = (pos[:2] - size[:2]).tolist() + (pos[:2] + size[:2]).tolist()
            if box:
                height = pos[2] + size[2]
                pieces.append((bounds, height, 0, 2, 2))
                np.maximum(region, height, out=region)
            else:
                rows, cols = model.hfield_nrow[field_id], model.hfield_ncol[field_id]
                start = model.hfield_adr[field_id]
                z = model.hfield_data[start:start + rows * cols].reshape(rows, cols)
                identity.update(np.array([rows, cols], dtype=np.int64).tobytes())
                identity.update(z.tobytes())
                samples.append((pos[2] + size[2] * z).ravel().astype(np.float32))
                pieces.append((bounds, pos[2] + size[2], sample_offset, rows, cols))
                sample_offset += rows * cols
                gx = np.clip((xs[x0:x1] - pos[0] + size[0]) / (2 * size[0]), 0, 1) * (cols - 1)
                gy = np.clip((ys[y0:y1] - pos[1] + size[1]) / (2 * size[1]), 0, 1) * (rows - 1)
                ix, iy = np.minimum(gx.astype(int), cols - 2), np.minimum(gy.astype(int), rows - 2)[:, None]
                fx, fy = gx - ix, gy[:, None] - iy
                a, b, c, d = z[iy, ix], z[iy, ix + 1], z[iy + 1, ix], z[iy + 1, ix + 1]
                grid = np.where(fx >= fy, a + fx * (b - a) + fy * (d - b), a + fy * (c - a) + fx * (d - c))
                np.maximum(region, pos[2] + size[2] * grid, out=region)

        self.h = torch.from_numpy(raster).to(device)
        self._build_surface_index(pieces, samples, half)
        edges = terrain_asset(terrain_map.cfg.seed)["edges"]
        identity.update(edges.tobytes())
        self.geometry_hash = identity.hexdigest()
        self._build_edge_index(edges, half)
        return self

    def _build_surface_index(self, pieces, samples, half):
        # 地形只有叠在地面上的墙／岛和相邻踏面；同一 XY 的表面按层高排序。
        self.surface_cell = 0.25
        self.surface_half = torch.as_tensor(half, dtype=torch.float32, device=self.device)
        shape = np.rint(2 * half / self.surface_cell).astype(int)
        bounds, top, addresses, heights, widths = zip(*pieces, strict=True)
        boxes = np.asarray(bounds)
        # 源资产的矩形可以在边界线共面接触；正面积重叠的地块依最大高度排序。
        candidate = {}
        for idx in np.argsort(top)[::-1]:
            first = np.floor((boxes[idx, :2] + half) / self.surface_cell).astype(int)
            last = np.floor((boxes[idx, 2:] + half) / self.surface_cell).astype(int)
            for x in range(max(first[0], 0), min(last[0] + 1, shape[0])):
                for y in range(max(first[1], 0), min(last[1] + 1, shape[1])):
                    candidate.setdefault((x, y), []).append(idx)
        # 首项为 z=0 的外部边框，未使用的格槽也引用这一项。
        table = np.zeros((*shape, max(map(len, candidate.values()))), dtype=np.int32)
        for cell, indices in candidate.items():
            table[cell][:len(indices)] = np.asarray(indices) + 1
        self.surface_ids = torch.from_numpy(table).to(self.device)
        bounds_with_border = np.vstack((np.array([0., 0., 1., 1.]), boxes))
        self.bounds = torch.tensor(bounds_with_border, dtype=torch.float32, device=self.device)
        self.flat_top = torch.tensor([0, *top], dtype=torch.float32, device=self.device)
        self.field_layout = torch.tensor([[0, 2, 2], *zip(addresses, heights, widths, strict=True)],
                                         dtype=torch.long, device=self.device)
        self.field_values = torch.from_numpy(np.concatenate(samples)).to(self.device)

    @torch.no_grad()
    def collision_height(self, xy):
        """按每点所在空间格的真实矩形碰撞范围和高度场三角面计算竖直射线高度。"""
        cell = ((xy + self.surface_half) / self.surface_cell).floor().long()
        cx, cy = cell.unbind(-1)
        ids = self.surface_ids[cx.clamp(0, self.surface_ids.shape[0] - 1),
                               cy.clamp(0, self.surface_ids.shape[1] - 1)]
        boxes = self.bounds[ids]
        point = xy.unsqueeze(-2)
        inside = (ids != 0) & (point >= boxes[..., :2]).all(-1) & (point <= boxes[..., 2:]).all(-1)
        # 候选按最高点降序，首个覆盖的层就是上表面；空格和整张地图外属于 z=0 边框。
        first = inside.int().argmax(dim=-1, keepdim=True)
        piece = torch.where(inside.any(dim=-1), ids.gather(-1, first).squeeze(-1), 0)
        rect = self.bounds[piece]
        addr, rows, cols = self.field_layout[piece].unbind(-1)
        uv = ((xy - rect[..., :2]) / (rect[..., 2:] - rect[..., :2])).clamp(0, 1)
        gx, gy = uv[..., 0] * (cols - 1), uv[..., 1] * (rows - 1)
        ix, iy = torch.minimum(gx.long(), cols - 2), torch.minimum(gy.long(), rows - 2)
        fx, fy = gx - ix, gy - iy
        i = addr + iy * cols + ix
        a, b = self.field_values[i], self.field_values[i + 1]
        c, d = self.field_values[i + cols], self.field_values[i + cols + 1]
        tri = torch.where(fx >= fy, a + fx * (b - a) + fy * (d - b), a + fy * (c - a) + fx * (d - c))
        return torch.where(addr == 0, self.flat_top[piece], tri)

    def height(self, xy):
        """前向使用实际碰撞几何；想象的位移梯度沿用本项目 2 厘米栅格的双线性导数。"""
        actual = self.collision_height(xy)
        if not xy.requires_grad:
            return actual
        grid = super().height(xy)
        return actual + (grid - grid.detach())

    def _build_edge_index(self, lines, half):
        # 半径扩展包围盒登记到所有相交的二维格，运行时只访问点所在格内的候选。
        self.edge_cell = 0.25
        self.edge_radius = 0.05
        shape = np.ceil(2 * half / self.edge_cell).astype(int)
        self.edge_half = torch.tensor(half, dtype=torch.float32, device=self.device)
        cells = {}
        lower = np.floor((lines[..., :2].min(axis=1) - self.edge_radius + half) / self.edge_cell).astype(int)
        upper = np.floor((lines[..., :2].max(axis=1) + self.edge_radius + half) / self.edge_cell).astype(int)
        for index, (lo, hi) in enumerate(zip(lower, upper, strict=True), start=1):
            for x in range(max(0, lo[0]), min(shape[0], hi[0] + 1)):
                for y in range(max(0, lo[1]), min(shape[1], hi[1] + 1)):
                    cells.setdefault((x, y), []).append(index)
        ids = np.zeros((*shape, max(map(len, cells.values()))), dtype=np.int32)
        for cell, indices in cells.items():
            ids[cell][:len(indices)] = indices
        self.edge_ids = torch.from_numpy(ids).to(self.device)
        segments = torch.as_tensor(lines, device=self.device)
        delta = segments[:, 1] - segments[:, 0]
        length = delta.norm(dim=-1, keepdim=True)
        # 第零行是填充槽，负段长使它不属于任何查询圆柱。
        empty = delta.new_tensor([[0., 0., 0., 1., 0., 0., -1.]])
        self.edge_data = torch.cat((empty, torch.cat((segments[:, 0], delta / length, length), dim=-1)))

    def edge_depth(self, points_w):
        """LocoLab 有限圆柱的最深穿透偏移模长，两端投影以外无惩罚。"""
        cell = ((points_w[..., :2] + self.edge_half) / self.edge_cell).floor().long()
        x, y = cell.unbind(-1)
        ids = self.edge_ids[x.clamp(0, self.edge_ids.shape[0] - 1), y.clamp(0, self.edge_ids.shape[1] - 1)]
        data = self.edge_data[ids]
        offset = points_w.unsqueeze(-2) - data[..., :3]
        t = (offset * data[..., 3:6]).sum(-1)
        distance = (offset - t.unsqueeze(-1) * data[..., 3:6]).norm(dim=-1)
        depth = torch.where((t >= 0) & (t <= data[..., 6]), (self.edge_radius - distance).relu(), 0.)
        # 作者先按深度选圆柱，再计算偏移向量；轴线上偏移严格为零。
        maximum, index = depth.max(dim=-1, keepdim=True)
        radial = distance.gather(-1, index)
        return (maximum * radial / radial.clamp_min(1e-8)).squeeze(-1)
