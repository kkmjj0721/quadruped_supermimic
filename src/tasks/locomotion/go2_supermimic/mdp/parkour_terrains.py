# Copyright (c) 2026-2027 zh
"""内容：装配预打包 Parkour 碰撞地形，查询精确表面、可微高程与有限圆柱锐边。"""

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


# MuJoCo 独立棱柱编译曾产生最大 2.2 nm 微缝；实体 XY 公差带上限为 1 µm。
PRISM_MAX_XY_EXPANSION_M = 1e-6
PRISM_COMPILE_ROUNDING_M = 2e-8
PROJECT_ROOT = Path(__file__).resolve().parents[5]


@lru_cache(maxsize=4)
def terrain_asset(asset_path):
    """读取预打包资产；同一次装配的各地块共享不可变数组。"""
    with np.load(PROJECT_ROOT / asset_path) as archive:
        return dict(archive)


@dataclass(kw_only=True)
class ProofColumnCfg(SubTerrainCfg):
    asset_path: str
    column: int
    num_rows: int

    def function(self, difficulty, spec, rng):
        # MjLab 课程的各行采用均匀 difficulty；资产保存了上游实际采样难度。
        row = round(difficulty * (self.num_rows - 1))
        key = f"r{row}c{self.column}"
        asset = terrain_asset(self.asset_path)
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
                # MuJoCo 要求高度场底层为正厚度；单面网格下方补 1 cm 实体。
                field = spec.add_hfield(name=field_name, nrow=heights.shape[0], ncol=heights.shape[1],
                                        size=(*half, top - low, max(low - bottom, 0.01)),
                                        userdata=((heights - low) / (top - low)).ravel())
                geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field_name, pos=(*center, low))
                geometries.append(TerrainGeometry(geom=geom, hfield=field))
        triangles = asset[key + "t"]
        for index, (triangle, bottom) in enumerate(zip(triangles, asset[key + "b"], strict=True)):
            # 三顶点及其正下方底点构成单个凸棱柱，禁止整片非凸地形取凸包。
            xy = triangle[:, :2]
            lengths = np.linalg.norm(xy[[1, 2, 0]] - xy[[2, 0, 1]], axis=-1)
            incenter = (xy * lengths[:, None]).sum(axis=0) / lengths.sum()
            offsets = xy - incenter
            # 预留实测编译舍入裕量，实际实体顶点仍须满足 1 µm 上限。
            displacement = offsets * ((PRISM_MAX_XY_EXPANSION_M - PRISM_COMPILE_ROUNDING_M)
                                      / np.linalg.norm(offsets, axis=-1).max())
            # 沿原顶面延拓高度，实体补缝与后续感知查询使用同一编译后几何。
            gradient = np.linalg.solve(xy[1:] - xy[0], triangle[1:, 2] - triangle[0, 2])
            expanded = np.column_stack((xy + displacement, triangle[:, 2] + displacement @ gradient))
            vertices = np.vstack((expanded, np.column_stack((expanded[:, :2], np.full(3, bottom)))))
            mesh_name = f"{key}t{index}"
            # 重基使输入顶点保持局部量级，减少独立 mesh 的单精度量化位移。
            spec.add_mesh(name=mesh_name, uservert=(vertices - triangle[0]).ravel())
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, pos=triangle[0])
            geometries.append(TerrainGeometry(geom=geom))
        return TerrainOutput(origin=asset["origins"][row, self.column], geometries=geometries)


def parkour_generator(cfg):
    """每列独立注册，地图布局完全由资产元数据决定。"""
    asset = terrain_asset(cfg.asset_path)
    names = tuple(asset["columns"].tolist())
    rows, cols = int(asset["num_rows"]), int(asset["num_cols"])
    generator = TerrainGeneratorCfg(
        seed=int(asset["seed"]), curriculum=True, num_rows=rows, num_cols=cols,
        size=tuple(asset["patch_size"]), border_width=float(asset["border_width"]), color_scheme="none",
        sub_terrains={
            f"{col}_{name}": ProofColumnCfg(asset_path=str(cfg.asset_path), column=col, num_rows=rows, proportion=1 / cols)
            for col, name in enumerate(names)
        },
    )
    return generator, names


class ParkourSurface(TerrainSurface):
    """精确碰撞前向与 2 cm 高程梯度代理共用实际 MuJoCo 碰撞模型。"""

    @classmethod
    def from_model(cls, model, device, terrain_map):
        self = cls()
        self.device = torch.device(device)
        self.res = terrain_map.cfg.field_resolution_m
        half = np.asarray((terrain_map.rows, terrain_map.cols)) * np.asarray(terrain_map.size) / 2
        self.origin = tuple(float(value) for value in -half - 2 * self.res)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        # 缓存标识延续现有高程缓存约定，由碰撞数据与扫描分辨率共同决定。
        identity = hashlib.sha256(b"proof_parkour_collision_v3")
        identity.update(np.float64(self.res).tobytes())
        identity.update(half.tobytes())
        pieces, samples, triangles = [], [np.zeros(4, dtype=np.float32)], []
        sample_offset = 4
        for geom_id in range(model.ngeom):
            body = model.body(model.geom_bodyid[geom_id]).name
            if body != "terrain" and not body.endswith("/terrain"):
                continue
            pos = data.geom_xpos[geom_id]
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[geom_id]
                start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                local = model.mesh_vert[start:start + count]
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                world = local @ rotation.T + pos
                identity.update(world.tobytes())
                # 装配契约规定前三个 mesh 顶点为原三角顶面；读取编译后的实际几何坐标。
                triangles.append(world[:3])
                continue
            box = model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
            field_id = model.geom_dataid[geom_id]
            size = model.geom_size[geom_id] if box else model.hfield_size[field_id]
            identity.update(np.int64(model.geom_type[geom_id]).tobytes())
            identity.update(pos.tobytes())
            identity.update(size.tobytes())
            bounds = (pos[:2] - size[:2]).tolist() + (pos[:2] + size[:2]).tolist()
            if box:
                pieces.append((bounds, pos[2] + size[2], 0, 2, 2))
            else:
                rows, cols = model.hfield_nrow[field_id], model.hfield_ncol[field_id]
                start = model.hfield_adr[field_id]
                z = model.hfield_data[start:start + rows * cols].reshape(rows, cols)
                identity.update(np.array([rows, cols], dtype=np.int64).tobytes())
                identity.update(z.tobytes())
                values = (pos[2] + size[2] * z).ravel().astype(np.float32)
                samples.append(values)
                pieces.append((bounds, pos[2] + size[2], sample_offset, rows, cols))
                sample_offset += rows * cols
        triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        asset = terrain_asset(terrain_map.cfg.asset_path)
        edges = asset["edges"]
        self.edge_radius = float(asset["edge_radius"])
        identity.update(edges.tobytes())
        identity.update(np.float64(self.edge_radius).tobytes())
        self.geometry_hash = identity.hexdigest()
        cache = Path(terrain_map.cfg.field_cache_dir) / f"heightfield_parkour_{self.geometry_hash[:12]}.pt"
        if cache.exists():
            self.h = torch.load(cache, map_location=self.device, weights_only=True)["h"]
        else:
            heights = self._rasterize(pieces, samples, triangles, half)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"h": heights, "origin": self.origin, "res": self.res}, cache)
            self.h = heights.to(self.device)
        self._build_surface_index(pieces, samples, half)
        self._build_triangle_index(triangles, half)
        self._build_edge_index(edges, half)
        return self

    def _rasterize(self, pieces, samples, triangles, half):
        """逐碰撞块填写扫描栅格；高度场使用 MuJoCo 的两三角面插值。"""
        xs, ys = [np.arange(round(2 * extent / self.res) + 5) * self.res + start
                  for extent, start in zip(half, self.origin, strict=True)]
        raster = np.full((len(ys), len(xs)), -np.inf, dtype=np.float32)
        values = np.concatenate(samples)
        for bounds, top, address, rows, cols in pieces:
            x0, x1 = np.searchsorted(xs, [bounds[0] - 1e-10, bounds[2] + 1e-10])
            y0, y1 = np.searchsorted(ys, [bounds[1] - 1e-10, bounds[3] + 1e-10])
            region = raster[y0:y1, x0:x1]
            if address == 0:
                np.maximum(region, top, out=region)
            else:
                z = values[address:address + rows * cols].reshape(rows, cols)
                gx = np.clip((xs[x0:x1] - bounds[0]) / (bounds[2] - bounds[0]), 0, 1) * (cols - 1)
                gy = np.clip((ys[y0:y1] - bounds[1]) / (bounds[3] - bounds[1]), 0, 1) * (rows - 1)
                ix, iy = np.minimum(gx.astype(int), cols - 2), np.minimum(gy.astype(int), rows - 2)[:, None]
                fx, fy = gx - ix, gy[:, None] - iy
                a, b, c, d = z[iy, ix], z[iy, ix + 1], z[iy + 1, ix], z[iy + 1, ix + 1]
                grid = np.where(fx >= fy, a + fx * (b - a) + fy * (d - b), a + fy * (c - a) + fx * (d - c))
                np.maximum(region, grid, out=region)
        for triangle in triangles:
            lower, upper = triangle[:, :2].min(axis=0), triangle[:, :2].max(axis=0)
            x0, x1 = np.searchsorted(xs, lower[0]), np.searchsorted(xs, upper[0], side="right")
            y0, y1 = np.searchsorted(ys, lower[1]), np.searchsorted(ys, upper[1], side="right")
            x, y = np.meshgrid(xs[x0:x1], ys[y0:y1])
            a, b, c = triangle
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            z = w0 * a[2] + w1 * b[2] + w2 * c[2]
            region = raster[y0:y1, x0:x1]
            np.maximum(region, np.where(inside, z, -np.inf), out=region)
        # 资产核心之外采用平面边框；保留内部缺失表面的非有限值以暴露资产错误。
        outside = (np.abs(ys[:, None]) > half[1]) | (np.abs(xs[None, :]) > half[0])
        raster[outside] = 0
        return torch.from_numpy(raster)

    def _build_surface_index(self, pieces, samples, half):
        self.surface_cell = 0.5
        self.surface_half = torch.as_tensor(half, dtype=torch.float32, device=self.device)
        shape = np.ceil(2 * half / self.surface_cell).astype(int)
        bounds, top, addresses, heights, widths = zip(*pieces, strict=True)
        boxes = np.asarray(bounds)
        candidates = {}
        for index in np.argsort(top)[::-1]:
            first = np.floor((boxes[index, :2] + half) / self.surface_cell).astype(int)
            last = np.floor((boxes[index, 2:] + half) / self.surface_cell).astype(int)
            for x in range(max(first[0], 0), min(last[0] + 1, shape[0])):
                for y in range(max(first[1], 0), min(last[1] + 1, shape[1])):
                    candidates.setdefault((x, y), []).append(index)
        # 第零项表示平面边框；每个空间格使用固定候选槽，适配 CUDA Graph。
        table = np.zeros((*shape, max(map(len, candidates.values()))), dtype=np.int32)
        for cell, indices in candidates.items():
            table[cell][:len(indices)] = np.asarray(indices) + 1
        self.surface_ids = torch.from_numpy(table).to(self.device)
        self.surface_counts = (self.surface_ids != 0).sum(-1)
        self.bounds = torch.tensor(np.vstack((np.array([0., 0., 1., 1.]), boxes)), dtype=torch.float64, device=self.device)
        self.flat_top = torch.tensor([0, *top], dtype=torch.float32, device=self.device)
        self.field_layout = torch.tensor([[0, 2, 2], *zip(addresses, heights, widths, strict=True)],
                                         dtype=torch.long, device=self.device)
        self.field_values = torch.from_numpy(np.concatenate(samples)).to(self.device)

    @torch.no_grad()
    def collision_height(self, xy):
        """按所在空间格查询实际碰撞高度，保留沟边与高台的真实位置。"""
        if self.device.type == "cuda":
            from .parkour_surface_kernel import parkour_collision_height_fp64

            return parkour_collision_height_fp64(
                xy, self.surface_half, self.surface_ids, self.surface_counts, self.bounds, self.flat_top,
                self.field_layout, self.field_values, self.triangle_ids, self.triangle_counts, self.triangle_origin,
                self.triangle_w0, self.triangle_w1, self.triangle_z, self.surface_cell, self.triangle_cell,
            )
        # 全局坐标绝对值可达 80 m；几何判定保留双精度，避免吞掉微米实体重叠。
        points = xy.to(dtype=torch.float64)
        rectangle_height, covered = self._rectangle_height(points)
        triangle_height = self._triangle_height(points)
        height = torch.maximum(rectangle_height, triangle_height)
        in_core = ((points >= -self.surface_half) & (points <= self.surface_half)).all(-1)
        return torch.where(in_core, height, torch.where(covered, rectangle_height, 0.)).to(dtype=xy.dtype)

    def _rectangle_height(self, xy):
        cell = ((xy + self.surface_half) / self.surface_cell).floor().long()
        cx, cy = cell.unbind(-1)
        ids = self.surface_ids[cx.clamp(0, self.surface_ids.shape[0] - 1),
                               cy.clamp(0, self.surface_ids.shape[1] - 1)]
        boxes = self.bounds.to(dtype=xy.dtype)[ids]
        point = xy.unsqueeze(-2)
        inside = (ids != 0) & (point >= boxes[..., :2]).all(-1) & (point <= boxes[..., 2:]).all(-1)
        addr, rows, cols = self.field_layout[ids].unbind(-1)
        uv = ((point - boxes[..., :2]) / (boxes[..., 2:] - boxes[..., :2])).clamp(0, 1)
        gx, gy = uv[..., 0] * (cols - 1), uv[..., 1] * (rows - 1)
        ix, iy = torch.minimum(gx.long(), cols - 2), torch.minimum(gy.long(), rows - 2)
        fx, fy = gx - ix, gy - iy
        i = addr + iy * cols + ix
        a, b = self.field_values[i], self.field_values[i + 1]
        c, d = self.field_values[i + cols], self.field_values[i + cols + 1]
        tri = torch.where(fx >= fy, a + fx * (b - a) + fy * (d - b), a + fy * (c - a) + fx * (d - c))
        candidate = torch.where(addr == 0, self.flat_top[ids], tri)
        # 高度场叠加时比较当前位置的实际面高，不能只按各块最高点选首项。
        top = torch.where(inside, candidate, -torch.inf).amax(dim=-1)
        return top, inside.any(dim=-1)

    def _build_triangle_index(self, triangles, half):
        """精确棱柱顶面使用细空间格，避免扫描时广播全部三角面。"""
        self.triangle_cell = 0.1
        shape = np.ceil(2 * half / self.triangle_cell).astype(int)
        lower, upper = triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
        first = np.floor((lower + half) / self.triangle_cell).astype(int)
        last = np.floor((upper + half) / self.triangle_cell).astype(int)
        cells = {}
        for index, (lo, hi) in enumerate(zip(first, last, strict=True), start=1):
            for x in range(max(0, lo[0]), min(shape[0], hi[0] + 1)):
                for y in range(max(0, lo[1]), min(shape[1], hi[1] + 1)):
                    cells.setdefault((x, y), []).append(index)
        ids = np.zeros((*shape, max([len(values) for values in cells.values()] + [1])), dtype=np.int32)
        for cell, indices in cells.items():
            ids[cell][:len(indices)] = indices
        self.triangle_ids = torch.from_numpy(ids).to(self.device)
        self.triangle_counts = (self.triangle_ids != 0).sum(-1)
        # 空槽是有非零投影面积的占位三角形；用 id 掩码表达空集合。
        empty = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        values = torch.as_tensor(np.concatenate((empty, triangles)), dtype=torch.float64, device=self.device)
        a, b, c = values.unbind(-2)
        determinant = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        self.triangle_origin = c[:, :2]
        self.triangle_w0 = torch.stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]), -1) / determinant[:, None]
        self.triangle_w1 = torch.stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]), -1) / determinant[:, None]
        self.triangle_z = values[:, :, 2]

    def _triangle_height(self, xy):
        cell = ((xy + self.surface_half) / self.triangle_cell).floor().long()
        x, y = cell.unbind(-1)
        ids = self.triangle_ids[x.clamp(0, self.triangle_ids.shape[0] - 1), y.clamp(0, self.triangle_ids.shape[1] - 1)]
        offset = xy.unsqueeze(-2) - self.triangle_origin.to(dtype=xy.dtype)[ids]
        w0 = (offset * self.triangle_w0.to(dtype=xy.dtype)[ids]).sum(-1)
        w1 = (offset * self.triangle_w1.to(dtype=xy.dtype)[ids]).sum(-1)
        w2 = 1 - w0 - w1
        inside = (ids != 0) & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = self.triangle_z.to(dtype=xy.dtype)[ids]
        height = w0 * z[..., 0] + w1 * z[..., 1] + w2 * z[..., 2]
        return torch.where(inside, height, -torch.inf).amax(dim=-1)

    def height(self, xy):
        """精确前向；想象反传沿用扫描栅格的双线性位置导数。"""
        actual = self.collision_height(xy)
        if not xy.requires_grad:
            return actual
        grid = super().height(xy)
        return actual + (grid - grid.detach())

    def _build_edge_index(self, lines, half):
        self.edge_cell = 0.25
        half = half + self.edge_radius
        shape = np.ceil(2 * half / self.edge_cell).astype(int)
        self.edge_half = torch.tensor(half, dtype=torch.float32, device=self.device)
        cells = {}
        lower = np.floor((lines[..., :2].min(axis=1) - self.edge_radius + half) / self.edge_cell).astype(int)
        upper = np.floor((lines[..., :2].max(axis=1) + self.edge_radius + half) / self.edge_cell).astype(int)
        for index, (lo, hi) in enumerate(zip(lower, upper, strict=True), start=1):
            for x in range(max(0, lo[0]), min(shape[0], hi[0] + 1)):
                for y in range(max(0, lo[1]), min(shape[1], hi[1] + 1)):
                    cells.setdefault((x, y), []).append(index)
        # 无锐边的自定义布局仍有一个空槽；它是空集合的固定 shape 表示。
        ids = np.zeros((*shape, max([len(values) for values in cells.values()] + [1])), dtype=np.int32)
        for cell, indices in cells.items():
            ids[cell][:len(indices)] = indices
        self.edge_ids = torch.from_numpy(ids).to(self.device)
        segments = torch.as_tensor(lines, device=self.device)
        delta = segments[:, 1] - segments[:, 0]
        length = delta.norm(dim=-1, keepdim=True)
        empty = delta.new_tensor([[0., 0., 0., 1., 0., 0., -1.]])
        self.edge_data = torch.cat((empty, torch.cat((segments[:, 0], delta / length, length), dim=-1)))

    def edge_depth(self, points_w):
        """返回 LocoLab 最深有限圆柱穿透偏移的模长；段外和轴线上均为零。"""
        cell = ((points_w[..., :2] + self.edge_half) / self.edge_cell).floor().long()
        x, y = cell.unbind(-1)
        ids = self.edge_ids[x.clamp(0, self.edge_ids.shape[0] - 1), y.clamp(0, self.edge_ids.shape[1] - 1)]
        data = self.edge_data[ids]
        offset = points_w.unsqueeze(-2) - data[..., :3]
        t = (offset * data[..., 3:6]).sum(-1)
        distance = (offset - t.unsqueeze(-1) * data[..., 3:6]).norm(dim=-1)
        within_grid = ((points_w[..., :2] >= -self.edge_half) & (points_w[..., :2] < self.edge_half)).all(-1)
        depth = torch.where((t >= 0) & (t <= data[..., 6]) & within_grid.unsqueeze(-1),
                            (self.edge_radius - distance).relu(), 0.)
        maximum, index = depth.max(dim=-1, keepdim=True)
        radial = distance.gather(-1, index)
        # LocoLab utils/warp/kernels.py:87 原式 max(dist, 1e-8)；轴线偏移严格为零。
        return (maximum * radial / radial.clamp_min(1e-8)).squeeze(-1)
