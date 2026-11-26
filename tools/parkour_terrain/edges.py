# Copyright (c) 2026-2027 zh
"""内容：提取并拼接官方可行走顶面锐边；来源 LocoLab@37e567d，许可证见 LICENSE_LocoLab。"""

import random
from itertools import pairwise
from types import SimpleNamespace

import numpy as np


class EdgeDetector:
    def _upper_surface_edges(self, mesh):
        excluded = mesh.face_attributes["exclude_virtual_obstacle"]
        if excluded.any():
            mesh = mesh.copy()
            mesh.update_faces(~excluded)
            mesh.remove_unreferenced_vertices()
        adjacency = mesh.face_adjacency
        if len(adjacency) == 0:
            return np.empty((0, 6), dtype=np.float32)
        keep = mesh.face_adjacency_angles > np.deg2rad(self.angle_threshold)
        min_nz = self.cfg.min_upward_normal_z
        pair = np.arange(len(adjacency))
        face_nz = mesh.face_normals[:, 2][adjacency]
        up_side = np.argmax(face_nz, axis=1)
        up_face = adjacency[pair, up_side]
        drop_face = adjacency[pair, 1 - up_side]
        centers = mesh.triangles_center
        signed_distance = np.einsum("ij,ij->i", mesh.face_normals[up_face], centers[drop_face] - centers[up_face])
        keep &= (face_nz.max(axis=1) > min_nz) & (signed_distance < 0.0)
        if not np.any(keep):
            return np.empty((0, 6), dtype=np.float32)
        edges = mesh.face_adjacency_edges[keep]
        vertices = mesh.vertices
        coordinates = np.hstack((vertices[edges[:, 0]], vertices[edges[:, 1]]))
        return self._exposed_surface_edges(mesh, coordinates)

    @staticmethod
    def _exposed_surface_edges(mesh, edges):
        epsilon = 0.0001
        pieces = []
        tree = mesh.triangles_tree
        for edge in edges:
            start, end = (edge[:3], edge[3:])
            direction = end - start
            length = np.linalg.norm(direction)
            if np.linalg.norm(direction[:2]) < epsilon:
                continue
            bounds = np.r_[np.minimum(start, end) - epsilon, np.maximum(start, end) + epsilon]
            neighbors = list(tree.intersection(bounds))
            vertices = mesh.triangles[neighbors].reshape(-1, 3)
            fractions = np.clip((vertices - start) @ direction / length**2, 0.0, 1.0)
            fractions = np.unique(np.r_[0.0, fractions, 1.0])
            for left, right in pairwise(fractions):
                if (right - left) * length > epsilon:
                    pieces.append(np.r_[start + left * direction, start + right * direction])
        if not pieces:
            return np.empty((0, 6), dtype=np.float32)
        pieces = np.asarray(pieces)
        midpoints = (pieces[:, :3] + pieces[:, 3:]) * 0.5
        directions = pieces[:, 3:] - pieces[:, :3]
        sideways = np.column_stack((-directions[:, 1], directions[:, 0], np.zeros(len(pieces))))
        sideways *= epsilon / np.linalg.norm(sideways, axis=1, keepdims=True)
        origins = np.concatenate((midpoints + sideways, midpoints - sideways))
        origins[:, 2] -= epsilon
        ray_directions = np.zeros_like(origins)
        ray_directions[:, 2] = 1.0
        _, ray_ids, face_ids = mesh.ray.intersects_location(origins, ray_directions, multiple_hits=True)
        winding = np.zeros(len(origins), dtype=np.int32)
        np.add.at(winding, ray_ids, np.sign(mesh.face_normals[face_ids, 2]).astype(np.int32))
        inside = (winding > 0).reshape(2, -1)
        keep = inside[0] != inside[1]
        return pieces[keep]

    def _filter_edges_by_direction(self, edge_coords):
        max_z_ratio = self.cfg.max_edge_direction_z_ratio
        if edge_coords.size == 0:
            return edge_coords
        directions = edge_coords[:, 3:6] - edge_coords[:, :3]
        lengths = np.linalg.norm(directions, axis=1)
        valid = lengths > 1e-06
        z_ratio = np.ones_like(lengths)
        z_ratio[valid] = np.abs(directions[valid, 2]) / lengths[valid]
        return edge_coords[valid & (z_ratio <= max_z_ratio)]

    def process_edges(self, edge_coords):
        line_pts = edge_coords.reshape(-1, 3)
        V, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
        E_pairs = inv_idx.reshape(-1, 2)
        adj_list = {i: set() for i in range(V.shape[0])}
        for u, v in E_pairs:
            if u != v:
                adj_list[u].add(v)
                adj_list[v].add(u)
        num_edges_V = np.array([len(adj_list[i]) for i in range(V.shape[0])], dtype=int)
        available_edges = set(np.where(num_edges_V > 0)[0])
        cos_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))
        processed_edge_coords = []

        def compute_max_distance_to_line_vec(V, v_set):
            A = V[v_set[0]]
            B = V[v_set[-1]]
            AB = B - A
            norm_AB = np.linalg.norm(AB)
            if norm_AB == 0:
                return (v_set[0], 0.0)
            pts = V[v_set]
            dists = np.linalg.norm(np.cross(pts - A, pts - B), axis=1) / norm_AB
            max_idx = np.argmax(dists)
            return (v_set[max_idx], dists[max_idx])

        while available_edges:
            selected_vertex = random.choice(list(available_edges))
            v_set = [selected_vertex]
            if adj_list[selected_vertex]:
                neighbor = next(iter(adj_list[selected_vertex]))
                v_set.append(neighbor)
                adj_list[selected_vertex].remove(neighbor)
                adj_list[neighbor].remove(selected_vertex)
                for vid in [selected_vertex, neighbor]:
                    num_edges_V[vid] -= 1
                    if num_edges_V[vid] == 0:
                        available_edges.discard(vid)
            while True:
                find_neighbor = False
                start, end = (v_set[0], v_set[-1])
                neighbors = list(adj_list[start])
                if neighbors:
                    dirs = V[start] - V[neighbors]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    start_dir = V[v_set[1]] - V[start]
                    start_dir /= np.linalg.norm(start_dir)
                    dots = dirs @ start_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.insert(0, n)
                        adj_list[start].remove(n)
                        adj_list[n].remove(start)
                        for vid in [start, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True
                neighbors = list(adj_list[end])
                if neighbors:
                    dirs = V[neighbors] - V[end]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    end_dir = V[end] - V[v_set[-2]]
                    end_dir /= np.linalg.norm(end_dir)
                    dots = dirs @ end_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.append(n)
                        adj_list[end].remove(n)
                        adj_list[n].remove(end)
                        for vid in [end, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True
                if not find_neighbor:
                    break
            while len(v_set) >= self.cfg.min_points:
                for i in range(len(v_set) - 1):
                    _max_vi, max_dist = compute_max_distance_to_line_vec(V, v_set[i:])
                    if max_dist < 0.05:
                        break
                if len(v_set) - i >= self.cfg.min_points:
                    processed_edge_coords.append(np.concatenate([V[v_set[i]], V[v_set[-1]]]))
                v_set = v_set[: i + 1]
        return np.array(processed_edge_coords, dtype=np.float32)


def select_mesh_region(mesh, xyz_range, origin):
    triangles = mesh.triangles - origin
    lower, upper = np.asarray(xyz_range).T
    selected = ((triangles.max(axis=1) >= lower) & (triangles.min(axis=1) <= upper)).all(axis=1)
    result = mesh.copy()
    result.update_faces(selected)
    result.remove_unreferenced_vertices()
    return result


def detect_edges(mesh):
    detector = EdgeDetector()
    detector.angle_threshold = 70.0
    detector.cfg = SimpleNamespace(
        min_upward_normal_z=0.35, max_edge_direction_z_ratio=0.2, adjacent_angle_threshold=30.0, min_points=2
    )
    raw = detector._filter_edges_by_direction(detector._upper_surface_edges(mesh))
    return detector.process_edges(raw).reshape(-1, 2, 3)
