# Copyright (c) 2026-2027 zh
"""内容：官方双层 Perlin 粗糙度采样与网格插值。

来源：LocoLab@37e567d，许可证见 LICENSE_LocoLab。
"""

from __future__ import annotations

import numpy as np


def _sample_perlin_noise(shape, cell_size):
    x, y = np.meshgrid(np.arange(shape[0]) / cell_size, np.arange(shape[1]) / cell_size, indexing="ij")
    x_cell = np.floor(x).astype(np.int32)
    y_cell = np.floor(y).astype(np.int32)
    x -= x_cell
    y -= y_cell
    angles = np.random.uniform(0.0, 2.0 * np.pi, size=(x_cell.max() + 2, y_cell.max() + 2))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def dot_grid(offset_x, offset_y):
        gradient = gradients[x_cell + offset_x, y_cell + offset_y]
        return (x - offset_x) * gradient[..., 0] + (y - offset_y) * gradient[..., 1]

    fade_x = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
    fade_y = 6.0 * y**5 - 15.0 * y**4 + 10.0 * y**3
    noise_x0 = dot_grid(0, 0) * (1.0 - fade_x) + dot_grid(1, 0) * fade_x
    noise_x1 = dot_grid(0, 1) * (1.0 - fade_x) + dot_grid(1, 1) * fade_x
    return np.sqrt(2.0) * (noise_x0 * (1.0 - fade_y) + noise_x1 * fade_y) * 0.5 + 0.5


def _sample_fractal_perlin_noise(shape, horizontal_scale, downsampled_scale):
    noise = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    amplitude_sum = 0.0
    cell_size = 2.0 * downsampled_scale / horizontal_scale
    for _ in range(2):
        noise += amplitude * _sample_perlin_noise(shape, cell_size)
        amplitude_sum += amplitude
        amplitude *= 0.25
        cell_size /= 2.0
    noise -= noise.mean()
    return np.clip(noise / (0.5 * amplitude_sum), -1.0, 1.0)


def generate_roughness_height_field(cfg, difficulty, shape):
    width_pixels, length_pixels = shape
    strength = roughness_strength(cfg, difficulty)
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)
    if strength == 0.0:
        return np.zeros((width_pixels, length_pixels), dtype=np.int16)
    downsampled_scale = cfg.downsampled_scale * np.exp2(np.random.random())
    noise = _sample_fractal_perlin_noise((width_pixels, length_pixels), cfg.horizontal_scale, downsampled_scale)
    noise = np.where(noise < 0.0, noise * abs(min(height_min, 0)), noise * max(height_max, 0))
    noise *= strength
    noise = np.rint(noise / height_step) * height_step
    scaled_min = int(np.rint(min(height_min * strength, height_max * strength) / height_step)) * height_step
    scaled_max = int(np.rint(max(height_min * strength, height_max * strength) / height_step)) * height_step
    return np.clip(noise, scaled_min, scaled_max).astype(np.int16)


def sample_height_field(height_field, x, y, horizontal_scale):
    rows, cols = height_field.shape
    fx = np.clip(np.asarray(x, dtype=np.float64) / horizontal_scale, 0.0, rows - 1.0)
    fy = np.clip(np.asarray(y, dtype=np.float64) / horizontal_scale, 0.0, cols - 1.0)
    x0 = np.floor(fx).astype(np.intp)
    y0 = np.floor(fy).astype(np.intp)
    x1 = np.minimum(x0 + 1, rows - 1)
    y1 = np.minimum(y0 + 1, cols - 1)
    tx = fx - x0
    ty = fy - y0
    v00 = height_field[x0, y0]
    v10 = height_field[x1, y0]
    v01 = height_field[x0, y1]
    v11 = height_field[x1, y1]
    return v00 * (1.0 - tx) * (1.0 - ty) + v10 * tx * (1.0 - ty) + v01 * (1.0 - tx) * ty + v11 * tx * ty


def roughness_strength(cfg, difficulty):
    if cfg.roughness_type == "difficulty":
        return float(np.clip(difficulty, 0.0, 1.0))
    if cfg.roughness_type == "random":
        return float(np.random.choice(cfg.roughness_strengths))
    return {"fixed": 1.0}[cfg.roughness_type]


def should_apply_roughness(cfg):
    return cfg.apply_roughness >= 1.0 or (cfg.apply_roughness > 0.0 and np.random.random() < cfg.apply_roughness)


def _finalize_height_field(cfg, hf_raw, difficulty):
    if should_apply_roughness(cfg):
        hf_raw += generate_roughness_height_field(cfg, difficulty, hf_raw.shape)
    return hf_raw
