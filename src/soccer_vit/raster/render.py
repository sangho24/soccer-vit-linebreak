from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RasterSpec:
    size: int = 224
    sigma_m: float = 1.2
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    channels: int = 5


def meters_to_pixel(x_m: float, y_m: float, spec: RasterSpec) -> tuple[float, float]:
    x = (float(x_m) + spec.pitch_length_m / 2.0) / spec.pitch_length_m * (spec.size - 1)
    y = (spec.pitch_width_m / 2.0 - float(y_m)) / spec.pitch_width_m * (spec.size - 1)
    return x, y


def _sigma_px(spec: RasterSpec) -> float:
    px_per_m_x = (spec.size - 1) / spec.pitch_length_m
    px_per_m_y = (spec.size - 1) / spec.pitch_width_m
    return float(spec.sigma_m * (px_per_m_x + px_per_m_y) / 2.0)


def draw_gaussian(canvas: np.ndarray, x_px: float, y_px: float, sigma_px: float, amp: float = 1.0) -> None:
    h, w = canvas.shape
    if not np.isfinite(x_px) or not np.isfinite(y_px):
        return
    r = max(2, int(round(3 * sigma_px)))
    x0 = max(0, int(np.floor(x_px)) - r)
    x1 = min(w - 1, int(np.ceil(x_px)) + r)
    y0 = max(0, int(np.floor(y_px)) - r)
    y1 = min(h - 1, int(np.ceil(y_px)) + r)
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    g = np.exp(-((xx - x_px) ** 2 + (yy - y_px) ** 2) / (2.0 * sigma_px**2))
    canvas[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(canvas[y0 : y1 + 1, x0 : x1 + 1], amp * g)


def render_snapshot(
    attacking_xy: np.ndarray,
    defending_xy: np.ndarray,
    ball_xy: tuple[float, float] | None,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    spec: RasterSpec = RasterSpec(),
) -> np.ndarray:
    img = np.zeros((spec.channels, spec.size, spec.size), dtype=np.float32)
    sigma_px = _sigma_px(spec)

    def _draw_many(ch: int, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return
        for x_m, y_m in pts:
            x_px, y_px = meters_to_pixel(x_m, y_m, spec)
            draw_gaussian(img[ch], x_px, y_px, sigma_px)

    _draw_many(0, np.asarray(attacking_xy, dtype=float))
    _draw_many(1, np.asarray(defending_xy, dtype=float))

    if ball_xy is not None and spec.channels >= 3:
        x_px, y_px = meters_to_pixel(ball_xy[0], ball_xy[1], spec)
        draw_gaussian(img[2], x_px, y_px, sigma_px * 1.1, amp=1.0)
    if spec.channels >= 4:
        x_px, y_px = meters_to_pixel(passer_xy[0], passer_xy[1], spec)
        draw_gaussian(img[3], x_px, y_px, sigma_px * 1.2, amp=1.0)
    if spec.channels >= 5:
        x_px, y_px = meters_to_pixel(receiver_xy[0], receiver_xy[1], spec)
        draw_gaussian(img[4], x_px, y_px, sigma_px * 1.2, amp=1.0)

    return np.clip(img, 0.0, 1.0)
