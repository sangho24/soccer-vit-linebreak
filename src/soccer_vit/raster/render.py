from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RasterSpec:
    size: int = 224
    sigma_m: float = 1.2
    line_sigma_m: float = 0.8
    corridor_w_m: float = 8.0
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


def _avg_px_per_m(spec: RasterSpec) -> float:
    px_per_m_x = (spec.size - 1) / spec.pitch_length_m
    px_per_m_y = (spec.size - 1) / spec.pitch_width_m
    return float((px_per_m_x + px_per_m_y) / 2.0)


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


def draw_pass_geometry(
    line_canvas: np.ndarray,
    corridor_canvas: np.ndarray | None,
    p0_px: tuple[float, float],
    p1_px: tuple[float, float],
    line_sigma_px: float,
    corridor_w_px: float,
) -> None:
    """Rasterize pass line and corridor in pixel space."""
    h, w = line_canvas.shape
    x0, y0 = float(p0_px[0]), float(p0_px[1])
    x1, y1 = float(p1_px[0]), float(p1_px[1])
    v = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    vv = float(np.dot(v, v))
    if vv <= 1e-8:
        draw_gaussian(line_canvas, x0, y0, max(1.0, line_sigma_px))
        if corridor_canvas is not None:
            draw_gaussian(corridor_canvas, x0, y0, max(1.0, corridor_w_px / 2.0))
        return

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    relx = xx - x0
    rely = yy - y0
    t = (relx * v[0] + rely * v[1]) / vv
    t_clip = np.clip(t, 0.0, 1.0)
    projx = x0 + t_clip * v[0]
    projy = y0 + t_clip * v[1]
    dist = np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)
    on_segment = (t >= 0.0) & (t <= 1.0)

    line = np.exp(-(dist**2) / (2.0 * max(1e-6, line_sigma_px) ** 2)).astype(np.float32)
    line *= on_segment.astype(np.float32)
    np.maximum(line_canvas, line, out=line_canvas)

    if corridor_canvas is not None:
        corr = ((dist <= corridor_w_px) & on_segment).astype(np.float32)
        np.maximum(corridor_canvas, corr, out=corridor_canvas)


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
    px_per_m = _avg_px_per_m(spec)

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
    if spec.channels >= 6:
        p0_px = meters_to_pixel(passer_xy[0], passer_xy[1], spec)
        p1_px = meters_to_pixel(receiver_xy[0], receiver_xy[1], spec)
        line_sigma_px = max(1.0, float(spec.line_sigma_m * px_per_m))
        corridor_w_px = max(1.0, float(spec.corridor_w_m * px_per_m))
        corridor_canvas = img[6] if spec.channels >= 7 else None
        draw_pass_geometry(
            line_canvas=img[5],
            corridor_canvas=corridor_canvas,
            p0_px=p0_px,
            p1_px=p1_px,
            line_sigma_px=line_sigma_px,
            corridor_w_px=corridor_w_px,
        )

    return np.clip(img, 0.0, 1.0)
