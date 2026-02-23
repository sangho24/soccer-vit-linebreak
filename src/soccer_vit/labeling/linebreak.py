from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LineBreakParams:
    min_forward_m: float = 5.0
    corridor_w_m: float = 8.0
    k_bypassed: int = 2
    drop_non_forward: bool = False


@dataclass
class LineBreakResult:
    label: int
    bypassed_count: int
    forward_m: float
    pass_length_m: float
    min_def_dist_to_line_m: float | None
    dropped: bool = False


def _as_np(arr: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"Expected 1D point, got shape {out.shape}")
    return out


def segment_projection_stats(p0: np.ndarray, p1: np.ndarray, d: np.ndarray) -> tuple[float, float, float]:
    """Return (t, perpendicular distance, segment_length)."""
    p0 = _as_np(p0)
    p1 = _as_np(p1)
    d = _as_np(d)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-8:
        return 0.0, float(np.linalg.norm(d - p0)), 0.0
    t = float(np.dot(d - p0, v) / np.dot(v, v))
    proj = p0 + t * v
    perp = float(np.linalg.norm(d - proj))
    return t, perp, L


def bypassed_defenders(
    p0: np.ndarray,
    p1: np.ndarray,
    defenders_xy: np.ndarray,
    corridor_w_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean mask, projection ts, perp distances for bypassed defenders."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    D = np.asarray(defenders_xy, dtype=float)
    if D.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,)), np.zeros((0,))

    v = p1 - p0
    vv = float(np.dot(v, v))
    if vv <= 1e-8:
        n = len(D)
        return np.zeros((n,), dtype=bool), np.zeros((n,)), np.linalg.norm(D - p0[None, :], axis=1)

    rel = D - p0[None, :]
    t = (rel @ v) / vv
    proj = p0[None, :] + t[:, None] * v[None, :]
    perp = np.linalg.norm(D - proj, axis=1)

    x0, x1 = float(p0[0]), float(p1[0])
    x_lo, x_hi = min(x0, x1), max(x0, x1)
    mask = (t > 0.0) & (t < 1.0) & (perp <= corridor_w_m) & (D[:, 0] > x_lo) & (D[:, 0] < x_hi)
    return mask, t, perp


def label_line_break(
    p0: np.ndarray,
    p1: np.ndarray,
    defenders_xy: np.ndarray,
    params: LineBreakParams = LineBreakParams(),
) -> LineBreakResult:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    forward_m = float(p1[0] - p0[0])
    pass_length_m = float(np.linalg.norm(p1 - p0))

    if forward_m < params.min_forward_m:
        return LineBreakResult(
            label=0,
            bypassed_count=0,
            forward_m=forward_m,
            pass_length_m=pass_length_m,
            min_def_dist_to_line_m=None,
            dropped=params.drop_non_forward,
        )

    mask, _, perp = bypassed_defenders(p0, p1, defenders_xy, corridor_w_m=params.corridor_w_m)
    bypassed = int(mask.sum())
    min_dist = float(np.min(perp)) if len(perp) else None
    label = int(bypassed >= params.k_bypassed)
    return LineBreakResult(
        label=label,
        bypassed_count=bypassed,
        forward_m=forward_m,
        pass_length_m=pass_length_m,
        min_def_dist_to_line_m=min_dist,
        dropped=False,
    )


def compute_baseline_features(
    p0: np.ndarray,
    p1: np.ndarray,
    defenders_xy: np.ndarray,
    params: LineBreakParams,
) -> dict[str, float]:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    pass_length = float(np.linalg.norm(v))
    angle = float(np.arctan2(v[1], v[0])) if pass_length > 1e-8 else 0.0
    mask, _, perp = bypassed_defenders(p0, p1, defenders_xy, corridor_w_m=params.corridor_w_m)
    forward_m = float(v[0])
    return {
        "forward_m": forward_m,
        "pass_length_m": pass_length,
        "pass_angle_rad": angle,
        "corridor_def_count": float(mask.sum()),
        "min_def_dist_to_line_m": float(np.min(perp)) if len(perp) else params.corridor_w_m * 2.0,
    }
