from __future__ import annotations

import numpy as np


def flip_y_coordinates(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).copy()
    if pts.size == 0:
        return pts
    pts[:, 1] *= -1.0
    return pts


def jitter_coordinates(points_xy: np.ndarray, jitter_m: float, rng: np.random.Generator) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).copy()
    if pts.size == 0 or jitter_m <= 0:
        return pts
    pts += rng.uniform(-jitter_m, jitter_m, size=pts.shape)
    return pts


def augment_snapshot_coords(
    attacking_xy: np.ndarray,
    defending_xy: np.ndarray,
    ball_xy: tuple[float, float] | None,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    jitter_m: float = 0.5,
    y_flip_prob: float = 0.5,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()
    att = jitter_coordinates(attacking_xy, jitter_m, rng)
    deff = jitter_coordinates(defending_xy, jitter_m, rng)
    passer = np.asarray(passer_xy, dtype=float) + rng.uniform(-jitter_m, jitter_m, size=(2,))
    receiver = np.asarray(receiver_xy, dtype=float) + rng.uniform(-jitter_m, jitter_m, size=(2,))
    ball = None
    if ball_xy is not None:
        ball = tuple((np.asarray(ball_xy, dtype=float) + rng.uniform(-jitter_m, jitter_m, size=(2,))).tolist())

    if rng.random() < y_flip_prob:
        att[:, 1] *= -1 if att.size else 1
        deff[:, 1] *= -1 if deff.size else 1
        passer[1] *= -1
        receiver[1] *= -1
        if ball is not None:
            ball_arr = np.asarray(ball, dtype=float)
            ball_arr[1] *= -1
            ball = tuple(ball_arr.tolist())
    return att, deff, ball, tuple(passer.tolist()), tuple(receiver.tolist())


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k.astype(np.float32)


def _separable_blur(image_chw: np.ndarray, sigma: float) -> np.ndarray:
    img = np.asarray(image_chw, dtype=np.float32)
    k = _gaussian_kernel1d(sigma)
    r = len(k) // 2
    out = img.copy()
    # Horizontal
    padded = np.pad(out, ((0, 0), (0, 0), (r, r)), mode="reflect")
    h = np.zeros_like(out)
    for i, kv in enumerate(k):
        h += kv * padded[:, :, i : i + out.shape[2]]
    # Vertical
    padded = np.pad(h, ((0, 0), (r, r), (0, 0)), mode="reflect")
    v = np.zeros_like(h)
    for i, kv in enumerate(k):
        v += kv * padded[:, i : i + h.shape[1], :]
    return v


def low_pass(image_chw: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(_separable_blur(image_chw, sigma), 0.0, 1.0)


def high_pass(image_chw: np.ndarray, sigma: float) -> np.ndarray:
    base = np.asarray(image_chw, dtype=np.float32)
    blur = _separable_blur(base, sigma)
    hp = base - blur
    hp_min = hp.min()
    hp_max = hp.max()
    if hp_max - hp_min < 1e-8:
        return np.zeros_like(base)
    return ((hp - hp_min) / (hp_max - hp_min)).astype(np.float32)
