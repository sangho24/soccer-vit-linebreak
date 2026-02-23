from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AttentionCapture:
    matrices: list[np.ndarray]


class ViTAttentionExtractor:
    """Captures per-block attention matrices from timm ViT models via forward hooks.

    Works on many timm ViT variants where blocks contain `attn` modules returning internal attention
    through `attn_map` or by exposing a `forward` path with `get_attention_map`. When unavailable,
    this extractor falls back to empty capture.
    """

    def __init__(self, model: Any):
        self.model = model
        self.hooks = []
        self.mats: list[np.ndarray] = []

    def _hook(self, module, _inputs, output):  # pragma: no cover - torch/timm dependent
        attn = output
        if attn is None:
            return
        try:
            arr = attn.detach().cpu().numpy()
        except Exception:
            return
        if arr.ndim in (3, 4):
            self.mats.append(arr)

    def __enter__(self):  # pragma: no cover - torch/timm dependent
        blocks = getattr(self.model, "blocks", [])
        for b in blocks:
            attn_mod = getattr(b, "attn", None)
            if attn_mod is not None:
                # timm ViT Attention typically passes attention probs through attn_drop before value aggregation.
                # Hooking attn_drop yields [B, H, T, T] attention matrices directly.
                target = getattr(attn_mod, "attn_drop", None) or attn_mod
                if hasattr(attn_mod, "fused_attn"):
                    try:
                        attn_mod.fused_attn = False
                    except Exception:
                        pass
                self.hooks.append(target.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - torch/timm dependent
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def capture(self) -> AttentionCapture:
        return AttentionCapture(matrices=list(self.mats))


def attention_rollout(attn_matrices: list[np.ndarray], alpha: float = 0.9) -> np.ndarray:
    """Compute residual-aware attention rollout for one sample.

    Each matrix expected shape: [B, H, T, T] or [H, T, T]. Uses head-mean and B=0.
    Returns rollout matrix [T, T].
    """
    if not attn_matrices:
        return np.eye(1, dtype=np.float32)

    R = None
    for A in attn_matrices:
        if A.ndim == 4:
            A0 = A[0].mean(axis=0)
        elif A.ndim == 3:
            A0 = A.mean(axis=0)
        else:
            continue
        T = A0.shape[-1]
        I = np.eye(T, dtype=np.float32)
        A_res = alpha * A0.astype(np.float32) + (1.0 - alpha) * I
        denom = A_res.sum(axis=-1, keepdims=True) + 1e-8
        A_res = A_res / denom
        R = A_res if R is None else (A_res @ R)
    if R is None:
        return np.eye(1, dtype=np.float32)
    return R


def cls_to_patch_heatmap(rollout: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    h, w = grid_hw
    if rollout.shape[0] <= 1:
        return np.zeros((h, w), dtype=np.float32)
    cls_attn = rollout[0, 1 : 1 + (h * w)]
    if cls_attn.size < h * w:
        padded = np.zeros((h * w,), dtype=np.float32)
        padded[: cls_attn.size] = cls_attn
        cls_attn = padded
    heat = cls_attn.reshape(h, w)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat.astype(np.float32)


def attention_distance(attn_matrices: list[np.ndarray], patch_size_m: float = 1.0) -> dict[str, float]:
    """Aggregate attention-weighted token distance.

    Approximates distance on token grid (excluding CLS) and reports mean across layers/heads.
    """
    vals: list[float] = []
    for A in attn_matrices:
        if A.ndim == 4:
            A_use = A[0]
        elif A.ndim == 3:
            A_use = A
        else:
            continue
        H, T, _ = A_use.shape
        n = T - 1 if T > 1 else T
        if n <= 0:
            continue
        side = int(np.sqrt(n))
        if side * side != n:
            continue
        coords = np.stack(np.mgrid[0:side, 0:side], axis=-1).reshape(-1, 2).astype(np.float32)
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1) * patch_size_m
        for hidx in range(H):
            Ah = A_use[hidx]
            A_patch = Ah[1:, 1:] if T > 1 else Ah
            den = float(A_patch.sum()) + 1e-8
            vals.append(float((A_patch * D).sum() / den))
    if not vals:
        return {"attention_distance_mean": float("nan"), "attention_distance_std": float("nan")}
    arr = np.asarray(vals, dtype=float)
    return {
        "attention_distance_mean": float(np.nanmean(arr)),
        "attention_distance_std": float(np.nanstd(arr)),
    }
