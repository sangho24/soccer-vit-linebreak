from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(np.unique(y_true)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        out["auroc"] = float("nan")
    return out


def find_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_grid: int = 101,
    min_thr: float = 0.0,
    max_thr: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Grid-search thresholds on validation predictions for F1 / Balanced Accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {}
    if n_grid < 2:
        n_grid = 2
    grid = np.linspace(min_thr, max_thr, n_grid)
    best_f1: tuple[float, float] | None = None  # (score, thr)
    best_bal: tuple[float, float] | None = None
    for thr in grid:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        f1 = float(m.get("f1", np.nan))
        bal = float(m.get("balanced_accuracy", np.nan))
        if not np.isnan(f1):
            if best_f1 is None or f1 > best_f1[0]:
                best_f1 = (f1, float(thr))
        if not np.isnan(bal):
            if best_bal is None or bal > best_bal[0]:
                best_bal = (bal, float(thr))
    out: dict[str, dict[str, Any]] = {}
    if best_f1 is not None:
        thr = best_f1[1]
        out["best_f1"] = {"threshold": thr, "val_metrics": binary_metrics(y_true, y_prob, threshold=thr)}
    if best_bal is not None:
        thr = best_bal[1]
        out["best_balanced_accuracy"] = {"threshold": thr, "val_metrics": binary_metrics(y_true, y_prob, threshold=thr)}
    return out


def nan_to_none(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [nan_to_none(v) for v in x]
    return x
