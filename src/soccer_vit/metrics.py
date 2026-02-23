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


def nan_to_none(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [nan_to_none(v) for v in x]
    return x
