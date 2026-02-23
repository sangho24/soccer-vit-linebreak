from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class BaselineModel:
    feature_names: list[str]
    estimator: LogisticRegression

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


def fit_logistic_baseline(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 1000,
    class_weight: str | dict[str, float] | None = "balanced",
) -> BaselineModel:
    est = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    est.fit(X, y)
    return BaselineModel(feature_names=[], estimator=est)


def save_baseline(model: BaselineModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_baseline(path: str | Path) -> BaselineModel:
    with open(path, "rb") as f:
        return pickle.load(f)
