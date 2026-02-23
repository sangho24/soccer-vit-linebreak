from __future__ import annotations

import numpy as np


def grad_cam_placeholder(image_chw: np.ndarray) -> np.ndarray:
    """Optional Grad-CAM placeholder for environments without a full CNN explain stack."""
    img = np.asarray(image_chw, dtype=np.float32)
    if img.ndim != 3:
        raise ValueError("Expected CHW image")
    heat = img.max(axis=0)
    heat -= heat.min()
    if heat.max() > 0:
        heat /= heat.max()
    return heat
