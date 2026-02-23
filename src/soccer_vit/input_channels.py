from __future__ import annotations

from typing import Any

import numpy as np

CHANNEL_ORDER = ["attacking", "defending", "ball", "passer", "receiver"]


def resolve_channel_selection(cfg: dict[str, Any], n_available: int = 5) -> tuple[list[int], list[str]]:
    """Resolve channel subset from config.

    Supported config shapes:
    - input.use_channels: [attacking, defending, ball, passer, receiver]
    - input.include_ball / input.include_passer / input.include_receiver (bool)
    Defaults to all available channels in CHANNEL_ORDER order.
    """
    input_cfg = cfg.get("input", {}) or {}
    available_names = CHANNEL_ORDER[:n_available]

    use_channels = input_cfg.get("use_channels")
    if use_channels:
        want = [str(x) for x in use_channels]
    else:
        include = {
            "ball": bool(input_cfg.get("include_ball", True)),
            "passer": bool(input_cfg.get("include_passer", True)),
            "receiver": bool(input_cfg.get("include_receiver", True)),
        }
        want = []
        for name in available_names:
            if name in include and not include[name]:
                continue
            want.append(name)

    idx = [available_names.index(name) for name in want if name in available_names]
    names = [available_names[i] for i in idx]
    if not idx:
        idx = list(range(n_available))
        names = available_names
    return idx, names


def select_image_channels(images: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    if images.ndim != 4:
        raise ValueError(f"Expected images as NCHW, got shape {images.shape}")
    idx, names = resolve_channel_selection(cfg, n_available=int(images.shape[1]))
    return images[:, idx, :, :], names

