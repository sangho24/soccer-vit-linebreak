from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data_pipeline import load_saved_dataset
from .input_channels import select_image_channels
from .labeling.linebreak import LineBreakParams, compute_baseline_features, label_line_break
from .metrics import binary_metrics, find_best_thresholds, nan_to_none
from .models.baselines import load_baseline
from .raster.augment import high_pass, low_pass
from .raster.render import RasterSpec, meters_to_pixel, render_snapshot
from .utils import ensure_dirs, load_yaml, save_json


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    p = cfg.get("paths", {})
    return {
        "processed": Path(p.get("processed_dir", "data/processed")),
        "reports": Path(p.get("reports_dir", "reports")),
        "figures": Path(p.get("figures_dir", "reports/figures")),
        "models": Path(p.get("models_dir", "reports/models")),
    }


def _load_splits(processed_dir: Path) -> dict[str, np.ndarray]:
    s = np.load(processed_dir / "splits.npz")
    return {k: s[k] for k in s.files}


def _subsample_eval_indices(idx: np.ndarray, y_all: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    y = y_all[idx]
    pos = idx[y == 1]
    neg = idx[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.sort(rng.choice(idx, size=max_n, replace=False))
    n_pos = max(1, int(round(max_n * (len(pos) / len(idx)))))
    n_pos = min(n_pos, len(pos), max_n - 1)
    n_neg = min(max_n - n_pos, len(neg))
    chosen = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ])
    if len(chosen) < max_n:
        remaining = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = rng.choice(remaining, size=min(max_n - len(chosen), len(remaining)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def _compute_threshold_tuning_summary(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test_by_condition: dict[str, np.ndarray],
    n_grid: int = 101,
) -> dict[str, Any]:
    best = find_best_thresholds(y_val, p_val, n_grid=n_grid)
    if not best:
        return {}
    out: dict[str, Any] = {"grid_size": int(n_grid)}
    for objective, info in best.items():
        thr = float(info["threshold"])
        cond_metrics = {}
        for cond, p_test in p_test_by_condition.items():
            cond_metrics[cond] = binary_metrics(y_test, p_test, threshold=thr)
        out[objective] = {
            "threshold": thr,
            "val_metrics": info["val_metrics"],
            "test_metrics": cond_metrics,
        }
    return out


def _predict_torch(model_name: str, images: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:  # pragma: no cover - torch dependent
    path = _paths(cfg)["models"] / f"{model_name}.pt"
    if not path.exists():
        return None
    try:
        import torch
        from .models.cnn import create_resnet18
        from .models.vit import ViTBuildConfig, create_vit_model
    except Exception:
        return None

    ckpt = torch.load(path, map_location="cpu")
    in_ch = int(images.shape[1])
    if model_name == "resnet18":
        model = create_resnet18(in_chans=in_ch, num_classes=2, pretrained=False)
    elif model_name == "vit_base":
        model = create_vit_model(ViTBuildConfig(model_name="vit_base_patch16_224", in_chans=in_ch, num_classes=2, pretrained=False))
    else:
        return None
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            xb = torch.tensor(images[i : i + 64], dtype=torch.float32)
            logits = model(xb)
            if logits.ndim == 2 and logits.shape[1] == 2:
                logits_pos = logits[:, 1] - logits[:, 0]
            else:
                logits_pos = logits.squeeze(-1)
            probs.append(torch.sigmoid(logits_pos).cpu().numpy())
    return np.concatenate(probs)


def _parse_points(json_str: str) -> np.ndarray:
    import json as _json

    arr = _json.loads(json_str) if isinstance(json_str, str) else []
    return np.asarray(arr, dtype=np.float32).reshape(-1, 2) if len(arr) else np.zeros((0, 2), dtype=np.float32)


def _counterfactual_for_row(meta_row, cfg: dict[str, Any]) -> dict[str, Any]:
    label_cfg = cfg.get("labeling", {})
    params = LineBreakParams(
        min_forward_m=float(label_cfg.get("min_forward_m", 5.0)),
        corridor_w_m=float(label_cfg.get("corridor_w_m", 8.0)),
        k_bypassed=int(label_cfg.get("k_bypassed", 2)),
        drop_non_forward=bool(label_cfg.get("drop_non_forward", False)),
    )
    p0 = np.array([meta_row.passer_x_m, meta_row.passer_y_m], dtype=float)
    p1 = np.array([meta_row.receiver_x_m, meta_row.receiver_y_m], dtype=float)
    defenders = _parse_points(meta_row.defending_xy_json)
    attackers = _parse_points(meta_row.attacking_xy_json)
    if len(defenders) == 0:
        return {}

    v = p1 - p0
    vv = float(np.dot(v, v))
    if vv <= 1e-8:
        return {}
    rel = defenders - p0[None, :]
    t = (rel @ v) / vv
    proj = p0[None, :] + t[:, None] * v[None, :]
    dist = np.linalg.norm(defenders - proj, axis=1)
    idx = int(np.argmin(dist))
    nearest = defenders[idx].copy()
    n = np.array([-v[1], v[0]], dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-8:
        n = np.array([0.0, 1.0])
        n_norm = 1.0
    n /= n_norm
    shift = float(cfg.get("eval", {}).get("counterfactual_shift_m", 10.0))

    on_line_def = defenders.copy()
    # Move nearest defender onto projected point on the pass line (increases corridor occupancy).
    on_line_def[idx] = proj[idx]

    off_line_def = defenders.copy()
    sign = 1.0 if np.dot(defenders[idx] - proj[idx], n) >= 0 else -1.0
    off_line_def[idx] = defenders[idx] + sign * shift * n

    feats_on_line = compute_baseline_features(p0, p1, on_line_def, params)
    feats_off_line = compute_baseline_features(p0, p1, off_line_def, params)
    lb_on_line = label_line_break(p0, p1, on_line_def, params)
    lb_off_line = label_line_break(p0, p1, off_line_def, params)

    return {
        "p0": p0,
        "p1": p1,
        "attackers": attackers,
        "defenders_orig": defenders,
        "defenders_on_line": on_line_def,
        "defenders_off_line": off_line_def,
        "label_on_line": lb_on_line.label,
        "label_off_line": lb_off_line.label,
        "feats_on_line": feats_on_line,
        "feats_off_line": feats_off_line,
        "nearest_idx": idx,
        "nearest_orig_dist_to_line_m": float(dist[idx]),
    }


def _save_freq_panels(images: np.ndarray, sample_ids: np.ndarray, fig_dir: Path, sigma: float, n: int = 20) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(images))
    for i in range(n):
        orig = images[i].max(axis=0)
        low = low_pass(images[i], sigma=sigma).max(axis=0)
        high = high_pass(images[i], sigma=sigma).max(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, im, title in zip(axes, [orig, low, high], ["orig", "low-pass", "high-pass"]):
            ax.imshow(im, cmap="viridis", origin="upper")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(str(sample_ids[i]))
        fig.tight_layout()
        fig.savefig(fig_dir / f"{sample_ids[i]}_freq_triplet.png", dpi=120)
        plt.close(fig)


def _save_counterfactual_panel(meta_row, cf: dict[str, Any], cfg: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    raster_cfg = cfg.get("raster", {})
    pitch_cfg = cfg.get("pitch", {})
    rs = RasterSpec(
        size=int(raster_cfg.get("size", 224)),
        sigma_m=float(raster_cfg.get("sigma_m", 1.2)),
        pitch_length_m=float(pitch_cfg.get("length_m", 105.0)),
        pitch_width_m=float(pitch_cfg.get("width_m", 68.0)),
        channels=int(raster_cfg.get("channels", 5)),
    )
    p0 = tuple(cf["p0"].tolist())
    p1 = tuple(cf["p1"].tolist())
    ball = (float(meta_row.ball_x_m), float(meta_row.ball_y_m)) if np.isfinite(getattr(meta_row, "ball_x_m", np.nan)) and np.isfinite(getattr(meta_row, "ball_y_m", np.nan)) else None
    imgs = [
        render_snapshot(cf["attackers"], cf["defenders_orig"], ball, p0, p1, rs).max(axis=0),
        render_snapshot(cf["attackers"], cf["defenders_on_line"], ball, p0, p1, rs).max(axis=0),
        render_snapshot(cf["attackers"], cf["defenders_off_line"], ball, p0, p1, rs).max(axis=0),
    ]
    titles = [
        f"orig (label={getattr(meta_row,'label', '?')})",
        f"on-line (label={cf.get('label_on_line', '?')})",
        f"off-line (label={cf.get('label_off_line', '?')})",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im, cmap="magma", origin="upper")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(str(meta_row.sample_id))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _counterfactual_render_spec(cfg: dict[str, Any]) -> RasterSpec:
    raster_cfg = cfg.get("raster", {})
    pitch_cfg = cfg.get("pitch", {})
    return RasterSpec(
        size=int(raster_cfg.get("size", 224)),
        sigma_m=float(raster_cfg.get("sigma_m", 1.2)),
        line_sigma_m=float(raster_cfg.get("line_sigma_m", 0.8)),
        corridor_w_m=float(raster_cfg.get("corridor_w_m", cfg.get("labeling", {}).get("corridor_w_m", 8.0))),
        pitch_length_m=float(pitch_cfg.get("length_m", 105.0)),
        pitch_width_m=float(pitch_cfg.get("width_m", 68.0)),
        channels=int(raster_cfg.get("channels", 5)),
    )


def _render_counterfactual_variant_images(meta_row, cf: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    rs = _counterfactual_render_spec(cfg)
    p0 = tuple(cf["p0"].tolist())
    p1 = tuple(cf["p1"].tolist())
    ball = (
        (float(meta_row.ball_x_m), float(meta_row.ball_y_m))
        if np.isfinite(getattr(meta_row, "ball_x_m", np.nan)) and np.isfinite(getattr(meta_row, "ball_y_m", np.nan))
        else None
    )
    return {
        "orig": render_snapshot(cf["attackers"], cf["defenders_orig"], ball, p0, p1, rs),
        "on_line": render_snapshot(cf["attackers"], cf["defenders_on_line"], ball, p0, p1, rs),
        "off_line": render_snapshot(cf["attackers"], cf["defenders_off_line"], ball, p0, p1, rs),
    }


def _build_counterfactual_rows_and_images(meta_test, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    max_cf_eval = cfg.get("eval", {}).get("max_counterfactual_eval_samples")
    rows_out: list[dict[str, Any]] = []
    orig_imgs: list[np.ndarray] = []
    on_imgs: list[np.ndarray] = []
    off_imgs: list[np.ndarray] = []
    for i, row in enumerate(meta_test.itertuples(index=False)):
        if max_cf_eval is not None and len(rows_out) >= int(max_cf_eval):
            break
        cf = _counterfactual_for_row(row, cfg)
        if not cf:
            continue
        imgs = _render_counterfactual_variant_images(row, cf, cfg)
        rows_out.append(
            {
                "sample_id": row.sample_id,
                "orig_label": int(row.label),
                "label_on_line": int(cf["label_on_line"]),
                "label_off_line": int(cf["label_off_line"]),
                "feats_on_line": cf["feats_on_line"],
                "feats_off_line": cf["feats_off_line"],
                "nearest_orig_dist_to_line_m": float(cf.get("nearest_orig_dist_to_line_m", np.nan)),
            }
        )
        orig_imgs.append(imgs["orig"])
        on_imgs.append(imgs["on_line"])
        off_imgs.append(imgs["off_line"])

    if not rows_out:
        return [], {}
    return rows_out, {
        "orig": np.stack(orig_imgs, axis=0).astype(np.float32),
        "on_line": np.stack(on_imgs, axis=0).astype(np.float32),
        "off_line": np.stack(off_imgs, axis=0).astype(np.float32),
    }


def _generate_vit_rollout_artifacts(
    X_img_test: np.ndarray,
    y_test: np.ndarray,
    sample_ids_test: np.ndarray,
    meta_test,
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    n_samples = int(cfg.get("eval", {}).get("n_rollout_samples", 30))
    n_samples = min(n_samples, len(X_img_test))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fallback: save channel-max heatmaps so artifact paths exist even without torch/timm.
    def _save_placeholder(i: int, heat: np.ndarray, suffix: str = "rollout"):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(X_img_test[i].max(axis=0), cmap="gray", origin="upper")
        ax[0].set_title("raster")
        ax[1].imshow(heat, cmap="inferno", origin="upper")
        ax[1].set_title(suffix)
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{sample_ids_test[i]} (y={int(y_test[i])})")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{sample_ids_test[i]}_rollout.png", dpi=130)
        plt.close(fig)

    vit_path = _paths(cfg)["models"] / "vit_base.pt"
    if not vit_path.exists():
        for i in range(n_samples):
            heat = X_img_test[i].max(axis=0)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            _save_placeholder(i, heat, "placeholder")
        return {"rollout_count": n_samples, "attention_distance": {"status": "vit_checkpoint_missing"}}

    try:  # pragma: no cover - torch/timm dependent
        import torch

        from .models.explain_vit import (
            ViTAttentionExtractor,
            attention_distance,
            attention_rollout,
            cls_to_patch_heatmap,
        )
        from .models.vit import ViTBuildConfig, create_vit_model
    except Exception:
        for i in range(n_samples):
            heat = X_img_test[i].max(axis=0)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            _save_placeholder(i, heat, "placeholder")
        return {"rollout_count": n_samples, "attention_distance": {"status": "vit_dependencies_missing"}}

    ckpt = torch.load(vit_path, map_location="cpu")
    model = create_vit_model(
        ViTBuildConfig(model_name="vit_base_patch16_224", in_chans=int(X_img_test.shape[1]), num_classes=2, pretrained=False)
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # Infer patch grid from model if possible.
    patch_size = 16
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        ps = model.patch_embed.patch_size
        patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
    grid_hw = (X_img_test.shape[-2] // patch_size, X_img_test.shape[-1] // patch_size)
    pitch_length = float(cfg.get("pitch", {}).get("length_m", 105.0))
    patch_size_m = pitch_length / grid_hw[1]

    dist_rows = []
    focus_rows = []
    for i in range(n_samples):
        xb = torch.tensor(X_img_test[i : i + 1], dtype=torch.float32)
        with ViTAttentionExtractor(model) as cap:
            _ = model(xb)
            mats = cap.capture().matrices
        if mats:
            roll = attention_rollout(mats)
            heat = cls_to_patch_heatmap(roll, grid_hw)
            # Upsample to raster size (nearest) without extra deps.
            heat_big = np.kron(heat, np.ones((patch_size, patch_size), dtype=np.float32))
            heat_big = heat_big[: X_img_test.shape[-2], : X_img_test.shape[-1]]
            dist_rows.append({"label": int(y_test[i]), **attention_distance(mats, patch_size_m=patch_size_m)})
            # Passer/receiver-centered rollout summary on patch grid.
            if meta_test is not None and i < len(meta_test):
                row = meta_test.iloc[i]
                rs = _counterfactual_render_spec(cfg)
                def _patch_focus(x_m: float, y_m: float, radius: int = 1) -> float:
                    x_px, y_px = meters_to_pixel(float(x_m), float(y_m), rs)
                    px = int(np.clip(np.floor(x_px / patch_size), 0, grid_hw[1] - 1))
                    py = int(np.clip(np.floor(y_px / patch_size), 0, grid_hw[0] - 1))
                    y0 = max(0, py - radius)
                    y1 = min(grid_hw[0], py + radius + 1)
                    x0 = max(0, px - radius)
                    x1 = min(grid_hw[1], px + radius + 1)
                    patch = heat[y0:y1, x0:x1]
                    return float(np.mean(patch)) if patch.size else float("nan")

                passer_focus = _patch_focus(row.passer_x_m, row.passer_y_m)
                receiver_focus = _patch_focus(row.receiver_x_m, row.receiver_y_m)
                focus_rows.append(
                    {
                        "label": int(y_test[i]),
                        "passer_focus_mean": passer_focus,
                        "receiver_focus_mean": receiver_focus,
                        "receiver_minus_passer": receiver_focus - passer_focus,
                    }
                )
        else:
            heat_big = X_img_test[i].max(axis=0)
            heat_big = (heat_big - heat_big.min()) / (heat_big.max() - heat_big.min() + 1e-8)
        _save_placeholder(i, heat_big, "rollout")

    if not dist_rows:
        return {"rollout_count": n_samples, "attention_distance": {"status": "attention_not_captured"}}

    # Aggregate label-wise means and low/high perturb deltas (using prediction-time attention distance only if available).
    dists = np.array([r["attention_distance_mean"] for r in dist_rows], dtype=float)
    labs = np.array([r["label"] for r in dist_rows], dtype=int)
    out = {
        "rollout_count": n_samples,
        "attention_distance": {
            "label0_mean": float(np.nanmean(dists[labs == 0])) if np.any(labs == 0) else None,
            "label1_mean": float(np.nanmean(dists[labs == 1])) if np.any(labs == 1) else None,
            "overall_mean": float(np.nanmean(dists)),
            "overall_std": float(np.nanstd(dists)),
        },
    }
    if focus_rows:
        f_lab = np.array([r["label"] for r in focus_rows], dtype=int)
        f_p = np.array([r["passer_focus_mean"] for r in focus_rows], dtype=float)
        f_r = np.array([r["receiver_focus_mean"] for r in focus_rows], dtype=float)
        f_d = np.array([r["receiver_minus_passer"] for r in focus_rows], dtype=float)
        out["rollout_focus"] = {
            "label0": {
                "passer_mean": float(np.nanmean(f_p[f_lab == 0])) if np.any(f_lab == 0) else None,
                "receiver_mean": float(np.nanmean(f_r[f_lab == 0])) if np.any(f_lab == 0) else None,
                "receiver_minus_passer_mean": float(np.nanmean(f_d[f_lab == 0])) if np.any(f_lab == 0) else None,
            },
            "label1": {
                "passer_mean": float(np.nanmean(f_p[f_lab == 1])) if np.any(f_lab == 1) else None,
                "receiver_mean": float(np.nanmean(f_r[f_lab == 1])) if np.any(f_lab == 1) else None,
                "receiver_minus_passer_mean": float(np.nanmean(f_d[f_lab == 1])) if np.any(f_lab == 1) else None,
            },
            "overall": {
                "passer_mean": float(np.nanmean(f_p)),
                "receiver_mean": float(np.nanmean(f_r)),
                "receiver_minus_passer_mean": float(np.nanmean(f_d)),
            },
        }
    return out


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    paths = _paths(cfg)
    ensure_dirs(paths["reports"], paths["figures"])
    arrays, meta = load_saved_dataset(cfg)
    splits = _load_splits(paths["processed"])
    eval_cfg = cfg.get("eval", {})
    te_idx = splits["test_idx"]
    te_idx = _subsample_eval_indices(
        te_idx,
        arrays["labels"].astype(int),
        eval_cfg.get("max_eval_samples"),
        seed=int(cfg.get("seed", 42)) + 99,
    )
    va_idx = splits["val_idx"] if len(splits["val_idx"]) else splits["test_idx"]
    va_idx = _subsample_eval_indices(
        va_idx,
        arrays["labels"].astype(int),
        eval_cfg.get("max_val_threshold_samples"),
        seed=int(cfg.get("seed", 42)) + 100,
    )
    y_val = arrays["labels"][va_idx].astype(int)
    y_test = arrays["labels"][te_idx].astype(int)
    X_feat_val = arrays["features"][va_idx]
    X_feat_test = arrays["features"][te_idx]
    X_img_val_full = arrays["images"][va_idx]
    X_img_test_full = arrays["images"][te_idx]
    X_img_val, selected_channels_val = select_image_channels(X_img_val_full, cfg)
    X_img_test, selected_channels = select_image_channels(X_img_test_full, cfg)
    if selected_channels_val != selected_channels:
        raise RuntimeError("Channel selection mismatch between val and test subsets")
    sample_ids_all = arrays.get("sample_ids")
    if sample_ids_all is not None and len(sample_ids_all) == len(arrays["labels"]):
        sample_ids_test = sample_ids_all[te_idx]
    else:
        sample_ids_test = np.asarray([str(i) for i in te_idx], dtype=object)
    meta_test = meta.iloc[te_idx].reset_index(drop=True)

    metrics: dict[str, Any] = {"models": {}, "conditions": {}, "counterfactual": {}}
    metrics["conditions"]["eval_n_test"] = int(len(te_idx))
    metrics["conditions"]["eval_n_val_for_threshold"] = int(len(va_idx))
    metrics["conditions"]["selected_channels"] = selected_channels
    metrics["conditions"]["counterfactual_eval_n"] = int(cfg.get("eval", {}).get("max_counterfactual_eval_samples", len(te_idx)))
    metrics["conditions"]["threshold_grid_size"] = int(eval_cfg.get("threshold_grid_size", 101))

    feature_names_dataset = [str(x) for x in arrays.get("feature_names", [])]
    cf_rows_shared, cf_images_shared = _build_counterfactual_rows_and_images(meta_test, cfg)
    baseline_ckpts = sorted(paths["models"].glob("baseline*.pkl"))
    for bpath in baseline_ckpts:
        model_name = bpath.stem
        baseline = load_baseline(bpath)
        feat_names = list(getattr(baseline, "feature_names", []) or feature_names_dataset)
        feat_idx = [feature_names_dataset.index(f) for f in feat_names]
        Xb_val = X_feat_val[:, feat_idx]
        Xb = X_feat_test[:, feat_idx]
        p_val = baseline.predict_proba(Xb_val)
        p_orig = baseline.predict_proba(Xb)
        metrics["models"][model_name] = {"original": nan_to_none(binary_metrics(y_test, p_orig))}
        # Frequency perturbs do not alter baseline features; log identical values for comparability.
        metrics["models"][model_name]["low_pass"] = metrics["models"][model_name]["original"]
        metrics["models"][model_name]["high_pass"] = metrics["models"][model_name]["original"]
        metrics["models"][model_name]["threshold_tuning"] = nan_to_none(
            _compute_threshold_tuning_summary(
                y_val=y_val,
                p_val=p_val,
                y_test=y_test,
                p_test_by_condition={"original": p_orig, "low_pass": p_orig, "high_pass": p_orig},
                n_grid=int(eval_cfg.get("threshold_grid_size", 101)),
            )
        )

        cf_rows = []
        for r in cf_rows_shared:
            x_on = np.asarray([[r["feats_on_line"][k] for k in feat_names]], dtype=np.float32)
            x_off = np.asarray([[r["feats_off_line"][k] for k in feat_names]], dtype=np.float32)
            p_on = float(baseline.predict_proba(x_on)[0])
            p_off = float(baseline.predict_proba(x_off)[0])
            cf_rows.append(
                {
                    "sample_id": r["sample_id"],
                    "orig_label": int(r["orig_label"]),
                    "label_on_line": int(r["label_on_line"]),
                    "label_off_line": int(r["label_off_line"]),
                    "p_on_line": p_on,
                    "p_off_line": p_off,
                    "delta_on_minus_off": p_on - p_off,
                    "nearest_orig_dist_to_line_m": r["nearest_orig_dist_to_line_m"],
                }
            )
        if cf_rows:
            delta = np.array([r["delta_on_minus_off"] for r in cf_rows], dtype=float)
            metrics["counterfactual"][model_name] = {
                "n": int(len(cf_rows)),
                "mean_delta_on_line_minus_off_line": float(np.mean(delta)),
                "on_line_greater_rate": float(np.mean(delta > 0)),
                "expected_direction": "on_line > off_line for geometry line-break label",
            }
            import pandas as pd

            pd.DataFrame(cf_rows).to_csv(paths["reports"] / f"counterfactual_{model_name}.csv", index=False)

    sigma = float(cfg.get("eval", {}).get("freq_blur_sigma", 2.0))
    n_freq_panels = int(cfg.get("eval", {}).get("n_freq_panels", 20))
    _save_freq_panels(X_img_test, sample_ids_test, paths["figures"], sigma=sigma, n=n_freq_panels)

    # Evaluate available image models under original/low/high perturbations.
    for model_name in ["resnet18", "vit_base"]:
        p_val = _predict_torch(model_name, X_img_val, cfg)
        p_orig = _predict_torch(model_name, X_img_test, cfg)
        if p_orig is None:
            continue
        X_low = np.stack([low_pass(x, sigma=sigma) for x in X_img_test], axis=0)
        X_high = np.stack([high_pass(x, sigma=sigma) for x in X_img_test], axis=0)
        p_low = _predict_torch(model_name, X_low, cfg)
        p_high = _predict_torch(model_name, X_high, cfg)
        metrics["models"][model_name] = {
            "original": nan_to_none(binary_metrics(y_test, p_orig)),
            "low_pass": nan_to_none(binary_metrics(y_test, p_low)) if p_low is not None else None,
            "high_pass": nan_to_none(binary_metrics(y_test, p_high)) if p_high is not None else None,
        }
        if p_val is not None:
            metrics["models"][model_name]["threshold_tuning"] = nan_to_none(
                _compute_threshold_tuning_summary(
                    y_val=y_val,
                    p_val=p_val,
                    y_test=y_test,
                    p_test_by_condition={
                        "original": p_orig,
                        "low_pass": p_low if p_low is not None else p_orig,
                        "high_pass": p_high if p_high is not None else p_orig,
                    },
                    n_grid=int(eval_cfg.get("threshold_grid_size", 101)),
                )
            )
        # Counterfactual probability shift for image models (same geometry manipulations rendered as raster).
        if cf_rows_shared and cf_images_shared:
            X_cf_on, _ = select_image_channels(cf_images_shared["on_line"], cfg)
            X_cf_off, _ = select_image_channels(cf_images_shared["off_line"], cfg)
            p_on = _predict_torch(model_name, X_cf_on, cfg)
            p_off = _predict_torch(model_name, X_cf_off, cfg)
            if p_on is not None and p_off is not None and len(p_on) == len(cf_rows_shared) and len(p_off) == len(cf_rows_shared):
                cf_rows = []
                for r, pon, poff in zip(cf_rows_shared, p_on, p_off):
                    cf_rows.append(
                        {
                            "sample_id": r["sample_id"],
                            "orig_label": int(r["orig_label"]),
                            "label_on_line": int(r["label_on_line"]),
                            "label_off_line": int(r["label_off_line"]),
                            "p_on_line": float(pon),
                            "p_off_line": float(poff),
                            "delta_on_minus_off": float(pon - poff),
                            "nearest_orig_dist_to_line_m": r["nearest_orig_dist_to_line_m"],
                        }
                    )
                delta = np.asarray([r["delta_on_minus_off"] for r in cf_rows], dtype=float)
                metrics["counterfactual"][model_name] = {
                    "n": int(len(cf_rows)),
                    "mean_delta_on_line_minus_off_line": float(np.mean(delta)),
                    "on_line_greater_rate": float(np.mean(delta > 0)),
                    "expected_direction": "on_line > off_line for geometry line-break label",
                }
                import pandas as pd

                pd.DataFrame(cf_rows).to_csv(paths["reports"] / f"counterfactual_{model_name}.csv", index=False)

    metrics["explainability"] = _generate_vit_rollout_artifacts(
        X_img_test=X_img_test,
        y_test=y_test,
        sample_ids_test=np.asarray(sample_ids_test),
        meta_test=meta_test.reset_index(drop=True),
        cfg=cfg,
        out_dir=paths["figures"],
    )

    # Counterfactual figures for representative samples (first few where constructible).
    n_cf_target = int(cfg.get("eval", {}).get("n_counterfactual_panels", 20))
    n_cf = 0
    for row in meta_test.itertuples(index=False):
        cf = _counterfactual_for_row(row, cfg)
        if not cf:
            continue
        _save_counterfactual_panel(row, cf, cfg, paths["figures"] / f"sample_{row.sample_id}_counterfactual.png")
        n_cf += 1
        if n_cf >= n_cf_target:
            break
    metrics["artifacts"] = {"counterfactual_panels": n_cf, "freq_triplets": int(min(n_freq_panels, len(X_img_test)))}

    save_json(paths["reports"] / "metrics.json", nan_to_none(metrics))
    print(json.dumps(nan_to_none(metrics), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained models and robustness experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/default.yaml")
    run.set_defaults(func=cmd_run)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
