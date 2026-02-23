from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dirs, load_yaml


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    p = cfg.get("paths", {})
    return {
        "reports": Path(p.get("reports_dir", "reports")),
        "figures": Path(p.get("figures_dir", "reports/figures")),
    }


def _load_metrics(report_dir: Path) -> dict[str, Any]:
    p = report_dir / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics.json in {report_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_named_reports(named_reports: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in str(named_reports).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = Path(v.strip())
    if not out:
        raise ValueError("No named reports provided")
    return out


def _get(d: dict[str, Any] | None, *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _plot_q1_no_passer(metrics_map: dict[str, dict[str, Any]], out_path: Path) -> bool:
    if not {"both", "no_passer"}.issubset(metrics_map.keys()):
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = []
    for key in ["both", "no_passer"]:
        m = metrics_map[key]
        vit = _get(m, "models", "vit_base", default={}) or {}
        orig = vit.get("original", {})
        cf = _get(m, "counterfactual", "vit_base", default={}) or {}
        rf = _get(m, "explainability", "rollout_focus", "overall", default={}) or {}
        rows.append(
            {
                "name": key,
                "AUROC": _safe_float(_get(orig, "auroc")),
                "F1": _safe_float(_get(orig, "f1")),
                "CF rate": _safe_float(cf.get("on_line_greater_rate")),
                "CF delta": _safe_float(cf.get("mean_delta_on_line_minus_off_line")),
                "Receiver-Passer": _safe_float(rf.get("receiver_minus_passer_mean")),
                "Corridor/Passer": _safe_float(rf.get("corridor_to_passer_ratio")),
                "NearestDef/Passer": _safe_float(rf.get("nearest_defender_to_passer_ratio")),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["AUROC", "F1", "CF rate", "CF delta"]
    x = np.arange(len(labels))
    width = 0.35
    for j, row in enumerate(rows):
        vals = [row[k] if row[k] is not None else 0.0 for k in labels]
        axes[0].bar(x + (j - 0.5) * width, vals, width=width, label=row["name"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title("Q1: no_passer vs both (performance + CF)")
    axes[0].legend()
    axes[0].set_ylim(min(-0.1, np.nanmin([r["CF delta"] or 0 for r in rows]) - 0.05), 1.0)

    labels2 = ["Receiver-Passer", "Corridor/Passer", "NearestDef/Passer"]
    x2 = np.arange(len(labels2))
    for j, row in enumerate(rows):
        vals = [row[k] if row[k] is not None else 0.0 for k in labels2]
        axes[1].bar(x2 + (j - 0.5) * width, vals, width=width, label=row["name"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=20)
    axes[1].set_title("Q1: focus balance (ViT rollout)")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_q2_counterfactual_flip(compare_metrics: dict[str, Any], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    models = ["resnet18", "vit_base"]
    rows = []
    for model in models:
        cf = _get(compare_metrics, "counterfactual", model, default={}) or {}
        rows.append(
            {
                "model": model,
                "overall_rate": _safe_float(cf.get("on_line_greater_rate")),
                "flip_rate": _safe_float(_get(cf, "stratified", "by_label_flip", "flip", "on_line_greater_rate")),
                "flip_n": _get(cf, "stratified", "by_label_flip", "flip", "n"),
                "overall_delta": _safe_float(cf.get("mean_delta_on_line_minus_off_line")),
                "flip_delta": _safe_float(_get(cf, "stratified", "by_label_flip", "flip", "mean_delta_on_line_minus_off_line")),
            }
        )
    if all(r["overall_rate"] is None and r["flip_rate"] is None for r in rows):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(rows))
    width = 0.36
    axes[0].bar(x - width / 2, [r["overall_rate"] or 0 for r in rows], width=width, label="overall")
    axes[0].bar(x + width / 2, [r["flip_rate"] or 0 for r in rows], width=width, label="label-flip subset")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["model"] for r in rows])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Q2: Counterfactual on-line > off-line rate")
    axes[0].legend()
    for i, r in enumerate(rows):
        if r["flip_n"] is not None:
            axes[0].text(i + width / 2, (r["flip_rate"] or 0) + 0.03, f"n={r['flip_n']}", ha="center", fontsize=8)

    axes[1].bar(x - width / 2, [r["overall_delta"] or 0 for r in rows], width=width, label="overall")
    axes[1].bar(x + width / 2, [r["flip_delta"] or 0 for r in rows], width=width, label="label-flip subset")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["model"] for r in rows])
    axes[1].set_title("Q2: Counterfactual ΔP(on-off)")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_q3_role_map(metrics_map: dict[str, dict[str, Any]], out_path: Path) -> bool:
    needed = ["line_only", "corridor_only", "both"]
    if not all(k in metrics_map for k in needed):
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = []
    for key in needed:
        m = metrics_map[key]
        vit = _get(m, "models", "vit_base", default={}) or {}
        orig = vit.get("original", {})
        cf = _get(m, "counterfactual", "vit_base", default={}) or {}
        rf = _get(m, "explainability", "rollout_focus", "overall", default={}) or {}
        rows.append(
            {
                "run": key,
                "AUROC": _safe_float(_get(orig, "auroc")) or 0.0,
                "F1_tuned": _safe_float(_get(vit, "threshold_tuning", "best_f1", "test_metrics", "original", "f1")) or 0.0,
                "CF_rate": _safe_float(cf.get("on_line_greater_rate")) or 0.0,
                "CF_flip_rate": _safe_float(_get(cf, "stratified", "by_label_flip", "flip", "on_line_greater_rate")) or 0.0,
                "Corridor/Passer": _safe_float(rf.get("corridor_to_passer_ratio")) or 0.0,
            }
        )
    metrics_names = ["AUROC", "F1_tuned", "CF_rate", "CF_flip_rate", "Corridor/Passer"]
    mat = np.array([[r[mn] for mn in metrics_names] for r in rows], dtype=float)
    # Column-wise min-max normalization for heatmap readability.
    mn = np.nanmin(mat, axis=0)
    mx = np.nanmax(mat, axis=0)
    den = np.where((mx - mn) > 1e-8, (mx - mn), 1.0)
    mat_norm = (mat - mn) / den

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    im = axes[0].imshow(mat_norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(metrics_names)))
    axes[0].set_xticklabels(metrics_names, rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(rows)))
    axes[0].set_yticklabels([r["run"] for r in rows])
    axes[0].set_title("Q3: Role heatmap (column-normalized)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[0].text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    x = np.array([r["AUROC"] for r in rows], dtype=float)
    y = np.array([r["CF_flip_rate"] for r in rows], dtype=float)
    axes[1].scatter(x, y, s=80, c=["tab:blue", "tab:orange", "tab:green"])
    for xi, yi, r in zip(x, y, rows):
        axes[1].text(xi + 0.002, yi + 0.01, r["run"], fontsize=9)
    axes[1].set_xlim(max(0, np.nanmin(x) - 0.05), min(1.0, np.nanmax(x) + 0.08))
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Discrimination (AUROC)")
    axes[1].set_ylabel("Structure sensitivity (CF flip-subset rate)")
    axes[1].set_title("Q3: Channel role map")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_q4_compare_focus(compare_metrics: dict[str, Any], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = []
    for model in ["resnet18", "vit_base"]:
        mm = _get(compare_metrics, "models", model, default={}) or {}
        cf = _get(compare_metrics, "counterfactual", model, default={}) or {}
        if model == "resnet18":
            rf = _get(compare_metrics, "explainability", "resnet18_focus", "focus", "overall", default={}) or {}
            method = _get(compare_metrics, "explainability", "resnet18_focus", "method", default="resnet_focus")
        else:
            rf = _get(compare_metrics, "explainability", "rollout_focus", "overall", default={}) or {}
            method = "vit_rollout"
        rows.append(
            {
                "model": model,
                "method": method,
                "AUROC": _safe_float(_get(mm, "original", "auroc")) or 0.0,
                "F1": _safe_float(_get(mm, "original", "f1")) or 0.0,
                "CF rate": _safe_float(cf.get("on_line_greater_rate")) or 0.0,
                "CF flip": _safe_float(_get(cf, "stratified", "by_label_flip", "flip", "on_line_greater_rate")) or 0.0,
                "Receiver-Passer": _safe_float(rf.get("receiver_minus_passer_mean")) or 0.0,
                "Corridor/Passer": _safe_float(rf.get("corridor_to_passer_ratio")) or 0.0,
                "NearestDef/Passer": _safe_float(rf.get("nearest_defender_to_passer_ratio")) or 0.0,
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    x = np.arange(len(rows))
    width = 0.18
    perf_cols = ["AUROC", "F1", "CF rate", "CF flip"]
    for j, col in enumerate(perf_cols):
        axes[0].bar(x + (j - 1.5) * width, [r[col] for r in rows], width=width, label=col)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["model"] for r in rows])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Q4: CNN vs ViT (performance + counterfactual)")
    axes[0].legend(fontsize=8)

    focus_cols = ["Receiver-Passer", "Corridor/Passer", "NearestDef/Passer"]
    width2 = 0.22
    for j, col in enumerate(focus_cols):
        axes[1].bar(x + (j - 1) * width2, [r[col] for r in rows], width=width2, label=col)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r['model']}\n({r['method']})" for r in rows], fontsize=8)
    axes[1].set_title("Q4: Common focus metrics")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _load_explainability_npz(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def _global_patch_centers_m(grid_h: int, grid_w: int, patch_size: int, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0) -> np.ndarray:
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    x_px = (xx + 0.5) * patch_size
    y_px = (yy + 0.5) * patch_size
    x_px = np.clip(x_px, 0, 223)
    y_px = np.clip(y_px, 0, 223)
    x_m = (x_px / 223.0) * pitch_length_m - pitch_length_m / 2.0
    y_m = pitch_width_m / 2.0 - (y_px / 223.0) * pitch_width_m
    return np.stack([x_m, y_m], axis=-1).astype(np.float32)


def _passcentric_average(
    sample_npz: dict[str, Any],
    corridor_w_m: float = 8.0,
    u_range: tuple[float, float] = (-0.25, 1.25),
    v_range: tuple[float, float] = (-2.0, 2.0),
    out_hw: tuple[int, int] = (80, 80),
) -> dict[str, Any] | None:
    heats = np.asarray(sample_npz.get("heat_patches"), dtype=np.float32)
    if heats.ndim != 3 or len(heats) == 0:
        return None
    labels = np.asarray(sample_npz.get("labels"), dtype=np.int64)
    px = int(np.asarray(sample_npz.get("patch_size"))[0])
    gh = int(np.asarray(sample_npz.get("grid_h"))[0])
    gw = int(np.asarray(sample_npz.get("grid_w"))[0])
    patch_centers = _global_patch_centers_m(gh, gw, px).reshape(-1, 2)
    p0x = np.asarray(sample_npz.get("passer_x_m"), dtype=np.float32)
    p0y = np.asarray(sample_npz.get("passer_y_m"), dtype=np.float32)
    p1x = np.asarray(sample_npz.get("receiver_x_m"), dtype=np.float32)
    p1y = np.asarray(sample_npz.get("receiver_y_m"), dtype=np.float32)

    H, W = out_hw
    num_all = np.zeros((H, W), dtype=np.float64)
    den_all = np.zeros((H, W), dtype=np.float64)
    num_l0 = np.zeros((H, W), dtype=np.float64)
    den_l0 = np.zeros((H, W), dtype=np.float64)
    num_l1 = np.zeros((H, W), dtype=np.float64)
    den_l1 = np.zeros((H, W), dtype=np.float64)

    u0, u1 = u_range
    v0, v1 = v_range
    for i in range(len(heats)):
        p0 = np.array([p0x[i], p0y[i]], dtype=np.float32)
        p1 = np.array([p1x[i], p1y[i]], dtype=np.float32)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if not np.isfinite(L) or L < 1e-6:
            continue
        e1 = v / L
        e2 = np.array([-e1[1], e1[0]], dtype=np.float32)
        rel = patch_centers - p0[None, :]
        u = (rel @ e1) / L
        vlat = (rel @ e2) / max(1e-6, float(corridor_w_m))
        uu = ((u - u0) / (u1 - u0) * (W - 1)).astype(int)
        vv = ((vlat - v0) / (v1 - v0) * (H - 1)).astype(int)
        mask = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        if not np.any(mask):
            continue
        hflat = heats[i].reshape(-1).astype(np.float64)
        for x_idx, y_idx, val in zip(uu[mask], vv[mask], hflat[mask]):
            num_all[y_idx, x_idx] += val
            den_all[y_idx, x_idx] += 1.0
            if int(labels[i]) == 1:
                num_l1[y_idx, x_idx] += val
                den_l1[y_idx, x_idx] += 1.0
            else:
                num_l0[y_idx, x_idx] += val
                den_l0[y_idx, x_idx] += 1.0

    def _mean(num, den):
        out = np.full_like(num, np.nan, dtype=np.float64)
        m = den > 0
        out[m] = num[m] / den[m]
        return out
    mean_all = _mean(num_all, den_all)
    mean_l0 = _mean(num_l0, den_l0)
    mean_l1 = _mean(num_l1, den_l1)
    diff_l1_l0 = mean_l1 - mean_l0
    return {
        "mean_all": mean_all,
        "mean_l0": mean_l0,
        "mean_l1": mean_l1,
        "diff_l1_l0": diff_l1_l0,
        "u_range": u_range,
        "v_range": v_range,
        "n_samples": int(len(heats)),
        "n_label1": int(np.sum(labels == 1)),
        "n_label0": int(np.sum(labels == 0)),
    }


def _plot_passcentric_heatmap(sample_npz_path: Path, out_path: Path, title_prefix: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    d = _load_explainability_npz(sample_npz_path)
    if d is None:
        return False
    acc = _passcentric_average(d)
    if acc is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    mean_all = acc["mean_all"]
    diff = acc["diff_l1_l0"]
    u0, u1 = acc["u_range"]
    v0, v1 = acc["v_range"]
    extent = [u0, u1, v0, v1]
    im0 = axes[0].imshow(mean_all, origin="lower", cmap="inferno", aspect="auto", extent=extent)
    axes[0].axhline(0, color="cyan", lw=0.8, alpha=0.7)
    axes[0].axvline(0, color="white", lw=0.8, alpha=0.7)
    axes[0].axvline(1, color="white", lw=0.8, alpha=0.7, ls="--")
    axes[0].set_title(f"{title_prefix}: pass-centric mean heat")
    axes[0].set_xlabel("longitudinal (pass-normalized, passer=0, receiver=1)")
    axes[0].set_ylabel("lateral (corridor-width normalized)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax = np.nanmax(np.abs(diff)) if np.isfinite(np.nanmax(np.abs(diff))) else 1.0
    vmax = max(vmax, 1e-6)
    im1 = axes[1].imshow(diff, origin="lower", cmap="coolwarm", aspect="auto", extent=extent, vmin=-vmax, vmax=vmax)
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.7)
    axes[1].axvline(0, color="black", lw=0.8, alpha=0.7)
    axes[1].axvline(1, color="black", lw=0.8, alpha=0.7, ls="--")
    axes[1].set_title(f"{title_prefix}: label1 - label0")
    axes[1].set_xlabel("longitudinal (pass-normalized)")
    axes[1].set_ylabel("lateral / corridor width")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_passcentric_compare(sample_npz_a: Path, label_a: str, sample_npz_b: Path, label_b: str, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    d_a = _load_explainability_npz(sample_npz_a)
    d_b = _load_explainability_npz(sample_npz_b)
    if d_a is None or d_b is None:
        return False
    a = _passcentric_average(d_a)
    b = _passcentric_average(d_b)
    if a is None or b is None:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    u0, u1 = a["u_range"]; v0, v1 = a["v_range"]
    extent = [u0, u1, v0, v1]
    arrs = [a["mean_all"], b["mean_all"], a["mean_all"] - b["mean_all"]]
    titles = [f"{label_a} mean", f"{label_b} mean", f"{label_a} - {label_b}"]
    cmaps = ["inferno", "inferno", "coolwarm"]
    vmax_diff = np.nanmax(np.abs(arrs[2])) if np.isfinite(np.nanmax(np.abs(arrs[2]))) else 1.0
    for ax, arr, ttl, cm in zip(axes, arrs, titles, cmaps):
        kwargs = {}
        if cm == "coolwarm":
            kwargs["vmin"] = -max(vmax_diff, 1e-6)
            kwargs["vmax"] = max(vmax_diff, 1e-6)
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cm, **kwargs)
        ax.axhline(0, color="white" if cm == "inferno" else "black", lw=0.8, alpha=0.7)
        ax.axvline(0, color="white" if cm == "inferno" else "black", lw=0.8, alpha=0.7)
        ax.axvline(1, color="white" if cm == "inferno" else "black", lw=0.8, alpha=0.7, ls="--")
        ax.set_title(ttl)
        ax.set_xlabel("u (pass-normalized)")
        ax.set_ylabel("v / corridor")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_metric_bars(metrics: dict[str, Any], fig_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = []
    for model, conds in metrics.get("models", {}).items():
        for cond, vals in conds.items():
            if not isinstance(vals, dict):
                continue
            if not {"auroc", "f1", "balanced_accuracy"}.issubset(vals.keys()):
                continue
            rows.append((model, cond, vals.get("auroc"), vals.get("f1"), vals.get("balanced_accuracy")))
    if not rows:
        return
    labels = [f"{m}\n{c}" for m, c, *_ in rows]
    aurocs = [r[2] if r[2] is not None else 0 for r in rows]
    f1s = [r[3] if r[3] is not None else 0 for r in rows]
    bals = [r[4] if r[4] is not None else 0 for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 4.5))
    width = 0.25
    ax.bar([i - width for i in x], aurocs, width=width, label="AUROC")
    ax.bar(list(x), f1s, width=width, label="F1")
    ax.bar([i + width for i in x], bals, width=width, label="Balanced Acc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model Performance by Condition")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _write_summary_md(metrics: dict[str, Any], out_path: Path, n_samples: int) -> None:
    lines = ["# Soccer ViT Line-Break Report", ""]
    art = metrics.get("artifacts", {})
    lines.append("## Summary")
    lines.append(f"- Requested showcase samples: {n_samples}")
    if art:
        lines.append(f"- Generated frequency triplet panels: {art.get('freq_triplets', 0)}")
        lines.append(f"- Generated counterfactual panels: {art.get('counterfactual_panels', 0)}")
    conds_top = metrics.get("conditions", {})
    if isinstance(conds_top, dict):
        ch = conds_top.get("selected_channels")
        if ch:
            lines.append(f"- Selected channels: {', '.join(map(str, ch))}")
        if conds_top.get("eval_n_val_for_threshold") is not None:
            lines.append(
                f"- Val samples for threshold tuning: {conds_top.get('eval_n_val_for_threshold')}"
            )
    cf_map = metrics.get("counterfactual", {})
    for model_name, cf in cf_map.items():
        if not isinstance(cf, dict):
            continue
        lines.append(
            f"- {model_name} counterfactual on-line>off-line rate: {cf.get('on_line_greater_rate')}"
        )
        strat = cf.get("stratified")
        if isinstance(strat, dict):
            by_lbl = strat.get("by_orig_label", {})
            if isinstance(by_lbl, dict):
                pos = by_lbl.get("1")
                neg = by_lbl.get("0")
                if isinstance(pos, dict):
                    lines.append(
                        f"- {model_name} CF (orig_label=1) on-line>off-line rate: {pos.get('on_line_greater_rate')}"
                    )
                if isinstance(neg, dict):
                    lines.append(
                        f"- {model_name} CF (orig_label=0) on-line>off-line rate: {neg.get('on_line_greater_rate')}"
                    )
            by_flip = strat.get("by_label_flip", {})
            if isinstance(by_flip, dict):
                flip = by_flip.get("flip")
                if isinstance(flip, dict):
                    lines.append(
                        f"- {model_name} CF (label-flip subset) on-line>off-line rate: {flip.get('on_line_greater_rate')}"
                    )
    rf = metrics.get("explainability", {}).get("rollout_focus")
    if isinstance(rf, dict):
        o = rf.get("overall", {})
        if isinstance(o, dict):
            lines.append(
                f"- Rollout focus overall (receiver-passer mean): {o.get('receiver_minus_passer_mean')}"
            )
            if o.get("corridor_mean") is not None:
                lines.append(f"- Rollout focus overall (corridor mean): {o.get('corridor_mean')}")
            if o.get("nearest_defender_mean") is not None:
                lines.append(f"- Rollout focus overall (nearest defender mean): {o.get('nearest_defender_mean')}")
            if o.get("corridor_to_passer_ratio") is not None:
                lines.append(f"- Rollout focus overall (corridor/passer ratio): {o.get('corridor_to_passer_ratio')}")
            if o.get("nearest_defender_to_passer_ratio") is not None:
                lines.append(
                    f"- Rollout focus overall (nearest-defender/passer ratio): {o.get('nearest_defender_to_passer_ratio')}"
                )
    cnn_focus = metrics.get("explainability", {}).get("resnet18_focus")
    if isinstance(cnn_focus, dict):
        cfo = (cnn_focus.get("focus") or {}).get("overall", {})
        if isinstance(cfo, dict):
            lines.append(
                f"- ResNet18 focus proxy overall (receiver-passer mean): {cfo.get('receiver_minus_passer_mean')}"
            )
            if cfo.get("corridor_to_passer_ratio") is not None:
                lines.append(
                    f"- ResNet18 focus proxy overall (corridor/passer ratio): {cfo.get('corridor_to_passer_ratio')}"
                )
    lines.append("")
    lines.append("## Models")
    for model, conds in metrics.get("models", {}).items():
        lines.append(f"### {model}")
        for cond, vals in conds.items():
            if not isinstance(vals, dict):
                continue
            if not {"auroc", "f1", "balanced_accuracy"}.issubset(vals.keys()):
                continue
            lines.append(
                f"- {cond}: AUROC={vals.get('auroc')} F1={vals.get('f1')} BalancedAcc={vals.get('balanced_accuracy')}"
            )
        tt = conds.get("threshold_tuning") if isinstance(conds, dict) else None
        if isinstance(tt, dict):
            for objective in ["best_f1", "best_balanced_accuracy"]:
                entry = tt.get(objective)
                if not isinstance(entry, dict):
                    continue
                thr = entry.get("threshold")
                tmo = entry.get("test_metrics", {}).get("original", {})
                if isinstance(tmo, dict):
                    lines.append(
                        f"- threshold_tuning/{objective}: thr={thr} "
                        f"test(original) F1={tmo.get('f1')} BalancedAcc={tmo.get('balanced_accuracy')} AUROC={tmo.get('auroc')}"
                    )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def cmd_make(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    paths = _paths(cfg)
    ensure_dirs(paths["reports"], paths["figures"])
    metrics_path = paths["reports"] / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run eval first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _plot_metric_bars(metrics, paths["figures"] / "performance_by_condition.png")
    _write_summary_md(metrics, paths["reports"] / "summary.md", n_samples=args.n_samples)
    print(json.dumps({"summary": str(paths['reports'] / 'summary.md')}, indent=2, ensure_ascii=False))


def cmd_questions(args: argparse.Namespace) -> None:
    named = _parse_named_reports(args.named_reports)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)
    metrics_map = {name: _load_metrics(path) for name, path in named.items()}
    created: list[str] = []

    # Q1: no_passer vs both
    if _plot_q1_no_passer(metrics_map, out_dir / "q1_no_passer_vs_both.png"):
        created.append(str(out_dir / "q1_no_passer_vs_both.png"))

    # Q2/Q4 from compare report if provided
    if "compare" in metrics_map:
        if _plot_q2_counterfactual_flip(metrics_map["compare"], out_dir / "q2_counterfactual_label_flip.png"):
            created.append(str(out_dir / "q2_counterfactual_label_flip.png"))
        if _plot_q4_compare_focus(metrics_map["compare"], out_dir / "q4_cnn_vs_vit_focus_compare.png"):
            created.append(str(out_dir / "q4_cnn_vs_vit_focus_compare.png"))

    # Q3: line-only / corridor-only / both role map
    if _plot_q3_role_map(metrics_map, out_dir / "q3_structure_channel_role_map.png"):
        created.append(str(out_dir / "q3_structure_channel_role_map.png"))

    # Pass-centric averages for any report dirs that have exported explainability NPZs.
    for name, rdir in named.items():
        vit_npz = rdir / "vit_rollout_samples.npz"
        if _plot_passcentric_heatmap(vit_npz, out_dir / f"passcentric_{name}_vit_rollout.png", f"{name}/ViT"):
            created.append(str(out_dir / f"passcentric_{name}_vit_rollout.png"))
        res_npz = rdir / "resnet18_focus_samples.npz"
        if _plot_passcentric_heatmap(res_npz, out_dir / f"passcentric_{name}_resnet18_focus.png", f"{name}/ResNet18"):
            created.append(str(out_dir / f"passcentric_{name}_resnet18_focus.png"))

    # Q1 pass-centric compare (both vs no_passer) if available.
    if "both" in named and "no_passer" in named:
        if _plot_passcentric_compare(
            named["both"] / "vit_rollout_samples.npz",
            "both",
            named["no_passer"] / "vit_rollout_samples.npz",
            "no_passer",
            out_dir / "q1_passcentric_both_vs_no_passer_vit.png",
        ):
            created.append(str(out_dir / "q1_passcentric_both_vs_no_passer_vit.png"))

    # Q4 pass-centric compare (ResNet vs ViT) from compare report if available.
    if "compare" in named:
        if _plot_passcentric_compare(
            named["compare"] / "vit_rollout_samples.npz",
            "ViT",
            named["compare"] / "resnet18_focus_samples.npz",
            "ResNet18",
            out_dir / "q4_passcentric_vit_vs_resnet.png",
        ):
            created.append(str(out_dir / "q4_passcentric_vit_vs_resnet.png"))

    # Save a machine-readable summary to make zipping/downloading easy.
    summary = {"out_dir": str(out_dir), "named_reports": {k: str(v) for k, v in named.items()}, "created": created}
    (out_dir / "questions_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate report figures and summary")
    sub = p.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make")
    make.add_argument("--config", default="configs/default.yaml")
    make.add_argument("--n-samples", type=int, default=30)
    make.set_defaults(func=cmd_make)

    q = sub.add_parser("questions", help="Generate question-oriented comparison figures across report directories")
    q.add_argument(
        "--named-reports",
        required=True,
        help="Comma-separated name=report_dir entries. Recommended names: both,no_passer,line_only,corridor_only,compare",
    )
    q.add_argument("--out-dir", default="reports/question_figures")
    q.set_defaults(func=cmd_questions)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
