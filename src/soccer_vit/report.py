from __future__ import annotations

import argparse
import csv
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


def _safe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _fmt_num(x: Any, nd: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def _nan_abs_max(arr: np.ndarray | list[np.ndarray], default: float = 1.0) -> float:
    if isinstance(arr, list):
        vals = []
        for a in arr:
            aa = np.asarray(a, dtype=float)
            if aa.size == 0:
                continue
            m = np.isfinite(aa)
            if np.any(m):
                vals.append(float(np.max(np.abs(aa[m]))))
        return max(vals) if vals else float(default)
    aa = np.asarray(arr, dtype=float)
    if aa.size == 0:
        return float(default)
    m = np.isfinite(aa)
    if not np.any(m):
        return float(default)
    return float(np.max(np.abs(aa[m])))


def _nan_min_max(arrs: list[np.ndarray], default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    vals = []
    for a in arrs:
        aa = np.asarray(a, dtype=float)
        if aa.size == 0:
            continue
        m = np.isfinite(aa)
        if np.any(m):
            vals.append(aa[m])
    if not vals:
        return float(default[0]), float(default[1])
    cat = np.concatenate(vals)
    return float(np.min(cat)), float(np.max(cat))


def _add_bar_value_labels(ax, bar_container, nd: int = 3, fontsize: int = 8) -> None:
    for rect in bar_container:
        h = float(rect.get_height())
        if not np.isfinite(h):
            continue
        x = float(rect.get_x() + rect.get_width() / 2.0)
        if h >= 0:
            y = h + 0.015
            va = "bottom"
        else:
            y = h - 0.015
            va = "top"
        ax.text(x, y, f"{h:.{nd}f}", ha="center", va=va, fontsize=fontsize)


def _load_counterfactual_csv_rows(report_dir: Path, model_name: str) -> list[dict[str, Any]]:
    path = report_dir / f"counterfactual_{model_name}.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            label_on = _safe_int(r.get("label_on_line"))
            label_off = _safe_int(r.get("label_off_line"))
            rows.append(
                {
                    "sample_id": r.get("sample_id"),
                    "delta_on_minus_off": _safe_float(r.get("delta_on_minus_off")),
                    "p_on_line": _safe_float(r.get("p_on_line")),
                    "p_off_line": _safe_float(r.get("p_off_line")),
                    "label_on_line": label_on,
                    "label_off_line": label_off,
                    "is_label_flip": (
                        None if label_on is None or label_off is None else bool(label_on != label_off)
                    ),
                }
            )
    return rows


def _mask_by_count(arr: np.ndarray, count: np.ndarray | None, min_count: float) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if count is not None:
        c = np.asarray(count, dtype=float)
        bad = (~np.isfinite(c)) | (c < float(min_count))
        out[bad] = np.nan
    return out


def _draw_passcentric_guides(ax, dark_bg: bool = False) -> None:
    line_color = "white" if dark_bg else "black"
    # Corridor zone in normalized lateral coordinates (|v| <= 1).
    ax.axhspan(-1.0, 1.0, color=(1, 1, 1, 0.05) if dark_bg else (0, 0, 0, 0.05), zorder=0)
    ax.axhline(0, color=line_color, lw=0.8, alpha=0.8, zorder=2)
    ax.axvline(0, color=line_color, lw=0.8, alpha=0.8, zorder=2)
    ax.axvline(1, color=line_color, lw=0.8, alpha=0.8, ls="--", zorder=2)
    ax.text(0.01, 0.98, "u=0 passer", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=line_color)
    ax.text(0.70, 0.98, "u=1 receiver", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=line_color)
    ax.text(0.01, 0.05, "v=0 pass line\n|v|<=1 corridor", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=line_color)


def _plot_q1_no_passer(
    metrics_map: dict[str, dict[str, Any]],
    out_path: Path,
    report_dirs: dict[str, Path] | None = None,
) -> bool:
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
                "report_dir": str((report_dirs or {}).get(key, "")),
                "AUROC": _safe_float(_get(orig, "auroc")),
                "F1": _safe_float(_get(orig, "f1")),
                "CF rate": _safe_float(cf.get("on_line_greater_rate")),
                "CF delta": _safe_float(cf.get("mean_delta_on_line_minus_off_line")),
                "Receiver-Passer": _safe_float(rf.get("receiver_minus_passer_mean")),
                "Corridor/Passer": _safe_float(rf.get("corridor_to_passer_ratio")),
                "NearestDef/Passer": _safe_float(rf.get("nearest_defender_to_passer_ratio")),
                "passer_mean": _safe_float(rf.get("passer_mean")),
                "receiver_mean": _safe_float(rf.get("receiver_mean")),
                "corridor_mean": _safe_float(rf.get("corridor_mean")),
                "nearest_defender_mean": _safe_float(rf.get("nearest_defender_mean")),
                "eval_n_test": _safe_int(_get(m, "conditions", "eval_n_test")),
                "cf_n": _safe_int(_get(m, "counterfactual", "vit_base", "n")),
                "rollout_n": _safe_int(_get(m, "explainability", "rollout_count")),
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    labels = ["AUROC", "F1", "CF rate", "CF delta"]
    x = np.arange(len(labels))
    width = 0.35
    for j, row in enumerate(rows):
        vals = [row[k] if row[k] is not None else 0.0 for k in labels]
        bars = axes[0].bar(x + (j - 0.5) * width, vals, width=width, label=row["name"])
        _add_bar_value_labels(axes[0], bars, nd=3, fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title("Q1: no_passer vs both (performance + CF)")
    axes[0].legend()
    axes[0].set_ylim(min(-0.1, np.nanmin([r["CF delta"] or 0 for r in rows]) - 0.05), 1.0)
    axes[0].grid(axis="y", alpha=0.2)

    labels2 = ["Receiver-Passer", "Corridor/Passer", "NearestDef/Passer"]
    x2 = np.arange(len(labels2))
    for j, row in enumerate(rows):
        vals = [row[k] if row[k] is not None else 0.0 for k in labels2]
        bars = axes[1].bar(x2 + (j - 0.5) * width, vals, width=width, label=row["name"])
        _add_bar_value_labels(axes[1], bars, nd=2, fontsize=7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=20)
    axes[1].set_title("Q1: focus balance (ViT rollout)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    # Attention budget view: normalize positive focus means into a 100% stacked bar.
    budget_parts = ["passer_mean", "receiver_mean", "corridor_mean", "nearest_defender_mean"]
    budget_labels = ["Passer", "Receiver", "Corridor", "NearestDef"]
    budget_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    x3 = np.arange(len(rows))
    bottoms = np.zeros(len(rows), dtype=float)
    for key, lbl, col in zip(budget_parts, budget_labels, budget_colors):
        vals = np.array([max(0.0, float(r.get(key) or 0.0)) for r in rows], dtype=float)
        totals = np.array(
            [sum(max(0.0, float(r.get(k) or 0.0)) for k in budget_parts) for r in rows],
            dtype=float,
        )
        pct = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 1e-12)
        axes[2].bar(x3, pct, bottom=bottoms, label=lbl, color=col, width=0.6)
        for i, p in enumerate(pct):
            if p >= 0.12:
                axes[2].text(i, bottoms[i] + p / 2.0, f"{p*100:.0f}%", ha="center", va="center", fontsize=7, color="white")
        bottoms += pct
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks(np.linspace(0, 1, 6))
    axes[2].set_yticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)])
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels([r["name"] for r in rows])
    axes[2].set_title("Q1: focus budget (100% stacked)")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(axis="y", alpha=0.2)

    both = next((r for r in rows if r["name"] == "both"), None)
    nop = next((r for r in rows if r["name"] == "no_passer"), None)
    eval_n = both.get("eval_n_test") if both else None
    rollout_n = both.get("rollout_n") if both else None
    cf_n = both.get("cf_n") if both else None
    cap1 = (
        f"Runs: both vs no_passer | eval_n={eval_n if eval_n is not None else 'NA'} | "
        f"rollout_n={rollout_n if rollout_n is not None else 'NA'} | CF_n={cf_n if cf_n is not None else 'NA'}"
    )
    cap2 = None
    if both and nop:
        auroc_gap = (both.get("AUROC") or 0.0) - (nop.get("AUROC") or 0.0)
        cf_delta_gap = (both.get("CF delta") or 0.0) - (nop.get("CF delta") or 0.0)
        cap2 = (
            f"Auto note: AUROC gap={auroc_gap:+.3f}, CF ΔP gap={cf_delta_gap:+.3f}. "
            "If AUROC is similar but CF ΔP is larger for 'both', passer channel may act as contextual support rather than a pure shortcut."
        )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.text(0.01, 0.05, cap1, ha="left", va="bottom", fontsize=8)
    if cap2:
        fig.text(0.01, 0.015, cap2, ha="left", va="bottom", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_q2_counterfactual_flip(compare_metrics: dict[str, Any], out_path: Path, compare_report_dir: Path | None = None) -> bool:
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
    cf_csv_rows = {m: _load_counterfactual_csv_rows(compare_report_dir, m) for m in models} if compare_report_dir else {}
    has_dot_panel = any(cf_csv_rows.get(m) for m in models)
    ncols = 3 if has_dot_panel else 2
    fig, axes = plt.subplots(1, ncols, figsize=(15.2 if has_dot_panel else 10.8, 4.6))
    if ncols == 2:
        axes = np.asarray(axes, dtype=object)
    x = np.arange(len(rows))
    width = 0.36
    bars0 = axes[0].bar(x - width / 2, [r["overall_rate"] or 0 for r in rows], width=width, label="overall")
    bars1 = axes[0].bar(x + width / 2, [r["flip_rate"] or 0 for r in rows], width=width, label="label-flip subset")
    _add_bar_value_labels(axes[0], bars0, nd=3, fontsize=7)
    _add_bar_value_labels(axes[0], bars1, nd=3, fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["model"] for r in rows])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Q2: Counterfactual on-line > off-line rate")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    for i, r in enumerate(rows):
        if r["flip_n"] is not None:
            axes[0].text(i + width / 2, min(0.98, (r["flip_rate"] or 0) + 0.06), f"flip n={r['flip_n']}", ha="center", fontsize=8)

    bars2 = axes[1].bar(x - width / 2, [r["overall_delta"] or 0 for r in rows], width=width, label="overall")
    bars3 = axes[1].bar(x + width / 2, [r["flip_delta"] or 0 for r in rows], width=width, label="label-flip subset")
    _add_bar_value_labels(axes[1], bars2, nd=3, fontsize=7)
    _add_bar_value_labels(axes[1], bars3, nd=3, fontsize=7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["model"] for r in rows])
    axes[1].set_title("Q2: Counterfactual ΔP(on-off)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    if has_dot_panel:
        ax = axes[2]
        ax.axhline(0, color="black", lw=0.8)
        colors = {"resnet18": "tab:blue", "vit_base": "tab:orange"}
        for i, model in enumerate(models):
            model_rows = cf_csv_rows.get(model, [])
            deltas = np.asarray([r.get("delta_on_minus_off", np.nan) for r in model_rows], dtype=float)
            if deltas.size == 0:
                continue
            flip = np.asarray([bool(r.get("is_label_flip")) if r.get("is_label_flip") is not None else False for r in model_rows], dtype=bool)
            jitter = np.linspace(-0.12, 0.12, len(deltas)) if len(deltas) > 1 else np.array([0.0], dtype=float)
            xj = i + jitter
            ax.scatter(xj, deltas, s=22, alpha=0.45, color="0.65", label="all samples" if i == 0 else None)
            if np.any(flip):
                ax.scatter(xj[flip], deltas[flip], s=30, alpha=0.9, color=colors[model], edgecolor="black", linewidth=0.3, label="label-flip subset" if i == 0 else None)
            mu = float(np.nanmean(deltas)) if np.any(np.isfinite(deltas)) else np.nan
            if np.isfinite(mu):
                ax.scatter([i], [mu], s=60, marker="D", color=colors[model], edgecolor="black", linewidth=0.5, zorder=4)
            ax.text(i, ax.get_ylim()[1] if np.isfinite(ax.get_ylim()[1]) else 0.0, "", alpha=0.0)  # keep autoscale stable before annotations
            ax.text(
                i,
                np.nanmax(deltas[np.isfinite(deltas)]) + 0.01 if np.any(np.isfinite(deltas)) else 0.02,
                f"n={len(deltas)}, flip={int(np.sum(flip))}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_title("Q2: Sample-level ΔP (local geometric intervention)")
        ax.set_ylabel("ΔP(on-line - off-line)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="best")

    flip_ns = [r.get("flip_n") for r in rows if r.get("flip_n") is not None]
    flip_n_note = f"label-flip subset n per model: {flip_ns}" if flip_ns else "label-flip subset n unavailable"
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.24, wspace=0.28)
    fig.text(0.01, 0.05, "Counterfactual here is a local geometric intervention (nearest-defender on-line vs off-line), not a full tactical counterfactual.", ha="left", va="bottom", fontsize=8)
    fig.text(0.01, 0.02, f"Small-n caveat: {flip_n_note}. Interpret flip-subset magnitudes as structure-sensitivity sanity checks.", ha="left", va="bottom", fontsize=8)
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

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    im = axes[0].imshow(mat_norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(metrics_names)))
    axes[0].set_xticklabels(metrics_names, rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(rows)))
    axes[0].set_yticklabels([r["run"] for r in rows])
    axes[0].set_title("Q3: Role heatmap (column-normalized background, raw values annotated)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[0].text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].text(
        0.01,
        -0.16,
        "Discrimination vs structure-sensitivity metrics are mixed columns; background is normalized, numbers are raw.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    x = np.array([r["AUROC"] for r in rows], dtype=float)
    y = np.array([r["CF_flip_rate"] for r in rows], dtype=float)
    axes[1].scatter(x, y, s=80, c=["tab:blue", "tab:orange", "tab:green"])
    text_offsets = {
        "line_only": (0.004, 0.015),
        "corridor_only": (-0.035, -0.035),
        "both": (0.006, 0.030),
    }
    for xi, yi, r in zip(x, y, rows):
        dx, dy = text_offsets.get(r["run"], (0.004, 0.01))
        axes[1].text(
            xi + dx,
            yi + dy,
            r["run"],
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axes[1].set_xlim(max(0, np.nanmin(x) - 0.05), min(1.0, np.nanmax(x) + 0.08))
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Discrimination (AUROC)")
    axes[1].set_ylabel("Structure sensitivity (CF flip-subset rate)")
    axes[1].set_title("Q3: Channel role map")
    axes[1].grid(alpha=0.2)
    axes[1].text(
        0.02,
        0.02,
        "Right = better discrimination (AUROC)\nUp = stronger structure sensitivity (CF flip-subset rate)",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
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
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    x = np.arange(len(rows))
    width = 0.18
    perf_cols = ["AUROC", "F1", "CF rate", "CF flip"]
    for j, col in enumerate(perf_cols):
        bars = axes[0].bar(x + (j - 1.5) * width, [r[col] for r in rows], width=width, label=col)
        _add_bar_value_labels(axes[0], bars, nd=3, fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["model"] for r in rows])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Q4: CNN vs ViT (performance + counterfactual)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    focus_cols = ["Receiver-Passer", "Corridor/Passer", "NearestDef/Passer"]
    width2 = 0.22
    for j, col in enumerate(focus_cols):
        bars = axes[1].bar(x + (j - 1) * width2, [r[col] for r in rows], width=width2, label=col)
        _add_bar_value_labels(axes[1], bars, nd=2, fontsize=7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["resnet18\n(proxy saliency)" if r["model"] == "resnet18" else "vit_base\n(attn rollout)" for r in rows],
        fontsize=8,
    )
    axes[1].set_title("Q4: Common focus metrics")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].text(
        0.02,
        0.02,
        "Receiver-Passer > 0: receiver-focused\nCorridor/Passer or NearestDef/Passer > 1: structure region exceeds passer focus",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(
        0.01,
        0.02,
        "ResNet18 focus uses input_grad * input patch-mean saliency proxy (not Grad-CAM).",
        ha="left",
        va="bottom",
        fontsize=8,
    )
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
    out_hw: tuple[int, int] = (96, 96),
    splat_sigma_cells: float = 0.85,
    splat_radius: int = 2,
) -> dict[str, Any] | None:
    heats = np.asarray(sample_npz.get("heat_patches"), dtype=np.float32)
    if heats.ndim != 3 or len(heats) == 0:
        return None
    labels = np.asarray(sample_npz.get("labels"), dtype=np.int64)
    if labels.shape[0] != heats.shape[0]:
        labels = np.zeros((heats.shape[0],), dtype=np.int64)
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
    sigma = max(1e-6, float(splat_sigma_cells))
    rad = max(1, int(splat_radius))
    n_used = 0
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
        xg = (u - u0) / (u1 - u0) * (W - 1)
        yg = (vlat - v0) / (v1 - v0) * (H - 1)
        mask = np.isfinite(xg) & np.isfinite(yg) & (xg >= 0) & (xg <= (W - 1)) & (yg >= 0) & (yg <= (H - 1))
        if not np.any(mask):
            continue
        hflat = heats[i].reshape(-1).astype(np.float64)
        n_used += 1
        for xc, yc, val in zip(xg[mask], yg[mask], hflat[mask]):
            if not np.isfinite(val):
                continue
            ix = int(np.floor(xc))
            iy = int(np.floor(yc))
            xs = np.arange(max(0, ix - rad), min(W - 1, ix + rad) + 1, dtype=int)
            ys = np.arange(max(0, iy - rad), min(H - 1, iy + rad) + 1, dtype=int)
            if xs.size == 0 or ys.size == 0:
                continue
            dx2 = (xs.astype(np.float64) - float(xc)) ** 2
            dy2 = (ys.astype(np.float64) - float(yc)) ** 2
            ker = np.exp(-(dy2[:, None] + dx2[None, :]) / (2.0 * sigma * sigma))
            s = float(np.sum(ker))
            if not np.isfinite(s) or s <= 0:
                continue
            ker = ker / s
            num_all[np.ix_(ys, xs)] += val * ker
            den_all[np.ix_(ys, xs)] += ker
            if int(labels[i]) == 1:
                num_l1[np.ix_(ys, xs)] += val * ker
                den_l1[np.ix_(ys, xs)] += ker
            else:
                num_l0[np.ix_(ys, xs)] += val * ker
                den_l0[np.ix_(ys, xs)] += ker

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
        "count_all": den_all,
        "count_l0": den_l0,
        "count_l1": den_l1,
        "u_range": u_range,
        "v_range": v_range,
        "n_samples": int(len(heats)),
        "n_used_samples": int(n_used),
        "n_label1": int(np.sum(labels == 1)),
        "n_label0": int(np.sum(labels == 0)),
        "method": str(np.asarray(sample_npz.get("method"), dtype=object)[0]) if sample_npz.get("method") is not None else "unknown",
    }


def _plot_passcentric_heatmap(
    sample_npz_path: Path,
    out_path: Path,
    title_prefix: str,
    *,
    grid_size: int = 96,
    splat_sigma_cells: float = 0.85,
    min_count: float = 2.0,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    d = _load_explainability_npz(sample_npz_path)
    if d is None:
        return False
    acc = _passcentric_average(d, out_hw=(int(grid_size), int(grid_size)), splat_sigma_cells=float(splat_sigma_cells))
    if acc is None:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8))
    axes = np.asarray(axes, dtype=object)
    mean_all = _mask_by_count(acc["mean_all"], acc.get("count_all"), min_count=min_count)
    mean_l0 = _mask_by_count(acc["mean_l0"], acc.get("count_l0"), min_count=min_count)
    mean_l1 = _mask_by_count(acc["mean_l1"], acc.get("count_l1"), min_count=min_count)
    count_shared = np.minimum(np.asarray(acc.get("count_l0"), dtype=float), np.asarray(acc.get("count_l1"), dtype=float))
    diff = _mask_by_count(acc["diff_l1_l0"], count_shared, min_count=min_count)
    u0, u1 = acc["u_range"]
    v0, v1 = acc["v_range"]
    extent = [u0, u1, v0, v1]
    inferno = plt.get_cmap("inferno").copy()
    inferno.set_bad("#d0d0d0")
    coolwarm = plt.get_cmap("coolwarm").copy()
    coolwarm.set_bad("#d0d0d0")
    mean_vmin, mean_vmax = _nan_min_max([mean_all, mean_l0, mean_l1], default=(0.0, 1.0))
    if not np.isfinite(mean_vmin):
        mean_vmin = 0.0
    if not np.isfinite(mean_vmax) or mean_vmax <= mean_vmin:
        mean_vmax = mean_vmin + 1e-6
    diff_vmax = max(_nan_abs_max(diff, default=1.0), 1e-6)
    panels = [
        (axes[0, 0], mean_all, acc.get("count_all"), f"{title_prefix}: all (n={acc['n_samples']}, used={acc.get('n_used_samples','NA')})", inferno, True, {"vmin": mean_vmin, "vmax": mean_vmax}),
        (axes[0, 1], mean_l0, acc.get("count_l0"), f"{title_prefix}: label0 (n0={acc['n_label0']})", inferno, True, {"vmin": mean_vmin, "vmax": mean_vmax}),
        (axes[1, 0], mean_l1, acc.get("count_l1"), f"{title_prefix}: label1 (n1={acc['n_label1']})", inferno, True, {"vmin": mean_vmin, "vmax": mean_vmax}),
        (axes[1, 1], diff, count_shared, f"{title_prefix}: label1 - label0", coolwarm, False, {"vmin": -diff_vmax, "vmax": diff_vmax}),
    ]
    for ax, arr, count_map, ttl, cmap, dark_bg, scale_kwargs in panels:
        im = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto", extent=extent, **scale_kwargs)
        _draw_passcentric_guides(ax, dark_bg=dark_bg)
        ax.set_title(ttl)
        ax.set_xlabel("u (pass-normalized; passer=0, receiver=1)")
        ax.set_ylabel("v / corridor width")
        cmax = _nan_abs_max(np.asarray(count_map), default=0.0) if count_map is not None else 0.0
        ax.text(
            0.99,
            0.02,
            f"min_count={min_count:g}\nmax occ={cmax:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Pass-centric heatmaps ({title_prefix}) | method={acc.get('method','unknown')} | Gaussian splat σ={splat_sigma_cells} cells",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.text(0.01, 0.005, "Gray cells indicate low occupancy / insufficient support after pass-centric alignment.", ha="left", va="bottom", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_passcentric_compare(
    sample_npz_a: Path,
    label_a: str,
    sample_npz_b: Path,
    label_b: str,
    out_path: Path,
    *,
    grid_size: int = 96,
    splat_sigma_cells: float = 0.85,
    min_count: float = 2.0,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    d_a = _load_explainability_npz(sample_npz_a)
    d_b = _load_explainability_npz(sample_npz_b)
    if d_a is None or d_b is None:
        return False
    a = _passcentric_average(d_a, out_hw=(int(grid_size), int(grid_size)), splat_sigma_cells=float(splat_sigma_cells))
    b = _passcentric_average(d_b, out_hw=(int(grid_size), int(grid_size)), splat_sigma_cells=float(splat_sigma_cells))
    if a is None or b is None:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    u0, u1 = a["u_range"]; v0, v1 = a["v_range"]
    extent = [u0, u1, v0, v1]
    a_mean = _mask_by_count(a["mean_all"], a.get("count_all"), min_count=min_count)
    b_mean = _mask_by_count(b["mean_all"], b.get("count_all"), min_count=min_count)
    shared_count = np.minimum(np.asarray(a.get("count_all"), dtype=float), np.asarray(b.get("count_all"), dtype=float))
    diff_mean = _mask_by_count(a["mean_all"] - b["mean_all"], shared_count, min_count=min_count)
    arrs = [a_mean, b_mean, diff_mean]
    titles = [
        f"{label_a} mean (n={a['n_samples']}, y1={a['n_label1']})",
        f"{label_b} mean (n={b['n_samples']}, y1={b['n_label1']})",
        f"{label_a} - {label_b}",
    ]
    cmaps = ["inferno", "inferno", "coolwarm"]
    inferno = plt.get_cmap("inferno").copy()
    inferno.set_bad("#d0d0d0")
    coolwarm = plt.get_cmap("coolwarm").copy()
    coolwarm.set_bad("#d0d0d0")
    mean_vmin, mean_vmax = _nan_min_max([a_mean, b_mean], default=(0.0, 1.0))
    if not np.isfinite(mean_vmin):
        mean_vmin = 0.0
    if not np.isfinite(mean_vmax) or mean_vmax <= mean_vmin:
        mean_vmax = mean_vmin + 1e-6
    vmax_diff = max(_nan_abs_max(diff_mean, default=1.0), 1e-6)
    for idx, (ax, arr, ttl, cm) in enumerate(zip(axes, arrs, titles, cmaps)):
        kwargs = {}
        if cm == "coolwarm":
            kwargs["vmin"] = -max(vmax_diff, 1e-6)
            kwargs["vmax"] = max(vmax_diff, 1e-6)
            count_map = shared_count
            cmap = coolwarm
            dark_bg = False
        else:
            kwargs["vmin"] = mean_vmin
            kwargs["vmax"] = mean_vmax
            count_map = a.get("count_all") if idx == 0 else b.get("count_all")
            cmap = inferno
            dark_bg = True
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cmap, **kwargs)
        _draw_passcentric_guides(ax, dark_bg=dark_bg)
        ax.set_title(ttl)
        ax.set_xlabel("u (pass-normalized)")
        ax.set_ylabel("v / corridor")
        cmax = _nan_abs_max(np.asarray(count_map), default=0.0) if count_map is not None else 0.0
        ax.text(
            0.99,
            0.02,
            f"min_count={min_count:g}\nmax occ={cmax:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Pass-centric compare | shared normalization for mean maps | Gaussian splat σ={splat_sigma_cells} cells",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.01, 0.005, "Gray cells indicate low occupancy / insufficient support after alignment.", ha="left", va="bottom", fontsize=8)
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
    passcentric_opts = {
        "grid_size": int(getattr(args, "passcentric_grid_size", 96)),
        "splat_sigma_cells": float(getattr(args, "passcentric_splat_sigma", 0.85)),
        "min_count": float(getattr(args, "passcentric_min_count", 2.0)),
    }

    # Q1: no_passer vs both
    if _plot_q1_no_passer(metrics_map, out_dir / "q1_no_passer_vs_both.png", report_dirs=named):
        created.append(str(out_dir / "q1_no_passer_vs_both.png"))

    # Q2/Q4 from compare report if provided
    if "compare" in metrics_map:
        if _plot_q2_counterfactual_flip(metrics_map["compare"], out_dir / "q2_counterfactual_label_flip.png", compare_report_dir=named.get("compare")):
            created.append(str(out_dir / "q2_counterfactual_label_flip.png"))
        if _plot_q4_compare_focus(metrics_map["compare"], out_dir / "q4_cnn_vs_vit_focus_compare.png"):
            created.append(str(out_dir / "q4_cnn_vs_vit_focus_compare.png"))

    # Q3: line-only / corridor-only / both role map
    if _plot_q3_role_map(metrics_map, out_dir / "q3_structure_channel_role_map.png"):
        created.append(str(out_dir / "q3_structure_channel_role_map.png"))

    # Pass-centric averages for any report dirs that have exported explainability NPZs.
    for name, rdir in named.items():
        vit_npz = rdir / "vit_rollout_samples.npz"
        if _plot_passcentric_heatmap(vit_npz, out_dir / f"passcentric_{name}_vit_rollout.png", f"{name}/ViT", **passcentric_opts):
            created.append(str(out_dir / f"passcentric_{name}_vit_rollout.png"))
        res_npz = rdir / "resnet18_focus_samples.npz"
        if _plot_passcentric_heatmap(res_npz, out_dir / f"passcentric_{name}_resnet18_focus.png", f"{name}/ResNet18", **passcentric_opts):
            created.append(str(out_dir / f"passcentric_{name}_resnet18_focus.png"))

    # Q1 pass-centric compare (both vs no_passer) if available.
    if "both" in named and "no_passer" in named:
        if _plot_passcentric_compare(
            named["both"] / "vit_rollout_samples.npz",
            "both",
            named["no_passer"] / "vit_rollout_samples.npz",
            "no_passer",
            out_dir / "q1_passcentric_both_vs_no_passer_vit.png",
            **passcentric_opts,
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
            **passcentric_opts,
        ):
            created.append(str(out_dir / "q4_passcentric_vit_vs_resnet.png"))

    # Save a machine-readable summary to make zipping/downloading easy.
    summary = {
        "out_dir": str(out_dir),
        "named_reports": {k: str(v) for k, v in named.items()},
        "created": created,
        "passcentric_options": passcentric_opts,
    }
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
    q.add_argument("--passcentric-grid-size", type=int, default=96)
    q.add_argument("--passcentric-splat-sigma", type=float, default=0.85, help="Gaussian splat sigma in output-grid cells")
    q.add_argument("--passcentric-min-count", type=float, default=2.0, help="Gray-mask cells with occupancy below this threshold")
    q.set_defaults(func=cmd_questions)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
