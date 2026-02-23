from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .utils import ensure_dirs, load_yaml


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    p = cfg.get("paths", {})
    return {
        "reports": Path(p.get("reports_dir", "reports")),
        "figures": Path(p.get("figures_dir", "reports/figures")),
    }


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate report figures and summary")
    sub = p.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make")
    make.add_argument("--config", default="configs/default.yaml")
    make.add_argument("--n-samples", type=int, default=30)
    make.set_defaults(func=cmd_make)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
