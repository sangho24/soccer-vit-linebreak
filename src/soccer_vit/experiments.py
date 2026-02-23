from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .metrics import nan_to_none
from .utils import load_yaml, save_json, slugify


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    p = cfg.get("paths", {})
    return {
        "processed": Path(p.get("processed_dir", "data/processed")),
        "reports": Path(p.get("reports_dir", "reports")),
        "figures": Path(p.get("figures_dir", "reports/figures")),
        "models": Path(p.get("models_dir", "reports/models")),
    }


def _run_module(module: str, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, "-m", module, *args]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _write_temp_cfg(cfg: dict[str, Any]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="soccer_vit_cfg_"))
    path = tmp_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _override_seed_and_paths(base_cfg: dict[str, Any], seed: int, suffix: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # cheap deep-copy of yaml-safe data
    cfg["seed"] = int(seed)
    train = cfg.setdefault("train", {})
    train["random_state"] = int(seed)
    p = cfg.setdefault("paths", {})
    base_reports = Path(p.get("reports_dir", "reports"))
    report_dir = str(base_reports.parent / f"{base_reports.name}_{suffix}")
    p["reports_dir"] = report_dir
    p["figures_dir"] = str(Path(report_dir) / "figures")
    p["models_dir"] = str(Path(report_dir) / "models")
    return cfg


def _extract_metric_block(metrics: dict[str, Any], model_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    model_block = (metrics.get("models") or {}).get(model_name, {})
    orig = model_block.get("original", {}) if isinstance(model_block, dict) else {}
    if isinstance(orig, dict):
        out["original_auroc"] = orig.get("auroc")
        out["original_f1"] = orig.get("f1")
        out["original_balanced_accuracy"] = orig.get("balanced_accuracy")
    tt = model_block.get("threshold_tuning", {}) if isinstance(model_block, dict) else {}
    for key in ("best_f1", "best_balanced_accuracy"):
        entry = tt.get(key, {}) if isinstance(tt, dict) else {}
        if isinstance(entry, dict):
            out[f"{key}_threshold"] = entry.get("threshold")
            tmo = (entry.get("test_metrics") or {}).get("original", {})
            if isinstance(tmo, dict):
                out[f"{key}_test_f1"] = tmo.get("f1")
                out[f"{key}_test_balanced_accuracy"] = tmo.get("balanced_accuracy")
    cf = (metrics.get("counterfactual") or {}).get(model_name, {})
    if isinstance(cf, dict):
        out["cf_on_line_greater_rate"] = cf.get("on_line_greater_rate")
        out["cf_mean_delta_on_minus_off"] = cf.get("mean_delta_on_line_minus_off_line")
        strat = cf.get("stratified", {})
        if isinstance(strat, dict):
            by_lbl = strat.get("by_orig_label", {})
            if isinstance(by_lbl, dict):
                for lbl in ("0", "1"):
                    blk = by_lbl.get(lbl, {})
                    if isinstance(blk, dict):
                        out[f"cf_orig{lbl}_on_line_greater_rate"] = blk.get("on_line_greater_rate")
    rf = (metrics.get("explainability") or {}).get("rollout_focus", {})
    if isinstance(rf, dict):
        overall = rf.get("overall", {})
        if isinstance(overall, dict):
            out["rollout_receiver_minus_passer"] = overall.get("receiver_minus_passer_mean")
            out["rollout_corridor_mean"] = overall.get("corridor_mean")
            out["rollout_nearest_defender_mean"] = overall.get("nearest_defender_mean")
    return out


def _mean_std_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_runs": 0}
    keys = sorted({k for r in rows for k in r.keys() if k not in {"seed", "report_dir"}})
    out: dict[str, Any] = {"n_runs": len(rows), "runs": rows, "summary": {}}
    for k in keys:
        vals = [r.get(k) for r in rows]
        vals_num = [float(v) for v in vals if isinstance(v, (int, float)) and v is not None]
        if not vals_num:
            continue
        arr = np.asarray(vals_num, dtype=float)
        out["summary"][k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
    return nan_to_none(out)


def _aggregate_seed_metrics(cwd: Path, report_dirs: list[Path], model_name: str, out_path: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rdir in report_dirs:
        mpath = rdir / "metrics.json"
        if not mpath.exists():
            continue
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
        row = {"report_dir": str(rdir)}
        row.update(_extract_metric_block(metrics, model_name))
        seed_guess = "".join(ch for ch in rdir.name.split("_")[-1] if ch.isdigit())
        if seed_guess:
            row["seed"] = int(seed_guess)
        rows.append(row)
    agg = _mean_std_summary(rows)
    agg["model"] = model_name
    agg["report_dirs"] = [str(p) for p in report_dirs]
    if out_path is not None:
        save_json(out_path, agg)
    return agg


def cmd_compare_models(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cwd = Path(args.cwd or ".").resolve()
    if args.build_dataset:
        _run_module("soccer_vit.train", ["build-dataset", "--config", str(cfg_path)], cwd=cwd)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        _run_module("soccer_vit.train", ["fit", "--model", m, "--config", str(cfg_path)], cwd=cwd)
    if args.eval:
        _run_module("soccer_vit.eval", ["run", "--config", str(cfg_path)], cwd=cwd)
    if args.report:
        _run_module("soccer_vit.report", ["make", "--config", str(cfg_path), "--n-samples", str(args.n_samples)], cwd=cwd)
    cfg = load_yaml(cfg_path)
    print(json.dumps({"done": True, "models": models, "reports_dir": str(_paths(cfg)["reports"])}, indent=2, ensure_ascii=False))


def cmd_seed_sweep(args: argparse.Namespace) -> None:
    base_cfg = load_yaml(args.config)
    cwd = Path(args.cwd or ".").resolve()
    model_name = args.model
    seeds = [int(s) for s in args.seeds.split(",") if str(s).strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    report_dirs: list[Path] = []
    for seed in seeds:
        suffix = f"seed{seed}"
        cfg = _override_seed_and_paths(base_cfg, seed=seed, suffix=suffix)
        tmp_cfg = _write_temp_cfg(cfg)
        if args.fit:
            _run_module("soccer_vit.train", ["fit", "--model", model_name, "--config", str(tmp_cfg)], cwd=cwd)
        if args.eval:
            _run_module("soccer_vit.eval", ["run", "--config", str(tmp_cfg)], cwd=cwd)
        if args.report:
            _run_module("soccer_vit.report", ["make", "--config", str(tmp_cfg), "--n-samples", str(args.n_samples)], cwd=cwd)
        report_dirs.append(_paths(cfg)["reports"])

    out_root = Path(args.out_dir) if args.out_dir else (_paths(base_cfg)["reports"].parent / f"{slugify(model_name)}_seed_sweep")
    out_root.mkdir(parents=True, exist_ok=True)
    agg = _aggregate_seed_metrics(cwd, report_dirs, model_name=model_name, out_path=out_root / "summary.json")
    print(json.dumps(agg, indent=2, ensure_ascii=False))


def cmd_aggregate(args: argparse.Namespace) -> None:
    report_dirs = [Path(p.strip()) for p in args.report_dirs.split(",") if p.strip()]
    agg = _aggregate_seed_metrics(Path(args.cwd or ".").resolve(), report_dirs, model_name=args.model)
    print(json.dumps(agg, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment helpers (compare models, seed sweeps, aggregation)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare-models", help="Train multiple models under one config, then eval/report")
    c.add_argument("--config", default="configs/vit_mid_linecorridor.yaml")
    c.add_argument("--models", default="baseline_strict,resnet18,vit_base")
    c.add_argument("--cwd", default=".")
    c.add_argument("--n-samples", type=int, default=8)
    c.add_argument("--build-dataset", action="store_true")
    c.add_argument("--no-eval", dest="eval", action="store_false")
    c.add_argument("--no-report", dest="report", action="store_false")
    c.set_defaults(func=cmd_compare_models, eval=True, report=True)

    s = sub.add_parser("seed-sweep", help="Repeat fit/eval/report for a model across seeds and aggregate metrics")
    s.add_argument("--config", default="configs/vit_mid_linecorridor.yaml")
    s.add_argument("--model", default="vit_base", choices=["resnet18", "vit_base"])
    s.add_argument("--seeds", default="41,42,43")
    s.add_argument("--cwd", default=".")
    s.add_argument("--out-dir", default="")
    s.add_argument("--n-samples", type=int, default=8)
    s.add_argument("--no-fit", dest="fit", action="store_false")
    s.add_argument("--no-eval", dest="eval", action="store_false")
    s.add_argument("--no-report", dest="report", action="store_false")
    s.set_defaults(func=cmd_seed_sweep, fit=True, eval=True, report=True)

    a = sub.add_parser("aggregate", help="Aggregate metrics.json across report directories")
    a.add_argument("--report-dirs", required=True, help="Comma-separated report directories")
    a.add_argument("--model", default="vit_base", choices=["resnet18", "vit_base"])
    a.add_argument("--cwd", default=".")
    a.set_defaults(func=cmd_aggregate)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
