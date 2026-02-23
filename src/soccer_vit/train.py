from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .data_pipeline import build_and_save_dataset, load_saved_dataset
from .input_channels import select_image_channels
from .metrics import binary_metrics, nan_to_none
from .models.baselines import fit_logistic_baseline, save_baseline
from .utils import ensure_dirs, load_yaml, set_seed


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    p = cfg.get("paths", {})
    return {
        "processed": Path(p.get("processed_dir", "data/processed")),
        "reports": Path(p.get("reports_dir", "reports")),
        "figures": Path(p.get("figures_dir", "reports/figures")),
        "models": Path(p.get("models_dir", "reports/models")),
    }


def _ensure_split(arrays: dict[str, np.ndarray], cfg: dict[str, Any], processed_dir: Path) -> dict[str, np.ndarray]:
    split_path = processed_dir / "splits.npz"
    if split_path.exists():
        s = np.load(split_path)
        return {k: s[k] for k in s.files}

    y = arrays["labels"].astype(int)
    idx = np.arange(len(y))
    train_cfg = cfg.get("train", {})
    test_size = float(train_cfg.get("test_size", 0.2))
    val_size = float(train_cfg.get("val_size", 0.1))
    random_state = int(train_cfg.get("random_state", 42))

    tr_idx, te_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    if val_size > 0:
        rel_val = val_size / max(1e-8, (1.0 - test_size))
        tr_idx, va_idx = train_test_split(
            tr_idx,
            test_size=rel_val,
            random_state=random_state,
            stratify=y[tr_idx] if len(np.unique(y[tr_idx])) > 1 else None,
        )
    else:
        va_idx = np.array([], dtype=int)

    np.savez(split_path, train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx)
    return {"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx}


def _subsample_indices(idx: np.ndarray, y_all: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    y = y_all[idx]
    pos = idx[y == 1]
    neg = idx[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        chosen = rng.choice(idx, size=max_n, replace=False)
        return np.sort(chosen)
    n_pos = max(1, int(round(max_n * (len(pos) / len(idx)))))
    n_pos = min(n_pos, len(pos), max_n - 1)
    n_neg = max_n - n_pos
    n_neg = min(n_neg, len(neg))
    if n_pos + n_neg < max_n:
        # Fill any shortfall from remaining indices.
        chosen = np.concatenate([
            rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
            rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
        ])
        remaining = np.setdiff1d(idx, chosen, assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(max_n - len(chosen), len(remaining)), replace=False)
            chosen = np.concatenate([chosen, extra])
        return np.sort(chosen)
    chosen = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ])
    return np.sort(chosen)


def _baseline_feature_subset(feature_names: list[str], variant: str) -> list[int]:
    idx_map = {name: i for i, name in enumerate(feature_names)}
    if variant in {"baseline", "baseline_rule_like"}:
        keep = feature_names
    elif variant == "baseline_strict":
        # Remove geometry-near-label features to reduce rule leakage.
        drop = {"corridor_def_count", "min_def_dist_to_line_m"}
        keep = [f for f in feature_names if f not in drop]
    else:
        raise ValueError(f"Unknown baseline variant: {variant}")
    missing = [f for f in keep if f not in idx_map]
    if missing:
        raise ValueError(f"Missing baseline features in dataset: {missing}")
    return [idx_map[f] for f in keep]


def _fit_baseline(cfg: dict[str, Any], variant: str = "baseline") -> dict[str, Any]:
    arrays, meta = load_saved_dataset(cfg)
    p = _paths(cfg)
    ensure_dirs(p["models"], p["reports"])
    splits = _ensure_split(arrays, cfg, p["processed"])

    X_full = arrays["features"].astype(np.float32)
    y = arrays["labels"].astype(int)
    tr, te = splits["train_idx"], splits["test_idx"]
    train_cfg = cfg.get("train", {})
    feature_names_all = [str(x) for x in arrays.get("feature_names", [])]
    feat_idx = _baseline_feature_subset(feature_names_all, variant)
    X = X_full[:, feat_idx]
    feat_names = [feature_names_all[i] for i in feat_idx]

    model = fit_logistic_baseline(
        X[tr],
        y[tr],
        max_iter=int(train_cfg.get("baseline_max_iter", 1000)),
        class_weight=train_cfg.get("class_weight", "balanced"),
    )
    model.feature_names = feat_names
    save_baseline(model, p["models"] / f"{variant}.pkl")
    if variant == "baseline":
        # Keep legacy alias path for downstream tooling.
        save_baseline(model, p["models"] / "baseline.pkl")

    prob_te = model.predict_proba(X[te])
    m = binary_metrics(y[te], prob_te)
    out = {
        "model": variant,
        "feature_names": feat_names,
        "metrics": nan_to_none(m),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }

    pred_rows = meta.iloc[te].copy()
    pred_rows["y_prob"] = prob_te
    pred_rows.to_csv(p["models"] / f"{variant}_test_predictions.csv", index=False)
    with open(p["models"] / f"{variant}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _device_from_cfg(cfg: dict[str, Any]):  # pragma: no cover - torch dependent
    import torch

    dev = str(cfg.get("train", {}).get("device", "auto"))
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def _make_torch_dataloaders(arrays: dict[str, np.ndarray], splits: dict[str, np.ndarray], cfg: dict[str, Any]):  # pragma: no cover - torch dependent
    import torch
    from torch.utils.data import DataLoader, Dataset

    class NpzDataset(Dataset):
        def __init__(self, X, y, idx):
            self.X = X[idx]
            self.y = y[idx]

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(int(self.y[i]), dtype=torch.long)

    X = arrays["images"].astype(np.float32)
    y = arrays["labels"].astype(int)
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    seed = int(cfg.get("seed", cfg.get("train", {}).get("random_state", 42)))
    train_idx = _subsample_indices(
        splits["train_idx"], y, cfg.get("train", {}).get("max_train_samples"), seed=seed
    )
    val_base_idx = splits["val_idx"] if len(splits["val_idx"]) else splits["test_idx"]
    val_idx = _subsample_indices(
        val_base_idx, y, cfg.get("train", {}).get("max_val_samples"), seed=seed + 1
    )
    test_idx = _subsample_indices(
        splits["test_idx"], y, cfg.get("train", {}).get("max_test_samples"), seed=seed + 2
    )
    train_ds = NpzDataset(X, y, train_idx)
    val_ds = NpzDataset(X, y, val_idx)
    test_ds = NpzDataset(X, y, test_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
    )


def _train_torch_model(model_name: str, cfg: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - torch dependent
    import torch
    import torch.nn.functional as F

    from .models.cnn import create_resnet18
    from .models.vit import ViTBuildConfig, create_vit_model, freeze_backbone_for_linear_probe, unfreeze_last_blocks

    arrays, meta = load_saved_dataset(cfg)
    arrays = dict(arrays)
    arrays["images"], selected_channels = select_image_channels(arrays["images"], cfg)
    p = _paths(cfg)
    ensure_dirs(p["models"], p["reports"])
    splits = _ensure_split(arrays, cfg, p["processed"])
    train_loader, val_loader, test_loader, used_idx = _make_torch_dataloaders(arrays, splits, cfg)
    y = arrays["labels"].astype(int)
    device = _device_from_cfg(cfg)

    train_cfg = cfg.get("train", {})
    pretrained_backbone = bool(train_cfg.get("pretrained_backbone", True))

    if model_name == "resnet18":
        model = create_resnet18(in_chans=arrays["images"].shape[1], num_classes=2, pretrained=pretrained_backbone)
    elif model_name == "vit_base":
        model = create_vit_model(
            ViTBuildConfig(
                model_name="vit_base_patch16_224",
                in_chans=int(arrays["images"].shape[1]),
                num_classes=2,
                pretrained=pretrained_backbone,
            )
        )
        freeze_backbone_for_linear_probe(model)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    pos_weight = None
    n_pos = max(1, int(y[used_idx["train_idx"]].sum()))
    n_neg = max(1, int(len(used_idx["train_idx"]) - n_pos))
    if n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        probs_all = []
        y_all = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if logits.ndim == 2 and logits.shape[1] == 2:
                # weighted BCE on positive logit difference
                logits_pos = logits[:, 1] - logits[:, 0]
            else:
                logits_pos = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits_pos, yb.float(), pos_weight=pos_weight)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * len(yb)
            probs = torch.sigmoid(logits_pos).detach().cpu().numpy()
            probs_all.append(probs)
            y_all.append(yb.detach().cpu().numpy())
        y_cat = np.concatenate(y_all) if y_all else np.array([])
        p_cat = np.concatenate(probs_all) if probs_all else np.array([])
        met = binary_metrics(y_cat, p_cat) if len(y_cat) else {}
        met["loss"] = total_loss / max(1, len(y_cat))
        return met, y_cat, p_cat

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    best_state = None
    best_val = -1.0
    history = []

    epochs_a = int(train_cfg.get("epochs_stage_a", 5))
    epochs_b = int(train_cfg.get("epochs_stage_b", 5)) if model_name == "vit_base" else 0

    for epoch in range(epochs_a):
        tr_m, *_ = run_epoch(train_loader, train=True)
        va_m, *_ = run_epoch(val_loader, train=False)
        history.append({"stage": "A", "epoch": epoch + 1, "train": tr_m, "val": va_m})
        score = float(va_m.get("auroc") or -1.0)
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if model_name == "vit_base" and epochs_b > 0:
        unfreeze_last_blocks(model, n_blocks=2)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr * 0.5, weight_decay=wd)
        for epoch in range(epochs_b):
            tr_m, *_ = run_epoch(train_loader, train=True)
            va_m, *_ = run_epoch(val_loader, train=False)
            history.append({"stage": "B", "epoch": epoch + 1, "train": tr_m, "val": va_m})
            score = float(va_m.get("auroc") or -1.0)
            if score > best_val:
                best_val = score
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_m, te_y, te_p = run_epoch(test_loader, train=False)
    ckpt = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "config": cfg,
        "image_channels": int(arrays["images"].shape[1]),
    }
    torch.save(ckpt, p["models"] / f"{model_name}.pt")

    te_idx = used_idx["test_idx"]
    pred_rows = meta.iloc[te_idx].copy()
    pred_rows["y_prob"] = te_p
    pred_rows.to_csv(p["models"] / f"{model_name}_test_predictions.csv", index=False)

    out = {
        "model": model_name,
        "selected_channels": selected_channels,
        "metrics": nan_to_none(te_m),
        "history": nan_to_none(history),
        "n_train": int(len(used_idx["train_idx"])),
        "n_test": int(len(te_idx)),
    }
    with open(p["models"] / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def cmd_build_dataset(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    summary = build_and_save_dataset(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_fit(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    model_name = args.model
    if model_name in {"baseline", "baseline_rule_like", "baseline_strict"}:
        out = _fit_baseline(cfg, variant=model_name)
    else:
        if not _torch_available():
            raise RuntimeError("torch/torchvision/timm not available. Install vision deps to train CNN/ViT.")
        out = _train_torch_model(model_name, cfg)
    print(json.dumps(nan_to_none(out), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/build dataset for soccer line-break project")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-dataset", help="Build raster/features dataset from Metrica CSV")
    p_build.add_argument("--config", default="configs/default.yaml")
    p_build.set_defaults(func=cmd_build_dataset)

    p_fit = sub.add_parser("fit", help="Train a model")
    p_fit.add_argument("--config", default="configs/default.yaml")
    p_fit.add_argument(
        "--model",
        choices=["baseline", "baseline_rule_like", "baseline_strict", "resnet18", "vit_base"],
        required=True,
    )
    p_fit.set_defaults(func=cmd_fit)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
