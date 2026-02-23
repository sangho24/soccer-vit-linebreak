from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labeling.linebreak import LineBreakParams, compute_baseline_features, label_line_break
from .metrica.align import build_tracking_index, infer_frame_for_event, lookup_tracking_row
from .metrica.io import discover_sample_games, load_event_game, load_tracking_game, tracking_player_columns
from .metrica.normalize import (
    PitchSpec,
    add_meter_coordinates_events,
    add_meter_coordinates_tracking,
    normalize_attack_direction,
    normalize_tracking_snapshot_for_team,
)
from .raster.render import RasterSpec, render_snapshot
from .utils import ensure_dirs


@dataclass
class SampleRecord:
    sample_id: str
    game_id: int
    frame_id: int
    period: int
    team: str
    label: int
    bypassed_count: int
    forward_m: float
    pass_length_m: float
    min_def_dist_to_line_m: float | None
    passer_x_m: float
    passer_y_m: float
    receiver_x_m: float
    receiver_y_m: float
    ball_x_m: float | None
    ball_y_m: float | None
    attacking_xy_json: str
    defending_xy_json: str


PASS_KEYWORDS = ("pass",)
EXCLUDE_SUBTYPE_KEYWORDS = ("clear", "clearance")


def _team_name_from_event(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if "home" in s:
        return "Home"
    if "away" in s:
        return "Away"
    return None


def _is_pass_event(row: pd.Series) -> bool:
    et = str(row.get("event_type", "")).lower()
    st = str(row.get("subtype", "")).lower()
    if not any(k in et for k in PASS_KEYWORDS):
        return False
    if any(k in st for k in EXCLUDE_SUBTYPE_KEYWORDS):
        return False
    return True


def _period_of_row(row: pd.Series) -> int:
    try:
        return int(float(row.get("period", 1)))
    except Exception:
        return 1


def _extract_players_from_snapshot(row_m: pd.Series) -> dict[str, np.ndarray]:
    cols = tracking_player_columns(pd.DataFrame([row_m]))
    out: dict[str, np.ndarray] = {}
    for team in ["Home", "Away"]:
        pts = []
        for _, xcol, ycol in cols.get(team, []):
            x_m = row_m.get(f"{xcol}_m")
            y_m = row_m.get(f"{ycol}_m")
            if pd.notna(x_m) and pd.notna(y_m):
                pts.append((float(x_m), float(y_m)))
        out[team] = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)
    return out


def _json_points(arr: np.ndarray) -> str:
    return json.dumps(np.asarray(arr, dtype=float).tolist(), ensure_ascii=False)


def _build_dataset_from_games(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    pitch_cfg = cfg.get("pitch", {})
    pitch = PitchSpec(
        length_m=float(pitch_cfg.get("length_m", 105.0)),
        width_m=float(pitch_cfg.get("width_m", 68.0)),
    )
    metrica_cfg = cfg.get("metrica", {})
    labeling_cfg = cfg.get("labeling", {})
    raster_cfg = cfg.get("raster", {})
    fps_default = float(metrica_cfg.get("fps_default", 25.0))
    params = LineBreakParams(
        min_forward_m=float(labeling_cfg.get("min_forward_m", 5.0)),
        corridor_w_m=float(labeling_cfg.get("corridor_w_m", 8.0)),
        k_bypassed=int(labeling_cfg.get("k_bypassed", labeling_cfg.get("k_bypassed", labeling_cfg.get("K", 2)))),
        drop_non_forward=bool(labeling_cfg.get("drop_non_forward", False)),
    )
    raster = RasterSpec(
        size=int(raster_cfg.get("size", 224)),
        sigma_m=float(raster_cfg.get("sigma_m", 1.2)),
        line_sigma_m=float(raster_cfg.get("line_sigma_m", 0.8)),
        corridor_w_m=float(raster_cfg.get("corridor_w_m", params.corridor_w_m)),
        pitch_length_m=pitch.length_m,
        pitch_width_m=pitch.width_m,
        channels=int(raster_cfg.get("channels", 5)),
    )

    game_files = discover_sample_games(metrica_cfg.get("root_dir", "data/external/sample-data"))
    wanted_games = set(int(g) for g in metrica_cfg.get("sample_games", [])) if metrica_cfg.get("sample_games") else None

    images: list[np.ndarray] = []
    labels: list[int] = []
    feature_rows: list[list[float]] = []
    feature_names = [
        "forward_m",
        "pass_length_m",
        "pass_angle_rad",
        "corridor_def_count",
        "min_def_dist_to_line_m",
    ]
    records: list[SampleRecord] = []

    for gf in game_files:
        if wanted_games is not None and gf.game_id not in wanted_games:
            continue
        events = load_event_game(gf.event_csv)
        tracking = load_tracking_game(gf.tracking_home_csv, gf.tracking_away_csv)
        events = add_meter_coordinates_events(events, pitch=pitch)
        tracking_m = add_meter_coordinates_tracking(tracking, pitch=pitch)
        tracking_m, events_m, direction_map = normalize_attack_direction(tracking_m, events, fps=fps_default)
        t_index = build_tracking_index(tracking_m)

        if not {"start_x_m", "start_y_m", "end_x_m", "end_y_m"}.issubset(events_m.columns):
            raise ValueError(
                "Event CSV missing pass coordinate columns after standardization. "
                f"Columns: {list(events_m.columns)}"
            )

        pass_events = events_m[events_m.apply(_is_pass_event, axis=1)].copy()
        for ridx, ev in pass_events.iterrows():
            team = _team_name_from_event(ev.get("team"))
            if team is None:
                continue
            try:
                frame_id = infer_frame_for_event(ev, fps=fps_default)
            except ValueError:
                continue
            snapshot = lookup_tracking_row(t_index, frame_id)
            if snapshot is None:
                continue

            period = _period_of_row(ev)
            snapshot = normalize_tracking_snapshot_for_team(snapshot, team, period, direction_map)
            players = _extract_players_from_snapshot(snapshot)
            attacking_xy = players[team]
            defending_xy = players["Away" if team == "Home" else "Home"]

            p0 = np.array([ev.get("start_x_m", np.nan), ev.get("start_y_m", np.nan)], dtype=float)
            p1 = np.array([ev.get("end_x_m", np.nan), ev.get("end_y_m", np.nan)], dtype=float)
            if not np.isfinite(p0).all() or not np.isfinite(p1).all():
                continue

            lb = label_line_break(p0, p1, defending_xy, params=params)
            if lb.dropped:
                continue

            feats = compute_baseline_features(p0, p1, defending_xy, params)
            feature_rows.append([feats[k] for k in feature_names])
            labels.append(lb.label)

            ball_xy = None
            bx = snapshot.get("ball_x_m")
            by = snapshot.get("ball_y_m")
            if pd.notna(bx) and pd.notna(by):
                ball_xy = (float(bx), float(by))

            img = render_snapshot(
                attacking_xy=attacking_xy,
                defending_xy=defending_xy,
                ball_xy=ball_xy,
                passer_xy=(float(p0[0]), float(p0[1])),
                receiver_xy=(float(p1[0]), float(p1[1])),
                spec=raster,
            )
            images.append(img)

            sample_id = f"g{gf.game_id}_r{int(ridx)}_f{frame_id}"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    game_id=gf.game_id,
                    frame_id=frame_id,
                    period=period,
                    team=team,
                    label=lb.label,
                    bypassed_count=lb.bypassed_count,
                    forward_m=lb.forward_m,
                    pass_length_m=lb.pass_length_m,
                    min_def_dist_to_line_m=lb.min_def_dist_to_line_m,
                    passer_x_m=float(p0[0]),
                    passer_y_m=float(p0[1]),
                    receiver_x_m=float(p1[0]),
                    receiver_y_m=float(p1[1]),
                    ball_x_m=float(ball_xy[0]) if ball_xy is not None else None,
                    ball_y_m=float(ball_xy[1]) if ball_xy is not None else None,
                    attacking_xy_json=_json_points(attacking_xy),
                    defending_xy_json=_json_points(defending_xy),
                )
            )

    if not records:
        raise RuntimeError("No pass samples were generated. Check data path and CSV column parsing.")

    X_img = np.stack(images, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    X_feat = np.asarray(feature_rows, dtype=np.float32)
    meta_df = pd.DataFrame([asdict(r) for r in records])
    arrays = {
        "images": X_img,
        "labels": y,
        "features": X_feat,
        "feature_names": np.asarray(feature_names, dtype=object),
        "sample_ids": meta_df["sample_id"].astype(str).to_numpy(dtype=object),
    }
    return arrays, meta_df


def build_and_save_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    arrays, meta_df = _build_dataset_from_games(cfg)
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    ensure_dirs(processed_dir)

    npz_path = processed_dir / "dataset.npz"
    csv_path = processed_dir / "samples.csv"
    np.savez_compressed(npz_path, **arrays)
    meta_df.to_csv(csv_path, index=False)

    summary = {
        "n_samples": int(len(meta_df)),
        "pos_rate": float(meta_df["label"].mean()) if len(meta_df) else 0.0,
        "npz_path": str(npz_path),
        "samples_csv": str(csv_path),
    }
    with open(processed_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def load_saved_dataset(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    npz_path = processed_dir / "dataset.npz"
    csv_path = processed_dir / "samples.csv"
    if not npz_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Dataset artifacts not found under {processed_dir}. Run build-dataset first.")
    npz = np.load(npz_path, allow_pickle=True)
    arrays = {k: npz[k] for k in npz.files}
    meta_df = pd.read_csv(csv_path)
    return arrays, meta_df
