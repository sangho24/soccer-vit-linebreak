from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .io import tracking_player_columns


@dataclass
class PitchSpec:
    length_m: float = 105.0
    width_m: float = 68.0


def norm_to_meters(x_norm: float, y_norm: float, pitch: PitchSpec = PitchSpec()) -> tuple[float, float]:
    x_m = (float(x_norm) - 0.5) * pitch.length_m
    y_m = (0.5 - float(y_norm)) * pitch.width_m
    return x_m, y_m


def meters_to_norm(x_m: float, y_m: float, pitch: PitchSpec = PitchSpec()) -> tuple[float, float]:
    x = (float(x_m) / pitch.length_m) + 0.5
    y = 0.5 - (float(y_m) / pitch.width_m)
    return x, y


def _detect_period_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"period", "half"} or "period" in lc or "half" in lc:
            return c
    return None


def _event_periods(events_df: pd.DataFrame) -> list[int]:
    period_col = _detect_period_col(events_df)
    if period_col is None:
        return [1]
    vals = pd.to_numeric(events_df[period_col], errors="coerce").dropna().astype(int)
    uniq = sorted(v for v in vals.unique() if v > 0)
    return uniq or [1]


def add_meter_coordinates_events(events_df: pd.DataFrame, pitch: PitchSpec = PitchSpec()) -> pd.DataFrame:
    out = events_df.copy()
    for prefix in ["start", "end"]:
        xcol, ycol = f"{prefix}_x", f"{prefix}_y"
        if xcol in out.columns and ycol in out.columns:
            xy = out[[xcol, ycol]].apply(pd.to_numeric, errors="coerce")
            out[f"{prefix}_x_m"] = (xy[xcol] - 0.5) * pitch.length_m
            out[f"{prefix}_y_m"] = (0.5 - xy[ycol]) * pitch.width_m
    return out


def add_meter_coordinates_tracking(tracking_df: pd.DataFrame, pitch: PitchSpec = PitchSpec()) -> pd.DataFrame:
    out = tracking_df.copy()
    for c in list(out.columns):
        if c.endswith("_x") or c.endswith("_y"):
            stem = c[:-2]
            axis = c[-1]
            if axis == "x":
                out[f"{stem}_x_m"] = (pd.to_numeric(out[c], errors="coerce") - 0.5) * pitch.length_m
            else:
                out[f"{stem}_y_m"] = (0.5 - pd.to_numeric(out[c], errors="coerce")) * pitch.width_m
    return out


def _infer_event_frame_col(events_df: pd.DataFrame) -> str | None:
    for c in ["start_frame", "frame"]:
        if c in events_df.columns:
            return c
    return None


def _first_half_window_frames(events_df: pd.DataFrame, period_value: int, seconds: float, fps: float) -> tuple[int | None, int | None]:
    period_col = _detect_period_col(events_df)
    frame_col = _infer_event_frame_col(events_df)
    if frame_col is not None:
        rows = events_df.copy()
        if period_col is not None:
            rows = rows[pd.to_numeric(rows[period_col], errors="coerce") == period_value]
        frames = pd.to_numeric(rows[frame_col], errors="coerce").dropna()
        if not frames.empty:
            start = int(frames.min())
            return start, int(start + round(seconds * fps))
    # Fallback to whole tracking range (None -> caller handles)
    return None, None


def _team_direction_sign_for_half(
    tracking_df_m: pd.DataFrame,
    events_df: pd.DataFrame,
    team: str,
    period_value: int,
    fps: float,
    window_seconds: float = 5.0,
) -> int:
    frame_lo, frame_hi = _first_half_window_frames(events_df, period_value, window_seconds, fps)
    df = tracking_df_m
    if frame_lo is not None and "frame" in df.columns:
        frames = pd.to_numeric(df["frame"], errors="coerce")
        df = df[(frames >= frame_lo) & (frames <= frame_hi)]
    player_cols = tracking_player_columns(df)
    x_cols = [x + "_m" if not x.endswith("_m") else x for _, x, _ in player_cols.get(team, [])]
    x_cols = [c for c in x_cols if c in df.columns]
    if not x_cols:
        return +1
    mean_x = pd.to_numeric(df[x_cols].stack(), errors="coerce").mean()
    if pd.isna(mean_x):
        return +1
    # Starts left (mean_x < 0) -> already attacking +x in that half.
    return +1 if mean_x < 0 else -1


def normalize_attack_direction(
    tracking_df_m: pd.DataFrame,
    events_df_m: pd.DataFrame,
    fps: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], int]]:
    """Rotate coordinates 180 deg for halves where team attacks -x.

    Returns normalized tracking/events plus direction map {(period, team): sign_before_norm}.
    sign_before_norm = +1 means no flip, -1 means flipped.
    """
    tr = tracking_df_m.copy()
    ev = events_df_m.copy()
    periods = _event_periods(ev)
    period_col = _detect_period_col(ev)
    direction_map: dict[tuple[int, str], int] = {}

    for period in periods:
        for team in ["Home", "Away"]:
            sign = _team_direction_sign_for_half(tr, ev, team, period, fps=fps)
            direction_map[(period, team)] = sign
            if sign == +1:
                continue

            # Tracking: rotate this team's players and ball only when acting as possessing team is ambiguous.
            # To avoid corrupting global scene geometry, flip the entire frame for that half in downstream event-aligned snapshot
            # via event normalization flags. Here we store direction map and flip event coordinates for the passing team only.

    # Normalize event coordinates by team/period.
    if period_col is None:
        period_series = pd.Series(1, index=ev.index)
    else:
        period_series = pd.to_numeric(ev[period_col], errors="coerce").fillna(1).astype(int)

    if "team" in ev.columns:
        team_series = ev["team"].astype(str).str.lower()
        team_norm = pd.Series("", index=ev.index, dtype=object)
        team_norm.loc[team_series.str.contains("home", na=False)] = "Home"
        team_norm.loc[team_series.str.contains("away", na=False)] = "Away"
    else:
        team_norm = pd.Series("", index=ev.index, dtype=object)

    for prefix in ["start", "end"]:
        xcol, ycol = f"{prefix}_x_m", f"{prefix}_y_m"
        if xcol not in ev.columns or ycol not in ev.columns:
            continue
        for idx in ev.index:
            team = str(team_norm.get(idx, ""))
            period = int(period_series.loc[idx])
            sign = direction_map.get((period, team), +1)
            if sign == -1:
                ev.at[idx, xcol] = -float(ev.at[idx, xcol])
                ev.at[idx, ycol] = -float(ev.at[idx, ycol])

    return tr, ev, direction_map


def normalize_tracking_snapshot_for_team(
    row: pd.Series,
    possessing_team: str,
    period: int,
    direction_map: dict[tuple[int, str], int],
) -> pd.Series:
    sign = direction_map.get((int(period), possessing_team), +1)
    if sign == +1:
        return row
    out = row.copy()
    for c in row.index:
        if c.endswith("_x_m") or c.endswith("_y_m"):
            v = row[c]
            if pd.notna(v):
                out[c] = -float(v)
    return out
