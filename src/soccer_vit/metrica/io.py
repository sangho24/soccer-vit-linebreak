from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HEADER_CANDIDATES = list(range(0, 6))


@dataclass
class GameFiles:
    game_id: int
    event_csv: Path
    tracking_home_csv: Path
    tracking_away_csv: Path


def _score_event_columns(cols: Iterable[str]) -> int:
    s = 0
    norm = [c.strip().lower() for c in cols]
    joined = "|".join(norm)
    for token in ["start", "end", "type", "team", "frame", "time"]:
        if token in joined:
            s += 1
    return s


def _score_tracking_columns(cols: Iterable[str]) -> int:
    s = 0
    norm = [c.strip().lower() for c in cols]
    joined = "|".join(norm)
    if "frame" in joined:
        s += 3
    if "time" in joined:
        s += 2
    if "home" in joined or "away" in joined:
        s += 2
    if "ball" in joined:
        s += 1
    return s


def read_csv_auto(path: str | Path, kind: str) -> pd.DataFrame:
    path = Path(path)
    best_df: pd.DataFrame | None = None
    best_score = -1
    last_err: Exception | None = None
    for header in HEADER_CANDIDATES:
        try:
            df = pd.read_csv(path, header=header)
            df = df.dropna(axis=1, how="all")
            score = _score_event_columns(df.columns) if kind == "event" else _score_tracking_columns(df.columns)
            if score > best_score:
                best_df = df
                best_score = score
        except Exception as e:  # pragma: no cover - best-effort parser fallback
            last_err = e
    if best_df is None:
        raise ValueError(f"Failed to read {kind} csv {path}: {last_err}")
    return best_df


def _clean_name(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def standardize_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap: dict[str, str] = {}
    for c in df.columns:
        lc = _clean_name(c).lower()
        if "period" in lc or lc == "half":
            colmap[c] = "period"
        elif "start frame" in lc:
            colmap[c] = "start_frame"
        elif "end frame" in lc:
            colmap[c] = "end_frame"
        elif "start time" in lc:
            colmap[c] = "start_time_s"
        elif "end time" in lc:
            colmap[c] = "end_time_s"
        elif "subtype" in lc or "sub type" in lc:
            colmap[c] = "subtype"
        elif lc == "type" or lc.endswith(" type"):
            colmap[c] = "event_type"
        elif lc == "team" or "team" in lc:
            colmap[c] = "team"
        elif re.search(r"\bfrom\b.*\bx\b", lc) or lc in {"start x", "x start", "x"}:
            colmap[c] = "start_x"
        elif re.search(r"\bfrom\b.*\by\b", lc) or lc in {"start y", "y start"}:
            colmap[c] = "start_y"
        elif re.search(r"\bto\b.*\bx\b", lc) or lc in {"end x", "x end", "target x"}:
            colmap[c] = "end_x"
        elif re.search(r"\bto\b.*\by\b", lc) or lc in {"end y", "y end", "target y"}:
            colmap[c] = "end_y"
        elif lc == "frame":
            colmap[c] = "frame"
        elif "id" == lc or lc.endswith(" id"):
            colmap[c] = "event_id"
    out = df.rename(columns=colmap).copy()
    # Numeric coercion for common fields.
    for c in [
        "period",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "frame",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def standardize_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap: dict[str, str] = {}
    for c in df.columns:
        lc = _clean_name(c).lower()
        if lc == "frame":
            colmap[c] = "frame"
        elif lc.startswith("time"):
            colmap[c] = "time_s"
        elif lc.replace(" ", "") in {"ball_x", "ballx"}:
            colmap[c] = "ball_x"
        elif lc.replace(" ", "") in {"ball_y", "bally"}:
            colmap[c] = "ball_y"
    out = df.rename(columns=colmap).copy()
    if "frame" in out.columns:
        out["frame"] = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")
    if "time_s" in out.columns:
        out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    # Standardize player column names like Home_1_x / Away_7_y.
    renamed: dict[str, str] = {}
    for c in out.columns:
        m = re.match(r"(?i)\s*(home|away)[ _-]?(\d+)[ _-]?([xy])\s*$", str(c))
        if m:
            renamed[c] = f"{m.group(1).title()}_{m.group(2)}_{m.group(3).lower()}"
    out = out.rename(columns=renamed)
    return out


def _try_read_metrica_tracking_multiheader(path: str | Path) -> pd.DataFrame | None:
    """Parse Metrica raw tracking CSVs with 3 header rows (team / jersey / labels)."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, low_memory=False)
    if raw.shape[0] < 4 or raw.shape[1] < 6:
        return None

    row0 = raw.iloc[0].astype(str).str.strip()
    row2 = raw.iloc[2].astype(str).str.strip()
    first3 = [x.lower() for x in row2.iloc[:3].tolist()]
    if not (len(first3) >= 3 and "period" in first3[0] and "frame" in first3[1] and "time" in first3[2]):
        return None

    # Infer team tag from the top header row.
    team_tag = None
    r0_join = "|".join(row0.fillna("").tolist()).lower()
    if "home" in r0_join:
        team_tag = "Home"
    elif "away" in r0_join:
        team_tag = "Away"
    else:
        return None

    col_names: list[str] = []
    j = 0
    ncols = raw.shape[1]
    while j < ncols:
        token = str(row2.iloc[j]).strip() if j < len(row2) else ""
        token_l = token.lower()
        if j == 0 and "period" in token_l:
            col_names.append("Period")
            j += 1
            continue
        if j == 1 and "frame" in token_l:
            col_names.append("Frame")
            j += 1
            continue
        if j == 2 and "time" in token_l:
            col_names.append("Time [s]")
            j += 1
            continue
        if token_l.startswith("player"):
            m = re.match(r"player\s*(\d+)", token_l)
            pid = m.group(1) if m else "unk"
            col_names.append(f"{team_tag}_{pid}_x")
            if j + 1 < ncols:
                col_names.append(f"{team_tag}_{pid}_y")
            j += 2
            continue
        if token_l == "ball":
            col_names.append("ball_x")
            if j + 1 < ncols:
                col_names.append("ball_y")
            j += 2
            continue
        # Preserve unknown columns but make them unique and discard later if empty.
        col_names.append(f"unnamed_{j}")
        j += 1

    # Align names to raw width.
    if len(col_names) < ncols:
        col_names.extend([f"unnamed_{k}" for k in range(len(col_names), ncols)])
    elif len(col_names) > ncols:
        col_names = col_names[:ncols]

    data = raw.iloc[3:].copy().reset_index(drop=True)
    data.columns = col_names
    # Drop fully empty/unnamed columns from trailing commas.
    drop_cols = [c for c in data.columns if c.startswith("unnamed_") and data[c].isna().all()]
    if drop_cols:
        data = data.drop(columns=drop_cols)
    return data


def discover_sample_games(root_dir: str | Path) -> list[GameFiles]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Metrica root not found: {root}")

    game_map: dict[int, dict[str, Path]] = {}
    for p in root.rglob("*.csv"):
        name_l = p.name.lower()
        m = re.search(r"sample[_ ]game[_ ](\d+)", str(p).lower())
        if not m:
            continue
        gid = int(m.group(1))
        slots = game_map.setdefault(gid, {})
        if "event" in name_l:
            slots["event"] = p
        elif "tracking" in name_l and "home" in name_l:
            slots["home"] = p
        elif "tracking" in name_l and "away" in name_l:
            slots["away"] = p

    out: list[GameFiles] = []
    for gid, slots in sorted(game_map.items()):
        missing = [k for k in ["event", "home", "away"] if k not in slots]
        if missing:
            continue
        out.append(
            GameFiles(
                game_id=gid,
                event_csv=slots["event"],
                tracking_home_csv=slots["home"],
                tracking_away_csv=slots["away"],
            )
        )
    if not out:
        raise FileNotFoundError(f"No Metrica sample game files found under {root}")
    return out


def load_tracking_game(home_csv: str | Path, away_csv: str | Path) -> pd.DataFrame:
    home_raw = _try_read_metrica_tracking_multiheader(home_csv)
    away_raw = _try_read_metrica_tracking_multiheader(away_csv)
    home = standardize_tracking_columns(home_raw if home_raw is not None else read_csv_auto(home_csv, kind="tracking"))
    away = standardize_tracking_columns(away_raw if away_raw is not None else read_csv_auto(away_csv, kind="tracking"))

    join_cols = [c for c in ["frame", "time_s"] if c in home.columns and c in away.columns]
    if not join_cols:
        raise ValueError(
            "Tracking CSVs missing frame/time columns. Home cols: "
            f"{list(home.columns)} | Away cols: {list(away.columns)}"
        )

    # Keep only frame/time + home player/ball from home, and away player cols from away.
    home_keep = join_cols + [c for c in home.columns if c.startswith("Home_") or c in {"ball_x", "ball_y"}]
    away_keep = join_cols + [c for c in away.columns if c.startswith("Away_")]
    merged = pd.merge(home[home_keep], away[away_keep], on=join_cols, how="inner")
    merged = merged.sort_values(join_cols).reset_index(drop=True)
    return merged


def load_event_game(event_csv: str | Path) -> pd.DataFrame:
    df = standardize_event_columns(read_csv_auto(event_csv, kind="event"))
    return df


def tracking_player_columns(tracking_df: pd.DataFrame) -> dict[str, list[tuple[str, str, str]]]:
    out: dict[str, list[tuple[str, str, str]]] = {"Home": [], "Away": []}
    players: dict[tuple[str, str], dict[str, str]] = {}
    for c in tracking_df.columns:
        m = re.match(r"^(Home|Away)_(\d+)_([xy])$", str(c))
        if not m:
            continue
        key = (m.group(1), m.group(2))
        players.setdefault(key, {})[m.group(3)] = c
    for (team, pid), cols in players.items():
        if "x" in cols and "y" in cols:
            out[team].append((pid, cols["x"], cols["y"]))
    for team in out:
        out[team].sort(key=lambda t: int(t[0]))
    return out
