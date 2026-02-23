from __future__ import annotations

import pandas as pd


def infer_frame_for_event(event_row: pd.Series, fps: float = 25.0) -> int:
    if "start_frame" in event_row and pd.notna(event_row.get("start_frame")):
        return int(round(float(event_row["start_frame"])))
    if "frame" in event_row and pd.notna(event_row.get("frame")):
        return int(round(float(event_row["frame"])))
    if "start_time_s" in event_row and pd.notna(event_row.get("start_time_s")):
        return int(round(float(event_row["start_time_s"]) * fps))
    raise ValueError(f"Could not infer frame from event row. Available columns: {list(event_row.index)}")


def build_tracking_index(tracking_df: pd.DataFrame) -> pd.DataFrame:
    if "frame" in tracking_df.columns:
        idx = pd.to_numeric(tracking_df["frame"], errors="coerce")
        out = tracking_df.copy()
        out["_frame_key"] = idx.round().astype("Int64")
        return out.dropna(subset=["_frame_key"]).set_index("_frame_key", drop=False)
    if "time_s" in tracking_df.columns:
        out = tracking_df.copy().set_index("time_s", drop=False)
        return out
    raise ValueError(f"Tracking dataframe missing frame/time columns. Columns: {list(tracking_df.columns)}")


def lookup_tracking_row(indexed_tracking_df: pd.DataFrame, frame_id: int) -> pd.Series | None:
    if frame_id in indexed_tracking_df.index:
        row = indexed_tracking_df.loc[frame_id]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row
    return None
