from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrackingOverview:
    summary: dict[str, Any]
    frame_summary: pd.DataFrame
    id_summary: pd.DataFrame
    tracking_with_heading: pd.DataFrame


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ID": ("id", "tag_id", "bee id", "bee_id"),
    "frame": ("frame number", "frame_number"),
    "centroidX": ("centroidx", "centroid_x", "x", "centerx", "centrex"),
    "centroidY": ("centroidy", "centroid_y", "y", "centery", "centrey"),
    "frontX": ("frontx", "front_x", "headx"),
    "frontY": ("fronty", "front_y", "heady"),
}


def normalize_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lowered = {str(col).strip().lower(): col for col in out.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        candidates = (canonical.lower(),) + aliases
        for alias in candidates:
            source = lowered.get(alias.lower())
            if source and source not in rename_map:
                rename_map[source] = canonical
                break
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _numeric_id_series(df: pd.DataFrame) -> pd.Series:
    if "ID" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df["ID"], errors="coerce")


def add_heading_angle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"centroidX", "centroidY", "frontX", "frontY"}
    if not required.issubset(set(out.columns)):
        return out

    dx = pd.to_numeric(out["frontX"], errors="coerce") - pd.to_numeric(out["centroidX"], errors="coerce")
    dy = pd.to_numeric(out["frontY"], errors="coerce") - pd.to_numeric(out["centroidY"], errors="coerce")
    out["heading_angle"] = np.arctan2(dy, dx)
    out["heading_angle_deg"] = np.degrees(out["heading_angle"]) % 360.0
    return out


def build_tracking_overview(df: pd.DataFrame, include_heading: bool = True) -> TrackingOverview:
    df = normalize_tracking_columns(df)

    if df.empty:
        empty = pd.DataFrame()
        return TrackingOverview(
            summary={
                "row_count": 0,
                "unique_ids": 0,
                "frame_min": None,
                "frame_max": None,
                "frame_count": 0,
                "detections_per_frame_mean": 0.0,
                "detections_per_frame_median": 0.0,
                "detections_per_frame_p95": 0.0,
            },
            frame_summary=empty,
            id_summary=empty,
            tracking_with_heading=df.copy(),
        )

    work = df.copy()
    if "frame" not in work.columns:
        raise ValueError(
            "Tracking CSV is missing required frame column. "
            "Expected one of: frame, frame number, frame_number"
        )

    work["frame"] = pd.to_numeric(work["frame"], errors="coerce")
    work = work.dropna(subset=["frame"])
    work["frame"] = work["frame"].astype(int)

    id_numeric = _numeric_id_series(work)
    unique_ids = int(id_numeric.dropna().nunique())

    frame_counts = (
        work.groupby("frame", as_index=False)
        .size()
        .rename(columns={"size": "detections"})
        .sort_values("frame")
        .reset_index(drop=True)
    )

    per_id_work = (
        work.assign(ID_num=id_numeric)
        .dropna(subset=["ID_num"])
        .assign(ID_num=lambda d: d["ID_num"].astype(int))
    )
    if per_id_work.empty:
        per_id = pd.DataFrame(columns=["ID_num", "detections", "first_frame", "last_frame"])
    else:
        agg_map: dict[str, tuple[str, str]] = {
            "detections": ("frame", "count"),
            "first_frame": ("frame", "min"),
            "last_frame": ("frame", "max"),
        }
        if "centroidX" in per_id_work.columns:
            agg_map["mean_centroid_x"] = ("centroidX", "mean")
        if "centroidY" in per_id_work.columns:
            agg_map["mean_centroid_y"] = ("centroidY", "mean")
        per_id = (
            per_id_work.groupby("ID_num", as_index=False)
            .agg(**agg_map)
            .sort_values("ID_num")
            .reset_index(drop=True)
        )
    if not per_id.empty:
        per_id["duration_frames"] = per_id["last_frame"] - per_id["first_frame"] + 1

    summary = {
        "row_count": int(len(work)),
        "unique_ids": unique_ids,
        "frame_min": int(work["frame"].min()),
        "frame_max": int(work["frame"].max()),
        "frame_count": int(work["frame"].nunique()),
        "detections_per_frame_mean": float(frame_counts["detections"].mean()),
        "detections_per_frame_median": float(frame_counts["detections"].median()),
        "detections_per_frame_p95": float(frame_counts["detections"].quantile(0.95)),
    }

    with_heading = add_heading_angle(work) if include_heading else work
    return TrackingOverview(
        summary=summary,
        frame_summary=frame_counts,
        id_summary=per_id,
        tracking_with_heading=with_heading,
    )


def write_tracking_overview(overview: TrackingOverview, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "tracking_overview.json"
    frame_summary_path = output_dir / "frame_summary.csv"
    id_summary_path = output_dir / "id_summary.csv"
    tracking_enriched_path = output_dir / "tracking_with_heading.csv"

    summary_path.write_text(json.dumps(overview.summary, indent=2))
    overview.frame_summary.to_csv(frame_summary_path, index=False)
    overview.id_summary.to_csv(id_summary_path, index=False)
    overview.tracking_with_heading.to_csv(tracking_enriched_path, index=False)

    return {
        "summary_json": str(summary_path),
        "frame_summary_csv": str(frame_summary_path),
        "id_summary_csv": str(id_summary_path),
        "tracking_with_heading_csv": str(tracking_enriched_path),
    }
