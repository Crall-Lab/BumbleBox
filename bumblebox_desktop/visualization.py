from __future__ import annotations

import math
import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd


@dataclass
class TrackingVideoRenderResult:
    video_input: str
    tracking_csv: str
    output_video: str
    frames_read: int
    frames_written: int
    detections_drawn: int
    fps: float
    width: int
    height: int


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ID": ("id", "tag_id", "bee id", "bee_id"),
    "frame": ("frame number", "frame_number"),
    "centroidX": ("centroidx", "centroid_x", "x", "centerx", "centrex"),
    "centroidY": ("centroidy", "centroid_y", "y", "centery", "centrey"),
    "frontX": ("frontx", "front_x", "headx"),
    "frontY": ("fronty", "front_y", "heady"),
}


def _normalize_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lowered = {str(col).strip().lower(): col for col in out.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in (canonical.lower(),) + aliases:
            source = lowered.get(alias.lower())
            if source and source not in rename_map:
                rename_map[source] = canonical
                break
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _stable_id_color(id_value: float) -> tuple[int, int, int]:
    if not math.isfinite(id_value):
        return (0, 200, 255)
    # Use a deterministic hue by tag ID so the same tag keeps the same color.
    hue = ((int(id_value) * 37) % 360) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return (int(blue * 255), int(green * 255), int(red * 255))


def _overlay_metrics(width: int, height: int) -> tuple[int, int, int, float, int]:
    base = max(1.0, min(width, height) / 1800.0)
    line_thickness = max(3, int(round(base * 3)))
    dot_radius = max(6, int(round(base * 5)))
    front_radius = max(4, int(round(base * 3.5)))
    label_scale = max(1.4, min(3.2, base * 1.8))
    label_thickness = max(3, int(round(base * 2.8)))
    return line_thickness, dot_radius, front_radius, label_scale, label_thickness


def render_tracking_video(
    *,
    video_path: str | Path,
    tracking_csv_path: str | Path,
    output_video_path: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
    draw_front: bool = True,
    draw_labels: bool = True,
) -> TrackingVideoRenderResult:
    video_path = Path(video_path).expanduser().resolve()
    tracking_csv_path = Path(tracking_csv_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not tracking_csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")

    if output_video_path is None:
        output_video_path = video_path.with_name(f"{video_path.stem}_tracked.mp4")
    output_video_path = Path(output_video_path).expanduser().resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracking_csv_path)
    df = _normalize_tracking_columns(df)
    required = {"frame", "centroidX", "centroidY"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Tracking CSV missing required columns after normalization: " + ", ".join(missing)
        )

    work = df.copy()
    work["frame"] = pd.to_numeric(work["frame"], errors="coerce")
    work["centroidX"] = pd.to_numeric(work["centroidX"], errors="coerce")
    work["centroidY"] = pd.to_numeric(work["centroidY"], errors="coerce")
    if "ID" in work.columns:
        work["ID"] = pd.to_numeric(work["ID"], errors="coerce")
    if "frontX" in work.columns:
        work["frontX"] = pd.to_numeric(work["frontX"], errors="coerce")
    if "frontY" in work.columns:
        work["frontY"] = pd.to_numeric(work["frontY"], errors="coerce")

    work = work.dropna(subset=["frame"])
    if work.empty:
        raise ValueError("Tracking CSV has no valid frame rows.")
    work["frame"] = work["frame"].astype(int)
    by_frame = {int(k): g for k, g in work.groupby("frame")}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frames_read = 0
    frames_written = 0
    detections_drawn = 0
    line_thickness, dot_radius, front_radius, label_scale, label_thickness = _overlay_metrics(width, height)
    try:
        while True:
            if max_frames is not None and frames_read >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_rows = by_frame.get(frames_read)
            if frame_rows is not None:
                for row in frame_rows.itertuples(index=False):
                    x = float(getattr(row, "centroidX"))
                    y = float(getattr(row, "centroidY"))
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    cx = int(round(x))
                    cy = int(round(y))
                    detections_drawn += 1

                    dot_color = (0, 200, 255)
                    if hasattr(row, "ID"):
                        id_value = float(getattr(row, "ID"))
                        if math.isfinite(id_value):
                            dot_color = _stable_id_color(id_value)

                    if draw_front and hasattr(row, "frontX") and hasattr(row, "frontY"):
                        fx = float(getattr(row, "frontX"))
                        fy = float(getattr(row, "frontY"))
                        if math.isfinite(fx) and math.isfinite(fy):
                            fxi = int(round(fx))
                            fyi = int(round(fy))
                            cv2.circle(frame, (fxi, fyi), front_radius + 2, (0, 0, 0), -1)
                            cv2.circle(frame, (fxi, fyi), front_radius, (0, 255, 0), -1)

                    cv2.circle(frame, (cx, cy), dot_radius + 2, (0, 0, 0), -1)
                    cv2.circle(frame, (cx, cy), dot_radius, dot_color, -1)

                    if draw_labels and hasattr(row, "ID"):
                        id_value = float(getattr(row, "ID"))
                        if math.isfinite(id_value):
                            label = str(int(id_value))
                            (text_w, text_h), baseline = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                label_scale,
                                label_thickness,
                            )
                            text_x = max(0, min(width - text_w, cx - (text_w // 2)))
                            text_y = max(text_h + 4, cy - dot_radius - 10)
                            cv2.putText(
                                frame,
                                label,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                label_scale,
                                (0, 0, 0),
                                label_thickness + 3,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                frame,
                                label,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                label_scale,
                                dot_color,
                                label_thickness,
                                cv2.LINE_AA,
                            )

            writer.write(frame)
            frames_read += 1
            frames_written += 1
    finally:
        cap.release()
        writer.release()

    return TrackingVideoRenderResult(
        video_input=str(video_path),
        tracking_csv=str(tracking_csv_path),
        output_video=str(output_video_path),
        frames_read=frames_read,
        frames_written=frames_written,
        detections_drawn=detections_drawn,
        fps=fps,
        width=width,
        height=height,
    )


def format_tracking_video_render_result(result: TrackingVideoRenderResult) -> str:
    lines = [
        "Tracking Video Render",
        "---------------------",
        f"Input video: {result.video_input}",
        f"Tracking CSV: {result.tracking_csv}",
        f"Output video: {result.output_video}",
        f"Frames read: {result.frames_read}",
        f"Frames written: {result.frames_written}",
        f"Detections drawn: {result.detections_drawn}",
        f"Video fps: {result.fps:.3f}",
        f"Resolution: {result.width}x{result.height}",
    ]
    return "\n".join(lines)
