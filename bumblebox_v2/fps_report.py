from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None


def _extract_float_tokens(line: str) -> List[float]:
    values: List[float] = []
    for token in line.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def load_timestamps(path: Path) -> List[float]:
    timestamps: List[float] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        floats = _extract_float_tokens(line)
        if floats:
            timestamps.append(floats[-1])

    return timestamps


def _safe_round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _candidate_timestamp_paths(video_path: Path) -> List[Path]:
    stem = video_path.with_suffix("")
    return [
        stem.with_name(stem.name + "_frame_timestamps.csv"),
        stem.with_name(stem.name + "_timestamps.csv"),
        stem.with_name(stem.name + "_actual_fps.txt"),
    ]


def build_fps_report(
    video_path: str | Path,
    timestamps_path: str | Path | None = None,
    recording_seconds: float | None = None,
) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for fps reporting. Install opencv-contrib-python.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    metadata_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    metadata_duration_s = None
    if metadata_fps > 0 and frame_count > 0:
        metadata_duration_s = frame_count / metadata_fps

    report: Dict[str, Any] = {
        "video": str(video_path),
        "frame_count": frame_count,
        "metadata_fps": _safe_round(metadata_fps),
        "metadata_duration_s": _safe_round(metadata_duration_s),
        "requested_recording_seconds": recording_seconds,
    }

    ts_path = Path(timestamps_path) if timestamps_path else None
    if ts_path is None:
        for candidate in _candidate_timestamp_paths(video_path):
            if candidate.exists():
                ts_path = candidate
                break

    if ts_path and ts_path.exists() and ts_path.suffix.lower() != ".txt":
        timestamps = load_timestamps(ts_path)
        if len(timestamps) >= 2:
            elapsed = timestamps[-1] - timestamps[0]
            actual_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else None

            intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b >= a]
            report.update(
                {
                    "timestamp_file": str(ts_path),
                    "captured_frames_from_timestamps": len(timestamps),
                    "elapsed_from_timestamps_s": _safe_round(elapsed),
                    "actual_fps_from_timestamps": _safe_round(actual_fps),
                    "frame_interval_mean_ms": _safe_round(mean(intervals) * 1000.0) if intervals else None,
                    "frame_interval_std_ms": _safe_round(pstdev(intervals) * 1000.0) if len(intervals) > 1 else 0.0,
                    "frame_interval_min_ms": _safe_round(min(intervals) * 1000.0) if intervals else None,
                    "frame_interval_max_ms": _safe_round(max(intervals) * 1000.0) if intervals else None,
                }
            )

            if metadata_fps > 0 and actual_fps:
                drift_pct = ((actual_fps - metadata_fps) / metadata_fps) * 100.0
                report["actual_vs_metadata_drift_pct"] = _safe_round(drift_pct, 3)
        else:
            report["timestamp_file"] = str(ts_path)
            report["timestamp_warning"] = "Timestamp file found, but fewer than 2 timestamps were parsed."

    elif ts_path and ts_path.exists() and ts_path.suffix.lower() == ".txt":
        floats = load_timestamps(ts_path)
        actual_fps = floats[0] if floats else None
        report.update(
            {
                "timestamp_file": str(ts_path),
                "actual_fps_from_sidecar_txt": _safe_round(actual_fps),
            }
        )

    if recording_seconds and frame_count > 0:
        expected_frames = recording_seconds * (metadata_fps if metadata_fps > 0 else 0)
        if expected_frames > 0:
            report["frames_vs_expected_pct"] = _safe_round((frame_count / expected_frames) * 100.0, 3)

    return report


def write_report_json(report: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False))
    return output_path


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Video: {report['video']}")
    lines.append(f"Frame count: {report.get('frame_count')}")
    lines.append(f"Metadata FPS: {report.get('metadata_fps')}")
    lines.append(f"Metadata duration (s): {report.get('metadata_duration_s')}")

    if report.get("actual_fps_from_timestamps") is not None:
        lines.append(f"Actual FPS (timestamps): {report.get('actual_fps_from_timestamps')}")
        lines.append(
            "Frame interval mean/std (ms): "
            f"{report.get('frame_interval_mean_ms')} / {report.get('frame_interval_std_ms')}"
        )
        lines.append(f"Frame interval min/max (ms): {report.get('frame_interval_min_ms')} / {report.get('frame_interval_max_ms')}")
        lines.append(f"Drift vs metadata (%): {report.get('actual_vs_metadata_drift_pct')}")
    elif report.get("actual_fps_from_sidecar_txt") is not None:
        lines.append(f"Actual FPS (sidecar): {report.get('actual_fps_from_sidecar_txt')}")

    if report.get("frames_vs_expected_pct") is not None and not math.isnan(report["frames_vs_expected_pct"]):
        lines.append(f"Frames vs expected (%): {report.get('frames_vs_expected_pct')}")

    if report.get("timestamp_warning"):
        lines.append(f"Warning: {report['timestamp_warning']}")

    return "\n".join(lines)

