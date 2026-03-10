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

    report: Dict[str, Any] = {
        "video": str(video_path),
        "video_size_bytes": int(video_path.stat().st_size) if video_path.exists() else None,
        "frame_count": None,
        "metadata_fps": None,
        "metadata_duration_s": None,
        "requested_recording_seconds": recording_seconds,
    }

    capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        metadata_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()

        metadata_duration_s = None
        if metadata_fps > 0 and frame_count > 0:
            metadata_duration_s = frame_count / metadata_fps

        report.update(
            {
                "frame_count": frame_count,
                "metadata_fps": _safe_round(metadata_fps),
                "metadata_duration_s": _safe_round(metadata_duration_s),
            }
        )
    else:
        capture.release()
        report["video_open_warning"] = f"Unable to open video with OpenCV: {video_path}"

    if report.get("video_size_bytes") is not None and int(report["video_size_bytes"]) < 1024:
        report["video_size_warning"] = (
            f"Video file is unusually small ({report['video_size_bytes']} bytes). "
            "Encoding may have failed or no frames were written."
        )

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

            metadata_fps_value = report.get("metadata_fps")
            if metadata_fps_value and actual_fps:
                drift_pct = ((actual_fps - float(metadata_fps_value)) / float(metadata_fps_value)) * 100.0
                report["actual_vs_metadata_drift_pct"] = _safe_round(drift_pct, 3)
            frame_count_value = report.get("frame_count")
            if frame_count_value is not None and int(frame_count_value) != len(timestamps):
                report["frame_count_mismatch_warning"] = (
                    "Encoded video frame count reported by OpenCV "
                    f"({int(frame_count_value)}) does not match captured timestamp count "
                    f"({len(timestamps)}). This usually means container/decoder metadata differs from the "
                    "capture-side count, not necessarily that capture failed."
                )
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

    frame_count_value = report.get("frame_count")
    metadata_fps_value = report.get("metadata_fps")
    if recording_seconds and frame_count_value:
        expected_frames = recording_seconds * (float(metadata_fps_value) if metadata_fps_value and float(metadata_fps_value) > 0 else 0)
        if expected_frames > 0:
            report["frames_vs_expected_pct"] = _safe_round((float(frame_count_value) / expected_frames) * 100.0, 3)
    elif recording_seconds and report.get("captured_frames_from_timestamps"):
        actual_frames = float(report["captured_frames_from_timestamps"])
        actual_fps_value = report.get("actual_fps_from_timestamps") or report.get("actual_fps_from_sidecar_txt")
        expected_frames = recording_seconds * (float(actual_fps_value) if actual_fps_value else 0.0)
        if expected_frames > 0:
            report["frames_vs_expected_pct"] = _safe_round((actual_frames / expected_frames) * 100.0, 3)

    return report


def write_report_json(report: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False))
    return output_path


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Video: {report['video']}")
    if report.get("video_size_bytes") is not None:
        lines.append(f"Video size (bytes): {report.get('video_size_bytes')}")
    if report.get("captured_frames_from_timestamps") is not None:
        lines.append(f"Captured frame count (timestamps): {report.get('captured_frames_from_timestamps')}")
    lines.append(f"Encoded frame count (OpenCV metadata estimate): {report.get('frame_count')}")
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
    if report.get("frame_count_mismatch_warning"):
        lines.append(f"Warning: {report['frame_count_mismatch_warning']}")
    if report.get("video_open_warning"):
        lines.append(f"Warning: {report['video_open_warning']}")
    if report.get("video_size_warning"):
        lines.append(f"Warning: {report['video_size_warning']}")

    return "\n".join(lines)
