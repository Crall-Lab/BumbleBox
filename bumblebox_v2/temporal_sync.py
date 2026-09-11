from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class TemporalOffsetEstimate:
    reference_stream: str
    sensor_stream: str
    offset_seconds: float
    equivalent_reference_frames: Optional[float]
    peak_correlation: float
    residual_jitter_seconds: Optional[float]
    matched_transition_count: int
    sample_hz: float
    max_lag_seconds: float
    convention: str = "reference_time = sensor_time + offset_seconds"


def _timestamp_rows(path: Path) -> tuple[list[float], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for key in ("captured_unix_s", "host_receive_unix_s", "time_s"):
        values = []
        try:
            for row in rows:
                raw = str(row.get(key) or "").strip()
                if raw:
                    values.append(float(raw))
        except ValueError:
            values = []
        if values:
            return values, key
    raise ValueError(f"No supported timestamp column was found in {path}")


def _crop_normalized(frame: Any, roi: Optional[tuple[float, float, float, float]]) -> Any:
    if roi is None:
        return frame
    x, y, width, height = roi
    frame_height, frame_width = frame.shape[:2]
    x0 = max(0, min(frame_width - 1, int(round(x * frame_width))))
    y0 = max(0, min(frame_height - 1, int(round(y * frame_height))))
    x1 = max(x0 + 1, min(frame_width, int(round((x + width) * frame_width))))
    y1 = max(y0 + 1, min(frame_height, int(round((y + height) * frame_height))))
    return frame[y0:y1, x0:x1]


def _motion_trace_from_video(
    path: Path, roi: Optional[tuple[float, float, float, float]]
) -> list[float]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"OpenCV and NumPy are required for timing-cue analysis: {exc}") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for timing-cue analysis: {path}")
    trace: list[float] = []
    previous = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame = _crop_normalized(frame, roi)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            scale = min(1.0, 240.0 / max(gray.shape[:2]))
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = gray.astype(np.float32)
            trace.append(0.0 if previous is None else float(np.mean(np.abs(gray - previous))))
            previous = gray
    finally:
        capture.release()
    if len(trace) < 3:
        raise RuntimeError(f"Fewer than three frames could be decoded from {path}")
    return trace


def _motion_trace_from_array(
    path: Path, roi: Optional[tuple[float, float, float, float]]
) -> list[float]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"NumPy is required for timing-cue analysis: {exc}") from exc
    frames = np.load(path, mmap_mode="r")
    if frames.ndim < 3 or int(frames.shape[0]) < 3:
        raise RuntimeError(f"Expected a frame stack in {path}, got shape {frames.shape}")
    trace: list[float] = []
    previous = None
    for frame in frames:
        current = _crop_normalized(frame, roi)
        stride = max(1, max(current.shape[:2]) // 240)
        current = current[::stride, ::stride].astype(np.float32)
        trace.append(0.0 if previous is None else float(np.mean(np.abs(current - previous))))
        previous = current
    return trace


def _normalize(values: Any) -> Any:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-9, 1.4826 * mad, float(np.std(values)))
    return (values - median) / scale


def _peak_times(times: Any, values: Any, minimum_spacing_s: float) -> list[float]:
    import numpy as np

    normalized = _normalize(values)
    threshold = max(1.0, float(np.median(normalized) + 1.5 * np.std(normalized)))
    candidates = [
        index
        for index in range(1, len(normalized) - 1)
        if normalized[index] >= threshold
        and normalized[index] >= normalized[index - 1]
        and normalized[index] >= normalized[index + 1]
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: float(normalized[item]), reverse=True):
        if all(abs(float(times[index]) - float(times[other])) >= minimum_spacing_s for other in selected):
            selected.append(index)
    return sorted(float(times[index]) for index in selected)


def estimate_trace_offset(
    reference_times: Sequence[float],
    reference_values: Sequence[float],
    sensor_times: Sequence[float],
    sensor_values: Sequence[float],
    *,
    reference_stream: str = "rgb",
    sensor_stream: str,
    max_lag_seconds: float = 2.0,
    sample_hz: Optional[float] = None,
) -> TemporalOffsetEstimate:
    import numpy as np

    ref_t = np.asarray(reference_times, dtype=np.float64)
    ref_v = np.asarray(reference_values, dtype=np.float64)
    sensor_t = np.asarray(sensor_times, dtype=np.float64)
    sensor_v = np.asarray(sensor_values, dtype=np.float64)
    ref_count = min(len(ref_t), len(ref_v))
    sensor_count = min(len(sensor_t), len(sensor_v))
    ref_t, ref_v = ref_t[:ref_count], ref_v[:ref_count]
    sensor_t, sensor_v = sensor_t[:sensor_count], sensor_v[:sensor_count]
    if ref_count < 3 or sensor_count < 3:
        raise ValueError("At least three timestamped frames are required in each stream")
    if sample_hz is None:
        intervals = np.concatenate((np.diff(ref_t), np.diff(sensor_t)))
        intervals = intervals[intervals > 0]
        sample_hz = min(120.0, max(10.0, 2.0 / float(np.median(intervals))))
    sample_hz = float(sample_hz)
    step = 1.0 / sample_hz
    overlap_start = max(float(ref_t[0]), float(sensor_t[0])) + max_lag_seconds
    overlap_end = min(float(ref_t[-1]), float(sensor_t[-1])) - max_lag_seconds
    if overlap_end - overlap_start < max(0.25, 3 * step):
        overlap_start = max(float(ref_t[0]), float(sensor_t[0]))
        overlap_end = min(float(ref_t[-1]), float(sensor_t[-1]))
    grid = np.arange(overlap_start, overlap_end + step * 0.5, step)
    if len(grid) < 3:
        raise ValueError("The timestamp files do not contain enough overlapping time")
    ref_grid = _normalize(np.interp(grid, ref_t, ref_v))
    offsets = np.arange(-max_lag_seconds, max_lag_seconds + step * 0.5, step)
    correlations = []
    for offset in offsets:
        sensor_grid = _normalize(np.interp(grid - offset, sensor_t, sensor_v))
        ref_centered = ref_grid - float(np.mean(ref_grid))
        sensor_centered = sensor_grid - float(np.mean(sensor_grid))
        denominator = float(
            np.sqrt(np.sum(ref_centered * ref_centered) * np.sum(sensor_centered * sensor_centered))
        )
        correlations.append(
            float(np.sum(ref_centered * sensor_centered) / denominator)
            if denominator > 1e-12
            else -1.0
        )
    best_index = int(np.argmax(correlations))
    best_offset = float(offsets[best_index])
    if 0 < best_index < len(offsets) - 1:
        left, center, right = correlations[best_index - 1 : best_index + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            best_offset += 0.5 * (left - right) / denominator * step

    min_spacing = max(2.0 * step, 0.05)
    ref_peaks = _peak_times(ref_t, ref_v, min_spacing)
    sensor_peaks = _peak_times(sensor_t, sensor_v, min_spacing)
    residuals = []
    for sensor_peak in sensor_peaks:
        corrected = sensor_peak + best_offset
        if ref_peaks:
            residual = min((ref_peak - corrected for ref_peak in ref_peaks), key=abs)
            if abs(residual) <= max(0.25, 3.0 * step):
                residuals.append(float(residual))
    jitter = float(np.std(residuals)) if len(residuals) >= 2 else None
    ref_intervals = np.diff(ref_t)
    ref_fps = 1.0 / float(np.median(ref_intervals[ref_intervals > 0])) if np.any(ref_intervals > 0) else None
    return TemporalOffsetEstimate(
        reference_stream=reference_stream,
        sensor_stream=sensor_stream,
        offset_seconds=best_offset,
        equivalent_reference_frames=(best_offset * ref_fps if ref_fps else None),
        peak_correlation=float(correlations[best_index]),
        residual_jitter_seconds=jitter,
        matched_transition_count=len(residuals),
        sample_hz=sample_hz,
        max_lag_seconds=float(max_lag_seconds),
    )


def _stream_source(summary: dict[str, Any], stream: str) -> tuple[Optional[Path], Optional[Path], str]:
    if stream == "rgb":
        return (
            Path(summary["video_path"]) if summary.get("video_path") else None,
            Path(summary["timestamp_path"]) if summary.get("timestamp_path") else None,
            "video",
        )
    if stream == "thermal":
        source = summary.get("thermal_raw_npy_path") or summary.get("thermal_preview_video_path")
        return (
            Path(source) if source else None,
            Path(summary["thermal_timestamp_path"]) if summary.get("thermal_timestamp_path") else None,
            "array" if source and str(source).lower().endswith(".npy") else "video",
        )
    source = summary.get("realsense_raw_depth_npy_path") or summary.get("realsense_color_video_path")
    return (
        Path(source) if source else None,
        Path(summary["realsense_timestamp_path"]) if summary.get("realsense_timestamp_path") else None,
        "array" if source and str(source).lower().endswith(".npy") else "video",
    )


def analyze_session_sync(
    summary_path: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    max_lag_seconds: float = 2.0,
    roi: Optional[tuple[float, float, float, float]] = None,
) -> dict[str, Any]:
    summary_file = Path(summary_path).expanduser().resolve()
    summary = json.loads(summary_file.read_text())
    sources = {}
    traces = {}
    timestamp_domains = {}
    errors = []
    for stream in ("rgb", "thermal", "realsense"):
        source, timestamps, kind = _stream_source(summary, stream)
        if source is None or timestamps is None:
            if stream != "rgb":
                continue
            raise ValueError("The run summary does not contain RGB video and timestamp paths")
        if not source.is_file() or not timestamps.is_file():
            errors.append(f"{stream}: source or timestamp file is unavailable on this computer")
            continue
        times, domain = _timestamp_rows(timestamps)
        values = (
            _motion_trace_from_array(source, roi)
            if kind == "array"
            else _motion_trace_from_video(source, roi)
        )
        count = min(len(times), len(values))
        sources[stream] = {"data": str(source), "timestamps": str(timestamps), "kind": kind}
        traces[stream] = (times[:count], values[:count])
        timestamp_domains[stream] = domain
    if "rgb" not in traces:
        raise RuntimeError("RGB timing-cue trace could not be loaded")
    estimates = []
    for stream in ("thermal", "realsense"):
        if stream not in traces:
            continue
        estimates.append(
            estimate_trace_offset(
                traces["rgb"][0],
                traces["rgb"][1],
                traces[stream][0],
                traces[stream][1],
                sensor_stream=stream,
                max_lag_seconds=max_lag_seconds,
            )
        )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "summary_path": str(summary_file),
        "method": "motion_energy_cross_correlation_with_transition_residuals",
        "roi_normalized": list(roi) if roi is not None else None,
        "sources": sources,
        "timestamp_domains": timestamp_domains,
        "estimates": [asdict(item) for item in estimates],
        "errors": errors,
        "success": bool(estimates),
    }
    destination = (
        Path(output_path).expanduser()
        if output_path is not None
        else summary_file.parent / "sync_validation_report.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    report["output_path"] = str(destination.resolve())
    return report


def format_sync_report(report: dict[str, Any]) -> str:
    lines = [
        "Multimodal Timing-Cue Analysis",
        "------------------------------",
        f"Summary: {report['summary_path']}",
        f"Method: {report['method']}",
    ]
    for estimate in report.get("estimates", []):
        jitter = estimate.get("residual_jitter_seconds")
        jitter_text = f"{float(jitter):.6f}s" if jitter is not None else "n/a"
        lines.append(
            f"- {estimate['sensor_stream']} -> {estimate['reference_stream']}: "
            f"offset={float(estimate['offset_seconds']):+.6f}s, "
            f"correlation={float(estimate['peak_correlation']):.3f}, jitter={jitter_text}"
        )
    lines.extend(f"- warning: {error}" for error in report.get("errors", []))
    lines.extend([f"Report: {report['output_path']}", f"Success: {report['success']}"])
    return "\n".join(lines)
