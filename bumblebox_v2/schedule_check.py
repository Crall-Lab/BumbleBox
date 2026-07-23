from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .camera_profiles import get_camera_model_info

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


@dataclass
class ScheduleCheckItem:
    name: str
    status: str
    message: str


@dataclass
class ScheduleCheckReport:
    items: List[ScheduleCheckItem]
    suggestions: List[str]
    metrics: Dict[str, float | int | str | bool]
    benchmark_used: bool

    @property
    def has_failures(self) -> bool:
        return any(item.status == "FAIL" for item in self.items)


def _total_ram_bytes() -> Optional[int]:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except Exception:
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(page_count)
    except Exception:
        return None


def _resolve_total_ram_bytes(
    config: Dict[str, Any],
    assume_ram_gb: Optional[float] = None,
) -> tuple[Optional[int], str]:
    if assume_ram_gb is not None:
        if assume_ram_gb <= 0:
            raise ValueError("assume_ram_gb must be > 0 when provided")
        return int(float(assume_ram_gb) * (1024**3)), "assumed_cli"

    override = config.get("system", {}).get("ram_gb_override")
    if override not in (None, "", 0):
        try:
            override_gb = float(override)
        except (TypeError, ValueError) as exc:
            raise ValueError("system.ram_gb_override must be a number when set") from exc
        if override_gb <= 0:
            raise ValueError("system.ram_gb_override must be > 0 when set")
        return int(override_gb * (1024**3)), "assumed_config"

    return _total_ram_bytes(), "detected_host"


def _pi_model_key(config: Dict[str, Any]) -> str:
    value = str(config.get("system", {}).get("pi_model", "auto")).strip().lower()
    if value in {"pi4", "pi5"}:
        return value
    return "auto"


def _estimate_tracking_fps_heuristic(config: Dict[str, Any]) -> float:
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    pixels = max(1, width * height)
    reference_pixels = 4056 * 3040

    base_by_pi = {
        "pi4": 0.95,
        "pi5": 1.75,
        "auto": 1.15,
    }
    base = base_by_pi[_pi_model_key(config)]
    scaled = base * (reference_pixels / pixels)

    if bool(config.get("pipeline", {}).get("parallel_tracking", False)):
        scaled *= 1.25
    if str(config.get("pipeline", {}).get("tracking_source", "ram")).strip().lower() == "video":
        scaled *= 0.85

    return max(0.20, scaled)


def benchmark_tracking_fps(
    input_path: str | Path,
    dictionary_name: str,
    aruco_params: Optional[Dict[str, Any]],
    sample_frames: int = 80,
) -> float:
    from .tracking_optimizer import load_sample_frames, normalize_dictionary_name

    if cv2 is None:
        raise RuntimeError("OpenCV is required for schedule benchmark input. Install opencv-contrib-python.")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is required for schedule benchmark input.")

    frames, _input_type, _total_count = load_sample_frames(input_path, sample_frames)
    if not frames:
        raise RuntimeError("No frames available for tracking benchmark.")

    detector_params = cv2.aruco.DetectorParameters()
    for key, value in (aruco_params or {}).items():
        if hasattr(detector_params, key):
            setattr(detector_params, key, value)

    normalized_dictionary = normalize_dictionary_name(dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, normalized_dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    start = time.perf_counter()
    for frame in frames:
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector.detectMarkers(gray)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return len(frames) / elapsed


def run_schedule_check(
    config: Dict[str, Any],
    benchmark_input: Optional[str | Path] = None,
    benchmark_frames: int = 80,
    assume_ram_gb: Optional[float] = None,
) -> ScheduleCheckReport:
    items: List[ScheduleCheckItem] = []
    suggestions: List[str] = []

    camera = config["camera"]
    capture = config["capture"]
    pipeline = config["pipeline"]
    tracking = config["tracking"]

    width = int(camera["width"])
    height = int(camera["height"])
    fps_target = float(camera["fps_target"])
    recording_seconds = float(capture["recording_seconds"])
    frame_count = max(1.0, fps_target * recording_seconds)

    frame_bytes = width * height * 1.5  # YUV420
    overhead_factor = 1.55
    if bool(pipeline.get("parallel_tracking", False)):
        overhead_factor += 0.25
    estimated_capture_ram = frame_count * frame_bytes * overhead_factor
    total_ram, ram_source = _resolve_total_ram_bytes(config, assume_ram_gb=assume_ram_gb)
    memory_ratio = (estimated_capture_ram / total_ram) if total_ram and total_ram > 0 else None

    if total_ram:
        source_message = {
            "assumed_cli": f"Using assumed RAM from CLI: {total_ram / (1024**3):.2f} GiB",
            "assumed_config": f"Using assumed RAM from config system.ram_gb_override: {total_ram / (1024**3):.2f} GiB",
            "detected_host": f"Using detected host RAM: {total_ram / (1024**3):.2f} GiB",
        }.get(ram_source, f"Using RAM estimate: {total_ram / (1024**3):.2f} GiB")
        items.append(ScheduleCheckItem("RAM source", "PASS", source_message))
    else:
        items.append(ScheduleCheckItem("RAM source", "WARN", "Could not detect RAM. Use --assume-ram-gb for a stable estimate."))

    if memory_ratio is None:
        items.append(
            ScheduleCheckItem(
                "Memory budget",
                "WARN",
                "Could not detect total RAM; memory risk estimate is unavailable.",
            )
        )
    elif memory_ratio >= 0.80:
        items.append(
            ScheduleCheckItem(
                "Memory budget",
                "FAIL",
                (
                    f"Estimated per-recording frame buffer is {estimated_capture_ram / (1024**3):.2f} GiB "
                    f"({memory_ratio * 100:.1f}% of total RAM). High risk of OOM or swap thrash."
                ),
            )
        )
    elif memory_ratio >= 0.60:
        items.append(
            ScheduleCheckItem(
                "Memory budget",
                "WARN",
                (
                    f"Estimated per-recording frame buffer is {estimated_capture_ram / (1024**3):.2f} GiB "
                    f"({memory_ratio * 100:.1f}% of total RAM). Tracking and other processes may cause instability."
                ),
            )
        )
    else:
        items.append(
            ScheduleCheckItem(
                "Memory budget",
                "PASS",
                (
                    f"Estimated frame buffer is {estimated_capture_ram / (1024**3):.2f} GiB "
                    f"({(memory_ratio or 0) * 100:.1f}% of total RAM)."
                ),
            )
        )

    camera_model = str(camera.get("model", "auto")).strip().lower()
    camera_model_info = get_camera_model_info(camera_model)
    expected_res = camera_model_info.max_resolution if camera_model_info else None
    if expected_res:
        if (width, height) == expected_res:
            items.append(
                ScheduleCheckItem(
                    "Camera resolution",
                    "PASS",
                    f"{camera_model} configured at expected default {expected_res[0]}x{expected_res[1]}",
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Camera resolution",
                    "WARN",
                    (
                        f"{camera_model} default is {expected_res[0]}x{expected_res[1]}, "
                        f"but config uses {width}x{height}."
                    ),
                )
            )

    benchmark_used = False
    benchmark_fps = None
    if benchmark_input:
        try:
            benchmark_fps = benchmark_tracking_fps(
                input_path=benchmark_input,
                dictionary_name=str(tracking.get("tag_dictionary", "4X4_50")),
                aruco_params=tracking.get("aruco_params"),
                sample_frames=max(10, int(benchmark_frames)),
            )
            benchmark_used = True
            items.append(
                ScheduleCheckItem(
                    "Tracking benchmark",
                    "PASS",
                    f"Measured detect throughput: {benchmark_fps:.3f} frames/sec",
                )
            )
        except Exception as exc:
            items.append(
                ScheduleCheckItem(
                    "Tracking benchmark",
                    "WARN",
                    f"Benchmark failed: {exc}. Falling back to heuristic estimate.",
                )
            )

    tracking_fps_est = benchmark_fps if benchmark_fps and benchmark_fps > 0 else _estimate_tracking_fps_heuristic(config)
    tracking_seconds_per_recording = frame_count / max(0.20, tracking_fps_est)
    mode = str(pipeline["mode"]).strip().lower()
    defer_tracking = bool(pipeline.get("defer_tracking_until_after_recording", False))
    record_interval_seconds = float(capture.get("record_interval_minutes", 1)) * 60.0
    track_interval_seconds = float(capture.get("track_interval_minutes", 1)) * 60.0

    if mode == "record_only":
        cycle_seconds = recording_seconds
        if cycle_seconds > record_interval_seconds:
            items.append(
                ScheduleCheckItem(
                    "Record schedule fit",
                    "FAIL",
                    (
                        f"Recording duration {cycle_seconds:.1f}s is longer than record interval "
                        f"{record_interval_seconds:.1f}s."
                    ),
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Record schedule fit",
                    "PASS",
                    f"Recording cycle ({cycle_seconds:.1f}s) fits in interval ({record_interval_seconds:.1f}s).",
                )
            )

    elif mode == "track_only":
        if tracking_seconds_per_recording > track_interval_seconds:
            items.append(
                ScheduleCheckItem(
                    "Track schedule fit",
                    "FAIL",
                    (
                        f"Estimated tracking time {tracking_seconds_per_recording:.1f}s exceeds track interval "
                        f"{track_interval_seconds:.1f}s."
                    ),
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Track schedule fit",
                    "PASS",
                    (
                        f"Estimated tracking time {tracking_seconds_per_recording:.1f}s fits in track interval "
                        f"{track_interval_seconds:.1f}s."
                    ),
                )
            )

    elif mode == "record_and_track":
        if defer_tracking:
            cycle_seconds = recording_seconds + tracking_seconds_per_recording
            label = "record + deferred tracking"
        else:
            cycle_seconds = max(recording_seconds, tracking_seconds_per_recording * 0.85)
            label = "record + overlapping tracking"

        if cycle_seconds > record_interval_seconds:
            items.append(
                ScheduleCheckItem(
                    "Record schedule fit",
                    "FAIL",
                    (
                        f"Estimated {label} cycle {cycle_seconds:.1f}s exceeds record interval "
                        f"{record_interval_seconds:.1f}s."
                    ),
                )
            )
        elif cycle_seconds > (0.80 * record_interval_seconds):
            items.append(
                ScheduleCheckItem(
                    "Record schedule fit",
                    "WARN",
                    (
                        f"Estimated {label} cycle {cycle_seconds:.1f}s uses most of interval "
                        f"{record_interval_seconds:.1f}s."
                    ),
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Record schedule fit",
                    "PASS",
                    (
                        f"Estimated {label} cycle {cycle_seconds:.1f}s fits in interval "
                        f"{record_interval_seconds:.1f}s."
                    ),
                )
            )

    elif mode == "mixed_schedule":
        record_cycle = recording_seconds
        track_cycle = tracking_seconds_per_recording
        utilization = (record_cycle / max(1.0, record_interval_seconds)) + (
            track_cycle / max(1.0, track_interval_seconds)
        )

        if record_cycle > record_interval_seconds:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule: record lane",
                    "FAIL",
                    f"Record cycle {record_cycle:.1f}s exceeds record interval {record_interval_seconds:.1f}s.",
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule: record lane",
                    "PASS",
                    f"Record cycle {record_cycle:.1f}s fits interval {record_interval_seconds:.1f}s.",
                )
            )

        if track_cycle > track_interval_seconds:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule: track lane",
                    "FAIL",
                    f"Track cycle {track_cycle:.1f}s exceeds track interval {track_interval_seconds:.1f}s.",
                )
            )
        elif track_cycle > (0.80 * track_interval_seconds):
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule: track lane",
                    "WARN",
                    f"Track cycle {track_cycle:.1f}s uses most of track interval {track_interval_seconds:.1f}s.",
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule: track lane",
                    "PASS",
                    f"Track cycle {track_cycle:.1f}s fits track interval {track_interval_seconds:.1f}s.",
                )
            )

        if utilization > 1.0:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule utilization",
                    "FAIL",
                    f"Estimated combined utilization is {utilization:.2f} (>1.0). Jobs will accumulate backlog.",
                )
            )
        elif utilization > 0.85:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule utilization",
                    "WARN",
                    f"Estimated combined utilization is {utilization:.2f}. Very little timing margin.",
                )
            )
        else:
            items.append(
                ScheduleCheckItem(
                    "Mixed schedule utilization",
                    "PASS",
                    f"Estimated combined utilization is {utilization:.2f}.",
                )
            )

    if memory_ratio is not None and memory_ratio >= 0.60:
        target_ratio = 0.45
        max_frames = (target_ratio * total_ram) / max(1.0, frame_bytes * overhead_factor) if total_ram else None
        if max_frames:
            recommended_fps = max(0.2, max_frames / max(1.0, recording_seconds))
            suggestions.append(
                f"Reduce camera.fps_target to around {recommended_fps:.2f} or lower to reduce RAM pressure."
            )
        suggestions.append("Consider reducing resolution or recording_seconds for RAM-limited Pi models.")
        suggestions.append("Consider tracking_source=video to lower RAM pressure during recording.")

    if mode in {"record_and_track", "mixed_schedule", "track_only"}:
        min_track_interval_sec = tracking_seconds_per_recording * 1.15
        suggestions.append(
            f"Use track interval >= {math.ceil(min_track_interval_sec / 60.0)} minute(s) for current settings."
        )

    if mode in {"record_only", "record_and_track", "mixed_schedule"}:
        min_record_interval_sec = recording_seconds
        if mode == "record_and_track" and defer_tracking:
            min_record_interval_sec += tracking_seconds_per_recording
        suggestions.append(
            f"Use record interval >= {math.ceil(min_record_interval_sec / 60.0)} minute(s) for current settings."
        )

    if not benchmark_used:
        suggestions.append(
            "For higher confidence, rerun schedule-check with --benchmark-input on representative footage."
        )

    metrics: Dict[str, float | int | str | bool] = {
        "camera_width": width,
        "camera_height": height,
        "fps_target": fps_target,
        "recording_seconds": recording_seconds,
        "estimated_frame_count": int(round(frame_count)),
        "estimated_frame_buffer_bytes": int(estimated_capture_ram),
        "total_ram_bytes": int(total_ram) if total_ram else 0,
        "ram_source": ram_source,
        "memory_ratio": float(memory_ratio) if memory_ratio is not None else -1.0,
        "tracking_fps_estimate": float(tracking_fps_est),
        "tracking_seconds_per_recording": float(tracking_seconds_per_recording),
        "benchmark_used": benchmark_used,
        "mode": mode,
        "execution_hint": "Use optimize-tracking with execution-target pi_safe on Pi hardware.",
    }

    return ScheduleCheckReport(
        items=items,
        suggestions=suggestions,
        metrics=metrics,
        benchmark_used=benchmark_used,
    )


def format_schedule_check_report(report: ScheduleCheckReport) -> str:
    lines = [
        "Schedule Check Report",
        "---------------------",
    ]
    for item in report.items:
        lines.append(f"[{item.status}] {item.name}: {item.message}")

    lines.append("")
    lines.append("Key metrics:")
    for key, value in report.metrics.items():
        lines.append(f"- {key}: {value}")

    if report.suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for idx, suggestion in enumerate(report.suggestions, start=1):
            lines.append(f"{idx}. {suggestion}")

    fail_count = sum(1 for item in report.items if item.status == "FAIL")
    warn_count = sum(1 for item in report.items if item.status == "WARN")
    lines.append("")
    lines.append(
        f"Summary: {fail_count} fail, {warn_count} warn, {len(report.items) - fail_count - warn_count} pass"
    )
    return "\n".join(lines)
