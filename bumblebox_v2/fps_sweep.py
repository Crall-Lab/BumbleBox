from __future__ import annotations

import gc
import json
import os
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .run_engine import (
    capture_probe,
    mjpeg_record_probe,
    open_capture_probe_session,
)
from .status_history import list_recent_run_records, load_run_summary

CAMERA_SWEEP_COOLDOWN_SECONDS = 0.75


SAFE_MEMORY_RATIO = 0.45
WARN_MEMORY_RATIO = 0.60
HIGH_RISK_MEMORY_RATIO = 0.80
SAFE_DISK_RATIO = 0.70
WARN_DISK_RATIO = 0.90
HIGH_RISK_DISK_RATIO = 0.97


@dataclass
class TrackingBenchmark:
    tracking_fps: float
    summary_path: str
    mode: str
    confidence: str
    source: str
    note: Optional[str]
    started_at: str


@dataclass
class FpsSweepPoint:
    target_fps: float
    actual_fps: float
    frames_captured: int
    probe_seconds: float
    capture_elapsed_seconds: float
    capacity_estimation_mode: str
    probe_output_bytes: Optional[int]
    probe_output_mib_per_second: Optional[float]
    probe_memory_mib_per_second: Optional[float]
    probe_ram_budget_bytes: Optional[int]
    max_recording_seconds_safe: Optional[float]
    max_recording_seconds_warn: Optional[float]
    max_recording_seconds_high_risk: Optional[float]
    estimated_tracking_seconds_safe: Optional[float]
    estimated_tracking_seconds_warn: Optional[float]
    estimated_tracking_seconds_high_risk: Optional[float]
    estimated_tracking_seconds_for_configured_recording: Optional[float]
    status: str
    error: Optional[str]


@dataclass
class FpsSweepReport:
    created_at: str
    recording_codec: str
    tracking_source: str
    capacity_mode: str
    capacity_note: str
    camera_width: int
    camera_height: int
    frame_bytes: int
    probe_seconds: float
    configured_recording_seconds: float
    overhead_factor: float
    capacity_total_bytes: Optional[int]
    capacity_source: str
    total_ram_bytes: Optional[int]
    ram_source: str
    fps_values: List[float]
    tracking_benchmark: Optional[TrackingBenchmark]
    points: List[FpsSweepPoint]

    @property
    def has_errors(self) -> bool:
        return any(point.status == "FAIL" for point in self.points)

    @property
    def success_count(self) -> int:
        return sum(1 for point in self.points if point.status == "PASS")


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_iso(value: str) -> Optional[datetime]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _elapsed_seconds(started_at: str, finished_at: str) -> Optional[float]:
    start = _parse_iso(started_at)
    finish = _parse_iso(finished_at)
    if not start or not finish:
        return None
    elapsed = (finish - start).total_seconds()
    if elapsed <= 0:
        return None
    return elapsed


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


def _resolve_storage_free_bytes(data_root: str | Path) -> tuple[Optional[int], str]:
    path = Path(data_root).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    target = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(target)
        return int(usage.free), str(target)
    except Exception:
        return None, str(target)


def _read_mem_available_bytes() -> Optional[int]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_process_rss_bytes() -> Optional[int]:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _capture_ram_snapshot() -> dict[str, Optional[int]]:
    return {
        "mem_available_bytes": _read_mem_available_bytes(),
        "process_rss_bytes": _read_process_rss_bytes(),
    }


def _observed_ram_probe_metrics(
    before: dict[str, Optional[int]],
    after: dict[str, Optional[int]],
    *,
    capture_elapsed: float,
    fallback_budget_bytes: Optional[int],
) -> tuple[Optional[float], Optional[float], Optional[int]]:
    mem_available_before = before.get("mem_available_bytes")
    mem_available_after = after.get("mem_available_bytes")
    process_rss_before = before.get("process_rss_bytes")
    process_rss_after = after.get("process_rss_bytes")

    deltas: list[int] = []
    if (
        mem_available_before is not None
        and mem_available_after is not None
        and mem_available_before >= mem_available_after
    ):
        deltas.append(int(mem_available_before - mem_available_after))
    if (
        process_rss_before is not None
        and process_rss_after is not None
        and process_rss_after >= process_rss_before
    ):
        deltas.append(int(process_rss_after - process_rss_before))

    observed_growth_bytes = max(deltas) if deltas else None
    if observed_growth_bytes is None or observed_growth_bytes <= 0 or capture_elapsed <= 0:
        return None, None, (
            int(mem_available_before)
            if mem_available_before is not None and mem_available_before > 0
            else fallback_budget_bytes
        )

    bytes_per_second = float(observed_growth_bytes) / float(capture_elapsed)
    budget_bytes = (
        int(mem_available_before)
        if mem_available_before is not None and mem_available_before > 0
        else fallback_budget_bytes
    )
    return bytes_per_second, round(bytes_per_second / (1024**2), 4), budget_bytes


def _capacity_mode_for_config(config: Dict[str, Any]) -> tuple[str, str]:
    codec = str(config.get("camera", {}).get("codec", "mp4")).strip().lower()
    tracking_source = str(config.get("pipeline", {}).get("tracking_source", "ram")).strip().lower()
    mode = str(config.get("pipeline", {}).get("mode", "record_and_track")).strip().lower()

    if mode == "record_and_track" and tracking_source == "ram":
        return (
            "ram",
            "RAM-backed capacity model: tracking_source=ram keeps frames in memory during recording, so RAM is the limiting factor even if codec is MJPEG.",
        )
    if codec == "mjpeg":
        return (
            "disk",
            "Disk-backed capacity model: MJPEG sweep writes probe video during capture and estimates recording duration from measured file growth plus free space under system.data_root.",
        )
    return (
        "ram",
        "RAM-backed capacity model: MP4 recording captures frames to RAM first and encodes afterward.",
    )


def parse_fps_values(csv_text: str) -> List[float]:
    values: List[float] = []
    for token in str(csv_text).split(","):
        text = token.strip()
        if not text:
            continue
        value = float(text)
        if value <= 0:
            raise ValueError("FPS values must be > 0")
        values.append(value)

    if not values:
        raise ValueError("No FPS values were parsed from --fps-values")
    return values


def fps_range(start: float, stop: float, step: float) -> List[float]:
    if start <= 0 or stop <= 0 or step <= 0:
        raise ValueError("fps-start, fps-stop, and fps-step must all be > 0")
    if stop < start:
        raise ValueError("fps-stop must be >= fps-start")

    values: List[float] = []
    current = float(start)
    guard = 0
    while current <= (stop + (step * 0.1)):
        values.append(round(current, 6))
        current += step
        guard += 1
        if guard > 500:
            raise ValueError("FPS range produced too many points; reduce range size.")
    return values


def _overhead_factor(config: Dict[str, Any]) -> float:
    factor = 1.55
    if bool(config.get("pipeline", {}).get("parallel_tracking", False)):
        factor += 0.25
    return factor


def _read_recording_seconds_from_snapshot(summary_payload: Dict[str, Any]) -> Optional[float]:
    path_text = str(summary_payload.get("config_snapshot_path", "")).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.exists() or not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    try:
        value = float(payload.get("capture", {}).get("recording_seconds"))
    except Exception:
        return None
    return value if value > 0 else None


def _tracking_benchmark_from_summary(
    summary_payload: Dict[str, Any],
    summary_path: Path,
) -> Optional[TrackingBenchmark]:
    mode = str(summary_payload.get("mode", "")).strip().lower()
    if mode not in {"track_only", "record_and_track"}:
        return None
    if not bool(summary_payload.get("success", False)):
        return None

    tracking_seconds = float(summary_payload.get("tracking_elapsed_seconds") or 0.0)
    tracking_frames = int(summary_payload.get("tracking_frames_processed") or 0)
    started_at = str(summary_payload.get("started_at", ""))

    if tracking_seconds > 0 and tracking_frames > 0:
        return TrackingBenchmark(
            tracking_fps=tracking_frames / tracking_seconds,
            summary_path=str(summary_path),
            mode=mode,
            confidence="high",
            source="tracking_elapsed_seconds",
            note=None,
            started_at=started_at,
        )

    wall_seconds = _elapsed_seconds(
        str(summary_payload.get("started_at", "")),
        str(summary_payload.get("finished_at", "")),
    )
    recording_seconds = _read_recording_seconds_from_snapshot(summary_payload)
    frames_captured = int(summary_payload.get("frames_captured") or 0)
    if wall_seconds and recording_seconds and frames_captured > 0 and wall_seconds > (recording_seconds + 0.5):
        estimated_tracking_seconds = wall_seconds - recording_seconds
        if estimated_tracking_seconds > 0:
            return TrackingBenchmark(
                tracking_fps=frames_captured / estimated_tracking_seconds,
                summary_path=str(summary_path),
                mode=mode,
                confidence="low",
                source="wall_time_minus_recording",
                note=(
                    "Estimated from run wall-time minus recording duration; includes some non-tracking overhead."
                ),
                started_at=started_at,
            )
    return None


def find_recent_tracking_benchmark(
    data_root: str | Path,
    session_start_iso: Optional[str] = None,
    limit: int = 120,
) -> Optional[TrackingBenchmark]:
    session_start = _parse_iso(session_start_iso or "") if session_start_iso else None
    records = list_recent_run_records(data_root, limit=limit)
    for record in records:
        if session_start:
            started = _parse_iso(record.started_at)
            if not started or started < session_start:
                continue

        try:
            payload = load_run_summary(record.summary_path)
        except Exception:
            continue

        benchmark = _tracking_benchmark_from_summary(payload, record.summary_path)
        if benchmark and benchmark.tracking_fps > 0:
            return benchmark
    return None


def _duration_budget_seconds(
    *,
    ram_bytes: Optional[int],
    memory_ratio: float,
    frame_bytes: int,
    effective_fps: float,
    overhead_factor: float,
) -> Optional[float]:
    if not ram_bytes or ram_bytes <= 0:
        return None
    if frame_bytes <= 0 or effective_fps <= 0 or overhead_factor <= 0:
        return None
    bytes_per_second = frame_bytes * effective_fps * overhead_factor
    if bytes_per_second <= 0:
        return None
    return (ram_bytes * memory_ratio) / bytes_per_second


def _duration_budget_from_bytes_per_second(
    *,
    total_bytes: Optional[int],
    usage_ratio: float,
    bytes_per_second: Optional[float],
) -> Optional[float]:
    if not total_bytes or total_bytes <= 0:
        return None
    if bytes_per_second is None or bytes_per_second <= 0:
        return None
    if usage_ratio <= 0:
        return None
    return (float(total_bytes) * float(usage_ratio)) / float(bytes_per_second)


def _estimate_tracking_seconds(
    *,
    recording_seconds: Optional[float],
    recording_fps: float,
    tracking_fps: Optional[float],
) -> Optional[float]:
    if recording_seconds is None or recording_seconds <= 0:
        return None
    if recording_fps <= 0:
        return None
    if tracking_fps is None or tracking_fps <= 0:
        return None
    estimated_frames = recording_fps * recording_seconds
    return estimated_frames / tracking_fps


def run_fps_sweep(
    config: Dict[str, Any],
    fps_values: Sequence[float],
    probe_seconds: float = 20.0,
    assume_ram_gb: Optional[float] = None,
    use_mock_camera: Optional[bool] = None,
    session_start_iso: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> FpsSweepReport:
    if probe_seconds <= 0:
        raise ValueError("probe_seconds must be > 0")
    if not fps_values:
        raise ValueError("fps_values cannot be empty")

    prepared_fps: List[float] = []
    for value in fps_values:
        parsed = float(value)
        if parsed <= 0:
            raise ValueError("All FPS targets must be > 0")
        prepared_fps.append(parsed)

    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    frame_bytes = int(width * height * 1.5)
    recording_codec = str(config.get("camera", {}).get("codec", "mp4")).strip().lower()
    tracking_source = str(config.get("pipeline", {}).get("tracking_source", "ram")).strip().lower()
    capacity_mode, capacity_note = _capacity_mode_for_config(config)
    configured_recording_seconds = float(config.get("capture", {}).get("recording_seconds", probe_seconds))
    overhead_factor = _overhead_factor(config)
    total_ram: Optional[int] = None
    ram_source = "not used"
    capacity_total_bytes: Optional[int] = None
    capacity_source = "unavailable"
    use_mock_capture = bool(use_mock_camera if use_mock_camera is not None else config.get("runtime", {}).get("use_mock_camera", False))
    if capacity_mode == "ram":
        total_ram, ram_source = _resolve_total_ram_bytes(config, assume_ram_gb=assume_ram_gb)
        capacity_total_bytes = total_ram
        capacity_source = ram_source
    else:
        storage_free_bytes, storage_source = _resolve_storage_free_bytes(config.get("system", {}).get("data_root", ""))
        capacity_total_bytes = storage_free_bytes
        capacity_source = storage_source
        if assume_ram_gb is not None:
            capacity_note += " Assume RAM GiB is ignored in disk-backed MJPEG mode."
    config_ram_override = config.get("system", {}).get("ram_gb_override")
    ram_empirical_enabled = (
        capacity_mode == "ram"
        and not use_mock_capture
        and assume_ram_gb is None
        and config_ram_override in (None, "", 0)
    )
    if capacity_mode == "ram" and ram_empirical_enabled:
        capacity_note = (
            "Empirical RAM-backed capacity model: each probe measures observed memory growth "
            "from Linux MemAvailable and process RSS while frames remain in memory. "
            "Safe/warn/high-risk durations use RAM available at the start of each probe. "
            "If memory samples are unavailable for a point, BumbleBox falls back to the frame-size heuristic."
        )
    elif capacity_mode == "ram":
        capacity_note = (
            "Heuristic RAM-backed capacity model: durations are estimated from frame size, measured FPS, "
            "and a fixed overhead factor. This path is used for simulated/mock runs or when RAM overrides are set."
        )
    benchmark = find_recent_tracking_benchmark(
        data_root=config.get("system", {}).get("data_root", ""),
        session_start_iso=session_start_iso,
    )

    def _build_point(
        *,
        target_fps: float,
        frames_captured: int,
        measured_fps: float,
        capture_elapsed: float,
        probe_output_bytes: Optional[int] = None,
        probe_memory_mib_per_second: Optional[float] = None,
        probe_ram_budget_bytes: Optional[int] = None,
    ) -> FpsSweepPoint:
        effective_fps = measured_fps if measured_fps > 0 else target_fps
        probe_output_mib_per_second: Optional[float] = None
        capacity_estimation_mode = "disk_empirical" if capacity_mode == "disk" else "ram_heuristic"
        if capacity_mode == "ram":
            if (
                probe_memory_mib_per_second is not None
                and probe_ram_budget_bytes is not None
                and capture_elapsed > 0
            ):
                bytes_per_second = float(probe_memory_mib_per_second) * (1024**2)
                safe_seconds = _duration_budget_from_bytes_per_second(
                    total_bytes=probe_ram_budget_bytes,
                    usage_ratio=SAFE_MEMORY_RATIO,
                    bytes_per_second=bytes_per_second,
                )
                warn_seconds = _duration_budget_from_bytes_per_second(
                    total_bytes=probe_ram_budget_bytes,
                    usage_ratio=WARN_MEMORY_RATIO,
                    bytes_per_second=bytes_per_second,
                )
                high_risk_seconds = _duration_budget_from_bytes_per_second(
                    total_bytes=probe_ram_budget_bytes,
                    usage_ratio=HIGH_RISK_MEMORY_RATIO,
                    bytes_per_second=bytes_per_second,
                )
                capacity_estimation_mode = "ram_empirical"
            else:
                safe_seconds = _duration_budget_seconds(
                    ram_bytes=total_ram,
                    memory_ratio=SAFE_MEMORY_RATIO,
                    frame_bytes=frame_bytes,
                    effective_fps=effective_fps,
                    overhead_factor=overhead_factor,
                )
                warn_seconds = _duration_budget_seconds(
                    ram_bytes=total_ram,
                    memory_ratio=WARN_MEMORY_RATIO,
                    frame_bytes=frame_bytes,
                    effective_fps=effective_fps,
                    overhead_factor=overhead_factor,
                )
                high_risk_seconds = _duration_budget_seconds(
                    ram_bytes=total_ram,
                    memory_ratio=HIGH_RISK_MEMORY_RATIO,
                    frame_bytes=frame_bytes,
                    effective_fps=effective_fps,
                    overhead_factor=overhead_factor,
                )
        else:
            bytes_per_second = None
            if probe_output_bytes is not None and capture_elapsed > 0:
                bytes_per_second = float(probe_output_bytes) / float(capture_elapsed)
                probe_output_mib_per_second = round(bytes_per_second / (1024**2), 4)
            safe_seconds = _duration_budget_from_bytes_per_second(
                total_bytes=capacity_total_bytes,
                usage_ratio=SAFE_DISK_RATIO,
                bytes_per_second=bytes_per_second,
            )
            warn_seconds = _duration_budget_from_bytes_per_second(
                total_bytes=capacity_total_bytes,
                usage_ratio=WARN_DISK_RATIO,
                bytes_per_second=bytes_per_second,
            )
            high_risk_seconds = _duration_budget_from_bytes_per_second(
                total_bytes=capacity_total_bytes,
                usage_ratio=HIGH_RISK_DISK_RATIO,
                bytes_per_second=bytes_per_second,
            )
        tracking_fps = benchmark.tracking_fps if benchmark else None
        return FpsSweepPoint(
            target_fps=float(target_fps),
            actual_fps=float(measured_fps),
            frames_captured=int(frames_captured),
            probe_seconds=float(probe_seconds),
            capture_elapsed_seconds=float(capture_elapsed),
            capacity_estimation_mode=capacity_estimation_mode,
            probe_output_bytes=(int(probe_output_bytes) if probe_output_bytes is not None else None),
            probe_output_mib_per_second=probe_output_mib_per_second,
            probe_memory_mib_per_second=probe_memory_mib_per_second,
            probe_ram_budget_bytes=(int(probe_ram_budget_bytes) if probe_ram_budget_bytes is not None else None),
            max_recording_seconds_safe=safe_seconds,
            max_recording_seconds_warn=warn_seconds,
            max_recording_seconds_high_risk=high_risk_seconds,
            estimated_tracking_seconds_safe=_estimate_tracking_seconds(
                recording_seconds=safe_seconds,
                recording_fps=effective_fps,
                tracking_fps=tracking_fps,
            ),
            estimated_tracking_seconds_warn=_estimate_tracking_seconds(
                recording_seconds=warn_seconds,
                recording_fps=effective_fps,
                tracking_fps=tracking_fps,
            ),
            estimated_tracking_seconds_high_risk=_estimate_tracking_seconds(
                recording_seconds=high_risk_seconds,
                recording_fps=effective_fps,
                tracking_fps=tracking_fps,
            ),
            estimated_tracking_seconds_for_configured_recording=_estimate_tracking_seconds(
                recording_seconds=configured_recording_seconds,
                recording_fps=effective_fps,
                tracking_fps=tracking_fps,
            ),
            status="PASS",
            error=None,
        )

    points: List[FpsSweepPoint] = []
    total = len(prepared_fps)
    use_persistent_session = not use_mock_capture
    session = open_capture_probe_session(config, use_mock_camera=use_mock_camera) if use_persistent_session else None
    if session is not None:
        session.start()
    probe_base = Path(capacity_source).expanduser() if capacity_mode == "disk" else Path(
        config.get("system", {}).get("data_root", "")
    ).expanduser()
    probe_root = probe_base / ".bbx_fps_sweep_probe"
    try:
        for index, target_fps in enumerate(prepared_fps, start=1):
            if progress_callback:
                progress_callback(index, total, target_fps)

            probe_cfg = deepcopy(config)
            probe_cfg.setdefault("camera", {})
            probe_cfg.setdefault("capture", {})
            probe_cfg.setdefault("runtime", {})
            probe_cfg["camera"]["fps_target"] = float(target_fps)
            probe_cfg["capture"]["recording_seconds"] = float(probe_seconds)
            if use_mock_camera is not None:
                probe_cfg["runtime"]["use_mock_camera"] = bool(use_mock_camera)

            try:
                if capacity_mode == "disk":
                    probe_root.mkdir(parents=True, exist_ok=True)
                    probe_path = probe_root / f"probe_{index:03d}_{target_fps:.3f}.avi"
                    try:
                        frames_captured, measured_fps, capture_elapsed, probe_output_bytes = mjpeg_record_probe(
                            probe_cfg,
                            output_path=probe_path,
                            fps_target=float(target_fps),
                            recording_seconds=float(probe_seconds),
                            use_mock_camera=use_mock_camera,
                            session=session,
                        )
                    finally:
                        try:
                            probe_path.unlink()
                        except Exception:
                            pass
                else:
                    probe_output_bytes = None
                    probe_memory_mib_per_second = None
                    probe_ram_budget_bytes = None
                    if session is not None:
                        gc.collect()
                        ram_snapshot_before = _capture_ram_snapshot() if ram_empirical_enabled else {}
                        start_time = time.perf_counter()
                        frames, _timestamps, measured_fps = session.capture_for(
                            fps=float(target_fps),
                            duration=float(probe_seconds),
                        )
                        capture_elapsed = time.perf_counter() - start_time
                        frames_captured = len(frames)
                        ram_snapshot_after = _capture_ram_snapshot() if ram_empirical_enabled else {}
                        if ram_empirical_enabled:
                            (
                                _probe_memory_bytes_per_second,
                                probe_memory_mib_per_second,
                                probe_ram_budget_bytes,
                            ) = _observed_ram_probe_metrics(
                                ram_snapshot_before,
                                ram_snapshot_after,
                                capture_elapsed=float(capture_elapsed),
                                fallback_budget_bytes=total_ram,
                            )
                        del frames
                        gc.collect()
                    else:
                        frames_captured, measured_fps, capture_elapsed = capture_probe(probe_cfg)
                points.append(
                    _build_point(
                        target_fps=float(target_fps),
                        frames_captured=int(frames_captured),
                        measured_fps=float(measured_fps),
                        capture_elapsed=float(capture_elapsed),
                        probe_output_bytes=probe_output_bytes,
                        probe_memory_mib_per_second=probe_memory_mib_per_second,
                        probe_ram_budget_bytes=probe_ram_budget_bytes,
                    )
                )
            except Exception as exc:
                points.append(
                    FpsSweepPoint(
                        target_fps=float(target_fps),
                        actual_fps=0.0,
                        frames_captured=0,
                        probe_seconds=float(probe_seconds),
                        capture_elapsed_seconds=0.0,
                        capacity_estimation_mode=("disk_empirical" if capacity_mode == "disk" else "ram_heuristic"),
                        probe_output_bytes=None,
                        probe_output_mib_per_second=None,
                        probe_memory_mib_per_second=None,
                        probe_ram_budget_bytes=None,
                        max_recording_seconds_safe=None,
                        max_recording_seconds_warn=None,
                        max_recording_seconds_high_risk=None,
                        estimated_tracking_seconds_safe=None,
                        estimated_tracking_seconds_warn=None,
                        estimated_tracking_seconds_high_risk=None,
                        estimated_tracking_seconds_for_configured_recording=None,
                        status="FAIL",
                        error=str(exc),
                    )
                )
            finally:
                if session is None and not bool(probe_cfg.get("runtime", {}).get("use_mock_camera", False)) and index < total:
                    time.sleep(CAMERA_SWEEP_COOLDOWN_SECONDS)
    finally:
        if session is not None:
            session.close()
        try:
            if probe_root.exists():
                probe_root.rmdir()
        except Exception:
            pass

    return FpsSweepReport(
        created_at=_iso_now(),
        recording_codec=recording_codec,
        tracking_source=tracking_source,
        capacity_mode=capacity_mode,
        capacity_note=capacity_note,
        camera_width=width,
        camera_height=height,
        frame_bytes=frame_bytes,
        probe_seconds=float(probe_seconds),
        configured_recording_seconds=float(configured_recording_seconds),
        overhead_factor=float(overhead_factor),
        capacity_total_bytes=int(capacity_total_bytes) if capacity_total_bytes else None,
        capacity_source=capacity_source,
        total_ram_bytes=int(total_ram) if total_ram else None,
        ram_source=ram_source,
        fps_values=[float(v) for v in prepared_fps],
        tracking_benchmark=benchmark,
        points=points,
    )


def _human_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 0:
        return "n/a"
    if seconds < 120:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 120:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def _capacity_mode_label(mode: str) -> str:
    if str(mode).strip().lower() == "disk":
        return "disk-backed MJPEG streaming"
    return "RAM-backed capture"


def format_fps_sweep_report(report: FpsSweepReport) -> str:
    empirical_ram_points = sum(1 for point in report.points if point.capacity_estimation_mode == "ram_empirical")
    mjpeg_ram_override = (
        report.recording_codec == "mjpeg"
        and report.capacity_mode == "ram"
        and report.tracking_source == "ram"
    )
    lines = [
        "FPS Sweep Report",
        "----------------",
        f"Created at: {report.created_at}",
        f"Recording codec: {report.recording_codec}",
        f"Tracking source: {report.tracking_source}",
        f"Capacity model: {_capacity_mode_label(report.capacity_mode)}",
        f"Resolution: {report.camera_width}x{report.camera_height}",
        f"Probe duration per FPS: {_human_seconds(report.probe_seconds)}",
        f"Configured recording duration: {_human_seconds(report.configured_recording_seconds)}",
    ]
    if report.capacity_mode == "disk":
        lines.append(f"Storage path checked: {report.capacity_source}")
        if report.capacity_total_bytes:
            lines.append(f"Free storage at start: {report.capacity_total_bytes / (1024**3):.2f} GiB")
        else:
            lines.append("Free storage at start: unavailable")
    else:
        lines.append(f"RAM source: {report.capacity_source}")
        if report.capacity_total_bytes:
            label = "Installed RAM detected" if empirical_ram_points else "RAM available to model"
            lines.append(f"{label}: {report.capacity_total_bytes / (1024**3):.2f} GiB")
        else:
            lines.append("RAM available to model: unavailable")
        if empirical_ram_points:
            lines.append(
                f"RAM probes using empirical memory growth: {empirical_ram_points}/{len(report.points)}"
            )
    if mjpeg_ram_override:
        lines.append(
            "MJPEG override: MJPEG is selected, but this sweep is RAM-backed because "
            "pipeline.mode=record_and_track and tracking_source=ram keep frames in memory during recording."
        )
    lines.append(f"Capacity note: {report.capacity_note}")

    if report.tracking_benchmark:
        lines.append(
            "Tracking benchmark: "
            f"{report.tracking_benchmark.tracking_fps:.3f} tracking-fps "
            f"({report.tracking_benchmark.confidence} confidence, {report.tracking_benchmark.source})"
        )
        lines.append(f"Tracking benchmark source: {report.tracking_benchmark.summary_path}")
        if report.tracking_benchmark.note:
            lines.append(f"Tracking benchmark note: {report.tracking_benchmark.note}")
    else:
        lines.append("Tracking benchmark: unavailable (run at least one tracking session first).")

    lines.append("")
    if report.capacity_mode == "disk":
        lines.append(
            "Columns: target_fps -> actual_fps | max recording @ safe/warn/high-risk free-space budgets | "
            "measured MJPEG write rate | est. tracking for configured recording"
        )
    else:
        lines.append(
            "Columns: target_fps -> actual_fps | max recording @ safe/warn/high-risk RAM budgets | "
            "measured RAM growth or heuristic fallback | "
            "est. tracking for configured recording"
        )
    for point in report.points:
        if point.status == "FAIL":
            lines.append(f"- {point.target_fps:.2f} -> FAIL ({point.error})")
            continue
        if report.capacity_mode == "disk":
            write_rate = (
                f"{point.probe_output_mib_per_second:.3f} MiB/s"
                if point.probe_output_mib_per_second is not None
                else "n/a MiB/s"
            )
            lines.append(
                "- "
                f"{point.target_fps:.2f} -> {point.actual_fps:.3f} | "
                f"{_human_seconds(point.max_recording_seconds_safe)} / "
                f"{_human_seconds(point.max_recording_seconds_warn)} / "
                f"{_human_seconds(point.max_recording_seconds_high_risk)} | "
                f"{write_rate} | "
                f"{_human_seconds(point.estimated_tracking_seconds_for_configured_recording)}"
            )
        else:
            ram_growth = (
                f"{point.probe_memory_mib_per_second:.3f} MiB/s RAM"
                if point.capacity_estimation_mode == "ram_empirical" and point.probe_memory_mib_per_second is not None
                else "heuristic RAM model"
            )
            lines.append(
                "- "
                f"{point.target_fps:.2f} -> {point.actual_fps:.3f} | "
                f"{_human_seconds(point.max_recording_seconds_safe)} / "
                f"{_human_seconds(point.max_recording_seconds_warn)} / "
                f"{_human_seconds(point.max_recording_seconds_high_risk)} | "
                f"{ram_growth} | "
                f"{_human_seconds(point.estimated_tracking_seconds_for_configured_recording)}"
            )
            if (
                point.capacity_estimation_mode == "ram_empirical"
                and point.probe_ram_budget_bytes is not None
            ):
                lines.append(
                    "  RAM available at probe start: "
                    f"{point.probe_ram_budget_bytes / (1024**3):.2f} GiB"
                )
        if point.estimated_tracking_seconds_safe is not None:
            lines.append(
                "  tracking for safe/warn/high-risk durations: "
                f"{_human_seconds(point.estimated_tracking_seconds_safe)} / "
                f"{_human_seconds(point.estimated_tracking_seconds_warn)} / "
                f"{_human_seconds(point.estimated_tracking_seconds_high_risk)}"
            )

    lines.append("")
    lines.append(f"Successful probes: {report.success_count}/{len(report.points)}")
    return "\n".join(lines)


def write_fps_sweep_json(report: FpsSweepReport, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    output.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return output
