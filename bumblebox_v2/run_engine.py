from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import gc
import json
import math
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .camera_controls import (
    apply_autofocus_before_start,
    apply_preflight_lens_lock,
    autofocus_preflight_settings,
    autofocus_settings,
    lock_autofocus_after_warmup,
    run_autofocus_preflight,
    start_autofocus_after_camera_start,
)
from .camera_profiles import apply_camera_profile, configured_profile_name, validate_camera_ir_compatibility
from .realsense_camera import (
    RealSenseRecordingResult,
    RealSenseRecordingSession,
    capture_simulated_realsense_recording,
)
from . import heartbeat
from .time_units import describe_seconds
from .simulated_capture import (
    simulated_frame_times,
    simulated_rgb_yuv420_frame,
    simulated_thermal_frame,
)
from .camera_select import (
    capture_lock_suffix,
    configured_camera_id,
    configured_camera_num,
    picamera2_camera_kwargs,
    session_tag,
)
from .capture_lock import DEFAULT_LOCK_WAIT_SECONDS, capture_lock, default_lock_path
from .storage_guard import ensure_storage_ready
from .stream_writer import (
    stream_chunk_frames,
    FRAME_BUFFER_STREAM,
    COLOUR_INPUT_FORMAT,
    GRAY_INPUT_FORMAT,
    full_range_output_format,
    StreamedVideo,
    StreamingVideoWriter,
    find_ffmpeg_binary,
    frame_buffer_mode,
    normalize_mp4_codec,
    resolve_mp4_preset,
    stream_queue_bytes,
)
from .image_controls import apply_image_controls
from .thermal_camera import resolve_thermal_device_path, _set_v4l2_y16_format
from .tracking_index import sync_run_summary_file
from .tuning import resolve_camera_tuning_file

CAMERA_REOPEN_RETRY_ATTEMPTS = 3
CAMERA_REOPEN_RETRY_DELAY_SECONDS = 0.75
CAMERA_RELEASE_SETTLE_SECONDS = 0.75
SYNC_CAPTURE_START_DELAY_SECONDS = 0.20
SYNC_CAPTURE_FLUSH_FRAMES = 4
PROGRESS_UPDATE_INTERVAL_SECONDS = 2.0
PROGRESS_UPDATE_STEPS = 10

_PROGRESS_PRINT_LOCK = threading.Lock()


UNFINISHED_RUN_ERROR = "Did not finish: the recorder was stopped, crashed, or lost power."


def _progress_message(stage: str, message: str) -> None:
    heartbeat.beat()
    with _PROGRESS_PRINT_LOCK:
        print(f"[{stage}] {message}", flush=True)


class _FrameProgress:
    def __init__(
        self,
        *,
        stage: str,
        label: str,
        total: int,
        unit: str = "frames",
    ) -> None:
        self.stage = str(stage)
        self.label = str(label)
        self.total = max(1, int(total))
        self.unit = str(unit)
        self.step = max(1, (self.total + PROGRESS_UPDATE_STEPS - 1) // PROGRESS_UPDATE_STEPS)
        self.next_count = self.step
        self.last_count = 0
        self.last_print_monotonic = time.perf_counter()

    def start(self, *, detail: Optional[str] = None) -> None:
        suffix = f" ({detail})" if detail else ""
        _progress_message(
            self.stage,
            f"{self.label}: starting; approximately {self.total} {self.unit}{suffix}.",
        )

    def update(self, completed: int) -> None:
        completed = max(0, int(completed))
        if completed < self.next_count:
            return
        now = time.perf_counter()
        if now - self.last_print_monotonic < PROGRESS_UPDATE_INTERVAL_SECONDS:
            return
        percent = min(100.0, 100.0 * float(completed) / float(self.total))
        _progress_message(
            self.stage,
            f"{self.label}: {completed}/{self.total} {self.unit} ({percent:.0f}%).",
        )
        self.last_count = completed
        self.last_print_monotonic = now
        self.next_count = ((completed // self.step) + 1) * self.step

    def finish(self, completed: int) -> None:
        completed = max(0, int(completed))
        _progress_message(
            self.stage,
            f"{self.label}: complete; {completed} {self.unit}.",
        )
        self.last_count = completed


@dataclass
class RunSummary:
    started_at: str
    finished_at: str
    mode: str
    session_name: str
    session_dir: str
    hostname: str
    python_executable: str
    camera_profile: str
    camera_model: str
    camera_infrared: Optional[bool]
    camera_monochrome_output: Optional[bool]
    camera_autofocus_mode: str
    camera_lens_position: Optional[float]
    camera_focus_lock_after_warmup: bool
    camera_autofocus_preflight_enabled: bool
    camera_autofocus_preflight_performed: bool
    camera_autofocus_preflight_lens_position: Optional[float]
    camera_autofocus_preflight_selection_reason: Optional[str]
    camera_autofocus_preflight_final_state: Optional[str]
    camera_autofocus_preflight_best_score: Optional[float]
    camera_autofocus_preflight_elapsed_seconds: Optional[float]
    camera_autofocus_preflight_notes: List[str]
    resolved_tuning_file: Optional[str]
    frames_captured: int
    actual_fps: float
    tracking_elapsed_seconds: Optional[float]
    tracking_frames_processed: int
    tracking_processing_fps: Optional[float]
    video_codec: Optional[str]
    video_path: Optional[str]
    capture_frame_buffer: str
    stream_encoder_stats: Optional[Dict[str, Any]]
    recording_preview_png_path: Optional[str]
    timestamp_path: Optional[str]
    thermal_enabled: bool
    thermal_device_path: Optional[str]
    thermal_frames_captured: int
    thermal_actual_fps: Optional[float]
    thermal_timestamp_path: Optional[str]
    thermal_raw_npy_path: Optional[str]
    thermal_preview_video_path: Optional[str]
    thermal_preview_png_path: Optional[str]
    thermal_metadata_json_path: Optional[str]
    thermal_side_by_side_video_path: Optional[str]
    thermal_side_by_side_png_path: Optional[str]
    realsense_enabled: bool
    realsense_device_serial: Optional[str]
    realsense_frames_captured: int
    realsense_actual_fps: Optional[float]
    realsense_depth_scale_meters: Optional[float]
    realsense_timestamp_path: Optional[str]
    realsense_raw_depth_npy_path: Optional[str]
    realsense_depth_preview_video_path: Optional[str]
    realsense_depth_preview_png_path: Optional[str]
    realsense_color_video_path: Optional[str]
    realsense_color_preview_png_path: Optional[str]
    realsense_metadata_json_path: Optional[str]
    tracked_video_path: Optional[str]
    tracked_thermal_side_by_side_video_path: Optional[str]
    tracked_thermal_side_by_side_png_path: Optional[str]
    raw_csv_path: Optional[str]
    noid_csv_path: Optional[str]
    cleaned_csv_path: Optional[str]
    fps_report_json: Optional[str]
    config_snapshot_path: Optional[str]
    local_index_path: Optional[str]
    local_index_manifest_path: Optional[str]
    warnings: List[str]
    errors: List[str]
    success: bool
    # Which camera of a two-camera Pi made this run. Older summaries lack
    # these and count as the first camera.
    camera_num: int = 0
    camera_id: Optional[str] = None
    camera_session_tag: str = ""
    # Every video file of a recording that was split into consecutive files,
    # in order; empty for a recording kept as one file (see video_path).
    video_parts: List[str] = field(default_factory=list)


@dataclass
class LiveRecordingResult:
    session_name: str
    session_dir: str
    requested_video_codec: str
    requested_mp4_codec: Optional[str]
    video_codec: str
    video_path: str
    video_size_bytes: int
    timestamp_path: Optional[str]
    recording_preview_png_path: Optional[str]
    frames_captured: int
    actual_fps: float
    configured_fps_target: float
    requested_recording_seconds: float
    video_write_warning: Optional[str]


@dataclass
class CameraResetResult:
    used_mock_camera: bool
    detected_cameras_before: Optional[int]
    detected_cameras_after: Optional[int]
    probe_width: int
    probe_height: int
    settle_seconds: float
    note: Optional[str]


@dataclass
class ThermalRecordingArtifacts:
    device_path: str
    raw16_layout: str
    frames_captured: int
    actual_fps: float
    timestamp_path: Optional[Path]
    raw_npy_path: Path
    preview_video_path: Path
    preview_png_path: Optional[Path]
    metadata_json_path: Path


@dataclass
class ThermalSideBySideArtifacts:
    frame_count: int
    video_path: Path
    midpoint_png_path: Optional[Path]


@dataclass
class TrackingVisualizationArtifacts:
    tracked_video_path: Path
    tracked_thermal_side_by_side_video_path: Optional[Path]
    tracked_thermal_side_by_side_png_path: Optional[Path]


@dataclass(frozen=True)
class FrameTimestampRecord:
    time_s: float
    captured_monotonic_s: float
    captured_unix_s: float
    sensor_timestamp_ns: Optional[int] = None
    timestamp_source: str = "host_receive"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_iso_utc(unix_s: float) -> str:
    return datetime.fromtimestamp(float(unix_s), tz=timezone.utc).isoformat(timespec="milliseconds")


def _to_iso_local(unix_s: float) -> str:
    return datetime.fromtimestamp(float(unix_s)).astimezone().isoformat(timespec="milliseconds")


def _build_timestamp_record(
    *,
    start_monotonic: float,
    captured_monotonic_s: float,
    monotonic_to_unix_offset_s: float,
    sensor_timestamp_ns: Optional[int] = None,
    timestamp_source: str = "host_receive",
) -> FrameTimestampRecord:
    return FrameTimestampRecord(
        time_s=float(captured_monotonic_s - start_monotonic),
        captured_monotonic_s=float(captured_monotonic_s),
        captured_unix_s=float(captured_monotonic_s + monotonic_to_unix_offset_s),
        sensor_timestamp_ns=sensor_timestamp_ns,
        timestamp_source=timestamp_source,
    )


def _guard_fleet_queen_local_pipeline(config: Dict[str, Any]) -> None:
    fleet = config.get("fleet", {})
    if not isinstance(fleet, dict):
        return
    role = str(fleet.get("role", "standalone")).strip().lower()
    local_enabled = bool(fleet.get("queen_local_pipeline_enabled", True))
    if role == "queen" and not local_enabled:
        raise ValueError(
            "This node is configured as fleet queen with local pipeline disabled "
            "(fleet.queen_local_pipeline_enabled=false). "
            "Enable it in config to run capture/tracking on queen."
        )


def _normalize_box_preset(box_preset: Any) -> Optional[str]:
    if box_preset is None:
        return None
    text = str(box_preset).strip().lower()
    if text in {"none", "null", ""}:
        return None
    return str(box_preset)


def _normalize_recording_codec(config: Dict[str, Any]) -> str:
    camera = config.get("camera", {})
    if not isinstance(camera, dict):
        return "mp4"
    raw = str(camera.get("codec", "mp4")).strip().lower()
    if raw in {"mjpeg", "mjpg"}:
        return "mjpeg"
    return "mp4"


def _detected_picamera2_camera_count() -> Optional[int]:
    try:
        from picamera2 import Picamera2
    except Exception:
        return None

    try:
        camera_info = Picamera2.global_camera_info()
    except Exception:
        return None
    if isinstance(camera_info, list):
        return len(camera_info)
    return None


def _native_camera_probe_command() -> Optional[list[str]]:
    for command_name in ("rpicam-hello", "libcamera-hello"):
        resolved = shutil.which(command_name)
        if resolved:
            return [resolved, "-t", "750", "--nopreview"]
    return None


def _make_session_paths(config: Dict[str, Any]) -> Tuple[str, Path]:
    hostname = socket.gethostname()
    now = datetime.now()
    # A second camera on the same Pi tags its sessions (for example "cam2"),
    # so two cameras recording in the same second never produce files with
    # the same name, in the data folder or in the local tracking index.
    tag = session_tag(config)
    device = f"{hostname}_{tag}" if tag else hostname
    base_name = f"{device}_{now.strftime('%Y-%m-%d_%H_%M_%S')}"

    data_root = Path(config["system"]["data_root"])
    day_dir = data_root / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    # Session names are only precise to the second, and two scheduled runs can
    # start in the same second (for example the record and track timers of a
    # mixed schedule). Creating the folder is atomic, so claim it exclusively
    # and add a suffix if it is taken; otherwise the runs would share a folder
    # and overwrite each other's files.
    for attempt in range(1, 1000):
        session_name = base_name if attempt == 1 else f"{base_name}_{attempt}"
        session_dir = day_dir / session_name
        try:
            session_dir.mkdir()
        except FileExistsError:
            continue
        return session_name, session_dir
    raise RuntimeError(f"Could not create a unique session folder under {day_dir} for {base_name}.")


def _write_timestamps(
    timestamps: List[float],
    session_dir: Path,
    session_name: str,
    *,
    timestamp_records: Optional[List[FrameTimestampRecord]] = None,
    node_name: Optional[str] = None,
    first_frame: int = 0,
) -> Path:
    path = session_dir / f"{session_name}_frame_timestamps.csv"
    with path.open("w") as f:
        f.write(
            "frame,node,sensor,timestamp_source,time_s,captured_monotonic_s,captured_unix_s,sensor_timestamp_ns,captured_iso_local,captured_iso_utc\n"
        )
        node = str(node_name or socket.gethostname())
        records = list(timestamp_records or [])
        for position, value in enumerate(timestamps):
            i = first_frame + position
            record = records[position] if position < len(records) else None
            if record is None:
                f.write(f"{i},{node},rgb,host_receive,{value:.6f},,,,,\n")
                continue
            f.write(
                f"{i},{node},rgb,{record.timestamp_source},{record.time_s:.6f},{record.captured_monotonic_s:.6f},{record.captured_unix_s:.6f},"
                f"{record.sensor_timestamp_ns if record.sensor_timestamp_ns is not None else ''},"
                f"{_to_iso_local(record.captured_unix_s)},{_to_iso_utc(record.captured_unix_s)}\n"
            )
    return path


def _write_named_timestamps(
    timestamps: List[float],
    session_dir: Path,
    stem: str,
    *,
    timestamp_records: Optional[List[FrameTimestampRecord]] = None,
    node_name: Optional[str] = None,
    sensor_name: str = "thermal",
) -> Path:
    path = session_dir / f"{stem}.csv"
    with path.open("w") as f:
        f.write(
            "frame,node,sensor,timestamp_source,time_s,captured_monotonic_s,captured_unix_s,sensor_timestamp_ns,captured_iso_local,captured_iso_utc\n"
        )
        node = str(node_name or socket.gethostname())
        records = list(timestamp_records or [])
        for i, value in enumerate(timestamps):
            record = records[i] if i < len(records) else None
            if record is None:
                f.write(f"{i},{node},{sensor_name},host_receive,{value:.6f},,,,,\n")
                continue
            f.write(
                f"{i},{node},{sensor_name},{record.timestamp_source},{record.time_s:.6f},{record.captured_monotonic_s:.6f},{record.captured_unix_s:.6f},"
                f"{record.sensor_timestamp_ns if record.sensor_timestamp_ns is not None else ''},"
                f"{_to_iso_local(record.captured_unix_s)},{_to_iso_utc(record.captured_unix_s)}\n"
            )
    return path


def _mock_capture_frames(
    config: Dict[str, Any],
    *,
    progress_label: Optional[str] = None,
    frame_sink: Optional[StreamingVideoWriter] = None,
) -> Tuple[List[Any], List[float], float, List[FrameTimestampRecord]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("numpy is required for mock capture mode") from exc

    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])

    scheduled_times = simulated_frame_times(duration, fps)
    frame_count = len(scheduled_times)
    progress = (
        _FrameProgress(stage="capture", label=progress_label, total=frame_count)
        if progress_label
        else None
    )
    if progress is not None:
        progress.start(detail=f"mock capture, {duration:.1f}s at {fps:.3f} FPS")
    frames = []
    timestamps = []
    timestamp_records: List[FrameTimestampRecord] = []
    start_monotonic = time.perf_counter()
    monotonic_to_unix_offset_s = time.time() - time.perf_counter()
    # The mock camera normally hands over all its frames at once. With
    # runtime.mock_camera_realtime it waits like a real camera, which is what
    # long-recording behaviour (file roll-over, the watchdog) needs to be tried.
    realtime = bool(config["runtime"].get("mock_camera_realtime", False))
    for i, relative_s in enumerate(scheduled_times):
        if realtime:
            wait = start_monotonic + relative_s - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        frame = simulated_rgb_yuv420_frame(width, height, relative_s, duration)
        record = _build_timestamp_record(
            start_monotonic=start_monotonic,
            captured_monotonic_s=start_monotonic + relative_s,
            monotonic_to_unix_offset_s=monotonic_to_unix_offset_s,
        )
        if frame_sink is None:
            frames.append(frame)
        else:
            frame_sink.add(frame, record)
        timestamps.append(relative_s)
        timestamp_records.append(record)
        heartbeat.beat()
        if progress is not None:
            progress.update(i + 1)

    actual_fps = fps
    if progress is not None:
        progress.finish(frame_count)
    return frames, timestamps, actual_fps, timestamp_records


def _capture_frames_from_started_picamera(
    picam2: Any,
    fps: float,
    duration: float,
    *,
    start_monotonic: float | None = None,
    progress_label: Optional[str] = None,
    frame_sink: Optional[StreamingVideoWriter] = None,
) -> Tuple[List[Any], List[float], float, List[FrameTimestampRecord]]:
    """Capture frames for ``duration`` seconds.

    Without ``frame_sink`` every frame is kept and returned (RAM mode). With
    one, each frame is handed to it as it arrives and the returned frame list
    is empty (stream mode); timestamps are returned either way.
    """
    frames: List[Any] = []
    timestamps: List[float] = []
    timestamp_records: List[FrameTimestampRecord] = []

    if start_monotonic is None:
        start = time.perf_counter()
    else:
        start = float(start_monotonic)
        while True:
            now = time.perf_counter()
            if now >= start:
                break
            time.sleep(min(0.001, max(0.0, start - now)))
    monotonic_to_unix_offset_s = time.time() - time.perf_counter()
    frame_index = 0
    target_interval = 1.0 / fps
    estimated_frames = max(1, int(math.ceil(float(duration) * float(fps))))
    progress = (
        _FrameProgress(stage="capture", label=progress_label, total=estimated_frames)
        if progress_label
        else None
    )
    if progress is not None:
        progress.start(detail=f"{duration:.1f}s at {fps:.3f} FPS")
    while (time.perf_counter() - start) < duration:
        now = time.perf_counter()
        expected = start + frame_index * target_interval
        if now >= expected:
            sensor_timestamp_ns = None
            capture_request = getattr(picam2, "capture_request", None)
            if callable(capture_request):
                request = capture_request()
                try:
                    yuv420 = request.make_array("main")
                    metadata = request.get_metadata()
                    raw_sensor_timestamp = metadata.get("SensorTimestamp")
                    if raw_sensor_timestamp is not None:
                        sensor_timestamp_ns = int(raw_sensor_timestamp)
                finally:
                    request.release()
            else:
                yuv420 = picam2.capture_array()
            captured_monotonic_s = time.perf_counter()
            record = _build_timestamp_record(
                start_monotonic=start,
                captured_monotonic_s=captured_monotonic_s,
                monotonic_to_unix_offset_s=monotonic_to_unix_offset_s,
                sensor_timestamp_ns=sensor_timestamp_ns,
                timestamp_source=("sensor_and_host" if sensor_timestamp_ns is not None else "host_receive"),
            )
            if frame_sink is None:
                frames.append(yuv420)
            else:
                frame_sink.add(yuv420, record)
            timestamps.append(record.time_s)
            timestamp_records.append(record)
            heartbeat.beat()
            frame_index += 1
            if progress is not None:
                progress.update(frame_index)
        else:
            # Not due yet: sleep briefly instead of spinning, so this loop does
            # not hold a CPU core (and the interpreter lock) that the encoder
            # and the other sensor threads need.
            time.sleep(min(0.001, max(0.0, expected - now)))

    frame_count = len(timestamps)
    elapsed = timestamps[-1] - timestamps[0] if frame_count > 1 else duration
    actual_fps = (frame_count - 1) / elapsed if elapsed > 0 and frame_count > 1 else float(frame_count) / max(duration, 1e-6)
    if progress is not None:
        progress.finish(frame_count)
    return frames, timestamps, actual_fps, timestamp_records


class _PicameraCaptureSession:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except Exception as exc:  # pragma: no cover - dependency/runtime
            raise RuntimeError(
                "picamera2/libcamera not available. Install on Raspberry Pi OS, or set runtime.use_mock_camera=true."
            ) from exc

        self._Picamera2 = Picamera2
        self._controls = controls
        self.width = int(config["camera"]["width"])
        self.height = int(config["camera"]["height"])
        self.shutter_us = int(config["camera"]["shutter_us"])
        self.digital_zoom = config["camera"].get("digital_zoom")
        self.monochrome_output = bool(config["camera"].get("monochrome_output", False))
        self.image_control_notes: list[str] = []
        self.warmup_s = float(config["runtime"].get("camera_warmup_seconds", 2.0))
        self.resolved_tuning_file = resolve_camera_tuning_file(config)
        self.autofocus_notes: list[str] = []
        self.picam2 = self._open_and_configure()
        self.started = False

    def _construct_picamera2(self) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, CAMERA_REOPEN_RETRY_ATTEMPTS + 1):
            try:
                camera_info = self._Picamera2.global_camera_info()
                if isinstance(camera_info, list) and len(camera_info) == 0:
                    raise RuntimeError(
                        "No camera detected by picamera2/libcamera. "
                        "Check ribbon cable orientation/seating and enable camera stack."
                    )
            except RuntimeError as exc:
                last_error = exc
                if attempt < CAMERA_REOPEN_RETRY_ATTEMPTS:
                    gc.collect()
                    time.sleep(CAMERA_REOPEN_RETRY_DELAY_SECONDS)
                    continue
                raise
            except Exception:
                pass

            try:
                camera_kwargs = picamera2_camera_kwargs(self.config, self._Picamera2)
                if self.resolved_tuning_file:
                    tuning = self._Picamera2.load_tuning_file(str(self.resolved_tuning_file))
                    return self._Picamera2(tuning=tuning, **camera_kwargs)
                return self._Picamera2(**camera_kwargs)
            except IndexError as exc:
                last_error = RuntimeError(
                    "No camera detected by picamera2/libcamera (IndexError during camera open). "
                    "Check ribbon cable orientation/seating, camera power, and that no other process owns the camera."
                )
                if attempt < CAMERA_REOPEN_RETRY_ATTEMPTS:
                    gc.collect()
                    time.sleep(CAMERA_REOPEN_RETRY_DELAY_SECONDS)
                    continue
                raise last_error from exc
            except RuntimeError as exc:
                last_error = exc
                if attempt < CAMERA_REOPEN_RETRY_ATTEMPTS:
                    gc.collect()
                    time.sleep(CAMERA_REOPEN_RETRY_DELAY_SECONDS)
                    continue
                raise
            except Exception as exc:
                if self.resolved_tuning_file:
                    raise RuntimeError(
                        f"Failed to open camera with tuning file '{self.resolved_tuning_file}': {exc}"
                    ) from exc
                raise RuntimeError(f"Failed to open camera: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to open camera for an unknown reason.")

    def _open_and_configure(self) -> Any:
        picam2 = self._construct_picamera2()
        try:
            self.autofocus_preflight_result = run_autofocus_preflight(
                self.config,
                picam2,
                self._controls,
                notes=self.autofocus_notes,
                strict=True,
            )
            preview = picam2.create_preview_configuration({"format": "YUV420", "size": (self.width, self.height)})
            picam2.align_configuration(preview)
            picam2.configure(preview)
        except IndexError as exc:
            try:
                picam2.close()
            except Exception:
                pass
            raise RuntimeError(
                "Camera opened but failed to configure capture stream (IndexError). "
                "This usually means libcamera could not enumerate valid sensor modes."
            ) from exc
        except Exception as exc:
            try:
                picam2.close()
            except Exception:
                pass
            raise RuntimeError(f"Failed to configure capture stream: {exc}") from exc

        picam2.set_controls({"ExposureTime": self.shutter_us})
        if self.monochrome_output:
            try:
                picam2.set_controls({"Saturation": 0.0})
            except Exception:
                pass

        # Sensor gain, white balance, contrast and noise reduction, where the
        # config fixes them. A control this camera lacks is noted, not fatal.
        apply_image_controls(self.config, picam2, self._controls, self.image_control_notes)
        for note in self.image_control_notes:
            print(f"[capture] {note}", flush=True)

        if isinstance(self.digital_zoom, (list, tuple)) and len(self.digital_zoom) == 4:
            picam2.set_controls({"ScalerCrop": tuple(self.digital_zoom)})
        if self.autofocus_preflight_result.performed:
            apply_preflight_lens_lock(
                self.autofocus_preflight_result,
                picam2,
                self._controls,
                notes=self.autofocus_notes,
                strict=True,
            )
        else:
            apply_autofocus_before_start(
                self.config,
                picam2,
                self._controls,
                notes=self.autofocus_notes,
                strict=True,
            )
        return picam2

    def start(self) -> None:
        if self.started:
            return
        self.picam2.start()
        self.started = True
        if not self.autofocus_preflight_result.performed:
            start_autofocus_after_camera_start(
                self.config,
                self.picam2,
                self._controls,
                notes=self.autofocus_notes,
                strict=True,
            )
        time.sleep(max(0.0, self.warmup_s))
        if not self.autofocus_preflight_result.performed:
            lock_autofocus_after_warmup(
                self.config,
                self.picam2,
                self._controls,
                notes=self.autofocus_notes,
                strict=True,
            )

    def capture_for(
        self,
        *,
        fps: float,
        duration: float,
        start_monotonic: float | None = None,
        progress_label: Optional[str] = None,
        frame_sink: Optional[StreamingVideoWriter] = None,
    ) -> Tuple[List[Any], List[float], float, List[FrameTimestampRecord]]:
        self.start()
        return _capture_frames_from_started_picamera(
            self.picam2,
            float(fps),
            float(duration),
            start_monotonic=start_monotonic,
            progress_label=progress_label,
            frame_sink=frame_sink,
        )

    def close(self) -> None:
        if self.started:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.started = False
        try:
            self.picam2.close()
        except Exception:
            pass
        self.picam2 = None
        gc.collect()
        time.sleep(CAMERA_RELEASE_SETTLE_SECONDS)

    def __enter__(self) -> "_PicameraCaptureSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _write_autofocus_capture_diagnostics(
    diagnostics: Optional[dict[str, Any]],
    session: _PicameraCaptureSession,
) -> None:
    if diagnostics is None:
        return
    diagnostics["autofocus_preflight"] = session.autofocus_preflight_result.to_dict()
    diagnostics["autofocus_notes"] = list(session.autofocus_notes)


def _capture_frames_picamera(
    config: Dict[str, Any],
    *,
    diagnostics: Optional[dict[str, Any]] = None,
    progress_label: Optional[str] = None,
    frame_sink: Optional[StreamingVideoWriter] = None,
) -> Tuple[List[Any], List[float], float, List[FrameTimestampRecord]]:
    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])
    with _PicameraCaptureSession(config) as session:
        _write_autofocus_capture_diagnostics(diagnostics, session)
        return session.capture_for(
            fps=fps,
            duration=duration,
            progress_label=progress_label,
            frame_sink=frame_sink,
        )


def _thermal_enabled_for_recording(config: Dict[str, Any]) -> bool:
    thermal = config.get("thermal", {})
    return isinstance(thermal, dict) and bool(thermal.get("enabled", False))


def _realsense_enabled_for_recording(config: Dict[str, Any]) -> bool:
    realsense = config.get("realsense", {})
    return isinstance(realsense, dict) and bool(realsense.get("enabled", False))


def _camera_monochrome_output_enabled(config: Dict[str, Any]) -> bool:
    camera = config.get("camera", {})
    return isinstance(camera, dict) and bool(camera.get("monochrome_output", False))


def _flush_rgb_capture_queue(picam2: Any, frame_count: int) -> None:
    for _ in range(max(0, int(frame_count))):
        try:
            picam2.capture_array()
        except Exception:
            break


def _flush_thermal_capture_queue(capture: Any, frame_count: int) -> None:
    for _ in range(max(0, int(frame_count))):
        try:
            ok, _frame = capture.read()
        except Exception:
            break
        if not ok:
            break


def _infer_thermal_raw16_layout(frame: Any) -> Optional[str]:
    shape = getattr(frame, "shape", None)
    dtype = str(getattr(frame, "dtype", ""))
    if shape is None:
        return None
    if dtype == "uint16" and len(shape) == 2:
        return "uint16_mono16"
    if dtype == "uint8" and len(shape) == 3 and int(shape[2]) == 2:
        return "uint8_2ch_packed16"
    return None


def _decode_thermal_raw16_frame(frame: Any, raw16_layout: str) -> Any:
    if raw16_layout == "uint16_mono16":
        return frame.copy()
    if raw16_layout == "uint8_2ch_packed16":
        packed = frame.copy()
        return packed.view("<u2").reshape(packed.shape[0], packed.shape[1])
    raise RuntimeError(f"Unsupported thermal raw16 layout: {raw16_layout}")


class _ThermalCaptureSession:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        thermal = config.get("thermal", {})
        if not isinstance(thermal, dict):
            raise RuntimeError("thermal config section is missing or malformed.")

        self.device_path = resolve_thermal_device_path(config)
        if not self.device_path:
            raise RuntimeError("No thermal capture device could be resolved from config.")
        self.width = int(thermal.get("width", 160))
        self.height = int(thermal.get("height", 120))
        self.pixel_format = str(thermal.get("pixel_format", "auto")).strip().lower() or "auto"
        if self.pixel_format not in {"auto", "y16"}:
            raise RuntimeError(
                "Synchronized thermal recording currently supports thermal.pixel_format 'auto' or 'y16' only."
            )

        try:
            import cv2
        except Exception as exc:  # pragma: no cover - dependency/runtime
            raise RuntimeError("OpenCV is required for thermal capture.") from exc

        self._cv2 = cv2
        if self.pixel_format in {"auto", "y16"}:
            try:
                _set_v4l2_y16_format(self.device_path, self.width, self.height)
            except Exception:
                pass
        self.capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open thermal device for synchronized recording: {self.device_path}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_CONVERT_RGB"):
            self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            self.capture.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*"Y16 ")))

        self.raw16_layout: Optional[str] = None

    def capture_for(
        self,
        *,
        fps: float,
        duration: float,
        start_monotonic: float | None = None,
        progress_label: Optional[str] = None,
    ) -> Tuple[List[Any], List[float], float, str, List[FrameTimestampRecord]]:
        frames: List[Any] = []
        timestamps: List[float] = []
        timestamp_records: List[FrameTimestampRecord] = []

        if start_monotonic is None:
            start = time.perf_counter()
        else:
            start = float(start_monotonic)
            while True:
                now = time.perf_counter()
                if now >= start:
                    break
                time.sleep(min(0.001, max(0.0, start - now)))
        monotonic_to_unix_offset_s = time.time() - time.perf_counter()

        frame_index = 0
        target_interval = 1.0 / float(fps)
        estimated_frames = max(1, int(math.ceil(float(duration) * float(fps))))
        progress = (
            _FrameProgress(stage="capture", label=progress_label, total=estimated_frames)
            if progress_label
            else None
        )
        if progress is not None:
            progress.start(detail=f"{duration:.1f}s at {float(fps):.3f} FPS")
        first_error: Optional[str] = None
        while (time.perf_counter() - start) < float(duration):
            now = time.perf_counter()
            expected = start + frame_index * target_interval
            if now < expected:
                time.sleep(min(0.001, max(0.0, expected - now)))
                continue

            ok, frame = self.capture.read()
            captured_monotonic_s = time.perf_counter()
            if not ok or frame is None:
                if first_error is None:
                    first_error = "Thermal device opened, but frame reads failed during synchronized capture."
                time.sleep(0.002)
                continue

            layout = _infer_thermal_raw16_layout(frame)
            if layout is None:
                if first_error is None:
                    first_error = (
                        "Thermal capture returned frames, but not in a usable raw16 layout during synchronized capture."
                    )
                time.sleep(0.002)
                continue

            if self.raw16_layout is None:
                self.raw16_layout = layout
            decoded = _decode_thermal_raw16_frame(frame, layout)
            frames.append(decoded)
            record = _build_timestamp_record(
                start_monotonic=start,
                captured_monotonic_s=captured_monotonic_s,
                monotonic_to_unix_offset_s=monotonic_to_unix_offset_s,
            )
            timestamps.append(record.time_s)
            timestamp_records.append(record)
            frame_index += 1
            if progress is not None:
                progress.update(frame_index)

        if self.raw16_layout is None:
            raise RuntimeError(first_error or "Thermal synchronized capture did not yield any usable raw16 frames.")

        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else float(duration)
        actual_fps = (
            (len(timestamps) - 1) / elapsed
            if elapsed > 0 and len(timestamps) > 1
            else float(len(frames)) / max(float(duration), 1e-6)
        )
        if progress is not None:
            progress.finish(len(frames))
        return frames, timestamps, float(actual_fps), self.raw16_layout, timestamp_records

    def close(self) -> None:
        try:
            self.capture.release()
        except Exception:
            pass

    def __enter__(self) -> "_ThermalCaptureSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _mock_capture_thermal_frames(
    config: Dict[str, Any],
    *,
    timestamps: List[float],
    actual_fps: float,
    timestamp_records: List[FrameTimestampRecord],
) -> Tuple[List[Any], List[float], float, str, str, List[FrameTimestampRecord]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("numpy is required for mock thermal capture mode") from exc

    thermal = config.get("thermal", {})
    width = int(thermal.get("width", 160))
    height = int(thermal.get("height", 120))
    duration = float(config.get("capture", {}).get("recording_seconds", 1.0))
    frames: List[Any] = []
    for timestamp in timestamps:
        frames.append(simulated_thermal_frame(width, height, timestamp, duration))
    return frames, list(timestamps), float(actual_fps), "uint16_mono16", "mock://thermal", list(timestamp_records)


def _capture_rgb_and_optional_sensors(
    config: Dict[str, Any],
    *,
    session_dir: Path,
    session_name: str,
    diagnostics: Optional[dict[str, Any]] = None,
    start_monotonic_override: Optional[float] = None,
    ready_callback: Optional[Any] = None,
    capture_node_name: Optional[str] = None,
    frame_sink: Optional[StreamingVideoWriter] = None,
) -> Tuple[
    List[Any],
    List[float],
    float,
    List[FrameTimestampRecord],
    Optional[dict[str, Any]],
    Optional[RealSenseRecordingResult],
]:
    # With frame_sink set (stream mode) RGB frames go to the sink as they
    # arrive and the returned RGB frame list is empty; count RGB frames by
    # their timestamps.
    thermal_enabled = _thermal_enabled_for_recording(config)
    realsense_enabled = _realsense_enabled_for_recording(config)
    if (
        not thermal_enabled
        and not realsense_enabled
        and start_monotonic_override is None
    ):
        _progress_message("capture", "Preparing RGB capture.")
        frames, timestamps, actual_fps, timestamp_records = _capture_frames(
            config,
            diagnostics=diagnostics,
            progress_label="RGB capture",
            frame_sink=frame_sink,
        )
        _progress_message(
            "capture",
            f"RGB acquisition finished: {len(timestamps)} frames at {actual_fps:.3f} FPS.",
        )
        return frames, timestamps, actual_fps, timestamp_records, None, None

    if bool(config["runtime"].get("use_mock_camera", False)):
        active_streams = ["RGB"]
        if thermal_enabled:
            active_streams.append("thermal")
        if realsense_enabled:
            active_streams.append("RealSense")
        sensor_label = "+".join(active_streams)
        _progress_message("capture", f"Preparing synchronized mock {sensor_label} capture.")
        if ready_callback is not None:
            ready_callback()
        if start_monotonic_override is not None:
            while time.perf_counter() < float(start_monotonic_override):
                time.sleep(
                    min(
                        0.001,
                        max(0.0, float(start_monotonic_override) - time.perf_counter()),
                    )
                )
        frames, timestamps, actual_fps, timestamp_records = _capture_frames(
            config,
            diagnostics=diagnostics,
            progress_label="RGB capture",
            frame_sink=frame_sink,
        )
        thermal_result = None
        if thermal_enabled:
            thermal_frames, thermal_timestamps, thermal_actual_fps, raw16_layout, device_path, thermal_timestamp_records = _mock_capture_thermal_frames(
                config,
                timestamps=timestamps,
                actual_fps=actual_fps,
                timestamp_records=timestamp_records,
            )
            thermal_result = {
                "device_path": device_path,
                "frames": thermal_frames,
                "timestamps": thermal_timestamps,
                "actual_fps": thermal_actual_fps,
                "raw16_layout": raw16_layout,
                "timestamp_records": thermal_timestamp_records,
            }

        realsense_artifacts = None
        if realsense_enabled:
            realsense_fps = float(config.get("realsense", {}).get("fps", 30))
            requested_frames = len(
                simulated_frame_times(float(config["capture"]["recording_seconds"]), realsense_fps)
            )
            progress = _FrameProgress(
                stage="capture",
                label="Simulated RealSense capture",
                total=requested_frames,
            )
            progress.start(
                detail=(
                    f"{float(config['capture']['recording_seconds']):.1f}s at "
                    f"{realsense_fps:.3f} FPS"
                )
            )
            depth_preview_progress: Optional[_FrameProgress] = None

            def update_depth_preview(completed: int, total: int) -> None:
                nonlocal depth_preview_progress
                if depth_preview_progress is None:
                    depth_preview_progress = _FrameProgress(
                        stage="output",
                        label="Simulated RealSense depth preview video",
                        total=total,
                    )
                    depth_preview_progress.start()
                depth_preview_progress.update(completed)
                if completed >= total:
                    depth_preview_progress.finish(completed)

            shared_start = (
                timestamp_records[0].captured_monotonic_s
                if timestamp_records
                else time.perf_counter()
            )
            realsense_artifacts = capture_simulated_realsense_recording(
                config,
                session_dir=session_dir,
                session_name=session_name,
                duration=float(config["capture"]["recording_seconds"]),
                start_monotonic=shared_start,
                progress_callback=lambda completed, _total: progress.update(completed),
                status_callback=lambda message: _progress_message("output", message),
                depth_preview_progress_callback=update_depth_preview,
                node_name=capture_node_name,
            )
            progress.finish(realsense_artifacts.frames_captured)

        counts = [f"{len(timestamps)} RGB"]
        if thermal_result is not None:
            counts.append(f"{len(thermal_result['frames'])} thermal")
        if realsense_artifacts is not None:
            counts.append(f"{realsense_artifacts.frames_captured} RealSense")
        _progress_message(
            "capture",
            "Synchronized mock acquisition finished: " + ", ".join(counts) + " frames.",
        )
        return (
            frames,
            timestamps,
            actual_fps,
            timestamp_records,
            thermal_result,
            realsense_artifacts,
        )

    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])

    rgb_result: dict[str, Any] = {}
    thermal_result: dict[str, Any] = {}
    realsense_result: dict[str, RealSenseRecordingResult] = {}
    errors: list[str] = []
    active_streams = ["RGB"]
    if thermal_enabled:
        active_streams.append("thermal")
    if realsense_enabled:
        active_streams.append("RealSense")
    _progress_message("capture", f"Preparing synchronized {'+'.join(active_streams)} capture.")

    def rgb_worker(session: _PicameraCaptureSession) -> None:
        try:
            frames, timestamps, actual_fps, timestamp_records = session.capture_for(
                fps=fps,
                duration=duration,
                start_monotonic=start_monotonic,
                progress_label="RGB capture",
                frame_sink=frame_sink,
            )
            rgb_result.update(
                {
                    "frames": frames,
                    "timestamps": timestamps,
                    "actual_fps": actual_fps,
                    "timestamp_records": timestamp_records,
                }
            )
        except Exception as exc:
            errors.append(f"RGB synchronized capture failed: {exc}")

    def thermal_worker(session: _ThermalCaptureSession) -> None:
        try:
            frames, timestamps, actual_fps, raw16_layout, timestamp_records = session.capture_for(
                fps=fps,
                duration=duration,
                start_monotonic=start_monotonic,
                progress_label="Thermal capture",
            )
            thermal_result.update(
                {
                    "device_path": session.device_path,
                    "frames": frames,
                    "timestamps": timestamps,
                    "actual_fps": actual_fps,
                    "raw16_layout": raw16_layout,
                    "timestamp_records": timestamp_records,
                }
            )
        except Exception as exc:
            errors.append(f"Thermal synchronized capture failed: {exc}")

    def realsense_worker(session: RealSenseRecordingSession) -> None:
        requested_frames = max(
            1,
            int(math.ceil(duration * float(config.get("realsense", {}).get("fps", 30)))),
        )
        progress = _FrameProgress(
            stage="capture",
            label="RealSense capture",
            total=requested_frames,
        )
        progress.start(
            detail=(
                f"{duration:.1f}s at {float(config.get('realsense', {}).get('fps', 30)):.3f} FPS"
            )
        )
        depth_preview_progress: Optional[_FrameProgress] = None

        def update_depth_preview(completed: int, total: int) -> None:
            nonlocal depth_preview_progress
            if depth_preview_progress is None:
                depth_preview_progress = _FrameProgress(
                    stage="output",
                    label="RealSense depth preview video",
                    total=total,
                )
                depth_preview_progress.start()
            depth_preview_progress.update(completed)
            if completed >= total:
                depth_preview_progress.finish(completed)

        try:
            result = session.capture_for(
                duration=duration,
                start_monotonic=start_monotonic,
                progress_callback=lambda completed, _total: progress.update(completed),
                status_callback=lambda message: _progress_message("output", message),
                depth_preview_progress_callback=update_depth_preview,
            )
            realsense_result["artifacts"] = result
            progress.finish(result.frames_captured)
        except Exception as exc:
            errors.append(f"RealSense synchronized capture failed: {exc}")

    with ExitStack() as stack:
        rgb_session = stack.enter_context(_PicameraCaptureSession(config))
        _write_autofocus_capture_diagnostics(diagnostics, rgb_session)
        rgb_session.start()
        thermal_session = (
            stack.enter_context(_ThermalCaptureSession(config)) if thermal_enabled else None
        )
        realsense_session = (
            stack.enter_context(
                RealSenseRecordingSession(
                    config,
                    session_dir=session_dir,
                    session_name=session_name,
                    node_name=capture_node_name,
                )
            )
            if realsense_enabled
            else None
        )

        # Flush queued frames immediately before the shared start so streams begin from current data
        # instead of buffered warmup frames.
        _flush_rgb_capture_queue(rgb_session.picam2, SYNC_CAPTURE_FLUSH_FRAMES)
        if thermal_session is not None:
            _flush_thermal_capture_queue(thermal_session.capture, SYNC_CAPTURE_FLUSH_FRAMES)
        start_monotonic = (
            float(start_monotonic_override)
            if start_monotonic_override is not None
            else time.perf_counter() + SYNC_CAPTURE_START_DELAY_SECONDS
        )
        if ready_callback is not None:
            ready_callback()
        _progress_message(
            "capture",
            f"Starting {len(active_streams)} streams from one host deadline for {duration:.1f}s.",
        )
        threads = [threading.Thread(target=rgb_worker, args=(rgb_session,), daemon=True)]
        if thermal_session is not None:
            threads.append(threading.Thread(target=thermal_worker, args=(thermal_session,), daemon=True))
        if realsense_session is not None:
            threads.append(threading.Thread(target=realsense_worker, args=(realsense_session,), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    if errors:
        raise RuntimeError(" | ".join(errors))
    counts = [f"{len(rgb_result['timestamps'])} RGB"]
    if thermal_enabled:
        counts.append(f"{len(thermal_result['frames'])} thermal")
    if realsense_enabled:
        counts.append(f"{realsense_result['artifacts'].frames_captured} RealSense")
    _progress_message(
        "capture",
        "Synchronized acquisition finished: " + ", ".join(counts) + " frames.",
    )
    return (
        rgb_result["frames"],
        rgb_result["timestamps"],
        float(rgb_result["actual_fps"]),
        rgb_result["timestamp_records"],
        thermal_result or None,
        realsense_result.get("artifacts"),
    )


def _write_thermal_recording_outputs(
    *,
    session_dir: Path,
    session_name: str,
    device_path: str,
    frames: List[Any],
    timestamps: List[float],
    actual_fps: float,
    raw16_layout: str,
    timestamp_records: Optional[List[FrameTimestampRecord]] = None,
    node_name: Optional[str] = None,
) -> ThermalRecordingArtifacts:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError(f"OpenCV/numpy are required for thermal recording outputs: {exc}") from exc

    if not frames:
        raise RuntimeError("No thermal frames were captured.")

    _progress_message("output", f"Writing thermal raw data for {len(frames)} frames.")
    stack = np.stack(frames, axis=0)
    raw_npy_path = session_dir / f"{session_name}_thermal_raw16.npy"
    np.save(raw_npy_path, stack)

    timestamp_path = _write_named_timestamps(
        timestamps,
        session_dir,
        f"{session_name}_thermal_frame_timestamps",
        timestamp_records=timestamp_records,
        node_name=node_name,
        sensor_name="thermal",
    ) if timestamps else None

    min_value = int(stack.min())
    max_value = int(stack.max())
    if max_value == min_value:
        preview8 = np.zeros_like(stack, dtype=np.uint8)
    else:
        preview8 = ((stack.astype(np.float32) - float(min_value)) * (255.0 / float(max_value - min_value))).clip(0, 255).astype(np.uint8)

    preview_video_path = session_dir / f"{session_name}_thermal_preview.avi"
    writer = cv2.VideoWriter(
        str(preview_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(actual_fps if actual_fps > 0 else 1.0),
        (int(stack.shape[2]), int(stack.shape[1])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open thermal preview video writer for {preview_video_path}")
    preview_progress = _FrameProgress(
        stage="output",
        label="Thermal preview video",
        total=int(stack.shape[0]),
    )
    preview_progress.start()
    try:
        for frame_index, frame8 in enumerate(preview8, start=1):
            writer.write(cv2.applyColorMap(frame8, cv2.COLORMAP_INFERNO))
            preview_progress.update(frame_index)
    finally:
        writer.release()
    preview_progress.finish(int(stack.shape[0]))

    mid_idx = max(0, min(int(stack.shape[0]) - 1, int(stack.shape[0]) // 2))
    preview_png_path = session_dir / f"{session_name}_thermal_midframe.png"
    preview_png_ok = cv2.imwrite(
        str(preview_png_path),
        cv2.applyColorMap(preview8[mid_idx], cv2.COLORMAP_INFERNO),
    )
    if not preview_png_ok:
        preview_png_path = None

    metadata_json_path = session_dir / f"{session_name}_thermal_metadata.json"
    metadata_json_path.write_text(
        json.dumps(
            {
                "captured_at": _now_iso(),
                "device_path": device_path,
                "raw16_layout": raw16_layout,
                "frame_count": int(stack.shape[0]),
                "frame_shape": [int(stack.shape[1]), int(stack.shape[2])],
                "actual_fps": float(actual_fps),
                "min_value": min_value,
                "max_value": max_value,
                "timestamp_csv": str(timestamp_path) if timestamp_path else None,
                "preview_video": str(preview_video_path),
                "raw_npy": str(raw_npy_path),
            },
            indent=2,
        )
    )
    _progress_message("output", "Thermal raw data, timestamps, preview, and metadata are complete.")

    return ThermalRecordingArtifacts(
        device_path=device_path,
        raw16_layout=raw16_layout,
        frames_captured=int(stack.shape[0]),
        actual_fps=float(actual_fps),
        timestamp_path=timestamp_path,
        raw_npy_path=raw_npy_path,
        preview_video_path=preview_video_path,
        preview_png_path=preview_png_path,
        metadata_json_path=metadata_json_path,
    )


def _draw_preview_label(image: Any, text: str) -> Any:
    try:
        import cv2
    except Exception:
        return image

    height = int(image.shape[0]) if getattr(image, "shape", None) is not None else 0
    font_scale = max(0.7, min(2.0, float(height) / 1400.0))
    thickness = max(1, int(round(font_scale * 2)))
    origin = (18, max(28, int(round(42 * font_scale))))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def _match_rgb_to_thermal_pairs(
    *,
    rgb_frame_count: int,
    rgb_timestamps: List[float],
    thermal_frame_count: int,
    thermal_timestamps: List[float],
    fps: float,
) -> List[tuple[int, int, float, float]]:
    thermal_times = list(thermal_timestamps)
    if not thermal_times:
        thermal_times = [float(i) / max(float(fps), 1e-6) for i in range(thermal_frame_count)]
    rgb_times = list(rgb_timestamps)
    if not rgb_times:
        rgb_times = [float(i) / max(float(fps), 1e-6) for i in range(rgb_frame_count)]

    thermal_times = thermal_times[:thermal_frame_count]
    rgb_times = rgb_times[:rgb_frame_count]
    if not rgb_times or not thermal_times:
        return []

    matched_pairs: List[tuple[int, int, float, float]] = []
    thermal_idx = 0
    for rgb_idx, rgb_time in enumerate(rgb_times):
        while (
            thermal_idx + 1 < len(thermal_times)
            and abs(thermal_times[thermal_idx + 1] - rgb_time) <= abs(thermal_times[thermal_idx] - rgb_time)
        ):
            thermal_idx += 1
        matched_pairs.append((rgb_idx, thermal_idx, rgb_time, thermal_times[thermal_idx]))
    return matched_pairs


def _write_rgb_thermal_side_by_side_outputs(
    *,
    session_dir: Path,
    session_name: str,
    rgb_frames: List[Any],
    rgb_timestamps: List[float],
    thermal_frames: List[Any],
    thermal_timestamps: List[float],
    fps: float,
    monochrome_output: bool = False,
) -> ThermalSideBySideArtifacts:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError(f"OpenCV/numpy are required for RGB+thermal comparison outputs: {exc}") from exc

    if not rgb_frames or not thermal_frames:
        raise RuntimeError("No overlapping RGB/thermal frames were available for side-by-side output.")

    matched_pairs = _match_rgb_to_thermal_pairs(
        rgb_frame_count=len(rgb_frames),
        rgb_timestamps=rgb_timestamps,
        thermal_frame_count=len(thermal_frames),
        thermal_timestamps=thermal_timestamps,
        fps=fps,
    )

    frame_count = len(matched_pairs)
    if frame_count <= 0:
        raise RuntimeError("No overlapping RGB/thermal frames were available for side-by-side output.")

    first_rgb = _capture_frame_to_bgr(rgb_frames[0], monochrome_output=monochrome_output)
    rgb_height, rgb_width = int(first_rgb.shape[0]), int(first_rgb.shape[1])
    thermal_height, thermal_width = int(thermal_frames[0].shape[0]), int(thermal_frames[0].shape[1])
    scaled_thermal_width = max(1, int(round(float(rgb_height) * float(thermal_width) / float(max(1, thermal_height)))))

    global_min = int(min(int(frame.min()) for frame in thermal_frames[:frame_count]))
    global_max = int(max(int(frame.max()) for frame in thermal_frames[:frame_count]))
    range_value = max(1, global_max - global_min)

    video_path = session_dir / f"{session_name}_rgb_thermal_side_by_side.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps if fps > 0 else 1.0),
        (rgb_width + scaled_thermal_width, rgb_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open RGB+thermal side-by-side writer for {video_path}")

    progress = _FrameProgress(
        stage="output",
        label="RGB+thermal side-by-side video",
        total=frame_count,
    )
    progress.start()
    midpoint_png_path: Optional[Path] = None
    midpoint_idx = max(0, min(frame_count - 1, frame_count // 2))
    try:
        for pair_index, (rgb_idx, thermal_idx, rgb_time, thermal_time) in enumerate(matched_pairs):
            rgb_bgr = _capture_frame_to_bgr(rgb_frames[rgb_idx], monochrome_output=monochrome_output)
            thermal_raw = thermal_frames[thermal_idx]
            thermal_8 = (
                ((thermal_raw.astype(np.float32) - float(global_min)) * (255.0 / float(range_value)))
                .clip(0, 255)
                .astype(np.uint8)
            )
            thermal_color = cv2.applyColorMap(thermal_8, cv2.COLORMAP_INFERNO)
            thermal_scaled = cv2.resize(
                thermal_color,
                (scaled_thermal_width, rgb_height),
                interpolation=cv2.INTER_NEAREST,
            )

            delta_text = f" dt={thermal_time - rgb_time:+.3f}s"
            rgb_bgr = _draw_preview_label(rgb_bgr, f"RGB #{rgb_idx} t={rgb_time:.3f}s")
            thermal_scaled = _draw_preview_label(
                thermal_scaled,
                f"Thermal #{thermal_idx} t={thermal_time:.3f}s{delta_text}",
            )

            combined = np.concatenate([rgb_bgr, thermal_scaled], axis=1)
            writer.write(combined)
            progress.update(pair_index + 1)
            if pair_index == midpoint_idx:
                midpoint_png_path = session_dir / f"{session_name}_rgb_thermal_side_by_side_midframe.png"
                cv2.imwrite(str(midpoint_png_path), combined)
    finally:
        writer.release()
    progress.finish(frame_count)

    return ThermalSideBySideArtifacts(
        frame_count=frame_count,
        video_path=video_path,
        midpoint_png_path=midpoint_png_path,
    )


def _write_tracked_rgb_thermal_side_by_side_outputs(
    *,
    session_dir: Path,
    session_name: str,
    tracked_video_path: Path,
    rgb_timestamps: List[float],
    thermal_frames: List[Any],
    thermal_timestamps: List[float],
    fps: float,
) -> ThermalSideBySideArtifacts:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError(f"OpenCV/numpy are required for tracked RGB+thermal comparison outputs: {exc}") from exc

    if not thermal_frames:
        raise RuntimeError("No thermal frames were available for tracked RGB+thermal side-by-side output.")

    capture = cv2.VideoCapture(str(tracked_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open tracked video for RGB+thermal side-by-side output: {tracked_video_path}")

    try:
        declared_rgb_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        rgb_frame_count = declared_rgb_frames if declared_rgb_frames > 0 else len(rgb_timestamps)
        rgb_frame_count = min(rgb_frame_count, len(rgb_timestamps)) if rgb_timestamps else rgb_frame_count
        matched_pairs = _match_rgb_to_thermal_pairs(
            rgb_frame_count=rgb_frame_count,
            rgb_timestamps=rgb_timestamps,
            thermal_frame_count=len(thermal_frames),
            thermal_timestamps=thermal_timestamps,
            fps=fps,
        )
        if not matched_pairs:
            raise RuntimeError("No overlapping tracked RGB/thermal frames were available for side-by-side output.")

        ok, first_rgb = capture.read()
        if not ok or first_rgb is None:
            raise RuntimeError(f"Could not read first frame from tracked video: {tracked_video_path}")

        rgb_height, rgb_width = int(first_rgb.shape[0]), int(first_rgb.shape[1])
        thermal_height, thermal_width = int(thermal_frames[0].shape[0]), int(thermal_frames[0].shape[1])
        scaled_thermal_width = max(1, int(round(float(rgb_height) * float(thermal_width) / float(max(1, thermal_height)))))

        global_min = int(min(int(frame.min()) for frame in thermal_frames[: len(matched_pairs)]))
        global_max = int(max(int(frame.max()) for frame in thermal_frames[: len(matched_pairs)]))
        range_value = max(1, global_max - global_min)

        video_path = session_dir / f"{session_name}_tracked_rgb_thermal_side_by_side.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(fps if fps > 0 else 1.0),
            (rgb_width + scaled_thermal_width, rgb_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open tracked RGB+thermal side-by-side writer for {video_path}")

        progress = _FrameProgress(
            stage="output",
            label="Tracked RGB+thermal side-by-side video",
            total=len(matched_pairs),
        )
        progress.start()
        midpoint_png_path: Optional[Path] = None
        midpoint_idx = max(0, min(len(matched_pairs) - 1, len(matched_pairs) // 2))
        current_rgb = first_rgb
        try:
            for pair_index, (_rgb_idx, thermal_idx, rgb_time, thermal_time) in enumerate(matched_pairs):
                if pair_index > 0:
                    ok, next_rgb = capture.read()
                    if not ok or next_rgb is None:
                        raise RuntimeError(
                            f"Tracked video ended early while building RGB+thermal side-by-side output: {tracked_video_path}"
                        )
                    current_rgb = next_rgb

                rgb_bgr = current_rgb.copy()
                thermal_raw = thermal_frames[thermal_idx]
                thermal_8 = (
                    ((thermal_raw.astype(np.float32) - float(global_min)) * (255.0 / float(range_value)))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                thermal_color = cv2.applyColorMap(thermal_8, cv2.COLORMAP_INFERNO)
                thermal_scaled = cv2.resize(
                    thermal_color,
                    (scaled_thermal_width, rgb_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                delta_text = f" dt={thermal_time - rgb_time:+.3f}s"
                rgb_bgr = _draw_preview_label(rgb_bgr, f"Tracked RGB t={rgb_time:.3f}s")
                thermal_scaled = _draw_preview_label(
                    thermal_scaled,
                    f"Thermal #{thermal_idx} t={thermal_time:.3f}s{delta_text}",
                )
                combined = np.concatenate([rgb_bgr, thermal_scaled], axis=1)
                writer.write(combined)
                progress.update(pair_index + 1)
                if pair_index == midpoint_idx:
                    midpoint_png_path = session_dir / f"{session_name}_tracked_rgb_thermal_side_by_side_midframe.png"
                    cv2.imwrite(str(midpoint_png_path), combined)
        finally:
            writer.release()
        progress.finish(len(matched_pairs))
    finally:
        capture.release()

    return ThermalSideBySideArtifacts(
        frame_count=len(matched_pairs),
        video_path=video_path,
        midpoint_png_path=midpoint_png_path,
    )


def _render_tracking_visualizations(
    *,
    config: Dict[str, Any],
    session_dir: Path,
    session_name: str,
    video_path: Path,
    raw_csv_path: Path,
    rgb_timestamps: List[float],
    thermal_frames: Optional[List[Any]],
    thermal_timestamps: Optional[List[float]],
    fps: float,
) -> TrackingVisualizationArtifacts:
    try:
        from bumblebox_desktop.visualization import render_tracking_video
    except Exception as exc:
        raise RuntimeError(f"Could not import tracked-video renderer: {exc}") from exc

    tracked_video_path = session_dir / f"{session_name}_tracked.mp4"
    render_tracking_video(
        video_path=video_path,
        tracking_csv_path=raw_csv_path,
        output_video_path=tracked_video_path,
    )

    tracked_side_by_side: Optional[ThermalSideBySideArtifacts] = None
    if thermal_frames and thermal_timestamps is not None:
        tracked_side_by_side = _write_tracked_rgb_thermal_side_by_side_outputs(
            session_dir=session_dir,
            session_name=session_name,
            tracked_video_path=tracked_video_path,
            rgb_timestamps=rgb_timestamps,
            thermal_frames=thermal_frames,
            thermal_timestamps=thermal_timestamps,
            fps=fps,
        )

    return TrackingVisualizationArtifacts(
        tracked_video_path=tracked_video_path,
        tracked_thermal_side_by_side_video_path=(
            tracked_side_by_side.video_path if tracked_side_by_side else None
        ),
        tracked_thermal_side_by_side_png_path=(
            tracked_side_by_side.midpoint_png_path if tracked_side_by_side else None
        ),
    )


def open_capture_probe_session(
    config: Dict[str, Any],
    *,
    use_mock_camera: Optional[bool] = None,
) -> Optional[_PicameraCaptureSession]:
    probe_config = deepcopy(config)
    probe_config.setdefault("runtime", {})
    if use_mock_camera is not None:
        probe_config["runtime"]["use_mock_camera"] = bool(use_mock_camera)
    if bool(probe_config["runtime"].get("use_mock_camera", False)):
        return None
    return _PicameraCaptureSession(probe_config)


def reset_camera_runtime(
    config: Dict[str, Any],
    *,
    settle_seconds: float = 1.5,
    probe_width: int = 640,
    probe_height: int = 480,
) -> CameraResetResult:
    reset_config = deepcopy(config)
    reset_config.setdefault("runtime", {})
    if bool(reset_config["runtime"].get("use_mock_camera", False)):
        return CameraResetResult(
            used_mock_camera=True,
            detected_cameras_before=None,
            detected_cameras_after=None,
            probe_width=int(probe_width),
            probe_height=int(probe_height),
            settle_seconds=0.0,
            note="runtime.use_mock_camera=true, so no hardware camera reset was performed.",
        )

    reset_config.setdefault("camera", {})
    reset_config["camera"]["width"] = int(probe_width)
    reset_config["camera"]["height"] = int(probe_height)
    reset_config["camera"]["digital_zoom"] = None
    reset_config["camera"]["noise_reduction"] = "Auto"
    reset_config["camera"]["tuning_file"] = None
    reset_config["camera"]["model"] = "auto"
    reset_config["camera"]["infrared"] = None
    reset_config["camera"]["autofocus_mode"] = "default"
    reset_config["camera"]["autofocus_preflight_enabled"] = False
    reset_config["camera"].setdefault("shutter_us", 2500)
    reset_config["runtime"]["camera_warmup_seconds"] = min(
        max(float(reset_config["runtime"].get("camera_warmup_seconds", 0.25)), 0.0),
        0.5,
    )

    detected_before = _detected_picamera2_camera_count()
    native_command = _native_camera_probe_command()
    native_probe_error: Optional[str] = None
    if native_command is not None:
        proc = subprocess.run(native_command, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            wait_seconds = max(float(settle_seconds), CAMERA_RELEASE_SETTLE_SECONDS)
            gc.collect()
            time.sleep(wait_seconds)
            return CameraResetResult(
                used_mock_camera=False,
                detected_cameras_before=detected_before,
                detected_cameras_after=None,
                probe_width=int(probe_width),
                probe_height=int(probe_height),
                settle_seconds=wait_seconds,
                note=(
                    f"Camera reset completed via native camera app '{Path(native_command[0]).name}'. "
                    "This bypasses the Picamera2 reset probe and is the preferred recovery path on Pi when available."
                ),
            )
        native_probe_error = (
            f"Native camera probe '{Path(native_command[0]).name}' failed "
            f"(exit {proc.returncode}). "
            f"stdout: {(proc.stdout or '').strip() or '(none)'} "
            f"stderr: {(proc.stderr or '').strip() or '(none)'}"
        )

    try:
        with _PicameraCaptureSession(reset_config) as session:
            session.start()
            try:
                session.picam2.capture_array()
            except Exception:
                pass
    except Exception as exc:
        detected_after_failure = _detected_picamera2_camera_count()
        raise RuntimeError(
            "Camera reset probe failed. "
            f"Detected cameras before reset: {detected_before if detected_before is not None else 'unknown'}. "
            f"Detected cameras after failure: {detected_after_failure if detected_after_failure is not None else 'unknown'}. "
            + (f"Native probe error: {native_probe_error}. " if native_probe_error else "")
            + f"Underlying error: {exc}"
        ) from exc

    wait_seconds = max(float(settle_seconds), CAMERA_RELEASE_SETTLE_SECONDS)
    gc.collect()
    time.sleep(wait_seconds)
    detected_after = _detected_picamera2_camera_count()

    note = (
        "Camera reset probe completed with a conservative 640x480 open/close cycle "
        "using default libcamera sensor tuning."
    )
    if detected_after == 0:
        note += " picamera2 still reports no cameras after reset."
    elif detected_before == 0 and detected_after and detected_after > 0:
        note += " The camera became visible again after reset."

    return CameraResetResult(
        used_mock_camera=False,
        detected_cameras_before=detected_before,
        detected_cameras_after=detected_after,
        probe_width=int(probe_width),
        probe_height=int(probe_height),
        settle_seconds=wait_seconds,
        note=note,
    )


def _capture_frames(
    config: Dict[str, Any],
    *,
    diagnostics: Optional[dict[str, Any]] = None,
    progress_label: Optional[str] = None,
    frame_sink: Optional[StreamingVideoWriter] = None,
) -> Tuple[List[Any], List[float], float, List[FrameTimestampRecord]]:
    if bool(config["runtime"].get("use_mock_camera", False)):
        return _mock_capture_frames(config, progress_label=progress_label, frame_sink=frame_sink)
    return _capture_frames_picamera(
        config,
        diagnostics=diagnostics,
        progress_label=progress_label,
        frame_sink=frame_sink,
    )


def capture_probe(
    config: Dict[str, Any],
    *,
    fps_target: Optional[float] = None,
    recording_seconds: Optional[float] = None,
    use_mock_camera: Optional[bool] = None,
) -> Tuple[int, float, float]:
    probe_config = deepcopy(config)
    probe_config.setdefault("camera", {})
    probe_config.setdefault("capture", {})
    probe_config.setdefault("runtime", {})

    if fps_target is not None:
        probe_config["camera"]["fps_target"] = float(fps_target)
    if recording_seconds is not None:
        probe_config["capture"]["recording_seconds"] = float(recording_seconds)
    if use_mock_camera is not None:
        probe_config["runtime"]["use_mock_camera"] = bool(use_mock_camera)

    start = time.perf_counter()
    frames, _timestamps, actual_fps, _timestamp_records = _capture_frames(probe_config)
    elapsed = time.perf_counter() - start
    frame_count = len(frames)
    del frames
    gc.collect()
    return frame_count, float(actual_fps), float(elapsed)


def _stream_mjpeg_from_started_picamera(
    picam2: Any,
    *,
    fps: float,
    duration: float,
    width: int,
    height: int,
    output_path: Path,
    monochrome_output: bool = False,
) -> Tuple[int, float, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("OpenCV is required for MJPEG sweep probes.") from exc

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MJPEG probe writer for {output_path}")

    timestamps: List[float] = []
    frame_count = 0
    try:
        start = time.perf_counter()
        target_interval = 1.0 / float(fps)
        while (time.perf_counter() - start) < float(duration):
            now = time.perf_counter()
            expected = start + frame_count * target_interval
            if now >= expected:
                yuv420 = picam2.capture_array()
                writer.write(_frame_to_bgr(yuv420, monochrome_output=monochrome_output))
                timestamps.append(now - start)
                frame_count += 1
    finally:
        writer.release()

    elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else float(duration)
    actual_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 and len(timestamps) > 1 else float(frame_count) / max(float(duration), 1e-6)
    size_bytes = int(output_path.stat().st_size) if output_path.exists() else 0
    return int(frame_count), float(actual_fps), size_bytes


def mjpeg_record_probe(
    config: Dict[str, Any],
    *,
    output_path: str | Path,
    fps_target: Optional[float] = None,
    recording_seconds: Optional[float] = None,
    use_mock_camera: Optional[bool] = None,
    session: Optional[_PicameraCaptureSession] = None,
) -> Tuple[int, float, float, int]:
    probe_config = deepcopy(config)
    probe_config.setdefault("camera", {})
    probe_config.setdefault("capture", {})
    probe_config.setdefault("runtime", {})

    if fps_target is not None:
        probe_config["camera"]["fps_target"] = float(fps_target)
    if recording_seconds is not None:
        probe_config["capture"]["recording_seconds"] = float(recording_seconds)
    if use_mock_camera is not None:
        probe_config["runtime"]["use_mock_camera"] = bool(use_mock_camera)

    fps = float(probe_config["camera"]["fps_target"])
    duration = float(probe_config["capture"]["recording_seconds"])
    width = int(probe_config["camera"]["width"])
    height = int(probe_config["camera"]["height"])
    output_path = Path(output_path)

    if bool(probe_config["runtime"].get("use_mock_camera", False)):
        start = time.perf_counter()
        frames, _timestamps, actual_fps, _timestamp_records = _mock_capture_frames(probe_config)
        capture_elapsed = time.perf_counter() - start
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency/runtime
            raise RuntimeError("OpenCV is required for MJPEG sweep probes.") from exc
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open MJPEG probe writer for {output_path}")
        try:
            for frame in frames:
                writer.write(
                    _frame_to_bgr(
                        frame,
                        monochrome_output=_camera_monochrome_output_enabled(probe_config),
                    )
                )
        finally:
            writer.release()
        frame_count = len(frames)
        del frames
        gc.collect()
        size_bytes = int(output_path.stat().st_size) if output_path.exists() else 0
        return int(frame_count), float(actual_fps), float(capture_elapsed), int(size_bytes)

    start = time.perf_counter()
    if session is not None:
        frame_count, actual_fps, size_bytes = _stream_mjpeg_from_started_picamera(
            session.picam2,
            fps=fps,
            duration=duration,
            width=width,
            height=height,
            output_path=output_path,
            monochrome_output=_camera_monochrome_output_enabled(probe_config),
        )
    else:
        with _PicameraCaptureSession(probe_config) as temp_session:
            temp_session.start()
            frame_count, actual_fps, size_bytes = _stream_mjpeg_from_started_picamera(
                temp_session.picam2,
                fps=fps,
                duration=duration,
                width=width,
                height=height,
                output_path=output_path,
                monochrome_output=_camera_monochrome_output_enabled(probe_config),
            )
    capture_elapsed = time.perf_counter() - start
    return int(frame_count), float(actual_fps), float(capture_elapsed), int(size_bytes)


def record_live_test_clip(
    config: Dict[str, Any],
    *,
    recording_seconds: Optional[float] = None,
) -> LiveRecordingResult:
    test_config = deepcopy(config)
    test_config.setdefault("capture", {})

    if recording_seconds is not None:
        test_config["capture"]["recording_seconds"] = float(recording_seconds)

    requested_recording_seconds = float(test_config["capture"]["recording_seconds"])
    base_session_name, base_session_dir = _make_session_paths(test_config)
    session_name = f"{base_session_name}_fps_test"
    session_dir = base_session_dir.parent / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        if base_session_dir != session_dir and base_session_dir.exists() and not any(base_session_dir.iterdir()):
            base_session_dir.rmdir()
    except Exception:
        pass

    frames, timestamps, actual_fps, timestamp_records = _capture_frames(test_config)
    timestamp_path: Optional[Path] = None
    recording_preview_png_path: Optional[Path] = None

    if timestamps:
        timestamp_path = _write_timestamps(
            timestamps,
            session_dir,
            session_name,
            timestamp_records=timestamp_records,
        )

    requested_video_codec = _normalize_recording_codec(test_config)
    requested_mp4_codec = (
        _normalize_ffmpeg_mp4_codec(str(test_config["camera"].get("mp4_codec", "libx264")))
        if requested_video_codec == "mp4"
        else None
    )
    video_codec = requested_video_codec
    video_write_warning: Optional[str] = None
    video_path = _write_recording_video(
        frames,
        session_dir,
        session_name,
        fps=float(actual_fps if actual_fps > 0 else test_config["camera"]["fps_target"]),
        width=int(test_config["camera"]["width"]),
        height=int(test_config["camera"]["height"]),
        recording_codec=requested_video_codec,
        mp4_codec=str(test_config["camera"].get("mp4_codec", "libx264")),
        monochrome_output=_camera_monochrome_output_enabled(test_config),
    )

    if not _video_file_is_readable(video_path):
        primary_size = int(video_path.stat().st_size) if video_path.exists() else 0
        if requested_video_codec == "mp4":
            fallback_session_name = f"{session_name}_fallback_mjpeg"
            fallback_path = _write_recording_video(
                frames,
                session_dir,
                fallback_session_name,
                fps=float(actual_fps if actual_fps > 0 else test_config["camera"]["fps_target"]),
                width=int(test_config["camera"]["width"]),
                height=int(test_config["camera"]["height"]),
                recording_codec="mjpeg",
                mp4_codec=str(test_config["camera"].get("mp4_codec", "libx264")),
                monochrome_output=_camera_monochrome_output_enabled(test_config),
            )
            if _video_file_is_readable(fallback_path):
                video_codec = "mjpeg"
                video_path = fallback_path
                video_write_warning = (
                    "Configured MP4 live-test output was unreadable "
                    f"({primary_size} bytes). MJPEG fallback clip was written instead."
                )
            else:
                fallback_size = int(fallback_path.stat().st_size) if fallback_path.exists() else 0
                video_write_warning = (
                    "Configured MP4 live-test output was unreadable "
                    f"({primary_size} bytes), and MJPEG fallback was also unreadable ({fallback_size} bytes). "
                    "FPS estimates can still be derived from timestamps."
                )
        else:
            video_write_warning = (
                f"Configured {requested_video_codec.upper()} live-test output was unreadable "
                f"({primary_size} bytes). FPS estimates can still be derived from timestamps."
            )

    preview_base_name = Path(video_path).stem
    recording_preview_png_path = _write_midpoint_preview_png(
        frames,
        session_dir,
        preview_base_name,
        monochrome_output=_camera_monochrome_output_enabled(test_config),
    )

    if requested_video_codec == "mp4":
        sidecar = session_dir / f"{session_name}_actual_fps.txt"
        sidecar.write_text(f"{actual_fps:.6f}\n")

    frame_count = len(frames)
    if frame_count == 0:
        del frames
        raise RuntimeError(
            "Live FPS test captured zero frames. The camera started, but no frames were returned before the test window ended."
        )
    del frames

    return LiveRecordingResult(
        session_name=session_name,
        session_dir=str(session_dir),
        requested_video_codec=requested_video_codec,
        requested_mp4_codec=requested_mp4_codec,
        video_codec=video_codec,
        video_path=str(video_path),
        video_size_bytes=int(video_path.stat().st_size) if video_path.exists() else 0,
        timestamp_path=str(timestamp_path) if timestamp_path else None,
        recording_preview_png_path=(
            str(recording_preview_png_path) if recording_preview_png_path else None
        ),
        frames_captured=frame_count,
        actual_fps=round(actual_fps, 6),
        configured_fps_target=float(test_config["camera"]["fps_target"]),
        requested_recording_seconds=requested_recording_seconds,
        video_write_warning=video_write_warning,
    )


def _write_recording_video(
    frames: List[Any],
    session_dir: Path,
    session_name: str,
    fps: float,
    width: int,
    height: int,
    recording_codec: str,
    mp4_codec: str,
    monochrome_output: bool = False,
    mp4_preset: str = "auto",
) -> Path:
    actual_width = int(width)
    actual_height = int(height)
    if frames:
        first_bgr = _frame_to_bgr(frames[0], monochrome_output=monochrome_output)
        actual_height = int(first_bgr.shape[0])
        actual_width = int(first_bgr.shape[1])

    codec_name = str(recording_codec).strip().lower()
    if codec_name == "mjpeg":
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency/runtime
            raise RuntimeError("OpenCV is required to write MJPEG recording output.") from exc

        # AVI retains the measured frame rate; a raw .mjpeg stream does not.
        output = session_dir / f"{session_name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (actual_width, actual_height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output}")

        progress = _FrameProgress(
            stage="output",
            label="RGB MJPEG video",
            total=len(frames),
        )
        progress.start()
        try:
            for frame_index, frame in enumerate(frames, start=1):
                writer.write(_frame_to_bgr(frame, monochrome_output=monochrome_output))
                progress.update(frame_index)
        finally:
            writer.release()
        progress.finish(len(frames))
        return output

    output = session_dir / f"{session_name}.mp4"
    return _write_mp4_video_with_ffmpeg(
        frames=frames,
        output=output,
        fps=fps,
        width=actual_width,
        height=actual_height,
        mp4_codec=mp4_codec,
        monochrome_output=monochrome_output,
        mp4_preset=mp4_preset,
    )


def _normalize_ffmpeg_mp4_codec(mp4_codec: str) -> str:
    return normalize_mp4_codec(mp4_codec)


def _find_ffmpeg_binary() -> Optional[str]:
    return find_ffmpeg_binary()


def _write_mp4_video_with_ffmpeg(
    frames: List[Any],
    output: Path,
    fps: float,
    width: int,
    height: int,
    mp4_codec: str,
    monochrome_output: bool = False,
    mp4_preset: str = "auto",
) -> Path:
    import numpy as np

    ffmpeg_bin = _find_ffmpeg_binary()
    if not ffmpeg_bin:
        raise RuntimeError(
            "MP4 recording requires ffmpeg, but it was not found on PATH. "
            "Install ffmpeg or switch camera.codec to 'mjpeg'."
        )

    ffmpeg_codec = _normalize_ffmpeg_mp4_codec(mp4_codec)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        # The same hand-over as stream mode: the camera's frame for colour, its
        # brightness plane for grayscale, with no RGB step in between.
        GRAY_INPUT_FORMAT if monochrome_output else COLOUR_INPUT_FORMAT,
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        ffmpeg_codec,
        *full_range_output_format(ffmpeg_codec),
    ]
    # Frames are already in memory here, so "auto" keeps the slower, smaller "fast" preset.
    preset = resolve_mp4_preset(mp4_preset, streaming=False)
    if ffmpeg_codec == "libx264":
        cmd.extend(["-preset", preset, "-crf", "18"])
    elif ffmpeg_codec == "libx265":
        cmd.extend(["-preset", preset, "-crf", "20"])
    elif ffmpeg_codec == "mpeg4":
        cmd.extend(["-q:v", "2"])
    cmd.extend(["-movflags", "+faststart", str(output)])

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    stderr_text = ""
    progress = _FrameProgress(
        stage="output",
        label=f"RGB MP4 video ({ffmpeg_codec})",
        total=len(frames),
    )
    progress.start()
    try:
        if process.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for MP4 encoding.")
        for frame_index, frame in enumerate(frames, start=1):
            if monochrome_output:
                process.stdin.write(_frame_luma(frame).tobytes())
            else:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            progress.update(frame_index)
    except BrokenPipeError as exc:
        stderr_bytes = b""
        if process.stderr is not None:
            stderr_bytes = process.stderr.read()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg stopped while encoding MP4 '{output.name}': {stderr_text or 'broken pipe'}"
        ) from exc
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass

    if process.stderr is not None:
        stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed while encoding MP4 '{output.name}' with codec "
            f"'{ffmpeg_codec}' (from camera.mp4_codec='{mp4_codec}'). "
            f"Details: {stderr_text or 'no stderr output'}"
        )
    if not _video_file_is_readable(output):
        size_bytes = int(output.stat().st_size) if output.exists() else 0
        raise RuntimeError(
            f"ffmpeg finished but MP4 output '{output.name}' is unreadable ({size_bytes} bytes)."
        )
    progress.finish(len(frames))
    return output


def _frame_to_bgr(frame: Any, monochrome_output: bool = False) -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("OpenCV is required for frame conversion.") from exc

    if monochrome_output:
        # Grayscale is the camera's own brightness plane, untouched. Going
        # through the colour conversion below instead would clip it: that
        # conversion treats brightness 16 as black and 235 as white, while the
        # camera uses the whole 0-255 range (see docs/Frame_Processing.md).
        return cv2.cvtColor(_frame_luma(frame), cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(_full_range_to_video_levels(frame), cv2.COLOR_YUV2BGR_I420)


def _video_level_tables() -> Tuple[Any, Any]:
    """Lookup tables that squeeze full-range brightness and colour values into video levels."""
    import numpy as np

    values = np.arange(256, dtype=np.float64)
    luma = np.rint(16.0 + values * 219.0 / 255.0).astype(np.uint8)
    chroma = np.rint(128.0 + (values - 128.0) * 224.0 / 255.0).astype(np.uint8)
    return luma, chroma


_VIDEO_LEVEL_TABLES: Optional[Tuple[Any, Any]] = None


def _full_range_to_video_levels(frame: Any) -> Any:
    """Prepare a camera frame for OpenCV's YUV-to-BGR conversion.

    OpenCV's conversion expects video levels (brightness 16-235) and stretches
    them to 0-255. The camera delivers 0-255 already, so converting its frames
    directly clips the darkest and brightest detail. Squeezing the frame into
    video levels first makes the stretch land where it started.
    """
    import cv2
    import numpy as np

    global _VIDEO_LEVEL_TABLES
    if _VIDEO_LEVEL_TABLES is None:
        _VIDEO_LEVEL_TABLES = _video_level_tables()
    luma, chroma = _VIDEO_LEVEL_TABLES
    array = np.ascontiguousarray(frame)
    height = int(array.shape[0]) * 2 // 3
    adjusted = np.empty_like(array)
    adjusted[:height] = cv2.LUT(array[:height], luma)
    adjusted[height:] = cv2.LUT(array[height:], chroma)
    return adjusted


def _frame_luma(frame: Any) -> Any:
    """The brightness image of a capture frame: what a grayscale view or recording shows.

    A YUV420 capture frame is the brightness (Y) plane followed by the two
    colour planes at half size, so the first two thirds of its rows are the
    brightness image. A BGR frame is reduced to brightness the usual way.
    """
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 3:
        import cv2

        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(array[: int(array.shape[0]) * 2 // 3])


def _capture_frame_to_bgr(frame: Any, monochrome_output: bool = False) -> Any:
    """BGR image for an RGB-camera frame.

    Accepts a raw YUV420 capture frame (RAM mode) or a BGR frame that was read
    back from the saved video (stream mode, see ``_VideoFrameSequence``).
    """
    if getattr(frame, "ndim", 2) == 3:
        return frame
    return _frame_to_bgr(frame, monochrome_output=monochrome_output)


class _VideoFrameSequence:
    """List-like, read-only view of a saved video's frames, decoded on demand.

    Stream mode does not keep RGB frames in memory, so outputs that need them
    after capture read them back from the recording one at a time. Access is
    cheapest in increasing index order.
    """

    def __init__(self, video_path: Any, frame_count: int) -> None:
        # One file, or the consecutive files of a recording that was split.
        paths = [video_path] if isinstance(video_path, (str, Path)) else list(video_path)
        self.video_paths = [Path(path) for path in paths]
        self.video_path = self.video_paths[0]
        self._frame_count = max(0, int(frame_count))
        self._capture: Any = None
        self._file_index = 0
        self._next_index = 0

    def __len__(self) -> int:
        return self._frame_count

    def __getitem__(self, index: int) -> Any:
        import cv2

        index = int(index)
        if index < 0:
            index += self._frame_count
        if index < 0 or index >= self._frame_count:
            raise IndexError(index)
        if self._capture is None or index < self._next_index:
            self.close()
            self._capture = cv2.VideoCapture(str(self.video_paths[0]))
        frame = None
        while self._next_index <= index:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                # This file is finished; carry on in the next one, if any.
                if self._file_index + 1 < len(self.video_paths):
                    self._capture.release()
                    self._file_index += 1
                    self._capture = cv2.VideoCapture(str(self.video_paths[self._file_index]))
                    continue
                raise IndexError(
                    f"{self.video_paths[self._file_index].name} ended at frame {self._next_index}; "
                    f"frame {index} was requested."
                )
            self._next_index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        self._file_index = 0
        self._next_index = 0


def _video_file_is_readable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        import cv2
    except ImportError:
        return path.stat().st_size >= 1024

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            return True
        ok, _frame = capture.read()
        return bool(ok)
    finally:
        capture.release()


def _write_midpoint_preview_png(
    frames: List[Any],
    session_dir: Path,
    session_name: str,
    monochrome_output: bool = False,
) -> Optional[Path]:
    if not frames:
        return None
    try:
        import cv2
    except ImportError:
        return None

    mid_idx = max(0, min(len(frames) - 1, len(frames) // 2))
    frame = frames[mid_idx]
    bgr = _frame_to_bgr(frame, monochrome_output=monochrome_output)
    out = session_dir / f"{session_name}_midframe.png"
    ok = cv2.imwrite(str(out), bgr)
    if not ok:
        return None
    return out


def _track_from_ram(
    config: Dict[str, Any],
    session_name: str,
    session_dir: Path,
    frames: List[Any],
) -> Tuple[Any, Any]:
    try:
        from tag_tracking_utils import trackTagsFromRAM, trackTagsFromRAM_parallel
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import tag_tracking_utils for RAM tracking.") from exc

    wrapped_frames = [[frame] for frame in frames]
    box_preset = _normalize_box_preset(config["tracking"].get("box_preset"))
    now_str = _now_iso()
    colony_id = config["system"]["colony_id"]
    aruco_params = config["tracking"].get("aruco_params")
    parallel = bool(config["pipeline"].get("parallel_tracking", False))

    if parallel:
        df, df2, _ = trackTagsFromRAM_parallel(
            session_name,
            str(session_dir),
            wrapped_frames,
            config["tracking"]["tag_dictionary"],
            box_preset,
            now_str,
            colony_id,
            aruco_params=aruco_params,
        )
    else:
        df, df2, _ = trackTagsFromRAM(
            session_name,
            str(session_dir),
            wrapped_frames,
            config["tracking"]["tag_dictionary"],
            box_preset,
            now_str,
            socket.gethostname(),
            colony_id,
            aruco_params=aruco_params,
        )
    return df, df2


def _track_from_video(
    config: Dict[str, Any],
    session_name: str,
    session_dir: Path,
    video_path: Path,
    video_parts: Optional[List[Path]] = None,
) -> Tuple[Any, Any]:
    try:
        from tag_tracking_utils import trackTagsFromVid
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import tag_tracking_utils for video tracking.") from exc

    box_preset = _normalize_box_preset(config["tracking"].get("box_preset"))
    now_str = _now_iso()
    colony_id = config["system"]["colony_id"]
    aruco_params = config["tracking"].get("aruco_params")

    parts = [Path(path) for path in (video_parts or [])]
    if len(parts) <= 1:
        df, df2, _ = trackTagsFromVid(
            str(video_path),
            str(session_dir),
            session_name,
            config["tracking"]["tag_dictionary"],
            box_preset,
            now_str,
            colony_id,
            aruco_params=aruco_params,
        )
        return df, df2

    # A split recording: track each file, then join the results with frame
    # numbers that run through the whole recording.
    import pandas as pd

    tracked, untagged = [], []
    first_frame = 0
    for number, part in enumerate(parts, start=1):
        _progress_message("tracking", f"Tracking video file {number} of {len(parts)}: {part.name}.")
        part_name = f"{session_name}_part{number:03d}"
        part_df, part_df2, part_frames = trackTagsFromVid(
            str(part),
            str(session_dir),
            part_name,
            config["tracking"]["tag_dictionary"],
            box_preset,
            now_str,
            colony_id,
            aruco_params=aruco_params,
        )
        for frame_table in (part_df, part_df2):
            frame_table["frame"] = frame_table["frame"] + first_frame
            frame_table["filename"] = session_name
        tracked.append(part_df)
        untagged.append(part_df2)
        first_frame += int(part_frames)
        for leftover in (session_dir / f"{part_name}_raw.csv", session_dir / f"{part_name}_noID.csv"):
            leftover.unlink(missing_ok=True)
    df = pd.concat(tracked, ignore_index=True)
    df2 = pd.concat(untagged, ignore_index=True)
    df.to_csv(session_dir / f"{session_name}_raw.csv", index=False)
    df2.to_csv(session_dir / f"{session_name}_noID.csv", index=False)
    return df, df2


def _run_cleaning(config: Dict[str, Any], df: Any, actual_fps: float) -> Any:
    try:
        import data_cleaning
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import data_cleaning module.") from exc

    cleaning = config["cleaning"]
    out = df
    if bool(cleaning.get("remove_jumps", False)) and not out.empty:
        out = data_cleaning.remove_jumps(out)
    if bool(cleaning.get("interpolate_data", False)) and not out.empty:
        out = data_cleaning.interpolate(out, float(cleaning.get("max_seconds_gap", 3)), float(actual_fps))
    if bool(cleaning.get("compute_heading_angle", False)) and not out.empty:
        out = data_cleaning.compute_heading_angle(out)
    return out


def _excluded_tracking_tag_ids(config: Dict[str, Any]) -> set[int]:
    raw_ids = config.get("tracking", {}).get("excluded_tag_ids", [])
    if not isinstance(raw_ids, list):
        return set()
    out: set[int] = set()
    for raw_id in raw_ids:
        try:
            out.add(int(raw_id))
        except (TypeError, ValueError):
            continue
    return out


def _apply_tracking_id_exclusions(df: Any, excluded_ids: set[int]) -> tuple[Any, int]:
    if not excluded_ids:
        return df, 0
    if getattr(df, "empty", True):
        return df, 0
    if "ID" not in getattr(df, "columns", []):
        return df, 0

    import pandas as pd

    work = df.copy()
    numeric_ids = pd.to_numeric(work["ID"], errors="coerce")
    keep_mask = ~numeric_ids.isin(list(excluded_ids))
    filtered = work.loc[keep_mask].copy()
    removed = int(len(work) - len(filtered))
    return filtered, removed


def _run_metrics(config: Dict[str, Any], df: Any, actual_fps: float, session_dir: Path, session_name: str, warnings: List[str]) -> None:
    if not bool(config["pipeline"].get("calculate_behavior_metrics", False)):
        return
    if df.empty:
        return

    try:
        import behavioral_metrics as bm
    except Exception as exc:  # pragma: no cover - dependency/runtime
        warnings.append(f"Metrics skipped: could not import behavioral_metrics ({exc})")
        return

    enabled = {str(name).strip().lower() for name in config["metrics"].get("enabled", [])}
    moving_threshold = float(config["metrics"].get("moving_threshold_px_per_frame", 3.16))
    speed_cutoff_seconds = int(config["runtime"].get("speed_cutoff_seconds", 4))
    pixel_contact_distance = float(config["metrics"].get("pixel_contact_distance", 0))

    try:
        if "speed" in enabled:
            df = bm.compute_speed(df, actual_fps, speed_cutoff_seconds, moving_threshold, str(session_dir), session_name)
        if "activity" in enabled:
            df = bm.compute_activity(df, actual_fps, speed_cutoff_seconds, moving_threshold, str(session_dir), session_name)
        if "distance_from_center" in enabled:
            df = bm.compute_social_center_distance(df, str(session_dir), session_name)

        pw_df = None
        if "pairwise_distance" in enabled or "contacts" in enabled:
            pw_df = bm.pairwise_distance(df, str(session_dir), session_name)

        if "contacts" in enabled and pw_df is not None:
            bm.contact_matrix(pw_df, str(session_dir), pixel_contact_distance, session_name)

        if "video_averages" in enabled:
            bm.compute_video_averages(df, str(session_dir), session_name)
    except Exception as exc:
        warnings.append(f"Metric calculation encountered an error: {exc}")


def _run_fps_report_if_needed(
    config: Dict[str, Any],
    session_dir: Path,
    session_name: str,
    video_path: Optional[Path],
    timestamp_path: Optional[Path],
) -> Optional[Path]:
    if video_path is None:
        return None
    if video_path.suffix.lower() != ".mp4":
        return None
    if not bool(config["runtime"].get("fps_report_on_each_recording", False)):
        return None

    try:
        from .fps_report import build_fps_report, write_report_json
    except Exception:
        return None

    try:
        report = build_fps_report(
            video_path=video_path,
            timestamps_path=timestamp_path,
            recording_seconds=float(config["capture"]["recording_seconds"]),
        )
        out = session_dir / f"{session_name}_fps_report.json"
        write_report_json(report, out)
        return out
    except Exception:
        return None


def run_once(
    config: Dict[str, Any],
    mode_override: Optional[str] = None,
    *,
    session_name_override: Optional[str] = None,
    session_dir_override: Optional[str | Path] = None,
    capture_start_unix_ns: Optional[int] = None,
    capture_ready_callback: Optional[Any] = None,
    sensor_names_override: Optional[List[str]] = None,
    capture_node_name: Optional[str] = None,
) -> RunSummary:
    config = apply_camera_profile(config)
    if sensor_names_override is not None:
        assigned_sensors = {
            str(sensor).strip().lower() for sensor in sensor_names_override
        }
        unknown_sensors = assigned_sensors - {"rgb", "thermal", "realsense"}
        if unknown_sensors:
            raise ValueError(
                "Unknown capture sensor assignment(s): "
                + ", ".join(sorted(unknown_sensors))
            )
        if "rgb" not in assigned_sensors:
            raise ValueError(
                "run_once requires the RGB sensor. Sensor-only distributed nodes use "
                "the capture-node runner instead."
            )
        config.setdefault("thermal", {})["enabled"] = "thermal" in assigned_sensors
        config.setdefault("realsense", {})["enabled"] = "realsense" in assigned_sensors
    ir_error = validate_camera_ir_compatibility(config)
    if ir_error:
        raise ValueError(ir_error)

    started_at = _now_iso()
    warnings: List[str] = []
    errors: List[str] = []

    _guard_fleet_queen_local_pipeline(config)

    configured_mode = str(config["pipeline"]["mode"])
    mode = mode_override or configured_mode

    if configured_mode == "mixed_schedule" and mode_override is None:
        raise ValueError(
            "pipeline.mode is mixed_schedule. Use mode override 'record_only' or 'track_only' when running run-once."
        )

    if mode not in {"record_only", "track_only", "record_and_track"}:
        raise ValueError(f"Unsupported run mode: {mode}")

    # Refuse to start before anything is written if the data drive is missing
    # or nearly full, so a run can never quietly fill the system disk instead.
    ensure_storage_ready(config, path=session_dir_override)

    if session_name_override is None and session_dir_override is None:
        session_name, session_dir = _make_session_paths(config)
    else:
        session_name = str(session_name_override or "").strip()
        if not session_name:
            raise ValueError("session_name_override must be set with session_dir_override")
        if session_dir_override is None:
            data_root = Path(config["system"]["data_root"])
            session_dir = data_root / datetime.now().strftime("%Y-%m-%d") / session_name
        else:
            session_dir = Path(session_dir_override).expanduser()
        session_dir.mkdir(parents=True, exist_ok=True)

    capture_start_monotonic: Optional[float] = None
    if capture_start_unix_ns is not None:
        remaining_seconds = (int(capture_start_unix_ns) - time.time_ns()) / 1_000_000_000.0
        capture_start_monotonic = time.perf_counter() + max(0.0, remaining_seconds)
    hostname = socket.gethostname()
    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera", {}), dict) else {}
    camera_profile = configured_profile_name(config)
    camera_model = str(camera_cfg.get("model", "auto"))
    raw_camera_infrared = camera_cfg.get("infrared")
    camera_infrared = raw_camera_infrared if isinstance(raw_camera_infrared, bool) else None
    raw_camera_monochrome_output = camera_cfg.get("monochrome_output")
    camera_monochrome_output = (
        raw_camera_monochrome_output if isinstance(raw_camera_monochrome_output, bool) else None
    )
    focus_settings = autofocus_settings(config)
    preflight_settings = autofocus_preflight_settings(config)
    resolved_tuning_file = resolve_camera_tuning_file(config)
    capture_diagnostics: dict[str, Any] = {}

    frames: List[Any] = []
    timestamps: List[float] = []
    timestamp_records: List[FrameTimestampRecord] = []
    actual_fps = 0.0
    # Stream mode keeps no RGB frames in memory, so count frames by their
    # timestamps rather than len(frames).
    frame_count = 0
    capture_frame_buffer = "ram"
    stream_writer: Optional[StreamingVideoWriter] = None
    streamed_video: Optional[StreamedVideo] = None
    stream_encoder_stats: Optional[Dict[str, Any]] = None
    tracking_elapsed_seconds: Optional[float] = None
    tracking_frames_processed = 0
    tracking_processing_fps: Optional[float] = None
    video_codec: Optional[str] = None
    video_path: Optional[Path] = None
    video_parts: List[Path] = []
    recording_preview_png_path: Optional[Path] = None
    timestamp_path: Optional[Path] = None
    thermal_capture: Optional[dict[str, Any]] = None
    thermal_artifacts: Optional[ThermalRecordingArtifacts] = None
    thermal_side_by_side_artifacts: Optional[ThermalSideBySideArtifacts] = None
    realsense_artifacts: Optional[RealSenseRecordingResult] = None
    tracking_visualization_artifacts: Optional[TrackingVisualizationArtifacts] = None
    raw_csv: Optional[Path] = None
    noid_csv: Optional[Path] = None
    cleaned_csv: Optional[Path] = None
    fps_report_json: Optional[Path] = None
    config_snapshot_path: Optional[Path] = None
    # Held only while the camera is in use, so a later run can start capturing
    # while this one is still encoding or tracking.
    camera_lock = ExitStack()

    try:
        should_record = mode in {"record_only", "record_and_track"}
        should_track = mode in {"track_only", "record_and_track"}
        _progress_message(
            "run",
            f"Session {session_name} started in {mode} mode.",
        )
        # A run that is killed, or loses power, never reaches the code that
        # writes its summary, and would then be missing from every list of
        # runs. So a summary saying "did not finish" is written now and
        # replaced by the real one at the end.
        unfinished: Dict[str, Any] = {
            "started_at": started_at,
            "finished_at": "",
            "mode": mode,
            "session_name": session_name,
            "session_dir": str(session_dir),
            "hostname": socket.gethostname(),
            "frames_captured": 0,
            "actual_fps": 0.0,
            "video_path": None,
            "video_parts": [],
            "capture_frame_buffer": frame_buffer_mode(config),
            "thermal_enabled": False,
            "realsense_enabled": False,
            "camera_num": configured_camera_num(config),
            "camera_id": configured_camera_id(config),
            "camera_session_tag": session_tag(config),
            "warnings": [],
            "errors": [UNFINISHED_RUN_ERROR],
            "success": False,
            "unfinished": True,
        }
        unfinished_path = session_dir / f"{session_name}_run_summary.json"

        def _save_unfinished() -> None:
            try:
                unfinished_path.write_text(json.dumps(unfinished, indent=2))
            except OSError:
                pass  # the real summary is written at the end regardless

        _save_unfinished()

        if should_record and frame_buffer_mode(config) == FRAME_BUFFER_STREAM:
            capture_frame_buffer = FRAME_BUFFER_STREAM
            expected_frames = max(
                1,
                int(round(float(config["capture"]["recording_seconds"]) * float(config["camera"]["fps_target"]))),
            )
            # A long recording is written as consecutive files, so a failure
            # late in the session costs one file rather than all of it. A
            # recording that fits in one file keeps its usual name.
            chunk_frames = stream_chunk_frames(config)
            if expected_frames <= chunk_frames:
                chunk_frames = 0

            def _part_saved(chunk: Any) -> None:
                # Runs on the writer's background thread as each file closes:
                # its timestamps go to disk now, not at the end of the session.
                _write_timestamps(
                    [float(getattr(record, "time_s", 0.0)) for record in chunk.records],
                    session_dir,
                    f"{session_name}_part{chunk.index:03d}",
                    timestamp_records=list(chunk.records),
                    first_frame=int(chunk.first_frame),
                )
                # Keep the "did not finish" summary current, so a run that is
                # cut short still lists the files it completed.
                unfinished["video_parts"].append(str(chunk.video_path))
                unfinished["video_path"] = unfinished["video_parts"][0]
                unfinished["frames_captured"] = int(chunk.first_frame) + int(chunk.frame_count)
                unfinished["errors"] = [
                    f"{UNFINISHED_RUN_ERROR} {len(unfinished['video_parts'])} video file(s) with "
                    f"{unfinished['frames_captured']} frames were saved before that."
                ]
                _save_unfinished()
                _progress_message(
                    "output",
                    f"Video file {chunk.index} saved: {Path(chunk.video_path).name} "
                    f"({chunk.frame_count} frames from frame {chunk.first_frame}).",
                )

            stream_writer = StreamingVideoWriter(
                output_dir=session_dir,
                stem=session_name,
                fps=float(config["camera"]["fps_target"]),
                recording_codec=_normalize_recording_codec(config),
                mp4_codec=str(config["camera"].get("mp4_codec", "libx264")),
                mp4_preset=str(config["camera"].get("mp4_preset", "auto")),
                monochrome_output=_camera_monochrome_output_enabled(config),
                max_queue_bytes=stream_queue_bytes(config),
                preview_frame_index=expected_frames // 2,
                chunk_frames=chunk_frames,
                on_chunk_closed=_part_saved,
            )
            if chunk_frames:
                _progress_message(
                    "capture",
                    f"The recording is split into a new video file every {chunk_frames} frames "
                    f"(about {describe_seconds(round(chunk_frames / float(config['camera']['fps_target'])))}).",
                )
            _progress_message(
                "capture",
                "Stream mode: frames are encoded to disk during capture "
                f"({stream_writer.encoder}{' ' + stream_writer.preset if stream_writer.preset else ''}, "
                f"up to {stream_writer.stats.queue_budget_mb:.0f} MB of frames waiting).",
            )

        if should_record or should_track:
            # Only one run may use the camera at a time. Wait briefly for a
            # run that is already capturing, then give up with a clear error.
            camera_lock.enter_context(
                capture_lock(
                    float(config["runtime"].get("capture_lock_wait_seconds", DEFAULT_LOCK_WAIT_SECONDS)),
                    # One lock per camera: two cameras on one Pi may record at
                    # the same time, two runs on the same camera may not.
                    lock_path=default_lock_path(capture_lock_suffix(config)),
                    on_wait=lambda holder: _progress_message(
                        "capture",
                        f"Waiting for another BumbleBox capture to release the camera ({holder}).",
                    ),
                )
            )
            optional_sensor_recording = (
                _thermal_enabled_for_recording(config)
                or _realsense_enabled_for_recording(config)
            )
            if should_record and (
                optional_sensor_recording or capture_start_monotonic is not None
            ):
                if _thermal_enabled_for_recording(config):
                    thermal_cfg = config.get("thermal", {})
                    try:
                        thermal_target_fps = float(
                            thermal_cfg.get("fps_target", config["camera"]["fps_target"])
                        )
                    except Exception:
                        thermal_target_fps = float(config["camera"]["fps_target"])
                    if abs(thermal_target_fps - float(config["camera"]["fps_target"])) > 1e-6:
                        warnings.append(
                            "Synchronized thermal recording follows camera.fps_target. "
                            f"thermal.fps_target={thermal_target_fps:.3f} was ignored for this run."
                        )
                (
                    frames,
                    timestamps,
                    actual_fps,
                    timestamp_records,
                    thermal_capture,
                    realsense_artifacts,
                ) = _capture_rgb_and_optional_sensors(
                    config,
                    session_dir=session_dir,
                    session_name=session_name,
                    diagnostics=capture_diagnostics,
                    start_monotonic_override=capture_start_monotonic,
                    ready_callback=capture_ready_callback,
                    capture_node_name=capture_node_name,
                    frame_sink=stream_writer,
                )
                frame_count = len(timestamps)
                # Capture is over: tell the watchdog, so that slow work from
                # here on (encoding, tracking) is not mistaken for a stall.
                heartbeat.capture_finished()
                if thermal_capture is not None:
                    thermal_artifacts = _write_thermal_recording_outputs(
                        session_dir=session_dir,
                        session_name=session_name,
                        device_path=str(thermal_capture["device_path"]),
                        frames=list(thermal_capture["frames"]),
                        timestamps=list(thermal_capture["timestamps"]),
                        actual_fps=float(thermal_capture["actual_fps"]),
                        raw16_layout=str(thermal_capture["raw16_layout"]),
                        timestamp_records=list(thermal_capture.get("timestamp_records") or []),
                        node_name=capture_node_name,
                    )
                    if thermal_artifacts.frames_captured != frame_count:
                        warnings.append(
                            "RGB and thermal synchronized recording captured different frame counts "
                            f"({frame_count} RGB vs {thermal_artifacts.frames_captured} thermal). "
                            "Use the separate timestamp CSV files for alignment."
                        )
                    if abs(float(actual_fps) - float(thermal_artifacts.actual_fps)) > 0.25:
                        warnings.append(
                            "RGB and thermal actual FPS differed during synchronized recording "
                            f"({actual_fps:.3f} RGB vs {thermal_artifacts.actual_fps:.3f} thermal)."
                        )
                else:
                    thermal_artifacts = None
                    thermal_side_by_side_artifacts = None
            else:
                _progress_message("capture", "Preparing RGB capture.")
                frames, timestamps, actual_fps, timestamp_records = _capture_frames(
                    config,
                    diagnostics=capture_diagnostics,
                    progress_label="RGB capture",
                    frame_sink=stream_writer,
                )
                frame_count = len(timestamps)
                # Capture is over: tell the watchdog, so that slow work from
                # here on (encoding, tracking) is not mistaken for a stall.
                heartbeat.capture_finished()
                _progress_message(
                    "capture",
                    f"RGB acquisition finished: {frame_count} frames at {actual_fps:.3f} FPS.",
                )
                if should_track and _thermal_enabled_for_recording(config) and not should_record:
                    warnings.append("thermal.enabled is set, but thermal capture runs only during recording modes.")
                if should_track and _realsense_enabled_for_recording(config) and not should_record:
                    warnings.append("realsense.enabled is set, but RealSense capture runs only during recording modes.")
            camera_lock.close()

        if bool(config["runtime"].get("save_frame_timestamps", True)) and timestamps:
            _progress_message("output", "Writing RGB frame timestamps.")
            timestamp_path = _write_timestamps(
                timestamps,
                session_dir,
                session_name,
                timestamp_records=timestamp_records,
                node_name=capture_node_name,
            )
            _progress_message("output", "RGB frame timestamps are complete.")

        if should_record:
            video_codec = _normalize_recording_codec(config)
            if stream_writer is not None:
                # The video was encoded during capture; wait for the encoder to
                # finish and set the measured frame rate on the final file.
                _progress_message(
                    "output",
                    f"Finishing streamed {video_codec.upper()} recording.",
                )
                streamed_video = stream_writer.finish(
                    actual_fps=float(actual_fps if actual_fps > 0 else config["camera"]["fps_target"])
                )
                video_path = streamed_video.video_path
                if streamed_video.chunks:
                    video_parts = list(streamed_video.video_paths)
                stream_encoder_stats = streamed_video.stats.to_dict()
                warnings.extend(streamed_video.warnings)
                if streamed_video.frame_count != frame_count:
                    warnings.append(
                        f"Stream recording wrote {streamed_video.frame_count} frames but "
                        f"{frame_count} were captured."
                    )
            else:
                _progress_message(
                    "output",
                    f"Encoding RGB recording as {video_codec.upper()}.",
                )
                video_path = _write_recording_video(
                    frames,
                    session_dir,
                    session_name,
                    fps=float(actual_fps if actual_fps > 0 else config["camera"]["fps_target"]),
                    width=int(config["camera"]["width"]),
                    height=int(config["camera"]["height"]),
                    recording_codec=video_codec,
                    mp4_codec=str(config["camera"].get("mp4_codec", "libx264")),
                    monochrome_output=_camera_monochrome_output_enabled(config),
                    mp4_preset=str(config["camera"].get("mp4_preset", "auto")),
                )
            _progress_message("output", f"RGB recording complete: {video_path.name}.")

            if thermal_capture is not None:
                # Stream mode has no RGB frames in memory: read them back from the recording.
                side_by_side_rgb_frames: Any = (
                    _VideoFrameSequence(video_parts or video_path, frame_count)
                    if stream_writer is not None
                    else frames
                )
                try:
                    thermal_side_by_side_artifacts = _write_rgb_thermal_side_by_side_outputs(
                        session_dir=session_dir,
                        session_name=session_name,
                        rgb_frames=side_by_side_rgb_frames,
                        rgb_timestamps=timestamps,
                        thermal_frames=list(thermal_capture["frames"]),
                        thermal_timestamps=list(thermal_capture["timestamps"]),
                        fps=float(actual_fps if actual_fps > 0 else config["camera"]["fps_target"]),
                        monochrome_output=_camera_monochrome_output_enabled(config),
                    )
                except Exception as exc:
                    warnings.append(f"RGB+thermal side-by-side preview write failed: {exc}")
                finally:
                    if isinstance(side_by_side_rgb_frames, _VideoFrameSequence):
                        side_by_side_rgb_frames.close()

            _progress_message("output", "Writing RGB midpoint preview.")
            if streamed_video is not None:
                preview_frames = [streamed_video.preview_frame] if streamed_video.preview_frame is not None else []
            else:
                preview_frames = frames
            recording_preview_png_path = _write_midpoint_preview_png(
                preview_frames,
                session_dir,
                session_name,
                monochrome_output=_camera_monochrome_output_enabled(config),
            )
            if recording_preview_png_path is None:
                warnings.append("Could not write midpoint preview PNG.")
            else:
                _progress_message(
                    "output",
                    f"RGB midpoint preview complete: {recording_preview_png_path.name}.",
                )

            if video_codec == "mp4" and bool(config["runtime"].get("save_mp4_sidecar_fps_txt", True)):
                sidecar = session_dir / f"{session_name}_actual_fps.txt"
                sidecar.write_text(f"{actual_fps:.6f}\n")

        if should_track:
            source = str(config["pipeline"].get("tracking_source", "ram"))
            if not bool(config["pipeline"].get("defer_tracking_until_after_recording", True)):
                warnings.append("Non-deferred tracking is not implemented yet; using deferred tracking.")

            if stream_writer is not None and source != "video":
                _progress_message(
                    "tracking",
                    "Stream mode keeps no frames in memory, so tracking reads the saved video.",
                )
                source = "video"

            tracking_started = time.perf_counter()
            _progress_message(
                "tracking",
                f"Starting tag tracking from {source.upper()} input for {frame_count} captured frames.",
            )
            if source == "video":
                if video_path is None:
                    warnings.append("tracking_source=video requires recording; falling back to RAM tracking.")
                    df, df2 = _track_from_ram(config, session_name, session_dir, frames)
                else:
                    df, df2 = _track_from_video(config, session_name, session_dir, video_path, video_parts)
            else:
                df, df2 = _track_from_ram(config, session_name, session_dir, frames)
            tracking_elapsed_seconds = time.perf_counter() - tracking_started
            tracking_frames_processed = frame_count
            if tracking_elapsed_seconds > 0 and tracking_frames_processed > 0:
                tracking_processing_fps = tracking_frames_processed / tracking_elapsed_seconds
            _progress_message(
                "tracking",
                "Tag tracking complete: "
                f"{tracking_frames_processed} frames in {tracking_elapsed_seconds:.1f}s.",
            )

            _progress_message("tracking", "Writing tracking CSV outputs.")
            raw_csv = session_dir / f"{session_name}_raw.csv"
            noid_csv = session_dir / f"{session_name}_noID.csv"
            excluded_tag_ids = _excluded_tracking_tag_ids(config)
            df, removed_rows = _apply_tracking_id_exclusions(df, excluded_tag_ids)
            if removed_rows > 0:
                warnings.append(
                    "Excluded tracking IDs were removed from outputs: "
                    + ", ".join(str(tag_id) for tag_id in sorted(excluded_tag_ids))
                    + f" ({removed_rows} rows dropped)."
                )
            df.to_csv(raw_csv, index=False)
            df2.to_csv(noid_csv, index=False)
            _progress_message("tracking", "Raw and unidentified-tag CSV outputs are complete.")

            if not df.empty:
                _progress_message("tracking", "Cleaning tracks and calculating configured metrics.")
                df_clean = _run_cleaning(config, df, actual_fps if actual_fps > 0 else float(config["camera"]["fps_target"]))
                cleaned_csv = session_dir / f"{session_name}_cleaned.csv"
                df_clean.to_csv(cleaned_csv, index=False)
                _run_metrics(
                    config=config,
                    df=df_clean,
                    actual_fps=actual_fps if actual_fps > 0 else float(config["camera"]["fps_target"]),
                    session_dir=session_dir,
                    session_name=session_name,
                    warnings=warnings,
                )
                _progress_message("tracking", "Track cleaning and metrics are complete.")

            if bool(config["runtime"].get("render_tracking_video", False)):
                if video_path is None:
                    warnings.append("Tracked video visualization skipped because no recording video was available.")
                elif raw_csv is None or not raw_csv.exists():
                    warnings.append("Tracked video visualization skipped because tracking raw CSV was not found.")
                else:
                    try:
                        _progress_message("tracking", "Rendering tracked RGB visualization.")
                        tracking_visualization_artifacts = _render_tracking_visualizations(
                            config=config,
                            session_dir=session_dir,
                            session_name=session_name,
                            video_path=video_path,
                            raw_csv_path=raw_csv,
                            rgb_timestamps=timestamps,
                            thermal_frames=(
                                list(thermal_capture["frames"])
                                if thermal_capture is not None and "frames" in thermal_capture
                                else None
                            ),
                            thermal_timestamps=(
                                list(thermal_capture["timestamps"])
                                if thermal_capture is not None and "timestamps" in thermal_capture
                                else None
                            ),
                            fps=float(actual_fps if actual_fps > 0 else config["camera"]["fps_target"]),
                        )
                        _progress_message("tracking", "Tracked visualization outputs are complete.")
                    except Exception as exc:
                        warnings.append(f"Tracked video visualization failed: {exc}")

        _progress_message("output", "Writing run reports and configuration snapshot.")
        fps_report_json = _run_fps_report_if_needed(config, session_dir, session_name, video_path, timestamp_path)

        try:
            config_snapshot_path = session_dir / f"{session_name}_config_snapshot.json"
            config_snapshot_path.write_text(json.dumps(config, indent=2))
        except Exception as exc:
            warnings.append(f"Config snapshot write failed: {exc}")
        _progress_message(
            "output",
            "Recording and analysis outputs are complete; finalizing the run summary.",
        )

    except (Exception, KeyboardInterrupt) as exc:
        # KeyboardInterrupt is Ctrl+C, or Cancel in the GUI (which sends
        # SIGINT). Record it like a failure so the run still gets a summary
        # and a partial stream recording is closed cleanly.
        message = (
            "Cancelled before it finished (interrupted by the user)."
            if isinstance(exc, KeyboardInterrupt)
            else str(exc)
        )
        errors.append(message)
        _progress_message("error", message)
        if stream_writer is not None and streamed_video is None:
            # Capture or encoding failed part-way. Let the encoder close its
            # file cleanly so whatever was recorded so far stays readable.
            stream_writer.abort()
            kept = [chunk for chunk in stream_writer.chunks if chunk.frame_count > 0]
            if kept:
                # A split recording: every file written so far is complete.
                video_parts = [Path(chunk.video_path or chunk.capture_path) for chunk in kept]
                video_path = video_parts[0]
                warnings.append(
                    f"The recording stopped early; {len(kept)} video file(s) with "
                    f"{sum(chunk.frame_count for chunk in kept)} frames were kept."
                )
            else:
                partial = stream_writer.capture_path
                if partial.exists() and partial.stat().st_size > 0:
                    warnings.append(f"A partial stream recording was kept: {partial.name}")
    finally:
        camera_lock.close()
        if stream_writer is not None:
            stream_writer.abort()
        if thermal_capture is not None and "frames" in thermal_capture:
            try:
                del thermal_capture["frames"]
            except Exception:
                pass

    finished_at = _now_iso()
    raw_preflight_result = capture_diagnostics.get("autofocus_preflight")
    preflight_result = raw_preflight_result if isinstance(raw_preflight_result, dict) else {}
    raw_autofocus_notes = capture_diagnostics.get("autofocus_notes")
    autofocus_notes = (
        [str(note) for note in raw_autofocus_notes]
        if isinstance(raw_autofocus_notes, list)
        else []
    )
    summary = RunSummary(
        started_at=started_at,
        finished_at=finished_at,
        mode=mode,
        session_name=session_name,
        session_dir=str(session_dir),
        hostname=hostname,
        python_executable=str(Path(sys.executable).resolve()),
        camera_profile=camera_profile,
        camera_model=camera_model,
        camera_infrared=camera_infrared,
        camera_monochrome_output=camera_monochrome_output,
        camera_autofocus_mode=focus_settings.mode,
        camera_lens_position=focus_settings.lens_position,
        camera_focus_lock_after_warmup=focus_settings.lock_after_warmup,
        camera_autofocus_preflight_enabled=preflight_settings.enabled,
        camera_autofocus_preflight_performed=bool(preflight_result.get("performed", False)),
        camera_autofocus_preflight_lens_position=(
            float(preflight_result["selected_lens_position"])
            if preflight_result.get("selected_lens_position") is not None
            else None
        ),
        camera_autofocus_preflight_selection_reason=(
            str(preflight_result["selection_reason"])
            if preflight_result.get("selection_reason") is not None
            else None
        ),
        camera_autofocus_preflight_final_state=(
            str(preflight_result["final_af_state"])
            if preflight_result.get("final_af_state") is not None
            else None
        ),
        camera_autofocus_preflight_best_score=(
            float(preflight_result["best_focus_score"])
            if preflight_result.get("best_focus_score") is not None
            else None
        ),
        camera_autofocus_preflight_elapsed_seconds=(
            float(preflight_result["elapsed_seconds"])
            if preflight_result.get("elapsed_seconds") is not None
            else None
        ),
        camera_autofocus_preflight_notes=autofocus_notes,
        resolved_tuning_file=resolved_tuning_file,
        frames_captured=frame_count,
        actual_fps=round(actual_fps, 6),
        tracking_elapsed_seconds=round(tracking_elapsed_seconds, 6) if tracking_elapsed_seconds is not None else None,
        tracking_frames_processed=tracking_frames_processed,
        tracking_processing_fps=(
            round(tracking_processing_fps, 6) if tracking_processing_fps is not None else None
        ),
        video_codec=video_codec,
        video_path=str(video_path) if video_path else None,
        capture_frame_buffer=capture_frame_buffer,
        stream_encoder_stats=stream_encoder_stats,
        recording_preview_png_path=str(recording_preview_png_path) if recording_preview_png_path else None,
        timestamp_path=str(timestamp_path) if timestamp_path else None,
        thermal_enabled=bool(_thermal_enabled_for_recording(config)),
        thermal_device_path=(thermal_artifacts.device_path if thermal_artifacts else None),
        thermal_frames_captured=(thermal_artifacts.frames_captured if thermal_artifacts else 0),
        thermal_actual_fps=(round(thermal_artifacts.actual_fps, 6) if thermal_artifacts else None),
        thermal_timestamp_path=(str(thermal_artifacts.timestamp_path) if thermal_artifacts and thermal_artifacts.timestamp_path else None),
        thermal_raw_npy_path=(str(thermal_artifacts.raw_npy_path) if thermal_artifacts else None),
        thermal_preview_video_path=(str(thermal_artifacts.preview_video_path) if thermal_artifacts else None),
        thermal_preview_png_path=(str(thermal_artifacts.preview_png_path) if thermal_artifacts and thermal_artifacts.preview_png_path else None),
        thermal_metadata_json_path=(str(thermal_artifacts.metadata_json_path) if thermal_artifacts else None),
        thermal_side_by_side_video_path=(
            str(thermal_side_by_side_artifacts.video_path)
            if thermal_side_by_side_artifacts
            else None
        ),
        thermal_side_by_side_png_path=(
            str(thermal_side_by_side_artifacts.midpoint_png_path)
            if thermal_side_by_side_artifacts and thermal_side_by_side_artifacts.midpoint_png_path
            else None
        ),
        realsense_enabled=bool(_realsense_enabled_for_recording(config)),
        realsense_device_serial=(
            realsense_artifacts.selected_serial if realsense_artifacts else None
        ),
        realsense_frames_captured=(
            realsense_artifacts.frames_captured if realsense_artifacts else 0
        ),
        realsense_actual_fps=(
            round(realsense_artifacts.actual_fps, 6) if realsense_artifacts else None
        ),
        realsense_depth_scale_meters=(
            realsense_artifacts.depth_scale_meters if realsense_artifacts else None
        ),
        realsense_timestamp_path=(
            realsense_artifacts.timestamp_path if realsense_artifacts else None
        ),
        realsense_raw_depth_npy_path=(
            realsense_artifacts.raw_depth_npy_path if realsense_artifacts else None
        ),
        realsense_depth_preview_video_path=(
            realsense_artifacts.depth_preview_video_path if realsense_artifacts else None
        ),
        realsense_depth_preview_png_path=(
            realsense_artifacts.depth_preview_png_path if realsense_artifacts else None
        ),
        realsense_color_video_path=(
            realsense_artifacts.color_video_path if realsense_artifacts else None
        ),
        realsense_color_preview_png_path=(
            realsense_artifacts.color_preview_png_path if realsense_artifacts else None
        ),
        realsense_metadata_json_path=(
            realsense_artifacts.metadata_json_path if realsense_artifacts else None
        ),
        tracked_video_path=(
            str(tracking_visualization_artifacts.tracked_video_path)
            if tracking_visualization_artifacts
            else None
        ),
        tracked_thermal_side_by_side_video_path=(
            str(tracking_visualization_artifacts.tracked_thermal_side_by_side_video_path)
            if tracking_visualization_artifacts
            and tracking_visualization_artifacts.tracked_thermal_side_by_side_video_path
            else None
        ),
        tracked_thermal_side_by_side_png_path=(
            str(tracking_visualization_artifacts.tracked_thermal_side_by_side_png_path)
            if tracking_visualization_artifacts
            and tracking_visualization_artifacts.tracked_thermal_side_by_side_png_path
            else None
        ),
        raw_csv_path=str(raw_csv) if raw_csv else None,
        noid_csv_path=str(noid_csv) if noid_csv else None,
        cleaned_csv_path=str(cleaned_csv) if cleaned_csv else None,
        fps_report_json=str(fps_report_json) if fps_report_json else None,
        config_snapshot_path=str(config_snapshot_path) if config_snapshot_path else None,
        local_index_path=None,
        local_index_manifest_path=None,
        warnings=warnings,
        errors=errors,
        success=(len(errors) == 0),
        camera_num=configured_camera_num(config),
        camera_id=configured_camera_id(config),
        camera_session_tag=session_tag(config),
        video_parts=[str(path) for path in video_parts],
    )

    summary_path = Path(summary.session_dir) / f"{summary.session_name}_run_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    try:
        index_result = sync_run_summary_file(
            config,
            summary_path,
            summary_payload=asdict(summary),
        )
        if index_result.enabled:
            summary.local_index_path = index_result.index_root
            summary.local_index_manifest_path = index_result.manifest_path
            for warning in index_result.warnings:
                summary.warnings.append(f"Local tracking index warning: {warning}")
            summary.success = len(summary.errors) == 0
            summary_path.write_text(json.dumps(asdict(summary), indent=2))
            sync_run_summary_file(
                config,
                summary_path,
                summary_payload=asdict(summary),
            )
    except Exception as exc:
        summary.warnings.append(f"Local tracking index update failed: {exc}")
        summary.success = len(summary.errors) == 0
        summary_path.write_text(json.dumps(asdict(summary), indent=2))
    _progress_message(
        "run",
        f"Session finalized with {len(summary.errors)} error(s) and "
        f"{len(summary.warnings)} warning(s): {session_dir}.",
    )
    return summary


def _format_stream_encoder_stats(stats: Optional[Dict[str, Any]]) -> str:
    if not stats:
        return "n/a"
    encoder = str(stats.get("encoder", "?"))
    if stats.get("preset"):
        encoder += f" {stats['preset']}"
    return (
        f"{encoder}; most frames waiting {stats.get('peak_queue_frames', 0)} "
        f"({float(stats.get('peak_queue_mb', 0.0)):.0f} of {float(stats.get('queue_budget_mb', 0.0)):.0f} MB); "
        f"capture waited {float(stats.get('capture_stall_seconds', 0.0)):.2f}s in "
        f"{stats.get('capture_stall_count', 0)} stall(s)"
    )


def format_run_summary(summary: RunSummary) -> str:
    lines = [
        f"Run mode: {summary.mode}",
        f"Session: {summary.session_name}",
        f"Directory: {summary.session_dir}",
        f"Python executable: {summary.python_executable}",
        f"Camera profile: {summary.camera_profile}",
        f"Camera model: {summary.camera_model}",
        f"Camera IR setting: {summary.camera_infrared if summary.camera_infrared is not None else 'n/a'}",
        f"Camera monochrome output: {summary.camera_monochrome_output if summary.camera_monochrome_output is not None else 'n/a'}",
        f"Camera autofocus mode: {summary.camera_autofocus_mode}",
        (
            "Camera manual lens position: "
            f"{summary.camera_lens_position if summary.camera_lens_position is not None else 'default'}"
        ),
        f"Camera focus lock after warmup: {summary.camera_focus_lock_after_warmup}",
        f"Camera autofocus preflight enabled: {summary.camera_autofocus_preflight_enabled}",
        f"Camera autofocus preflight performed: {summary.camera_autofocus_preflight_performed}",
        (
            "Camera autofocus preflight lens position: "
            f"{summary.camera_autofocus_preflight_lens_position if summary.camera_autofocus_preflight_lens_position is not None else 'n/a'}"
        ),
        (
            "Camera autofocus preflight selection: "
            f"{summary.camera_autofocus_preflight_selection_reason or 'n/a'}"
        ),
        (
            "Camera autofocus preflight final state: "
            f"{summary.camera_autofocus_preflight_final_state or 'n/a'}"
        ),
        (
            "Camera autofocus preflight best score: "
            f"{summary.camera_autofocus_preflight_best_score if summary.camera_autofocus_preflight_best_score is not None else 'n/a'}"
        ),
        (
            "Camera autofocus preflight elapsed (s): "
            f"{summary.camera_autofocus_preflight_elapsed_seconds if summary.camera_autofocus_preflight_elapsed_seconds is not None else 'n/a'}"
        ),
        f"Resolved tuning file: {summary.resolved_tuning_file or 'default'}",
        f"Frames captured: {summary.frames_captured}",
        f"Actual FPS: {summary.actual_fps}",
        f"Tracking elapsed (s): {summary.tracking_elapsed_seconds if summary.tracking_elapsed_seconds is not None else 'n/a'}",
        f"Tracking frames processed: {summary.tracking_frames_processed}",
        f"Tracking processing FPS: {summary.tracking_processing_fps if summary.tracking_processing_fps is not None else 'n/a'}",
        f"Recording codec: {summary.video_codec or 'n/a'}",
        f"Frame buffer: {summary.capture_frame_buffer}",
        f"Stream encoder: {_format_stream_encoder_stats(summary.stream_encoder_stats)}",
        f"Video: {summary.video_path or 'none'}"
        + (f" (first of {len(summary.video_parts)} files)" if len(summary.video_parts) > 1 else ""),
        f"Midpoint PNG: {summary.recording_preview_png_path or 'none'}",
        f"Timestamps: {summary.timestamp_path or 'none'}",
        f"Thermal enabled: {summary.thermal_enabled}",
        f"Thermal device: {summary.thermal_device_path or 'none'}",
        f"Thermal frames captured: {summary.thermal_frames_captured}",
        f"Thermal actual FPS: {summary.thermal_actual_fps if summary.thermal_actual_fps is not None else 'n/a'}",
        f"Thermal timestamps: {summary.thermal_timestamp_path or 'none'}",
        f"Thermal raw NPY: {summary.thermal_raw_npy_path or 'none'}",
        f"Thermal preview video: {summary.thermal_preview_video_path or 'none'}",
        f"Thermal midpoint PNG: {summary.thermal_preview_png_path or 'none'}",
        f"Thermal metadata JSON: {summary.thermal_metadata_json_path or 'none'}",
        f"RGB+thermal side-by-side video: {summary.thermal_side_by_side_video_path or 'none'}",
        f"RGB+thermal side-by-side PNG: {summary.thermal_side_by_side_png_path or 'none'}",
        f"RealSense enabled: {summary.realsense_enabled}",
        f"RealSense device serial: {summary.realsense_device_serial or 'none'}",
        f"RealSense frames captured: {summary.realsense_frames_captured}",
        f"RealSense actual FPS: {summary.realsense_actual_fps if summary.realsense_actual_fps is not None else 'n/a'}",
        f"RealSense depth scale (meters/unit): {summary.realsense_depth_scale_meters if summary.realsense_depth_scale_meters is not None else 'n/a'}",
        f"RealSense timestamps: {summary.realsense_timestamp_path or 'none'}",
        f"RealSense raw depth NPY: {summary.realsense_raw_depth_npy_path or 'none'}",
        f"RealSense depth preview video: {summary.realsense_depth_preview_video_path or 'none'}",
        f"RealSense depth midpoint PNG: {summary.realsense_depth_preview_png_path or 'none'}",
        f"RealSense color video: {summary.realsense_color_video_path or 'none'}",
        f"RealSense color midpoint PNG: {summary.realsense_color_preview_png_path or 'none'}",
        f"RealSense metadata JSON: {summary.realsense_metadata_json_path or 'none'}",
        f"Tracked video: {summary.tracked_video_path or 'none'}",
        f"Tracked RGB+thermal side-by-side video: {summary.tracked_thermal_side_by_side_video_path or 'none'}",
        f"Tracked RGB+thermal side-by-side PNG: {summary.tracked_thermal_side_by_side_png_path or 'none'}",
        f"Raw CSV: {summary.raw_csv_path or 'none'}",
        f"NoID CSV: {summary.noid_csv_path or 'none'}",
        f"Cleaned CSV: {summary.cleaned_csv_path or 'none'}",
        f"FPS report: {summary.fps_report_json or 'none'}",
        f"Config snapshot: {summary.config_snapshot_path or 'none'}",
        f"Local tracking index: {summary.local_index_path or 'none'}",
        f"Local index manifest: {summary.local_index_manifest_path or 'none'}",
        f"Warnings: {len(summary.warnings)}",
        f"Errors: {len(summary.errors)}",
        f"Success: {summary.success}",
    ]
    if summary.camera_autofocus_preflight_notes:
        lines.append("")
        lines.append("Autofocus preflight details:")
        for note in summary.camera_autofocus_preflight_notes:
            lines.append(f"- {note}")
    if summary.warnings:
        lines.append("")
        lines.append("Warning details:")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
    if summary.errors:
        lines.append("")
        lines.append("Error details:")
        for error in summary.errors:
            lines.append(f"- {error}")
    return "\n".join(lines)
