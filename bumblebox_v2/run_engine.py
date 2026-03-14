from __future__ import annotations

from copy import deepcopy
import gc
import json
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .thermal_camera import resolve_thermal_device_path
from .tuning import resolve_camera_tuning_file

CAMERA_REOPEN_RETRY_ATTEMPTS = 3
CAMERA_REOPEN_RETRY_DELAY_SECONDS = 0.75
CAMERA_RELEASE_SETTLE_SECONDS = 0.75
SYNC_CAPTURE_START_DELAY_SECONDS = 0.20
SYNC_CAPTURE_FLUSH_FRAMES = 4


@dataclass
class RunSummary:
    started_at: str
    finished_at: str
    mode: str
    session_name: str
    session_dir: str
    hostname: str
    frames_captured: int
    actual_fps: float
    tracking_elapsed_seconds: Optional[float]
    tracking_frames_processed: int
    tracking_processing_fps: Optional[float]
    video_codec: Optional[str]
    video_path: Optional[str]
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
    raw_csv_path: Optional[str]
    noid_csv_path: Optional[str]
    cleaned_csv_path: Optional[str]
    fps_report_json: Optional[str]
    config_snapshot_path: Optional[str]
    warnings: List[str]
    errors: List[str]
    success: bool


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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    dt = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    session_name = f"{hostname}_{dt}"

    data_root = Path(config["system"]["data_root"])
    day_dir = data_root / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    session_dir = day_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_name, session_dir


def _write_timestamps(timestamps: List[float], session_dir: Path, session_name: str) -> Path:
    path = session_dir / f"{session_name}_frame_timestamps.csv"
    with path.open("w") as f:
        f.write("frame,time_s\n")
        for i, value in enumerate(timestamps):
            f.write(f"{i},{value:.6f}\n")
    return path


def _write_named_timestamps(
    timestamps: List[float],
    session_dir: Path,
    stem: str,
) -> Path:
    path = session_dir / f"{stem}.csv"
    with path.open("w") as f:
        f.write("frame,time_s\n")
        for i, value in enumerate(timestamps):
            f.write(f"{i},{value:.6f}\n")
    return path


def _mock_capture_frames(config: Dict[str, Any]) -> Tuple[List[Any], List[float], float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("numpy is required for mock capture mode") from exc

    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])

    frame_count = max(1, int(round(duration * fps)))
    frames = []
    timestamps = []
    for i in range(frame_count):
        frame = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        frame[:height, :] = (i * 17) % 255
        frames.append(frame)
        timestamps.append(i / fps)

    actual_fps = frame_count / duration if duration > 0 else fps
    return frames, timestamps, actual_fps


def _capture_frames_from_started_picamera(
    picam2: Any,
    fps: float,
    duration: float,
    *,
    start_monotonic: float | None = None,
) -> Tuple[List[Any], List[float], float]:
    frames: List[Any] = []
    timestamps: List[float] = []

    if start_monotonic is None:
        start = time.perf_counter()
    else:
        start = float(start_monotonic)
        while True:
            now = time.perf_counter()
            if now >= start:
                break
            time.sleep(min(0.001, max(0.0, start - now)))
    frame_index = 0
    target_interval = 1.0 / fps
    while (time.perf_counter() - start) < duration:
        now = time.perf_counter()
        expected = start + frame_index * target_interval
        if now >= expected:
            yuv420 = picam2.capture_array()
            frames.append(yuv420)
            timestamps.append(now - start)
            frame_index += 1

    elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else duration
    actual_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 and len(timestamps) > 1 else float(len(frames)) / max(duration, 1e-6)
    return frames, timestamps, actual_fps


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
        self.noise_reduction = config["camera"].get("noise_reduction", "Auto")
        self.warmup_s = float(config["runtime"].get("camera_warmup_seconds", 2.0))
        self.resolved_tuning_file = resolve_camera_tuning_file(config)
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
                if self.resolved_tuning_file:
                    tuning = self._Picamera2.load_tuning_file(str(self.resolved_tuning_file))
                    return self._Picamera2(tuning=tuning)
                return self._Picamera2()
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

        if self.noise_reduction != "Auto":
            try:
                mode = getattr(self._controls.draft.NoiseReductionModeEnum, str(self.noise_reduction))
                picam2.set_controls({"NoiseReductionMode": mode})
            except Exception:
                pass

        if isinstance(self.digital_zoom, (list, tuple)) and len(self.digital_zoom) == 4:
            picam2.set_controls({"ScalerCrop": tuple(self.digital_zoom)})
        return picam2

    def start(self) -> None:
        if self.started:
            return
        self.picam2.start()
        self.started = True
        time.sleep(max(0.0, self.warmup_s))

    def capture_for(
        self,
        *,
        fps: float,
        duration: float,
        start_monotonic: float | None = None,
    ) -> Tuple[List[Any], List[float], float]:
        self.start()
        return _capture_frames_from_started_picamera(
            self.picam2,
            float(fps),
            float(duration),
            start_monotonic=start_monotonic,
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


def _capture_frames_picamera(config: Dict[str, Any]) -> Tuple[List[Any], List[float], float]:
    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])
    with _PicameraCaptureSession(config) as session:
        return session.capture_for(fps=fps, duration=duration)


def _thermal_enabled_for_recording(config: Dict[str, Any]) -> bool:
    thermal = config.get("thermal", {})
    return isinstance(thermal, dict) and bool(thermal.get("enabled", False))


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
        self.capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open thermal device for synchronized recording: {self.device_path}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
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
    ) -> Tuple[List[Any], List[float], float, str]:
        frames: List[Any] = []
        timestamps: List[float] = []

        if start_monotonic is None:
            start = time.perf_counter()
        else:
            start = float(start_monotonic)
            while True:
                now = time.perf_counter()
                if now >= start:
                    break
                time.sleep(min(0.001, max(0.0, start - now)))

        frame_index = 0
        target_interval = 1.0 / float(fps)
        first_error: Optional[str] = None
        while (time.perf_counter() - start) < float(duration):
            now = time.perf_counter()
            expected = start + frame_index * target_interval
            if now < expected:
                time.sleep(min(0.001, max(0.0, expected - now)))
                continue

            ok, frame = self.capture.read()
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
            timestamps.append(now - start)
            frame_index += 1

        if self.raw16_layout is None:
            raise RuntimeError(first_error or "Thermal synchronized capture did not yield any usable raw16 frames.")

        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else float(duration)
        actual_fps = (
            (len(timestamps) - 1) / elapsed
            if elapsed > 0 and len(timestamps) > 1
            else float(len(frames)) / max(float(duration), 1e-6)
        )
        return frames, timestamps, float(actual_fps), self.raw16_layout

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
) -> Tuple[List[Any], List[float], float, str, str]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("numpy is required for mock thermal capture mode") from exc

    thermal = config.get("thermal", {})
    width = int(thermal.get("width", 160))
    height = int(thermal.get("height", 120))
    frames: List[Any] = []
    for idx, _timestamp in enumerate(timestamps):
        base = np.full((height, width), 29000 + (idx % 97), dtype=np.uint16)
        frames.append(base)
    return frames, list(timestamps), float(actual_fps), "uint16_mono16", "mock://thermal"


def _capture_rgb_and_optional_thermal(
    config: Dict[str, Any],
) -> Tuple[List[Any], List[float], float, Optional[dict[str, Any]]]:
    if not _thermal_enabled_for_recording(config):
        frames, timestamps, actual_fps = _capture_frames(config)
        return frames, timestamps, actual_fps, None

    if bool(config["runtime"].get("use_mock_camera", False)):
        frames, timestamps, actual_fps = _capture_frames(config)
        thermal_frames, thermal_timestamps, thermal_actual_fps, raw16_layout, device_path = _mock_capture_thermal_frames(
            config,
            timestamps=timestamps,
            actual_fps=actual_fps,
        )
        return frames, timestamps, actual_fps, {
            "device_path": device_path,
            "frames": thermal_frames,
            "timestamps": thermal_timestamps,
            "actual_fps": thermal_actual_fps,
            "raw16_layout": raw16_layout,
        }

    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])

    rgb_result: dict[str, Any] = {}
    thermal_result: dict[str, Any] = {}
    errors: list[str] = []

    def rgb_worker(session: _PicameraCaptureSession) -> None:
        try:
            frames, timestamps, actual_fps = session.capture_for(
                fps=fps,
                duration=duration,
                start_monotonic=start_monotonic,
            )
            rgb_result.update(
                {
                    "frames": frames,
                    "timestamps": timestamps,
                    "actual_fps": actual_fps,
                }
            )
        except Exception as exc:
            errors.append(f"RGB synchronized capture failed: {exc}")

    def thermal_worker(session: _ThermalCaptureSession) -> None:
        try:
            frames, timestamps, actual_fps, raw16_layout = session.capture_for(
                fps=fps,
                duration=duration,
                start_monotonic=start_monotonic,
            )
            thermal_result.update(
                {
                    "device_path": session.device_path,
                    "frames": frames,
                    "timestamps": timestamps,
                    "actual_fps": actual_fps,
                    "raw16_layout": raw16_layout,
                }
            )
        except Exception as exc:
            errors.append(f"Thermal synchronized capture failed: {exc}")

    with _PicameraCaptureSession(config) as rgb_session:
        rgb_session.start()
        with _ThermalCaptureSession(config) as thermal_session:
            # Flush queued frames immediately before the shared start so both streams begin from current data
            # instead of buffered warmup frames.
            _flush_rgb_capture_queue(rgb_session.picam2, SYNC_CAPTURE_FLUSH_FRAMES)
            _flush_thermal_capture_queue(thermal_session.capture, SYNC_CAPTURE_FLUSH_FRAMES)
            start_monotonic = time.perf_counter() + SYNC_CAPTURE_START_DELAY_SECONDS
            rgb_thread = threading.Thread(target=rgb_worker, args=(rgb_session,), daemon=True)
            thermal_thread = threading.Thread(target=thermal_worker, args=(thermal_session,), daemon=True)
            rgb_thread.start()
            thermal_thread.start()
            rgb_thread.join()
            thermal_thread.join()

    if errors:
        raise RuntimeError(" | ".join(errors))
    return (
        rgb_result["frames"],
        rgb_result["timestamps"],
        float(rgb_result["actual_fps"]),
        thermal_result or None,
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
) -> ThermalRecordingArtifacts:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError(f"OpenCV/numpy are required for thermal recording outputs: {exc}") from exc

    if not frames:
        raise RuntimeError("No thermal frames were captured.")

    stack = np.stack(frames, axis=0)
    raw_npy_path = session_dir / f"{session_name}_thermal_raw16.npy"
    np.save(raw_npy_path, stack)

    timestamp_path = _write_named_timestamps(
        timestamps,
        session_dir,
        f"{session_name}_thermal_frame_timestamps",
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
    try:
        for frame8 in preview8:
            writer.write(cv2.applyColorMap(frame8, cv2.COLORMAP_INFERNO))
    finally:
        writer.release()

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


def _write_rgb_thermal_side_by_side_outputs(
    *,
    session_dir: Path,
    session_name: str,
    rgb_frames: List[Any],
    rgb_timestamps: List[float],
    thermal_frames: List[Any],
    thermal_timestamps: List[float],
    fps: float,
) -> ThermalSideBySideArtifacts:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError(f"OpenCV/numpy are required for RGB+thermal comparison outputs: {exc}") from exc

    if not rgb_frames or not thermal_frames:
        raise RuntimeError("No overlapping RGB/thermal frames were available for side-by-side output.")

    thermal_times = list(thermal_timestamps)
    if not thermal_times:
        thermal_times = [float(i) / max(float(fps), 1e-6) for i in range(len(thermal_frames))]
    rgb_times = list(rgb_timestamps)
    if not rgb_times:
        rgb_times = [float(i) / max(float(fps), 1e-6) for i in range(len(rgb_frames))]

    matched_pairs: List[tuple[int, int, float, float]] = []
    thermal_idx = 0
    for rgb_idx, rgb_time in enumerate(rgb_times[: len(rgb_frames)]):
        while (
            thermal_idx + 1 < len(thermal_frames)
            and abs(thermal_times[thermal_idx + 1] - rgb_time) <= abs(thermal_times[thermal_idx] - rgb_time)
        ):
            thermal_idx += 1
        matched_pairs.append((rgb_idx, thermal_idx, rgb_time, thermal_times[thermal_idx]))

    frame_count = len(matched_pairs)
    if frame_count <= 0:
        raise RuntimeError("No overlapping RGB/thermal frames were available for side-by-side output.")

    first_rgb = _frame_to_bgr(rgb_frames[0])
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

    midpoint_png_path: Optional[Path] = None
    midpoint_idx = max(0, min(frame_count - 1, frame_count // 2))
    try:
        for pair_index, (rgb_idx, thermal_idx, rgb_time, thermal_time) in enumerate(matched_pairs):
            rgb_bgr = _frame_to_bgr(rgb_frames[rgb_idx])
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
            if pair_index == midpoint_idx:
                midpoint_png_path = session_dir / f"{session_name}_rgb_thermal_side_by_side_midframe.png"
                cv2.imwrite(str(midpoint_png_path), combined)
    finally:
        writer.release()

    return ThermalSideBySideArtifacts(
        frame_count=frame_count,
        video_path=video_path,
        midpoint_png_path=midpoint_png_path,
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


def _capture_frames(config: Dict[str, Any]) -> Tuple[List[Any], List[float], float]:
    if bool(config["runtime"].get("use_mock_camera", False)):
        return _mock_capture_frames(config)
    return _capture_frames_picamera(config)


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
    frames, _timestamps, actual_fps = _capture_frames(probe_config)
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
                writer.write(_frame_to_bgr(yuv420))
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
        frames, _timestamps, actual_fps = _mock_capture_frames(probe_config)
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
                writer.write(_frame_to_bgr(frame))
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

    frames, timestamps, actual_fps = _capture_frames(test_config)
    timestamp_path: Optional[Path] = None
    recording_preview_png_path: Optional[Path] = None

    if timestamps:
        timestamp_path = _write_timestamps(timestamps, session_dir, session_name)

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
        fps=float(test_config["camera"]["fps_target"]),
        width=int(test_config["camera"]["width"]),
        height=int(test_config["camera"]["height"]),
        recording_codec=requested_video_codec,
        mp4_codec=str(test_config["camera"].get("mp4_codec", "libx264")),
    )

    if not _video_file_is_readable(video_path):
        primary_size = int(video_path.stat().st_size) if video_path.exists() else 0
        if requested_video_codec == "mp4":
            fallback_session_name = f"{session_name}_fallback_mjpeg"
            fallback_path = _write_recording_video(
                frames,
                session_dir,
                fallback_session_name,
                fps=float(test_config["camera"]["fps_target"]),
                width=int(test_config["camera"]["width"]),
                height=int(test_config["camera"]["height"]),
                recording_codec="mjpeg",
                mp4_codec=str(test_config["camera"].get("mp4_codec", "libx264")),
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
    recording_preview_png_path = _write_midpoint_preview_png(frames, session_dir, preview_base_name)

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
) -> Path:
    codec_name = str(recording_codec).strip().lower()
    if codec_name == "mjpeg":
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency/runtime
            raise RuntimeError("OpenCV is required to write MJPEG recording output.") from exc

        output = session_dir / f"{session_name}.mjpeg"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output}")

        for frame in frames:
            writer.write(_frame_to_bgr(frame))

        writer.release()
        return output

    output = session_dir / f"{session_name}.mp4"
    return _write_mp4_video_with_ffmpeg(
        frames=frames,
        output=output,
        fps=fps,
        width=width,
        height=height,
        mp4_codec=mp4_codec,
    )


def _normalize_ffmpeg_mp4_codec(mp4_codec: str) -> str:
    raw = str(mp4_codec or "").strip().lower()
    aliases = {
        "mp4v": "mpeg4",
        "mpeg4": "mpeg4",
        "h264": "libx264",
        "x264": "libx264",
        "libx264": "libx264",
        "avc": "libx264",
        "h265": "libx265",
        "hevc": "libx265",
        "x265": "libx265",
        "libx265": "libx265",
    }
    return aliases.get(raw, raw or "mpeg4")


def _find_ffmpeg_binary() -> Optional[str]:
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    for candidate in ("/usr/bin/ffmpeg", "/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    return None


def _write_mp4_video_with_ffmpeg(
    frames: List[Any],
    output: Path,
    fps: float,
    width: int,
    height: int,
    mp4_codec: str,
) -> Path:
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
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        ffmpeg_codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if ffmpeg_codec == "libx264":
        cmd.extend(["-preset", "fast", "-crf", "18"])
    elif ffmpeg_codec == "libx265":
        cmd.extend(["-preset", "fast", "-crf", "20"])
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
    try:
        if process.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for MP4 encoding.")
        for frame in frames:
            process.stdin.write(_frame_to_bgr(frame).tobytes())
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
    return output


def _frame_to_bgr(frame: Any) -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("OpenCV is required for frame conversion.") from exc

    return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)


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
) -> Optional[Path]:
    if not frames:
        return None
    try:
        import cv2
    except ImportError:
        return None

    mid_idx = max(0, min(len(frames) - 1, len(frames) // 2))
    frame = frames[mid_idx]
    bgr = _frame_to_bgr(frame)
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
) -> Tuple[Any, Any]:
    try:
        from tag_tracking_utils import trackTagsFromVid
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import tag_tracking_utils for video tracking.") from exc

    box_preset = _normalize_box_preset(config["tracking"].get("box_preset"))
    now_str = _now_iso()
    colony_id = config["system"]["colony_id"]
    aruco_params = config["tracking"].get("aruco_params")

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


def run_once(config: Dict[str, Any], mode_override: Optional[str] = None) -> RunSummary:
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

    session_name, session_dir = _make_session_paths(config)
    hostname = socket.gethostname()

    frames: List[Any] = []
    timestamps: List[float] = []
    actual_fps = 0.0
    tracking_elapsed_seconds: Optional[float] = None
    tracking_frames_processed = 0
    tracking_processing_fps: Optional[float] = None
    video_codec: Optional[str] = None
    video_path: Optional[Path] = None
    recording_preview_png_path: Optional[Path] = None
    timestamp_path: Optional[Path] = None
    thermal_artifacts: Optional[ThermalRecordingArtifacts] = None
    thermal_side_by_side_artifacts: Optional[ThermalSideBySideArtifacts] = None
    raw_csv: Optional[Path] = None
    noid_csv: Optional[Path] = None
    cleaned_csv: Optional[Path] = None
    fps_report_json: Optional[Path] = None
    config_snapshot_path: Optional[Path] = None

    try:
        should_record = mode in {"record_only", "record_and_track"}
        should_track = mode in {"track_only", "record_and_track"}

        if should_record or should_track:
            if should_record and _thermal_enabled_for_recording(config):
                thermal_cfg = config.get("thermal", {})
                try:
                    thermal_target_fps = float(thermal_cfg.get("fps_target", config["camera"]["fps_target"]))
                except Exception:
                    thermal_target_fps = float(config["camera"]["fps_target"])
                if abs(thermal_target_fps - float(config["camera"]["fps_target"])) > 1e-6:
                    warnings.append(
                        "Synchronized thermal recording follows camera.fps_target. "
                        f"thermal.fps_target={thermal_target_fps:.3f} was ignored for this run."
                    )
                frames, timestamps, actual_fps, thermal_capture = _capture_rgb_and_optional_thermal(config)
                if thermal_capture is not None:
                    thermal_artifacts = _write_thermal_recording_outputs(
                        session_dir=session_dir,
                        session_name=session_name,
                        device_path=str(thermal_capture["device_path"]),
                        frames=list(thermal_capture["frames"]),
                        timestamps=list(thermal_capture["timestamps"]),
                        actual_fps=float(thermal_capture["actual_fps"]),
                        raw16_layout=str(thermal_capture["raw16_layout"]),
                    )
                    if thermal_artifacts.frames_captured != len(frames):
                        warnings.append(
                            "RGB and thermal synchronized recording captured different frame counts "
                            f"({len(frames)} RGB vs {thermal_artifacts.frames_captured} thermal). "
                            "Use the separate timestamp CSV files for alignment."
                        )
                    if abs(float(actual_fps) - float(thermal_artifacts.actual_fps)) > 0.25:
                        warnings.append(
                            "RGB and thermal actual FPS differed during synchronized recording "
                            f"({actual_fps:.3f} RGB vs {thermal_artifacts.actual_fps:.3f} thermal)."
                        )
                    try:
                        thermal_side_by_side_artifacts = _write_rgb_thermal_side_by_side_outputs(
                            session_dir=session_dir,
                            session_name=session_name,
                            rgb_frames=frames,
                            rgb_timestamps=timestamps,
                            thermal_frames=list(thermal_capture["frames"]),
                            thermal_timestamps=list(thermal_capture["timestamps"]),
                            fps=float(actual_fps if actual_fps > 0 else config["camera"]["fps_target"]),
                        )
                    except Exception as exc:
                        warnings.append(f"RGB+thermal side-by-side preview write failed: {exc}")
                    del thermal_capture["frames"]
                else:
                    thermal_artifacts = None
                    thermal_side_by_side_artifacts = None
            else:
                frames, timestamps, actual_fps = _capture_frames(config)
                if should_track and _thermal_enabled_for_recording(config) and not should_record:
                    warnings.append("thermal.enabled is set, but thermal capture runs only during recording modes.")

        if bool(config["runtime"].get("save_frame_timestamps", True)) and timestamps:
            timestamp_path = _write_timestamps(timestamps, session_dir, session_name)

        if should_record:
            video_codec = _normalize_recording_codec(config)
            video_path = _write_recording_video(
                frames,
                session_dir,
                session_name,
                fps=float(config["camera"]["fps_target"]),
                width=int(config["camera"]["width"]),
                height=int(config["camera"]["height"]),
                recording_codec=video_codec,
                mp4_codec=str(config["camera"].get("mp4_codec", "libx264")),
            )

            recording_preview_png_path = _write_midpoint_preview_png(frames, session_dir, session_name)
            if recording_preview_png_path is None:
                warnings.append("Could not write midpoint preview PNG.")

            if video_codec == "mp4" and bool(config["runtime"].get("save_mp4_sidecar_fps_txt", True)):
                sidecar = session_dir / f"{session_name}_actual_fps.txt"
                sidecar.write_text(f"{actual_fps:.6f}\n")

        if should_track:
            source = str(config["pipeline"].get("tracking_source", "ram"))
            if not bool(config["pipeline"].get("defer_tracking_until_after_recording", True)):
                warnings.append("Non-deferred tracking is not implemented yet; using deferred tracking.")

            tracking_started = time.perf_counter()
            if source == "video":
                if video_path is None:
                    warnings.append("tracking_source=video requires recording; falling back to RAM tracking.")
                    df, df2 = _track_from_ram(config, session_name, session_dir, frames)
                else:
                    df, df2 = _track_from_video(config, session_name, session_dir, video_path)
            else:
                df, df2 = _track_from_ram(config, session_name, session_dir, frames)
            tracking_elapsed_seconds = time.perf_counter() - tracking_started
            tracking_frames_processed = len(frames)
            if tracking_elapsed_seconds > 0 and tracking_frames_processed > 0:
                tracking_processing_fps = tracking_frames_processed / tracking_elapsed_seconds

            raw_csv = session_dir / f"{session_name}_raw.csv"
            noid_csv = session_dir / f"{session_name}_noID.csv"

            if not df.empty:
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

        fps_report_json = _run_fps_report_if_needed(config, session_dir, session_name, video_path, timestamp_path)

        try:
            config_snapshot_path = session_dir / f"{session_name}_config_snapshot.json"
            config_snapshot_path.write_text(json.dumps(config, indent=2))
        except Exception as exc:
            warnings.append(f"Config snapshot write failed: {exc}")

    except Exception as exc:
        errors.append(str(exc))

    finished_at = _now_iso()
    summary = RunSummary(
        started_at=started_at,
        finished_at=finished_at,
        mode=mode,
        session_name=session_name,
        session_dir=str(session_dir),
        hostname=hostname,
        frames_captured=len(frames),
        actual_fps=round(actual_fps, 6),
        tracking_elapsed_seconds=round(tracking_elapsed_seconds, 6) if tracking_elapsed_seconds is not None else None,
        tracking_frames_processed=tracking_frames_processed,
        tracking_processing_fps=(
            round(tracking_processing_fps, 6) if tracking_processing_fps is not None else None
        ),
        video_codec=video_codec,
        video_path=str(video_path) if video_path else None,
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
        raw_csv_path=str(raw_csv) if raw_csv else None,
        noid_csv_path=str(noid_csv) if noid_csv else None,
        cleaned_csv_path=str(cleaned_csv) if cleaned_csv else None,
        fps_report_json=str(fps_report_json) if fps_report_json else None,
        config_snapshot_path=str(config_snapshot_path) if config_snapshot_path else None,
        warnings=warnings,
        errors=errors,
        success=(len(errors) == 0),
    )

    summary_path = Path(summary.session_dir) / f"{summary.session_name}_run_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    return summary


def format_run_summary(summary: RunSummary) -> str:
    lines = [
        f"Run mode: {summary.mode}",
        f"Session: {summary.session_name}",
        f"Directory: {summary.session_dir}",
        f"Frames captured: {summary.frames_captured}",
        f"Actual FPS: {summary.actual_fps}",
        f"Tracking elapsed (s): {summary.tracking_elapsed_seconds if summary.tracking_elapsed_seconds is not None else 'n/a'}",
        f"Tracking frames processed: {summary.tracking_frames_processed}",
        f"Tracking processing FPS: {summary.tracking_processing_fps if summary.tracking_processing_fps is not None else 'n/a'}",
        f"Recording codec: {summary.video_codec or 'n/a'}",
        f"Video: {summary.video_path or 'none'}",
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
        f"Raw CSV: {summary.raw_csv_path or 'none'}",
        f"NoID CSV: {summary.noid_csv_path or 'none'}",
        f"Cleaned CSV: {summary.cleaned_csv_path or 'none'}",
        f"FPS report: {summary.fps_report_json or 'none'}",
        f"Config snapshot: {summary.config_snapshot_path or 'none'}",
        f"Warnings: {len(summary.warnings)}",
        f"Errors: {len(summary.errors)}",
        f"Success: {summary.success}",
    ]
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
