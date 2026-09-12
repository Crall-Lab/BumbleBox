from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Dict, Optional

from .simulated_capture import (
    simulated_frame_times,
    simulated_realsense_color_frame,
    simulated_realsense_depth_frame,
)


@dataclass
class RealSenseDevice:
    serial: str
    name: str
    product_line: Optional[str]
    firmware_version: Optional[str]
    usb_type: Optional[str]
    physical_port: Optional[str]
    depth_profiles: list[str]
    color_profiles: list[str]


@dataclass
class RealSenseCheckResult:
    module_available: bool
    sdk_version: Optional[str]
    preferred_serial: Optional[str]
    selected_serial: Optional[str]
    devices: list[RealSenseDevice]
    requested_depth_profile: str
    requested_color_profile: str
    align_to: str
    probe_attempted: bool
    probe_succeeded: bool
    depth_shape: Optional[str]
    depth_dtype: Optional[str]
    color_shape: Optional[str]
    color_dtype: Optional[str]
    depth_scale_meters: Optional[float]
    device_timestamp_ms: Optional[float]
    warnings: list[str]
    errors: list[str]


@dataclass
class RealSenseSnapshotResult:
    selected_serial: str
    device_name: str
    depth_shape: str
    depth_dtype: str
    color_shape: str
    color_dtype: str
    depth_scale_meters: float
    device_timestamp_ms: float
    color_device_timestamp_ms: float
    depth_timestamp_domain: str
    color_timestamp_domain: str
    host_receive_monotonic_seconds: float
    host_receive_unix_seconds: float
    captured_at_utc: str
    raw_depth_npy_path: str
    raw_depth_png16_path: str
    colorized_depth_png_path: str
    color_png_path: str
    metadata_json_path: str


@dataclass
class RealSenseRecordingResult:
    selected_serial: str
    device_name: str
    frames_captured: int
    actual_fps: float
    depth_scale_meters: float
    first_host_receive_monotonic_seconds: float
    last_host_receive_monotonic_seconds: float
    timestamp_path: str
    raw_depth_npy_path: Optional[str]
    depth_preview_video_path: Optional[str]
    depth_preview_png_path: Optional[str]
    color_video_path: Optional[str]
    color_preview_png_path: Optional[str]
    metadata_json_path: str


def _load_realsense():
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        raise RuntimeError(
            "RealSense Python support is not available. Install the librealsense SDK and "
            "pyrealsense2 for this Raspberry Pi runtime, then rerun the check. "
            f"Import error: {exc}"
        ) from exc
    return rs


def _safe_device_info(device: Any, field: Any) -> Optional[str]:
    try:
        if device.supports(field):
            value = str(device.get_info(field)).strip()
            return value or None
    except Exception:
        pass
    return None


def _device_stream_profiles(rs: Any, device: Any) -> tuple[list[str], list[str]]:
    depth_profiles: set[str] = set()
    color_profiles: set[str] = set()
    try:
        sensors = device.query_sensors()
    except Exception:
        return [], []
    for sensor in sensors:
        try:
            profiles = sensor.get_stream_profiles()
        except Exception:
            continue
        for profile in profiles:
            try:
                stream_type = profile.stream_type()
                if stream_type not in {rs.stream.depth, rs.stream.color}:
                    continue
                video = profile.as_video_stream_profile()
                value = f"{video.width()}x{video.height()}@{profile.fps()} {profile.format()}"
                if stream_type == rs.stream.depth:
                    depth_profiles.add(value)
                else:
                    color_profiles.add(value)
            except Exception:
                continue
    return sorted(depth_profiles), sorted(color_profiles)


def _device_record(rs: Any, device: Any) -> RealSenseDevice:
    serial = _safe_device_info(device, rs.camera_info.serial_number) or "unknown"
    depth_profiles, color_profiles = _device_stream_profiles(rs, device)
    return RealSenseDevice(
        serial=serial,
        name=_safe_device_info(device, rs.camera_info.name) or "RealSense device",
        product_line=_safe_device_info(device, rs.camera_info.product_line),
        firmware_version=_safe_device_info(device, rs.camera_info.firmware_version),
        usb_type=_safe_device_info(device, rs.camera_info.usb_type_descriptor),
        physical_port=_safe_device_info(device, rs.camera_info.physical_port),
        depth_profiles=depth_profiles,
        color_profiles=color_profiles,
    )


def discover_realsense_devices() -> tuple[Any, list[RealSenseDevice]]:
    rs = _load_realsense()
    context = rs.context()
    devices = [_device_record(rs, device) for device in context.query_devices()]
    return rs, devices


def _configured_serial(config: Dict[str, Any], override: Optional[str] = None) -> Optional[str]:
    raw = override if override is not None else config.get("realsense", {}).get("device_serial", "auto")
    value = str(raw or "auto").strip()
    return None if value.lower() in {"", "auto", "default"} else value


def _select_device(devices: list[RealSenseDevice], preferred_serial: Optional[str]) -> Optional[RealSenseDevice]:
    if preferred_serial:
        return next((device for device in devices if device.serial == preferred_serial), None)
    return devices[0] if devices else None


def _stream_settings(config: Dict[str, Any]) -> dict[str, int | str]:
    section = config.get("realsense", {})
    return {
        "depth_width": int(section.get("depth_width", 848)),
        "depth_height": int(section.get("depth_height", 480)),
        "color_width": int(section.get("color_width", 848)),
        "color_height": int(section.get("color_height", 480)),
        "fps": int(section.get("fps", 30)),
        "warmup_frames": int(section.get("warmup_frames", 15)),
        "align_to": str(section.get("align_to", "none")).strip().lower(),
    }


def _matching_profile_fps(
    profiles: list[str],
    *,
    width: int,
    height: int,
    pixel_format: str,
) -> list[int]:
    prefix = f"{width}x{height}@"
    suffix = f" format.{pixel_format}"
    matches: set[int] = set()
    for profile in profiles:
        if not profile.startswith(prefix) or not profile.endswith(suffix):
            continue
        raw_fps = profile[len(prefix) : -len(suffix)]
        try:
            matches.add(int(raw_fps))
        except ValueError:
            continue
    return sorted(matches)


def _profile_errors(device: RealSenseDevice, settings: dict[str, int | str]) -> list[str]:
    errors: list[str] = []
    fps = int(settings["fps"])
    requested = {
        "depth": (
            device.depth_profiles,
            int(settings["depth_width"]),
            int(settings["depth_height"]),
            "z16",
        ),
        "color": (
            device.color_profiles,
            int(settings["color_width"]),
            int(settings["color_height"]),
            "bgr8",
        ),
    }
    for stream_name, (profiles, width, height, pixel_format) in requested.items():
        available_fps = _matching_profile_fps(
            profiles,
            width=width,
            height=height,
            pixel_format=pixel_format,
        )
        if fps in available_fps:
            continue
        alternatives = ", ".join(str(value) for value in available_fps) or "none"
        errors.append(
            f"Requested RealSense {stream_name} profile {width}x{height}@{fps} {pixel_format} "
            f"is not advertised. Available FPS at this resolution/format: {alternatives}."
        )
    return errors


def _capture_frame_pair(
    rs: Any,
    config: Dict[str, Any],
    serial: str,
    *,
    warmup_frames_override: Optional[int] = None,
) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"NumPy is required for RealSense capture: {exc}") from exc

    settings = _stream_settings(config)
    pipeline = rs.pipeline()
    pipeline_config = rs.config()
    pipeline_config.enable_device(serial)
    pipeline_config.enable_stream(
        rs.stream.depth,
        int(settings["depth_width"]),
        int(settings["depth_height"]),
        rs.format.z16,
        int(settings["fps"]),
    )
    pipeline_config.enable_stream(
        rs.stream.color,
        int(settings["color_width"]),
        int(settings["color_height"]),
        rs.format.bgr8,
        int(settings["fps"]),
    )

    profile = pipeline.start(pipeline_config)
    try:
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        warmup_frames = (
            int(warmup_frames_override)
            if warmup_frames_override is not None
            else int(settings["warmup_frames"])
        )
        frames = None
        for _ in range(max(1, warmup_frames + 1)):
            frames = pipeline.wait_for_frames(5000)
        if frames is None:
            raise RuntimeError("RealSense did not return a frameset.")

        align_to = str(settings["align_to"])
        if align_to == "color":
            frames = rs.align(rs.stream.color).process(frames)
        elif align_to == "depth":
            frames = rs.align(rs.stream.depth).process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame:
            raise RuntimeError("RealSense frameset did not contain a depth frame.")
        if not color_frame:
            raise RuntimeError("RealSense frameset did not contain a color frame.")

        host_receive_monotonic_seconds = time.monotonic()
        host_receive_unix_seconds = time.time()
        return {
            "depth": np.asanyarray(depth_frame.get_data()).copy(),
            "color": np.asanyarray(color_frame.get_data()).copy(),
            "depth_scale_meters": depth_scale,
            "device_timestamp_ms": float(depth_frame.get_timestamp()),
            "color_device_timestamp_ms": float(color_frame.get_timestamp()),
            "depth_timestamp_domain": str(depth_frame.get_frame_timestamp_domain()),
            "color_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
            "host_receive_monotonic_seconds": host_receive_monotonic_seconds,
            "host_receive_unix_seconds": host_receive_unix_seconds,
            "depth_frame_number": int(depth_frame.get_frame_number()),
            "color_frame_number": int(color_frame.get_frame_number()),
        }
    finally:
        pipeline.stop()


def _iso_local(unix_seconds: float) -> str:
    return datetime.fromtimestamp(float(unix_seconds)).astimezone().isoformat(timespec="milliseconds")


def _iso_utc(unix_seconds: float) -> str:
    return datetime.fromtimestamp(float(unix_seconds), tz=timezone.utc).isoformat(timespec="milliseconds")


def _write_recording_timestamps(
    path: Path,
    records: list[dict[str, Any]],
    *,
    node_name: Optional[str] = None,
) -> None:
    node = str(node_name or socket.gethostname())
    header = (
        "frame,node,sensor,timestamp_source,time_s,host_receive_monotonic_s,host_receive_unix_s,"
        "host_receive_iso_local,host_receive_iso_utc,depth_device_timestamp_ms,"
        "color_device_timestamp_ms,depth_frame_number,color_frame_number,"
        "depth_timestamp_domain,color_timestamp_domain\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for index, record in enumerate(records):
            unix_seconds = float(record["host_receive_unix_seconds"])
            handle.write(
                f"{index},{node},realsense,device_and_host,{float(record['time_s']):.6f},"
                f"{float(record['host_receive_monotonic_seconds']):.6f},"
                f"{unix_seconds:.6f},{_iso_local(unix_seconds)},{_iso_utc(unix_seconds)},"
                f"{float(record['depth_device_timestamp_ms']):.6f},"
                f"{float(record['color_device_timestamp_ms']):.6f},"
                f"{int(record['depth_frame_number'])},{int(record['color_frame_number'])},"
                f"{record['depth_timestamp_domain']},{record['color_timestamp_domain']}\n"
            )


def _frame_number_gaps(records: list[dict[str, Any]], key: str) -> int:
    values = [int(record[key]) for record in records]
    return sum(max(0, current - previous - 1) for previous, current in zip(values, values[1:]))


def _write_depth_preview_artifacts(
    np_module: Any,
    cv2_module: Any,
    session_dir: Path,
    session_name: str,
    raw_depth_path: Path,
    *,
    frame_count: int,
    fps: float,
    min_value: int,
    max_value: int,
    midpoint_index: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[Path, Optional[Path]]:
    depth_stack = np_module.load(raw_depth_path, mmap_mode="r")
    height, width = int(depth_stack.shape[1]), int(depth_stack.shape[2])
    video_path = session_dir / f"{session_name}_realsense_depth_preview.avi"
    writer = cv2_module.VideoWriter(
        str(video_path),
        cv2_module.VideoWriter_fourcc(*"MJPG"),
        float(fps if fps > 0 else 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open RealSense depth preview writer: {video_path}")
    midpoint_colorized = None
    denominator = max(1, int(max_value) - int(min_value))
    try:
        for index in range(int(frame_count)):
            depth = depth_stack[index]
            scaled = (
                (depth.astype(np_module.float32) - float(min_value))
                * (255.0 / float(denominator))
            ).clip(0, 255).astype(np_module.uint8)
            scaled[depth == 0] = 0
            colorized = cv2_module.applyColorMap(scaled, cv2_module.COLORMAP_TURBO)
            writer.write(colorized)
            if index == midpoint_index:
                midpoint_colorized = colorized.copy()
            if progress_callback is not None:
                progress_callback(index + 1, int(frame_count))
    finally:
        writer.release()
        del depth_stack

    midpoint_path: Optional[Path] = None
    if midpoint_colorized is not None:
        candidate = session_dir / f"{session_name}_realsense_depth_midframe.png"
        if cv2_module.imwrite(str(candidate), midpoint_colorized):
            midpoint_path = candidate
    return video_path, midpoint_path


def capture_simulated_realsense_recording(
    config: Dict[str, Any],
    *,
    session_dir: str | Path,
    session_name: str,
    duration: float,
    start_monotonic: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    depth_preview_progress_callback: Optional[Callable[[int, int], None]] = None,
    node_name: Optional[str] = None,
) -> RealSenseRecordingResult:
    """Generate RealSense-compatible artifacts on a shared synthetic timeline."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"OpenCV and NumPy are required for simulated RealSense recording: {exc}") from exc

    output_dir = Path(session_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _stream_settings(config)
    section = config.get("realsense", {})
    save_depth = bool(section.get("save_depth", True))
    save_color = bool(section.get("save_color", True))
    duration = float(duration)
    fps = float(settings["fps"])
    frame_times = simulated_frame_times(duration, fps)
    frame_count = len(frame_times)
    start = float(start_monotonic) if start_monotonic is not None else time.perf_counter()
    monotonic_to_unix_offset = time.time() - time.perf_counter()

    align_to = str(settings["align_to"])
    depth_height = int(settings["color_height"] if align_to == "color" else settings["depth_height"])
    depth_width = int(settings["color_width"] if align_to == "color" else settings["depth_width"])
    color_height = int(settings["depth_height"] if align_to == "depth" else settings["color_height"])
    color_width = int(settings["depth_width"] if align_to == "depth" else settings["color_width"])

    raw_depth_path = output_dir / f"{session_name}_realsense_depth_raw16.npy"
    timestamp_path = output_dir / f"{session_name}_realsense_frame_timestamps.csv"
    metadata_path = output_dir / f"{session_name}_realsense_metadata.json"
    color_video_path = output_dir / f"{session_name}_realsense_color.avi"
    depth_stack = None
    color_writer = None
    records: list[dict[str, Any]] = []
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    midpoint_index = min(frame_count - 1, frame_count // 2)
    depth_midpoint = None
    color_midpoint = None

    if status_callback is not None:
        status_callback("Generating synchronized simulated RealSense depth and color frames.")
    try:
        if save_depth:
            depth_stack = np.lib.format.open_memmap(
                raw_depth_path,
                mode="w+",
                dtype=np.uint16,
                shape=(frame_count, depth_height, depth_width),
            )
        if save_color:
            color_writer = cv2.VideoWriter(
                str(color_video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (color_width, color_height),
            )
            if not color_writer.isOpened():
                raise RuntimeError(f"Failed to open simulated RealSense color writer: {color_video_path}")

        for index, relative_time in enumerate(frame_times):
            depth = simulated_realsense_depth_frame(
                depth_width,
                depth_height,
                relative_time,
                duration,
            )
            color = simulated_realsense_color_frame(
                color_width,
                color_height,
                relative_time,
                duration,
            )
            if depth_stack is not None:
                depth_stack[index] = depth
            if color_writer is not None:
                color_writer.write(color)

            frame_min = int(depth.min())
            frame_max = int(depth.max())
            min_value = frame_min if min_value is None else min(min_value, frame_min)
            max_value = frame_max if max_value is None else max(max_value, frame_max)
            if index == midpoint_index:
                depth_midpoint = depth.copy()
                color_midpoint = color.copy()

            host_monotonic = start + relative_time
            records.append(
                {
                    "time_s": relative_time,
                    "host_receive_monotonic_seconds": host_monotonic,
                    "host_receive_unix_seconds": host_monotonic + monotonic_to_unix_offset,
                    "depth_device_timestamp_ms": relative_time * 1000.0,
                    "color_device_timestamp_ms": relative_time * 1000.0 + 0.2,
                    "depth_frame_number": index + 1,
                    "color_frame_number": index + 1,
                    "depth_timestamp_domain": "simulated_clock",
                    "color_timestamp_domain": "simulated_clock",
                }
            )
            if progress_callback is not None:
                progress_callback(index + 1, frame_count)
    finally:
        if depth_stack is not None:
            depth_stack.flush()
            del depth_stack
        if color_writer is not None:
            color_writer.release()

    _write_recording_timestamps(timestamp_path, records, node_name=node_name)
    actual_fps = fps

    depth_video_path: Optional[Path] = None
    depth_png_path: Optional[Path] = None
    if save_depth:
        if status_callback is not None:
            status_callback("Writing simulated RealSense depth preview video.")
        depth_video_path, depth_png_path = _write_depth_preview_artifacts(
            np,
            cv2,
            output_dir,
            session_name,
            raw_depth_path,
            frame_count=frame_count,
            fps=actual_fps,
            min_value=int(min_value or 0),
            max_value=int(max_value or 0),
            midpoint_index=midpoint_index,
            progress_callback=depth_preview_progress_callback,
        )
        if depth_midpoint is not None:
            raw_midpoint_path = output_dir / f"{session_name}_realsense_depth_midframe_raw16.png"
            cv2.imwrite(str(raw_midpoint_path), depth_midpoint)

    color_png_path: Optional[Path] = None
    if save_color and color_midpoint is not None:
        candidate = output_dir / f"{session_name}_realsense_color_midframe.png"
        if cv2.imwrite(str(candidate), color_midpoint):
            color_png_path = candidate

    result = RealSenseRecordingResult(
        selected_serial="mock://realsense",
        device_name="Simulated RealSense",
        frames_captured=frame_count,
        actual_fps=actual_fps,
        depth_scale_meters=0.001,
        first_host_receive_monotonic_seconds=float(records[0]["host_receive_monotonic_seconds"]),
        last_host_receive_monotonic_seconds=float(records[-1]["host_receive_monotonic_seconds"]),
        timestamp_path=str(timestamp_path),
        raw_depth_npy_path=str(raw_depth_path) if save_depth else None,
        depth_preview_video_path=str(depth_video_path) if depth_video_path else None,
        depth_preview_png_path=str(depth_png_path) if depth_png_path else None,
        color_video_path=str(color_video_path) if save_color else None,
        color_preview_png_path=str(color_png_path) if color_png_path else None,
        metadata_json_path=str(metadata_path),
    )
    metadata = asdict(result)
    metadata.update(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "simulated": True,
            "simulation_signal": "shared_normalized_moving_target",
            "requested_duration_seconds": duration,
            "requested_depth_profile": (
                f"{settings['depth_width']}x{settings['depth_height']}@{settings['fps']} z16"
            ),
            "requested_color_profile": (
                f"{settings['color_width']}x{settings['color_height']}@{settings['fps']} bgr8"
            ),
            "align_to": align_to,
            "depth_min_nonzero_value": int(min_value or 0),
            "depth_max_value": int(max_value or 0),
            "depth_frame_number_gaps": 0,
            "color_frame_number_gaps": 0,
            "depth_timestamp_domains": ["simulated_clock"],
            "color_timestamp_domains": ["simulated_clock"],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if status_callback is not None:
        status_callback("Simulated RealSense artifacts are complete.")
    return result


class RealSenseRecordingSession:
    """Stream one RealSense recording to disk without retaining all frames in RAM."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        session_dir: str | Path,
        session_name: str,
        node_name: Optional[str] = None,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            raise RuntimeError(f"OpenCV and NumPy are required for RealSense recording: {exc}") from exc

        self._cv2 = cv2
        self._np = np
        self.config = config
        self.settings = _stream_settings(config)
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = str(session_name)
        self.node_name = str(node_name or socket.gethostname())
        section = config.get("realsense", {})
        self.save_depth = bool(section.get("save_depth", True))
        self.save_color = bool(section.get("save_color", True))

        self.rs, devices = discover_realsense_devices()
        preferred_serial = _configured_serial(config)
        self.device = _select_device(devices, preferred_serial)
        if self.device is None:
            if preferred_serial:
                raise RuntimeError(f"Configured RealSense serial was not found: {preferred_serial}")
            raise RuntimeError("No RealSense camera was discovered for synchronized recording.")
        profile_errors = _profile_errors(self.device, self.settings)
        if profile_errors:
            raise RuntimeError(" ".join(profile_errors))

        self.pipeline = self.rs.pipeline()
        pipeline_config = self.rs.config()
        pipeline_config.enable_device(self.device.serial)
        pipeline_config.enable_stream(
            self.rs.stream.depth,
            int(self.settings["depth_width"]),
            int(self.settings["depth_height"]),
            self.rs.format.z16,
            int(self.settings["fps"]),
        )
        pipeline_config.enable_stream(
            self.rs.stream.color,
            int(self.settings["color_width"]),
            int(self.settings["color_height"]),
            self.rs.format.bgr8,
            int(self.settings["fps"]),
        )
        self.pipeline_config = pipeline_config
        self.profile = None
        self.depth_scale_meters: Optional[float] = None
        self.aligner = None
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        try:
            self.profile = self.pipeline.start(self.pipeline_config)
        except Exception as exc:
            raise RuntimeError(f"Failed to start configured RealSense streams: {exc}") from exc
        self.started = True
        self.depth_scale_meters = float(
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        align_to = str(self.settings["align_to"])
        if align_to == "color":
            self.aligner = self.rs.align(self.rs.stream.color)
        elif align_to == "depth":
            self.aligner = self.rs.align(self.rs.stream.depth)

        try:
            for _ in range(max(0, int(self.settings["warmup_frames"]))):
                self.pipeline.wait_for_frames(5000)
        except Exception as exc:
            self.close()
            raise RuntimeError(f"RealSense warmup failed: {exc}") from exc

    def _process_frameset(self, frameset: Any) -> tuple[Any, Any, Any, Any]:
        if self.aligner is not None:
            frameset = self.aligner.process(frameset)
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense frameset did not contain both depth and color frames.")
        depth = self._np.asanyarray(depth_frame.get_data()).copy()
        color = self._np.asanyarray(color_frame.get_data()).copy()
        if depth.ndim != 2 or str(depth.dtype) != "uint16":
            raise RuntimeError(f"Unexpected RealSense depth layout: shape={depth.shape}, dtype={depth.dtype}")
        if color.ndim != 3 or int(color.shape[2]) != 3:
            raise RuntimeError(f"Unexpected RealSense color layout: shape={color.shape}, dtype={color.dtype}")
        return depth, color, depth_frame, color_frame

    def _grow_depth_memmap(self, depth_memmap: Any, count: int, new_capacity: int, partial: Path) -> Any:
        depth_memmap.flush()
        del depth_memmap
        current = self._np.load(partial, mmap_mode="r")
        grown_path = partial.with_name(partial.stem + "_grown.npy")
        grown = self._np.lib.format.open_memmap(
            grown_path,
            mode="w+",
            dtype=self._np.uint16,
            shape=(int(new_capacity), int(current.shape[1]), int(current.shape[2])),
        )
        grown[:count] = current[:count]
        grown.flush()
        del grown
        del current
        os.replace(grown_path, partial)
        return self._np.lib.format.open_memmap(partial, mode="r+")

    def _finalize_depth_npy(self, depth_memmap: Any, count: int, partial: Path, output: Path) -> None:
        depth_memmap.flush()
        del depth_memmap
        current = self._np.load(partial, mmap_mode="r")
        final_partial = output.with_name(output.stem + "_finalizing.npy")
        final = self._np.lib.format.open_memmap(
            final_partial,
            mode="w+",
            dtype=self._np.uint16,
            shape=(int(count), int(current.shape[1]), int(current.shape[2])),
        )
        chunk_size = max(1, min(32, int(count)))
        for offset in range(0, int(count), chunk_size):
            end = min(offset + chunk_size, int(count))
            final[offset:end] = current[offset:end]
        final.flush()
        del final
        del current
        os.replace(final_partial, output)
        partial.unlink(missing_ok=True)

    def _write_depth_previews(
        self,
        raw_depth_path: Path,
        *,
        frame_count: int,
        fps: float,
        min_value: int,
        max_value: int,
        midpoint_index: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[Path, Optional[Path]]:
        return _write_depth_preview_artifacts(
            self._np,
            self._cv2,
            self.session_dir,
            self.session_name,
            raw_depth_path,
            frame_count=frame_count,
            fps=fps,
            min_value=min_value,
            max_value=max_value,
            midpoint_index=midpoint_index,
            progress_callback=progress_callback,
        )

    def capture_for(
        self,
        *,
        duration: float,
        start_monotonic: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        depth_preview_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RealSenseRecordingResult:
        self.start()
        duration = float(duration)
        fps = float(self.settings["fps"])
        estimated_frames = max(1, int(math.ceil(duration * fps)))
        start = float(start_monotonic) if start_monotonic is not None else None

        records: list[dict[str, Any]] = []
        depth_memmap = None
        depth_capacity = max(8, estimated_frames + 8)
        depth_partial = self.session_dir / f".{self.session_name}_realsense_depth_partial.npy"
        raw_depth_path = self.session_dir / f"{self.session_name}_realsense_depth_raw16.npy"
        timestamp_path = self.session_dir / f"{self.session_name}_realsense_frame_timestamps.csv"
        metadata_path = self.session_dir / f"{self.session_name}_realsense_metadata.json"
        color_video_path = self.session_dir / f"{self.session_name}_realsense_color.avi"
        color_writer = None
        depth_midpoint = None
        color_midpoint = None
        midpoint_index = 0
        midpoint_error = float("inf")
        min_value: Optional[int] = None
        max_value: Optional[int] = None

        try:
            align_to = str(self.settings["align_to"])
            depth_height = int(
                self.settings["color_height"] if align_to == "color" else self.settings["depth_height"]
            )
            depth_width = int(
                self.settings["color_width"] if align_to == "color" else self.settings["depth_width"]
            )
            color_height = int(
                self.settings["depth_height"] if align_to == "depth" else self.settings["color_height"]
            )
            color_width = int(
                self.settings["depth_width"] if align_to == "depth" else self.settings["color_width"]
            )
            if self.save_depth:
                depth_memmap = self._np.lib.format.open_memmap(
                    depth_partial,
                    mode="w+",
                    dtype=self._np.uint16,
                    shape=(depth_capacity, depth_height, depth_width),
                )
            if self.save_color:
                color_writer = self._cv2.VideoWriter(
                    str(color_video_path),
                    self._cv2.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (color_width, color_height),
                )
                if not color_writer.isOpened():
                    raise RuntimeError(f"Failed to open RealSense color video writer: {color_video_path}")

            if start is None:
                start = time.perf_counter()
            while time.perf_counter() < start:
                time.sleep(min(0.001, max(0.0, start - time.perf_counter())))
            poll_for_frames = getattr(self.pipeline, "poll_for_frames", None)
            if callable(poll_for_frames):
                while poll_for_frames():
                    pass

            while True:
                if records and (time.perf_counter() - start) >= duration:
                    break
                frameset = self.pipeline.wait_for_frames(5000)
                host_monotonic = time.perf_counter()
                host_unix = time.time()
                depth, color, depth_frame, color_frame = self._process_frameset(frameset)
                relative_time = float(host_monotonic - start)

                if self.save_depth:
                    if tuple(depth.shape) != tuple(depth_memmap.shape[1:]):
                        raise RuntimeError(
                            f"RealSense depth shape changed during recording: {depth_memmap.shape[1:]} to {depth.shape}"
                        )
                    if len(records) >= depth_capacity:
                        depth_capacity = max(depth_capacity + 8, int(math.ceil(depth_capacity * 1.5)))
                        depth_memmap = self._grow_depth_memmap(
                            depth_memmap,
                            len(records),
                            depth_capacity,
                            depth_partial,
                        )
                    depth_memmap[len(records)] = depth

                valid_depth = depth[depth > 0]
                if valid_depth.size:
                    frame_min = int(valid_depth.min())
                    frame_max = int(valid_depth.max())
                    min_value = frame_min if min_value is None else min(min_value, frame_min)
                    max_value = frame_max if max_value is None else max(max_value, frame_max)

                if self.save_color:
                    if (int(color.shape[1]), int(color.shape[0])) != (color_width, color_height):
                        raise RuntimeError(
                            "RealSense color shape did not match the configured/aligned output: "
                            f"expected {(color_height, color_width)}, got {color.shape[:2]}"
                        )
                    color_writer.write(color)

                distance_from_midpoint = abs(relative_time - duration / 2.0)
                if distance_from_midpoint < midpoint_error:
                    midpoint_error = distance_from_midpoint
                    midpoint_index = len(records)
                    if self.save_depth:
                        depth_midpoint = depth.copy()
                    if self.save_color:
                        color_midpoint = color.copy()

                records.append(
                    {
                        "time_s": relative_time,
                        "host_receive_monotonic_seconds": host_monotonic,
                        "host_receive_unix_seconds": host_unix,
                        "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
                        "color_device_timestamp_ms": float(color_frame.get_timestamp()),
                        "depth_frame_number": int(depth_frame.get_frame_number()),
                        "color_frame_number": int(color_frame.get_frame_number()),
                        "depth_timestamp_domain": str(depth_frame.get_frame_timestamp_domain()),
                        "color_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                    }
                )
                if progress_callback is not None:
                    progress_callback(len(records), estimated_frames)
        except Exception:
            if depth_memmap is not None:
                depth_memmap.flush()
                del depth_memmap
            depth_partial.unlink(missing_ok=True)
            raise
        finally:
            if color_writer is not None:
                color_writer.release()

        if not records:
            raise RuntimeError("RealSense synchronized capture did not yield any complete framesets.")
        if status_callback is not None:
            status_callback("RealSense acquisition complete; finalizing raw depth and timestamps.")
        if self.save_depth and depth_memmap is not None:
            self._finalize_depth_npy(depth_memmap, len(records), depth_partial, raw_depth_path)

        elapsed = float(records[-1]["time_s"] - records[0]["time_s"]) if len(records) > 1 else duration
        actual_fps = (
            float(len(records) - 1) / elapsed
            if elapsed > 0 and len(records) > 1
            else float(len(records)) / max(duration, 1e-6)
        )
        _write_recording_timestamps(timestamp_path, records, node_name=self.node_name)

        depth_video_path: Optional[Path] = None
        depth_png_path: Optional[Path] = None
        if self.save_depth:
            if status_callback is not None:
                status_callback("Writing RealSense depth preview video.")
            depth_video_path, depth_png_path = self._write_depth_previews(
                raw_depth_path,
                frame_count=len(records),
                fps=actual_fps,
                min_value=int(min_value or 0),
                max_value=int(max_value or 0),
                midpoint_index=midpoint_index,
                progress_callback=depth_preview_progress_callback,
            )
            if depth_midpoint is not None:
                raw_midpoint_path = self.session_dir / f"{self.session_name}_realsense_depth_midframe_raw16.png"
                self._cv2.imwrite(str(raw_midpoint_path), depth_midpoint)

        color_png_path: Optional[Path] = None
        if self.save_color and color_midpoint is not None:
            candidate = self.session_dir / f"{self.session_name}_realsense_color_midframe.png"
            if self._cv2.imwrite(str(candidate), color_midpoint):
                color_png_path = candidate

        result = RealSenseRecordingResult(
            selected_serial=self.device.serial,
            device_name=self.device.name,
            frames_captured=len(records),
            actual_fps=float(actual_fps),
            depth_scale_meters=float(self.depth_scale_meters or 0.0),
            first_host_receive_monotonic_seconds=float(records[0]["host_receive_monotonic_seconds"]),
            last_host_receive_monotonic_seconds=float(records[-1]["host_receive_monotonic_seconds"]),
            timestamp_path=str(timestamp_path),
            raw_depth_npy_path=str(raw_depth_path) if self.save_depth else None,
            depth_preview_video_path=str(depth_video_path) if depth_video_path else None,
            depth_preview_png_path=str(depth_png_path) if depth_png_path else None,
            color_video_path=str(color_video_path) if self.save_color else None,
            color_preview_png_path=str(color_png_path) if color_png_path else None,
            metadata_json_path=str(metadata_path),
        )
        metadata = asdict(result)
        metadata.update(
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "requested_duration_seconds": duration,
                "requested_depth_profile": (
                    f"{self.settings['depth_width']}x{self.settings['depth_height']}@{self.settings['fps']} z16"
                ),
                "requested_color_profile": (
                    f"{self.settings['color_width']}x{self.settings['color_height']}@{self.settings['fps']} bgr8"
                ),
                "align_to": str(self.settings["align_to"]),
                "depth_min_nonzero_value": int(min_value or 0),
                "depth_max_value": int(max_value or 0),
                "depth_frame_number_gaps": _frame_number_gaps(records, "depth_frame_number"),
                "color_frame_number_gaps": _frame_number_gaps(records, "color_frame_number"),
                "depth_timestamp_domains": sorted({str(item["depth_timestamp_domain"]) for item in records}),
                "color_timestamp_domains": sorted({str(item["color_timestamp_domain"]) for item in records}),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if status_callback is not None:
            status_callback("RealSense raw depth, color, timestamps, previews, and metadata are complete.")
        return result

    def close(self) -> None:
        if self.started:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.started = False

    def __enter__(self) -> "RealSenseRecordingSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_realsense_check(
    config: Dict[str, Any],
    *,
    serial_override: Optional[str] = None,
    probe: bool = True,
) -> RealSenseCheckResult:
    warnings: list[str] = []
    errors: list[str] = []
    preferred_serial = _configured_serial(config, serial_override)
    settings = _stream_settings(config)
    requested_depth_profile = (
        f"{settings['depth_width']}x{settings['depth_height']}@{settings['fps']} z16"
    )
    requested_color_profile = (
        f"{settings['color_width']}x{settings['color_height']}@{settings['fps']} bgr8"
    )
    try:
        rs, devices = discover_realsense_devices()
    except Exception as exc:
        return RealSenseCheckResult(
            module_available=False,
            sdk_version=None,
            preferred_serial=preferred_serial,
            selected_serial=None,
            devices=[],
            requested_depth_profile=requested_depth_profile,
            requested_color_profile=requested_color_profile,
            align_to=str(settings["align_to"]),
            probe_attempted=False,
            probe_succeeded=False,
            depth_shape=None,
            depth_dtype=None,
            color_shape=None,
            color_dtype=None,
            depth_scale_meters=None,
            device_timestamp_ms=None,
            warnings=[],
            errors=[str(exc)],
        )

    selected = _select_device(devices, preferred_serial)
    if preferred_serial and selected is None:
        errors.append(f"Configured RealSense serial was not found: {preferred_serial}")
    elif selected is None:
        errors.append("No RealSense camera was discovered.")

    capture = None
    requested_profile_errors = _profile_errors(selected, settings) if selected else []
    if requested_profile_errors:
        errors.extend(requested_profile_errors)
    if probe and selected is not None and not requested_profile_errors:
        try:
            capture = _capture_frame_pair(rs, config, selected.serial, warmup_frames_override=2)
        except Exception as exc:
            errors.append(f"RealSense stream probe failed: {exc}")

    if selected and selected.usb_type and not selected.usb_type.startswith("3"):
        warnings.append(
            f"RealSense reports USB {selected.usb_type}; a USB 3 connection is recommended for depth streaming."
        )
        if requested_profile_errors:
            warnings.append(
                "Reconnect the RealSense camera through a USB 3 port and USB 3-capable cable "
                "to make higher-FPS profiles available."
            )
    if selected and "d405" in selected.name.lower():
        same_stream_size = (
            settings["depth_width"] == settings["color_width"]
            and settings["depth_height"] == settings["color_height"]
        )
        if not same_stream_size:
            warnings.append(
                "D405 depth and color streams should use the same resolution; update the configured profiles."
            )

    depth = capture.get("depth") if capture else None
    color = capture.get("color") if capture else None
    return RealSenseCheckResult(
        module_available=True,
        sdk_version=str(getattr(rs, "__version__", "unknown")),
        preferred_serial=preferred_serial,
        selected_serial=selected.serial if selected else None,
        devices=devices,
        requested_depth_profile=requested_depth_profile,
        requested_color_profile=requested_color_profile,
        align_to=str(settings["align_to"]),
        probe_attempted=bool(probe and selected is not None and not requested_profile_errors),
        probe_succeeded=capture is not None,
        depth_shape="x".join(str(value) for value in depth.shape) if depth is not None else None,
        depth_dtype=str(depth.dtype) if depth is not None else None,
        color_shape="x".join(str(value) for value in color.shape) if color is not None else None,
        color_dtype=str(color.dtype) if color is not None else None,
        depth_scale_meters=float(capture["depth_scale_meters"]) if capture else None,
        device_timestamp_ms=float(capture["device_timestamp_ms"]) if capture else None,
        warnings=warnings,
        errors=errors,
    )


def apply_detected_realsense_config(
    config: Dict[str, Any], result: RealSenseCheckResult
) -> Dict[str, Any]:
    if result.errors or not result.selected_serial or not result.probe_succeeded:
        raise RuntimeError("RealSense settings can only be applied after a successful stream probe.")
    from copy import deepcopy

    updated = deepcopy(config)
    section = updated.setdefault("realsense", {})
    section["enabled"] = True
    section["device_serial"] = result.selected_serial
    return updated


def capture_realsense_snapshot(
    config: Dict[str, Any],
    *,
    serial_override: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> RealSenseSnapshotResult:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"OpenCV and NumPy are required for RealSense snapshot saving: {exc}") from exc

    rs, devices = discover_realsense_devices()
    preferred_serial = _configured_serial(config, serial_override)
    selected = _select_device(devices, preferred_serial)
    if selected is None:
        if preferred_serial:
            raise RuntimeError(f"Configured RealSense serial was not found: {preferred_serial}")
        raise RuntimeError("No RealSense camera was discovered.")

    captured = _capture_frame_pair(rs, config, selected.serial)
    depth = captured["depth"]
    color = captured["color"]
    now = datetime.now().astimezone()
    captured_at_utc = datetime.now(timezone.utc).isoformat()
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    if output_dir is None:
        data_root = Path(config.get("system", {}).get("data_root", ".")).expanduser()
        output_root = data_root / now.strftime("%Y-%m-%d") / "realsense"
    else:
        output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_npy = output_root / f"{stamp}_realsense_depth_raw.npy"
    raw_png = output_root / f"{stamp}_realsense_depth_raw16.png"
    colorized_png = output_root / f"{stamp}_realsense_depth_colorized.png"
    color_png = output_root / f"{stamp}_realsense_color.png"
    metadata_json = output_root / f"{stamp}_realsense_snapshot.json"

    np.save(raw_npy, depth)
    if not cv2.imwrite(str(raw_png), depth):
        raise RuntimeError(f"Failed to write raw RealSense depth PNG: {raw_png}")
    # Recolor from the copied depth array so saved visualization and raw data share one frame.
    depth_8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_8, cv2.COLORMAP_TURBO)
    if not cv2.imwrite(str(colorized_png), depth_color):
        raise RuntimeError(f"Failed to write colorized RealSense depth PNG: {colorized_png}")
    if not cv2.imwrite(str(color_png), color):
        raise RuntimeError(f"Failed to write RealSense color PNG: {color_png}")

    result = RealSenseSnapshotResult(
        selected_serial=selected.serial,
        device_name=selected.name,
        depth_shape="x".join(str(value) for value in depth.shape),
        depth_dtype=str(depth.dtype),
        color_shape="x".join(str(value) for value in color.shape),
        color_dtype=str(color.dtype),
        depth_scale_meters=float(captured["depth_scale_meters"]),
        device_timestamp_ms=float(captured["device_timestamp_ms"]),
        color_device_timestamp_ms=float(captured["color_device_timestamp_ms"]),
        depth_timestamp_domain=str(captured["depth_timestamp_domain"]),
        color_timestamp_domain=str(captured["color_timestamp_domain"]),
        host_receive_monotonic_seconds=float(captured["host_receive_monotonic_seconds"]),
        host_receive_unix_seconds=float(captured["host_receive_unix_seconds"]),
        captured_at_utc=captured_at_utc,
        raw_depth_npy_path=str(raw_npy.resolve()),
        raw_depth_png16_path=str(raw_png.resolve()),
        colorized_depth_png_path=str(colorized_png.resolve()),
        color_png_path=str(color_png.resolve()),
        metadata_json_path=str(metadata_json.resolve()),
    )
    metadata = asdict(result)
    metadata["depth_frame_number"] = int(captured["depth_frame_number"])
    metadata["color_frame_number"] = int(captured["color_frame_number"])
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
    return result


def format_realsense_check_result(result: RealSenseCheckResult) -> str:
    lines = [
        "RealSense Camera Check",
        "----------------------",
        f"Python module available: {'yes' if result.module_available else 'no'}",
        f"SDK version: {result.sdk_version or 'unknown'}",
        f"Preferred serial: {result.preferred_serial or 'auto'}",
        f"Selected serial: {result.selected_serial or 'none'}",
        f"Requested depth: {result.requested_depth_profile}",
        f"Requested color: {result.requested_color_profile}",
        f"Alignment: {result.align_to}",
        f"Devices discovered: {len(result.devices)}",
    ]
    for device in result.devices:
        lines.append(
            f"- {device.name} | serial={device.serial} | product={device.product_line or 'unknown'} | "
            f"firmware={device.firmware_version or 'unknown'} | USB={device.usb_type or 'unknown'}"
        )
        if device.depth_profiles:
            lines.append("  depth profiles: " + ", ".join(device.depth_profiles))
        if device.color_profiles:
            lines.append("  color profiles: " + ", ".join(device.color_profiles))
    lines.extend(
        [
            f"Probe attempted: {'yes' if result.probe_attempted else 'no'}",
            f"Probe succeeded: {'yes' if result.probe_succeeded else 'no'}",
            f"Depth frame: {result.depth_shape or 'none'} {result.depth_dtype or ''}".rstrip(),
            f"Color frame: {result.color_shape or 'none'} {result.color_dtype or ''}".rstrip(),
            f"Depth scale (meters/unit): {result.depth_scale_meters if result.depth_scale_meters is not None else 'unknown'}",
        ]
    )
    if result.warnings:
        lines.extend(["", "Warnings", "--------"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "Errors", "------"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)


def format_realsense_snapshot_result(result: RealSenseSnapshotResult) -> str:
    return "\n".join(
        [
            f"Selected RealSense: {result.device_name} ({result.selected_serial})",
            f"Depth frame: {result.depth_shape} {result.depth_dtype}",
            f"Color frame: {result.color_shape} {result.color_dtype}",
            f"Depth scale (meters/unit): {result.depth_scale_meters}",
            f"Raw depth NPY: {result.raw_depth_npy_path}",
            f"Raw 16-bit depth PNG: {result.raw_depth_png16_path}",
            f"Colorized depth PNG: {result.colorized_depth_png_path}",
            f"Color PNG: {result.color_png_path}",
            f"Metadata JSON: {result.metadata_json_path}",
        ]
    )


def write_realsense_check_json(result: RealSenseCheckResult, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return path
