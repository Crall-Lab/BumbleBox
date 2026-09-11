from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional


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
    if probe and selected is not None:
        try:
            capture = _capture_frame_pair(rs, config, selected.serial, warmup_frames_override=2)
        except Exception as exc:
            errors.append(f"RealSense stream probe failed: {exc}")

    if selected and selected.usb_type and not selected.usb_type.startswith("3"):
        warnings.append(
            f"RealSense reports USB {selected.usb_type}; a USB 3 connection is recommended for depth streaming."
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
        probe_attempted=bool(probe and selected is not None),
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
