from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


THERMAL_PIXEL_FORMATS = {"auto", "y16", "gray8", "rgb"}
THERMAL_CANDIDATE_TOKENS = ("purethermal", "lepton", "flir", "groupgets")


@dataclass
class ThermalDevice:
    device_path: str
    label: str
    vendor: Optional[str]
    model: Optional[str]
    bus_info: Optional[str]
    sysfs_name: Optional[str]
    format_codes: List[str]
    supports_y16: bool
    is_candidate: bool
    notes: List[str]


@dataclass
class ThermalCheckResult:
    selected_device: Optional[str]
    preferred_device: Optional[str]
    requested_width: int
    requested_height: int
    devices: List[ThermalDevice]
    probe_opened: bool
    probe_frame_read: bool
    probe_backend: Optional[str]
    observed_width: Optional[int]
    observed_height: Optional[int]
    frame_shape: Optional[str]
    frame_dtype: Optional[str]
    warnings: List[str]
    errors: List[str]


def _run_command(command: List[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        return False, str(exc)
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, output


def _looks_like_thermal_device(text: str) -> bool:
    haystack = str(text or "").strip().lower()
    return any(token in haystack for token in THERMAL_CANDIDATE_TOKENS)


def _parse_v4l2_list_devices() -> Dict[str, str]:
    command = shutil.which("v4l2-ctl")
    if not command:
        return {}
    ok, output = _run_command([command, "--list-devices"])
    if not ok or not output:
        return {}

    labels: Dict[str, str] = {}
    current_label = ""
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current_label = line[:-1].strip()
            continue
        stripped = line.strip()
        if stripped.startswith("/dev/video") and current_label:
            labels[stripped] = current_label
    return labels


def _sysfs_video_devices() -> Dict[str, str]:
    found: Dict[str, str] = {}
    for video_dir in sorted(Path("/sys/class/video4linux").glob("video*")):
        dev_name = video_dir.name
        device_path = f"/dev/{dev_name}"
        try:
            label = (video_dir / "name").read_text().strip()
        except Exception:
            label = dev_name
        found[device_path] = label
    return found


def _udevadm_properties(device_path: str) -> Dict[str, str]:
    command = shutil.which("udevadm")
    if not command:
        return {}
    ok, output = _run_command([command, "info", "--name", device_path, "--query", "property"])
    if not ok or not output:
        return {}
    props: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key.strip()] = value.strip()
    return props


def _format_codes_for_device(device_path: str) -> List[str]:
    command = shutil.which("v4l2-ctl")
    if not command:
        return []
    ok, output = _run_command([command, "-d", device_path, "--list-formats-ext"])
    if not ok or not output:
        return []
    codes = sorted({match.strip() for match in re.findall(r"'([^']+)'", output) if match.strip()})
    return codes


def discover_thermal_devices() -> List[ThermalDevice]:
    labels = _parse_v4l2_list_devices()
    if not labels:
        labels = _sysfs_video_devices()

    devices: List[ThermalDevice] = []
    for device_path in sorted(labels.keys()):
        label = labels.get(device_path, Path(device_path).name)
        props = _udevadm_properties(device_path)
        format_codes = _format_codes_for_device(device_path)
        vendor = props.get("ID_VENDOR_FROM_DATABASE") or props.get("ID_VENDOR")
        model = props.get("ID_MODEL_FROM_DATABASE") or props.get("ID_MODEL")
        bus_info = props.get("ID_PATH") or props.get("ID_USB_PATH")
        sysfs_name = props.get("ID_V4L_PRODUCT") or label
        haystack = " ".join(
            [
                label,
                vendor or "",
                model or "",
                sysfs_name or "",
            ]
        )
        supports_y16 = any(code.strip().upper() == "Y16" for code in format_codes)
        is_candidate = _looks_like_thermal_device(haystack)
        notes: List[str] = []
        if supports_y16:
            notes.append("V4L2 advertises Y16, which is the preferred radiometric-ish format to inspect first.")
        if not format_codes:
            notes.append("Could not read V4L2 format list; install v4l-utils for better diagnostics.")
        devices.append(
            ThermalDevice(
                device_path=device_path,
                label=label,
                vendor=vendor,
                model=model,
                bus_info=bus_info,
                sysfs_name=sysfs_name,
                format_codes=format_codes,
                supports_y16=supports_y16,
                is_candidate=is_candidate,
                notes=notes,
            )
        )
    return devices


def _config_thermal(config: Dict[str, Any]) -> Dict[str, Any]:
    thermal = config.get("thermal", {})
    return thermal if isinstance(thermal, dict) else {}


def _preferred_thermal_device(config: Dict[str, Any], device_override: str | None) -> Optional[str]:
    if device_override:
        return str(device_override).strip()
    thermal = _config_thermal(config)
    text = str(thermal.get("device_path", "auto")).strip()
    if not text or text.lower() == "auto":
        return None
    return text


def _select_device(devices: List[ThermalDevice], preferred_device: str | None) -> Optional[ThermalDevice]:
    if preferred_device:
        preferred = str(preferred_device).strip()
        for device in devices:
            if device.device_path == preferred:
                return device
    for device in devices:
        if device.is_candidate:
            return device
    return devices[0] if devices else None


def run_thermal_check(
    config: Dict[str, Any],
    *,
    device_override: str | None = None,
    width_override: int | None = None,
    height_override: int | None = None,
) -> ThermalCheckResult:
    thermal = _config_thermal(config)
    requested_width = int(width_override or thermal.get("width", 160))
    requested_height = int(height_override or thermal.get("height", 120))
    preferred_device = _preferred_thermal_device(config, device_override=device_override)
    devices = discover_thermal_devices()
    selected = _select_device(devices, preferred_device=preferred_device)

    warnings: List[str] = []
    errors: List[str] = []

    if shutil.which("v4l2-ctl") is None:
        warnings.append("v4l2-ctl not found. Install v4l-utils for better thermal camera diagnostics.")
    if preferred_device and selected is None:
        errors.append(f"Preferred thermal device was not found: {preferred_device}")
    if selected is None:
        errors.append("No V4L2 video device was found for the thermal camera.")
        return ThermalCheckResult(
            selected_device=None,
            preferred_device=preferred_device,
            requested_width=requested_width,
            requested_height=requested_height,
            devices=devices,
            probe_opened=False,
            probe_frame_read=False,
            probe_backend=None,
            observed_width=None,
            observed_height=None,
            frame_shape=None,
            frame_dtype=None,
            warnings=warnings,
            errors=errors,
        )

    probe_opened = False
    probe_frame_read = False
    probe_backend: Optional[str] = None
    observed_width: Optional[int] = None
    observed_height: Optional[int] = None
    frame_shape: Optional[str] = None
    frame_dtype: Optional[str] = None

    try:
        import cv2
    except Exception as exc:
        warnings.append(f"OpenCV not available for thermal probe capture: {exc}")
        return ThermalCheckResult(
            selected_device=selected.device_path,
            preferred_device=preferred_device,
            requested_width=requested_width,
            requested_height=requested_height,
            devices=devices,
            probe_opened=False,
            probe_frame_read=False,
            probe_backend=None,
            observed_width=None,
            observed_height=None,
            frame_shape=None,
            frame_dtype=None,
            warnings=warnings,
            errors=errors,
        )

    capture = cv2.VideoCapture(selected.device_path, cv2.CAP_V4L2)
    try:
        probe_opened = bool(capture.isOpened())
        if not probe_opened:
            errors.append(f"OpenCV could not open the selected thermal device: {selected.device_path}")
        else:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(requested_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(requested_height))
            if hasattr(cv2, "CAP_PROP_CONVERT_RGB"):
                capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            probe_frame_read, frame = capture.read()
            observed_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            observed_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            if hasattr(capture, "getBackendName"):
                try:
                    probe_backend = str(capture.getBackendName())
                except Exception:
                    probe_backend = None
            if probe_frame_read and frame is not None:
                frame_shape = "x".join(str(part) for part in getattr(frame, "shape", ()))
                frame_dtype = str(getattr(frame, "dtype", "unknown"))
            else:
                errors.append("Thermal device opened, but no frame was read.")
    finally:
        capture.release()

    if selected.supports_y16 and frame_dtype and "uint16" not in frame_dtype:
        warnings.append(
            "Device advertises Y16, but OpenCV did not return uint16 in this probe. "
            "That usually means the default capture path is converting formats."
        )

    return ThermalCheckResult(
        selected_device=selected.device_path,
        preferred_device=preferred_device,
        requested_width=requested_width,
        requested_height=requested_height,
        devices=devices,
        probe_opened=probe_opened,
        probe_frame_read=probe_frame_read,
        probe_backend=probe_backend,
        observed_width=observed_width,
        observed_height=observed_height,
        frame_shape=frame_shape,
        frame_dtype=frame_dtype,
        warnings=warnings,
        errors=errors,
    )


def format_thermal_check_result(result: ThermalCheckResult) -> str:
    lines = [
        "Thermal Camera Check",
        "--------------------",
        f"Preferred device: {result.preferred_device or 'auto'}",
        f"Selected device: {result.selected_device or 'none'}",
        f"Requested probe size: {result.requested_width}x{result.requested_height}",
        f"Probe opened: {'yes' if result.probe_opened else 'no'}",
        f"Probe frame read: {'yes' if result.probe_frame_read else 'no'}",
    ]
    if result.probe_backend:
        lines.append(f"OpenCV backend: {result.probe_backend}")
    if result.observed_width and result.observed_height:
        lines.append(f"Observed probe size: {result.observed_width}x{result.observed_height}")
    if result.frame_shape:
        lines.append(f"Frame shape: {result.frame_shape}")
    if result.frame_dtype:
        lines.append(f"Frame dtype: {result.frame_dtype}")

    lines.append("")
    lines.append(f"Discovered video devices: {len(result.devices)}")
    for device in result.devices:
        marker = "*" if device.device_path == result.selected_device else "-"
        candidate_text = "candidate" if device.is_candidate else "other"
        lines.append(
            f"{marker} {device.device_path} | {device.label} | {candidate_text} | "
            f"formats={','.join(device.format_codes) or 'unknown'}"
        )
        if device.vendor or device.model:
            lines.append(f"  vendor/model: {(device.vendor or 'unknown')} / {(device.model or 'unknown')}")
        if device.bus_info:
            lines.append(f"  bus: {device.bus_info}")
        if device.notes:
            for note in device.notes:
                lines.append(f"  note: {note}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings")
        lines.append("--------")
        for warning in result.warnings:
            lines.append(f"- {warning}")

    if result.errors:
        lines.append("")
        lines.append("Errors")
        lines.append("------")
        for error in result.errors:
            lines.append(f"- {error}")
    else:
        lines.append("")
        lines.append("Result")
        lines.append("------")
        lines.append(
            "Thermal discovery/probe succeeded. The next step is to confirm the selected device remains stable "
            "across reboots and that Y16 is available if you need radiometric-style processing."
        )

    return "\n".join(lines)


def write_thermal_check_json(result: ThermalCheckResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))
    return path
