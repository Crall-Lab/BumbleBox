from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .tuning import TUNING_SEARCH_DIRS, inspect_camera_tuning_resolution


@dataclass
class CheckResult:
    name: str
    status: str
    message: str


def _read_pi_model() -> str | None:
    model_paths = [
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
    ]

    for path in model_paths:
        if path.exists():
            raw = path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore")
            return raw.strip()
    return None


def _run_command(command: List[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None


def _status_for_pi(config_pi_model: str, detected_model: str | None) -> CheckResult:
    if not detected_model:
        return CheckResult(
            name="Pi model",
            status="WARN",
            message="Could not detect Raspberry Pi model (expected on Pi hardware).",
        )

    lower = detected_model.lower()
    if "raspberry pi 4" in lower:
        detected = "pi4"
    elif "raspberry pi 5" in lower:
        detected = "pi5"
    else:
        detected = "other"

    if config_pi_model == "auto":
        if detected in {"pi4", "pi5"}:
            return CheckResult("Pi model", "PASS", f"Detected {detected_model}")
        return CheckResult(
            "Pi model",
            "WARN",
            f"Detected {detected_model}. BumbleBox V2 is tuned for Raspberry Pi 4/5.",
        )

    if config_pi_model == detected:
        return CheckResult("Pi model", "PASS", f"Config matches detected hardware: {detected_model}")

    return CheckResult(
        "Pi model",
        "FAIL",
        f"Config expects {config_pi_model}, but detected {detected_model}",
    )


def _dependency_check(module_name: str, install_hint: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
        return CheckResult(f"Dependency: {module_name}", "PASS", "Installed")
    except ImportError:
        return CheckResult(
            f"Dependency: {module_name}",
            "FAIL",
            f"Missing (install with: {install_hint})",
        )


def _picamera2_check(detected_model: str | None) -> CheckResult:
    try:
        importlib.import_module("picamera2")
        return CheckResult("Dependency: picamera2", "PASS", "Installed")
    except Exception:
        pass

    is_pi = bool(detected_model and "raspberry pi" in detected_model.lower())
    if not is_pi:
        return CheckResult(
            "Dependency: picamera2",
            "FAIL",
            "Missing (install with: pip3 install picamera2)",
        )

    apt_installed = False
    dpkg = _run_command(["dpkg-query", "-W", "-f=${Status}", "python3-picamera2"])
    if dpkg is not None and dpkg.returncode == 0:
        text = f"{dpkg.stdout}\n{dpkg.stderr}".lower()
        apt_installed = "install ok installed" in text

    system_python_has_module = False
    system_py = _run_command(["/usr/bin/python3", "-c", "import picamera2"])
    if system_py is not None and system_py.returncode == 0:
        system_python_has_module = True

    if apt_installed or system_python_has_module:
        return CheckResult(
            "Dependency: picamera2",
            "WARN",
            (
                "python3-picamera2 appears installed for system Python, but not in the current interpreter. "
                "You are likely using a venv without system-site-packages. "
                "Recreate with: bash scripts/setup_venv.sh --system-site-packages"
            ),
        )

    return CheckResult(
        "Dependency: picamera2",
        "FAIL",
        (
            "Missing (install with: sudo apt install python3-picamera2 on Pi, or pip3 install picamera2). "
            "If using BumbleBox venv, prefer setup with --system-site-packages on Pi."
        ),
    )


def _opencv_aruco_check() -> CheckResult:
    try:
        import cv2
    except Exception:
        return CheckResult(
            "OpenCV ArUco",
            "FAIL",
            "cv2 import failed. Install opencv-contrib-python.",
        )

    if not hasattr(cv2, "aruco"):
        return CheckResult(
            "OpenCV ArUco",
            "FAIL",
            "cv2.aruco missing. Install opencv-contrib-python (not opencv-python).",
        )
    if not hasattr(cv2.aruco, "ArucoDetector"):
        return CheckResult(
            "OpenCV ArUco",
            "WARN",
            "cv2.aruco is present, but ArucoDetector API is missing (prefer OpenCV 4.7+).",
        )
    return CheckResult("OpenCV ArUco", "PASS", "cv2.aruco with ArucoDetector is available.")


def _camera_stack_check() -> CheckResult:
    command_names = ["rpicam-hello", "libcamera-hello"]
    any_found = False
    errors: List[str] = []

    for command_name in command_names:
        result = _run_command([command_name, "--list-cameras"])
        if result is None:
            continue

        any_found = True
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "no output"
            errors.append(f"{command_name}: {details}")
            continue

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        camera_lines = [line for line in lines if "camera" in line.lower() or "imx" in line.lower()]
        if not camera_lines:
            errors.append(f"{command_name}: command worked but no cameras were listed")
            continue

        return CheckResult(
            "Camera stack",
            "PASS",
            f"Detected camera stack via {command_name} with {len(camera_lines)} camera line(s).",
        )

    if not any_found:
        return CheckResult(
            "Camera stack",
            "WARN",
            (
                "Neither rpicam-hello nor libcamera-hello was found. "
                "Install Raspberry Pi camera apps (libcamera/rpicam)."
            ),
        )

    return CheckResult(
        "Camera stack",
        "WARN",
        "Camera command(s) found but camera listing failed: " + "; ".join(errors),
    )


def _ffmpeg_check(config: Dict[str, Any]) -> CheckResult:
    codec = str(config.get("camera", {}).get("codec", "mp4")).strip().lower()
    resolved = shutil.which("ffmpeg")
    if resolved:
        if codec == "mp4":
            return CheckResult("Dependency: ffmpeg", "PASS", f"Installed at {resolved} (required for MP4 recording).")
        return CheckResult("Dependency: ffmpeg", "PASS", f"Installed at {resolved}.")

    if codec == "mp4":
        return CheckResult(
            "Dependency: ffmpeg",
            "FAIL",
            "Missing. MP4 recording now requires ffmpeg. Install with: sudo apt install ffmpeg",
        )
    return CheckResult(
        "Dependency: ffmpeg",
        "WARN",
        "Missing. Not required for MJPEG recording, but needed if you switch codec to MP4.",
    )


def _data_root_check(data_root: str) -> CheckResult:
    path = Path(data_root)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        return CheckResult(
            "Data root",
            "FAIL",
            (
                f"Cannot create {path}: {exc}. "
                "Run 'bbx storage setup --apply-config' for external storage, or set a writable path with "
                "'bbx storage set-mount-point --mount-point /home/<user>/BumbleBoxData'."
            ),
        )
    except Exception as exc:
        return CheckResult("Data root", "FAIL", f"Cannot create {path}: {exc}")

    if not os.access(path, os.W_OK):
        return CheckResult("Data root", "FAIL", f"Path is not writable: {path}")

    return CheckResult("Data root", "PASS", f"Writable: {path}")


def _camera_tuning_check(config: Dict[str, Any]) -> CheckResult:
    info = inspect_camera_tuning_resolution(config)
    source = str(info.get("source", "default"))
    requested = info.get("requested")
    resolved = info.get("resolved")
    resolved_path = info.get("resolved_path")
    camera_model = info.get("camera_model")
    camera_profile = info.get("camera_profile")
    infrared = info.get("infrared")
    search_paths = ", ".join(str(path) for path in TUNING_SEARCH_DIRS)

    if source == "default":
        return CheckResult(
            "Camera tuning",
            "PASS",
            (
                "No tuning file selected; using libcamera default sensor tuning "
                f"for camera.profile={camera_profile}, camera.model={camera_model}."
            ),
        )

    if resolved_path:
        if source == "explicit":
            return CheckResult(
                "Camera tuning",
                "PASS",
                f"Explicit tuning_file='{requested}' resolved to {resolved_path}.",
            )
        return CheckResult(
            "Camera tuning",
            "PASS",
            (
                f"Auto-selected tuning='{resolved}' for camera.model={camera_model}, "
                f"camera.profile={camera_profile}, camera.infrared={infrared}; resolved to {resolved_path}."
            ),
        )

    if source == "explicit":
        return CheckResult(
            "Camera tuning",
            "WARN",
            (
                f"Explicit tuning_file='{requested}' was not found in common Pi tuning paths. "
                f"Searched: {search_paths}. "
                "Capture may fail if libcamera cannot resolve this file."
            ),
        )

    return CheckResult(
        "Camera tuning",
        "WARN",
        (
            f"Auto-selected tuning='{resolved}' for camera.model={camera_model}, "
            f"camera.infrared={infrared}, but file was not found in common Pi tuning paths. "
            f"Searched: {search_paths}."
        ),
    )


def _findmnt_target(path: Path) -> tuple[str, str] | None:
    result = _run_command(
        ["findmnt", "--noheadings", "--output", "SOURCE,TARGET", "--target", str(path)]
    )
    if result is None or result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _normalize_mount_path(path_text: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path_text).strip()))


def _fstab_entry_for_mount(mount_point: str) -> str | None:
    fstab_path = Path("/etc/fstab")
    if not fstab_path.exists():
        return None
    try:
        text = fstab_path.read_text()
    except Exception:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1] == mount_point:
            return line
    return None


def _uuid_fstab_note_needed(source: str | None, fstab_line: str | None) -> bool:
    if not source or not source.startswith("/dev/sd"):
        return False
    if fstab_line and "UUID=" in fstab_line:
        return False
    return True


def _data_root_mount_check(data_root: str) -> CheckResult:
    path = Path(data_root)
    if not path.exists():
        return CheckResult("Data root mount", "WARN", f"Path does not exist yet: {path}")

    mount = _findmnt_target(path)
    if mount is None:
        if str(path).startswith("/mnt/"):
            return CheckResult(
                "Data root mount",
                "WARN",
                (
                    f"{path} is under /mnt but no active mount was detected. "
                    "External storage may not be mounted."
                ),
            )
        return CheckResult("Data root mount", "WARN", "Could not resolve mount source with findmnt.")

    source, target = mount
    if _normalize_mount_path(target) != _normalize_mount_path(str(path)):
        return CheckResult(
            "Data root mount",
            "WARN",
            (
                f"{path} currently resides on {source} mounted at {target}. "
                "That does not mean the chosen data root is its own active mount point."
            ),
        )
    fstab_line = _fstab_entry_for_mount(str(path))
    if _uuid_fstab_note_needed(source, fstab_line):
        return CheckResult(
            "Data root mount",
            "WARN",
            (
                f"Mounted at {target} from {source}. Device names like sda1/sdb1 can change; "
                "prefer UUID entries in /etc/fstab."
            ),
        )
    return CheckResult("Data root mount", "PASS", f"Resolved mount {target} from {source}.")


def run_doctor(config: Dict[str, Any]) -> List[CheckResult]:
    results: List[CheckResult] = []
    py_version = tuple(int(part) for part in platform.python_version_tuple()[:3])

    results.append(
        CheckResult(
            "Python",
            "PASS" if py_version >= (3, 9, 0) else "FAIL",
            f"Running Python {platform.python_version()}",
        )
    )

    detected_model = _read_pi_model()
    results.append(_status_for_pi(config["system"]["pi_model"], detected_model))

    results.append(_dependency_check("cv2", "pip3 install opencv-contrib-python"))
    results.append(_opencv_aruco_check())
    results.append(_picamera2_check(detected_model))
    results.append(_dependency_check("yaml", "pip3 install pyyaml"))
    results.append(_dependency_check("pandas", "pip3 install pandas"))
    results.append(_ffmpeg_check(config))
    results.append(_camera_stack_check())
    results.append(_camera_tuning_check(config))
    results.append(_data_root_check(config["system"]["data_root"]))
    results.append(_data_root_mount_check(config["system"]["data_root"]))

    mode = config["pipeline"]["mode"]
    source = config["pipeline"]["tracking_source"]
    results.append(
        CheckResult(
            "Pipeline mode",
            "PASS",
            f"Configured mode={mode}, tracking_source={source}",
        )
    )

    return results


def format_report(results: List[CheckResult]) -> str:
    lines = []
    for result in results:
        lines.append(f"[{result.status}] {result.name}: {result.message}")

    fail_count = sum(1 for result in results if result.status == "FAIL")
    warn_count = sum(1 for result in results if result.status == "WARN")
    lines.append("")
    lines.append(f"Summary: {fail_count} fail, {warn_count} warn, {len(results) - fail_count - warn_count} pass")
    return "\n".join(lines)


def has_failures(results: List[CheckResult]) -> bool:
    return any(result.status == "FAIL" for result in results)
