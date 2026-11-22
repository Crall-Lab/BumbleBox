"""Choosing which Raspberry Pi camera a config records from.

A Raspberry Pi 5 has two camera connectors. libcamera numbers the cameras it
finds 0, 1, ... in the order it detects them, and that order is not tied to
a connector: with one camera unplugged, the other becomes camera 0. A config
that only said "camera 0" would then record the wrong view into its data
without any error.

So a config can name its camera two ways:

* ``camera.camera_num``: the number, used when nothing else is known.
* ``camera.camera_id``: the libcamera Id of one physical camera connection
  (for example ``/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a``). When this
  is set it decides, and a camera that is not connected is an error rather
  than a silent switch to the other one.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

SESSION_TAG_PATTERN = re.compile(r"^[A-Za-z0-9-]{0,16}$")


class CameraSelectionError(RuntimeError):
    """The camera a config asks for is not connected."""


@dataclass
class DetectedCamera:
    num: int
    model: str
    camera_id: str

    @property
    def description(self) -> str:
        """For example 'imx477 on connector i2c@88000'."""
        where = connector_hint(self.camera_id)
        return f"{self.model or 'camera'} on {where}" if where else f"{self.model or 'camera'} (number {self.num})"


def connector_hint(camera_id: str) -> str:
    """The part of a libcamera Id that tells two connectors apart, if any."""
    match = re.search(r"(i2c@[0-9a-fA-F]+)", str(camera_id or ""))
    return f"connector {match.group(1)}" if match else ""


def _camera_section(config: Dict[str, Any]) -> Dict[str, Any]:
    camera = config.get("camera", {})
    return camera if isinstance(camera, dict) else {}


def configured_camera_num(config: Dict[str, Any]) -> int:
    try:
        return max(0, int(_camera_section(config).get("camera_num", 0) or 0))
    except (TypeError, ValueError):
        return 0


def configured_camera_id(config: Dict[str, Any]) -> Optional[str]:
    value = str(_camera_section(config).get("camera_id") or "").strip()
    return value or None


def session_tag(config: Dict[str, Any]) -> str:
    """Text added to session names so two cameras on one Pi never share a name."""
    value = str(_camera_section(config).get("session_tag") or "").strip()
    return value if SESSION_TAG_PATTERN.match(value) else ""


def parse_camera_info(camera_info: Any) -> List[DetectedCamera]:
    """Turn ``Picamera2.global_camera_info()`` into a list of cameras."""
    cameras: List[DetectedCamera] = []
    if not isinstance(camera_info, list):
        return cameras
    for position, item in enumerate(camera_info):
        if not isinstance(item, dict):
            continue
        try:
            num = int(item.get("Num", position))
        except (TypeError, ValueError):
            num = position
        cameras.append(DetectedCamera(num=num, model=str(item.get("Model") or ""), camera_id=str(item.get("Id") or "")))
    return cameras


def detect_cameras() -> List[DetectedCamera]:
    """Cameras libcamera can see right now; empty when picamera2 is unavailable."""
    try:
        from picamera2 import Picamera2

        return parse_camera_info(Picamera2.global_camera_info())
    except Exception:
        return []


def resolve_camera_num(config: Dict[str, Any], cameras: Optional[List[DetectedCamera]]) -> int:
    """The libcamera number to open for ``config``.

    ``cameras`` is what is connected now, or ``None`` when that could not be
    read (then the configured number is used as it is).
    """
    wanted_id = configured_camera_id(config)
    wanted_num = configured_camera_num(config)
    if cameras is None:
        return wanted_num

    listing = "; ".join(f"{camera.num}: {camera.description}" for camera in cameras) or "none"
    if wanted_id:
        for camera in cameras:
            if camera.camera_id == wanted_id:
                return camera.num
        raise CameraSelectionError(
            "The camera assigned to this config is not connected "
            f"(camera.camera_id = {wanted_id}). Cameras detected: {listing}. "
            "Check its ribbon cable. Recording was refused rather than using a different camera."
        )
    if cameras and wanted_num >= len(cameras):
        raise CameraSelectionError(
            f"This config records from camera {wanted_num} (camera.camera_num), but only "
            f"{len(cameras)} camera(s) are connected. Cameras detected: {listing}."
        )
    return wanted_num


def picamera2_camera_kwargs(config: Dict[str, Any], picamera2_class: Any) -> Dict[str, Any]:
    """Keyword arguments that make ``Picamera2(...)`` open the configured camera.

    Empty for the default (first camera, not pinned), so a single-camera box
    opens its camera exactly as it always has.
    """
    if configured_camera_id(config) is None and configured_camera_num(config) == 0:
        return {}
    try:
        cameras: Optional[List[DetectedCamera]] = parse_camera_info(picamera2_class.global_camera_info())
    except Exception:
        cameras = None
    num = resolve_camera_num(config, cameras)
    return {"camera_num": num} if num != 0 else {}


def capture_lock_suffix(config: Dict[str, Any]) -> str:
    """Distinguishes the capture locks of two cameras; empty for the first camera."""
    num = configured_camera_num(config)
    return f"cam{num}" if num > 0 else ""


def selected_camera_model(config: Dict[str, Any], picamera2_class: Any) -> Optional[str]:
    """The sensor name libcamera reports for the camera ``config`` opens, for example ``imx477``.

    ``None`` when the cameras cannot be listed or the configured one is not
    there (opening it then reports the real problem).
    """
    try:
        cameras = parse_camera_info(picamera2_class.global_camera_info())
        num = resolve_camera_num(config, cameras)
    except Exception:
        return None
    for camera in cameras:
        if camera.num == num:
            return camera.model or None
    return None


def connected_camera_model(config: Dict[str, Any]) -> Optional[str]:
    """As ``selected_camera_model``, where the camera library has not been imported yet."""
    try:
        from picamera2 import Picamera2
    except Exception:
        return None
    return selected_camera_model(config, Picamera2)
