"""Which libcamera tuning file a camera is opened with.

A tuning file tells the camera how to build a picture from the sensor: white
balance, colour, lens shading, noise handling. Raspberry Pi supplies two for
most sensors, ``<sensor>.json`` and ``<sensor>_noir.json``. The second is for
a camera with no infrared filter: a NoIR camera, or an ordinary one whose
filter was taken off.

* **NoIR or standard** is decided by ``camera.infrared`` alone, whatever the
  camera model.
* **Which sensor's file** comes from the camera that is connected, when that
  can be read. Otherwise from ``camera.model``, otherwise from the recording
  size.
* ``camera.tuning_file`` names a file by hand and overrides both.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

from .camera_profiles import configured_profile_name, get_camera_model_info, normalize_camera_model

# Where libcamera keeps tuning files, one folder per Raspberry Pi camera
# pipeline: "pisp" on a Pi 5, "vc4" on a Pi 4 and earlier. A file from the
# other pipeline's folder is refused by the camera, so only the folder for
# this computer is searched.
TUNING_ROOTS = (
    Path("/usr/local/share/libcamera/ipa/rpi"),
    Path("/usr/share/libcamera/ipa/rpi"),
)
PIPELINES = ("pisp", "vc4")
PI_MODEL_PATHS = (
    Path("/proc/device-tree/model"),
    Path("/sys/firmware/devicetree/base/model"),
)
TUNING_ENV = "LIBCAMERA_RPI_TUNING_FILE"

SENSOR_SOURCE_CAMERA = "connected camera"
SENSOR_SOURCE_MODEL = "camera.model"
SENSOR_SOURCE_SIZE = "recording size"


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _camera(config: Any) -> Dict[str, Any]:
    camera = config.get("camera", {}) if isinstance(config, dict) else {}
    return camera if isinstance(camera, dict) else {}


def pi_pipeline() -> Optional[str]:
    """``"pisp"`` on a Raspberry Pi 5 family board, ``"vc4"`` on an earlier Pi, ``None`` elsewhere."""
    for path in PI_MODEL_PATHS:
        try:
            text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore")
        except OSError:
            continue
        if "Raspberry Pi" not in text:
            return None
        # "Raspberry Pi 5 Model B", "Raspberry Pi 500", "Raspberry Pi Compute Module 5".
        return "pisp" if re.search(r"Raspberry Pi (Compute Module )?5", text) else "vc4"
    return None


def tuning_search_dirs() -> tuple[Path, ...]:
    """Folders searched for a tuning file on this computer."""
    pipeline = pi_pipeline()
    pipelines = (pipeline,) if pipeline else PIPELINES
    return tuple(root / name for name in pipelines for root in TUNING_ROOTS)


def _detect_sensor_from_model(model: Any) -> Optional[str]:
    text = normalize_camera_model(model)
    info = get_camera_model_info(text)
    if info is not None and info.sensor:
        return info.sensor
    if "imx477" in text or "hq" in text:
        return "imx477"
    if "imx708" in text or "module3" in text:
        return "imx708"
    if "ov64a40" in text or "owlsight" in text:
        return "ov64a40"
    if "imx219" in text:
        return "imx219"
    return None


def _detect_sensor_from_resolution(width: Any, height: Any) -> Optional[str]:
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return None
    if (w, h) == (4056, 3040):
        return "imx477"
    if (w, h) == (4608, 2592):
        return "imx708"
    if (w, h) == (9248, 6944):
        return "ov64a40"
    return None


def _sensor_from_camera(name: Any) -> Optional[str]:
    """The sensor name the connected camera reports, without a ``_noir`` ending.

    libcamera names its tuning files after this name (``imx477``,
    ``imx708_wide``), so it leads straight to the right file.
    """
    text = str(name or "").strip().lower()
    if text.endswith(".json"):
        text = text[: -len(".json")]
    if text.endswith("_noir"):
        text = text[: -len("_noir")]
    return text if re.fullmatch(r"[a-z0-9_]+", text) else None


def infrared_enabled(config: Any) -> bool:
    """Whether the NoIR tuning file is asked for.

    ``camera.infrared`` decides, whatever the camera model: an ordinary camera
    with its infrared filter taken off needs NoIR tuning as much as a camera
    sold as NoIR. Only a config that leaves the setting out falls back to the
    model's name.
    """
    camera_cfg = _camera(config)
    chosen = _parse_bool(camera_cfg.get("infrared"))
    if chosen is not None:
        return chosen
    return normalize_camera_model(camera_cfg.get("model", "auto")) in {"hq_noir", "module3_noir"}


def _find_tuning_file(candidate: str) -> Optional[Path]:
    value = str(candidate or "").strip()
    if not value:
        return None

    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()

    basename = Path(value).name
    for directory in tuning_search_dirs():
        path = directory / basename
        if path.is_file():
            return path.resolve()
    return None


def inspect_camera_tuning_resolution(config: dict[str, Any], detected_sensor: Any = None) -> Dict[str, Any]:
    """How the tuning file is chosen for ``config``, for the recorder and for reports.

    ``detected_sensor`` is the sensor name the connected camera reports
    (``camera_select.selected_camera_model``), or ``None`` when no camera has
    been asked.
    """
    camera_cfg = _camera(config)
    infrared = infrared_enabled(config)
    info: Dict[str, Any] = {
        "source": "default",
        "requested": None,
        "resolved": None,
        "resolved_path": None,
        "camera_model": str(camera_cfg.get("model", "auto")),
        "camera_profile": configured_profile_name(config) if isinstance(config, dict) else "custom",
        "infrared": infrared,
        "sensor": None,
        "sensor_source": None,
        "missing": None,
    }

    explicit = camera_cfg.get("tuning_file")
    if explicit not in (None, ""):
        found = _find_tuning_file(str(explicit))
        info.update(
            source="explicit",
            requested=str(explicit),
            resolved=str(found) if found else str(explicit),
            resolved_path=str(found) if found else None,
        )
        return info

    sensor = _sensor_from_camera(detected_sensor)
    sensor_source = SENSOR_SOURCE_CAMERA if sensor else None
    if sensor is None:
        sensor = _detect_sensor_from_model(camera_cfg.get("model", "auto"))
        sensor_source = SENSOR_SOURCE_MODEL if sensor else None
    if sensor is None:
        sensor = _detect_sensor_from_resolution(camera_cfg.get("width"), camera_cfg.get("height"))
        sensor_source = SENSOR_SOURCE_SIZE if sensor else None
    if sensor is None:
        return info
    info.update(sensor=sensor, sensor_source=sensor_source)

    basename = f"{sensor}{'_noir' if infrared else ''}.json"
    found = _find_tuning_file(basename)
    if found is not None:
        info.update(source="auto", resolved=str(found), resolved_path=str(found))
        return info

    # Not found. Where this computer's tuning folder can be seen, the file is
    # really absent: leave the camera on its own standard tuning rather than
    # fail to open it. (The OwlSight has no file in the usual folders at all.)
    # Elsewhere, pass the name on and let the camera library look for it.
    folders_present = any(directory.is_dir() for directory in tuning_search_dirs())
    if folders_present or sensor == "ov64a40":
        info["missing"] = basename
        return info
    info.update(source="auto", resolved=basename)
    return info


def resolve_camera_tuning_file(config: dict[str, Any], detected_sensor: Any = None) -> Optional[str]:
    """The tuning file to open the camera with, or ``None`` for the camera's own standard tuning.

    Precedence:
    1. ``camera.tuning_file`` (named by hand).
    2. The file for the sensor (connected camera, else ``camera.model``, else
       the recording size), in its NoIR version when ``camera.infrared`` is true.
    3. ``None``.
    """
    return inspect_camera_tuning_resolution(config, detected_sensor).get("resolved")


def describe_tuning(config: dict[str, Any], detected_sensor: Any = None) -> str:
    """The tuning choice in words, for a status line."""
    info = inspect_camera_tuning_resolution(config, detected_sensor)
    if info["source"] == "explicit":
        return f"tuning file {Path(str(info['requested'])).name} (named in the config)"
    kind = "NoIR tuning" if info["infrared"] else "standard tuning"
    # A sensor guessed from the recording size is only a guess: the camera is
    # asked when it is opened, and its answer decides.
    known = info["sensor_source"] in {SENSOR_SOURCE_CAMERA, SENSOR_SOURCE_MODEL}
    if info["resolved"] and known:
        return f"{kind} ({Path(str(info['resolved'])).name})"
    if info["missing"] and known:
        if info["infrared"]:
            return f"standard tuning, because no NoIR file ({info['missing']}) is installed for this camera"
        return "standard tuning"
    if info["infrared"]:
        return "NoIR tuning (the connected camera's NoIR file, chosen when the camera is opened)"
    return "standard tuning"


def release_camera_manager(picamera2_class: Any) -> bool:
    """Let go of libcamera's camera manager, if no camera is open in this process.

    libcamera chooses each camera's tuning file when its camera manager
    starts, and the manager starts the first time the cameras are listed.
    Picamera2 then keeps that manager until a camera is closed. BumbleBox
    lists the cameras before opening one, so without this a tuning file named
    at opening time arrives too late and is ignored.

    Returns whether a manager was let go.
    """
    try:
        manager = picamera2_class._cm
        if getattr(manager, "cameras", None) or getattr(manager, "running", False):
            return False  # a camera is open here; its manager has to stay
        if vars(manager).get("_cms") is None:
            return False
        manager._cms = None
        gc.collect()
        return True
    except Exception:
        return False


def open_picamera2(
    picamera2_class: Any,
    tuning_file: Optional[str],
    camera_kwargs: Optional[Dict[str, Any]] = None,
    notes: Optional[list] = None,
) -> Any:
    """Open a camera so that ``tuning_file`` is the tuning libcamera really uses.

    See ``release_camera_manager`` for why the manager is let go first.

    If the camera cannot be opened with the tuning file but can without it,
    it is opened on its own standard tuning and a line saying so is printed
    and added to ``notes``: a recording with standard tuning is worth more
    than no recording. If it cannot be opened either way, the first error is
    raised.
    """
    kwargs = dict(camera_kwargs or {})

    def attempt(name: Optional[str]) -> Any:
        release_camera_manager(picamera2_class)
        try:
            if name:
                tuning = picamera2_class.load_tuning_file(str(name))
                return picamera2_class(tuning=tuning, **kwargs)
            return picamera2_class(**kwargs)
        finally:
            # Picamera2 hands the tuning to libcamera through this variable
            # and leaves it naming a temporary file it has already deleted.
            # The next listing of cameras would start libcamera with that
            # missing file.
            os.environ.pop(TUNING_ENV, None)

    if not tuning_file:
        return attempt(None)
    try:
        return attempt(str(tuning_file))
    except Exception as first_error:
        try:
            camera = attempt(None)
        except Exception:
            raise first_error
        message = (
            f"The camera could not be opened with tuning file '{tuning_file}' ({first_error}). "
            "It was opened with its standard tuning instead."
        )
        print(f"[capture] {message}", flush=True)
        if notes is not None:
            notes.append(message)
        return camera
