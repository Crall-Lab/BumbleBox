from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


TUNING_SEARCH_DIRS = (
    Path("/usr/share/libcamera/ipa/rpi/pisp"),  # Raspberry Pi 5 pipeline.
    Path("/usr/share/libcamera/ipa/rpi/vc4"),   # Raspberry Pi 4 and earlier pipeline.
)

CAMERA_MODEL_TO_SENSOR = {
    "hq": "imx477",
    "hq_noir": "imx477",
    "module3": "imx708",
    "module3_standard": "imx708",
    "module3_wide": "imx708",
    "module3_noir": "imx708",
}


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


def _detect_sensor_from_model(model: Any) -> Optional[str]:
    text = str(model or "").strip().lower()
    if text in CAMERA_MODEL_TO_SENSOR:
        return CAMERA_MODEL_TO_SENSOR[text]
    if "imx477" in text or "hq" in text:
        return "imx477"
    if "imx708" in text or "module3" in text:
        return "imx708"
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
    return None


def _is_noir_variant(model: Any, infrared: Any) -> bool:
    model_text = str(model or "").strip().lower()
    if model_text in {"hq_noir", "module3_noir"}:
        return True
    if model_text in {"hq", "module3", "module3_standard", "module3_wide"}:
        return False
    infrared_bool = _parse_bool(infrared)
    return bool(infrared_bool) if infrared_bool is not None else False


def _find_tuning_file(candidate: str) -> Optional[Path]:
    value = str(candidate or "").strip()
    if not value:
        return None

    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()

    basename = Path(value).name
    for directory in TUNING_SEARCH_DIRS:
        path = directory / basename
        if path.is_file():
            return path.resolve()
    return None


def resolve_tuning_file(
    *,
    explicit_tuning_file: Any = None,
    camera_model: Any = None,
    width: Any = None,
    height: Any = None,
    infrared: Any = None,
    default_sensor: str = "imx477",
) -> str:
    """Resolve legacy tuning file path/name with explicit override precedence."""
    if explicit_tuning_file not in (None, ""):
        explicit = str(explicit_tuning_file)
        found = _find_tuning_file(explicit)
        return str(found) if found else explicit

    sensor = _detect_sensor_from_model(camera_model)
    if sensor is None:
        sensor = _detect_sensor_from_resolution(width, height)
    if sensor is None:
        sensor = default_sensor

    use_noir = _is_noir_variant(camera_model, infrared)
    basename = f"{sensor}{'_noir' if use_noir else ''}.json"
    found = _find_tuning_file(basename)
    return str(found) if found else basename


def resolve_recording_tuning_from_config(config: dict[str, Any]) -> str:
    camera = config.get("camera_settings", {}) if isinstance(config, dict) else {}
    recording = config.get("recording_options", {}) if isinstance(config, dict) else {}
    return resolve_tuning_file(
        explicit_tuning_file=camera.get("tuning_file"),
        camera_model=camera.get("model"),
        width=camera.get("width"),
        height=camera.get("height"),
        infrared=recording.get("infrared_recording"),
        default_sensor="imx477",
    )
