from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


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
    infrared_bool = _parse_bool(infrared)
    if infrared_bool is not None:
        return bool(infrared_bool)

    model_text = str(model or "").strip().lower()
    if model_text in {"hq_noir", "module3_noir"}:
        return True
    if model_text in {"hq", "module3", "module3_standard", "module3_wide"}:
        return False
    return False


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


def resolve_camera_tuning_file(config: dict[str, Any]) -> Optional[str]:
    """Resolve a tuning file path/name from BumbleBox V2 config.

    Precedence:
    1. camera.tuning_file (explicit override)
    2. Derived default from camera.model (+ camera.infrared), with resolution hint fallback.
    3. None (libcamera default tuning).
    """
    camera_cfg = config.get("camera", {}) if isinstance(config, dict) else {}
    explicit = camera_cfg.get("tuning_file")
    if explicit not in (None, ""):
        found = _find_tuning_file(str(explicit))
        return str(found) if found else str(explicit)

    camera_model = camera_cfg.get("model", "auto")
    sensor = _detect_sensor_from_model(camera_model)
    if sensor is None:
        sensor = _detect_sensor_from_resolution(
            camera_cfg.get("width"),
            camera_cfg.get("height"),
        )
    if sensor is None:
        return None

    use_noir = _is_noir_variant(camera_model, camera_cfg.get("infrared"))
    basename = f"{sensor}{'_noir' if use_noir else ''}.json"
    found = _find_tuning_file(basename)
    return str(found) if found else basename


def inspect_camera_tuning_resolution(config: dict[str, Any]) -> Dict[str, Any]:
    """Return details about how camera tuning is resolved for diagnostics/UI."""
    camera_cfg = config.get("camera", {}) if isinstance(config, dict) else {}
    explicit = camera_cfg.get("tuning_file")
    explicit_text = None if explicit in (None, "") else str(explicit)

    resolved = resolve_camera_tuning_file(config)
    resolved_path: Optional[Path] = None
    if resolved:
        resolved_path = _find_tuning_file(str(resolved))

    if explicit_text is not None:
        source = "explicit"
    elif resolved is not None:
        source = "auto"
    else:
        source = "default"

    return {
        "source": source,
        "requested": explicit_text,
        "resolved": resolved,
        "resolved_path": str(resolved_path) if resolved_path else None,
        "camera_model": str(camera_cfg.get("model", "auto")),
        "infrared": _parse_bool(camera_cfg.get("infrared")),
    }
