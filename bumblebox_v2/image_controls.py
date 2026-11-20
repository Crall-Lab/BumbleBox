"""Camera settings that shape the picture itself.

Sensor gain, white balance, contrast and noise reduction are applied inside
the camera, before a frame reaches BumbleBox. They therefore change the
recordings, the live monitor and what the tag detector sees, all alike.

Left on automatic (the default), the camera picks the gain and white balance
frame by frame. Under infrared or red light that choice can drift, and the
white balance in particular is guessing at colours that are not there. Fixed
values keep one recording comparable with the next.

Nothing here imports the camera library: the values are worked out from the
config, and ``apply_image_controls`` hands them to an open camera.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

AUTO = "auto"
DEFAULT_CONTRAST = 1.0
# Limits for the config. A camera may accept less (an HQ camera's gain stops
# near 22); it then uses the nearest value it can.
GAIN_RANGE = (1.0, 64.0)
COLOUR_GAIN_RANGE = (0.1, 32.0)
CONTRAST_RANGE = (0.0, 32.0)

# camera.noise_reduction -> the camera library's name for the mode. "Auto"
# sends nothing, which leaves the default of the camera setup BumbleBox uses:
# Picamera2's preview configuration asks for "Minimal".
NOISE_REDUCTION_MODES = ("Auto", "Off", "Minimal", "Fast", "HighQuality")
# What default_config.yaml gives a new setup. A config that leaves the key out
# of a hand-built dictionary still reads as "Auto": nothing is sent.
DEFAULT_NOISE_REDUCTION = "HighQuality"
NOISE_REDUCTION_WORDS = {
    "Auto": "left to the camera",
    "Off": "off",
    "Minimal": "minimal",
    "Fast": "fast",
    "HighQuality": "high quality",
}
_NOISE_REDUCTION_ALIASES = {
    "auto": "Auto",
    "default": "Auto",
    "off": "Off",
    "none": "Off",
    "minimal": "Minimal",
    "fast": "Fast",
    "highquality": "HighQuality",
    "hq": "HighQuality",
}


class ImageSettingError(ValueError):
    """A picture setting in the config that cannot be used."""


def _camera(config: Dict[str, Any]) -> Dict[str, Any]:
    camera = config.get("camera", {}) if isinstance(config, dict) else {}
    return camera if isinstance(camera, dict) else {}


def _is_auto(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {AUTO, ""})


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImageSettingError(f"{name} must be a number, got: {value!r}")
    number = float(value)
    if not (low <= number <= high):
        raise ImageSettingError(f"{name} must be between {low:g} and {high:g}, got: {number:g}")
    return number


def analogue_gain(config: Dict[str, Any]) -> Optional[float]:
    """The fixed sensor gain, or ``None`` when the camera chooses it."""
    value = _camera(config).get("analogue_gain", AUTO)
    if _is_auto(value):
        return None
    return _number(value, "camera.analogue_gain", *GAIN_RANGE)


def colour_gains(config: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The fixed white balance as (red gain, blue gain), or ``None`` when the camera chooses it."""
    value = _camera(config).get("colour_gains", AUTO)
    if _is_auto(value):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ImageSettingError(
            f'camera.colour_gains must be "auto" or two numbers [red, blue], got: {value!r}'
        )
    red = _number(value[0], "camera.colour_gains red", *COLOUR_GAIN_RANGE)
    blue = _number(value[1], "camera.colour_gains blue", *COLOUR_GAIN_RANGE)
    return red, blue


def contrast(config: Dict[str, Any]) -> float:
    """Contrast applied in the camera. 1.0 is the camera's own picture."""
    value = _camera(config).get("contrast", DEFAULT_CONTRAST)
    if value is None:
        return DEFAULT_CONTRAST
    return _number(value, "camera.contrast", *CONTRAST_RANGE)


def noise_reduction(config: Dict[str, Any]) -> str:
    """The noise reduction mode by its camera-library name; ``"Auto"`` leaves the camera's default."""
    value = _camera(config).get("noise_reduction", "Auto")
    if value is None:
        return "Auto"
    key = str(value).strip().lower().replace("_", "").replace(" ", "").replace("-", "")
    mode = _NOISE_REDUCTION_ALIASES.get(key)
    if mode is None:
        raise ImageSettingError(
            "camera.noise_reduction must be one of " + ", ".join(NOISE_REDUCTION_MODES) + f", got: {value!r}"
        )
    return mode


def validate_image_settings(config: Dict[str, Any]) -> None:
    """Raise ``ImageSettingError`` for a picture setting that cannot be used."""
    analogue_gain(config)
    colour_gains(config)
    contrast(config)
    noise_reduction(config)


def requested_controls(config: Dict[str, Any]) -> List[Tuple[str, Any, str]]:
    """The controls these settings ask for: (control name, value, config key).

    Only settings moved away from "leave it to the camera" produce a control.
    A new setup asks for one thing, high quality noise reduction. The noise
    reduction mode is given by name; ``apply_image_controls`` turns it into
    the camera library's value.
    """
    controls: List[Tuple[str, Any, str]] = []
    gain = analogue_gain(config)
    if gain is not None:
        controls.append(("AnalogueGain", gain, "camera.analogue_gain"))
    gains = colour_gains(config)
    if gains is not None:
        # Fixed gains switch automatic white balance off; saying so as well is harmless.
        controls.append(("AwbEnable", False, "camera.colour_gains"))
        controls.append(("ColourGains", gains, "camera.colour_gains"))
    level = contrast(config)
    if abs(level - DEFAULT_CONTRAST) > 1e-6:
        controls.append(("Contrast", level, "camera.contrast"))
    mode = noise_reduction(config)
    if mode != "Auto":
        controls.append(("NoiseReductionMode", mode, "camera.noise_reduction"))
    return controls


def apply_image_controls(
    config: Dict[str, Any],
    picam2: Any,
    libcamera_controls: Any = None,
    notes: Optional[List[str]] = None,
) -> List[str]:
    """Hand the picture settings to an open camera. Returns the controls it accepted.

    Each control is sent on its own, so a camera that lacks one still gets the
    others; what was refused is added to ``notes`` in words.
    """
    applied: List[str] = []
    for name, value, key in requested_controls(config):
        try:
            if name == "NoiseReductionMode":
                if libcamera_controls is None:
                    from libcamera import controls as libcamera_controls  # type: ignore[no-redef]
                value = getattr(libcamera_controls.draft.NoiseReductionModeEnum, str(value))
            picam2.set_controls({name: value})
            applied.append(name)
        except Exception as exc:
            if notes is not None:
                notes.append(f"The camera did not accept {name} ({key}): {exc}")
    return applied


def describe_settings(config: Dict[str, Any]) -> str:
    """One line saying how the picture settings stand, for a status line or a report."""
    gain = analogue_gain(config)
    gains = colour_gains(config)
    parts = [
        f"sensor gain {'automatic' if gain is None else f'fixed at {gain:g}'}",
        "white balance automatic" if gains is None else f"white balance fixed (red {gains[0]:g}, blue {gains[1]:g})",
    ]
    level = contrast(config)
    if abs(level - DEFAULT_CONTRAST) > 1e-6:
        parts.append(f"contrast {level:g}")
    parts.append(f"noise reduction {NOISE_REDUCTION_WORDS[noise_reduction(config)]}")
    return "; ".join(parts)


def camera_readout(metadata: Any) -> str:
    """What the camera is using right now, from a frame's metadata.

    These are the numbers to copy into the fixed settings: let the camera
    settle on automatic, read them here, then fix them.
    """
    if not isinstance(metadata, dict):
        return ""
    parts: List[str] = []
    gain = metadata.get("AnalogueGain")
    if isinstance(gain, (int, float)) and not isinstance(gain, bool):
        parts.append(f"sensor gain {float(gain):.2f}")
    gains = metadata.get("ColourGains")
    if isinstance(gains, (list, tuple)) and len(gains) == 2:
        try:
            parts.append(f"white balance red {float(gains[0]):.2f}, blue {float(gains[1]):.2f}")
        except (TypeError, ValueError):
            pass
    return "  |  ".join(parts)
