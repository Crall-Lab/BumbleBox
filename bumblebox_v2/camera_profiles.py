from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CameraModelInfo:
    key: str
    label: str
    sensor: Optional[str]
    max_resolution: Optional[tuple[int, int]]
    supports_infrared: bool
    notes: str = ""


@dataclass(frozen=True)
class CameraProfile:
    key: str
    label: str
    description: str
    camera_values: Dict[str, Any]


CAMERA_MODELS: Dict[str, CameraModelInfo] = {
    "auto": CameraModelInfo(
        key="auto",
        label="Auto-detect",
        sensor=None,
        max_resolution=None,
        supports_infrared=True,
        notes="Use libcamera/Picamera2 defaults and resolution hints.",
    ),
    "hq": CameraModelInfo(
        key="hq",
        label="Raspberry Pi HQ Camera",
        sensor="imx477",
        max_resolution=(4056, 3040),
        supports_infrared=True,
        notes="HQ sensor. IR behavior depends on lens/filter configuration.",
    ),
    "hq_noir": CameraModelInfo(
        key="hq_noir",
        label="Raspberry Pi HQ Camera (NoIR/IR)",
        sensor="imx477",
        max_resolution=(4056, 3040),
        supports_infrared=True,
        notes="HQ sensor configured for IR/NoIR BumbleBox recording.",
    ),
    "module3": CameraModelInfo(
        key="module3",
        label="Raspberry Pi Camera Module 3",
        sensor="imx708",
        max_resolution=(4608, 2592),
        supports_infrared=True,
    ),
    "module3_standard": CameraModelInfo(
        key="module3_standard",
        label="Raspberry Pi Camera Module 3 Standard",
        sensor="imx708",
        max_resolution=(4608, 2592),
        supports_infrared=False,
    ),
    "module3_wide": CameraModelInfo(
        key="module3_wide",
        label="Raspberry Pi Camera Module 3 Wide",
        sensor="imx708",
        max_resolution=(4608, 2592),
        supports_infrared=False,
    ),
    "module3_noir": CameraModelInfo(
        key="module3_noir",
        label="Raspberry Pi Camera Module 3 NoIR",
        sensor="imx708",
        max_resolution=(4608, 2592),
        supports_infrared=True,
    ),
    "owlsight_64mp": CameraModelInfo(
        key="owlsight_64mp",
        label="Arducam OwlSight 64MP",
        sensor="ov64a40",
        max_resolution=(9248, 6944),
        supports_infrared=False,
        notes="Stock OwlSight/OV64A40 has an IR-cut filter; use visible illumination.",
    ),
}

CAMERA_MODEL_ALIASES = {
    "owlsight": "owlsight_64mp",
    "owl_sight": "owlsight_64mp",
    "arducam_owlsight": "owlsight_64mp",
    "ov64a40": "owlsight_64mp",
    "b0483": "owlsight_64mp",
}

CAMERA_PROFILES: Dict[str, CameraProfile] = {
    "custom": CameraProfile(
        key="custom",
        label="Custom",
        description="Use the explicit camera fields without applying a preset.",
        camera_values={},
    ),
    "hq_reference": CameraProfile(
        key="hq_reference",
        label="HQ Reference",
        description="Known-good HQ/NoIR comparison profile at full HQ resolution and 7 fps.",
        camera_values={
            "model": "hq_noir",
            "width": 4056,
            "height": 3040,
            "fps_target": 7.0,
            "shutter_us": 2500,
            "codec": "mp4",
            "infrared": True,
            "monochrome_output": False,
            "noise_reduction": "Auto",
            "tuning_file": None,
            "digital_zoom": None,
            "autofocus_mode": "default",
            "lens_position": None,
            "focus_lock_after_warmup": False,
            "autofocus_range": "normal",
            "autofocus_speed": "normal",
            "autofocus_preflight_enabled": False,
            "autofocus_preflight_width": 1920,
            "autofocus_preflight_height": 1440,
            "autofocus_preflight_timeout_seconds": 8.0,
            "autofocus_preflight_stable_frames": 3,
        },
    ),
    "owlsight_reference": CameraProfile(
        key="owlsight_reference",
        label="OwlSight Reference",
        description=(
            "Arducam OwlSight comparison profile using the same 4056x3040 @ 7 fps "
            "capture envelope as the HQ reference, with a low-resolution autofocus "
            "preflight followed by a locked recording focus."
        ),
        camera_values={
            "model": "owlsight_64mp",
            "width": 4056,
            "height": 3040,
            "fps_target": 7.0,
            "shutter_us": 2500,
            "codec": "mp4",
            "infrared": False,
            "monochrome_output": False,
            "noise_reduction": "Auto",
            "tuning_file": None,
            "digital_zoom": None,
            "autofocus_mode": "continuous",
            "lens_position": None,
            "focus_lock_after_warmup": False,
            "autofocus_range": "normal",
            "autofocus_speed": "normal",
            "autofocus_preflight_enabled": True,
            "autofocus_preflight_width": 1920,
            "autofocus_preflight_height": 1440,
            "autofocus_preflight_timeout_seconds": 8.0,
            "autofocus_preflight_stable_frames": 3,
        },
    ),
}


def normalize_camera_model(model: Any) -> str:
    value = str(model or "auto").strip().lower()
    if not value:
        return "auto"
    return CAMERA_MODEL_ALIASES.get(value, value)


def normalize_camera_profile(profile: Any) -> str:
    value = str(profile or "custom").strip().lower()
    if not value:
        return "custom"
    return value


def camera_model_choices() -> list[str]:
    return list(CAMERA_MODELS.keys())


def camera_profile_choices() -> list[str]:
    return list(CAMERA_PROFILES.keys())


def get_camera_model_info(model: Any) -> CameraModelInfo | None:
    return CAMERA_MODELS.get(normalize_camera_model(model))


def get_camera_profile(profile: Any) -> CameraProfile | None:
    return CAMERA_PROFILES.get(normalize_camera_profile(profile))


def apply_camera_profile(config: Dict[str, Any], profile: Any | None = None) -> Dict[str, Any]:
    """Apply a named camera profile to a config copy and return the updated copy."""
    updated = deepcopy(config)
    camera_cfg = updated.setdefault("camera", {})
    selected = normalize_camera_profile(profile if profile is not None else camera_cfg.get("profile", "custom"))
    camera_cfg["profile"] = selected
    profile_info = get_camera_profile(selected)
    if profile_info is None or selected == "custom":
        if "model" in camera_cfg:
            camera_cfg["model"] = normalize_camera_model(camera_cfg.get("model"))
        return updated

    for key, value in profile_info.camera_values.items():
        camera_cfg[key] = deepcopy(value)
    camera_cfg["profile"] = selected
    camera_cfg["model"] = normalize_camera_model(camera_cfg.get("model"))
    return updated


def configured_profile_name(config: Dict[str, Any]) -> str:
    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera", {}), dict) else {}
    return normalize_camera_profile(camera_cfg.get("profile", "custom"))


def validate_camera_ir_compatibility(config: Dict[str, Any]) -> Optional[str]:
    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera", {}), dict) else {}
    model_info = get_camera_model_info(camera_cfg.get("model", "auto"))
    if model_info is None:
        return None
    if bool(camera_cfg.get("infrared", False)) and not model_info.supports_infrared:
        return (
            f"camera.model={model_info.key!r} is marked as not IR-capable, but camera.infrared=true. "
            f"{model_info.notes or 'Use visible illumination or choose an IR-capable camera profile.'}"
        )
    return None
