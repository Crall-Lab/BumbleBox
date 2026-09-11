from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    label: str
    description: str
    thermal_enabled: bool
    realsense_enabled: bool


HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "rgb_only": HardwareProfile(
        key="rgb_only",
        label="RGB camera only",
        description="Use an HQ, OwlSight, or other libcamera-compatible RGB camera.",
        thermal_enabled=False,
        realsense_enabled=False,
    ),
    "rgb_thermal": HardwareProfile(
        key="rgb_thermal",
        label="RGB + thermal",
        description="Record the RGB camera with the PureThermal/Lepton camera.",
        thermal_enabled=True,
        realsense_enabled=False,
    ),
    "rgb_depth": HardwareProfile(
        key="rgb_depth",
        label="RGB + RealSense depth",
        description="Use an RGB camera with a RealSense depth camera.",
        thermal_enabled=False,
        realsense_enabled=True,
    ),
    "multimodal": HardwareProfile(
        key="multimodal",
        label="RGB + thermal + RealSense",
        description="Enable all three acquisition systems for calibration and integration work.",
        thermal_enabled=True,
        realsense_enabled=True,
    ),
    "custom": HardwareProfile(
        key="custom",
        label="Custom hardware",
        description="Choose thermal and RealSense support independently.",
        thermal_enabled=False,
        realsense_enabled=False,
    ),
}


def normalize_hardware_profile(value: Any) -> str:
    normalized = str(value or "custom").strip().lower()
    return normalized or "custom"


def hardware_profile_choices() -> list[str]:
    return list(HARDWARE_PROFILES.keys())


def get_hardware_profile(value: Any) -> HardwareProfile | None:
    return HARDWARE_PROFILES.get(normalize_hardware_profile(value))


def apply_hardware_profile(config: Dict[str, Any], profile: Any) -> Dict[str, Any]:
    """Apply optional-device defaults without changing RGB camera settings."""
    updated = deepcopy(config)
    selected = normalize_hardware_profile(profile)
    profile_info = get_hardware_profile(selected)
    if profile_info is None:
        raise ValueError(f"Unknown hardware profile: {profile}")

    setup = updated.setdefault("setup", {})
    setup["hardware_profile"] = selected
    if selected != "custom":
        updated.setdefault("thermal", {})["enabled"] = profile_info.thermal_enabled
        updated.setdefault("realsense", {})["enabled"] = profile_info.realsense_enabled
    return updated
