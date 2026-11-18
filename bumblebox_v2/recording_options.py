"""Small recording choices that can be flipped without opening the Settings page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .camera_profiles import CAMERA_PROFILES
from .config import load_config, save_config


def recording_is_grayscale(config: Dict[str, Any]) -> bool:
    """Whether recordings made with this config are saved in grayscale."""
    camera = config.get("camera", {}) if isinstance(config.get("camera"), dict) else {}
    return bool(camera.get("monochrome_output", False))


@dataclass
class RecordingColorChange:
    grayscale: bool
    # Label of the camera profile that had to be left for the setting to hold, if any.
    left_profile: Optional[str] = None

    @property
    def message(self) -> str:
        text = f"Recordings from this camera will be saved in {'grayscale' if self.grayscale else 'colour'}."
        if self.left_profile:
            text += (
                f" The camera profile “{self.left_profile}” fixes that setting, so the camera now uses "
                "Custom settings with the same values otherwise."
            )
        return text


def set_recording_grayscale(config_path: str | Path, enabled: bool) -> RecordingColorChange:
    """Save whether this config's recordings are grayscale (``camera.monochrome_output``).

    A named camera profile re-applies its own value every time the config is
    saved. When that value is not the one wanted, the config is switched to
    the Custom profile, which keeps every other camera value as it is.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    camera = config.setdefault("camera", {})
    left_profile: Optional[str] = None
    key = str(camera.get("profile", "custom") or "custom")
    profile = CAMERA_PROFILES.get(key)
    if (
        profile is not None
        and key != "custom"
        and "monochrome_output" in profile.camera_values
        and bool(profile.camera_values["monochrome_output"]) != bool(enabled)
    ):
        camera["profile"] = "custom"
        left_profile = profile.label
    camera["monochrome_output"] = bool(enabled)
    save_config(config_path, config)
    return RecordingColorChange(grayscale=bool(enabled), left_profile=left_profile)
