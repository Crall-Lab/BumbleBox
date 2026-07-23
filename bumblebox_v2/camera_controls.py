from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


VALID_AUTOFOCUS_MODES = {"default", "manual", "auto", "continuous"}


@dataclass(frozen=True)
class AutofocusSettings:
    mode: str
    lens_position: Optional[float]
    lock_after_warmup: bool


def normalize_autofocus_mode(value: Any) -> str:
    mode = str(value or "default").strip().lower()
    return mode or "default"


def autofocus_settings(config: dict[str, Any]) -> AutofocusSettings:
    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera", {}), dict) else {}
    raw_lens_position = camera_cfg.get("lens_position")
    lens_position = (
        None
        if raw_lens_position is None or str(raw_lens_position).strip() == ""
        else float(raw_lens_position)
    )
    return AutofocusSettings(
        mode=normalize_autofocus_mode(camera_cfg.get("autofocus_mode", "default")),
        lens_position=lens_position,
        lock_after_warmup=bool(camera_cfg.get("focus_lock_after_warmup", False)),
    )


def _control_is_advertised(picam2: Any, control_name: str) -> bool:
    try:
        advertised = getattr(picam2, "camera_controls", None)
    except Exception:
        return True
    if isinstance(advertised, dict) and advertised:
        return control_name in advertised
    return True


def _focus_failure(message: str, *, notes: Optional[list[str]], strict: bool) -> bool:
    if notes is not None:
        notes.append(message)
    if strict:
        raise RuntimeError(message)
    return False


def apply_autofocus_before_start(
    config: dict[str, Any],
    picam2: Any,
    controls: Any,
    *,
    notes: Optional[list[str]] = None,
    strict: bool = False,
) -> bool:
    settings = autofocus_settings(config)
    if settings.mode == "default":
        return True

    if not _control_is_advertised(picam2, "AfMode"):
        return _focus_failure(
            f"Camera does not advertise AfMode; autofocus_mode='{settings.mode}' cannot be applied.",
            notes=notes,
            strict=strict,
        )

    mode_names = {
        "manual": "Manual",
        "auto": "Auto",
        "continuous": "Continuous",
    }
    enum_name = mode_names.get(settings.mode)
    if enum_name is None:
        return _focus_failure(
            f"Unsupported autofocus mode: {settings.mode}",
            notes=notes,
            strict=strict,
        )

    requested_controls: dict[str, Any] = {
        "AfMode": getattr(controls.AfModeEnum, enum_name),
    }
    if settings.mode == "manual" and settings.lens_position is not None:
        if not _control_is_advertised(picam2, "LensPosition"):
            return _focus_failure(
                "Camera does not advertise LensPosition; manual lens position cannot be applied.",
                notes=notes,
                strict=strict,
            )
        requested_controls["LensPosition"] = settings.lens_position

    try:
        picam2.set_controls(requested_controls)
    except Exception as exc:
        return _focus_failure(
            f"Could not apply autofocus_mode='{settings.mode}': {exc}",
            notes=notes,
            strict=strict,
        )
    return True


def start_autofocus_after_camera_start(
    config: dict[str, Any],
    picam2: Any,
    controls: Any,
    *,
    notes: Optional[list[str]] = None,
    strict: bool = False,
) -> bool:
    settings = autofocus_settings(config)
    if settings.mode != "auto":
        return True

    if not _control_is_advertised(picam2, "AfTrigger"):
        return _focus_failure(
            "Camera does not advertise AfTrigger; one-shot autofocus cannot be started.",
            notes=notes,
            strict=strict,
        )
    try:
        picam2.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
    except Exception as exc:
        return _focus_failure(
            f"Could not trigger one-shot autofocus: {exc}",
            notes=notes,
            strict=strict,
        )
    return True


def lock_autofocus_after_warmup(
    config: dict[str, Any],
    picam2: Any,
    controls: Any,
    *,
    notes: Optional[list[str]] = None,
    strict: bool = False,
) -> Optional[float]:
    settings = autofocus_settings(config)
    if not settings.lock_after_warmup:
        return None

    try:
        metadata = picam2.capture_metadata()
        lens_position = metadata.get("LensPosition") if isinstance(metadata, dict) else None
        if lens_position is None:
            _focus_failure(
                "Could not lock autofocus: capture metadata did not include LensPosition.",
                notes=notes,
                strict=strict,
            )
            return None
        lens_position = float(lens_position)
        picam2.set_controls(
            {
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": lens_position,
            }
        )
    except Exception as exc:
        _focus_failure(
            f"Could not lock autofocus after warmup: {exc}",
            notes=notes,
            strict=strict,
        )
        return None

    if notes is not None:
        notes.append(f"Autofocus locked after warmup at lens position {lens_position:.3f}.")
    return lens_position
