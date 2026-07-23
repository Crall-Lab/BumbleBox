from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
import time
from typing import Any, Optional


VALID_AUTOFOCUS_MODES = {"default", "manual", "auto", "continuous"}
VALID_AUTOFOCUS_RANGES = {"normal", "macro", "full"}
VALID_AUTOFOCUS_SPEEDS = {"normal", "fast"}


@dataclass(frozen=True)
class AutofocusSettings:
    mode: str
    lens_position: Optional[float]
    lock_after_warmup: bool
    autofocus_range: str
    autofocus_speed: str


@dataclass(frozen=True)
class AutofocusPreflightSettings:
    enabled: bool
    width: int
    height: int
    timeout_seconds: float
    stable_frames: int


@dataclass
class AutofocusPreflightResult:
    performed: bool
    selected_lens_position: Optional[float]
    selection_reason: str
    best_focus_score: Optional[float]
    final_af_state: str
    frames_observed: int
    elapsed_seconds: float
    requested_width: int
    requested_height: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_autofocus_mode(value: Any) -> str:
    mode = str(value or "default").strip().lower()
    return mode or "default"


def normalize_autofocus_range(value: Any) -> str:
    autofocus_range = str(value or "normal").strip().lower()
    return autofocus_range or "normal"


def normalize_autofocus_speed(value: Any) -> str:
    autofocus_speed = str(value or "normal").strip().lower()
    return autofocus_speed or "normal"


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
        autofocus_range=normalize_autofocus_range(camera_cfg.get("autofocus_range", "normal")),
        autofocus_speed=normalize_autofocus_speed(camera_cfg.get("autofocus_speed", "normal")),
    )


def autofocus_preflight_settings(config: dict[str, Any]) -> AutofocusPreflightSettings:
    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera", {}), dict) else {}
    return AutofocusPreflightSettings(
        enabled=bool(camera_cfg.get("autofocus_preflight_enabled", False)),
        width=int(camera_cfg.get("autofocus_preflight_width", 1920)),
        height=int(camera_cfg.get("autofocus_preflight_height", 1440)),
        timeout_seconds=float(camera_cfg.get("autofocus_preflight_timeout_seconds", 8.0)),
        stable_frames=int(camera_cfg.get("autofocus_preflight_stable_frames", 3)),
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
    if _control_is_advertised(picam2, "AfRange") and hasattr(controls, "AfRangeEnum"):
        range_names = {"normal": "Normal", "macro": "Macro", "full": "Full"}
        requested_controls["AfRange"] = getattr(
            controls.AfRangeEnum,
            range_names[settings.autofocus_range],
        )
    if _control_is_advertised(picam2, "AfSpeed") and hasattr(controls, "AfSpeedEnum"):
        speed_names = {"normal": "Normal", "fast": "Fast"}
        requested_controls["AfSpeed"] = getattr(
            controls.AfSpeedEnum,
            speed_names[settings.autofocus_speed],
        )
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


def _autofocus_state_name(value: Any) -> str:
    if value is None:
        return "unknown"
    name = getattr(value, "name", None)
    if name:
        return str(name).strip().lower()
    if isinstance(value, int):
        return {
            0: "idle",
            1: "scanning",
            2: "focused",
            3: "failed",
        }.get(value, str(value))
    text = str(value).strip().lower()
    for state in ("idle", "scanning", "focused", "failed"):
        if state in text:
            return state
    return text or "unknown"


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_autofocus_preflight(
    config: dict[str, Any],
    picam2: Any,
    controls: Any,
    *,
    notes: Optional[list[str]] = None,
    strict: bool = True,
) -> AutofocusPreflightResult:
    settings = autofocus_preflight_settings(config)
    if not settings.enabled:
        return AutofocusPreflightResult(
            performed=False,
            selected_lens_position=None,
            selection_reason="disabled",
            best_focus_score=None,
            final_af_state="not_run",
            frames_observed=0,
            elapsed_seconds=0.0,
            requested_width=settings.width,
            requested_height=settings.height,
            notes=[],
        )

    result_notes: list[str] = []
    started = False
    operation_start = time.perf_counter()
    scan_start = operation_start
    frames_observed = 0
    final_state = "unknown"
    best_focus_score: Optional[float] = None
    best_lens_position: Optional[float] = None
    focused_positions: list[float] = []
    focused_scores: list[float] = []
    saw_scanning = False
    selection_reason = "unavailable"

    print(
        "[camera] Autofocus preflight: "
        f"{settings.width}x{settings.height}, timeout={settings.timeout_seconds:g}s",
        flush=True,
    )
    try:
        preflight_config = picam2.create_preview_configuration(
            main={"size": (settings.width, settings.height)}
        )
        picam2.align_configuration(preflight_config)
        picam2.configure(preflight_config)
        if not apply_autofocus_before_start(
            config,
            picam2,
            controls,
            notes=result_notes,
            strict=strict,
        ):
            raise RuntimeError("Could not initialize autofocus preflight.")

        picam2.start()
        started = True
        scan_start = time.perf_counter()
        start_autofocus_after_camera_start(
            config,
            picam2,
            controls,
            notes=result_notes,
            strict=strict,
        )

        while (time.perf_counter() - scan_start) < settings.timeout_seconds:
            metadata = picam2.capture_metadata()
            frames_observed += 1
            final_state = _autofocus_state_name(
                metadata.get("AfState") if isinstance(metadata, dict) else None
            )
            lens_position = _optional_float(
                metadata.get("LensPosition") if isinstance(metadata, dict) else None
            )
            focus_score = _optional_float(
                metadata.get("FocusFoM") if isinstance(metadata, dict) else None
            )
            if final_state == "scanning":
                saw_scanning = True

            if (
                lens_position is not None
                and focus_score is not None
                and (best_focus_score is None or focus_score > best_focus_score)
            ):
                best_focus_score = focus_score
                best_lens_position = lens_position

            if final_state == "focused" and lens_position is not None:
                focused_positions.append(lens_position)
                if focus_score is not None:
                    focused_scores.append(focus_score)
                if len(focused_positions) >= settings.stable_frames:
                    break
            else:
                focused_positions.clear()
                focused_scores.clear()

            if final_state == "failed" and saw_scanning and best_lens_position is not None:
                break
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Autofocus preflight failed: {exc}") from exc
        result_notes.append(f"Autofocus preflight failed: {exc}")
    finally:
        if started:
            try:
                picam2.stop()
            except Exception as exc:
                result_notes.append(f"Could not stop autofocus preflight stream cleanly: {exc}")

    if len(focused_positions) >= settings.stable_frames:
        selected_lens_position = float(median(focused_positions[-settings.stable_frames :]))
        selection_reason = "stable_focused_state"
        if focused_scores:
            best_focus_score = max(
                best_focus_score if best_focus_score is not None else focused_scores[0],
                max(focused_scores),
            )
    elif best_lens_position is not None:
        selected_lens_position = best_lens_position
        selection_reason = (
            "highest_focus_score_after_failed_state"
            if final_state == "failed" and saw_scanning
            else "highest_focus_score_before_timeout"
        )
    else:
        selected_lens_position = None

    elapsed_seconds = max(0.0, time.perf_counter() - operation_start)
    if selected_lens_position is None:
        message = (
            "Autofocus preflight did not report a usable LensPosition/FocusFoM pair; "
            "the full-resolution stream cannot be focus-locked."
        )
        result_notes.append(message)
        if strict:
            raise RuntimeError(message)
    else:
        result_notes.append(
            "Autofocus preflight selected lens position "
            f"{selected_lens_position:.3f} using {selection_reason}."
        )
        print(
            "[camera] Autofocus preflight selected "
            f"lens={selected_lens_position:.3f}, state={final_state}, "
            f"score={best_focus_score if best_focus_score is not None else 'n/a'}, "
            f"reason={selection_reason}, elapsed={elapsed_seconds:.2f}s",
            flush=True,
        )

    if notes is not None:
        notes.extend(result_notes)
    return AutofocusPreflightResult(
        performed=True,
        selected_lens_position=selected_lens_position,
        selection_reason=selection_reason,
        best_focus_score=best_focus_score,
        final_af_state=final_state,
        frames_observed=frames_observed,
        elapsed_seconds=round(elapsed_seconds, 3),
        requested_width=settings.width,
        requested_height=settings.height,
        notes=result_notes,
    )


def apply_preflight_lens_lock(
    result: AutofocusPreflightResult,
    picam2: Any,
    controls: Any,
    *,
    notes: Optional[list[str]] = None,
    strict: bool = True,
) -> bool:
    if not result.performed:
        return False
    if result.selected_lens_position is None:
        return _focus_failure(
            "Autofocus preflight did not select a lens position.",
            notes=notes,
            strict=strict,
        )
    try:
        picam2.set_controls(
            {
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": float(result.selected_lens_position),
            }
        )
    except Exception as exc:
        return _focus_failure(
            f"Could not apply the autofocus preflight lens position: {exc}",
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
