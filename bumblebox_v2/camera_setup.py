from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Optional

from .camera_controls import (
    apply_autofocus_before_start,
    apply_preflight_lens_lock,
    autofocus_preflight_settings,
    autofocus_settings,
    AutofocusPreflightResult,
    lock_autofocus_after_warmup,
    run_autofocus_preflight,
    start_autofocus_after_camera_start,
)
from .camera_profiles import (
    apply_camera_profile,
    configured_profile_name,
    get_camera_model_info,
    validate_camera_ir_compatibility,
)
from .qt_env import sanitize_current_qt_env
from .tuning import inspect_camera_tuning_resolution, resolve_camera_tuning_file


PREVIEW_WINDOWS = {"QTGL", "QT", "DRM"}


@dataclass
class CameraPreviewResult:
    preview_seconds: float
    elapsed_seconds: float
    window: str
    width: int
    height: int
    shutter_us: int
    noise_reduction: str
    digital_zoom_applied: bool
    tuning_file: Optional[str]
    autofocus_mode: str
    focus_lock_after_warmup: bool
    locked_lens_position: Optional[float]
    autofocus_preflight_performed: bool
    autofocus_preflight_selection_reason: str
    autofocus_preflight_final_state: str
    autofocus_preflight_best_score: Optional[float]
    autofocus_preflight_elapsed_seconds: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraTrackingTestResult:
    requested_seconds: float
    elapsed_seconds: float
    frames_processed: int
    frames_with_tags: int
    mean_tags_per_frame: float
    median_tags_per_frame: float
    max_tags_in_frame: int
    detection_rate: float
    dictionary_name: str
    box_preset: Optional[str]
    display_width: int
    terminated_early: bool
    unique_ids_detected: list[int]
    rejected_quads_total: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraCheckResult:
    camera_profile: str
    camera_model: str
    requested_width: int
    requested_height: int
    requested_fps: float
    shutter_us: int
    infrared: bool
    monochrome_output: bool
    autofocus_mode: str
    lens_position: Optional[float]
    focus_lock_after_warmup: bool
    autofocus_range: str
    autofocus_speed: str
    autofocus_preflight_enabled: bool
    autofocus_preflight_width: int
    autofocus_preflight_height: int
    autofocus_preflight_timeout_seconds: float
    autofocus_controls_available: list[str]
    codec: str
    resolved_tuning_file: Optional[str]
    tuning_source: str
    tuning_resolved_path: Optional[str]
    model_supports_infrared: Optional[bool]
    ir_compatibility_error: Optional[str]
    picamera2_available: bool
    detected_cameras: list[dict[str, Any]]
    opened_camera_properties: dict[str, Any]
    sensor_modes: list[dict[str, Any]]
    notes: list[str]
    connection_status: str
    connection_findings: list[str]
    recommended_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _summarize_sensor_modes(sensor_modes: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(sensor_modes, (list, tuple)):
        return out
    for mode in sensor_modes:
        if not isinstance(mode, dict):
            out.append({"mode": _jsonable(mode)})
            continue
        summarized: dict[str, Any] = {}
        for key in ("format", "size", "fps", "crop_limits", "bit_depth", "unpacked", "exposure_limits"):
            if key in mode:
                summarized[key] = _jsonable(mode.get(key))
        if not summarized:
            summarized = _jsonable(mode)
        out.append(summarized)
    return out


def _normalize_dictionary_name(dictionary_name: str) -> str:
    name = str(dictionary_name or "4X4_50").strip().upper()
    if not name.startswith("DICT_"):
        name = f"DICT_{name}"
    return name


def _normalize_box_preset(box_preset: Any) -> Optional[str]:
    if box_preset is None:
        return None
    text = str(box_preset).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"custom", "koppert"}:
        return text
    return None


def interpret_camera_connection_diagnostics(
    *,
    expected_sensor: Optional[str],
    detected_cameras: list[dict[str, Any]],
    picamera2_available: bool,
    diagnostic_text: str = "",
) -> tuple[str, list[str], list[str]]:
    """Convert low-level camera failures into concise operator guidance."""
    expected = str(expected_sensor or "").strip().lower()
    combined_detected = json.dumps(detected_cameras, sort_keys=True).lower()
    diagnostic_lower = str(diagnostic_text).lower()
    is_owlsight = expected == "ov64a40"
    expected_detected = bool(expected and expected in combined_detected)

    if detected_cameras and (not expected or expected_detected):
        label = "OwlSight/OV64A40" if is_owlsight else expected_sensor or "camera"
        return "ready", [f"{label} is visible to Picamera2/libcamera."], []

    if detected_cameras and expected and not expected_detected:
        return (
            "mismatch",
            [
                f"The configured sensor '{expected_sensor}' was not found, although another camera was detected."
            ],
            [
                "Confirm the selected BumbleBox camera profile matches the physically connected camera.",
                "Run `rpicam-hello --list-cameras` and verify the reported sensor name before recording.",
            ],
        )

    if not picamera2_available:
        return (
            "unavailable",
            ["Picamera2 is unavailable, so the CSI camera connection could not be checked."],
            ["Install the Raspberry Pi camera packages, then rerun `./bbx camera-check`."],
        )

    chip_id_failure = "failed to read chip id" in diagnostic_lower
    remote_io_failure = any(
        token in diagnostic_lower
        for token in ("error -121", "-121", "remote i/o", "remote io")
    )
    if is_owlsight and chip_id_failure and remote_io_failure:
        return (
            "connection_failure",
            [
                "The OV64A40 driver loaded, but the OwlSight sensor did not answer its I2C chip-ID request (error -121 / remote I/O).",
                "This points to the physical CSI connection or sensor power, not autofocus, resolution, or tuning settings.",
            ],
            [
                "Shut the Raspberry Pi down completely before touching the camera cable; do not reseat CSI hardware while powered.",
                "Reseat both ribbon-cable ends, confirm conductor orientation, and close both connector latches evenly.",
                "Inspect the ribbon for creases or damaged contacts; if possible, test a known-good cable and the other Pi CSI connector.",
                "Power the Pi back on, run `rpicam-hello --list-cameras`, then rerun `./bbx camera-check`.",
            ],
        )

    findings = ["No CSI camera is currently visible to Picamera2/libcamera."]
    if is_owlsight:
        findings.append(
            "The configured OwlSight/OV64A40 sensor was not enumerated; capture settings cannot fix a camera that is absent at this stage."
        )
    actions = [
        "Shut the Pi down, reseat both ends of the CSI ribbon cable, verify orientation and latch closure, then reboot.",
        "Run `rpicam-hello --list-cameras`; continue with BumbleBox only after the sensor appears there.",
        "If it remains absent, test a known-good ribbon cable and alternate CSI connector before changing software settings.",
    ]
    return "missing", findings, actions


def _camera_system_diagnostic_text() -> str:
    outputs: list[str] = []
    rpicam = shutil.which("rpicam-hello") or shutil.which("libcamera-hello")
    commands: list[list[str]] = []
    if rpicam:
        commands.append([rpicam, "--list-cameras"])
    dmesg = shutil.which("dmesg")
    if dmesg:
        commands.append([dmesg, "--color=never"])

    for command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception:
            continue
        text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        relevant = [
            line
            for line in text.splitlines()
            if any(
                token in line.lower()
                for token in ("ov64a40", "chip id", "error -121", "remote i/o", "no cameras")
            )
        ]
        outputs.extend(relevant[-20:])
    return "\n".join(outputs)


def run_camera_check(config: dict[str, Any]) -> CameraCheckResult:
    config = apply_camera_profile(config)
    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera", {}), dict) else {}
    camera_model = str(camera_cfg.get("model", "auto"))
    model_info = get_camera_model_info(camera_model)
    tuning_info = inspect_camera_tuning_resolution(config)
    ir_error = validate_camera_ir_compatibility(config)
    notes: list[str] = []
    detected_cameras: list[dict[str, Any]] = []
    opened_camera_properties: dict[str, Any] = {}
    sensor_modes: list[dict[str, Any]] = []
    autofocus_controls_available: list[str] = []

    try:
        from picamera2 import Picamera2

        picamera2_available = True
    except Exception as exc:  # pragma: no cover - runtime dependency
        picamera2_available = False
        notes.append(f"Picamera2 unavailable: {exc}")
    else:
        picam2 = None
        try:
            camera_info = Picamera2.global_camera_info()
            if isinstance(camera_info, list):
                detected_cameras = [_jsonable(item) for item in camera_info]
            else:
                detected_cameras = [{"camera_info": _jsonable(camera_info)}]
        except Exception as exc:
            notes.append(f"Could not read Picamera2.global_camera_info(): {exc}")

        try:
            resolved_tuning_file = tuning_info.get("resolved")
            if resolved_tuning_file:
                tuning = Picamera2.load_tuning_file(str(resolved_tuning_file))
                picam2 = Picamera2(tuning=tuning)
            else:
                picam2 = Picamera2()
            properties = getattr(picam2, "camera_properties", {})
            if isinstance(properties, dict):
                opened_camera_properties = _jsonable(properties)
            sensor_modes = _summarize_sensor_modes(getattr(picam2, "sensor_modes", None))
            advertised_controls = getattr(picam2, "camera_controls", {})
            if isinstance(advertised_controls, dict):
                autofocus_controls_available = [
                    name
                    for name in (
                        "AfMode",
                        "AfTrigger",
                        "AfState",
                        "AfRange",
                        "AfSpeed",
                        "AfMetering",
                        "AfWindows",
                        "LensPosition",
                    )
                    if name in advertised_controls
                ]
        except Exception as exc:
            notes.append(f"Could not open Picamera2 for properties/modes: {exc}")
        finally:
            if picam2 is not None:
                try:
                    picam2.close()
                except Exception:
                    pass

    expected_sensor = model_info.sensor if model_info is not None else None
    diagnostic_text = "\n".join(notes)
    if expected_sensor == "ov64a40" and not detected_cameras:
        system_diagnostics = _camera_system_diagnostic_text()
        if system_diagnostics:
            diagnostic_text = f"{diagnostic_text}\n{system_diagnostics}"
    connection_status, connection_findings, recommended_actions = (
        interpret_camera_connection_diagnostics(
            expected_sensor=expected_sensor,
            detected_cameras=detected_cameras,
            picamera2_available=picamera2_available,
            diagnostic_text=diagnostic_text,
        )
    )

    focus_settings = autofocus_settings(config)
    preflight_settings = autofocus_preflight_settings(config)
    return CameraCheckResult(
        camera_profile=configured_profile_name(config),
        camera_model=camera_model,
        requested_width=int(camera_cfg.get("width", 4056)),
        requested_height=int(camera_cfg.get("height", 3040)),
        requested_fps=float(camera_cfg.get("fps_target", 5.0)),
        shutter_us=int(camera_cfg.get("shutter_us", 2500)),
        infrared=bool(camera_cfg.get("infrared", False)),
        monochrome_output=bool(camera_cfg.get("monochrome_output", False)),
        autofocus_mode=focus_settings.mode,
        lens_position=focus_settings.lens_position,
        focus_lock_after_warmup=focus_settings.lock_after_warmup,
        autofocus_range=focus_settings.autofocus_range,
        autofocus_speed=focus_settings.autofocus_speed,
        autofocus_preflight_enabled=preflight_settings.enabled,
        autofocus_preflight_width=preflight_settings.width,
        autofocus_preflight_height=preflight_settings.height,
        autofocus_preflight_timeout_seconds=preflight_settings.timeout_seconds,
        autofocus_controls_available=autofocus_controls_available,
        codec=str(camera_cfg.get("codec", "mp4")),
        resolved_tuning_file=tuning_info.get("resolved"),
        tuning_source=str(tuning_info.get("source", "default")),
        tuning_resolved_path=tuning_info.get("resolved_path"),
        model_supports_infrared=(model_info.supports_infrared if model_info is not None else None),
        ir_compatibility_error=ir_error,
        picamera2_available=picamera2_available,
        detected_cameras=detected_cameras,
        opened_camera_properties=opened_camera_properties,
        sensor_modes=sensor_modes,
        notes=notes,
        connection_status=connection_status,
        connection_findings=connection_findings,
        recommended_actions=recommended_actions,
    )


def format_camera_check_result(result: CameraCheckResult) -> str:
    lines = [
        "Camera Check",
        "------------",
        f"Camera profile: {result.camera_profile}",
        f"Camera model: {result.camera_model}",
        f"Requested size/FPS: {result.requested_width}x{result.requested_height} @ {result.requested_fps:g} fps",
        f"Shutter: {result.shutter_us} us",
        f"Codec: {result.codec}",
        f"IR lighting/config: {result.infrared}",
        f"Monochrome output: {result.monochrome_output}",
        f"Autofocus mode: {result.autofocus_mode}",
        f"Manual lens position: {result.lens_position if result.lens_position is not None else 'default'}",
        f"Focus lock after warmup: {result.focus_lock_after_warmup}",
        f"Autofocus range/speed: {result.autofocus_range}/{result.autofocus_speed}",
        f"Low-resolution autofocus preflight: {result.autofocus_preflight_enabled}",
        (
            "Autofocus preflight stream: "
            f"{result.autofocus_preflight_width}x{result.autofocus_preflight_height}"
        ),
        f"Autofocus preflight timeout: {result.autofocus_preflight_timeout_seconds:g} s",
        (
            "Autofocus controls advertised: "
            + (", ".join(result.autofocus_controls_available) or "none reported")
        ),
        f"Model supports IR: {result.model_supports_infrared if result.model_supports_infrared is not None else 'unknown'}",
        f"Tuning source: {result.tuning_source}",
        f"Resolved tuning file: {result.resolved_tuning_file or 'default'}",
        f"Resolved tuning path: {result.tuning_resolved_path or 'none'}",
        f"Picamera2 available: {result.picamera2_available}",
        f"Detected cameras: {len(result.detected_cameras)}",
        f"Sensor modes reported: {len(result.sensor_modes)}",
        f"Connection status: {result.connection_status}",
    ]
    if result.ir_compatibility_error:
        lines.extend(["", "Errors", "------", f"- {result.ir_compatibility_error}"])
    if result.detected_cameras:
        lines.extend(["", "Detected Camera Info", "--------------------"])
        for idx, camera_info in enumerate(result.detected_cameras):
            lines.append(f"- camera[{idx}]: {json.dumps(camera_info, sort_keys=True)}")
    if result.sensor_modes:
        lines.extend(["", "Sensor Modes", "------------"])
        for idx, mode in enumerate(result.sensor_modes):
            lines.append(f"- mode[{idx}]: {json.dumps(mode, sort_keys=True)}")
    if result.connection_findings:
        lines.extend(["", "Connection Diagnosis", "--------------------"])
        lines.extend(f"- {finding}" for finding in result.connection_findings)
    if result.recommended_actions:
        lines.extend(["", "Recommended Actions", "-------------------"])
        lines.extend(
            f"{index}. {action}"
            for index, action in enumerate(result.recommended_actions, start=1)
        )
    if result.notes:
        lines.extend(["", "Notes", "-----"])
        lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines)


def _apply_preset_aruco_params(parameters: Any, box_preset: Optional[str]) -> None:
    if box_preset == "custom":
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeMax = 29
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.06
    elif box_preset == "koppert":
        # Koppert preset remains aligned with custom until a dedicated profile is finalized.
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeMax = 29
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.06


def _apply_custom_aruco_params(parameters: Any, aruco_params: Any, notes: list[str]) -> None:
    if not isinstance(aruco_params, dict):
        return
    for key, value in aruco_params.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)
        else:
            notes.append(f"Ignored unknown ArUco parameter: {key}")


def _open_picamera2(
    config: dict[str, Any],
    width: int,
    height: int,
    frame_format: Optional[str],
    *,
    notes: Optional[list[str]] = None,
) -> tuple[Any, Optional[str], AutofocusPreflightResult]:
    try:
        from picamera2 import Picamera2
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "picamera2 is not available. Install Raspberry Pi camera stack and retry."
        ) from exc

    def _construct_picamera2(resolved_tuning_file: Optional[str]) -> Any:
        try:
            camera_info = Picamera2.global_camera_info()
            if isinstance(camera_info, list) and len(camera_info) == 0:
                raise RuntimeError(
                    "No camera detected by picamera2/libcamera. "
                    "Check ribbon cable orientation/seating and enable camera stack."
                )
        except RuntimeError:
            raise
        except Exception:
            # Continue; some environments may not support this probe cleanly.
            pass

        try:
            if resolved_tuning_file:
                tuning = Picamera2.load_tuning_file(str(resolved_tuning_file))
                return Picamera2(tuning=tuning)
            return Picamera2()
        except IndexError as exc:
            raise RuntimeError(
                "No camera detected by picamera2/libcamera (IndexError during camera open). "
                "Check ribbon cable orientation/seating, camera power, and that no other process owns the camera."
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            if resolved_tuning_file:
                raise RuntimeError(
                    f"Failed to open camera with tuning file '{resolved_tuning_file}': {exc}"
                ) from exc
            raise RuntimeError(f"Failed to open camera: {exc}") from exc

    resolved_tuning_file = resolve_camera_tuning_file(config)
    picam2 = _construct_picamera2(resolved_tuning_file)
    try:
        from libcamera import controls
    except Exception as exc:  # pragma: no cover - runtime dependency
        try:
            picam2.close()
        except Exception:
            pass
        raise RuntimeError("libcamera controls are required for camera setup.") from exc

    try:
        preflight_result = run_autofocus_preflight(
            config,
            picam2,
            controls,
            notes=notes,
            strict=True,
        )
    except Exception:
        try:
            picam2.close()
        except Exception:
            pass
        raise

    main_stream = {"size": (int(width), int(height))}
    if frame_format:
        main_stream["format"] = frame_format

    try:
        camera_config = picam2.create_preview_configuration(main_stream)
        picam2.align_configuration(camera_config)
        picam2.configure(camera_config)
    except IndexError as exc:
        try:
            picam2.close()
        except Exception:
            pass
        raise RuntimeError(
            "Camera opened but failed to configure stream (IndexError). "
            "This usually means libcamera could not enumerate valid sensor modes."
        ) from exc
    except Exception as exc:
        try:
            picam2.close()
        except Exception:
            pass
        raise RuntimeError(f"Failed to configure camera preview stream: {exc}") from exc
    return picam2, resolved_tuning_file, preflight_result


def _apply_camera_controls(config: dict[str, Any], picam2: Any, notes: list[str]) -> bool:
    camera_cfg = config.get("camera", {})
    shutter_us = int(camera_cfg.get("shutter_us", 2500))
    picam2.set_controls({"ExposureTime": shutter_us})

    if bool(camera_cfg.get("monochrome_output", False)):
        try:
            picam2.set_controls({"Saturation": 0.0})
        except Exception:
            notes.append("Could not apply camera.monochrome_output.")

    noise_reduction = str(camera_cfg.get("noise_reduction", "Auto"))
    if noise_reduction != "Auto":
        try:
            from libcamera import controls

            mode = getattr(controls.draft.NoiseReductionModeEnum, noise_reduction)
            picam2.set_controls({"NoiseReductionMode": mode})
        except Exception:
            notes.append(f"Could not apply noise_reduction='{noise_reduction}'.")

    digital_zoom = camera_cfg.get("digital_zoom")
    digital_zoom_applied = False
    if isinstance(digital_zoom, (list, tuple)) and len(digital_zoom) == 4:
        try:
            picam2.set_controls({"ScalerCrop": tuple(digital_zoom)})
            digital_zoom_applied = True
        except Exception:
            notes.append("Could not apply camera.digital_zoom.")
    elif digital_zoom is not None:
        notes.append("camera.digital_zoom must be null or 4 values; ignoring invalid value.")

    return digital_zoom_applied


def run_camera_preview(
    config: dict[str, Any],
    *,
    preview_seconds: float = 20.0,
    window: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> CameraPreviewResult:
    config = apply_camera_profile(config)
    ir_error = validate_camera_ir_compatibility(config)
    if ir_error:
        raise ValueError(ir_error)

    if preview_seconds <= 0:
        raise ValueError("preview_seconds must be > 0")

    # pip OpenCV can overwrite Qt plugin env vars at import time; clear those before Preview.QT starts.
    sanitize_current_qt_env()

    camera_cfg = config.get("camera", {})
    width = int(width if width is not None else camera_cfg.get("width", 4056))
    height = int(height if height is not None else camera_cfg.get("height", 3040))
    window_name = str(window or camera_cfg.get("preview_window", "QT")).strip().upper()
    if window_name not in PREVIEW_WINDOWS:
        raise ValueError(f"window must be one of {sorted(PREVIEW_WINDOWS)}")

    try:
        from libcamera import controls
        from picamera2 import Preview
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("picamera2/libcamera Preview backend is not available.") from exc

    preview_mode = getattr(Preview, window_name)
    notes: list[str] = []
    picam2, resolved_tuning_file, preflight_result = _open_picamera2(
        config,
        width=width,
        height=height,
        frame_format=None,
        notes=notes,
    )
    digital_zoom_applied = _apply_camera_controls(config, picam2, notes)
    if preflight_result.performed:
        apply_preflight_lens_lock(
            preflight_result,
            picam2,
            controls,
            notes=notes,
            strict=True,
        )
    else:
        apply_autofocus_before_start(config, picam2, controls, notes=notes)
    focus_settings = autofocus_settings(config)

    started = False
    preview_started = False
    locked_lens_position: Optional[float] = preflight_result.selected_lens_position
    t0 = time.perf_counter()
    try:
        picam2.start_preview(preview_mode)
        preview_started = True
        picam2.start()
        started = True
        if not preflight_result.performed:
            start_autofocus_after_camera_start(config, picam2, controls, notes=notes)
        if focus_settings.lock_after_warmup and not preflight_result.performed:
            warmup_s = min(
                float(preview_seconds),
                max(0.0, float(config.get("runtime", {}).get("camera_warmup_seconds", 2.0))),
            )
            time.sleep(warmup_s)
            locked_lens_position = lock_autofocus_after_warmup(
                config,
                picam2,
                controls,
                notes=notes,
            )
            time.sleep(max(0.0, float(preview_seconds) - warmup_s))
        else:
            time.sleep(float(preview_seconds))
    finally:
        if preview_started:
            try:
                picam2.stop_preview()
            except Exception:
                pass
        if started:
            try:
                picam2.stop()
            except Exception:
                pass
        try:
            picam2.close()
        except Exception:
            pass

    elapsed = max(0.0, time.perf_counter() - t0)
    return CameraPreviewResult(
        preview_seconds=float(preview_seconds),
        elapsed_seconds=round(elapsed, 3),
        window=window_name,
        width=width,
        height=height,
        shutter_us=int(camera_cfg.get("shutter_us", 2500)),
        noise_reduction=str(camera_cfg.get("noise_reduction", "Auto")),
        digital_zoom_applied=digital_zoom_applied,
        tuning_file=resolved_tuning_file,
        autofocus_mode=focus_settings.mode,
        focus_lock_after_warmup=focus_settings.lock_after_warmup,
        locked_lens_position=locked_lens_position,
        autofocus_preflight_performed=preflight_result.performed,
        autofocus_preflight_selection_reason=preflight_result.selection_reason,
        autofocus_preflight_final_state=preflight_result.final_af_state,
        autofocus_preflight_best_score=preflight_result.best_focus_score,
        autofocus_preflight_elapsed_seconds=preflight_result.elapsed_seconds,
        notes=notes,
    )


def run_camera_tracking_test(
    config: dict[str, Any],
    *,
    test_seconds: float = 20.0,
    display_width: int = 1280,
    dictionary_name: Optional[str] = None,
    box_preset: Optional[str] = None,
    show_rejected: bool = False,
    use_clahe: bool = True,
    window_title: str = "BumbleBox Live Tracking Test (ESC to stop)",
) -> CameraTrackingTestResult:
    config = apply_camera_profile(config)
    ir_error = validate_camera_ir_compatibility(config)
    if ir_error:
        raise ValueError(ir_error)

    if test_seconds <= 0:
        raise ValueError("test_seconds must be > 0")
    if display_width <= 0:
        raise ValueError("display_width must be > 0")

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required for camera-test-tracking.") from exc

    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError(
            "cv2.aruco/ArucoDetector is not available. Install opencv-contrib-python."
        )

    tracking_cfg = config.get("tracking", {})
    camera_cfg = config.get("camera", {})
    resolved_dictionary = _normalize_dictionary_name(
        dictionary_name or tracking_cfg.get("tag_dictionary", "4X4_50")
    )
    if not hasattr(cv2.aruco, resolved_dictionary):
        raise ValueError(f"Unknown ArUco dictionary: {resolved_dictionary}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, resolved_dictionary))
    params = cv2.aruco.DetectorParameters()
    notes: list[str] = []
    resolved_box_preset = _normalize_box_preset(
        box_preset if box_preset is not None else tracking_cfg.get("box_preset")
    )
    _apply_preset_aruco_params(params, resolved_box_preset)
    _apply_custom_aruco_params(params, tracking_cfg.get("aruco_params"), notes)
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    width = int(camera_cfg.get("width", 4056))
    height = int(camera_cfg.get("height", 3040))
    picam2, _resolved_tuning_file, preflight_result = _open_picamera2(
        config,
        width=width,
        height=height,
        frame_format="YUV420",
        notes=notes,
    )
    _apply_camera_controls(config, picam2, notes)
    try:
        from libcamera import controls
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("libcamera controls are required for camera-test-tracking.") from exc
    if preflight_result.performed:
        apply_preflight_lens_lock(
            preflight_result,
            picam2,
            controls,
            notes=notes,
            strict=True,
        )
    else:
        apply_autofocus_before_start(config, picam2, controls, notes=notes)
    warmup_s = float(config.get("runtime", {}).get("camera_warmup_seconds", 2.0))

    frame_tag_counts: list[int] = []
    frames_with_tags = 0
    unique_ids: set[int] = set()
    rejected_quads_total = 0
    terminated_early = False

    started = False
    t0 = time.perf_counter()
    try:
        picam2.start()
        started = True
        if not preflight_result.performed:
            start_autofocus_after_camera_start(config, picam2, controls, notes=notes)
        time.sleep(max(0.0, warmup_s))
        if not preflight_result.performed:
            lock_autofocus_after_warmup(config, picam2, controls, notes=notes)
        start = time.perf_counter()
        while (time.perf_counter() - start) < float(test_seconds):
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            corners, ids, rejected = detector.detectMarkers(gray)
            detected_count = int(len(ids)) if ids is not None else 0
            frame_tag_counts.append(detected_count)
            if detected_count > 0:
                frames_with_tags += 1
                unique_ids.update(int(x) for x in ids.flatten().tolist())

            rejected_count = int(len(rejected)) if rejected is not None else 0
            rejected_quads_total += rejected_count

            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ids is not None and detected_count > 0:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
            if show_rejected and rejected is not None:
                for quad in rejected:
                    points = quad.reshape(-1, 1, 2).astype("int32")
                    cv2.polylines(display, [points], True, (0, 140, 255), 1)

            elapsed = time.perf_counter() - start
            cv2.putText(
                display,
                f"t={elapsed:05.1f}s  tags={detected_count}  rejected={rejected_count}",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "Press ESC to stop early",
                (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            h, w = display.shape[:2]
            scale = float(display_width) / float(w) if w > display_width else 1.0
            out_w = int(round(w * scale))
            out_h = int(round(h * scale))
            if out_w <= 0 or out_h <= 0:
                out_w, out_h = w, h
            shown = cv2.resize(display, (out_w, out_h), interpolation=cv2.INTER_AREA)

            cv2.imshow(window_title, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                terminated_early = True
                break
    finally:
        if started:
            try:
                picam2.stop()
            except Exception:
                pass
        try:
            picam2.close()
        except Exception:
            pass
        try:
            cv2.destroyWindow(window_title)
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    elapsed_total = max(0.0, time.perf_counter() - t0)
    frames_processed = len(frame_tag_counts)
    mean_tags = float(sum(frame_tag_counts) / frames_processed) if frames_processed else 0.0
    median_tags = float(median(frame_tag_counts)) if frames_processed else 0.0
    max_tags = int(max(frame_tag_counts)) if frames_processed else 0
    detection_rate = (float(frames_with_tags) / float(frames_processed)) if frames_processed else 0.0

    if frames_processed == 0:
        notes.append("No frames were processed. Check camera startup and display environment.")
    elif detection_rate < 0.15:
        notes.append("Low detection rate. Check focus, lighting, tag size, and camera distance.")

    return CameraTrackingTestResult(
        requested_seconds=float(test_seconds),
        elapsed_seconds=round(elapsed_total, 3),
        frames_processed=frames_processed,
        frames_with_tags=frames_with_tags,
        mean_tags_per_frame=round(mean_tags, 4),
        median_tags_per_frame=round(median_tags, 4),
        max_tags_in_frame=max_tags,
        detection_rate=round(detection_rate, 4),
        dictionary_name=resolved_dictionary,
        box_preset=resolved_box_preset,
        display_width=int(display_width),
        terminated_early=terminated_early,
        unique_ids_detected=sorted(unique_ids),
        rejected_quads_total=rejected_quads_total,
        notes=notes,
    )


def write_tracking_test_json(result: CameraTrackingTestResult, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return out


def write_camera_check_json(result: CameraCheckResult, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return out


def format_camera_preview_result(result: CameraPreviewResult) -> str:
    lines = [
        "Camera Preview",
        "--------------",
        f"Window: {result.window}",
        f"Configured size: {result.width}x{result.height}",
        f"Requested seconds: {result.preview_seconds}",
        f"Elapsed seconds: {result.elapsed_seconds}",
        f"Shutter (us): {result.shutter_us}",
        f"Noise reduction: {result.noise_reduction}",
        f"Digital zoom applied: {result.digital_zoom_applied}",
        f"Autofocus mode: {result.autofocus_mode}",
        f"Focus lock after warmup: {result.focus_lock_after_warmup}",
        (
            "Locked lens position: "
            f"{result.locked_lens_position if result.locked_lens_position is not None else 'n/a'}"
        ),
        f"Autofocus preflight performed: {result.autofocus_preflight_performed}",
        f"Autofocus preflight selection: {result.autofocus_preflight_selection_reason}",
        f"Autofocus preflight final state: {result.autofocus_preflight_final_state}",
        (
            "Autofocus preflight best score: "
            f"{result.autofocus_preflight_best_score if result.autofocus_preflight_best_score is not None else 'n/a'}"
        ),
        f"Autofocus preflight elapsed (s): {result.autofocus_preflight_elapsed_seconds}",
        f"Tuning file: {result.tuning_file or '(default)'}",
    ]
    if result.notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines)


def format_tracking_test_result(result: CameraTrackingTestResult) -> str:
    lines = [
        "Camera Live Tracking Test",
        "-------------------------",
        f"Dictionary: {result.dictionary_name}",
        f"Box preset: {result.box_preset}",
        f"Requested seconds: {result.requested_seconds}",
        f"Elapsed seconds: {result.elapsed_seconds}",
        f"Frames processed: {result.frames_processed}",
        f"Frames with >=1 tag: {result.frames_with_tags}",
        f"Detection rate: {result.detection_rate}",
        f"Mean tags/frame: {result.mean_tags_per_frame}",
        f"Median tags/frame: {result.median_tags_per_frame}",
        f"Max tags/frame: {result.max_tags_in_frame}",
        f"Unique IDs detected: {len(result.unique_ids_detected)} -> {result.unique_ids_detected}",
        f"Rejected quads (total): {result.rejected_quads_total}",
        f"Ended early (ESC/q): {result.terminated_early}",
    ]
    if result.notes:
        lines.append("")
        lines.append("Notes:")
        for note in result.notes:
            lines.append(f"- {note}")
    return "\n".join(lines)
