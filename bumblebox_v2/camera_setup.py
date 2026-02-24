from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Optional

from .tuning import resolve_camera_tuning_file


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


def _apply_preset_aruco_params(parameters: Any, box_preset: Optional[str]) -> None:
    if box_preset == "custom":
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 31
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.08
    elif box_preset == "koppert":
        # Koppert preset remains aligned with custom until a dedicated profile is finalized.
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 31
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.08


def _apply_custom_aruco_params(parameters: Any, aruco_params: Any, notes: list[str]) -> None:
    if not isinstance(aruco_params, dict):
        return
    for key, value in aruco_params.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)
        else:
            notes.append(f"Ignored unknown ArUco parameter: {key}")


def _open_picamera2(
    config: dict[str, Any], width: int, height: int, frame_format: Optional[str]
) -> tuple[Any, Optional[str]]:
    try:
        from picamera2 import Picamera2
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "picamera2 is not available. Install Raspberry Pi camera stack and retry."
        ) from exc

    resolved_tuning_file = resolve_camera_tuning_file(config)
    if resolved_tuning_file:
        try:
            tuning = Picamera2.load_tuning_file(str(resolved_tuning_file))
            picam2 = Picamera2(tuning=tuning)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tuning file '{resolved_tuning_file}': {exc}"
            ) from exc
    else:
        picam2 = Picamera2()

    main_stream = {"size": (int(width), int(height))}
    if frame_format:
        main_stream["format"] = frame_format

    camera_config = picam2.create_preview_configuration(main_stream)
    picam2.align_configuration(camera_config)
    picam2.configure(camera_config)
    return picam2, resolved_tuning_file


def _apply_camera_controls(config: dict[str, Any], picam2: Any, notes: list[str]) -> bool:
    camera_cfg = config.get("camera", {})
    shutter_us = int(camera_cfg.get("shutter_us", 2500))
    picam2.set_controls({"ExposureTime": shutter_us})

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
    if preview_seconds <= 0:
        raise ValueError("preview_seconds must be > 0")

    camera_cfg = config.get("camera", {})
    width = int(width if width is not None else camera_cfg.get("width", 4056))
    height = int(height if height is not None else camera_cfg.get("height", 3040))
    window_name = str(window or camera_cfg.get("preview_window", "QTGL")).strip().upper()
    if window_name not in PREVIEW_WINDOWS:
        raise ValueError(f"window must be one of {sorted(PREVIEW_WINDOWS)}")

    try:
        from picamera2 import Preview
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("picamera2 Preview backend is not available.") from exc

    preview_mode = getattr(Preview, window_name)
    notes: list[str] = []
    picam2, resolved_tuning_file = _open_picamera2(
        config,
        width=width,
        height=height,
        frame_format=None,
    )
    digital_zoom_applied = _apply_camera_controls(config, picam2, notes)

    started = False
    preview_started = False
    t0 = time.perf_counter()
    try:
        picam2.start_preview(preview_mode)
        preview_started = True
        picam2.start()
        started = True
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
    picam2, _resolved_tuning_file = _open_picamera2(
        config,
        width=width,
        height=height,
        frame_format="YUV420",
    )
    _apply_camera_controls(config, picam2, notes)
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
        time.sleep(max(0.0, warmup_s))
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
        f"Tuning file: {result.tuning_file or '(default)'}",
    ]
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
