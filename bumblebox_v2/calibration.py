from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from math import hypot
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None


@dataclass
class CalibrationResult:
    method: str
    pixels_per_cm: float
    pixel_distance: float
    real_distance_cm: float
    notes: str = ""


def parse_point(point_text: str) -> Tuple[float, float]:
    try:
        x_text, y_text = point_text.split(",", maxsplit=1)
        return float(x_text.strip()), float(y_text.strip())
    except Exception as exc:
        raise ValueError(f"Point must be in 'x,y' format, got: {point_text}") from exc


def calibrate_from_points(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    real_distance_cm: float,
) -> CalibrationResult:
    if real_distance_cm <= 0:
        raise ValueError("real_distance_cm must be > 0")

    pixel_distance = hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
    if pixel_distance <= 0:
        raise ValueError("Selected points are identical; pixel distance is zero.")

    pixels_per_cm = pixel_distance / real_distance_cm
    return CalibrationResult(
        method="manual_points",
        pixels_per_cm=pixels_per_cm,
        pixel_distance=pixel_distance,
        real_distance_cm=real_distance_cm,
        notes="Computed from two manually selected points.",
    )


def _resolve_aruco_dictionary(dictionary_name: str):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for ArUco calibration.")

    if "DICT_" not in dictionary_name:
        dictionary_name = f"DICT_{dictionary_name}"
    dictionary_name = dictionary_name.upper()

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def calibrate_from_aruco_image(
    image_path: str | Path,
    marker_size_mm: float,
    dictionary_name: str = "4X4_50",
    marker_id: Optional[int] = None,
) -> CalibrationResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for ArUco calibration.")
    if marker_size_mm <= 0:
        raise ValueError("marker_size_mm must be > 0")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = _resolve_aruco_dictionary(dictionary_name)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco markers detected in image.")

    selected_index = None
    if marker_id is not None:
        for idx, detected_id in enumerate(ids.flatten().tolist()):
            if detected_id == marker_id:
                selected_index = idx
                break
        if selected_index is None:
            raise RuntimeError(f"Marker ID {marker_id} was not found in the image.")
    else:
        max_perimeter = -1.0
        for idx, marker_corners in enumerate(corners):
            pts = marker_corners[0]
            perimeter = 0.0
            for i in range(4):
                a = pts[i]
                b = pts[(i + 1) % 4]
                perimeter += hypot(a[0] - b[0], a[1] - b[1])
            if perimeter > max_perimeter:
                max_perimeter = perimeter
                selected_index = idx

    assert selected_index is not None
    pts = corners[selected_index][0]

    side_lengths = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        side_lengths.append(hypot(a[0] - b[0], a[1] - b[1]))

    mean_side_px = sum(side_lengths) / 4.0
    real_distance_cm = marker_size_mm / 10.0
    pixels_per_cm = mean_side_px / real_distance_cm

    detected_id = int(ids[selected_index][0])
    return CalibrationResult(
        method="aruco_marker",
        pixels_per_cm=pixels_per_cm,
        pixel_distance=mean_side_px,
        real_distance_cm=real_distance_cm,
        notes=f"Computed from marker ID {detected_id} using dictionary {dictionary_name}.",
    )


def apply_scale_to_config(config: Dict[str, Any], calibration: CalibrationResult) -> Dict[str, Any]:
    updated = dict(config)
    updated.setdefault("calibration", {})
    updated["calibration"]["pixels_per_cm"] = round(calibration.pixels_per_cm, 6)
    updated["calibration"]["method"] = calibration.method
    updated["calibration"]["last_updated"] = datetime.now().isoformat(timespec="seconds")

    metrics = updated.setdefault("metrics", {})
    contact_distance_cm = metrics.get("contact_distance_cm")
    if isinstance(contact_distance_cm, (int, float)) and contact_distance_cm > 0:
        metrics["pixel_contact_distance"] = round(contact_distance_cm * calibration.pixels_per_cm, 3)

    return updated


def format_calibration(calibration: CalibrationResult, pixel_contact_distance: float | None = None) -> str:
    lines = [
        f"Calibration method: {calibration.method}",
        f"Pixels per cm: {round(calibration.pixels_per_cm, 6)}",
        f"Measured pixel distance: {round(calibration.pixel_distance, 3)}",
        f"Real distance (cm): {round(calibration.real_distance_cm, 3)}",
    ]
    if pixel_contact_distance is not None:
        lines.append(f"Derived contact threshold (px): {round(pixel_contact_distance, 3)}")
    if calibration.notes:
        lines.append(f"Notes: {calibration.notes}")
    return "\n".join(lines)


def capture_calibration_image(
    config: Dict[str, Any],
    *,
    output_dir: str | Path,
    filename_prefix: str = "calibration_capture",
) -> Path:
    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except Exception as exc:
        raise RuntimeError(
            "picamera2/libcamera is required to capture calibration images on Pi."
        ) from exc

    from .tuning import resolve_camera_tuning_file

    camera_cfg = config.get("camera", {})
    runtime_cfg = config.get("runtime", {})
    width = int(camera_cfg.get("width", 4056))
    height = int(camera_cfg.get("height", 3040))
    shutter_us = int(camera_cfg.get("shutter_us", 2500))
    noise_reduction = str(camera_cfg.get("noise_reduction", "Auto"))
    digital_zoom = camera_cfg.get("digital_zoom")
    warmup_s = float(runtime_cfg.get("camera_warmup_seconds", 2.0))

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"{filename_prefix}_{stamp}.png"

    tuning_file = resolve_camera_tuning_file(config)
    if tuning_file:
        tuning = Picamera2.load_tuning_file(str(tuning_file))
        picam2 = Picamera2(tuning=tuning)
    else:
        picam2 = Picamera2()

    started = False
    try:
        still = picam2.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam2.align_configuration(still)
        picam2.configure(still)
        picam2.set_controls({"ExposureTime": shutter_us})

        if noise_reduction != "Auto":
            try:
                mode = getattr(controls.draft.NoiseReductionModeEnum, noise_reduction)
                picam2.set_controls({"NoiseReductionMode": mode})
            except Exception:
                pass

        if isinstance(digital_zoom, (list, tuple)) and len(digital_zoom) == 4:
            try:
                picam2.set_controls({"ScalerCrop": tuple(digital_zoom)})
            except Exception:
                pass

        picam2.start()
        started = True
        time.sleep(max(0.0, warmup_s))
        picam2.capture_file(str(output_path))
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

    return output_path


def extract_points_from_labelme_json(
    json_path: str | Path,
) -> tuple[Tuple[float, float], Tuple[float, float], str]:
    path = Path(json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LabelMe JSON not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read LabelMe JSON: {path}") from exc

    shapes = payload.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise RuntimeError("LabelMe JSON has no shapes. Add two point annotations and save.")

    explicit_points: list[tuple[float, float]] = []
    fallback_points: list[tuple[float, float]] = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        shape_points = shape.get("points")
        if not isinstance(shape_points, list):
            continue

        shape_type = str(shape.get("shape_type", "")).strip().lower()
        if shape_type == "point" and len(shape_points) >= 1:
            point = shape_points[0]
            if isinstance(point, list) and len(point) >= 2:
                explicit_points.append((float(point[0]), float(point[1])))

        for point in shape_points:
            if isinstance(point, list) and len(point) >= 2:
                fallback_points.append((float(point[0]), float(point[1])))

    if len(explicit_points) >= 2:
        point_a, point_b = explicit_points[0], explicit_points[1]
        return point_a, point_b, "Loaded first two LabelMe point annotations."

    if len(fallback_points) >= 2:
        point_a, point_b = fallback_points[0], fallback_points[1]
        return (
            point_a,
            point_b,
            "Loaded first two points from saved shapes (no explicit point annotations found).",
        )

    raise RuntimeError("Could not find at least two points in LabelMe JSON.")
