from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot
from pathlib import Path
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
