from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class ThermalRegistrationResult:
    method: str
    rgb_image_path: str
    thermal_image_path: str
    point_count: int
    homography: List[List[float]]
    inlier_mask: List[int]
    point_pairs: List[Dict[str, List[float]]]
    output_dir: str
    warped_thermal_png_path: str
    overlay_png_path: str
    correspondence_png_path: str
    registration_json_path: str
    notes: str


def _load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def _to_display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    if image.ndim == 3 and image.shape[2] == 1:
        single = image[:, :, 0]
        normalized = cv2.normalize(single, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    if image.dtype == np.uint16:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    return image.copy()


def _fit_scale(width: int, height: int, *, max_width: int = 1400, max_height: int = 900) -> float:
    return min(1.0, max_width / max(1, width), max_height / max(1, height))


def _draw_point_annotations(
    image: np.ndarray,
    points: Sequence[Point],
    *,
    scale: float = 1.0,
    footer_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    out = image.copy()
    for idx, (x, y) in enumerate(points, start=1):
        px = int(round(x * scale))
        py = int(round(y * scale))
        cv2.circle(out, (px, py), 8, (0, 0, 0), -1)
        cv2.circle(out, (px, py), 6, (0, 255, 255), -1)
        label = str(idx)
        cv2.putText(out, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if footer_lines:
        line_height = 24
        footer_height = (len(footer_lines) * line_height) + 12
        out = cv2.copyMakeBorder(out, 0, footer_height, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        base_y = image.shape[0] + 24
        for line in footer_lines:
            cv2.putText(out, line, (12, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
            base_y += line_height
    return out


def pick_points_on_image(
    image_path: str | Path,
    *,
    window_title: str,
    min_points: int = 4,
) -> List[Point]:
    image = _load_image(image_path)
    display = _to_display_bgr(image)
    height, width = display.shape[:2]
    scale = _fit_scale(width, height)
    if scale < 1.0:
        shown = cv2.resize(display, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    else:
        shown = display.copy()

    points: List[Point] = []
    instructions = [
        "Left click: add point",
        "u/backspace: undo last point",
        "r: reset",
        f"enter/space: finish when >= {min_points} points",
        "esc: cancel",
    ]

    def _redraw() -> np.ndarray:
        return _draw_point_annotations(shown, points, scale=scale, footer_lines=instructions)

    state = {"frame": _redraw()}

    def _mouse_callback(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y >= shown.shape[0]:
            return
        points.append((x / scale, y / scale))
        state["frame"] = _redraw()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, _mouse_callback)
    try:
        while True:
            cv2.imshow(window_title, state["frame"])
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                if len(points) >= min_points:
                    break
            elif key in (27,):
                raise RuntimeError(f"Point picking canceled for {window_title}.")
            elif key in (8, 127, ord("u"), ord("U")):
                if points:
                    points.pop()
                    state["frame"] = _redraw()
            elif key in (ord("r"), ord("R")):
                points.clear()
                state["frame"] = _redraw()

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError(f"Point picking window closed before completion: {window_title}")
    finally:
        cv2.destroyWindow(window_title)
    return points


def _create_correspondence_preview(
    rgb_image: np.ndarray,
    thermal_image: np.ndarray,
    rgb_points: Sequence[Point],
    thermal_points: Sequence[Point],
) -> np.ndarray:
    rgb_bgr = _to_display_bgr(rgb_image)
    thermal_bgr = _to_display_bgr(thermal_image)

    max_h = max(rgb_bgr.shape[0], thermal_bgr.shape[0])
    if rgb_bgr.shape[0] != max_h:
        scale = max_h / rgb_bgr.shape[0]
        rgb_bgr = cv2.resize(rgb_bgr, (int(round(rgb_bgr.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)
        rgb_scale = scale
    else:
        rgb_scale = 1.0
    if thermal_bgr.shape[0] != max_h:
        scale = max_h / thermal_bgr.shape[0]
        thermal_bgr = cv2.resize(thermal_bgr, (int(round(thermal_bgr.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)
        thermal_scale = scale
    else:
        thermal_scale = 1.0

    rgb_marked = _draw_point_annotations(rgb_bgr, rgb_points, scale=rgb_scale)
    thermal_marked = _draw_point_annotations(thermal_bgr, thermal_points, scale=thermal_scale)

    gap = np.full((max_h, 20, 3), 24, dtype=np.uint8)
    combined = np.hstack([rgb_marked, gap, thermal_marked])
    cv2.putText(combined, "RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        combined,
        "Thermal",
        (rgb_marked.shape[1] + gap.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combined


def register_rgb_thermal_pair(
    *,
    rgb_image_path: str | Path,
    thermal_image_path: str | Path,
    output_dir: str | Path,
    min_points: int = 4,
) -> ThermalRegistrationResult:
    rgb_path = Path(rgb_image_path).expanduser().resolve()
    thermal_path = Path(thermal_image_path).expanduser().resolve()
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not thermal_path.exists():
        raise FileNotFoundError(f"Thermal image not found: {thermal_path}")

    rgb_image = _load_image(rgb_path)
    thermal_image = _load_image(thermal_path)
    rgb_display = _to_display_bgr(rgb_image)
    thermal_display = _to_display_bgr(thermal_image)

    rgb_points = pick_points_on_image(rgb_path, window_title="RGB Registration Points", min_points=min_points)
    thermal_points = pick_points_on_image(
        thermal_path,
        window_title="Thermal Registration Points",
        min_points=len(rgb_points),
    )

    if len(rgb_points) != len(thermal_points):
        raise RuntimeError("RGB and thermal point counts do not match.")
    if len(rgb_points) < max(4, int(min_points)):
        raise RuntimeError("At least 4 corresponding points are required for homography registration.")

    rgb_pts = np.array(rgb_points, dtype=np.float32)
    thermal_pts = np.array(thermal_points, dtype=np.float32)
    homography, mask = cv2.findHomography(thermal_pts, rgb_pts, method=0)
    if homography is None:
        raise RuntimeError("OpenCV could not compute a homography from the selected point pairs.")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    warped_thermal = cv2.warpPerspective(
        thermal_display,
        homography,
        (rgb_display.shape[1], rgb_display.shape[0]),
    )
    overlay = cv2.addWeighted(rgb_display, 0.68, warped_thermal, 0.42, 0.0)
    correspondence = _create_correspondence_preview(rgb_image, thermal_image, rgb_points, thermal_points)

    warped_path = output_root / "warped_thermal.png"
    overlay_path = output_root / "rgb_thermal_overlay.png"
    correspondence_path = output_root / "rgb_thermal_correspondences.png"
    cv2.imwrite(str(warped_path), warped_thermal)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(correspondence_path), correspondence)

    point_pairs = [
        {"rgb": [float(rx), float(ry)], "thermal": [float(tx), float(ty)]}
        for (rx, ry), (tx, ty) in zip(rgb_points, thermal_points)
    ]
    payload = {
        "method": "manual_homography_2d",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rgb_image_path": str(rgb_path),
        "thermal_image_path": str(thermal_path),
        "point_count": len(point_pairs),
        "point_pairs": point_pairs,
        "homography": homography.tolist(),
        "inlier_mask": mask.astype(int).ravel().tolist() if mask is not None else [],
        "warped_thermal_png_path": str(warped_path),
        "overlay_png_path": str(overlay_path),
        "correspondence_png_path": str(correspondence_path),
        "notes": (
            "Approximate 2D registration from manually selected correspondence points. "
            "This does not model full 3D nest structure."
        ),
    }
    json_path = output_root / "thermal_registration.json"
    json_path.write_text(json.dumps(payload, indent=2))

    return ThermalRegistrationResult(
        method="manual_homography_2d",
        rgb_image_path=str(rgb_path),
        thermal_image_path=str(thermal_path),
        point_count=len(point_pairs),
        homography=homography.tolist(),
        inlier_mask=payload["inlier_mask"],
        point_pairs=point_pairs,
        output_dir=str(output_root),
        warped_thermal_png_path=str(warped_path),
        overlay_png_path=str(overlay_path),
        correspondence_png_path=str(correspondence_path),
        registration_json_path=str(json_path),
        notes=str(payload["notes"]),
    )


def apply_thermal_registration_to_config(
    config: Dict[str, Any],
    registration: ThermalRegistrationResult,
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(config))
    thermal_cfg = updated.setdefault("thermal", {})
    registration_cfg = thermal_cfg.setdefault("registration", {})
    registration_cfg["method"] = registration.method
    registration_cfg["last_updated"] = datetime.now().isoformat(timespec="seconds")
    registration_cfg["rgb_image_path"] = registration.rgb_image_path
    registration_cfg["thermal_image_path"] = registration.thermal_image_path
    registration_cfg["point_count"] = registration.point_count
    registration_cfg["point_pairs"] = registration.point_pairs
    registration_cfg["homography"] = registration.homography
    registration_cfg["inlier_mask"] = registration.inlier_mask
    registration_cfg["warped_thermal_png_path"] = registration.warped_thermal_png_path
    registration_cfg["overlay_png_path"] = registration.overlay_png_path
    registration_cfg["correspondence_png_path"] = registration.correspondence_png_path
    registration_cfg["registration_json_path"] = registration.registration_json_path
    registration_cfg["notes"] = registration.notes
    return updated


def format_thermal_registration(registration: ThermalRegistrationResult) -> str:
    lines = [
        "Thermal Registration",
        "--------------------",
        f"Method: {registration.method}",
        f"RGB image: {registration.rgb_image_path}",
        f"Thermal image: {registration.thermal_image_path}",
        f"Point pairs: {registration.point_count}",
        f"Output dir: {registration.output_dir}",
        f"Warped thermal PNG: {registration.warped_thermal_png_path}",
        f"Overlay PNG: {registration.overlay_png_path}",
        f"Correspondence PNG: {registration.correspondence_png_path}",
        f"Registration JSON: {registration.registration_json_path}",
    ]
    if registration.notes:
        lines.append(f"Notes: {registration.notes}")
    return "\n".join(lines)
