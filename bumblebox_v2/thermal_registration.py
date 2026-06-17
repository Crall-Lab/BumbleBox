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
    overlay_with_tracking_png_path: Optional[str]
    tracking_annotation_count: int
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


def _fit_pair_scales(
    rgb_width: int,
    rgb_height: int,
    thermal_width: int,
    thermal_height: int,
    *,
    max_width: int = 1500,
    max_height: int = 860,
    gap: int = 28,
    zoom: float = 1.0,
) -> tuple[float, float]:
    target_height = min(float(max_height), float(max(1, rgb_height)))
    rgb_scale = min(1.0, target_height / max(1, rgb_height))
    thermal_scale = target_height / max(1, thermal_height)
    total_width = (rgb_width * rgb_scale) + (thermal_width * thermal_scale) + gap
    if total_width > max_width:
        usable_width = max(1.0, float(max_width - gap))
        shrink = usable_width / max(1.0, (rgb_width * rgb_scale) + (thermal_width * thermal_scale))
        rgb_scale *= shrink
        thermal_scale *= shrink
    zoom = max(0.2, float(zoom))
    return rgb_scale * zoom, thermal_scale * zoom


def _draw_point_annotations(
    image: np.ndarray,
    points: Sequence[Point],
    *,
    scale: float = 1.0,
    footer_lines: Optional[Sequence[str]] = None,
    active_index: Optional[int] = None,
) -> np.ndarray:
    out = image.copy()
    for idx, (x, y) in enumerate(points, start=1):
        px = int(round(x * scale))
        py = int(round(y * scale))
        fill_color = (0, 255, 255) if active_index != idx - 1 else (0, 180, 255)
        cv2.circle(out, (px, py), 4, (0, 0, 0), -1)
        cv2.circle(out, (px, py), 3, fill_color, -1)
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


def _pending_side(rgb_points: Sequence[Point], thermal_points: Sequence[Point]) -> str:
    return "rgb" if len(rgb_points) == len(thermal_points) else "thermal"


def _delete_pair_at_index(rgb_points: List[Point], thermal_points: List[Point], index: int) -> None:
    if 0 <= index < len(rgb_points):
        rgb_points.pop(index)
    if 0 <= index < len(thermal_points):
        thermal_points.pop(index)


def _undo_last(rgb_points: List[Point], thermal_points: List[Point]) -> None:
    if len(rgb_points) > len(thermal_points):
        rgb_points.pop()
        return
    if thermal_points:
        thermal_points.pop()
    if rgb_points:
        rgb_points.pop()


def _find_nearest_point_index(
    points: Sequence[Point],
    *,
    scale: float,
    mouse_x: int,
    mouse_y: int,
    threshold_px: float = 16.0,
) -> Optional[int]:
    best_index: Optional[int] = None
    best_distance = threshold_px
    for idx, (x, y) in enumerate(points):
        px = float(x * scale)
        py = float(y * scale)
        distance = float(((px - mouse_x) ** 2 + (py - mouse_y) ** 2) ** 0.5)
        if distance <= best_distance:
            best_distance = distance
            best_index = idx
    return best_index


def _clamp_point(x: float, y: float, width: int, height: int) -> Point:
    return (
        min(max(0.0, x), max(0.0, float(width - 1))),
        min(max(0.0, y), max(0.0, float(height - 1))),
    )


def pick_point_pairs_on_images(
    rgb_image: np.ndarray,
    thermal_image: np.ndarray,
    *,
    window_title: str,
    min_points: int = 4,
) -> tuple[List[Point], List[Point]]:
    rgb_display = _to_display_bgr(rgb_image)
    thermal_display = _to_display_bgr(thermal_image)
    rgb_height, rgb_width = rgb_display.shape[:2]
    thermal_height, thermal_width = thermal_display.shape[:2]

    gap = 28
    border = 4
    header_height = 42
    pad_x = 10
    pad_y = 10
    line_height = 24

    rgb_points: List[Point] = []
    thermal_points: List[Point] = []
    state: Dict[str, Any] = {
        "frame": None,
        "dragging": None,
        "zoom": 1.0,
        "view": {},
    }

    def _build_view(*, active_drag: Optional[tuple[str, int]] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        rgb_scale, thermal_scale = _fit_pair_scales(
            rgb_width,
            rgb_height,
            thermal_width,
            thermal_height,
            gap=gap,
            zoom=state["zoom"],
        )
        rgb_shown = cv2.resize(
            rgb_display,
            (max(1, int(round(rgb_width * rgb_scale))), max(1, int(round(rgb_height * rgb_scale)))),
            interpolation=cv2.INTER_AREA if rgb_scale < 1.0 else cv2.INTER_LINEAR,
        )
        thermal_shown = cv2.resize(
            thermal_display,
            (max(1, int(round(thermal_width * thermal_scale))), max(1, int(round(thermal_height * thermal_scale)))),
            interpolation=cv2.INTER_AREA if thermal_scale < 1.0 else cv2.INTER_NEAREST,
        )

        rgb_panel_w = rgb_shown.shape[1] + (border * 2)
        rgb_panel_h = rgb_shown.shape[0] + (border * 2)
        thermal_panel_w = thermal_shown.shape[1] + (border * 2)
        thermal_panel_h = thermal_shown.shape[0] + (border * 2)
        panels_top = header_height + pad_y
        rgb_panel_x = pad_x
        rgb_panel_y = panels_top
        thermal_panel_x = rgb_panel_x + rgb_panel_w + gap
        thermal_panel_y = panels_top
        rgb_image_x = rgb_panel_x + border
        rgb_image_y = rgb_panel_y + border
        thermal_image_x = thermal_panel_x + border
        thermal_image_y = thermal_panel_y + border

        pending = _pending_side(rgb_points, thermal_points)
        complete_pairs = min(len(rgb_points), len(thermal_points))
        next_pair_index = max(len(rgb_points), len(thermal_points)) + (1 if len(rgb_points) == len(thermal_points) else 0)
        instructions = [
            f"Complete pairs: {complete_pairs}    Next pair: {next_pair_index} -> click {pending.upper()} image    Zoom: {state['zoom']:.2f}x",
            "Left click: add next point on the highlighted image, or drag an existing point to adjust it",
            "Mouse wheel: zoom    Right click near a point: delete that pair    u/backspace: undo last    r: reset    enter/space: finish    esc: cancel",
        ]
        footer_height = (len(instructions) * line_height) + 16
        frame_h = panels_top + max(rgb_panel_h, thermal_panel_h) + footer_height + pad_y
        frame_w = thermal_panel_x + thermal_panel_w + pad_x
        frame = np.full((frame_h, frame_w, 3), 24, dtype=np.uint8)

        rgb_active_index = active_drag[1] if active_drag and active_drag[0] == "rgb" else None
        thermal_active_index = active_drag[1] if active_drag and active_drag[0] == "thermal" else None
        rgb_marked = _draw_point_annotations(rgb_shown, rgb_points, scale=rgb_scale, active_index=rgb_active_index)
        thermal_marked = _draw_point_annotations(
            thermal_shown,
            thermal_points,
            scale=thermal_scale,
            active_index=thermal_active_index,
        )

        rgb_border_color = (70, 170, 70) if pending == "rgb" else (78, 78, 78)
        thermal_border_color = (70, 170, 70) if pending == "thermal" else (78, 78, 78)
        rgb_panel = cv2.copyMakeBorder(rgb_marked, border, border, border, border, cv2.BORDER_CONSTANT, value=rgb_border_color)
        thermal_panel = cv2.copyMakeBorder(
            thermal_marked,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=thermal_border_color,
        )

        frame[rgb_panel_y : rgb_panel_y + rgb_panel.shape[0], rgb_panel_x : rgb_panel_x + rgb_panel.shape[1]] = rgb_panel
        frame[
            thermal_panel_y : thermal_panel_y + thermal_panel.shape[0],
            thermal_panel_x : thermal_panel_x + thermal_panel.shape[1],
        ] = thermal_panel

        rgb_title = "RGB" + ("  <- click next point here" if pending == "rgb" else "")
        thermal_title = "Thermal" + ("  <- click matching point here" if pending == "thermal" else "")
        cv2.putText(frame, rgb_title, (rgb_panel_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            thermal_title,
            (thermal_panel_x, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        footer_y = panels_top + max(rgb_panel_h, thermal_panel_h) + 28
        for line in instructions:
            cv2.putText(frame, line, (pad_x, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
            footer_y += line_height

        view = {
            "rgb_scale": rgb_scale,
            "thermal_scale": thermal_scale,
            "rgb_width": rgb_width,
            "rgb_height": rgb_height,
            "thermal_width": thermal_width,
            "thermal_height": thermal_height,
            "rgb_image_x": rgb_image_x,
            "rgb_image_y": rgb_image_y,
            "thermal_image_x": thermal_image_x,
            "thermal_image_y": thermal_image_y,
            "rgb_shown_w": rgb_shown.shape[1],
            "rgb_shown_h": rgb_shown.shape[0],
            "thermal_shown_w": thermal_shown.shape[1],
            "thermal_shown_h": thermal_shown.shape[0],
        }
        return frame, view

    def _pane_at(x: int, y: int) -> tuple[Optional[str], Optional[tuple[float, float]]]:
        view = state["view"]
        rgb_image_x = int(view["rgb_image_x"])
        rgb_image_y = int(view["rgb_image_y"])
        thermal_image_x = int(view["thermal_image_x"])
        thermal_image_y = int(view["thermal_image_y"])
        rgb_shown_w = int(view["rgb_shown_w"])
        rgb_shown_h = int(view["rgb_shown_h"])
        thermal_shown_w = int(view["thermal_shown_w"])
        thermal_shown_h = int(view["thermal_shown_h"])
        rgb_scale = float(view["rgb_scale"])
        thermal_scale = float(view["thermal_scale"])
        if (
            rgb_image_x <= x < rgb_image_x + rgb_shown_w
            and rgb_image_y <= y < rgb_image_y + rgb_shown_h
        ):
            return "rgb", ((x - rgb_image_x) / rgb_scale, (y - rgb_image_y) / rgb_scale)
        if (
            thermal_image_x <= x < thermal_image_x + thermal_shown_w
            and thermal_image_y <= y < thermal_image_y + thermal_shown_h
        ):
            return "thermal", ((x - thermal_image_x) / thermal_scale, (y - thermal_image_y) / thermal_scale)
        return None, None

    def _redraw(active_drag: Optional[tuple[str, int]] = None) -> np.ndarray:
        frame, view = _build_view(active_drag=active_drag)
        state["view"] = view
        return frame

    def _mouse_callback(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 0
            if hasattr(cv2, "getMouseWheelDelta"):
                try:
                    delta = int(cv2.getMouseWheelDelta(_flags))
                except Exception:
                    delta = 0
            if delta == 0:
                delta = 1 if _flags > 0 else -1
            factor = 1.12 if delta > 0 else (1.0 / 1.12)
            state["zoom"] = min(4.0, max(0.45, float(state["zoom"]) * factor))
            state["frame"] = _redraw(active_drag=state["dragging"])
            return

        pane, local = _pane_at(x, y)
        if pane is None or local is None:
            if event == cv2.EVENT_LBUTTONUP:
                state["dragging"] = None
                state["frame"] = _redraw()
            return

        source_x, source_y = local
        points = rgb_points if pane == "rgb" else thermal_points
        view = state["view"]
        scale = float(view["rgb_scale"] if pane == "rgb" else view["thermal_scale"])
        width = int(view["rgb_width"] if pane == "rgb" else view["thermal_width"])
        height = int(view["rgb_height"] if pane == "rgb" else view["thermal_height"])
        display_x = int(round(source_x * scale))
        display_y = int(round(source_y * scale))
        nearest_idx = _find_nearest_point_index(points, scale=scale, mouse_x=display_x, mouse_y=display_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if nearest_idx is not None:
                state["dragging"] = (pane, nearest_idx)
                state["frame"] = _redraw(active_drag=state["dragging"])
                return

            if pane != _pending_side(rgb_points, thermal_points):
                return

            points.append(_clamp_point(source_x, source_y, width, height))
            state["frame"] = _redraw()
            return

        if event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
            drag_pane, drag_idx = state["dragging"]
            drag_points = rgb_points if drag_pane == "rgb" else thermal_points
            drag_width = rgb_width if drag_pane == "rgb" else thermal_width
            drag_height = rgb_height if drag_pane == "rgb" else thermal_height
            if pane != drag_pane:
                return
            drag_points[drag_idx] = _clamp_point(source_x, source_y, drag_width, drag_height)
            state["frame"] = _redraw(active_drag=state["dragging"])
            return

        if event == cv2.EVENT_LBUTTONUP:
            if state["dragging"] is not None:
                drag_pane, drag_idx = state["dragging"]
                if pane == drag_pane:
                    drag_points = rgb_points if drag_pane == "rgb" else thermal_points
                    drag_width = rgb_width if drag_pane == "rgb" else thermal_width
                    drag_height = rgb_height if drag_pane == "rgb" else thermal_height
                    drag_points[drag_idx] = _clamp_point(source_x, source_y, drag_width, drag_height)
            state["dragging"] = None
            state["frame"] = _redraw()
            return

        if event == cv2.EVENT_RBUTTONDOWN and nearest_idx is not None:
            _delete_pair_at_index(rgb_points, thermal_points, nearest_idx)
            state["dragging"] = None
            state["frame"] = _redraw()

    state["frame"] = _redraw()
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, _mouse_callback)
    try:
        while True:
            cv2.imshow(window_title, state["frame"])
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                if len(rgb_points) == len(thermal_points) and len(rgb_points) >= min_points:
                    break
            elif key in (27,):
                raise RuntimeError("Thermal registration point picking was canceled.")
            elif key in (8, 127, ord("u"), ord("U")):
                if rgb_points or thermal_points:
                    _undo_last(rgb_points, thermal_points)
                    state["dragging"] = None
                    state["frame"] = _redraw()
            elif key in (ord("r"), ord("R")):
                rgb_points.clear()
                thermal_points.clear()
                state["dragging"] = None
                state["frame"] = _redraw()

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError(f"Point picking window closed before completion: {window_title}")
    finally:
        cv2.destroyWindow(window_title)
    return rgb_points, thermal_points


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


def _raw16_to_apparent_celsius(raw_value: float) -> float:
    # Lepton radiometric/TLinear frames are typically centi-Kelvin.
    return (float(raw_value) / 100.0) - 273.15


def _sample_apparent_temperature_c(
    thermal_frame: np.ndarray,
    *,
    x: float,
    y: float,
    radius: int = 2,
) -> Optional[float]:
    if thermal_frame.ndim != 2:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= int(thermal_frame.shape[1]) or yi >= int(thermal_frame.shape[0]):
        return None
    x0 = max(0, xi - radius)
    x1 = min(int(thermal_frame.shape[1]), xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(int(thermal_frame.shape[0]), yi + radius + 1)
    region = thermal_frame[y0:y1, x0:x1]
    if region.size == 0:
        return None
    return _raw16_to_apparent_celsius(float(region.max()))


def annotate_registration_overlay_with_session_tracking(
    *,
    registration: ThermalRegistrationResult,
    run_summary: Dict[str, Any],
) -> tuple[ThermalRegistrationResult, Optional[str]]:
    tracking_csv_path = str(run_summary.get("raw_csv_path") or "").strip()
    thermal_raw_npy_path = str(run_summary.get("thermal_raw_npy_path") or "").strip()
    if not tracking_csv_path:
        return registration, "No raw tracking CSV was found for the selected session."
    if not thermal_raw_npy_path:
        return registration, "No thermal raw stack was found for the selected session."

    tracking_csv = Path(tracking_csv_path).expanduser().resolve()
    thermal_raw_npy = Path(thermal_raw_npy_path).expanduser().resolve()
    if not tracking_csv.exists():
        return registration, f"Tracking CSV not found: {tracking_csv}"
    if not thermal_raw_npy.exists():
        return registration, f"Thermal raw stack not found: {thermal_raw_npy}"

    try:
        import pandas as pd
    except Exception as exc:
        return registration, f"pandas is required for tracked thermal overlay annotations: {exc}"

    try:
        frame_count = int(run_summary.get("frames_captured", 0) or 0)
        thermal_frame_count = int(run_summary.get("thermal_frames_captured", 0) or 0)
    except Exception as exc:
        return registration, f"Could not determine midpoint frame index from run summary: {exc}"
    if frame_count <= 0 or thermal_frame_count <= 0:
        return registration, "Selected session did not report RGB and thermal frame counts."

    rgb_mid_idx = max(0, min(frame_count - 1, frame_count // 2))
    thermal_mid_idx = max(0, min(thermal_frame_count - 1, thermal_frame_count // 2))

    df = pd.read_csv(tracking_csv)
    required = {"frame", "centroidX", "centroidY"}
    missing = sorted(required - set(df.columns))
    if missing:
        return registration, "Tracking CSV is missing required columns: " + ", ".join(missing)

    work = df.copy()
    work["frame"] = pd.to_numeric(work["frame"], errors="coerce")
    work["centroidX"] = pd.to_numeric(work["centroidX"], errors="coerce")
    work["centroidY"] = pd.to_numeric(work["centroidY"], errors="coerce")
    if "ID" in work.columns:
        work["ID"] = pd.to_numeric(work["ID"], errors="coerce")
    frame_rows = work.loc[work["frame"] == rgb_mid_idx].copy()
    frame_rows = frame_rows.dropna(subset=["centroidX", "centroidY"])
    if frame_rows.empty:
        return registration, f"No tracked detections were found on midpoint RGB frame {rgb_mid_idx}."

    thermal_stack = np.load(thermal_raw_npy, mmap_mode="r")
    if getattr(thermal_stack, "ndim", 0) != 3:
        return registration, "Thermal raw stack did not have the expected 3D shape."
    thermal_frame = thermal_stack[thermal_mid_idx]

    homography = np.array(registration.homography, dtype=np.float64)
    try:
        rgb_to_thermal = np.linalg.inv(homography)
    except Exception as exc:
        return registration, f"Could not invert homography for thermal sampling: {exc}"

    overlay_path = Path(registration.overlay_png_path).expanduser().resolve()
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if overlay is None:
        return registration, f"Could not load overlay image for tracking annotation: {overlay_path}"

    annotations_drawn = 0
    for row in frame_rows.itertuples(index=False):
        rgb_x = float(getattr(row, "centroidX"))
        rgb_y = float(getattr(row, "centroidY"))
        rgb_point = np.array([[[rgb_x, rgb_y]]], dtype=np.float32)
        thermal_point = cv2.perspectiveTransform(rgb_point, rgb_to_thermal)[0, 0]
        temperature_c = _sample_apparent_temperature_c(
            thermal_frame,
            x=float(thermal_point[0]),
            y=float(thermal_point[1]),
        )
        if temperature_c is None:
            continue

        cx = int(round(rgb_x))
        cy = int(round(rgb_y))
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 0), -1)
        cv2.circle(overlay, (cx, cy), 3, (0, 220, 255), -1)

        label_parts = []
        if hasattr(row, "ID"):
            id_value = getattr(row, "ID")
            if pd.notna(id_value):
                label_parts.append(str(int(float(id_value))))
        label_parts.append(f"{temperature_c:.1f}C")
        label = " ".join(label_parts)
        text_x = max(0, cx + 8)
        text_y = max(12, cy - 8)
        overlay_h, overlay_w = overlay.shape[:2]
        label_base = max(1.0, min(overlay_w, overlay_h) / 1800.0)
        label_scale = max(0.9, min(2.2, label_base * 1.25))
        label_thickness = max(2, int(round(label_base * 2.0)))
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (0, 0, 0),
            label_thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )
        annotations_drawn += 1

    if annotations_drawn <= 0:
        return registration, "Tracked detections were found, but no thermal temperatures could be sampled."

    output_path = overlay_path.with_name(f"{overlay_path.stem}_with_tracking.png")
    if not cv2.imwrite(str(output_path), overlay):
        return registration, f"Could not write tracked thermal overlay image: {output_path}"

    registration.overlay_with_tracking_png_path = str(output_path)
    registration.tracking_annotation_count = int(annotations_drawn)
    try:
        json_path = Path(registration.registration_json_path).expanduser().resolve()
        if json_path.exists():
            payload = json.loads(json_path.read_text())
            payload["overlay_with_tracking_png_path"] = registration.overlay_with_tracking_png_path
            payload["tracking_annotation_count"] = registration.tracking_annotation_count
            json_path.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return registration, None


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

    rgb_points, thermal_points = pick_point_pairs_on_images(
        rgb_image,
        thermal_image,
        window_title="Thermal Registration Points",
        min_points=min_points,
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
    overlay = cv2.addWeighted(rgb_display, 0.68, warped_thermal, 0.84, 0.0)
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
        "overlay_with_tracking_png_path": None,
        "tracking_annotation_count": 0,
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
        overlay_with_tracking_png_path=None,
        tracking_annotation_count=0,
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
    registration_cfg["overlay_with_tracking_png_path"] = registration.overlay_with_tracking_png_path
    registration_cfg["tracking_annotation_count"] = int(registration.tracking_annotation_count)
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
    if registration.overlay_with_tracking_png_path:
        lines.append(f"Overlay PNG with tracked temperatures: {registration.overlay_with_tracking_png_path}")
        lines.append(f"Tracked annotations: {registration.tracking_annotation_count}")
    if registration.notes:
        lines.append(f"Notes: {registration.notes}")
    return "\n".join(lines)
