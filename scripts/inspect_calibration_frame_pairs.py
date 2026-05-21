#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sorted_image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _fit_scales(
    rgb_width: int,
    rgb_height: int,
    thermal_width: int,
    thermal_height: int,
    *,
    max_width: int,
    max_height: int,
    gap: int = 24,
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
    return rgb_scale, thermal_scale


def _resize_for_display(image: np.ndarray, *, scale: float, pane: str) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else (cv2.INTER_NEAREST if pane == "thermal" else cv2.INTER_LINEAR)
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _load_frame(path: Optional[Path], *, fallback_size: tuple[int, int], label: str) -> np.ndarray:
    if path is None:
        width, height = fallback_size
        blank = np.full((height, width, 3), 20, dtype=np.uint8)
        cv2.putText(blank, f"No {label} frame", (20, max(40, height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        return blank
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        width, height = fallback_size
        blank = np.full((height, width, 3), 20, dtype=np.uint8)
        cv2.putText(blank, f"Could not load {label}", (20, max(40, height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        return blank
    return frame


def _build_display(
    *,
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    rgb_path: Optional[Path],
    thermal_path: Optional[Path],
    rgb_index: int,
    thermal_index: Optional[int],
    thermal_offset: int,
    rgb_total: int,
    thermal_total: int,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    gap = 24
    pad = 12
    title_height = 42
    footer_height = 90

    rgb_scale, thermal_scale = _fit_scales(
        rgb_frame.shape[1],
        rgb_frame.shape[0],
        thermal_frame.shape[1],
        thermal_frame.shape[0],
        max_width=max_width - (pad * 2),
        max_height=max_height - title_height - footer_height - (pad * 2),
        gap=gap,
    )
    rgb_shown = _resize_for_display(rgb_frame, scale=rgb_scale, pane="rgb")
    thermal_shown = _resize_for_display(thermal_frame, scale=thermal_scale, pane="thermal")

    content_height = max(rgb_shown.shape[0], thermal_shown.shape[0])
    frame_width = rgb_shown.shape[1] + thermal_shown.shape[1] + gap + (pad * 2)
    frame_height = title_height + content_height + footer_height + (pad * 2)
    canvas = np.full((frame_height, frame_width, 3), 24, dtype=np.uint8)

    rgb_x = pad
    thermal_x = rgb_x + rgb_shown.shape[1] + gap
    content_y = title_height
    canvas[content_y : content_y + rgb_shown.shape[0], rgb_x : rgb_x + rgb_shown.shape[1]] = rgb_shown
    canvas[content_y : content_y + thermal_shown.shape[0], thermal_x : thermal_x + thermal_shown.shape[1]] = thermal_shown

    rgb_label = f"RGB  idx {rgb_index}/{max(0, rgb_total - 1)}"
    thermal_label = f"Thermal  idx {thermal_index}/{max(0, thermal_total - 1)}" if thermal_index is not None else "Thermal  idx none"
    cv2.putText(canvas, rgb_label, (rgb_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, thermal_label, (thermal_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    footer_y = title_height + content_height + 28
    lines = [
        f"Thermal offset: {thermal_offset:+d}    Pairing rule: thermal_index = rgb_index + offset",
        "Left/Right or A/D: previous/next RGB frame    Up/Down or W/S or ]/[ : adjust thermal offset",
        "PageUp/PageDown: jump 10    Home/End: first/last    Q or Esc: quit",
    ]
    for line in lines:
        cv2.putText(canvas, line, (pad, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        footer_y += 24

    if rgb_path is not None:
        cv2.putText(canvas, rgb_path.name, (pad, frame_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 180, 180), 1, cv2.LINE_AA)
    if thermal_path is not None:
        cv2.putText(canvas, thermal_path.name, (thermal_x, frame_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect extracted RGB and thermal frame folders side by side. "
            "Designed for frame-by-frame lag checking after running extract_calibration_video_frames.py."
        )
    )
    parser.add_argument(
        "--frame-root",
        default="",
        help="Root folder containing rgb_frames/ and thermal_frames/.",
    )
    parser.add_argument(
        "--rgb-dir",
        default="",
        help="Explicit RGB frame directory. Use instead of --frame-root if needed.",
    )
    parser.add_argument(
        "--thermal-dir",
        default="",
        help="Explicit thermal frame directory. Use instead of --frame-root if needed.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Starting RGB frame index.")
    parser.add_argument("--thermal-offset", type=int, default=0, help="Initial thermal frame offset.")
    parser.add_argument("--window-title", default="Calibration Frame Inspector", help="OpenCV window title.")
    parser.add_argument("--max-width", type=int, default=1600, help="Maximum display width.")
    parser.add_argument("--max-height", type=int, default=980, help="Maximum display height.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    frame_root = Path(args.frame_root).expanduser().resolve() if args.frame_root else None
    rgb_dir = Path(args.rgb_dir).expanduser().resolve() if args.rgb_dir else None
    thermal_dir = Path(args.thermal_dir).expanduser().resolve() if args.thermal_dir else None

    if frame_root is not None:
        rgb_dir = frame_root / "rgb_frames"
        thermal_dir = frame_root / "thermal_frames"

    if rgb_dir is None or thermal_dir is None:
        raise SystemExit("Provide --frame-root or both --rgb-dir and --thermal-dir.")
    if not rgb_dir.exists():
        raise SystemExit(f"RGB frame directory not found: {rgb_dir}")
    if not thermal_dir.exists():
        raise SystemExit(f"Thermal frame directory not found: {thermal_dir}")

    rgb_frames = _sorted_image_paths(rgb_dir)
    thermal_frames = _sorted_image_paths(thermal_dir)
    if not rgb_frames:
        raise SystemExit(f"No image frames found in RGB directory: {rgb_dir}")
    if not thermal_frames:
        raise SystemExit(f"No image frames found in thermal directory: {thermal_dir}")

    rgb_index = max(0, min(len(rgb_frames) - 1, int(args.start_index)))
    thermal_offset = int(args.thermal_offset)
    window_title = str(args.window_title)

    rgb_probe = cv2.imread(str(rgb_frames[0]), cv2.IMREAD_COLOR)
    thermal_probe = cv2.imread(str(thermal_frames[0]), cv2.IMREAD_COLOR)
    if rgb_probe is None:
        raise SystemExit(f"Could not load first RGB frame: {rgb_frames[0]}")
    if thermal_probe is None:
        raise SystemExit(f"Could not load first thermal frame: {thermal_frames[0]}")
    rgb_fallback = (rgb_probe.shape[1], rgb_probe.shape[0])
    thermal_fallback = (thermal_probe.shape[1], thermal_probe.shape[0])

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    try:
        while True:
            thermal_index = rgb_index + thermal_offset
            thermal_path = thermal_frames[thermal_index] if 0 <= thermal_index < len(thermal_frames) else None
            rgb_path = rgb_frames[rgb_index]

            rgb_frame = _load_frame(rgb_path, fallback_size=rgb_fallback, label="RGB")
            thermal_frame = _load_frame(thermal_path, fallback_size=thermal_fallback, label="thermal")
            display = _build_display(
                rgb_frame=rgb_frame,
                thermal_frame=thermal_frame,
                rgb_path=rgb_path,
                thermal_path=thermal_path,
                rgb_index=rgb_index,
                thermal_index=(thermal_index if thermal_path is not None else None),
                thermal_offset=thermal_offset,
                rgb_total=len(rgb_frames),
                thermal_total=len(thermal_frames),
                max_width=max(480, int(args.max_width)),
                max_height=max(360, int(args.max_height)),
            )
            cv2.imshow(window_title, display)

            key = cv2.waitKeyEx(0)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (2555904, 83, 65363, ord("d"), ord("D"), ord("l"), ord("L"), 32):
                rgb_index = min(len(rgb_frames) - 1, rgb_index + 1)
                continue
            if key in (2424832, 81, 65361, ord("a"), ord("A"), ord("h"), ord("H")):
                rgb_index = max(0, rgb_index - 1)
                continue
            if key in (2490368, 82, 65362, ord("w"), ord("W"), ord("]")):
                thermal_offset += 1
                continue
            if key in (2621440, 84, 65364, ord("s"), ord("S"), ord("[")):
                thermal_offset -= 1
                continue
            if key in (2162688, 85, 65365):
                rgb_index = max(0, rgb_index - 10)
                continue
            if key in (2228224, 86, 65366):
                rgb_index = min(len(rgb_frames) - 1, rgb_index + 10)
                continue
            if key in (2359296, 80, 65360):
                rgb_index = 0
                continue
            if key in (2293760, 87, 65367):
                rgb_index = len(rgb_frames) - 1
                continue
    finally:
        cv2.destroyWindow(window_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
