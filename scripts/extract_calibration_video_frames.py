#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bumblebox_v2.config import DEFAULT_USER_CONFIG_PATH, load_config


@dataclass
class VideoExtractionResult:
    source_video: str
    output_dir: str
    frame_count: int
    fps: float
    frame_table_csv: str


def _default_output_name() -> str:
    return f"video_frame_export_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()

    config = load_config(args.config)
    data_root = Path(str(config["system"]["data_root"])).expanduser().resolve()
    output_name = args.output_name.strip() if args.output_name else _default_output_name()
    return data_root / "calibration" / output_name


def _extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    csv_path: Path,
) -> VideoExtractionResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_seconds", "time_msec", "png_path"])

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            png_name = f"frame_{frame_count:06d}.png"
            png_path = output_dir / png_name
            if not cv2.imwrite(str(png_path), frame):
                capture.release()
                raise RuntimeError(f"Failed to write frame PNG: {png_path}")

            time_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            writer.writerow([frame_count, f"{time_msec / 1000.0:.6f}", f"{time_msec:.3f}", str(png_path)])
            frame_count += 1

    capture.release()
    return VideoExtractionResult(
        source_video=str(video_path),
        output_dir=str(output_dir),
        frame_count=frame_count,
        fps=fps,
        frame_table_csv=str(csv_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract every frame from corresponding RGB and thermal videos into PNG folders "
            "under the BumbleBox calibration directory."
        )
    )
    parser.add_argument("--rgb-video", required=True, help="Path to the RGB video file.")
    parser.add_argument("--thermal-video", required=True, help="Path to the thermal video file.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_USER_CONFIG_PATH),
        help="BumbleBox config path used to resolve system.data_root when --output-dir is not provided.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Name of the folder to create under <data_root>/calibration/. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. If set, this overrides the config-derived calibration path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rgb_video = Path(args.rgb_video).expanduser().resolve()
    thermal_video = Path(args.thermal_video).expanduser().resolve()
    if not rgb_video.exists():
        raise SystemExit(f"RGB video not found: {rgb_video}")
    if not thermal_video.exists():
        raise SystemExit(f"Thermal video not found: {thermal_video}")

    output_root = _resolve_output_root(args)
    if output_root.exists() and not args.overwrite:
        raise SystemExit(
            f"Output directory already exists: {output_root}\n"
            "Use --overwrite to reuse it, or choose a different --output-name/--output-dir."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    rgb_output_dir = output_root / "rgb_frames"
    thermal_output_dir = output_root / "thermal_frames"
    rgb_csv = output_root / "rgb_frames.csv"
    thermal_csv = output_root / "thermal_frames.csv"
    if args.overwrite:
        for path in (rgb_output_dir, thermal_output_dir):
            if path.exists():
                shutil.rmtree(path)
        for path in (rgb_csv, thermal_csv, output_root / "frame_extraction_manifest.json"):
            if path.exists():
                path.unlink()

    rgb_result = _extract_video_frames(
        video_path=rgb_video,
        output_dir=rgb_output_dir,
        csv_path=rgb_csv,
    )
    thermal_result = _extract_video_frames(
        video_path=thermal_video,
        output_dir=thermal_output_dir,
        csv_path=thermal_csv,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "rgb": asdict(rgb_result),
        "thermal": asdict(thermal_result),
        "frame_count_difference": int(rgb_result.frame_count - thermal_result.frame_count),
    }
    manifest_path = output_root / "frame_extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Calibration Frame Extraction")
    print("----------------------------")
    print(f"Output root: {output_root}")
    print(f"RGB frames: {rgb_result.frame_count} -> {rgb_output_dir}")
    print(f"Thermal frames: {thermal_result.frame_count} -> {thermal_output_dir}")
    print(f"RGB frame table CSV: {rgb_csv}")
    print(f"Thermal frame table CSV: {thermal_csv}")
    print(f"Manifest JSON: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
