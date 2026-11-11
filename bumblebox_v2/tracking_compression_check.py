"""Check what video compression costs the tag tracker, on the same frames.

RAM recordings are tracked from the camera's raw frames. Stream recordings
(``capture.frame_buffer: stream``) are tracked from the saved video, which has
been compressed. This check captures one short burst of raw frames, tracks
them directly, then encodes those same frames with each candidate encoder and
tracks the resulting videos. Because every source holds the same frames, any
difference in what the tracker finds comes from the video path alone.

A lossless video is included as a reference. If it matches the raw frames, the
video path itself is harmless and any remaining loss is caused by compression,
so a higher-quality encoder would fix it.
"""

from __future__ import annotations

import json
import socket
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .encode_benchmark import DEFAULT_ENCODERS, parse_encoder_spec
from .run_engine import (
    _apply_tracking_id_exclusions,
    _camera_monochrome_output_enabled,
    _capture_frames,
    _excluded_tracking_tag_ids,
    _track_from_ram,
    _track_from_video,
)
from .stream_writer import StreamingVideoWriter, find_ffmpeg_binary

DEFAULT_CHECK_SECONDS = 4.0
RAW_LABEL = "raw frames (RAM mode)"
LOSSLESS_LABEL = "lossless video (reference)"
# A source is treated as equivalent to raw frames at or above this share of
# raw detections kept, with no more than this share of extra detections.
EQUIVALENT_RETAINED_PERCENT = 99.5
EQUIVALENT_EXTRA_PERCENT = 0.5


@dataclass
class SourceResult:
    label: str
    kind: str  # "raw", "lossless" or "encoder"
    frames: int = 0
    detections: int = 0
    tags_per_frame: float = 0.0
    matched: int = 0
    missed: int = 0
    extra: int = 0
    retained_percent: Optional[float] = None
    net_change_percent: Optional[float] = None
    median_shift_px: Optional[float] = None
    max_shift_px: Optional[float] = None
    megabytes_per_frame: Optional[float] = None
    raw_csv_path: Optional[str] = None
    video_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TrackingCompressionReport:
    output_dir: str
    width: int
    height: int
    frames: int
    actual_fps: float
    monochrome_output: bool
    used_mock_camera: bool
    results: List[SourceResult] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_detections(raw_df: Any, other_df: Any) -> Dict[str, Any]:
    """Compare two tracking tables frame by frame and tag by tag.

    A detection is identified by its (frame, ID) pair. Returns how many raw
    detections the other source also found, how many it missed, how many it
    found that raw did not, and how far matched tag centres moved.
    """
    import numpy as np

    def keyed(df: Any) -> Any:
        if df is None or getattr(df, "empty", True):
            return None
        table = df[["frame", "ID", "centroidX", "centroidY"]].copy()
        table["frame"] = table["frame"].astype(int)
        table["ID"] = table["ID"].astype(int)
        return table.drop_duplicates(subset=["frame", "ID"]).set_index(["frame", "ID"])

    raw = keyed(raw_df)
    other = keyed(other_df)
    raw_keys = set(raw.index) if raw is not None else set()
    other_keys = set(other.index) if other is not None else set()
    matched_keys = sorted(raw_keys & other_keys)

    median_shift = max_shift = None
    if matched_keys:
        raw_xy = raw.loc[matched_keys, ["centroidX", "centroidY"]].to_numpy(dtype=float)
        other_xy = other.loc[matched_keys, ["centroidX", "centroidY"]].to_numpy(dtype=float)
        shifts = np.sqrt(((raw_xy - other_xy) ** 2).sum(axis=1))
        median_shift = round(float(np.median(shifts)), 3)
        max_shift = round(float(shifts.max()), 3)

    return {
        "detections": len(other_keys),
        "matched": len(matched_keys),
        "missed": len(raw_keys - other_keys),
        "extra": len(other_keys - raw_keys),
        "retained_percent": round(100.0 * len(matched_keys) / len(raw_keys), 2) if raw_keys else None,
        "median_shift_px": median_shift,
        "max_shift_px": max_shift,
    }


def _safe_name(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.lower())
    return "_".join(part for part in cleaned.split("_") if part) or "source"


def _decoded_frame_count(video_path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count


def _default_output_dir(config: Dict[str, Any]) -> Path:
    now = datetime.now()
    name = f"{socket.gethostname()}_{now.strftime('%Y-%m-%d_%H_%M_%S')}_tracking_compression_check"
    return Path(str(config["system"]["data_root"])).expanduser() / now.strftime("%Y-%m-%d") / name


def run_tracking_compression_check(
    config: Dict[str, Any],
    *,
    seconds: float = DEFAULT_CHECK_SECONDS,
    encoders: Optional[Sequence[str]] = None,
    include_lossless: bool = True,
    keep_videos: bool = False,
    output_dir: Optional[str | Path] = None,
    status: Optional[Callable[[str], None]] = None,
) -> TrackingCompressionReport:
    def say(message: str) -> None:
        if status is not None:
            status(message)

    if not find_ffmpeg_binary():
        raise RuntimeError("ffmpeg was not found on PATH; it is required to encode the test videos.")

    work_config = deepcopy(config)
    work_config["capture"]["recording_seconds"] = float(seconds)
    work_config["capture"]["frame_buffer"] = "ram"  # the burst must be raw frames in memory
    monochrome = _camera_monochrome_output_enabled(work_config)
    used_mock = bool(work_config.get("runtime", {}).get("use_mock_camera", False))

    out_dir = Path(output_dir).expanduser() if output_dir else _default_output_dir(work_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    say(f"Capturing {float(seconds):g}s of raw frames into memory.")
    frames, timestamps, actual_fps, _records = _capture_frames(work_config)
    if not frames:
        raise RuntimeError("The camera returned no frames for the check.")
    frame_height = int(frames[0].shape[0]) * 2 // 3
    frame_width = int(frames[0].shape[1])
    report = TrackingCompressionReport(
        output_dir=str(out_dir),
        width=frame_width,
        height=frame_height,
        frames=len(frames),
        actual_fps=round(float(actual_fps), 6),
        monochrome_output=monochrome,
        used_mock_camera=used_mock,
    )
    excluded_ids = _excluded_tracking_tag_ids(work_config)

    say(f"Tracking {len(frames)} raw frames.")
    raw_df, _raw_noid = _track_from_ram(work_config, "raw_frames", out_dir, frames)
    raw_df, _removed = _apply_tracking_id_exclusions(raw_df, excluded_ids)
    raw_csv = out_dir / "raw_frames_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    baseline = compare_detections(raw_df, raw_df)
    report.results.append(
        SourceResult(
            label=RAW_LABEL,
            kind="raw",
            frames=len(frames),
            detections=baseline["detections"],
            tags_per_frame=round(baseline["detections"] / len(frames), 2),
            matched=baseline["matched"],
            retained_percent=baseline["retained_percent"],
            megabytes_per_frame=round(frames[0].nbytes / 1e6, 2),
            raw_csv_path=str(raw_csv),
        )
    )

    # (label, kind, codec, mp4_codec, mp4_preset, lossless)
    sources: List[tuple[str, str, str, Optional[str], Optional[str], bool]] = []
    if include_lossless:
        sources.append((LOSSLESS_LABEL, "lossless", "mp4", "libx264", "ultrafast", True))
    for spec in encoders or DEFAULT_ENCODERS:
        label, codec, mp4_codec, mp4_preset = parse_encoder_spec(spec)
        sources.append((label, "encoder", codec, mp4_codec, mp4_preset, False))

    for label, kind, codec, mp4_codec, mp4_preset, lossless in sources:
        result = SourceResult(label=label, kind=kind)
        stem = f"video_{_safe_name(label)}"
        video_path: Optional[Path] = None
        try:
            say(f"Encoding and tracking: {label}.")
            writer = StreamingVideoWriter(
                output_dir=out_dir,
                stem=stem,
                fps=float(actual_fps) if actual_fps > 0 else float(work_config["camera"]["fps_target"]),
                recording_codec=codec,
                mp4_codec=mp4_codec or "libx264",
                mp4_preset=mp4_preset or "auto",
                monochrome_output=monochrome,
                max_queue_bytes=2 * int(frames[0].nbytes),
                lossless=lossless,
            )
            try:
                for frame in frames:
                    writer.add(frame)
                streamed = writer.finish(actual_fps=float(actual_fps) if actual_fps > 0 else None)
            finally:
                writer.abort()
            video_path = streamed.video_path
            decoded = _decoded_frame_count(video_path)
            if decoded != len(frames):
                raise RuntimeError(
                    f"The encoded video holds {decoded} frames but {len(frames)} were captured, "
                    "so frames cannot be compared one to one."
                )

            video_df, _video_noid = _track_from_video(work_config, stem, out_dir, video_path)
            video_df, _removed = _apply_tracking_id_exclusions(video_df, excluded_ids)
            video_csv = out_dir / f"{stem}_raw.csv"
            video_df.to_csv(video_csv, index=False)

            comparison = compare_detections(raw_df, video_df)
            result.frames = decoded
            result.detections = comparison["detections"]
            result.tags_per_frame = round(comparison["detections"] / max(1, decoded), 2)
            result.matched = comparison["matched"]
            result.missed = comparison["missed"]
            result.extra = comparison["extra"]
            result.retained_percent = comparison["retained_percent"]
            if baseline["detections"] > 0:
                result.net_change_percent = round(
                    100.0 * (comparison["detections"] - baseline["detections"]) / baseline["detections"], 2
                )
            result.median_shift_px = comparison["median_shift_px"]
            result.max_shift_px = comparison["max_shift_px"]
            result.megabytes_per_frame = round(video_path.stat().st_size / 1e6 / max(1, decoded), 2)
            result.raw_csv_path = str(video_csv)
            if keep_videos:
                result.video_path = str(video_path)
        except Exception as exc:
            result.error = str(exc).splitlines()[0][:300] if str(exc) else exc.__class__.__name__
        finally:
            if not keep_videos:
                for leftover in (video_path, out_dir / f"{stem}.capture.mkv"):
                    if leftover is not None:
                        try:
                            Path(leftover).unlink(missing_ok=True)
                        except Exception:
                            pass
        report.results.append(result)

    report.findings = _findings(report)
    (out_dir / "tracking_compression_check.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report


def _is_equivalent(result: SourceResult, raw_detections: int) -> bool:
    if result.error is not None or result.retained_percent is None or raw_detections <= 0:
        return False
    extra_percent = 100.0 * result.extra / raw_detections
    return result.retained_percent >= EQUIVALENT_RETAINED_PERCENT and extra_percent <= EQUIVALENT_EXTRA_PERCENT


def _findings(report: TrackingCompressionReport) -> List[str]:
    findings: List[str] = []
    raw = next((r for r in report.results if r.kind == "raw"), None)
    raw_detections = raw.detections if raw else 0
    if raw_detections <= 0:
        return [
            "The tracker found no tags in the raw frames, so nothing can be compared. "
            "Point the camera at tagged bees (or a printed tag sheet) and run the check again."
        ]
    if raw_detections < 200:
        findings.append(
            f"Only {raw_detections} raw detections were available. Small differences are not "
            "meaningful at this sample size; use more seconds or more tags in view."
        )

    lossless = next((r for r in report.results if r.kind == "lossless"), None)
    if lossless is not None and lossless.error is None:
        if _is_equivalent(lossless, raw_detections):
            findings.append(
                "The lossless video matches the raw frames, so reading tags from a video costs nothing "
                "by itself. Any difference in the rows below it comes from compression."
            )
        else:
            findings.append(
                f"The lossless video is not identical to the raw frames for the tracker: {lossless.missed} "
                f"missed and {lossless.extra} extra ({_signed(lossless.net_change_percent)}% net). No pixel "
                "was lost to compression in that row, so this is what reading from a video costs by itself "
                "(the image is converted on the way in), and it is the floor for every encoder."
            )

    encoders = [r for r in report.results if r.kind == "encoder" and r.error is None]
    equivalent = [r.label for r in encoders if _is_equivalent(r, raw_detections)]
    different = [r for r in encoders if not _is_equivalent(r, raw_detections)]
    if equivalent:
        findings.append(
            "Equivalent to raw frames on this footage (at least "
            f"{EQUIVALENT_RETAINED_PERCENT:g}% kept, at most {EQUIVALENT_EXTRA_PERCENT:g}% extra): "
            + ", ".join(equivalent)
            + "."
        )
    for result in different:
        findings.append(
            f"{result.label}: {result.missed} missed and {result.extra} extra "
            f"({_signed(result.net_change_percent)}% net detections)."
        )
    findings.append(
        "Missed and extra detections are mostly tags at the edge of readability flipping one way "
        "or the other. Extra ones can be real tags the raw pass missed or false readings; this "
        "check cannot tell which. A large negative net change is the sign of real loss."
    )
    return findings


def _signed(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+g}"


def format_tracking_compression_check(report: TrackingCompressionReport) -> str:
    lines = [
        "Tracking compression check",
        f"Frames: {report.frames} at {report.width}x{report.height} "
        f"({'monochrome' if report.monochrome_output else 'colour'}), captured at {report.actual_fps:g} frames/s"
        + (" from the mock camera" if report.used_mock_camera else ""),
        f"Output folder: {report.output_dir}",
        "",
        f"{'source':<28}{'tags':>7}{'per frame':>10}{'kept %':>8}{'missed':>8}{'extra':>7}{'net %':>8}{'shift px':>10}{'MB/frame':>10}",
    ]
    for result in report.results:
        if result.error:
            lines.append(f"{result.label:<28}failed: {result.error}")
            continue
        is_raw = result.kind == "raw"
        kept = "-" if is_raw or result.retained_percent is None else f"{result.retained_percent:.1f}"
        net = "-" if is_raw or result.net_change_percent is None else f"{result.net_change_percent:+.1f}"
        shift = "-" if is_raw or result.median_shift_px is None else f"{result.median_shift_px:.2f}"
        size = "-" if result.megabytes_per_frame is None else f"{result.megabytes_per_frame:.2f}"
        missed = "-" if is_raw else str(result.missed)
        extra = "-" if is_raw else str(result.extra)
        lines.append(
            f"{result.label:<28}{result.detections:>7}{result.tags_per_frame:>10.1f}{kept:>8}"
            f"{missed:>8}{extra:>7}{net:>8}{shift:>10}{size:>10}"
        )
    lines.append("")
    lines.append(
        "kept % = share of raw-frame detections also found in that video; missed = found in raw "
        "frames only; extra = found in the video only; net % = change in total detections; "
        "shift px = median distance between matched tag centres."
    )
    for finding in report.findings:
        lines.append("")
        lines.append(finding)
    return "\n".join(lines)
