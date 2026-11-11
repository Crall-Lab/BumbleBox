"""Measure how fast this machine can encode video for stream recording.

Stream recording (``capture.frame_buffer: stream``) only works if the encoder
keeps up with the camera. This benchmark pushes full-size frames through the
same writer the recorder uses, once per candidate encoder, and reports how
many frames per second each one managed and how large its output was.

It measures the encoder alone. During a real recording the camera pipeline
also uses CPU, so treat "marginal" results as "probably too slow" and confirm
the chosen encoder with a real ``bbx run-once`` in stream mode.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .stream_writer import (
    StreamingVideoWriter,
    find_ffmpeg_binary,
    normalize_mp4_codec,
)

# (label, camera.codec, camera.mp4_codec, camera.mp4_preset)
DEFAULT_ENCODERS: tuple[str, ...] = (
    "mjpeg",
    "libx264:ultrafast",
    "libx264:veryfast",
    "libx264:fast",
)
DEFAULT_BENCHMARK_FRAMES = 24
# An encoder needs headroom over the camera's frame rate, because the camera
# pipeline competes for the same CPU cores during a real recording.
HEADROOM_FOR_YES = 1.5


@dataclass
class EncoderResult:
    label: str
    codec: str
    mp4_codec: Optional[str]
    mp4_preset: Optional[str]
    frames: int = 0
    seconds: float = 0.0
    frames_per_second: float = 0.0
    megabytes_per_frame: float = 0.0
    keeps_up: str = "no"
    error: Optional[str] = None


@dataclass
class EncodeBenchmarkReport:
    width: int
    height: int
    frames: int
    fps_target: float
    monochrome_output: bool
    source: str
    work_dir: str
    results: List[EncoderResult] = field(default_factory=list)
    recommended: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_encoder_spec(spec: str) -> tuple[str, str, Optional[str], Optional[str]]:
    """Turn "mjpeg" or "libx264:veryfast" into (label, codec, mp4_codec, mp4_preset)."""
    text = str(spec).strip().lower()
    if not text:
        raise ValueError("Empty encoder name.")
    if text in {"mjpeg", "mjpg"}:
        return "mjpeg", "mjpeg", None, None
    name, _, preset = text.partition(":")
    mp4_codec = normalize_mp4_codec(name)
    if mp4_codec in {"libx264", "libx265"}:
        preset = preset or "ultrafast"
        return f"{mp4_codec}:{preset}", "mp4", mp4_codec, preset
    return mp4_codec, "mp4", mp4_codec, None


def synthetic_frames(width: int, height: int, count: int, *, seed: int = 1) -> List[Any]:
    """YUV420 frames that are roughly as hard to encode as nest footage.

    A static textured background, sensor-like noise that changes every frame,
    and dark blobs that move a little between frames.
    """
    import cv2
    import numpy as np

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Benchmark frame size must be positive and even in both dimensions.")
    rng = np.random.default_rng(seed)
    background = np.zeros((height, width), np.float32)
    for scale, amplitude in ((64, 40.0), (16, 20.0), (4, 8.0)):
        coarse = rng.normal(0, 1, (height // scale + 2, width // scale + 2)).astype(np.float32)
        background += amplitude * cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    background = np.clip(background + 110.0, 20.0, 230.0)

    blob_count = max(4, int(150 * (width * height) / (4056 * 3040)))
    blob_size = (max(3, width // 90), max(2, width // 180))
    positions = rng.uniform([0, 0], [width, height], (blob_count, 2))
    velocities = rng.normal(0, max(1.0, width / 700.0), (blob_count, 2))

    frames: List[Any] = []
    for _ in range(max(1, int(count))):
        luma = background.copy()
        positions += velocities
        for x, y in positions:
            cv2.ellipse(luma, (int(x) % width, int(y) % height), blob_size, float((x + y) % 180), 0, 360, 35, -1)
        luma += rng.normal(0, 2.0, luma.shape).astype(np.float32)
        frame = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
        frame[:height] = np.clip(luma, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def frames_from_video(video_path: str | Path, count: int) -> List[Any]:
    """The first ``count`` frames of a real recording, as YUV420 frames."""
    import cv2

    path = Path(video_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: List[Any] = []
    try:
        while len(frames) < max(1, int(count)):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                break
            height, width = bgr.shape[:2]
            if width % 2 or height % 2:
                bgr = bgr[: height - height % 2, : width - width % 2]
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from video: {path}")
    return frames


def _benchmark_work_dir(config: Dict[str, Any], work_dir: Optional[str | Path]) -> Path:
    """Write test output where recordings go, so disk speed is part of the result."""
    candidates: List[Path] = []
    if work_dir:
        candidates.append(Path(work_dir).expanduser())
    data_root = config.get("system", {}).get("data_root") if isinstance(config.get("system"), dict) else None
    if data_root:
        candidates.append(Path(str(data_root)).expanduser())
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return Path(tempfile.mkdtemp(prefix="bbx-encode-benchmark-", dir=str(candidate)))
        except Exception:
            continue
    return Path(tempfile.mkdtemp(prefix="bbx-encode-benchmark-"))


def _keeps_up(frames_per_second: float, fps_target: float) -> str:
    if fps_target <= 0:
        return "n/a"
    if frames_per_second >= HEADROOM_FOR_YES * fps_target:
        return "yes"
    if frames_per_second >= fps_target:
        return "marginal"
    return "no"


def run_encode_benchmark(
    config: Dict[str, Any],
    *,
    frames: int = DEFAULT_BENCHMARK_FRAMES,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps_target: Optional[float] = None,
    encoders: Optional[Sequence[str]] = None,
    source_video: Optional[str | Path] = None,
    monochrome_output: Optional[bool] = None,
    work_dir: Optional[str | Path] = None,
) -> EncodeBenchmarkReport:
    camera = config.get("camera", {}) if isinstance(config.get("camera"), dict) else {}
    target = float(fps_target if fps_target is not None else camera.get("fps_target", 5.0))
    mono = bool(camera.get("monochrome_output", False)) if monochrome_output is None else bool(monochrome_output)
    if not find_ffmpeg_binary():
        raise RuntimeError("ffmpeg was not found on PATH; it is required for stream recording.")

    if source_video:
        test_frames = frames_from_video(source_video, frames)
        source = f"video: {Path(source_video).expanduser()}"
    else:
        test_frames = synthetic_frames(
            int(width if width is not None else camera.get("width", 4056)),
            int(height if height is not None else camera.get("height", 3040)),
            frames,
        )
        source = "synthetic nest-like frames"
    frame_height = int(test_frames[0].shape[0]) * 2 // 3
    frame_width = int(test_frames[0].shape[1])

    out_dir = _benchmark_work_dir(config, work_dir)
    report = EncodeBenchmarkReport(
        width=frame_width,
        height=frame_height,
        frames=len(test_frames),
        fps_target=target,
        monochrome_output=mono,
        source=source,
        work_dir=str(out_dir.parent),
    )
    try:
        for index, spec in enumerate(encoders or DEFAULT_ENCODERS):
            label, codec, mp4_codec, mp4_preset = parse_encoder_spec(spec)
            result = EncoderResult(label=label, codec=codec, mp4_codec=mp4_codec, mp4_preset=mp4_preset)
            try:
                writer = StreamingVideoWriter(
                    output_dir=out_dir,
                    stem=f"benchmark_{index}",
                    fps=target if target > 0 else 5.0,
                    recording_codec=codec,
                    mp4_codec=mp4_codec or "libx264",
                    mp4_preset=mp4_preset or "auto",
                    monochrome_output=mono,
                    # A tiny queue makes add() wait for the encoder, so the
                    # elapsed time is the encoder's own throughput.
                    max_queue_bytes=2 * int(test_frames[0].nbytes),
                )
                started = time.perf_counter()
                try:
                    for frame in test_frames:
                        writer.add(frame)
                    streamed = writer.finish(finalize=False)
                finally:
                    writer.abort()
                elapsed = max(1e-9, time.perf_counter() - started)
                result.frames = streamed.frame_count
                result.seconds = round(elapsed, 3)
                result.frames_per_second = round(streamed.frame_count / elapsed, 3)
                result.megabytes_per_frame = round(
                    streamed.video_path.stat().st_size / 1e6 / max(1, streamed.frame_count), 3
                )
                result.keeps_up = _keeps_up(result.frames_per_second, target)
            except Exception as exc:
                result.error = str(exc).splitlines()[0][:300] if str(exc) else exc.__class__.__name__
            report.results.append(result)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    report.recommended = _recommend(report.results)
    report.notes.append(
        "This measures the encoder alone. The camera pipeline also uses CPU during a real "
        "recording, so confirm the choice with a stream-mode `bbx run-once` and check that "
        "the run summary reports no stalls."
    )
    if not source_video:
        report.notes.append(
            "Synthetic frames were used. For a closer estimate, pass --source-video with one of your own recordings."
        )
    return report


def _recommend(results: Sequence[EncoderResult]) -> Optional[str]:
    """Prefer the smallest files among encoders that keep up with headroom."""
    good = [r for r in results if r.error is None and r.keeps_up == "yes"]
    if not good:
        return None
    return min(good, key=lambda r: r.megabytes_per_frame).label


def config_lines_for(label: str) -> List[str]:
    """Config settings that select the encoder with this label."""
    _label, codec, mp4_codec, mp4_preset = parse_encoder_spec(label)
    lines = ["capture.frame_buffer: stream", f"camera.codec: {codec}"]
    if mp4_codec:
        lines.append(f"camera.mp4_codec: {mp4_codec}")
    if mp4_preset:
        lines.append(f"camera.mp4_preset: {mp4_preset}")
    return lines


def format_encode_benchmark(report: EncodeBenchmarkReport) -> str:
    megabytes_per_frame_raw = report.width * report.height * 1.5 / 1e6
    lines = [
        "Stream encoding benchmark",
        f"Frames: {report.frames} at {report.width}x{report.height} "
        f"({'monochrome' if report.monochrome_output else 'colour'}; {megabytes_per_frame_raw:.1f} MB each unencoded)",
        f"Source: {report.source}",
        f"Output written under: {report.work_dir}",
        f"Camera target: {report.fps_target:g} frames/s",
        "",
        f"{'encoder':<22}{'frames/s':>10}{'x target':>10}{'MB/frame':>10}{'GB/hour':>9}  keeps up",
    ]
    for result in report.results:
        if result.error:
            lines.append(f"{result.label:<22}failed: {result.error}")
            continue
        ratio = result.frames_per_second / report.fps_target if report.fps_target > 0 else 0.0
        gb_per_hour = result.megabytes_per_frame * report.fps_target * 3600.0 / 1000.0
        lines.append(
            f"{result.label:<22}{result.frames_per_second:>10.2f}{ratio:>9.1f}x"
            f"{result.megabytes_per_frame:>10.2f}{gb_per_hour:>9.1f}  {result.keeps_up}"
        )
    lines.append("")
    lines.append(
        f"keeps up: yes = at least {HEADROOM_FOR_YES:g}x the camera target; "
        "marginal = faster than the target but with little headroom; no = slower than the target."
    )
    if report.recommended:
        lines.append("")
        lines.append(f"Suggested encoder: {report.recommended} (smallest files among those that keep up). Config:")
        lines.extend(f"  {line}" for line in config_lines_for(report.recommended))
    else:
        lines.append("")
        lines.append(
            "No encoder kept up with headroom at this size and frame rate. Lower camera.fps_target or "
            "the resolution, or keep capture.frame_buffer: ram for short recordings."
        )
    for note in report.notes:
        lines.append("")
        lines.append(f"Note: {note}")
    return "\n".join(lines)
