"""Encode a recording to disk while it is being captured ("stream" mode).

In the default "ram" mode every frame of a recording is held in memory and the
video is encoded after capture ends, so recording length is limited by RAM.
``StreamingVideoWriter`` removes that limit: the capture loop hands each frame
to a background thread, which feeds an ffmpeg process as frames arrive. The
capture loop never waits on the encoder unless the encoder falls so far behind
that the waiting frames would exceed a fixed memory budget.

The recording is written to a Matroska file first, because a Matroska file
stays readable if the run is killed part-way. When capture ends it is copied
(not re-encoded) into the final ``.mp4`` or ``.avi`` with the measured frame
rate.
"""

from __future__ import annotations

import collections
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

FRAME_BUFFER_RAM = "ram"
FRAME_BUFFER_STREAM = "stream"
VALID_FRAME_BUFFERS = {FRAME_BUFFER_RAM, FRAME_BUFFER_STREAM}

MP4_PRESET_AUTO = "auto"
VALID_MP4_PRESETS = {
    MP4_PRESET_AUTO,
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
}
# "auto" keeps the long-standing preset for RAM recordings (encoded after
# capture, so speed does not matter) and picks the fastest one for streaming.
RAM_MP4_PRESET = "fast"
STREAM_MP4_PRESET = "ultrafast"

DEFAULT_STREAM_QUEUE_MB = 512.0
ENCODER_EXIT_TIMEOUT_SECONDS = 600.0
FRAME_RATE_CORRECTION_TOLERANCE = 0.001
# Shorter waits than this are not worth a warning in the run summary.
STALL_WARNING_SECONDS = 0.25


def find_ffmpeg_binary() -> Optional[str]:
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    for candidate in ("/usr/bin/ffmpeg", "/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    return None


def normalize_mp4_codec(mp4_codec: Any) -> str:
    raw = str(mp4_codec or "").strip().lower()
    aliases = {
        "mp4v": "mpeg4",
        "mpeg4": "mpeg4",
        "h264": "libx264",
        "x264": "libx264",
        "libx264": "libx264",
        "avc": "libx264",
        "h265": "libx265",
        "hevc": "libx265",
        "x265": "libx265",
        "libx265": "libx265",
    }
    return aliases.get(raw, raw or "mpeg4")


def resolve_mp4_preset(value: Any, *, streaming: bool) -> str:
    preset = str(value or MP4_PRESET_AUTO).strip().lower()
    if preset == MP4_PRESET_AUTO or preset not in VALID_MP4_PRESETS:
        return STREAM_MP4_PRESET if streaming else RAM_MP4_PRESET
    return preset


def frame_buffer_mode(config: Dict[str, Any]) -> str:
    capture = config.get("capture", {})
    if not isinstance(capture, dict):
        return FRAME_BUFFER_RAM
    mode = str(capture.get("frame_buffer", FRAME_BUFFER_RAM)).strip().lower()
    return mode if mode in VALID_FRAME_BUFFERS else FRAME_BUFFER_RAM


def stream_queue_bytes(config: Dict[str, Any]) -> int:
    capture = config.get("capture", {})
    raw = capture.get("stream_queue_mb", DEFAULT_STREAM_QUEUE_MB) if isinstance(capture, dict) else DEFAULT_STREAM_QUEUE_MB
    try:
        megabytes = float(raw)
    except (TypeError, ValueError):
        megabytes = DEFAULT_STREAM_QUEUE_MB
    if megabytes <= 0:
        megabytes = DEFAULT_STREAM_QUEUE_MB
    return int(megabytes * 1024 * 1024)


@dataclass
class StreamStats:
    encoder: str
    preset: Optional[str]
    frames_written: int = 0
    width: int = 0
    height: int = 0
    queue_budget_mb: float = 0.0
    peak_queue_frames: int = 0
    peak_queue_mb: float = 0.0
    capture_stall_count: int = 0
    capture_stall_seconds: float = 0.0
    encode_seconds: float = 0.0
    encoded_megabytes: float = 0.0
    nominal_fps: float = 0.0
    container_fps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("queue_budget_mb", "peak_queue_mb", "capture_stall_seconds", "encode_seconds", "encoded_megabytes"):
            payload[key] = round(float(payload[key]), 3)
        return payload


@dataclass
class StreamedVideo:
    video_path: Path
    recording_codec: str
    frame_count: int
    preview_frame: Any
    stats: StreamStats
    warnings: List[str] = field(default_factory=list)


class StreamingVideoWriter:
    """Feeds frames to ffmpeg from a background thread as they are captured.

    Frames are YUV420 (I420) arrays as returned by Picamera2: a 2-D uint8
    array of shape ``(height * 3 / 2, width)``.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        stem: str,
        fps: float,
        recording_codec: str = "mp4",
        mp4_codec: str = "libx264",
        mp4_preset: str = MP4_PRESET_AUTO,
        monochrome_output: bool = False,
        max_queue_bytes: int = int(DEFAULT_STREAM_QUEUE_MB * 1024 * 1024),
        preview_frame_index: int = 0,
        ffmpeg_bin: Optional[str] = None,
        lossless: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.stem = str(stem)
        self.fps = float(fps) if float(fps) > 0 else 1.0
        self.recording_codec = "mjpeg" if str(recording_codec).strip().lower() in {"mjpeg", "mjpg"} else "mp4"
        if self.recording_codec == "mjpeg":
            self.encoder = "mjpeg"
            self.preset: Optional[str] = None
            final_suffix = ".avi"
        else:
            self.encoder = normalize_mp4_codec(mp4_codec)
            self.preset = (
                resolve_mp4_preset(mp4_preset, streaming=True)
                if self.encoder in {"libx264", "libx265"}
                else None
            )
            final_suffix = ".mp4"
        # Lossless output reproduces every pixel exactly, at several times the
        # file size. It is used as a reference when checking what compression
        # costs the tracker (see tracking_compression_check).
        self.lossless = bool(lossless)
        if self.lossless and self.encoder not in {"libx264", "libx265"}:
            raise ValueError(f"Lossless output is not available for encoder '{self.encoder}'.")
        self.monochrome_output = bool(monochrome_output)
        self.max_queue_bytes = max(1, int(max_queue_bytes))
        self.preview_frame_index = max(0, int(preview_frame_index))
        self.capture_path = self.output_dir / f"{self.stem}.capture.mkv"
        self.final_path = self.output_dir / f"{self.stem}{final_suffix}"

        self._ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()
        if not self._ffmpeg_bin:
            raise RuntimeError(
                "Stream recording requires ffmpeg, but it was not found on PATH. "
                "Install ffmpeg or set capture.frame_buffer to 'ram' with camera.codec 'mjpeg'."
            )

        self._cond = threading.Condition()
        self._queue: Deque[Any] = collections.deque()
        self._queued_bytes = 0
        self._closing = False
        self._closed = False
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._stderr_file: Optional[Any] = None
        self._neutral_chroma: Optional[bytes] = None
        self._width = 0
        self._height = 0
        self._frames_added = 0
        self._preview_frame: Any = None
        self._first_write_monotonic: Optional[float] = None
        self._last_write_monotonic: Optional[float] = None
        self.stats = StreamStats(
            encoder=self.encoder,
            preset=self.preset,
            queue_budget_mb=self.max_queue_bytes / (1024 * 1024),
            nominal_fps=self.fps,
        )

    # ------------------------------------------------------------------
    # Capture side
    # ------------------------------------------------------------------
    def add(self, frame: Any) -> None:
        """Queue one frame. Returns at once unless the memory budget is full."""
        nbytes = int(frame.nbytes)
        with self._cond:
            if self._closing or self._closed:
                raise RuntimeError("Stream writer is already closed.")
            self._raise_if_failed()
            stalled_from: Optional[float] = None
            # Always let one frame in, so a budget smaller than a frame still works.
            while self._queued_bytes > 0 and self._queued_bytes + nbytes > self.max_queue_bytes:
                if self._error is not None:
                    break
                if stalled_from is None:
                    stalled_from = time.perf_counter()
                    self.stats.capture_stall_count += 1
                self._cond.wait(timeout=0.25)
            if stalled_from is not None:
                self.stats.capture_stall_seconds += time.perf_counter() - stalled_from
            self._raise_if_failed()

            self._queue.append(frame)
            self._queued_bytes += nbytes
            self.stats.peak_queue_frames = max(self.stats.peak_queue_frames, len(self._queue))
            self.stats.peak_queue_mb = max(self.stats.peak_queue_mb, self._queued_bytes / (1024 * 1024))
            if self._frames_added <= self.preview_frame_index:
                self._preview_frame = frame
            self._frames_added += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="bbx-stream-writer", daemon=True)
                self._thread.start()
            self._cond.notify_all()

    @property
    def frames_added(self) -> int:
        return self._frames_added

    # ------------------------------------------------------------------
    # Finishing
    # ------------------------------------------------------------------
    def finish(self, actual_fps: Optional[float] = None, *, finalize: bool = True) -> StreamedVideo:
        """Wait for the encoder to drain, then produce the final video file.

        With ``finalize=False`` the Matroska capture file is returned as-is.
        """
        self._drain_and_close_encoder()
        if self._error is not None:
            raise RuntimeError(self._error)
        if self.stats.frames_written <= 0:
            raise RuntimeError("Stream recording captured zero frames; no video was written.")

        warnings: List[str] = []
        if self.stats.capture_stall_seconds >= STALL_WARNING_SECONDS:
            warnings.append(
                "Stream encoder fell behind the camera: capture waited "
                f"{self.stats.capture_stall_seconds:.1f}s in total across "
                f"{self.stats.capture_stall_count} stall(s) with a "
                f"{self.stats.queue_budget_mb:.0f} MB queue, which lowers the achieved frame rate. "
                "Use a faster encoder (camera.mp4_preset or camera.codec: mjpeg), lower the frame "
                "rate or resolution, or raise capture.stream_queue_mb."
            )

        video_path = self.capture_path
        if finalize:
            video_path, finalize_warning = self._finalize(actual_fps)
            if finalize_warning:
                warnings.append(finalize_warning)
        if video_path.exists():
            self.stats.encoded_megabytes = video_path.stat().st_size / 1e6
        return StreamedVideo(
            video_path=video_path,
            recording_codec=self.recording_codec,
            frame_count=self.stats.frames_written,
            preview_frame=self._preview_frame,
            stats=self.stats,
            warnings=warnings,
        )

    def abort(self) -> None:
        """Stop after a failed capture, keeping what was captured so far.

        Frames already handed over are still encoded, and the Matroska capture
        file is left in place so the partial recording can be read. Does
        nothing if the writer has already finished.
        """
        try:
            self._drain_and_close_encoder()
        except Exception:
            pass

    def _drain_and_close_encoder(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closing = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join()

        process = self._process
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
            try:
                return_code = process.wait(timeout=ENCODER_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return_code = -9
                self._set_error(
                    f"ffmpeg did not finish within {ENCODER_EXIT_TIMEOUT_SECONDS:.0f}s after capture ended and was stopped."
                )
            if return_code != 0 and self._error is None:
                self._set_error(
                    f"ffmpeg failed while encoding '{self.capture_path.name}' with encoder "
                    f"'{self.encoder}': {self._read_encoder_stderr() or f'exit code {return_code}'}"
                )
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None
        if self._first_write_monotonic is not None and self._last_write_monotonic is not None:
            self.stats.encode_seconds = max(0.0, time.perf_counter() - self._first_write_monotonic)
        with self._cond:
            self._closed = True

    def _finalize(self, actual_fps: Optional[float]) -> tuple[Path, Optional[str]]:
        """Copy the capture file into its final container with the measured frame rate."""
        measured = float(actual_fps) if actual_fps and float(actual_fps) > 0 else self.fps
        cmd: List[str] = [str(self._ffmpeg_bin), "-y", "-hide_banner", "-loglevel", "error"]
        if self.recording_codec == "mjpeg":
            # AVI is constant-rate: restating the input rate is exact.
            cmd += ["-r", f"{measured:.6f}", "-i", str(self.capture_path), "-c", "copy", str(self.final_path)]
        else:
            ratio = self.fps / measured
            if abs(ratio - 1.0) > FRAME_RATE_CORRECTION_TOLERANCE:
                cmd += ["-itsscale", f"{ratio:.9f}"]
            cmd += ["-i", str(self.capture_path), "-c", "copy", "-movflags", "+faststart", str(self.final_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=ENCODER_EXIT_TIMEOUT_SECONDS)
            failure = result.stderr.strip() if result.returncode != 0 else ""
        except Exception as exc:  # timeout or launch failure
            failure = str(exc)
        if failure or not self.final_path.exists() or self.final_path.stat().st_size < 1024:
            try:
                self.final_path.unlink(missing_ok=True)
            except Exception:
                pass
            return self.capture_path, (
                f"Could not write the final {self.final_path.suffix} file ({failure or 'empty output'}); "
                f"the recording was kept as {self.capture_path.name} at the nominal {self.fps:.3f} FPS."
            )
        self.stats.container_fps = round(measured, 6)
        try:
            self.capture_path.unlink()
        except Exception:
            pass
        return self.final_path, None

    # ------------------------------------------------------------------
    # Encoder side (background thread)
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._queue and not self._closing:
                        self._cond.wait(timeout=0.25)
                    if not self._queue:
                        return
                    frame = self._queue.popleft()
                if self._process is None:
                    self._start_encoder(frame)
                self._write_frame(frame)
                with self._cond:
                    self._queued_bytes = max(0, self._queued_bytes - int(frame.nbytes))
                    self.stats.frames_written += 1
                    self._cond.notify_all()
                del frame
        except BaseException as exc:  # includes BrokenPipeError when ffmpeg exits early
            detail = self._read_encoder_stderr()
            message = (
                f"Stream encoder '{self.encoder}' stopped after {self.stats.frames_written} frame(s): "
                f"{detail or exc}"
            )
            self._set_error(message)

    def _set_error(self, message: str) -> None:
        with self._cond:
            if self._error is None:
                self._error = str(message)
            self._queue.clear()
            self._queued_bytes = 0
            self._cond.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(self._error)

    def _read_encoder_stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0)
            return self._stderr_file.read().decode("utf-8", errors="replace").strip()[-2000:]
        except Exception:
            return ""

    def _start_encoder(self, first_frame: Any) -> None:
        shape = getattr(first_frame, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[0]) % 3 != 0:
            raise RuntimeError(
                f"Stream recording expects YUV420 frames shaped (height*3/2, width); got shape {shape}."
            )
        self._height = int(shape[0]) * 2 // 3
        self._width = int(shape[1])
        if self._width % 2 or self._height % 2:
            raise RuntimeError(
                f"Stream recording needs even frame dimensions; got {self._width}x{self._height}."
            )
        self.stats.width = self._width
        self.stats.height = self._height
        if self.monochrome_output:
            # Grey video is the luma plane plus neutral chroma, so ffmpeg does
            # no colour conversion and levels match the RAM-mode output.
            self._neutral_chroma = bytes([128]) * (self._width * self._height // 2)

        cmd: List[str] = [
            str(self._ffmpeg_bin),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s:v",
            f"{self._width}x{self._height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.encoder,
        ]
        if self.encoder == "mjpeg":
            cmd += ["-q:v", "2", "-pix_fmt", "yuvj420p"]
        elif self.encoder == "libx264":
            quality = ["-qp", "0"] if self.lossless else ["-crf", "18"]
            cmd += ["-preset", str(self.preset), *quality, "-pix_fmt", "yuv420p"]
        elif self.encoder == "libx265":
            quality = ["-x265-params", "lossless=1"] if self.lossless else ["-crf", "20"]
            cmd += ["-preset", str(self.preset), *quality, "-pix_fmt", "yuv420p"]
        elif self.encoder == "mpeg4":
            cmd += ["-q:v", "2", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-pix_fmt", "yuv420p"]
        cmd += ["-f", "matroska", str(self.capture_path)]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_file = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )
        self._enlarge_pipe(self._process)

    @staticmethod
    def _enlarge_pipe(process: subprocess.Popen) -> None:
        # A larger pipe means fewer wake-ups per multi-megabyte frame. Linux only.
        try:
            import fcntl

            f_setpipe_sz = getattr(fcntl, "F_SETPIPE_SZ", 1031)
            if process.stdin is not None:
                fcntl.fcntl(process.stdin.fileno(), f_setpipe_sz, 1024 * 1024)
        except Exception:
            pass

    def _write_frame(self, frame: Any) -> None:
        import numpy as np

        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Stream encoder is not running.")
        if tuple(frame.shape) != (self._height * 3 // 2, self._width):
            raise RuntimeError(
                f"Frame size changed during recording: expected {(self._height * 3 // 2, self._width)}, got {tuple(frame.shape)}."
            )
        now = time.perf_counter()
        if self._first_write_monotonic is None:
            self._first_write_monotonic = now
        if self.monochrome_output:
            luma = np.ascontiguousarray(frame[: self._height])
            process.stdin.write(luma.reshape(-1).data)
            process.stdin.write(self._neutral_chroma or b"")
        else:
            process.stdin.write(np.ascontiguousarray(frame).reshape(-1).data)
        self._last_write_monotonic = time.perf_counter()
