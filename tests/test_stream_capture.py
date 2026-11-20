"""Tests for stream recording (capture.frame_buffer: stream) and the encode benchmark.

Stream mode encodes to disk while frames are captured, so these tests need
ffmpeg and are skipped where it is not installed. No camera is needed.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from bumblebox_v2 import cli, run_engine
from bumblebox_v2.config import ConfigError, load_defaults, save_config, validate_config
from bumblebox_v2.encode_benchmark import parse_encoder_spec, run_encode_benchmark
from bumblebox_v2.schedule_check import run_schedule_check
from bumblebox_v2.stream_writer import StreamingVideoWriter, find_ffmpeg_binary

try:
    from tests.test_tracking_and_metrics import TAG_IDS, tagged_yuv420_frame
except ImportError:  # running with `unittest discover -s tests`
    from test_tracking_and_metrics import TAG_IDS, tagged_yuv420_frame

HAVE_FFMPEG = find_ffmpeg_binary() is not None
needs_ffmpeg = unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is required for stream recording")


def _frames(count: int, width: int = 320, height: int = 240) -> list[np.ndarray]:
    """Blocky YUV420 frames with a bright square that moves every frame."""
    rng = np.random.default_rng(0)
    frames = []
    for index in range(count):
        frame = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
        blocks = rng.integers(60, 190, (height // 8, width // 8), dtype=np.uint8)
        frame[:height] = blocks.repeat(8, axis=0).repeat(8, axis=1)
        frame[20:60, 10 + 5 * index : 50 + 5 * index] = 250
        frames.append(frame)
    return frames


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return frames, fps


def _stream_config(tmp: str, **camera: object) -> dict:
    config = load_defaults()
    config["system"]["data_root"] = str(Path(tmp) / "data")
    config["local_index"]["enabled"] = False
    config["camera"].update({"width": 640, "height": 480})
    config["camera"].update(camera)
    config["capture"]["recording_seconds"] = 2
    config["capture"]["frame_buffer"] = "stream"
    config["runtime"].update({"use_mock_camera": True, "fps_report_on_each_recording": False})
    return config


def _in_directory(test: unittest.TestCase, path: str) -> None:
    previous = os.getcwd()
    os.chdir(path)
    test.addCleanup(os.chdir, previous)


@needs_ffmpeg
class StreamingVideoWriterTests(unittest.TestCase):
    def test_mp4_keeps_every_frame_and_takes_the_measured_frame_rate(self) -> None:
        frames = _frames(30)
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoWriter(
                output_dir=Path(tmp), stem="clip", fps=5.0, recording_codec="mp4", preview_frame_index=12
            )
            for frame in frames:
                writer.add(frame)
            result = writer.finish(actual_fps=4.4)

            self.assertEqual(result.video_path, Path(tmp) / "clip.mp4")
            self.assertFalse(writer.capture_path.exists(), "the intermediate capture file should be removed")
            decoded, fps = _read_video(result.video_path)

        self.assertEqual(len(decoded), len(frames))
        self.assertEqual(result.frame_count, len(frames))
        self.assertAlmostEqual(fps, 4.4, delta=0.1)
        self.assertEqual(result.warnings, [])
        self.assertIs(result.preview_frame, frames[12])
        self.assertEqual(result.stats.encoder, "libx264")
        self.assertEqual(result.stats.preset, "ultrafast")  # "auto" picks the fastest preset when streaming

    def test_mjpeg_is_written_as_avi_at_the_measured_frame_rate(self) -> None:
        frames = _frames(20)
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoWriter(output_dir=Path(tmp), stem="clip", fps=5.0, recording_codec="mjpeg")
            for frame in frames:
                writer.add(frame)
            result = writer.finish(actual_fps=4.4)
            decoded, fps = _read_video(result.video_path)

        self.assertEqual(result.video_path.suffix, ".avi")
        self.assertEqual(len(decoded), len(frames))
        self.assertAlmostEqual(fps, 4.4, places=2)

    def test_decoded_levels_match_the_ram_mode_conversion(self) -> None:
        frames = _frames(6)
        for monochrome in (False, True):
            for codec in ("mp4", "mjpeg"):
                with self.subTest(codec=codec, monochrome=monochrome), tempfile.TemporaryDirectory() as tmp:
                    writer = StreamingVideoWriter(
                        output_dir=Path(tmp), stem="clip", fps=5.0, recording_codec=codec, monochrome_output=monochrome
                    )
                    for frame in frames:
                        writer.add(frame)
                    decoded, _fps = _read_video(writer.finish(actual_fps=5.0).video_path)
                    # RAM mode converts each frame with _frame_to_bgr before encoding.
                    expected = run_engine._frame_to_bgr(frames[0], monochrome_output=monochrome)
                    difference = np.abs(decoded[0].astype(int) - expected.astype(int)).mean()
                    self.assertLess(difference, 3.0)

    def test_queue_budget_bounds_memory_without_dropping_frames(self) -> None:
        frames = _frames(60)
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoWriter(
                output_dir=Path(tmp),
                stem="clip",
                fps=5.0,
                recording_codec="mp4",
                max_queue_bytes=2 * frames[0].nbytes,
            )
            for frame in frames:
                writer.add(frame)
            result = writer.finish(actual_fps=5.0)
            decoded, _fps = _read_video(result.video_path)

        self.assertEqual(len(decoded), len(frames))
        self.assertLessEqual(result.stats.peak_queue_frames, 2)
        self.assertLessEqual(result.stats.peak_queue_mb, result.stats.queue_budget_mb + 1e-6)

    def test_encoder_failure_raises_instead_of_hanging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoWriter(
                output_dir=Path(tmp), stem="clip", fps=5.0, recording_codec="mp4", mp4_codec="not_a_real_codec"
            )
            with self.assertRaises(RuntimeError) as raised:
                for frame in _frames(40):
                    writer.add(frame)
                    time.sleep(0.01)
                writer.finish(actual_fps=5.0)
            writer.abort()
        self.assertIn("not_a_real_codec", str(raised.exception))

    def test_no_frames_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoWriter(output_dir=Path(tmp), stem="clip", fps=5.0)
            with self.assertRaises(RuntimeError):
                writer.finish()


@needs_ffmpeg
class StreamRunTests(unittest.TestCase):
    def _run(self, config: dict) -> run_engine.RunSummary:
        with patch.object(run_engine, "simulated_rgb_yuv420_frame", tagged_yuv420_frame):
            return run_engine.run_once(config=config)

    def test_stream_run_records_then_tracks_from_the_saved_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _in_directory(self, tmp)
            summary = self._run(_stream_config(tmp))

            self.assertEqual(summary.errors, [])
            self.assertEqual(summary.warnings, [])
            self.assertTrue(summary.success)
            self.assertEqual(summary.capture_frame_buffer, "stream")
            self.assertIsNotNone(summary.stream_encoder_stats)
            self.assertEqual(summary.stream_encoder_stats["frames_written"], summary.frames_captured)

            video = Path(summary.video_path)
            self.assertEqual(video.suffix, ".mp4")
            decoded, fps = _read_video(video)
            self.assertEqual(len(decoded), summary.frames_captured)
            self.assertAlmostEqual(fps, summary.actual_fps, delta=0.1)
            self.assertEqual(list(video.parent.glob("*.capture.mkv")), [])
            self.assertTrue(Path(summary.recording_preview_png_path).exists())

            raw = pd.read_csv(summary.raw_csv_path)
            self.assertEqual(sorted(raw["ID"].unique()), list(TAG_IDS))
            self.assertEqual(len(raw), summary.frames_captured * len(TAG_IDS))
            self.assertTrue(Path(summary.cleaned_csv_path).exists())
            self.assertEqual(len(list(video.parent.glob("*_averages.csv"))), 1)

    def test_stream_and_ram_runs_find_the_same_tags(self) -> None:
        tracks = {}
        with tempfile.TemporaryDirectory() as tmp:
            _in_directory(self, tmp)
            for mode in ("ram", "stream"):
                config = _stream_config(str(Path(tmp) / mode))
                config["capture"]["frame_buffer"] = mode
                summary = self._run(config)
                self.assertTrue(summary.success, summary.errors)
                self.assertEqual(summary.capture_frame_buffer, mode)
                raw = pd.read_csv(summary.raw_csv_path)
                tracks[mode] = raw.sort_values(["frame", "ID"]).reset_index(drop=True)

        self.assertEqual(
            list(zip(tracks["ram"]["frame"], tracks["ram"]["ID"])),
            list(zip(tracks["stream"]["frame"], tracks["stream"]["ID"])),
        )
        for column in ("centroidX", "centroidY"):
            self.assertLess((tracks["ram"][column] - tracks["stream"][column]).abs().max(), 1.5)

    def test_stream_run_with_mjpeg_and_thermal_writes_side_by_side_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _in_directory(self, tmp)
            config = _stream_config(tmp, codec="mjpeg", monochrome_output=True)
            config["pipeline"]["mode"] = "record_only"
            config["thermal"].update({"enabled": True, "width": 16, "height": 12, "fps_target": 5.0})
            summary = self._run(config)

            self.assertTrue(summary.success, summary.errors)
            self.assertEqual(Path(summary.video_path).suffix, ".avi")
            self.assertEqual(summary.thermal_frames_captured, summary.frames_captured)
            side_by_side, _fps = _read_video(Path(summary.thermal_side_by_side_video_path))
            self.assertEqual(len(side_by_side), summary.frames_captured)

    def test_failed_capture_keeps_the_frames_recorded_so_far(self) -> None:
        calls = {"count": 0}

        def failing_frame(width, height, time_s, duration):
            calls["count"] += 1
            if calls["count"] > 6:
                raise RuntimeError("camera unplugged")
            return tagged_yuv420_frame(width, height, time_s, duration)

        with tempfile.TemporaryDirectory() as tmp:
            _in_directory(self, tmp)
            with patch.object(run_engine, "simulated_rgb_yuv420_frame", failing_frame):
                summary = run_engine.run_once(config=_stream_config(tmp))

            self.assertFalse(summary.success)
            self.assertIn("camera unplugged", " ".join(summary.errors))
            partial = list((Path(tmp) / "data").rglob("*.capture.mkv"))
            self.assertEqual(len(partial), 1)
            self.assertIn(partial[0].name, " ".join(summary.warnings))
            decoded, _fps = _read_video(partial[0])
            self.assertEqual(len(decoded), 6)

    def test_run_once_cli_accepts_a_frame_buffer_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _in_directory(self, tmp)
            config = _stream_config(tmp)
            config["capture"]["frame_buffer"] = "ram"
            config_path = Path(tmp) / "config.yaml"
            save_config(config_path, config)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exit_code = cli.main(
                    [
                        "run-once",
                        "--config",
                        str(config_path),
                        "--mock-camera",
                        "--mode",
                        "record_only",
                        "--frame-buffer",
                        "stream",
                        "--mp4-preset",
                        "veryfast",
                    ]
                )
            self.assertEqual(exit_code, 0, output.getvalue())
            self.assertIn("Frame buffer: stream", output.getvalue())
            self.assertIn("libx264 veryfast", output.getvalue())
            self.assertEqual(len(list((Path(tmp) / "data").rglob("*.mp4"))), 1)


class StreamConfigTests(unittest.TestCase):
    def test_defaults_keep_ram_mode(self) -> None:
        config = load_defaults()
        validate_config(config)
        self.assertEqual(config["capture"]["frame_buffer"], "ram")
        self.assertEqual(config["camera"]["mp4_preset"], "auto")

    def test_bad_stream_settings_are_rejected(self) -> None:
        for section, key, value in (
            ("capture", "frame_buffer", "disk"),
            ("capture", "stream_queue_mb", 0),
            ("capture", "stream_queue_mb", "lots"),
            ("camera", "mp4_preset", "warp9"),
        ):
            with self.subTest(key=f"{section}.{key}", value=value):
                config = load_defaults()
                config[section][key] = value
                with self.assertRaises(ConfigError):
                    validate_config(config)

    def test_schedule_check_memory_estimate_is_bounded_in_stream_mode(self) -> None:
        def memory_item(mode: str):
            config = load_defaults()
            config["capture"]["frame_buffer"] = mode
            config["capture"]["recording_seconds"] = 600
            report = run_schedule_check(config, assume_ram_gb=8)
            return next(item for item in report.items if item.name == "Memory budget")

        self.assertEqual(memory_item("ram").status, "FAIL")
        self.assertEqual(memory_item("stream").status, "PASS")


@needs_ffmpeg
class EncodeBenchmarkTests(unittest.TestCase):
    def test_benchmark_measures_each_encoder_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_encode_benchmark(
                load_defaults(),
                frames=6,
                width=320,
                height=240,
                fps_target=5.0,
                encoders=["mjpeg", "libx264:ultrafast", "not_a_real_codec"],
                work_dir=tmp,
            )
            self.assertEqual(os.listdir(tmp), [])

        self.assertEqual((report.width, report.height, report.frames), (320, 240, 6))
        by_label = {result.label: result for result in report.results}
        for label in ("mjpeg", "libx264:ultrafast"):
            self.assertIsNone(by_label[label].error)
            self.assertEqual(by_label[label].frames, 6)
            self.assertGreater(by_label[label].frames_per_second, 0)
            self.assertGreater(by_label[label].megabytes_per_frame, 0)
        self.assertIsNotNone(by_label["not_a_real_codec"].error)
        self.assertIn(report.recommended, ("mjpeg", "libx264:ultrafast"))

    def test_encoder_names_are_parsed(self) -> None:
        self.assertEqual(parse_encoder_spec("MJPEG"), ("mjpeg", "mjpeg", None, None))
        self.assertEqual(parse_encoder_spec("h264:veryfast"), ("libx264:veryfast", "mp4", "libx264", "veryfast"))
        self.assertEqual(parse_encoder_spec("libx264"), ("libx264:ultrafast", "mp4", "libx264", "ultrafast"))

    def test_encode_benchmark_command_prints_a_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exit_code = cli.main(
                    [
                        "encode-benchmark",
                        "--config",
                        str(Path(tmp) / "missing-config.yaml"),
                        "--frames",
                        "4",
                        "--width",
                        "320",
                        "--height",
                        "240",
                        "--encoders",
                        "mjpeg",
                        "--work-dir",
                        tmp,
                    ]
                )
        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertIn("frames/s", output.getvalue())
        self.assertIn("mjpeg", output.getvalue())


def _brightness_ramp(width: int = 512, height: int = 64) -> np.ndarray:
    """A YUV420 frame as the camera delivers it: brightness 0..255 across the width, no colour."""
    frame = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
    frame[:height] = np.repeat(np.arange(256, dtype=np.uint8), width // 256)[None, :]
    return frame


def _decoded_levels(video_path: Path, levels: tuple[int, ...], width: int = 512) -> list[float]:
    """The brightness a recording of ``_brightness_ramp`` plays back at each camera level."""
    frames, _fps = _read_video(video_path)
    row = frames[0][frames[0].shape[0] // 2].astype(float).mean(axis=1)
    step = width // 256
    return [float(row[level * step : (level + 1) * step].mean()) for level in levels]


class GrayscaleLevelTests(unittest.TestCase):
    """A grayscale recording is the camera's brightness image, with nothing clipped.

    The camera uses the whole 0-255 range. The colour conversion treats 16 as
    black and 235 as white, so grayscale made from the colour picture lost the
    darkest and brightest detail.
    """

    LEVELS = (0, 8, 16, 32, 128, 200, 235, 245, 255)

    def test_grayscale_frame_is_the_brightness_plane(self) -> None:
        frame = _brightness_ramp()
        gray = run_engine._frame_to_bgr(frame, monochrome_output=True)
        self.assertEqual(gray.shape, (64, 512, 3))
        for channel in range(3):
            np.testing.assert_array_equal(gray[:, :, channel], frame[:64])
        np.testing.assert_array_equal(run_engine._frame_luma(frame), frame[:64])

    def test_brightness_of_a_colour_picture(self) -> None:
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :] = (0, 255, 0)
        self.assertEqual(run_engine._frame_luma(bgr).shape, (4, 4))
        self.assertEqual(int(run_engine._frame_luma(bgr)[0, 0]), 150)

    def test_colour_picture_keeps_the_dark_and_bright_ends(self) -> None:
        bgr = run_engine._frame_to_bgr(_brightness_ramp())
        row = bgr[10].astype(int)
        for level in self.LEVELS:
            for channel in range(3):
                self.assertAlmostEqual(int(row[level * 2, channel]), level, delta=1, msg=f"camera level {level}")

    def test_colours_follow_the_cameras_own_formula(self) -> None:
        # Full-range ("JPEG") YUV to RGB, the matrix Picamera2 itself uses for this camera setup.
        rng = np.random.default_rng(3)
        height, width = 32, 32
        frame = rng.integers(40, 216, (height * 3 // 2, width), dtype=np.uint8)
        frame[:height] = np.repeat(np.repeat(frame[: height // 2, : width // 2], 2, axis=0), 2, axis=1)
        y = frame[:height].astype(float)
        u = frame[height : height + height // 4].reshape(height // 2, width // 2).astype(float) - 128.0
        v = frame[height + height // 4 :].reshape(height // 2, width // 2).astype(float) - 128.0
        u, v = (np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1) for plane in (u, v))
        expected = np.stack([y + 1.772 * u, y - 0.344 * u - 0.714 * v, y + 1.402 * v], axis=2).clip(0, 255)
        got = run_engine._frame_to_bgr(frame).astype(float)
        self.assertLess(float(np.abs(got - expected).mean()), 2.0)
        self.assertLess(float(np.abs(got - expected).max()), 6.0)

    def test_midframe_picture_keeps_the_dark_and_bright_ends(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            png = run_engine._write_midpoint_preview_png([_brightness_ramp()], Path(tmp), "clip", monochrome_output=True)
            image = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        self.assertEqual([int(image[10, level * 2]) for level in self.LEVELS], list(self.LEVELS))

    @needs_ffmpeg
    def test_recordings_play_back_at_the_cameras_levels(self) -> None:
        frames = [_brightness_ramp() for _ in range(5)]
        cases = [
            (buffered, codec, mp4_codec, monochrome)
            for buffered in ("ram", "stream")
            for codec, mp4_codec in (("mp4", "libx264"), ("mp4", "mpeg4"), ("mjpeg", "libx264"))
            for monochrome in (True, False)
        ]
        for buffered, codec, mp4_codec, monochrome in cases:
            if True:  # keeps the block below at one indent for both modes
                with self.subTest(
                    mode=buffered, codec=codec, mp4_codec=mp4_codec, grayscale=monochrome
                ), tempfile.TemporaryDirectory() as tmp:
                    if buffered == "ram":
                        with contextlib.redirect_stdout(io.StringIO()):
                            video = run_engine._write_recording_video(
                                frames, Path(tmp), "clip", 5.0, 512, 64, codec, mp4_codec, monochrome_output=monochrome
                            )
                    else:
                        writer = StreamingVideoWriter(
                            output_dir=Path(tmp),
                            stem="clip",
                            fps=5.0,
                            recording_codec=codec,
                            mp4_codec=mp4_codec,
                            monochrome_output=monochrome,
                        )
                        for frame in frames:
                            writer.add(frame)
                        video = writer.finish(actual_fps=5.0).video_path
                    played = _decoded_levels(video, self.LEVELS)
                    for level, value in zip(self.LEVELS, played):
                        # Before: 16 played back as 0 and 235 as 255.
                        self.assertAlmostEqual(value, level, delta=3.0, msg=f"camera level {level}")

    def test_h264_and_h265_store_the_cameras_values_and_mark_the_file_full_range(self) -> None:
        from bumblebox_v2.stream_writer import full_range_output_format

        self.assertEqual(full_range_output_format("libx264"), ["-pix_fmt", "yuvj420p", "-color_range", "pc"])
        self.assertEqual(full_range_output_format("libx265"), ["-pix_fmt", "yuvj420p", "-color_range", "pc"])
        self.assertEqual(full_range_output_format("mpeg4"), ["-pix_fmt", "yuv420p"])

    @needs_ffmpeg
    def test_colour_recording_is_the_cameras_frame_bit_for_bit_when_lossless(self) -> None:
        # No RGB step and no level change: lossless H.264 gives back exactly what the camera delivered.
        import subprocess

        frames = _frames(4)
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoWriter(output_dir=Path(tmp), stem="clip", fps=5.0, lossless=True)
            for frame in frames:
                writer.add(frame)
            video = writer.finish(actual_fps=5.0).video_path
            raw = subprocess.run(
                [find_ffmpeg_binary(), "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "yuvj420p", "-"],
                capture_output=True,
                check=True,
            ).stdout
        size = frames[0].size
        self.assertEqual(len(raw), size * len(frames))
        for index, frame in enumerate(frames):
            decoded = np.frombuffer(raw[index * size : (index + 1) * size], dtype=np.uint8).reshape(frame.shape)
            np.testing.assert_array_equal(decoded, frame)


if __name__ == "__main__":
    unittest.main()
