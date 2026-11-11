"""Tests for the raw-frames versus compressed-video tracking comparison."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from bumblebox_v2 import cli, run_engine
from bumblebox_v2.config import load_defaults, save_config
from bumblebox_v2.stream_writer import StreamingVideoWriter, find_ffmpeg_binary
from bumblebox_v2.tracking_compression_check import (
    LOSSLESS_LABEL,
    RAW_LABEL,
    compare_detections,
    format_tracking_compression_check,
    run_tracking_compression_check,
)

try:
    from tests.test_tracking_and_metrics import TAG_IDS, tagged_yuv420_frame
except ImportError:  # running with `unittest discover -s tests`
    from test_tracking_and_metrics import TAG_IDS, tagged_yuv420_frame

needs_ffmpeg = unittest.skipUnless(find_ffmpeg_binary() is not None, "ffmpeg is required to encode test videos")


def _detections(rows: list[tuple[int, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "ID", "centroidX", "centroidY"])


def _mock_config(tmp: str) -> dict:
    config = load_defaults()
    config["system"]["data_root"] = str(Path(tmp) / "data")
    config["local_index"]["enabled"] = False
    config["camera"].update({"width": 640, "height": 480})
    config["runtime"]["use_mock_camera"] = True
    return config


class CompareDetectionsTests(unittest.TestCase):
    def test_counts_matched_missed_and_extra_detections(self) -> None:
        raw = _detections([(0, 1, 10.0, 10.0), (0, 2, 50.0, 50.0), (1, 1, 12.0, 10.0)])
        video = _detections([(0, 1, 13.0, 14.0), (1, 1, 12.0, 10.0), (1, 3, 90.0, 90.0)])

        result = compare_detections(raw, video)

        self.assertEqual(result["detections"], 3)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["missed"], 1)  # (0, 2) is only in the raw frames
        self.assertEqual(result["extra"], 1)  # (1, 3) is only in the video
        self.assertAlmostEqual(result["retained_percent"], 66.67, places=2)
        self.assertAlmostEqual(result["max_shift_px"], 5.0)  # (0, 1) moved by a 3-4-5 step
        self.assertAlmostEqual(result["median_shift_px"], 2.5)

    def test_same_tag_twice_in_a_frame_counts_once(self) -> None:
        raw = _detections([(0, 1, 10.0, 10.0), (0, 1, 300.0, 300.0)])
        result = compare_detections(raw, raw)
        self.assertEqual((result["detections"], result["matched"], result["missed"], result["extra"]), (1, 1, 0, 0))

    def test_empty_video_tracking_misses_everything(self) -> None:
        raw = _detections([(0, 1, 10.0, 10.0), (1, 1, 11.0, 10.0)])
        result = compare_detections(raw, _detections([]))
        self.assertEqual((result["matched"], result["missed"], result["extra"]), (0, 2, 0))
        self.assertEqual(result["retained_percent"], 0.0)
        self.assertIsNone(result["median_shift_px"])

    def test_no_raw_detections_gives_no_percentage(self) -> None:
        result = compare_detections(_detections([]), _detections([(0, 1, 10.0, 10.0)]))
        self.assertIsNone(result["retained_percent"])
        self.assertEqual(result["extra"], 1)


@needs_ffmpeg
class TrackingCompressionCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        previous = os.getcwd()
        os.chdir(self._tmp.name)
        self.addCleanup(os.chdir, previous)

    def _run(self, **kwargs):
        with patch.object(run_engine, "simulated_rgb_yuv420_frame", tagged_yuv420_frame):
            return run_tracking_compression_check(_mock_config(self._tmp.name), seconds=2, **kwargs)

    def test_every_source_is_compared_with_the_raw_frames(self) -> None:
        report = self._run(encoders=["mjpeg", "libx264:ultrafast"])

        labels = [result.label for result in report.results]
        self.assertEqual(labels, [RAW_LABEL, LOSSLESS_LABEL, "mjpeg", "libx264:ultrafast"])
        raw = report.results[0]
        self.assertEqual(raw.detections, report.frames * len(TAG_IDS))
        for result in report.results[1:]:
            with self.subTest(source=result.label):
                self.assertIsNone(result.error)
                self.assertEqual(result.frames, report.frames)
                # Large, clean synthetic tags survive every encoder.
                self.assertEqual((result.matched, result.missed, result.extra), (raw.detections, 0, 0))
                self.assertEqual(result.retained_percent, 100.0)
                self.assertEqual(result.net_change_percent, 0.0)
                self.assertLess(result.max_shift_px, 1.5)
                self.assertTrue(Path(result.raw_csv_path).exists())

        out_dir = Path(report.output_dir)
        saved = json.loads((out_dir / "tracking_compression_check.json").read_text())
        self.assertEqual(len(saved["results"]), 4)
        # Test videos are removed unless asked for.
        self.assertEqual([p.name for p in out_dir.iterdir() if p.suffix in {".mp4", ".avi", ".mkv"}], [])
        self.assertIn("Equivalent to raw frames", " ".join(report.findings))
        self.assertIn("kept %", format_tracking_compression_check(report))

    def test_videos_can_be_kept_and_the_lossless_reference_skipped(self) -> None:
        report = self._run(encoders=["mjpeg"], include_lossless=False, keep_videos=True)

        self.assertEqual([result.label for result in report.results], [RAW_LABEL, "mjpeg"])
        self.assertTrue(Path(report.results[1].video_path).exists())

    def test_a_failing_encoder_is_reported_without_stopping_the_check(self) -> None:
        report = self._run(encoders=["not_a_real_codec", "mjpeg"], include_lossless=False)

        by_label = {result.label: result for result in report.results}
        self.assertIsNotNone(by_label["not_a_real_codec"].error)
        self.assertIsNone(by_label["mjpeg"].error)
        self.assertIn("failed", format_tracking_compression_check(report))

    def test_scene_without_tags_is_called_out(self) -> None:
        # The stock mock scene has no tags in it.
        report = run_tracking_compression_check(
            _mock_config(self._tmp.name), seconds=1, encoders=["mjpeg"], include_lossless=False
        )
        self.assertEqual(report.results[0].detections, 0)
        self.assertIn("found no tags", report.findings[0])

    def test_lossless_video_reproduces_the_frames_exactly(self) -> None:
        import cv2
        import numpy as np

        frames = [tagged_yuv420_frame(640, 480, index / 5.0) for index in range(4)]
        writer = StreamingVideoWriter(
            output_dir=Path(self._tmp.name), stem="lossless", fps=5.0, mp4_preset="ultrafast", lossless=True
        )
        for frame in frames:
            writer.add(frame)
        video = writer.finish(actual_fps=5.0).video_path

        capture = cv2.VideoCapture(str(video))
        ok, decoded = capture.read()
        capture.release()
        self.assertTrue(ok)
        expected = run_engine._frame_to_bgr(frames[0])
        self.assertLessEqual(int(np.abs(decoded.astype(int) - expected.astype(int)).max()), 1)

    def test_command_prints_the_comparison(self) -> None:
        config_path = Path(self._tmp.name) / "config.yaml"
        save_config(config_path, _mock_config(self._tmp.name))
        with patch.object(run_engine, "simulated_rgb_yuv420_frame", tagged_yuv420_frame):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exit_code = cli.main(
                    [
                        "tracking-compression-check",
                        "--config",
                        str(config_path),
                        "--mock-camera",
                        "--seconds",
                        "2",
                        "--encoders",
                        "mjpeg",
                    ]
                )
        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertIn(RAW_LABEL, output.getvalue())
        self.assertIn(LOSSLESS_LABEL, output.getvalue())
        self.assertIn("mjpeg", output.getvalue())


if __name__ == "__main__":
    unittest.main()
