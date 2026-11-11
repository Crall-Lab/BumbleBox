"""Regression tests for the recording-time tracking and metrics path.

These cover the pieces a default ``record_and_track`` run depends on: parallel
RAM tracking, the behavioural metrics module, and the full ``run_once``
pipeline with the shipped pipeline defaults. They use synthetic frames with
real ArUco markers drawn into them, so no camera is needed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
from cv2 import aruco
import numpy as np
import pandas as pd

import behavioral_metrics
import tag_tracking_utils
from bumblebox_v2 import run_engine
from bumblebox_v2.config import load_defaults


TAG_IDS = (1, 2, 3, 4)  # 17 and 37 are excluded by the default config
_DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
_MARKERS = {
    tag_id: cv2.copyMakeBorder(
        aruco.generateImageMarker(_DICTIONARY, tag_id, 48),
        12, 12, 12, 12,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    for tag_id in TAG_IDS
}


def tagged_yuv420_frame(width: int, height: int, time_s: float, duration: float = 1.0) -> np.ndarray:
    """A YUV420 frame, shaped like a camera frame, with four moving tags."""
    frame = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
    luma = frame[:height]
    luma[:] = 110
    for index, (tag_id, marker) in enumerate(_MARKERS.items()):
        x = int(60 + 140 * index + 10 * time_s * (index + 1))
        y = int(80 + 70 * index + 8 * math.sin(time_s * 3 + index))
        luma[y:y + marker.shape[0], x:x + marker.shape[1]] = marker
    return frame


def _in_directory(test: unittest.TestCase, path: str) -> None:
    """Run the rest of the test from ``path`` (jump logs are written to the cwd)."""
    previous = os.getcwd()
    os.chdir(path)
    test.addCleanup(os.chdir, previous)


class ParallelRamTrackingTests(unittest.TestCase):
    def test_parallel_tracking_finds_the_same_tags_as_serial_tracking(self) -> None:
        frames = [[tagged_yuv420_frame(640, 480, index / 5.0)] for index in range(6)]
        cv_threads_before = cv2.getNumThreads()

        with tempfile.TemporaryDirectory() as tmp:
            serial, _, serial_count = tag_tracking_utils.trackTagsFromRAM(
                "serial", tmp, frames, "4X4_50", "custom", "now", "host", "01",
            )
            parallel, _, parallel_count = tag_tracking_utils.trackTagsFromRAM_parallel(
                "parallel", tmp, frames, "4X4_50", "custom", "now", "01", n_chunks=3,
            )
            self.assertTrue((Path(tmp) / "parallel_raw.csv").exists())
            self.assertTrue((Path(tmp) / "parallel_noID.csv").exists())

        self.assertEqual(serial_count, len(frames))
        self.assertEqual(parallel_count, len(frames))
        self.assertEqual(len(serial), len(frames) * len(TAG_IDS))

        columns = ["frame", "ID", "centroidX", "centroidY", "frontX", "frontY"]
        serial_rows = serial[columns].sort_values(["frame", "ID"]).reset_index(drop=True)
        parallel_rows = parallel[columns].sort_values(["frame", "ID"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(serial_rows, parallel_rows)

        # The parallel tracker turns OpenCV's own threading off while it runs.
        self.assertEqual(cv2.getNumThreads(), cv_threads_before)

    def test_parallel_tracking_accepts_an_empty_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df, df2, count = tag_tracking_utils.trackTagsFromRAM_parallel(
                "empty", tmp, [], "4X4_50", "custom", "now", "01",
            )
        self.assertEqual(count, 0)
        self.assertTrue(df.empty)
        self.assertTrue(df2.empty)


class BehavioralMetricsTests(unittest.TestCase):
    def _tracks(self) -> pd.DataFrame:
        # Frame 0: a 3-4-5 layout. Frame 1: bee 3 is not detected.
        return pd.DataFrame(
            {
                "frame": [0, 0, 0, 1, 1],
                "ID": [1, 2, 3, 1, 2],
                "centroidX": [0.0, 3.0, 0.0, 0.0, 6.0],
                "centroidY": [0.0, 4.0, 8.0, 0.0, 8.0],
            }
        )

    def test_pairwise_distance_reports_distances_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pairwise = behavioral_metrics.pairwise_distance(self._tracks(), tmp, "clip")
            self.assertTrue((Path(tmp) / "clip_pairwise_distance.csv").exists())

        def row(frame: int, bee: int) -> pd.Series:
            match = pairwise[(pairwise["frame"] == frame) & (pairwise["ID"] == bee)]
            self.assertEqual(len(match), 1)
            return match.iloc[0]

        first = row(0, 1)
        self.assertTrue(math.isnan(first[1]))
        self.assertAlmostEqual(first[2], 5.0)
        self.assertAlmostEqual(first[3], 8.0)
        self.assertAlmostEqual(first["avg_distance"], 6.5)
        self.assertAlmostEqual(first["min_distance"], 5.0)
        self.assertAlmostEqual(first["max_distance"], 8.0)

        second = row(1, 2)
        self.assertAlmostEqual(second[1], 10.0)
        self.assertTrue(math.isnan(second[3]))

    def test_contact_matrix_thresholds_pairwise_distances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pairwise = behavioral_metrics.pairwise_distance(self._tracks(), tmp, "clip")
            contacts = behavioral_metrics.contact_matrix(pairwise.copy(), tmp, 6.0, "clip")
            self.assertTrue((Path(tmp) / "clip_contacts.csv").exists())

        first = contacts[(contacts["frame"] == 0) & (contacts["ID"] == 1)].iloc[0]
        self.assertEqual(first[2], 1)  # 5 px apart: contact
        self.assertEqual(first[3], 0)  # 8 px apart: no contact


class DefaultPipelineTests(unittest.TestCase):
    def test_run_once_with_default_pipeline_tracks_tags_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _in_directory(self, tmp)
            config = load_defaults()
            # The pipeline, tracking, cleaning and metrics sections stay at
            # their shipped defaults on purpose: this is the path a new
            # install runs.
            self.assertEqual(config["pipeline"]["mode"], "record_and_track")
            self.assertEqual(config["pipeline"]["tracking_source"], "ram")
            self.assertTrue(config["pipeline"]["parallel_tracking"])
            self.assertTrue(config["pipeline"]["calculate_behavior_metrics"])

            config["system"]["data_root"] = str(Path(tmp) / "data")
            config["local_index"]["enabled"] = False
            config["camera"].update({"width": 640, "height": 480, "codec": "mjpeg"})
            config["capture"]["recording_seconds"] = 2
            config["runtime"].update(
                {
                    "use_mock_camera": True,
                    "fps_report_on_each_recording": False,
                    "render_tracking_video": False,
                }
            )

            with patch.object(run_engine, "simulated_rgb_yuv420_frame", tagged_yuv420_frame):
                summary = run_engine.run_once(config=config)

            self.assertEqual(summary.errors, [])
            self.assertEqual(summary.warnings, [])
            self.assertTrue(summary.success)

            def only(suffix: str) -> Path:
                matches = list((Path(tmp) / "data").rglob(f"*{suffix}"))
                self.assertEqual(len(matches), 1, f"expected exactly one *{suffix}, found {matches}")
                return matches[0]

            raw = pd.read_csv(only("_raw.csv"))
            frame_count = int(raw["frame"].max()) + 1
            self.assertGreaterEqual(frame_count, 8)
            self.assertEqual(sorted(raw["ID"].unique()), list(TAG_IDS))
            self.assertEqual(len(raw), frame_count * len(TAG_IDS))

            only("_cleaned.csv")
            only("_pairwise_distance.csv")
            only("_contacts.csv")
            updated = pd.read_csv(only("_updated.csv"))
            for column in ("speed", "activity", "distance_from_center"):
                self.assertIn(column, updated.columns)
            averages = pd.read_csv(only("_averages.csv"))
            self.assertEqual(sorted(averages["ID"]), list(TAG_IDS))
            self.assertTrue((averages["frames tracked in video"] == frame_count).all())


if __name__ == "__main__":
    unittest.main()
