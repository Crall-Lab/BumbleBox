from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bumblebox_v2.config import load_defaults
from bumblebox_v2.run_engine import run_once
from bumblebox_v2.simulated_capture import (
    simulated_realsense_depth_frame,
    simulated_target_center,
    simulated_thermal_frame,
)


class SimulatedCaptureTests(unittest.TestCase):
    def test_shared_target_uses_the_same_normalized_position(self) -> None:
        time_s = 0.75
        duration = 2.0
        thermal_center = simulated_target_center(160, 120, time_s, duration)
        depth_center = simulated_target_center(848, 480, time_s, duration)

        self.assertAlmostEqual(thermal_center[0] / 159.0, depth_center[0] / 847.0, places=2)
        self.assertAlmostEqual(thermal_center[1] / 119.0, depth_center[1] / 479.0, places=2)

        thermal = simulated_thermal_frame(160, 120, time_s, duration)
        depth = simulated_realsense_depth_frame(848, 480, time_s, duration)
        self.assertEqual(int(thermal[thermal_center[1], thermal_center[0]]), 30500)
        self.assertEqual(int(depth[depth_center[1], depth_center[0]]), 525)

    def test_run_once_writes_three_camera_mock_artifacts_on_one_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_defaults()
            config["system"]["data_root"] = tmp
            config["system"]["colony_id"] = "mock"
            config["local_index"]["enabled"] = False
            config["pipeline"]["mode"] = "record_only"
            config["camera"].update(
                {
                    "profile": "custom",
                    "model": "auto",
                    "width": 64,
                    "height": 48,
                    "fps_target": 4.0,
                    "codec": "mjpeg",
                    "infrared": False,
                }
            )
            config["capture"]["recording_seconds"] = 0.5
            config["thermal"].update(
                {
                    "enabled": True,
                    "width": 16,
                    "height": 12,
                    "fps_target": 4.0,
                    "pixel_format": "y16",
                }
            )
            config["realsense"].update(
                {
                    "enabled": True,
                    "depth_width": 64,
                    "depth_height": 48,
                    "color_width": 64,
                    "color_height": 48,
                    "fps": 6,
                    "save_depth": True,
                    "save_color": True,
                }
            )
            config["runtime"].update(
                {
                    "use_mock_camera": True,
                    "fps_report_on_each_recording": False,
                    "render_tracking_video": False,
                }
            )

            summary = run_once(config, mode_override="record_only")

            self.assertTrue(summary.success, summary.errors)
            self.assertEqual(summary.frames_captured, 2)
            self.assertEqual(summary.thermal_frames_captured, 2)
            self.assertEqual(summary.realsense_frames_captured, 3)
            self.assertEqual(summary.thermal_device_path, "mock://thermal")
            self.assertEqual(summary.realsense_device_serial, "mock://realsense")

            artifact_paths = [
                summary.video_path,
                summary.timestamp_path,
                summary.thermal_raw_npy_path,
                summary.thermal_timestamp_path,
                summary.thermal_preview_video_path,
                summary.thermal_side_by_side_video_path,
                summary.realsense_raw_depth_npy_path,
                summary.realsense_timestamp_path,
                summary.realsense_depth_preview_video_path,
                summary.realsense_color_video_path,
                summary.realsense_metadata_json_path,
            ]
            for path in artifact_paths:
                self.assertIsNotNone(path)
                self.assertTrue(Path(str(path)).exists(), path)

            thermal_stack = np.load(str(summary.thermal_raw_npy_path))
            depth_stack = np.load(str(summary.realsense_raw_depth_npy_path))
            self.assertEqual(thermal_stack.shape, (2, 12, 16))
            self.assertEqual(depth_stack.shape, (3, 48, 64))

            def read_rows(path: str) -> list[dict[str, str]]:
                with Path(path).open(newline="") as handle:
                    return list(csv.DictReader(handle))

            rgb_rows = read_rows(str(summary.timestamp_path))
            thermal_rows = read_rows(str(summary.thermal_timestamp_path))
            depth_rows = read_rows(str(summary.realsense_timestamp_path))
            self.assertEqual(len(rgb_rows), 2)
            self.assertEqual(len(thermal_rows), 2)
            self.assertEqual(len(depth_rows), 3)
            self.assertAlmostEqual(
                float(rgb_rows[0]["captured_monotonic_s"]),
                float(thermal_rows[0]["captured_monotonic_s"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(rgb_rows[0]["captured_monotonic_s"]),
                float(depth_rows[0]["host_receive_monotonic_s"]),
                places=6,
            )

            metadata = json.loads(Path(str(summary.realsense_metadata_json_path)).read_text())
            self.assertTrue(metadata["simulated"])
            self.assertEqual(metadata["simulation_signal"], "shared_normalized_moving_target")
            self.assertEqual(metadata["depth_frame_number_gaps"], 0)


if __name__ == "__main__":
    unittest.main()
