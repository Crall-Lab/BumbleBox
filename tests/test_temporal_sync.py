from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bumblebox_v2.config import load_defaults
from bumblebox_v2.run_engine import run_once
from bumblebox_v2.temporal_sync import analyze_session_sync, estimate_trace_offset


class TemporalSyncTests(unittest.TestCase):
    def test_estimates_known_shared_cue_offset_and_jitter(self) -> None:
        times = np.arange(0.0, 10.0, 0.05)
        reference = np.exp(-((times - 3.0) / 0.08) ** 2) + np.exp(
            -((times - 6.2) / 0.08) ** 2
        )
        sensor = np.exp(-((times - 3.3) / 0.08) ** 2) + np.exp(
            -((times - 6.5) / 0.08) ** 2
        )
        estimate = estimate_trace_offset(
            times,
            reference,
            times,
            sensor,
            sensor_stream="thermal",
            max_lag_seconds=1.0,
            sample_hz=40.0,
        )
        self.assertAlmostEqual(estimate.offset_seconds, -0.3, places=2)
        self.assertGreater(estimate.peak_correlation, 0.95)
        self.assertEqual(estimate.matched_transition_count, 2)
        self.assertLess(float(estimate.residual_jitter_seconds or 0.0), 0.01)

    def test_analyzes_generated_three_sensor_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_defaults()
            config["system"]["data_root"] = str(root)
            config["local_index"]["enabled"] = False
            config["runtime"].update(
                use_mock_camera=True,
                fps_report_on_each_recording=False,
                render_tracking_video=False,
            )
            config["capture"]["recording_seconds"] = 1.0
            config["camera"].update(width=160, height=120, fps_target=6, codec="mjpeg")
            config["thermal"].update(enabled=True, width=32, height=24)
            config["realsense"].update(
                enabled=True,
                depth_width=64,
                depth_height=48,
                color_width=64,
                color_height=48,
                fps=8,
            )
            with redirect_stdout(io.StringIO()):
                summary = run_once(config, mode_override="record_only")
            summary_path = (
                Path(summary.session_dir) / f"{summary.session_name}_run_summary.json"
            )
            report = analyze_session_sync(summary_path, max_lag_seconds=0.4)
            self.assertTrue(report["success"])
            self.assertEqual(
                {item["sensor_stream"] for item in report["estimates"]},
                {"thermal", "realsense"},
            )
            self.assertTrue(Path(report["output_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
