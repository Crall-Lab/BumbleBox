from __future__ import annotations

import time
import unittest

import numpy as np

from bumblebox_v2.cli import build_parser
from bumblebox_v2.config import load_defaults
from bumblebox_v2.live_monitor import (
    DepthColorizer,
    LiveMonitorBackend,
    LiveMonitorOptions,
    ThermalColorizer,
    resolve_local_monitor_sensors,
)


class LiveMonitorTests(unittest.TestCase):
    def test_cli_defaults_use_reasonable_owlsight_preview(self) -> None:
        args = build_parser().parse_args(["live-monitor"])

        self.assertEqual(args.rgb_width, 1920)
        self.assertEqual(args.rgb_height, 1440)
        self.assertEqual(args.display_fps, 10.0)

    def test_local_sensor_assignment_excludes_remote_camera(self) -> None:
        config = load_defaults()
        config["thermal"]["enabled"] = True
        config["realsense"]["enabled"] = True
        config["distributed_capture"].update(
            {
                "enabled": True,
                "nodes": [
                    {
                        "name": "primary",
                        "local": True,
                        "enabled": True,
                        "sensors": ["rgb", "thermal"],
                    },
                    {
                        "name": "worker",
                        "local": False,
                        "enabled": True,
                        "sensors": ["realsense"],
                    },
                ],
            }
        )

        sensors = resolve_local_monitor_sensors(config)

        self.assertEqual(sensors, {"rgb", "thermal"})

    def test_colorizers_return_display_ready_bgr(self) -> None:
        thermal = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16) + 29000
        depth = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16) + 300
        depth[0, 0] = 0

        thermal_bgr = ThermalColorizer().colorize(thermal)
        depth_bgr = DepthColorizer().colorize(depth)

        self.assertEqual(thermal_bgr.shape, (12, 16, 3))
        self.assertEqual(depth_bgr.shape, (12, 16, 3))
        self.assertEqual(thermal_bgr.dtype, np.uint8)
        self.assertEqual(depth_bgr.dtype, np.uint8)
        self.assertEqual(depth_bgr[0, 0].tolist(), [0, 0, 0])

    def test_mock_backend_keeps_latest_frame_for_all_four_views(self) -> None:
        config = load_defaults()
        config["runtime"]["use_mock_camera"] = True
        config["thermal"]["enabled"] = True
        config["realsense"]["enabled"] = True
        config["thermal"].update({"width": 16, "height": 12, "fps_target": 8.0})
        config["realsense"].update(
            {
                "depth_width": 32,
                "depth_height": 24,
                "color_width": 32,
                "color_height": 24,
            }
        )
        statuses: list[tuple[str, str, str]] = []
        backend = LiveMonitorBackend(
            config,
            LiveMonitorOptions(rgb_width=64, rgb_height=48, display_fps=12.0),
            status_callback=lambda *args: statuses.append(args),
        )

        backend.start()
        deadline = time.monotonic() + 2.0
        packets = {}
        while time.monotonic() < deadline:
            packets = backend.frame_store.snapshot()
            if set(packets) == {
                "rgb",
                "thermal",
                "realsense_color",
                "realsense_depth",
            }:
                break
            time.sleep(0.02)
        backend.stop()

        self.assertEqual(
            set(packets),
            {"rgb", "thermal", "realsense_color", "realsense_depth"},
        )
        self.assertEqual(packets["rgb"].image_bgr.shape, (48, 64, 3))
        self.assertEqual(packets["thermal"].image_bgr.shape, (12, 16, 3))
        self.assertTrue(any(stream == "rgb" and state == "live" for stream, state, _ in statuses))

    def test_options_reject_odd_yuv_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            LiveMonitorOptions(rgb_width=1919, rgb_height=1440).validate()


if __name__ == "__main__":
    unittest.main()
