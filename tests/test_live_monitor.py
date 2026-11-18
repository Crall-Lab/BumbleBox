from __future__ import annotations

import os
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
    create_live_monitor_window,
    monitor_grid,
    monitor_streams,
    resolve_local_monitor_sensors,
    rgb_stream_title,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from bumblebox_v2.qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
except Exception:  # pragma: no cover - optional on non-GUI test hosts
    QApplication = None


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


class MonitorLayoutTests(unittest.TestCase):
    def test_only_the_sensors_in_use_get_a_feed(self) -> None:
        self.assertEqual(monitor_streams({"rgb"}), ["rgb"])
        self.assertEqual(monitor_streams({"rgb", "thermal"}), ["rgb", "thermal"])
        self.assertEqual(monitor_streams({"rgb", "realsense"}), ["rgb", "realsense_color", "realsense_depth"])
        self.assertEqual(
            monitor_streams({"rgb", "thermal", "realsense"}),
            ["rgb", "thermal", "realsense_color", "realsense_depth"],
        )
        self.assertEqual(monitor_streams({"thermal"}), ["thermal"])
        self.assertEqual(monitor_streams(set()), [])

    def test_grid_for_one_to_four_feeds(self) -> None:
        self.assertEqual(monitor_grid(0), [])
        self.assertEqual(monitor_grid(1), [(0, 0, 1, 1)])
        self.assertEqual(monitor_grid(2), [(0, 0, 1, 1), (0, 1, 1, 1)])
        # Three feeds: the first is large on the left, the other two stacked beside it.
        self.assertEqual(monitor_grid(3), [(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)])
        self.assertEqual(monitor_grid(4), [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)])

    def test_rgb_feed_is_named_after_the_camera_profile(self) -> None:
        config = load_defaults()
        self.assertEqual(rgb_stream_title(config), "RGB camera")
        config["camera"]["profile"] = "hq_reference"
        self.assertEqual(rgb_stream_title(config), "RGB camera (HQ Reference)")


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class MonitorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def window(self, *, thermal: bool = False, depth: bool = False, configure=None):
        config = load_defaults()
        config["runtime"]["use_mock_camera"] = True
        config["thermal"]["enabled"] = thermal
        config["realsense"]["enabled"] = depth
        if configure is not None:
            configure(config)
        window = create_live_monitor_window(config, LiveMonitorOptions(rgb_width=64, rgb_height=48))
        self.addCleanup(window.close)
        window.show()
        return window

    def shown(self, window) -> list[str]:
        return [stream for stream, panel in window.panels.items() if not panel.isHidden()]

    def test_camera_only_setup_shows_one_feed(self) -> None:
        window = self.window()
        self.assertEqual(list(window.panels), ["rgb"])
        self.assertEqual(self.shown(window), ["rgb"])
        self.assertTrue(window.empty_label.isHidden())
        row, column, row_span, column_span = window.grid.getItemPosition(window.grid.indexOf(window.panels["rgb"]))
        self.assertEqual((row, column, row_span, column_span), (0, 0, 1, 1))
        # A status for a sensor this setup does not use must not create or need a panel.
        window._set_status("thermal", "disabled", "Not enabled on this Pi.")
        self.assertEqual(list(window.panels), ["rgb"])

    def test_each_enabled_sensor_adds_its_feeds(self) -> None:
        self.assertEqual(list(self.window(thermal=True).panels), ["rgb", "thermal"])
        three = self.window(depth=True)
        self.assertEqual(list(three.panels), ["rgb", "realsense_color", "realsense_depth"])
        position = three.grid.getItemPosition(three.grid.indexOf(three.panels["rgb"]))
        self.assertEqual(position, (0, 0, 2, 1))  # the main camera spans both rows
        self.assertEqual(
            list(self.window(thermal=True, depth=True).panels),
            ["rgb", "thermal", "realsense_color", "realsense_depth"],
        )

    def test_any_feed_can_fill_the_screen_and_come_back(self) -> None:
        window = self.window(thermal=True, depth=True)
        window.toggle_full_view("thermal")
        self.assertEqual(window.full_view, "thermal")
        self.assertEqual(self.shown(window), ["thermal"])
        self.assertTrue(window.isFullScreen())
        self.assertTrue(window.header.isHidden())
        self.assertTrue(window.note.isHidden())
        self.assertEqual(window.panels["thermal"].full_view_button.text(), "Exit Full Screen (Esc)")
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(window.panels["thermal"])), (0, 0, 1, 1))

        # Straight from one feed to another.
        window.toggle_full_view("realsense_depth")
        self.assertEqual(self.shown(window), ["realsense_depth"])
        self.assertEqual(window.panels["thermal"].full_view_button.text(), "Full Screen")

        window.toggle_full_view("realsense_depth")
        self.assertIsNone(window.full_view)
        self.assertFalse(window.isFullScreen())
        self.assertEqual(self.shown(window), ["rgb", "thermal", "realsense_color", "realsense_depth"])
        self.assertFalse(window.header.isHidden())
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(window.panels["realsense_depth"])), (1, 1, 1, 1))

    def test_button_double_click_and_escape(self) -> None:
        window = self.window(thermal=True)
        window.panels["rgb"].full_view_button.click()
        self.assertEqual(window.full_view, "rgb")
        QTest.keyClick(window, Qt.Key_Escape)
        self.assertIsNone(window.full_view)

        QTest.mouseDClick(window.panels["thermal"].surface, Qt.LeftButton)
        self.assertEqual(window.full_view, "thermal")
        QTest.mouseDClick(window.panels["thermal"].surface, Qt.LeftButton)
        self.assertIsNone(window.full_view)

        QTest.keyClick(window, Qt.Key_F11)
        self.assertEqual(window.full_view, "rgb")
        QTest.keyClick(window, Qt.Key_F11)
        self.assertIsNone(window.full_view)

    def test_single_feed_setup_can_go_full_screen_too(self) -> None:
        window = self.window()
        window.toggle_full_view("rgb")
        self.assertTrue(window.isFullScreen())
        window.exit_full_view()
        self.assertEqual(self.shown(window), ["rgb"])

    def test_frames_still_arrive_in_full_screen(self) -> None:
        window = self.window(thermal=True)
        window.toggle_full_view("rgb")
        deadline = time.monotonic() + 5
        while window.panels["rgb"].last_sequence == 0 and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertGreater(window.panels["rgb"].last_sequence, 0)
        self.assertEqual(window.panels["rgb"].state, "live")

    def test_setup_with_no_local_camera_says_so(self) -> None:
        def other_pi_has_everything(config):
            config["distributed_capture"]["enabled"] = True
            config["distributed_capture"]["nodes"] = [
                {"name": "this", "local": True, "sensors": []},
                {"name": "other", "local": False, "host": "pi2.local", "sensors": ["rgb"]},
            ]

        window = self.window(configure=other_pi_has_everything)
        self.assertEqual(window.panels, {})
        self.assertFalse(window.empty_label.isHidden())
        window.toggle_full_view("rgb")  # nothing to show; must not fail
        self.assertIsNone(window.full_view)


if __name__ == "__main__":
    unittest.main()
