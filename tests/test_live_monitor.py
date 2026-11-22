from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from bumblebox_v2.cli import build_parser
from bumblebox_v2.config import load_config, load_defaults, save_config
from bumblebox_v2.recording_options import recording_is_grayscale, set_recording_grayscale
from bumblebox_v2.live_monitor import (
    DepthColorizer,
    LiveMonitorBackend,
    LiveMonitorOptions,
    ThermalColorizer,
    bgr_to_gray,
    boost_contrast,
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

    def test_two_cameras_share_the_top_row_with_the_other_feeds_below(self) -> None:
        self.assertEqual(monitor_grid(2, pair_first=True), [(0, 0, 1, 1), (0, 1, 1, 1)])  # side by side
        self.assertEqual(monitor_grid(3, pair_first=True), [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 2)])
        self.assertEqual(
            monitor_grid(4, pair_first=True), [(0, 0, 1, 2), (0, 2, 1, 2), (1, 0, 1, 2), (1, 2, 1, 2)]
        )
        self.assertEqual(
            monitor_grid(5, pair_first=True),
            [(0, 0, 1, 3), (0, 3, 1, 3), (1, 0, 1, 2), (1, 2, 1, 2), (1, 4, 1, 2)],
        )

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


class GrayscaleHelperTests(unittest.TestCase):
    def test_luma_of_primary_colours(self) -> None:
        image = np.zeros((1, 5, 3), dtype=np.uint8)
        image[0, 0] = (255, 0, 0)  # blue (BGR)
        image[0, 1] = (0, 255, 0)  # green
        image[0, 2] = (0, 0, 255)  # red
        image[0, 3] = (255, 255, 255)
        image[0, 4] = (90, 90, 90)
        gray = bgr_to_gray(image)
        self.assertEqual(gray.shape, (1, 5))
        self.assertEqual(gray.dtype, np.uint8)
        blue, green, red, white, mid = (int(value) for value in gray[0])
        self.assertTrue(green > red > blue)  # the eye, and the camera's luma, weight green most
        self.assertEqual(white, 255)
        self.assertEqual(mid, 90)
        with self.assertRaises(ValueError):
            bgr_to_gray(np.zeros((4, 4), dtype=np.uint8))

    def test_contrast_boost_brings_out_faint_detail(self) -> None:
        # A dim picture with a faint pattern, as in a dark nest.
        rng = np.random.default_rng(1)
        faint = (40 + rng.integers(0, 12, (96, 128))).astype(np.uint8)
        boosted = boost_contrast(faint)
        self.assertEqual(boosted.shape, faint.shape)
        self.assertEqual(boosted.dtype, np.uint8)
        self.assertGreater(float(boosted.std()), 2.0 * float(faint.std()))
        with self.assertRaises(ValueError):
            boost_contrast(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_recording_setting_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_config(path, load_defaults())
            self.assertFalse(recording_is_grayscale(load_config(path)))
            change = set_recording_grayscale(path, True)
            self.assertTrue(change.grayscale)
            self.assertIsNone(change.left_profile)
            self.assertTrue(recording_is_grayscale(load_config(path)))
            self.assertIn("saved in grayscale", change.message)
            self.assertFalse(set_recording_grayscale(path, False).grayscale)
            self.assertFalse(load_config(path)["camera"]["monochrome_output"])

    def test_profile_that_fixes_colour_is_left_only_when_it_has_to_be(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config = load_defaults()
            config["camera"]["profile"] = "hq_reference"  # this profile records in colour
            save_config(path, config)
            unchanged = set_recording_grayscale(path, False)
            self.assertIsNone(unchanged.left_profile)
            self.assertEqual(load_config(path)["camera"]["profile"], "hq_reference")

            change = set_recording_grayscale(path, True)
            self.assertEqual(change.left_profile, "HQ Reference")
            self.assertIn("Custom settings", change.message)
            saved = load_config(path)
            self.assertEqual(saved["camera"]["profile"], "custom")
            self.assertTrue(saved["camera"]["monochrome_output"])
            self.assertEqual(saved["camera"]["fps_target"], 7.0)  # the profile's other values stay


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class MonitorGrayscaleTests(unittest.TestCase):
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

    @staticmethod
    def colour_frame() -> np.ndarray:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:, :] = (30, 200, 90)  # clearly not gray
        return frame

    @staticmethod
    def is_gray(surface) -> bool:
        pixel = surface._source.toImage().pixelColor(4, 4)
        return pixel.red() == pixel.green() == pixel.blue()

    def test_grayscale_view_changes_the_picture_of_the_colour_cameras(self) -> None:
        window = self.window(thermal=True, depth=True)
        for panel in window.panels.values():
            panel.surface.set_bgr(self.colour_frame())
        self.assertFalse(self.is_gray(window.panels["rgb"].surface))
        self.assertEqual(window.gray_view_button.text(), "Grayscale View: Off")

        window.gray_view_button.click()
        self.assertTrue(window.grayscale_view)
        self.assertEqual(window.gray_view_button.text(), "Grayscale View: On")
        self.assertTrue(self.is_gray(window.panels["rgb"].surface))  # redrawn at once, even with no new frame
        self.assertTrue(self.is_gray(window.panels["realsense_color"].surface))
        # Thermal and depth use a colour scale that carries the measurement; they stay as they are.
        self.assertFalse(self.is_gray(window.panels["thermal"].surface))
        self.assertFalse(self.is_gray(window.panels["realsense_depth"].surface))

        window.gray_view_button.click()
        self.assertFalse(self.is_gray(window.panels["rgb"].surface))

    def test_g_key_switches_the_view_also_in_full_screen(self) -> None:
        window = self.window()
        window.panels["rgb"].surface.set_bgr(self.colour_frame())
        window.toggle_full_view("rgb")
        QTest.keyClick(window, Qt.Key_G)
        self.assertTrue(window.grayscale_view)
        self.assertTrue(window.gray_view_button.isChecked())
        self.assertTrue(self.is_gray(window.panels["rgb"].surface))
        QTest.keyClick(window, Qt.Key_G)
        self.assertFalse(window.grayscale_view)

    @staticmethod
    def drawn(surface) -> np.ndarray:
        """The picture on a surface, before it is scaled to the window, as gray values."""
        image = surface._source.toImage()
        return np.array(
            [[image.pixelColor(x, y).red() for x in range(image.width())] for y in range(image.height())],
            dtype=np.uint8,
        )

    @staticmethod
    def camera_brightness() -> np.ndarray:
        """A brightness image with values the colour conversion would clip (below 16, above 235)."""
        gray = np.zeros((12, 16), dtype=np.uint8)
        gray[:, :] = np.linspace(2, 250, 16, dtype=np.uint8)[None, :]
        return gray

    def test_grayscale_view_shows_the_cameras_brightness_image_untouched(self) -> None:
        window = self.window()
        surface = window.panels["rgb"].surface
        gray = self.camera_brightness()
        surface.set_bgr(self.colour_frame(), gray=gray)
        window.gray_view_button.click()
        np.testing.assert_array_equal(self.drawn(surface), gray)
        # Without a brightness image the colour picture is reduced to gray instead.
        surface.set_bgr(self.colour_frame())
        np.testing.assert_array_equal(self.drawn(surface), bgr_to_gray(self.colour_frame()))
        # One that does not fit the picture is ignored.
        surface.set_bgr(self.colour_frame(), gray=np.zeros((3, 3), dtype=np.uint8))
        np.testing.assert_array_equal(self.drawn(surface), bgr_to_gray(self.colour_frame()))

    def test_the_camera_feed_carries_its_brightness_image(self) -> None:
        from bumblebox_v2.live_monitor import LiveFramePacket
        from bumblebox_v2.run_engine import _frame_luma
        from bumblebox_v2.simulated_capture import simulated_rgb_yuv420_frame

        config = load_defaults()
        config["runtime"]["use_mock_camera"] = True
        backend = LiveMonitorBackend(config, LiveMonitorOptions(rgb_width=64, rgb_height=48, display_fps=20.0))
        backend.start()
        deadline = time.monotonic() + 2.0
        packet = None
        while time.monotonic() < deadline and packet is None:
            packet = backend.frame_store.snapshot().get("rgb")
            time.sleep(0.02)
        backend.stop()
        self.assertIsNotNone(packet)
        self.assertEqual(packet.image_gray.shape, (48, 64))
        self.assertEqual(packet.image_bgr.shape, (48, 64, 3))
        frame = simulated_rgb_yuv420_frame(64, 48, 0.0, 4.0)
        np.testing.assert_array_equal(_frame_luma(frame), frame[:48])
        self.assertIsNone(LiveFramePacket("thermal", np.zeros((2, 2, 3)), 1, 0.0).image_gray)

    def test_contrast_boost_is_a_grayscale_view(self) -> None:
        window = self.window(depth=True)
        surface = window.panels["rgb"].surface
        gray = self.camera_brightness()
        surface.set_bgr(self.colour_frame(), gray=gray)
        self.assertEqual(window.boost_button.text(), "Contrast Boost: Off")

        window.boost_button.click()  # turns the grayscale view on with it
        self.assertTrue(window.contrast_boost and window.grayscale_view)
        self.assertTrue(window.gray_view_button.isChecked())
        self.assertEqual(window.boost_button.text(), "Contrast Boost: On")
        self.assertEqual(window.gray_view_button.text(), "Grayscale View: On")
        np.testing.assert_array_equal(self.drawn(surface), boost_contrast(gray))

        window.boost_button.click()  # boost off: plain grayscale stays
        self.assertFalse(window.contrast_boost)
        self.assertTrue(window.grayscale_view)
        np.testing.assert_array_equal(self.drawn(surface), gray)

        window.boost_button.click()
        window.gray_view_button.click()  # back to colour: the boost goes with it
        self.assertFalse(window.contrast_boost or window.grayscale_view)
        self.assertFalse(window.boost_button.isChecked())
        self.assertEqual(window.boost_button.text(), "Contrast Boost: Off")
        self.assertFalse(self.is_gray(surface))

    def test_c_key_switches_the_boost_also_in_full_screen(self) -> None:
        window = self.window()
        window.panels["rgb"].surface.set_bgr(self.colour_frame(), gray=self.camera_brightness())
        window.toggle_full_view("rgb")
        QTest.keyClick(window, Qt.Key_C)
        self.assertTrue(window.contrast_boost and window.grayscale_view)
        QTest.keyClick(window, Qt.Key_C)
        self.assertFalse(window.contrast_boost)
        self.assertTrue(window.grayscale_view)

    def test_the_camera_panel_shows_the_gain_and_white_balance_in_use(self) -> None:
        from bumblebox_v2.live_monitor import LiveFramePacket

        window = self.window()
        panel = window.panels["rgb"]
        now = time.perf_counter()
        panel.set_packet(
            LiveFramePacket("rgb", self.colour_frame(), 9001, now, info="sensor gain 3.81  |  white balance red 1.92, blue 1.45")
        )
        self.assertTrue(panel.status_label.text().endswith("sensor gain 3.81  |  white balance red 1.92, blue 1.45"))
        panel.set_packet(LiveFramePacket("rgb", self.colour_frame(), 9002, now))
        self.assertNotIn("sensor gain", panel.status_label.text())  # the mock camera reports none

    def test_monitor_only_looks_it_has_no_recording_controls(self) -> None:
        window = self.window()
        self.assertFalse(hasattr(window, "record_gray_button"))
        labels = [button.text() for button in window.findChildren(type(window.gray_view_button))]
        self.assertFalse(any("Record" in label for label in labels), labels)
        self.assertIn("nothing here changes how recordings are made", window.note.text())

    def test_no_view_button_without_a_colour_feed(self) -> None:
        def thermal_only_here(config):
            config["distributed_capture"]["enabled"] = True
            config["distributed_capture"]["nodes"] = [
                {"name": "this", "local": True, "host": "pi1.local", "sensors": ["thermal"]},
                {"name": "other", "local": False, "host": "pi2.local", "sensors": ["rgb"]},
            ]

        window = self.window(thermal=True, configure=thermal_only_here)
        self.assertEqual(list(window.panels), ["thermal"])
        self.assertTrue(window.gray_view_button.isHidden())
        self.assertTrue(window.boost_button.isHidden())



class FakeCameraSession:
    """Stands in for the camera session: records how each opening chose its tuning."""

    openings: list = []

    def __init__(self, config, *, tuning_config=None, **given) -> None:
        from bumblebox_v2.tuning import resolve_camera_tuning_file

        self.config = config
        self.given = given
        self.size = (config["camera"]["width"], config["camera"]["height"])
        if "tuning_file" in given:
            self.resolved_tuning_file = given["tuning_file"]
        else:
            # No camera to ask here, so the file comes from the config alone.
            self.resolved_tuning_file = resolve_camera_tuning_file(
                tuning_config if tuning_config is not None else config
            )
        self.picam2 = self
        self.closed = False
        FakeCameraSession.openings.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.closed = True

    def start(self) -> None:
        pass

    def capture_array(self):
        from bumblebox_v2.simulated_capture import simulated_rgb_yuv420_frame

        return simulated_rgb_yuv420_frame(self.size[0], self.size[1], 0.0, 4.0)


class MonitorNoirBackendTests(unittest.TestCase):
    """The monitor's own NoIR switch reopens its camera; recordings are not touched."""

    def setUp(self) -> None:
        FakeCameraSession.openings = []
        patcher = mock.patch("bumblebox_v2.run_engine._PicameraCaptureSession", FakeCameraSession)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.statuses: list[tuple[str, str, str]] = []

    def backend(self, configure=None, **options) -> LiveMonitorBackend:
        config = load_defaults()
        if configure is not None:
            configure(config)
        self.config = config
        backend = LiveMonitorBackend(
            config,
            LiveMonitorOptions(rgb_width=64, rgb_height=48, display_fps=50.0, **options),
            status_callback=lambda *args: self.statuses.append(args),
        )
        self.addCleanup(backend.stop)
        return backend

    def wait(self, until, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not until():
            if time.monotonic() > deadline:
                self.fail("timed out waiting for the monitor's camera")
            time.sleep(0.01)

    def test_the_monitor_uses_the_recordings_tuning_file_at_its_own_smaller_size(self) -> None:
        # Camera model on auto: the file used to be guessed from the size the
        # camera was opened at, and the monitor's size matched nothing.
        backend = self.backend()
        backend.start()
        self.wait(lambda: backend.frame_store.snapshot().get("rgb") is not None)
        opening = FakeCameraSession.openings[0]
        self.assertEqual(opening.size, (64, 48))
        self.assertEqual(opening.resolved_tuning_file, "imx477_noir.json")
        packet = backend.frame_store.snapshot()["rgb"]
        self.assertTrue(packet.info.startswith("NoIR tuning (imx477_noir.json)"), packet.info)
        self.assertTrue(any("NoIR tuning (imx477_noir.json)" in message for _, state, message in self.statuses if state == "live"))

    def test_switching_noir_off_and_on_reopens_the_camera(self) -> None:
        backend = self.backend()
        self.assertTrue(backend.noir)  # starts where recordings are
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 1)

        backend.set_noir(False)
        self.wait(lambda: len(FakeCameraSession.openings) == 2)
        self.assertTrue(FakeCameraSession.openings[0].closed)  # one camera cannot be open twice
        self.assertEqual(FakeCameraSession.openings[1].resolved_tuning_file, "imx477.json")
        self.wait(lambda: "tuning imx477.json" in backend.frame_store.snapshot()["rgb"].info)
        self.assertTrue(any("Opening the camera again" in message for _, state, message in self.statuses if state == "starting"))

        backend.set_noir(True)
        self.wait(lambda: len(FakeCameraSession.openings) == 3)
        self.assertEqual(FakeCameraSession.openings[2].resolved_tuning_file, "imx477_noir.json")

        backend.set_noir(True)  # no change: the camera is left running
        time.sleep(0.1)
        self.assertEqual(len(FakeCameraSession.openings), 3)

    def test_the_switch_never_changes_the_recording_setting(self) -> None:
        backend = self.backend()
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 1)
        backend.set_noir(False)
        self.wait(lambda: len(FakeCameraSession.openings) == 2)
        self.assertIs(backend.config["camera"]["infrared"], True)
        self.assertIs(self.config["camera"]["infrared"], True)
        self.assertTrue(backend.recording_noir)

    def test_the_monitor_can_start_with_its_own_choice(self) -> None:
        backend = self.backend(noir_override=False)
        self.assertFalse(backend.noir)
        self.assertTrue(backend.recording_noir)
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 1)
        self.assertEqual(FakeCameraSession.openings[0].resolved_tuning_file, "imx477.json")

        def not_infrared(config):
            config["camera"]["infrared"] = False

        FakeCameraSession.openings = []
        backend = self.backend(not_infrared, noir_override=True)
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 1)
        self.assertEqual(FakeCameraSession.openings[0].resolved_tuning_file, "imx477_noir.json")

    def test_a_camera_that_failed_to_open_is_tried_again_on_the_next_switch(self) -> None:
        attempts = []

        class FailsWithNoir(FakeCameraSession):
            def __init__(self, config, *, tuning_config=None, **given) -> None:
                attempts.append(tuning_config["camera"]["infrared"])
                if tuning_config["camera"]["infrared"]:
                    raise RuntimeError("Failed to open camera with tuning file 'imx477_noir.json'")
                super().__init__(config, tuning_config=tuning_config, **given)

        with mock.patch("bumblebox_v2.run_engine._PicameraCaptureSession", FailsWithNoir):
            backend = self.backend()
            backend.start()
            self.wait(lambda: any(state == "error" for _, state, _ in self.statuses))
            self.assertIn("imx477_noir.json", next(message for _, state, message in self.statuses if state == "error"))
            backend.set_noir(False)
            self.wait(lambda: backend.frame_store.snapshot().get("rgb") is not None)
        self.assertEqual(attempts, [True, False])

    def test_cli_flags(self) -> None:
        parser = build_parser()
        self.assertIsNone(parser.parse_args(["live-monitor"]).noir)
        self.assertIs(parser.parse_args(["live-monitor", "--noir"]).noir, True)
        self.assertIs(parser.parse_args(["live-monitor", "--no-noir"]).noir, False)


class MonitorNoirWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def window(self, configure=None, **options):
        config = load_defaults()
        config["runtime"]["use_mock_camera"] = True
        config["thermal"]["enabled"] = False
        config["realsense"]["enabled"] = False
        if configure is not None:
            configure(config)
        window = create_live_monitor_window(config, LiveMonitorOptions(rgb_width=64, rgb_height=48, **options))
        self.addCleanup(window.close)
        window.show()
        return window

    def test_the_switch_starts_where_recordings_are(self) -> None:
        window = self.window()
        self.assertFalse(window.noir_button.isHidden())
        self.assertTrue(window.noir_button.isChecked())
        self.assertEqual(window.noir_button.text(), "NoIR Tuning: On")
        self.assertNotIn("for recordings", window.note.text())

        def not_infrared(config):
            config["camera"]["infrared"] = False

        window = self.window(not_infrared)
        self.assertFalse(window.noir_button.isChecked())
        self.assertEqual(window.noir_button.text(), "NoIR Tuning: Off")

    def test_button_and_n_key_switch_it_and_say_when_it_differs_from_recordings(self) -> None:
        window = self.window()
        window.noir_button.click()
        self.assertFalse(window.noir)
        self.assertFalse(window.backend.noir)
        self.assertEqual(window.noir_button.text(), "NoIR Tuning: Off")
        self.assertIn("NoIR tuning is off here and on for recordings", window.note.text())
        self.assertIn("nothing here changes how recordings are made", window.note.text())

        QTest.keyClick(window, Qt.Key_N)
        self.assertTrue(window.noir)
        self.assertTrue(window.backend.noir)
        self.assertTrue(window.noir_button.isChecked())
        self.assertNotIn("for recordings", window.note.text())

        window.enter_full_view("rgb")
        QTest.keyClick(window, Qt.Key_N)  # also in full screen, where the button is out of sight
        self.assertFalse(window.backend.noir)

    def test_starting_option_shows_in_the_button(self) -> None:
        window = self.window(noir_override=False)
        self.assertFalse(window.noir_button.isChecked())
        self.assertIn("NoIR tuning is off here and on for recordings", window.note.text())

    def test_a_tuning_file_named_by_hand_leaves_nothing_to_switch(self) -> None:
        def named(config):
            config["camera"]["tuning_file"] = "my_tuning.json"

        window = self.window(named)
        self.assertFalse(window.noir_button.isEnabled())
        self.assertIn("camera.tuning_file", window.noir_button.toolTip())
        QTest.keyClick(window, Qt.Key_N)
        self.assertTrue(window.backend.noir)
        self.assertNotIn("for recordings", window.note.text())

    def test_no_switch_without_the_rgb_camera(self) -> None:
        def thermal_only_here(config):
            config["thermal"]["enabled"] = True
            config["distributed_capture"]["enabled"] = True
            config["distributed_capture"]["nodes"] = [
                {"name": "this", "local": True, "host": "pi1.local", "sensors": ["thermal"]},
                {"name": "other", "local": False, "host": "pi2.local", "sensors": ["rgb"]},
            ]

        window = self.window(thermal_only_here)
        self.assertTrue(window.noir_button.isHidden())



class MonitorNoirMemoryTests(unittest.TestCase):
    """The monitor opens with its NoIR switch where it was last left, per config file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from bumblebox_v2 import live_monitor

        self.live_monitor = live_monitor
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        patcher = mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(self.home)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state_file = self.home / "BumbleBox" / "live_monitor.json"

    def backend(self, key="/boxes/a/config.yaml", infrared=True, **options) -> LiveMonitorBackend:
        config = load_defaults()
        config["runtime"]["use_mock_camera"] = True
        config["camera"]["infrared"] = infrared
        return LiveMonitorBackend(config, LiveMonitorOptions(rgb_width=64, rgb_height=48, remember_key=key, **options))

    def test_it_is_kept_next_to_the_other_per_computer_files(self) -> None:
        self.assertEqual(self.live_monitor.monitor_state_path(), self.state_file)
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": ""}):
            self.assertEqual(
                self.live_monitor.monitor_state_path(), Path.home() / ".local" / "share" / "BumbleBox" / "live_monitor.json"
            )

    def test_a_switch_that_was_never_used_follows_the_recording_setting(self) -> None:
        self.assertTrue(self.backend(infrared=True).noir)
        self.assertFalse(self.backend(infrared=False).noir)
        self.assertFalse(self.state_file.exists())  # looking writes nothing

    def test_the_switch_comes_back_where_it_was_left(self) -> None:
        self.backend().set_noir(False)
        self.assertFalse(self.backend().noir)
        # Its own setting: it stays there even when the recording setting says otherwise.
        self.assertFalse(self.backend(infrared=True).noir)
        self.backend().set_noir(True)
        self.assertTrue(self.backend(infrared=False).noir)
        self.assertTrue(self.backend(infrared=False).recording_noir is False)

    def test_each_config_file_remembers_its_own(self) -> None:
        self.backend(key="/boxes/a/config.yaml").set_noir(False)
        self.assertFalse(self.backend(key="/boxes/a/config.yaml").noir)
        self.assertTrue(self.backend(key="/boxes/a/config.camera2.yaml").noir)  # the second camera was not touched
        self.assertTrue(self.backend(key=None).noir)

    def test_a_starting_option_wins_for_one_session_and_is_not_remembered(self) -> None:
        self.backend().set_noir(False)
        self.assertTrue(self.backend(noir_override=True).noir)
        self.assertFalse(self.backend().noir)

    def test_the_config_file_is_never_written(self) -> None:
        config_path = self.home / "config.yaml"
        config = load_defaults()
        save_config(config_path, config)
        before = config_path.read_text()
        backend = LiveMonitorBackend(
            load_config(config_path), LiveMonitorOptions(rgb_width=64, rgb_height=48, remember_key=str(config_path))
        )
        backend.set_noir(False)
        self.assertEqual(config_path.read_text(), before)
        self.assertEqual(sorted(path.name for path in self.home.iterdir()), ["BumbleBox", "config.yaml"])

    def test_a_damaged_or_unwritable_memory_never_stops_the_monitor(self) -> None:
        self.state_file.parent.mkdir(parents=True)
        for junk in ("not json", "[1, 2]", '{"noir": "yes"}', '{"noir": {"/boxes/a/config.yaml": "off"}}'):
            with self.subTest(junk=junk):
                self.state_file.write_text(junk)
                self.assertTrue(self.backend().noir)  # falls back to the recording setting
        self.backend().set_noir(False)  # and a damaged file is replaced by a good one
        self.assertFalse(self.backend().noir)

        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/proc/no-such-place"}):
            backend = self.backend()
            backend.set_noir(False)  # cannot be written: no error
            self.assertFalse(backend.noir)

    def test_the_window_remembers_what_its_button_was_left_on(self) -> None:
        def window():
            config = load_defaults()
            config["runtime"]["use_mock_camera"] = True
            config["thermal"]["enabled"] = False
            config["realsense"]["enabled"] = False
            opened = create_live_monitor_window(
                config, LiveMonitorOptions(rgb_width=64, rgb_height=48, remember_key="/boxes/a/config.yaml")
            )
            self.addCleanup(opened.close)
            opened.show()
            return opened

        first = window()
        self.assertTrue(first.noir_button.isChecked())
        first.noir_button.click()
        first.close()

        second = window()
        self.assertFalse(second.noir_button.isChecked())
        self.assertEqual(second.noir_button.text(), "NoIR Tuning: Off")
        self.assertIn("NoIR tuning is off here and on for recordings", second.note.text())
        self.assertIn("where you last left it", second.noir_button.toolTip())

    def test_the_command_remembers_per_config_file_but_not_for_the_simulated_camera(self) -> None:
        config_path = self.home / "config.yaml"
        save_config(config_path, load_defaults())
        seen = []
        with mock.patch.object(
            self.live_monitor, "launch_live_monitor", lambda config, options, other=None: seen.append(options) or 0
        ):
            for extra in ([], ["--mock"]):
                args = build_parser().parse_args(["live-monitor", "--config", str(config_path), *extra])
                self.assertEqual(args.func(args), 0)
        self.assertEqual(seen[0].remember_key, str(config_path.resolve()))
        self.assertIsNone(seen[1].remember_key)



def second_camera(configure=None, *, first=False, title="Camera 2 (colony 02)"):
    from bumblebox_v2.live_monitor import MonitorCamera

    config = load_defaults()
    config["camera"]["camera_num"] = 0 if first else 1
    config["thermal"]["enabled"] = False
    config["realsense"]["enabled"] = False
    if configure is not None:
        configure(config)
    return MonitorCamera(config=config, title=title, first=first)


class BothCamerasBackendTests(unittest.TestCase):
    """Both cameras of a two-camera Pi in one monitor."""

    def setUp(self) -> None:
        FakeCameraSession.openings = []
        for patcher in (
            mock.patch("bumblebox_v2.run_engine._PicameraCaptureSession", FakeCameraSession),
            mock.patch("bumblebox_v2.camera_select.connected_camera_model", lambda config: None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.statuses: list[tuple[str, str, str]] = []

    def backend(self, other, configure=None, **options) -> LiveMonitorBackend:
        config = load_defaults()
        if configure is not None:
            configure(config)
        backend = LiveMonitorBackend(
            config,
            LiveMonitorOptions(rgb_width=64, rgb_height=48, display_fps=50.0, **options),
            status_callback=lambda *args: self.statuses.append(args),
            other_camera=other,
        )
        self.addCleanup(backend.stop)
        return backend

    def wait(self, until, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not until():
            if time.monotonic() > deadline:
                self.fail("timed out waiting for the monitor's cameras")
            time.sleep(0.01)

    @staticmethod
    def cameras_open() -> list[int]:
        return [session.config["camera"]["camera_num"] for session in FakeCameraSession.openings if not session.closed]

    def test_one_camera_is_shown_until_both_are_asked_for(self) -> None:
        backend = self.backend(second_camera())
        self.assertTrue(backend.can_show_both)
        self.assertFalse(backend.both)
        self.assertEqual(backend.streams(), ["rgb"])
        backend.start()
        self.wait(lambda: backend.frame_store.snapshot().get("rgb") is not None)
        self.assertEqual(self.cameras_open(), [0])  # the second camera is not opened for nothing

    def test_both_cameras_feed_their_own_panels(self) -> None:
        backend = self.backend(second_camera(), both_override=True)
        self.assertEqual(backend.streams(), ["rgb", "rgb2"])
        backend.start()
        self.wait(lambda: {"rgb", "rgb2"} <= set(backend.frame_store.snapshot()))
        self.assertEqual(sorted(self.cameras_open()), [0, 1])
        for stream in ("rgb", "rgb2"):
            self.assertTrue(any(name == stream and state == "live" for name, state, _ in self.statuses), stream)
            self.assertEqual(backend.frame_store.snapshot()[stream].image_bgr.shape, (48, 64, 3))

    def test_switching_between_one_and_both_reopens_the_cameras(self) -> None:
        backend = self.backend(second_camera())
        backend.start()
        self.wait(lambda: self.cameras_open() == [0])

        backend.set_both(True)
        self.wait(lambda: sorted(self.cameras_open()) == [0, 1])
        self.assertEqual(backend.streams(), ["rgb", "rgb2"])
        self.wait(lambda: "rgb2" in backend.frame_store.snapshot())

        backend.set_both(False)
        self.wait(lambda: self.cameras_open() == [0] and len(FakeCameraSession.openings) == 4)
        self.assertEqual(backend.streams(), ["rgb"])

    def test_two_cameras_that_want_the_same_tuning_file_both_get_it(self) -> None:
        backend = self.backend(second_camera(), both_override=True)
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 2)
        # The file is handed to both: libcamera loads one choice for the whole program.
        self.assertEqual([s.given.get("tuning_file") for s in FakeCameraSession.openings], ["imx477_noir.json"] * 2)
        self.wait(lambda: "rgb2" in backend.frame_store.snapshot())
        self.assertTrue(backend.frame_store.snapshot()["rgb2"].info.startswith("NoIR tuning (imx477_noir.json)"))

    def test_the_noir_switch_applies_to_both_cameras(self) -> None:
        backend = self.backend(second_camera(), both_override=True)
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 2)
        backend.set_noir(False)
        self.wait(lambda: len(FakeCameraSession.openings) == 4)
        self.assertEqual([s.given.get("tuning_file") for s in FakeCameraSession.openings[2:]], ["imx477.json"] * 2)

    def test_two_cameras_that_want_different_files_are_left_on_standard_tuning(self) -> None:
        def owlsight(config):
            config["camera"].update(model="owlsight_64mp", infrared=False)

        backend = self.backend(second_camera(owlsight), both_override=True)
        backend.start()
        self.wait(lambda: len(FakeCameraSession.openings) == 2)
        # Neither camera is handed the other sensor's file.
        self.assertEqual([s.given["tuning_file"] for s in FakeCameraSession.openings], [None, None])
        self.wait(lambda: {"rgb", "rgb2"} <= set(backend.frame_store.snapshot()))
        for stream in ("rgb", "rgb2"):
            self.assertIn("need different tuning files", backend.frame_store.snapshot()[stream].info)

    def test_a_camera_that_will_not_open_does_not_take_the_other_with_it(self) -> None:
        class SecondUnplugged(FakeCameraSession):
            def __init__(self, config, *, tuning_config=None, **given) -> None:
                if config["camera"]["camera_num"] == 1:
                    raise RuntimeError("The camera assigned to this config is not connected")
                super().__init__(config, tuning_config=tuning_config, **given)

        with mock.patch("bumblebox_v2.run_engine._PicameraCaptureSession", SecondUnplugged):
            backend = self.backend(second_camera(), both_override=True)
            backend.start()
            self.wait(lambda: backend.frame_store.snapshot().get("rgb") is not None)
            self.wait(lambda: any(name == "rgb2" and state == "error" for name, state, _ in self.statuses))
        self.assertNotIn("rgb2", backend.frame_store.snapshot())
        self.assertFalse(any(name == "rgb" and state == "error" for name, state, _ in self.statuses))

    def test_camera_one_is_always_on_the_left(self) -> None:
        self.assertEqual(self.backend(second_camera(), both_override=True).streams(), ["rgb", "rgb2"])
        # Opened from the second camera: the other one is camera 1, and goes first.
        opened_from_second = self.backend(second_camera(first=True, title="Camera 1 (colony 01)"), both_override=True)
        self.assertEqual(opened_from_second.streams(), ["rgb2", "rgb"])

    def test_the_other_sensors_keep_their_feeds(self) -> None:
        def with_thermal(config):
            config["runtime"]["use_mock_camera"] = True
            config["thermal"]["enabled"] = True

        backend = self.backend(second_camera(), with_thermal, both_override=True)
        self.assertEqual(backend.streams(), ["rgb", "rgb2", "thermal"])
        backend.start()
        self.wait(lambda: {"rgb", "rgb2", "thermal"} <= set(backend.frame_store.snapshot()))

    def test_a_single_camera_pi_has_nothing_to_switch(self) -> None:
        backend = self.backend(None, both_override=True)
        self.assertFalse(backend.can_show_both)
        self.assertFalse(backend.both)
        backend.set_both(True)
        self.assertEqual(backend.streams(), ["rgb"])


class BothCamerasWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"XDG_DATA_HOME": tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def window(self, other="default", configure=None, **options):
        config = load_defaults()
        config["runtime"]["use_mock_camera"] = True
        config["thermal"]["enabled"] = False
        config["realsense"]["enabled"] = False
        if configure is not None:
            configure(config)
        options.setdefault("rgb_title", "Camera 1 (colony 01)")
        window = create_live_monitor_window(
            config,
            LiveMonitorOptions(rgb_width=64, rgb_height=48, **options),
            second_camera() if other == "default" else other,
        )
        self.addCleanup(window.close)
        window.show()
        return window

    def pump(self, window, until, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not until():
            if time.monotonic() > deadline:
                self.fail("timed out waiting for the monitor window")
            self.app.processEvents()
            window._refresh()
            time.sleep(0.02)

    @staticmethod
    def title(panel) -> str:
        return next(label.text() for label in panel.findChildren(type(panel.status_label)) if label.objectName() == "VideoTitle")

    def test_a_pi_with_one_camera_has_no_such_button(self) -> None:
        window = self.window(other=None, rgb_title=None)
        self.assertTrue(window.both_button.isHidden())
        self.assertEqual(list(window.panels), ["rgb"])
        self.assertEqual(self.title(window.panels["rgb"]), "RGB camera")
        QTest.keyClick(window, Qt.Key_B)
        self.assertEqual(window.streams, ["rgb"])

    def test_the_button_puts_the_two_cameras_side_by_side(self) -> None:
        window = self.window()
        self.assertFalse(window.both_button.isHidden())
        self.assertEqual(window.both_button.text(), "Both Cameras: Off")
        self.assertEqual(window.streams, ["rgb"])
        self.assertTrue(window.panels["rgb2"].isHidden())
        self.assertIn("1 camera feed", window.subtitle.text())

        window.both_button.click()
        self.assertEqual(window.both_button.text(), "Both Cameras: On")
        self.assertEqual(window.streams, ["rgb", "rgb2"])
        self.assertIn("2 camera feeds", window.subtitle.text())
        left, right = window.panels["rgb"], window.panels["rgb2"]
        self.assertFalse(left.isHidden() or right.isHidden())
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(left)), (0, 0, 1, 1))
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(right)), (0, 1, 1, 1))
        self.assertEqual(self.title(left), "Camera 1 (colony 01)")
        self.assertEqual(self.title(right), "Camera 2 (colony 02)")
        self.pump(window, lambda: left.state == "live" and right.state == "live")
        self.assertEqual(left.dimensions, (64, 48))
        self.assertEqual(right.dimensions, (64, 48))

        window.both_button.click()
        self.assertEqual(window.streams, ["rgb"])
        self.assertTrue(right.isHidden())

    def test_b_key_switches_it_and_full_screen_still_works(self) -> None:
        window = self.window()
        QTest.keyClick(window, Qt.Key_B)
        self.assertTrue(window.both)
        window.enter_full_view("rgb2")
        self.assertEqual(window.full_view, "rgb2")
        self.assertTrue(window.panels["rgb"].isHidden())
        # Going back to one camera while the other fills the screen leaves full screen.
        QTest.keyClick(window, Qt.Key_B)
        self.assertFalse(window.both)
        self.assertIsNone(window.full_view)
        self.assertFalse(window.panels["rgb"].isHidden())
        self.assertTrue(window.panels["rgb2"].isHidden())

    def test_grayscale_and_contrast_boost_apply_to_both_cameras(self) -> None:
        window = self.window(both_override=True)
        self.pump(window, lambda: all(window.panels[s].state == "live" for s in ("rgb", "rgb2")))
        window.set_contrast_boost(True)
        for stream in ("rgb", "rgb2"):
            surface = window.panels[stream].surface
            self.assertTrue(surface._grayscale and surface._contrast_boost, stream)

    def test_no_old_frame_is_left_showing_when_a_camera_is_opened_again(self) -> None:
        window = self.window(both_override=True)
        right = window.panels["rgb2"]
        self.pump(window, lambda: right.state == "live")
        window.set_both(False)
        self.assertIsNone(right.surface._source)
        self.assertEqual(right.status_label.text(), "Starting camera...")

    def test_opened_from_the_second_camera_camera_one_is_still_on_the_left(self) -> None:
        window = self.window(
            other=second_camera(first=True, title="Camera 1 (colony 01)"),
            both_override=True,
            rgb_title="Camera 2 (colony 02)",
        )
        self.assertEqual(window.streams, ["rgb2", "rgb"])
        left, right = window.panels["rgb2"], window.panels["rgb"]
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(left)), (0, 0, 1, 1))
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(right)), (0, 1, 1, 1))
        self.assertEqual(self.title(left), "Camera 1 (colony 01)")
        self.assertEqual(self.title(right), "Camera 2 (colony 02)")

    def test_with_a_thermal_camera_the_two_cameras_keep_the_top_row(self) -> None:
        def with_thermal(config):
            config["thermal"]["enabled"] = True

        window = self.window(configure=with_thermal, both_override=True)
        self.assertEqual(window.streams, ["rgb", "rgb2", "thermal"])
        positions = {s: window.grid.getItemPosition(window.grid.indexOf(window.panels[s])) for s in window.streams}
        self.assertEqual(positions, {"rgb": (0, 0, 1, 1), "rgb2": (0, 1, 1, 1), "thermal": (1, 0, 1, 2)})
        window.set_both(False)  # back to the usual two-feed layout
        positions = {s: window.grid.getItemPosition(window.grid.indexOf(window.panels[s])) for s in window.streams}
        self.assertEqual(positions, {"rgb": (0, 0, 1, 1), "thermal": (0, 1, 1, 1)})

    def test_the_monitor_opens_the_way_it_was_left(self) -> None:
        first = self.window(box_key="/boxes/a/config.yaml")
        self.assertFalse(first.both)
        first.both_button.click()
        first.close()

        second = self.window(box_key="/boxes/a/config.yaml")
        self.assertTrue(second.both)
        self.assertTrue(second.both_button.isChecked())
        self.assertEqual(second.streams, ["rgb", "rgb2"])
        second.close()

        # For one session only, and not remembered.
        self.assertFalse(self.window(box_key="/boxes/a/config.yaml", both_override=False).both)
        self.assertTrue(self.window(box_key="/boxes/a/config.yaml").both)
        self.assertFalse(self.window(box_key="/boxes/b/config.yaml").both)  # another box


class BothCamerasCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        from bumblebox_v2 import live_monitor
        from bumblebox_v2.multi_camera import create_second_camera

        self.live_monitor = live_monitor
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.primary = self.root / "config.yaml"
        config = load_defaults()
        config["system"]["data_root"] = str(self.root / "data")
        config["system"]["colony_id"] = "01"
        save_config(self.primary, config)
        self.create_second_camera = lambda: create_second_camera(self.primary, cameras=[])

    def launch(self, *arguments):
        seen = []
        with mock.patch.object(
            self.live_monitor, "launch_live_monitor", lambda config, options, other=None: seen.append((config, options, other)) or 0
        ):
            args = build_parser().parse_args(["live-monitor", *arguments])
            self.assertEqual(args.func(args), 0)
        return seen[0]

    def test_a_single_camera_box_is_opened_as_before(self) -> None:
        _, options, other = self.launch("--config", str(self.primary))
        self.assertIsNone(other)
        self.assertIsNone(options.rgb_title)
        self.assertIsNone(options.both_override)

    def test_either_cameras_config_brings_the_other_camera_along(self) -> None:
        from bumblebox_v2.multi_camera import primary_config_path, second_camera_config_path

        second = self.create_second_camera().path
        self.assertEqual(second, second_camera_config_path(self.primary))
        self.assertEqual(primary_config_path(second), self.primary)
        self.assertEqual(primary_config_path(self.primary), self.primary)

        _, options, other = self.launch("--config", str(self.primary))
        self.assertEqual(options.rgb_title, "Camera 1 (colony 01)")
        self.assertEqual(other.title, "Camera 2 (colony 02)")
        self.assertFalse(other.first)
        self.assertEqual(other.config["camera"]["session_tag"], "cam2")
        self.assertEqual(options.box_key, str(self.primary.resolve()))

        config, options, other = self.launch("--config", str(second))
        self.assertEqual(config["camera"]["session_tag"], "cam2")
        self.assertEqual(options.rgb_title, "Camera 2 (colony 02)")
        self.assertEqual(other.title, "Camera 1 (colony 01)")
        self.assertTrue(other.first)
        # Remembered for the box, whichever camera the monitor was opened from.
        self.assertEqual(options.box_key, str(self.primary.resolve()))
        self.assertEqual(options.remember_key, str(second.resolve()))

    def test_starting_options(self) -> None:
        self.create_second_camera()
        self.assertIs(self.launch("--config", str(self.primary), "--both-cameras")[1].both_override, True)
        self.assertIs(self.launch("--config", str(self.primary), "--one-camera")[1].both_override, False)

    def test_a_file_that_only_looks_like_a_second_camera_is_its_own_box(self) -> None:
        from bumblebox_v2.multi_camera import primary_config_path

        lone = self.root / "elsewhere.camera2.yaml"
        save_config(lone, load_defaults())
        self.assertEqual(primary_config_path(lone), lone)


if __name__ == "__main__":
    unittest.main()
