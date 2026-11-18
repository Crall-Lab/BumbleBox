"""Qt GUI: cancel, stop confirmation, hardware-aware results, storage estimate, settings page."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from bumblebox_v2.qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    from PyQt5.QtCore import QEvent, QPoint, QPointF, QProcess, Qt, QTimer
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication, QMessageBox

    from bumblebox_v2 import box_status, qt_gui
    from bumblebox_v2.config import load_config, load_defaults, save_config
    from bumblebox_v2.qt_gui import BumbleBoxQtGUI
except Exception:  # pragma: no cover - optional on non-GUI test hosts
    QApplication = None


SLEEPY_SCRIPT = textwrap.dedent(
    """
    import signal, sys, time
    if "--ignore-sigint" in sys.argv:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("started " + sys.argv[1], flush=True)
    if "--quick" in sys.argv:
        sys.exit(0)
    time.sleep(30)
    """
)


def _quiet_status(free_gb: float = 212.0, schedule_state: str = "off"):
    level = box_status.LEVEL_OFF if schedule_state == box_status.SCHEDULE_OFF else box_status.LEVEL_OK
    return box_status.BoxStatus(
        storage=box_status.StatusLine(
            box_status.LEVEL_OK, f"Drive mounted · {free_gb:g} GB free", "/mnt/bumblebox/data", free_gb=free_gb
        ),
        schedule=box_status.ScheduleStatus(
            schedule_state,
            level,
            "Off" if schedule_state == box_status.SCHEDULE_OFF else "On · next recording 14:30 (in 12 min)",
            next_run=None if schedule_state == box_status.SCHEDULE_OFF else datetime.now() + timedelta(minutes=12),
        ),
        checked_at=datetime.now(),
    )


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class QtControlTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @contextmanager
    def window(self, configure=None):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "bumblebox_v2.qt_gui.collect_box_status", return_value=_quiet_status()
        ):
            root = Path(directory)
            self.root = root
            config_path = root / "config.yaml"
            config = load_defaults()
            config["setup"]["completed"] = True
            config["system"]["data_root"] = str(root / "data")
            if configure is not None:
                configure(config)
            save_config(config_path, config)
            window = BumbleBoxQtGUI(config_path)
            window.encode_benchmark_path = root / "bench.json"
            window._show_failure_dialog = lambda explanation: self.fail(f"unexpected failure dialog: {explanation}")
            try:
                yield window
            finally:
                if window.process.state() != QProcess.NotRunning:
                    window.process.kill()
                    window.process.waitForFinished(3000)
                if window._settings_dirty():
                    window._discard_settings()
                window.close()

    def pump(self, until, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not until():
            if time.monotonic() > deadline:
                self.fail("timed out waiting for the window")
            self.app.processEvents()
            time.sleep(0.02)


class CancelTests(QtControlTestCase):
    @contextmanager
    def fake_bbx(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_bbx.py"
            script.write_text(SLEEPY_SCRIPT)
            with mock.patch.object(qt_gui, "BBX_SCRIPT", script):
                yield

    def start(self, window, *commands):
        window.run_bbx_commands([list(command) for command in commands])
        self.pump(lambda: "started" in "".join(window._command_output))
        self.assertTrue(window.cancel_button.isVisibleTo(window))

    def test_cancel_stops_the_command_and_the_queue_without_a_failure_dialog(self) -> None:
        with self.window() as window, self.fake_bbx():
            self.start(window, ["sleepy"], ["second"])
            window._cancel_command()
            self.assertEqual(window.cancel_button.text(), "Cancelling...")
            self.pump(lambda: window.process.state() == QProcess.NotRunning)
            self.pump(lambda: window.command_status.objectName() == "StatusWarn")
            self.assertEqual(window.command_status.text(), "bbx sleepy cancelled.")
            self.assertTrue(window.cancel_button.isHidden())
            self.assertNotIn("started second", window.console.toPlainText())
            self.assertEqual(window.command_queue, [])

    def test_command_that_ignores_the_interrupt_is_terminated(self) -> None:
        with self.window() as window, self.fake_bbx(), mock.patch.object(qt_gui, "CANCEL_INTERRUPT_GRACE_MS", 300):
            self.start(window, ["stubborn", "--ignore-sigint"])
            started = time.monotonic()
            window._cancel_command()
            self.pump(lambda: window.process.state() == QProcess.NotRunning)
            self.assertLess(time.monotonic() - started, 5.0)
            self.assertIn("terminating it", window.console.toPlainText())
            self.assertEqual(window.command_status.objectName(), "StatusWarn")

    def test_cancelling_a_recording_asks_first(self) -> None:
        with self.window() as window, self.fake_bbx():
            self.start(window, ["run-once", "--mode", "record_only"])
            with mock.patch.object(window, "_confirm_cancel_recording", return_value=False):
                window._cancel_command()
            self.assertFalse(window._cancel_requested)
            self.assertEqual(window.process.state(), QProcess.Running)

            with mock.patch.object(window, "_confirm_cancel_recording", return_value=True):
                window._cancel_command()
            self.pump(lambda: window.process.state() == QProcess.NotRunning)
            self.pump(lambda: window.command_status.objectName() == "StatusWarn")
            self.assertTrue(window.command_status.text().startswith("Recording cancelled."))
            self.assertIn("listed as failed in Results", window.command_status.text())

    def test_command_that_completes_anyway_is_reported_as_completed(self) -> None:
        with self.window() as window:
            # Cancel was pressed with one more step queued, but the command
            # exited cleanly before the interrupt reached it.
            window._current_command = ["sleepy"]
            window._cancel_requested = True
            window._cancelled_steps = 1
            window._process_finished(0, QProcess.NormalExit)
            self.assertEqual(window.command_status.objectName(), "StatusGood")
            self.assertEqual(
                window.command_status.text(),
                "bbx sleepy finished before it could be cancelled. The steps after it were not run.",
            )

    def test_finished_command_hides_the_cancel_button(self) -> None:
        with self.window() as window, self.fake_bbx():
            window.run_bbx_commands([["quick", "--quick"]])
            self.pump(lambda: window.command_status.objectName() == "StatusGood")
            self.assertTrue(window.cancel_button.isHidden())


class StopConfirmationTests(QtControlTestCase):
    def test_stop_asks_when_the_schedule_is_on_and_does_nothing_if_declined(self) -> None:
        with self.window() as window:
            window._apply_box_status(_quiet_status(schedule_state=box_status.SCHEDULE_ON))
            config = load_config(window.config_path)
            config["scheduling"]["enabled"] = True
            save_config(window.config_path, config)
            ran = []
            window.run_bbx_command = lambda arguments, **kwargs: ran.append(arguments)

            with mock.patch.object(window, "_confirm_stop_automation", return_value=False):
                window._stop_automation()
            self.assertEqual(ran, [])
            self.assertTrue(load_config(window.config_path)["scheduling"]["enabled"])

            with mock.patch.object(window, "_confirm_stop_automation", return_value=True):
                window._stop_automation()
            self.assertEqual(ran, [["systemd-disable"]])
            self.assertFalse(load_config(window.config_path)["scheduling"]["enabled"])

    def test_dialog_explains_what_stopping_does_and_defaults_to_keep(self) -> None:
        with self.window() as window:
            window._apply_box_status(_quiet_status(schedule_state=box_status.SCHEDULE_ON))
            seen = {}

            def answer():
                box = self.app.activeModalWidget()
                seen.update(
                    title=box.windowTitle(),
                    text=box.text(),
                    info=box.informativeText(),
                    default=box.defaultButton().text(),
                )
                next(button for button in box.buttons() if button.text() == "Keep Recording").click()

            QTimer.singleShot(100, answer)
            self.assertFalse(window._confirm_stop_automation())
            self.assertEqual(seen["title"], "Stop automated recording?")
            self.assertIn("No more recordings will start", seen["text"])
            self.assertIn("already in progress finishes normally", seen["info"])
            self.assertIn("The next one is due", seen["info"])
            self.assertEqual(seen["default"], "Keep Recording")

    def test_no_question_when_nothing_is_scheduled(self) -> None:
        with self.window() as window:
            window._apply_box_status(_quiet_status(schedule_state=box_status.SCHEDULE_OFF))
            self.assertTrue(window._confirm_stop_automation())


class ResultButtonTests(QtControlTestCase):
    @staticmethod
    def payload(**extra):
        payload = {
            "session_name": "s",
            "started_at": datetime.now().isoformat(),
            "mode": "record_only",
            "success": True,
            "frames_captured": 10,
            "actual_fps": 5.0,
            "warnings": [],
            "errors": [],
            "thermal_enabled": False,
            "realsense_enabled": False,
        }
        payload.update(extra)
        return payload

    def visible(self, window):
        return {key for key, button in window.result_button_order if not button.isHidden()}

    def test_camera_only_box_shows_only_camera_outputs(self) -> None:
        with self.window() as window:
            window._run_history_loaded(window._run_history_generation, [self.payload()], None)
            self.assertEqual(self.visible(window), {"session", "rgb"})
            self.assertTrue(window.results_table.isColumnHidden(4))
            self.assertTrue(window.results_table.isColumnHidden(5))
            detail = window.result_detail.text()
            self.assertIn("RGB: 10 frames", detail)
            for absent in ("Thermal:", "RealSense:", "Distributed capture:"):
                self.assertNotIn(absent, detail)

    def test_a_recording_that_used_depth_keeps_its_buttons(self) -> None:
        with self.window() as window:
            window._run_history_loaded(
                window._run_history_generation, [self.payload(realsense_enabled=True, realsense_frames_captured=30)], None
            )
            self.assertEqual(
                self.visible(window),
                {"session", "rgb", "depth", "depth_color", "depth_data", "metadata", "calibration", "sync"},
            )
            self.assertTrue(window.results_table.isColumnHidden(4))
            self.assertFalse(window.results_table.isColumnHidden(5))
            self.assertIn("RealSense: 30 frames", window.result_detail.text())
            self.assertNotIn("Thermal:", window.result_detail.text())

    def test_configured_thermal_camera_shows_its_buttons(self) -> None:
        def thermal(config):
            config["thermal"]["enabled"] = True

        with self.window(thermal) as window:
            window._set_result_artifact_buttons(None)
            self.assertIn("thermal", self.visible(window))
            self.assertIn("calibration", self.visible(window))
            self.assertNotIn("manifest", self.visible(window))
            self.assertFalse(window.results_table.isColumnHidden(4))


class StorageCardTests(QtControlTestCase):
    def test_storage_card_shows_use_per_day_and_space_left(self) -> None:
        with self.window() as window:
            run = ResultButtonTests.payload(
                mode="record_and_track",
                frames_captured=100,
                _session_bytes=150e6,
                _recording_signature=box_status.recording_signature(window.config),
            )
            window._run_history_loaded(window._run_history_generation, [run], None)
            window._apply_box_status(_quiet_status(free_gb=212))
            detail = window.storage_card.detail_label.text()
            self.assertIn("About 7.2 GB per day · about 29 days of space left", detail)
            self.assertEqual(window.storage_card.property("tone"), "mint")

            window._apply_box_status(_quiet_status(free_gb=30))
            self.assertIn("about 4 days of space left", window.storage_card.detail_label.text())
            self.assertEqual(window.storage_card.property("tone"), "warn")

    def test_refused_storage_does_not_show_an_estimate(self) -> None:
        with self.window() as window:
            status = _quiet_status()
            status.storage = box_status.StatusLine(box_status.LEVEL_FAIL, "Data drive not mounted", "refused")
            window._apply_box_status(status)
            self.assertEqual(window.storage_card.detail_label.text(), "refused")
            self.assertEqual(window.storage_card.property("tone"), "fail")


class SettingsPageTests(QtControlTestCase):
    def test_form_shows_the_saved_settings(self) -> None:
        with self.window() as window:
            self.assertEqual(window.settings_mode.currentData(), "record_and_track")
            self.assertEqual(window.settings_recording_seconds.value(), 20)
            self.assertEqual(window.settings_record_interval.value(), 30)
            self.assertEqual(window.settings_frame_buffer.currentData(), "ram")
            self.assertEqual(window.settings_preset.currentData(), "auto")
            self.assertFalse(window._settings_dirty())
            self.assertFalse(window.settings_save_button.isEnabled())
            self.assertTrue(window.settings_track_interval.isHidden())

    def test_edit_save_and_discard(self) -> None:
        with self.window() as window:
            window.settings_recording_seconds.setValue(45)
            window.settings_frame_buffer.setCurrentIndex(window.settings_frame_buffer.findData("stream"))
            window.settings_mode.setCurrentIndex(window.settings_mode.findData("mixed_schedule"))
            self.assertTrue(window._settings_dirty())
            self.assertTrue(window.settings_save_button.isEnabled())
            self.assertFalse(window.settings_track_interval.isHidden())

            self.assertTrue(window._save_settings(offer_apply=False))
            saved = load_config(window.config_path)
            self.assertEqual(saved["capture"]["recording_seconds"], 45)
            self.assertEqual(saved["capture"]["frame_buffer"], "stream")
            self.assertEqual(saved["pipeline"]["mode"], "mixed_schedule")
            self.assertFalse(window._settings_dirty())

            window.settings_recording_seconds.setValue(90)
            window._discard_settings()
            self.assertEqual(window.settings_recording_seconds.value(), 45)
            self.assertFalse(window._settings_dirty())

    def test_profile_settings_are_locked_until_switched_to_custom(self) -> None:
        def hq(config):
            config["camera"]["profile"] = "hq_reference"

        with self.window(hq) as window:
            self.assertFalse(window.settings_fps.isEnabled())
            self.assertFalse(window.settings_codec.isEnabled())
            self.assertFalse(window.settings_custom_camera_button.isHidden())
            self.assertEqual(window.settings_fps.value(), 7.0)

            window.settings_fps.setValue(3.0)  # not editable by the user; must not be saved
            window.settings_recording_seconds.setValue(30)
            window._save_settings(offer_apply=False)
            saved = load_config(window.config_path)
            self.assertEqual(saved["camera"]["fps_target"], 7.0)
            self.assertEqual(saved["capture"]["recording_seconds"], 30)

            window._use_custom_camera_settings()
            self.assertTrue(window.settings_fps.isEnabled())
            self.assertTrue(window._settings_dirty())
            window.settings_fps.setValue(9.0)
            window._save_settings(offer_apply=False)
            saved = load_config(window.config_path)
            self.assertEqual(saved["camera"]["profile"], "custom")
            self.assertEqual(saved["camera"]["fps_target"], 9.0)
            self.assertTrue(window.settings_custom_camera_button.isHidden())

    def test_preview_flags_settings_that_cannot_work(self) -> None:
        with self.window() as window:
            window._refresh_settings_preview()
            self.assertEqual(window.settings_check_status.objectName(), "StatusGood")

            window.settings_recording_seconds.setValue(3600)
            window._refresh_settings_preview()
            self.assertEqual(window.settings_check_status.objectName(), "StatusFail")
            self.assertIn("Problem: Memory budget", window.settings_issues_label.text())
            self.assertIn("Unsaved changes", window.settings_summary_label.text())

    def test_preview_uses_the_saved_benchmark(self) -> None:
        with self.window() as window:
            window._refresh_settings_preview()
            self.assertIn("Not run on this computer yet", window.settings_benchmark_label.text())
            window.encode_benchmark_path.write_text(
                json.dumps(
                    {
                        "width": 4056,
                        "height": 3040,
                        "monochrome_output": False,
                        "source": "synthetic nest-like frames",
                        "results": [
                            {"label": "libx264:ultrafast", "codec": "mp4", "mp4_codec": "libx264",
                             "mp4_preset": "ultrafast", "frames_per_second": 3.0, "megabytes_per_frame": 1.9},
                            {"label": "libx264:fast", "codec": "mp4", "mp4_codec": "libx264",
                             "mp4_preset": "fast", "frames_per_second": 1.4, "megabytes_per_frame": 1.1},
                        ],
                    }
                )
            )
            window._refresh_settings_preview()
            self.assertIn("Selected encoder: 1.4 fps, measured today", window.settings_benchmark_label.text())
            self.assertIn("only sets how long encoding takes", window.settings_benchmark_label.text())
            self.assertEqual(window.settings_check_status.objectName(), "StatusGood")

            # Encoding while recording with an encoder slower than the camera cannot work.
            window.settings_frame_buffer.setCurrentIndex(window.settings_frame_buffer.findData("stream"))
            window._refresh_settings_preview()
            self.assertIn("is slower than the camera at 5 fps", window.settings_benchmark_label.text())
            self.assertEqual(window.settings_check_status.objectName(), "StatusFail")
            self.assertIn("Problem: Encoder", window.settings_issues_label.text())

    def test_benchmark_runs_on_the_latest_recording_with_the_form_values(self) -> None:
        with self.window() as window:
            video = self.root / "session" / "a.mp4"
            video.parent.mkdir()
            video.write_bytes(b"x" * 2000)
            run = ResultButtonTests.payload(
                video_path=str(video), _recording_signature=box_status.recording_signature(window.config)
            )
            window._run_history_loaded(window._run_history_generation, [run], None)
            window.settings_fps.setValue(8.0)
            window.settings_preset.setCurrentIndex(window.settings_preset.findData("medium"))
            ran = []
            window.run_bbx_command = lambda arguments, **kwargs: ran.append(arguments)
            window._run_encode_benchmark()
            arguments = ran[0]
            self.assertEqual(arguments[0], "encode-benchmark")
            self.assertEqual(arguments[arguments.index("--fps") + 1], "8")
            self.assertIn("--color", arguments)
            self.assertEqual(arguments[arguments.index("--json-out") + 1], str(window.encode_benchmark_path))
            self.assertEqual(arguments[arguments.index("--source-video") + 1], str(video.resolve()))
            self.assertIn("libx264:medium", arguments[arguments.index("--encoders") + 1])

    def test_unsaved_settings_are_settled_before_a_recording(self) -> None:
        with self.window() as window, mock.patch(
            "bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False
        ):
            ran = []
            window.run_bbx_command = lambda arguments, **kwargs: ran.append(arguments)
            window.settings_recording_seconds.setValue(33)

            with mock.patch.object(window, "_ask_unsaved_settings", return_value="cancel"):
                window._run_once()
            self.assertEqual(ran, [])

            with mock.patch.object(window, "_ask_unsaved_settings", return_value="ignore"):
                window._run_once()
            self.assertEqual(len(ran), 1)
            self.assertEqual(load_config(window.config_path)["capture"]["recording_seconds"], 20)

            with mock.patch.object(window, "_ask_unsaved_settings", return_value="save"):
                window._run_once()
            self.assertEqual(len(ran), 2)
            self.assertEqual(load_config(window.config_path)["capture"]["recording_seconds"], 33)

    def test_closing_with_unsaved_settings_asks(self) -> None:
        with self.window() as window:
            window.show()
            window.settings_recording_seconds.setValue(33)
            with mock.patch.object(window, "_ask_unsaved_settings_on_close", return_value="cancel"):
                window.close()
            self.assertTrue(window.isVisible())
            with mock.patch.object(window, "_ask_unsaved_settings_on_close", return_value="discard"):
                window.close()
            self.assertFalse(window.isVisible())

    def test_saving_offers_to_apply_a_changed_interval_to_the_running_schedule(self) -> None:
        with self.window() as window:
            window._apply_box_status(_quiet_status(schedule_state=box_status.SCHEDULE_ON))
            started = []
            window._start_automation = lambda targets=None: started.append(True)
            window.settings_record_interval.setValue(15)
            with mock.patch.object(qt_gui, "installed_units_are_current", return_value=False), mock.patch.object(
                window, "_ask_apply_schedule", return_value=True
            ):
                window._save_settings(offer_apply=True)
            self.assertEqual(started, [True])

            with mock.patch.object(qt_gui, "installed_units_are_current", return_value=True), mock.patch.object(
                window, "_ask_apply_schedule", side_effect=AssertionError("should not ask")
            ):
                window.settings_record_interval.setValue(20)
                window._save_settings(offer_apply=True)
            self.assertEqual(started, [True])

    def test_live_monitor_button_is_on_overview_not_run(self) -> None:
        with self.window() as window:
            overview, run = window.pages.widget(qt_gui.PAGE_INDEX_OVERVIEW), window.pages.widget(qt_gui.PAGE_INDEX_RUN)
            self.assertTrue(overview.isAncestorOf(window.live_monitor_button))
            self.assertFalse(run.isAncestorOf(window.live_monitor_button))
            self.assertEqual(window.live_monitor_button.text(), "Open Live Camera Monitor")
            started = []
            window.run_bbx_command = lambda arguments, **kwargs: started.append((arguments, kwargs))
            with mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False):
                window.live_monitor_button.click()
            self.assertEqual(started, [(["live-monitor"], {"detached": True})])

    def test_record_in_grayscale_button_on_run_saves_at_once(self) -> None:
        with self.window() as window:
            self.assertTrue(window.pages.widget(qt_gui.PAGE_INDEX_RUN).isAncestorOf(window.record_gray_button))
            self.assertEqual(window.record_gray_button.text(), "Record in Grayscale: Off")
            self.assertFalse(window.record_gray_button.isChecked())

            window.record_gray_button.click()
            self.assertTrue(load_config(window.config_path)["camera"]["monochrome_output"])
            self.assertEqual(window.record_gray_button.text(), "Record in Grayscale: On")
            self.assertTrue(window.record_gray_button.isChecked())
            self.assertTrue(window.settings_monochrome.isChecked())  # the Settings page agrees
            self.assertFalse(window._settings_dirty())

            window.record_gray_button.click()
            self.assertFalse(load_config(window.config_path)["camera"]["monochrome_output"])
            self.assertEqual(window.record_gray_button.text(), "Record in Grayscale: Off")

    def test_record_in_grayscale_follows_the_settings_page_and_the_shown_config(self) -> None:
        def gray(config):
            config["camera"]["monochrome_output"] = True

        with self.window(gray) as window:
            self.assertTrue(window.record_gray_button.isChecked())
            self.assertEqual(window.record_gray_button.text(), "Record in Grayscale: On")
            window.settings_monochrome.setChecked(False)
            window._save_settings(offer_apply=False)
            self.assertFalse(window.record_gray_button.isChecked())

    def test_record_in_grayscale_says_when_it_leaves_a_camera_profile(self) -> None:
        def hq(config):
            config["camera"]["profile"] = "hq_reference"

        with self.window(hq) as window:
            with mock.patch.object(QMessageBox, "information") as told:
                window.record_gray_button.click()
            self.assertEqual(told.call_args[0][1], "Camera profile changed")
            self.assertIn("HQ Reference", told.call_args[0][2])
            saved = load_config(window.config_path)
            self.assertEqual(saved["camera"]["profile"], "custom")
            self.assertTrue(saved["camera"]["monochrome_output"])
            self.assertTrue(window.settings_monochrome.isEnabled())  # no longer locked by the profile

    def test_failed_recording_toggle_puts_the_button_back(self) -> None:
        with self.window() as window:
            with mock.patch.object(qt_gui, "set_recording_grayscale", side_effect=OSError("read-only")), mock.patch.object(
                QMessageBox, "critical"
            ) as refused:
                window.record_gray_button.click()
            self.assertEqual(refused.call_args[0][1], "Recording setting not changed")
            self.assertFalse(window.record_gray_button.isChecked())
            self.assertEqual(window.record_gray_button.text(), "Record in Grayscale: Off")

    def test_setting_changed_outside_the_window_is_picked_up(self) -> None:
        from bumblebox_v2.recording_options import set_recording_grayscale

        with self.window() as window:
            self.assertFalse(window._reload_if_config_changed())  # nothing has changed
            self.assertFalse(window.settings_monochrome.isChecked())
            set_recording_grayscale(window.config_path, True)  # what the live monitor's button does
            self.assertTrue(window._reload_if_config_changed())
            self.assertTrue(window.config["camera"]["monochrome_output"])
            self.assertTrue(window.settings_monochrome.isChecked())
            self.assertFalse(window._settings_dirty())
            self.assertFalse(window._reload_if_config_changed())

    def test_saving_does_not_undo_a_setting_changed_outside_the_window(self) -> None:
        from bumblebox_v2.recording_options import set_recording_grayscale

        with self.window() as window:
            window.settings_recording_seconds.setValue(45)  # an unsaved edit
            set_recording_grayscale(window.config_path, True)
            window._reload_if_config_changed()
            self.assertEqual(window.settings_recording_seconds.value(), 45)  # the edit is kept
            self.assertTrue(window._settings_dirty())

            self.assertTrue(window._save_settings(offer_apply=False))
            saved = load_config(window.config_path)
            self.assertEqual(saved["capture"]["recording_seconds"], 45)
            self.assertTrue(saved["camera"]["monochrome_output"])  # not overwritten by the form's older value
            self.assertTrue(window.settings_monochrome.isChecked())

    def test_scrolling_over_a_field_does_not_change_it(self) -> None:
        with self.window() as window:
            spin = window.settings_recording_seconds.number
            event = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
            )
            QApplication.sendEvent(spin, event)
            self.assertEqual(spin.value(), 20)
            self.assertEqual(window.settings_recording_seconds.value(), 20)


class ScheduleFieldTests(QtControlTestCase):
    """Lengths and intervals in seconds, minutes, hours or days, and the time of day."""

    @staticmethod
    def choose(field, number: float, unit: str) -> None:
        field.number.setValue(number)
        field.unit.setCurrentIndex(field.unit.findData(unit))

    def test_saved_values_are_shown_in_their_natural_unit(self) -> None:
        def daily(config):
            config["capture"].update({"recording_seconds": 7200, "record_interval_minutes": 1440})
            config["scheduling"]["start_time"] = "10:00"

        with self.window(daily) as window:
            length, every = window.settings_recording_seconds, window.settings_record_interval
            self.assertEqual((length.number.value(), length.unit.currentData()), (2.0, "h"))
            self.assertEqual(length.unit.currentText(), "hours")
            self.assertEqual((every.number.value(), every.unit.currentData()), (1.0, "day"))
            self.assertEqual(every.unit.currentText(), "day")
            self.assertEqual(window.settings_start_time.time().toString("HH:mm"), "10:00")
            self.assertEqual(window.settings_start_hint.text(), "It starts at this time every day.")
            self.assertFalse(window._settings_dirty())
            window._refresh_settings_preview()
            self.assertIn("Records 2 h once a day at 10:00, then tracks.", window.settings_effects_label.text())

    def test_two_hours_once_a_day_at_ten_is_saved_in_seconds_and_minutes(self) -> None:
        with self.window() as window:
            self.assertEqual(window.settings_recording_seconds.unit.currentData(), "s")
            self.assertEqual(
                window.settings_start_hint.text(), "Falls on 00:00 and every 30 min after it, around the clock."
            )
            self.choose(window.settings_recording_seconds, 2, "h")
            self.choose(window.settings_record_interval, 1, "day")
            window.settings_start_time.setTime(qt_gui.QTime(10, 0))
            self.assertTrue(window._settings_dirty())
            self.assertTrue(window._save_settings(offer_apply=False))
            saved = load_config(window.config_path)
            self.assertEqual(saved["capture"]["recording_seconds"], 7200)
            self.assertEqual(saved["capture"]["record_interval_minutes"], 1440)
            self.assertEqual(saved["scheduling"]["start_time"], "10:00")
            self.assertFalse(window._settings_dirty())

    def test_choosing_a_unit_keeps_the_number_as_typed(self) -> None:
        with self.window() as window:
            field = window.settings_recording_seconds
            field.number.setValue(90)
            self.assertEqual(field.value(), 90)
            field.unit.setCurrentIndex(field.unit.findData("min"))
            self.assertEqual(field.number.value(), 90)
            self.assertEqual(field.value(), 5400)
            self.choose(field, 1.5, "h")
            self.assertEqual(field.value(), 5400)
            self.assertEqual(field.number.text(), "1.5")
            self.choose(field, 2.5, "s")
            self.assertEqual(field.value(), 2.5)

    def test_any_number_can_be_typed_before_the_unit_is_chosen(self) -> None:
        with self.window() as window:
            every = window.settings_record_interval
            self.choose(every, 1, "day")
            every.number.setValue(12)  # typing 12 while the unit still says days
            self.assertEqual(every.number.value(), 12)
            every.unit.setCurrentIndex(every.unit.findData("h"))
            self.assertEqual(every.value(), 720)
            self.assertIsNone(every.problem())
            self.choose(every, 1.5, "h")
            self.assertEqual(every.value(), 90)
            self.assertIsInstance(every.value(), int)

    def test_out_of_range_values_are_flagged_and_not_saved(self) -> None:
        with self.window() as window:
            length, every = window.settings_recording_seconds, window.settings_record_interval
            self.choose(length, 100, "h")  # a recording is at most a day
            self.assertEqual(length.problem(), "at most 24 hours")
            self.choose(every, 5, "day")  # timers repeat daily at most
            self.assertEqual(every.problem(), "at most 1 day")
            window._refresh_settings_preview()
            self.assertEqual(window.settings_check_status.objectName(), "StatusFail")
            self.assertIn("Problem: Record for: at most 24 hours.", window.settings_issues_label.text())
            self.assertIn("Problem: Record every: at most 1 day.", window.settings_issues_label.text())
            with mock.patch.object(QMessageBox, "critical") as refused:
                self.assertFalse(window._save_settings(offer_apply=False))
            self.assertIn("Record every: at most 1 day.", refused.call_args[0][2])
            saved = load_config(window.config_path)
            self.assertEqual((saved["capture"]["recording_seconds"], saved["capture"]["record_interval_minutes"]), (20, 30))

            self.choose(length, 2, "h")
            self.choose(every, 0.5, "day")
            self.assertEqual(window._settings_field_problems(), [])
            self.assertTrue(window._save_settings(offer_apply=False))
            self.assertEqual(load_config(window.config_path)["capture"]["record_interval_minutes"], 720)

    def test_unit_names_follow_the_number(self) -> None:
        with self.window() as window:
            field = window.settings_record_interval
            self.choose(field, 1, "h")
            self.assertEqual(field.unit.currentText(), "hour")
            field.number.setValue(2)
            self.assertEqual(field.unit.currentText(), "hours")

    def test_changed_start_time_offers_to_restart_the_running_schedule(self) -> None:
        with self.window() as window:
            window._apply_box_status(_quiet_status(schedule_state=box_status.SCHEDULE_ON))
            installed = self.root / "installed"
            from bumblebox_v2.systemd_units import write_systemd_units

            write_systemd_units(config=window.config, config_path=window.config_path, output_dir=installed)
            real_check = box_status.installed_units_are_current
            started = []
            window._start_automation = lambda targets=None: started.append(targets)
            # 09:10, not 09:30: with a 30-minute interval, 09:30 gives the same slots as midnight.
            window.settings_start_time.setTime(qt_gui.QTime(9, 10))
            with mock.patch.object(
                qt_gui,
                "installed_units_are_current",
                side_effect=lambda config, path: real_check(config, path, installed_dir=installed),
            ), mock.patch.object(window, "_ask_apply_schedule", return_value=True):
                window._save_settings(offer_apply=True)
            self.assertEqual(len(started), 1)

    def test_file_splitting_and_restart_settings(self) -> None:
        with self.window() as window:
            self.assertEqual(window.settings_chunk_minutes.value(), 10.0)
            self.assertEqual(window.settings_restart.currentData(), "auto")
            # Splitting only exists when encoding while recording.
            self.assertTrue(window.settings_chunk_minutes.isHidden())
            window.settings_frame_buffer.setCurrentIndex(window.settings_frame_buffer.findData("stream"))
            self.assertFalse(window.settings_chunk_minutes.isHidden())

            window.settings_chunk_minutes.setValue(5)
            window.settings_restart.setCurrentIndex(window.settings_restart.findData("never"))
            self.assertTrue(window._save_settings(offer_apply=False))
            saved = load_config(window.config_path)
            self.assertEqual(saved["capture"]["chunk_minutes"], 5.0)
            self.assertIs(saved["capture"]["restart_on_failure"], False)
            self.assertEqual(window.settings_restart.currentData(), "never")  # and it reads back as saved

            window.settings_chunk_minutes.setValue(0)
            self.assertEqual(window.settings_chunk_minutes.text(), "Never (one file)")
            window.settings_restart.setCurrentIndex(window.settings_restart.findData("always"))
            self.assertTrue(window._save_settings(offer_apply=False))
            saved = load_config(window.config_path)
            self.assertEqual(saved["capture"]["chunk_minutes"], 0.0)
            self.assertIs(saved["capture"]["restart_on_failure"], True)
            self.assertEqual(window.settings_restart.currentData(), "always")
            self.assertFalse(window._settings_dirty())

    def test_results_show_split_recordings_and_runs_that_did_not_finish(self) -> None:
        with self.window() as window:
            split = ResultButtonTests.payload(video_parts=["a_part001.mp4", "a_part002.mp4", "a_part003.mp4"])
            window._run_history_loaded(window._run_history_generation, [split], None)
            self.assertIn("Video: 3 files, one after the other", window.result_detail.text())

            unfinished = ResultButtonTests.payload(
                success=False,
                unfinished=True,
                errors=["Did not finish: the recorder was stopped, crashed, or lost power. 2 video file(s) with 40 frames were saved before that."],
                video_parts=["b_part001.mp4", "b_part002.mp4"],
            )
            window._run_history_loaded(window._run_history_generation, [unfinished], None)
            self.assertIn("FAILED", window.latest_card.value_label.text())
            self.assertIn("Did not finish", window.latest_card.detail_label.text())
            self.assertIn("2 video file(s) with 40 frames were saved", window.latest_card.detail_label.text())
            self.assertEqual(window.results_table.item(0, 1).text(), "Failed")

    def test_wizard_uses_the_same_fields(self) -> None:
        with self.window() as window:
            wizard = qt_gui.BumbleBoxSetupWizard(window.config_path, window.config, window)
            try:
                page = wizard.experiment_page
                self.assertEqual((page.recording_seconds.value(), page.interval_minutes.value()), (20, 30))
                self.choose(page.recording_seconds, 10, "min")
                self.choose(page.interval_minutes, 2, "h")
                page.start_time.setTime(qt_gui.QTime(8, 15))
                self.assertEqual((page.recording_seconds.value(), page.interval_minutes.value()), (600, 120))
                self.assertEqual(page.start_time.time().toString("HH:mm"), "08:15")
            finally:
                wizard.close()


CAM_A = "/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a"
CAM_B = "/base/axi/pcie@120000/rp1/i2c@80000/imx477@1a"


class TwoCameraTests(QtControlTestCase):
    """A Pi with two cameras: one config each, one window that switches between them."""

    def detected(self):
        from bumblebox_v2.camera_select import DetectedCamera

        return [DetectedCamera(0, "imx477", CAM_A), DetectedCamera(1, "imx477", CAM_B)]

    def add_second(self, window, *, pinned: bool = True, colony: str = "07"):
        cameras = self.detected() if pinned else []
        with mock.patch.object(qt_gui, "detect_cameras", return_value=cameras), mock.patch.object(
            window, "_ask_second_camera_details", return_value=(colony, str(self.root / "data"))
        ), mock.patch.object(QMessageBox, "information") as told:
            window._add_second_camera()
        return told

    def second_path(self, window) -> Path:
        return window.primary_config_path.with_name("config.camera2.yaml")

    def test_one_camera_box_shows_no_camera_controls(self) -> None:
        with self.window() as window:
            self.assertEqual(len(window.cameras), 1)
            for hidden in (window.camera_switch, window.cameras_strip, window.run_both_button,
                           window.both_cameras_checkbox, window.pin_cameras_button,
                           window.remove_second_camera_button):
                self.assertTrue(hidden.isHidden())
            self.assertFalse(window.add_second_camera_button.isHidden())
            self.assertEqual(window.page_title.text(), "OVERVIEW")
            self.assertEqual(window.run_once_button.text(), "Run Once Now")

    def test_adding_a_second_camera_creates_its_settings_and_shows_the_switcher(self) -> None:
        with self.window() as window:
            told = self.add_second(window)
            self.assertEqual(told.call_args[0][1], "Second camera added")
            self.assertEqual(len(window.cameras), 2)
            second = load_config(self.second_path(window))
            self.assertEqual(second["system"]["colony_id"], "07")
            self.assertEqual(second["camera"]["camera_id"], CAM_B)
            self.assertEqual(window.config["camera"]["camera_id"], CAM_A)  # the first camera was pinned too
            self.assertFalse(window.camera_switch.isHidden())
            self.assertFalse(window.cameras_strip.isHidden())
            self.assertEqual(window.camera_buttons[2].text(), "Camera 2 (colony 07)")
            self.assertTrue(window.camera_buttons[1].isChecked())
            self.assertTrue(window.add_second_camera_button.isHidden())
            self.assertTrue(window.pin_cameras_button.isHidden())
            self.assertFalse(window.remove_second_camera_button.isHidden())
            self.assertEqual(window.page_title.text(), "OVERVIEW  ·  CAMERA 1")
            self.assertEqual(window.run_once_button.text(), "Record Camera 1 Now")

    def test_cancelling_the_dialog_adds_nothing(self) -> None:
        with self.window() as window:
            with mock.patch.object(qt_gui, "detect_cameras", return_value=[]), mock.patch.object(
                window, "_ask_second_camera_details", return_value=None
            ):
                window._add_second_camera()
            self.assertEqual(len(window.cameras), 1)
            self.assertFalse(self.second_path(window).exists())

    def test_unpinned_cameras_offer_pinning(self) -> None:
        with self.window() as window:
            self.add_second(window, pinned=False)
            self.assertFalse(window.pin_cameras_button.isHidden())
            self.assertIn("not pinned", window.second_camera_label.text())
            with mock.patch.object(qt_gui, "pin_cameras", return_value=False), mock.patch.object(
                QMessageBox, "warning"
            ) as warned:
                window._pin_cameras()
            self.assertEqual(warned.call_args[0][1], "Cameras not pinned")

            from bumblebox_v2 import multi_camera

            with mock.patch.object(multi_camera, "detect_cameras", return_value=self.detected()), mock.patch.object(
                QMessageBox, "information"
            ):
                window._pin_cameras()
            self.assertTrue(window.pin_cameras_button.isHidden())
            self.assertIn("pinned to one physical camera", window.second_camera_label.text())

    def test_switching_shows_the_other_cameras_settings_and_status(self) -> None:
        with self.window() as window:
            self.add_second(window)
            second = load_config(self.second_path(window))
            second["capture"]["recording_seconds"] = 44
            save_config(self.second_path(window), second)
            on = _quiet_status(schedule_state=box_status.SCHEDULE_ON)
            window._box_status_loaded(window._box_status_generation, {1: _quiet_status(), 2: on})
            self.assertIn("Camera 2 (colony 07): On", window.camera_status_labels[2].text())
            self.assertIn("(shown below)", window.camera_status_labels[1].text())

            window._select_camera(2)
            self.assertEqual(window.config_path, self.second_path(window))
            self.assertEqual(window.settings_recording_seconds.value(), 44)
            self.assertEqual(window.page_title.text(), "OVERVIEW  ·  CAMERA 2")
            self.assertTrue(window.camera_buttons[2].isChecked())
            self.assertEqual(window.schedule_card.value_label.text(), "On · next recording 14:30 (in 12 min)")
            self.assertEqual(window.camera_card.title_label.text(), "CAMERA 2")
            self.assertIn("(shown below)", window.camera_status_labels[2].text())

            window._select_camera(1)
            self.assertEqual(window.config_path, window.primary_config_path)
            self.assertEqual(window.settings_recording_seconds.value(), 20)
            self.assertEqual(window.schedule_card.value_label.text(), "Off")

    def test_unsaved_settings_hold_the_switch(self) -> None:
        with self.window() as window:
            self.add_second(window)
            window.settings_recording_seconds.setValue(33)
            with mock.patch.object(window, "_ask_unsaved_settings_on_close", return_value="cancel"):
                window._select_camera(2)
            self.assertEqual(window.camera_index, 1)
            self.assertTrue(window.camera_buttons[1].isChecked())
            # Recording length is part of the schedule, so saving also asks whether
            # Camera 2 should follow; here it should not.
            with mock.patch.object(window, "_ask_unsaved_settings_on_close", return_value="save"), mock.patch.object(
                window, "_ask_copy_schedule", return_value=False
            ):
                window._select_camera(2)
            self.assertEqual(window.camera_index, 2)
            self.assertEqual(load_config(window.primary_config_path)["capture"]["recording_seconds"], 33)

    def test_each_camera_sees_only_its_own_recordings(self) -> None:
        with self.window() as window:
            self.add_second(window)
            day = self.root / "data" / "2026-11-18"
            for name, tag in (("host_2026-11-18_14_00_00", ""), ("host_cam2_2026-11-18_14_00_00", "cam2")):
                session = day / name
                session.mkdir(parents=True)
                (session / f"{name}_run_summary.json").write_text(
                    json.dumps(
                        {"session_name": name, "session_dir": str(session), "started_at": "2026-11-18T14:00:00",
                         "mode": "record_only", "success": True, "frames_captured": 10, "actual_fps": 5.0,
                         "camera_session_tag": tag}
                    )
                )
            window.refresh_results()
            # A scan may already have been running while the files were being written;
            # wait for one that has seen both.
            self.pump(
                lambda: bool(window._run_history_payloads)
                and bool(window._other_camera_payloads.get(2))
                and not window._run_history_loading
            )
            self.assertEqual([p["session_name"] for p in window._run_history_payloads], ["host_2026-11-18_14_00_00"])
            self.assertEqual(
                [p["session_name"] for p in window._other_camera_payloads[2]], ["host_cam2_2026-11-18_14_00_00"]
            )
            # Switch while a scan for Camera 1 is still running: its result must be dropped.
            window.refresh_results()
            window._select_camera(2)
            self.pump(lambda: not window._run_history_loading and bool(window._run_history_payloads))
            self.assertEqual(
                [p["session_name"] for p in window._run_history_payloads], ["host_cam2_2026-11-18_14_00_00"]
            )
            self.assertEqual(window.results_table.rowCount(), 1)

    def test_storage_card_counts_both_cameras_on_the_same_drive(self) -> None:
        with self.window() as window:
            self.add_second(window)
            signature = box_status.recording_signature(window.config)

            def run(tag):
                return ResultButtonTests.payload(
                    mode="record_and_track", frames_captured=100, _session_bytes=150e6,
                    _recording_signature=signature, camera_session_tag=tag,
                )

            window._run_history_loaded(window._run_history_generation, [run("")], None, {2: [run("cam2")]})
            window._apply_box_status(_quiet_status(free_gb=212))
            detail = window.storage_card.detail_label.text()
            self.assertIn("About 7.2 GB per day (14 GB with Camera 2)", detail)
            self.assertIn("about 15 days of space left", detail)

    def test_start_and_stop_act_on_both_cameras_by_default(self) -> None:
        with self.window() as window, mock.patch.object(
            qt_gui, "review_schedule_start", return_value=box_status.StartReview()
        ), mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False):
            self.add_second(window)
            first, second = str(window.primary_config_path), str(self.second_path(window))
            started = []
            window.run_bbx_commands = lambda commands, **kwargs: started.append((commands, kwargs))
            window._start_automation()
            commands, kwargs = started[0]
            self.assertEqual(
                commands,
                [["systemd-write", "--config", first], ["systemd-install", "--config", first],
                 ["systemd-write", "--config", second], ["systemd-install", "--config", second]],
            )
            self.assertEqual(kwargs["success_message"][0], "Automated recording started for both cameras")
            self.assertTrue(load_config(first)["scheduling"]["enabled"])
            self.assertTrue(load_config(second)["scheduling"]["enabled"])

            on = _quiet_status(schedule_state=box_status.SCHEDULE_ON)
            window._box_status_loaded(window._box_status_generation, {1: on, 2: on})
            with mock.patch.object(window, "_confirm_stop_automation", return_value=True) as asked:
                window._stop_automation()
            self.assertEqual(len(asked.call_args[0][0]), 2)
            self.assertEqual(
                started[1][0], [["systemd-disable", "--config", first], ["systemd-disable", "--config", second]]
            )
            self.assertFalse(load_config(second)["scheduling"]["enabled"])

    def test_one_camera_can_be_started_and_stopped_alone(self) -> None:
        with self.window() as window, mock.patch.object(
            qt_gui, "review_schedule_start", return_value=box_status.StartReview()
        ), mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False):
            self.add_second(window)
            window.both_cameras_checkbox.setChecked(False)
            started = []
            window.run_bbx_commands = lambda commands, **kwargs: started.append(commands)
            window._start_automation()
            first = str(window.primary_config_path)
            self.assertEqual(started[0], [["systemd-write", "--config", first], ["systemd-install", "--config", first]])
            self.assertFalse(load_config(self.second_path(window))["scheduling"]["enabled"])

            # With both ticked, a camera whose schedule is already off is left out of Stop.
            window.both_cameras_checkbox.setChecked(True)
            window._box_status_loaded(
                window._box_status_generation,
                {1: _quiet_status(schedule_state=box_status.SCHEDULE_ON), 2: _quiet_status()},
            )
            stopped = []
            window.run_bbx_command = lambda arguments, **kwargs: stopped.append(arguments)
            with mock.patch.object(window, "_confirm_stop_automation", return_value=True):
                window._stop_automation()
            self.assertEqual(stopped, [["systemd-disable", "--config", first]])

    def test_too_much_memory_for_both_cameras_is_raised_before_starting(self) -> None:
        from bumblebox_v2.multi_camera import CombinedLoad

        with self.window() as window, mock.patch.object(
            qt_gui, "review_schedule_start", return_value=box_status.StartReview()
        ), mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False), mock.patch.object(
            qt_gui, "combined_memory", return_value=CombinedLoad("fail", "Both cameras need 7.0 GiB of 8.0 GiB.")
        ):
            self.add_second(window)
            started = []
            window.run_bbx_commands = lambda commands, **kwargs: started.append(commands)
            asked = []
            window._confirm_start_despite_problems = lambda review: asked.append(review) or False
            window._start_automation()
            self.assertEqual(started, [])
            self.assertIn("Both cameras need 7.0 GiB of 8.0 GiB.", asked[0].problems)

    def test_record_both_runs_one_command_for_the_pair(self) -> None:
        with self.window() as window, mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False):
            self.add_second(window)
            ran = []
            window.run_bbx_command = lambda arguments, **kwargs: ran.append(arguments)
            window._run_both_once()
            self.assertEqual(
                ran,
                [["run-once", "--all-cameras", "--mode", "record_and_track", "--config", str(window.primary_config_path)]],
            )

    def test_record_both_through_the_real_command(self) -> None:
        def mock_cameras(config):
            config["local_index"]["enabled"] = False
            config["pipeline"]["mode"] = "record_only"
            config["camera"].update({"width": 64, "height": 48, "codec": "mjpeg"})
            config["capture"]["recording_seconds"] = 0.5
            config["runtime"].update({"use_mock_camera": True, "fps_report_on_each_recording": False})

        with tempfile.TemporaryDirectory() as locks, mock.patch.dict(
            os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(Path(locks) / "capture.lock")}
        ), self.window(mock_cameras) as window, mock.patch(
            "bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False
        ):
            self.add_second(window, pinned=False)
            window.run_mode.setCurrentIndex(window.run_mode.findData("record_only"))
            window._run_both_once()
            self.pump(lambda: window.command_status.objectName() == "StatusGood", timeout=60)
            output = window.console.toPlainText()
            self.assertIn("Camera 1: recorded.", output)
            self.assertIn("Camera 2: recorded.", output)
            self.assertEqual(window.command_status.text(), "Recording finished.")

            self.pump(lambda: bool(window._run_history_payloads) and bool(window._other_camera_payloads.get(2)))
            self.assertNotIn("_cam2_", window._run_history_payloads[0]["session_name"])
            self.assertIn("_cam2_", window._other_camera_payloads[2][0]["session_name"])
            self.assertEqual(window.results_table.rowCount(), 1)

    def test_changed_schedule_can_be_given_to_the_other_camera(self) -> None:
        with self.window() as window:
            self.add_second(window)
            window.settings_record_interval.setValue(15)
            with mock.patch.object(window, "_ask_copy_schedule", return_value=False) as asked:
                window._save_settings(offer_apply=False)
            self.assertEqual(asked.call_count, 1)
            self.assertEqual(load_config(self.second_path(window))["capture"]["record_interval_minutes"], 30)

            window.settings_record_interval.setValue(10)
            with mock.patch.object(window, "_ask_copy_schedule", return_value=True):
                window._save_settings(offer_apply=False)
            self.assertEqual(load_config(self.second_path(window))["capture"]["record_interval_minutes"], 10)

            # A change that does not touch the schedule asks nothing.
            window.settings_fps.setValue(4.0)
            with mock.patch.object(window, "_ask_copy_schedule", side_effect=AssertionError("should not ask")):
                window._save_settings(offer_apply=False)

    def test_removing_the_second_camera_stops_its_schedule_first(self) -> None:
        with self.window() as window:
            self.add_second(window)
            window._box_status_loaded(
                window._box_status_generation,
                {1: _quiet_status(), 2: _quiet_status(schedule_state=box_status.SCHEDULE_ON)},
            )
            window._select_camera(2)
            queued = []
            window.run_bbx_commands = lambda commands, **kwargs: queued.append((commands, kwargs))
            with mock.patch.object(window, "_confirm_remove_second_camera", return_value=False):
                window._remove_second_camera()
            self.assertEqual(queued, [])

            with mock.patch.object(window, "_confirm_remove_second_camera", return_value=True):
                window._remove_second_camera()
            commands, kwargs = queued[0]
            self.assertEqual(commands, [["systemd-disable", "--config", str(self.second_path(window))]])
            self.assertTrue(self.second_path(window).exists())  # not removed until its timers are off

            with mock.patch.object(QMessageBox, "information") as told:
                kwargs["on_success"]()
            self.assertEqual(told.call_args[0][1], "Second camera removed")
            self.assertFalse(self.second_path(window).exists())
            self.assertEqual(len(list(self.root.glob("config.camera2.removed-*.yaml"))), 1)
            self.assertEqual((len(window.cameras), window.camera_index), (1, 1))
            self.assertEqual(window.config_path, window.primary_config_path)
            self.assertTrue(window.camera_switch.isHidden())

    def test_second_camera_with_no_schedule_is_removed_at_once(self) -> None:
        with self.window() as window:
            self.add_second(window)
            window._box_status_loaded(window._box_status_generation, {1: _quiet_status(), 2: _quiet_status()})
            window.run_bbx_commands = lambda commands, **kwargs: self.fail("no command should be needed")
            with mock.patch.object(window, "_confirm_remove_second_camera", return_value=True), mock.patch.object(
                QMessageBox, "information"
            ):
                window._remove_second_camera()
            self.assertEqual(len(window.cameras), 1)

    def test_the_add_dialog_explains_and_prefills(self) -> None:
        with self.window() as window:
            dialog = qt_gui.SecondCameraDialog(window, detected=self.detected(), colony_id="02", data_root="/data")
            try:
                self.assertEqual(dialog.colony_id.text(), "02")
                self.assertEqual(dialog.data_root.text(), "/data")
                self.assertIn("imx477 on connector i2c@88000", dialog.detected_label.text())
                self.assertIn("pinned", dialog.detected_label.text())
            finally:
                dialog.close()
            alone = qt_gui.SecondCameraDialog(window, detected=self.detected()[:1], colony_id="02", data_root="/data")
            try:
                self.assertIn("1 camera(s) detected", alone.detected_label.text())
            finally:
                alone.close()


if __name__ == "__main__":
    unittest.main()
