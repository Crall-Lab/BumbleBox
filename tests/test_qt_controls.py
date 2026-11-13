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
            window._start_automation = lambda: started.append(True)
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

    def test_scrolling_over_a_field_does_not_change_it(self) -> None:
        with self.window() as window:
            spin = window.settings_recording_seconds
            event = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
            )
            QApplication.sendEvent(spin, event)
            self.assertEqual(spin.value(), 20)


if __name__ == "__main__":
    unittest.main()
