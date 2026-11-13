from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from bumblebox_v2.qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    from PyQt5.QtCore import QProcess
    from PyQt5.QtWidgets import QApplication

    from bumblebox_v2 import box_status
    from bumblebox_v2.config import load_config, load_defaults, save_config
    from bumblebox_v2.qt_gui import (
        BumbleBoxQtGUI,
        BumbleBoxSetupWizard,
        PAGE_DISTRIBUTED,
        PAGE_EXPERIMENT,
        PAGE_THERMAL,
    )
except Exception:  # pragma: no cover - optional on non-GUI test hosts
    QApplication = None


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class QtGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_window_and_conditional_wizard_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = load_defaults()
            config["setup"]["completed"] = True
            save_config(config_path, config)
            window = BumbleBoxQtGUI(config_path)
            wizard = BumbleBoxSetupWizard(config_path, config)
            try:
                self.assertEqual(window.pages.count(), 5)
                self.assertEqual(window.live_monitor_button.text(), "Open Live Camera Monitor")

                wizard.hardware_page.profile.setCurrentIndex(
                    wizard.hardware_page.profile.findData("rgb_only")
                )
                self.assertEqual(wizard.rgb_page.nextId(), PAGE_EXPERIMENT)

                wizard.hardware_page.profile.setCurrentIndex(
                    wizard.hardware_page.profile.findData("multimodal")
                )
                self.assertEqual(wizard.rgb_page.nextId(), PAGE_THERMAL)
                self.assertTrue(wizard.uses_thermal())
                self.assertTrue(wizard.uses_realsense())

                wizard.hardware_page.second_pi.setChecked(True)
                self.assertTrue(wizard.uses_second_pi())
                self.assertEqual(wizard.realsense_page.nextId(), PAGE_DISTRIBUTED)
                self.assertEqual(
                    wizard.distributed_page.realsense_location.currentData(), "worker"
                )
            finally:
                wizard.close()
                window.close()

    def test_results_page_reports_realsense_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = load_defaults()
            config["setup"]["completed"] = True
            config["system"]["data_root"] = str(root)
            save_config(config_path, config)
            session_dir = root / "session-a"
            session_dir.mkdir()
            rgb_video = session_dir / "session-a.mp4"
            depth_video = session_dir / "session-a_realsense_depth_preview.avi"
            rgb_video.touch()
            depth_video.touch()
            payload = {
                "session_name": "session-a",
                "session_dir": str(session_dir),
                "_session_dir_local": str(session_dir),
                "started_at": "2026-09-11T12:00:00-05:00",
                "mode": "record_only",
                "success": True,
                "frames_captured": 70,
                "actual_fps": 7.0,
                "video_path": str(rgb_video),
                "thermal_enabled": False,
                "thermal_frames_captured": 0,
                "realsense_enabled": True,
                "realsense_frames_captured": 300,
                "realsense_actual_fps": 30.0,
                "realsense_depth_scale_meters": 0.001,
                "realsense_depth_preview_video_path": str(depth_video),
                "warnings": [],
                "errors": [],
            }
            window = BumbleBoxQtGUI(config_path)
            try:
                window._run_history_loaded(window._run_history_generation, [payload], None)
                self.app.processEvents()
                self.assertEqual(window.results_table.rowCount(), 1)
                self.assertEqual(window.results_table.item(0, 4).text(), "Off")
                self.assertEqual(window.results_table.item(0, 5).text(), "300")
                self.assertTrue(window.latest_depth_button.isEnabled())
                self.assertIn("RealSense: 300 frames", window.result_detail.text())
                self.assertTrue(window.result_artifact_buttons["rgb"][0].isEnabled())
                self.assertTrue(window.result_artifact_buttons["depth"][0].isEnabled())
                self.assertFalse(window.result_artifact_buttons["thermal"][0].isEnabled())
            finally:
                window.close()


DRIVE_REASON = (
    "The data drive is not mounted: /etc/fstab lists a drive for /mnt/bumblebox/data, "
    "but nothing is mounted there."
)


def _status(storage_level: str, storage_headline: str, schedule_level: str, schedule_headline: str,
            schedule_detail: str = "Records 20 s every 30 min, then tracks."):
    return box_status.BoxStatus(
        storage=box_status.StatusLine(storage_level, storage_headline, "/mnt/bumblebox/data"),
        schedule=box_status.ScheduleStatus(
            box_status.SCHEDULE_ON, schedule_level, schedule_headline, schedule_detail
        ),
        checked_at=datetime.now(),
    )


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class QtBoxStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @contextmanager
    def window(self):
        """A window whose background status check returns a fixed, quiet answer."""
        quiet = _status(box_status.LEVEL_OK, "Drive mounted · 212 GB free", box_status.LEVEL_OFF, "Off")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "bumblebox_v2.qt_gui.collect_box_status", return_value=quiet
        ):
            root = Path(directory)
            config_path = root / "config.yaml"
            config = load_defaults()
            config["setup"]["completed"] = True
            config["system"]["data_root"] = str(root)
            save_config(config_path, config)
            window = BumbleBoxQtGUI(config_path)
            try:
                yield window
            finally:
                window.close()

    def test_cards_and_schedule_panel_show_the_box_status(self) -> None:
        with self.window() as window:
            self.assertEqual(window.storage_card.value_label.text(), "Checking...")

            window._apply_box_status(
                _status(
                    box_status.LEVEL_FAIL,
                    "Data drive not mounted",
                    box_status.LEVEL_FAIL,
                    "On · next recording 14:30 (in 12 min)",
                    f"Last scheduled run (14:00) FAILED: Data drive not mounted.\n{DRIVE_REASON}",
                )
            )
            self.assertEqual(window.storage_card.value_label.text(), "Data drive not mounted")
            self.assertEqual(window.storage_card.property("tone"), "fail")
            self.assertEqual(window.schedule_card.value_label.text(), "On · next recording 14:30 (in 12 min)")
            self.assertEqual(window.schedule_card.property("tone"), "fail")
            self.assertIn(DRIVE_REASON, window.schedule_card.detail_label.text())
            self.assertEqual(window.schedule_status_label.objectName(), "StatusFail")
            self.assertIn("On · next recording 14:30", window.schedule_status_label.text())
            self.assertIn("FAILED: Data drive not mounted", window.schedule_detail_label.text())

            window._apply_box_status(
                _status(
                    box_status.LEVEL_OK,
                    "Drive mounted · 212 GB free",
                    box_status.LEVEL_OK,
                    "On · next recording 15:00 (in 20 min)",
                )
            )
            self.assertEqual(window.storage_card.property("tone"), "mint")
            self.assertEqual(window.schedule_card.property("tone"), "mint")
            self.assertEqual(window.schedule_status_label.objectName(), "StatusGood")

    def test_status_check_runs_in_the_background_and_reaches_the_cards(self) -> None:
        with self.window() as window:
            for _ in range(200):
                if window._box_status is not None:
                    break
                window.thread_pool.waitForDone(20)
                self.app.processEvents()
            self.assertIsNotNone(window._box_status)
            self.assertEqual(window.storage_card.value_label.text(), "Drive mounted · 212 GB free")
            self.assertEqual(window.schedule_card.value_label.text(), "Off")
            self.assertEqual(window.schedule_card.property("tone"), "neutral")

    def test_failed_command_is_explained_in_plain_words(self) -> None:
        with self.window() as window:
            shown = []
            window._show_failure_dialog = shown.append
            self.assertTrue(window.command_status.isHidden())

            window._current_command = ["run-once", "--mode", "record_only"]
            window._command_output = [f"Run failed to start: {DRIVE_REASON}\n"]
            window.command_queue = [["systemd-install"]]
            window._process_finished(1, QProcess.NormalExit)

            self.assertEqual(len(shown), 1)
            self.assertEqual(shown[0].title, "Data drive not mounted")
            self.assertEqual(shown[0].reason, DRIVE_REASON)
            self.assertEqual(window.command_status.objectName(), "StatusFail")
            self.assertIn("Data drive not mounted", window.command_status.text())
            self.assertIn(DRIVE_REASON, window.command_status.text())
            self.assertFalse(window.command_status.isHidden())
            self.assertEqual(window.command_queue, [])

    def test_refused_recording_reaches_the_window_through_the_real_command(self) -> None:
        with self.window() as window:
            # No drive can have this much free space, so the storage guard
            # refuses the run before the camera is touched.
            config = load_config(window.config_path)
            config["system"]["min_free_space_gb"] = 10_000_000
            save_config(window.config_path, config)
            shown = []
            window._show_failure_dialog = shown.append

            window.run_bbx_command(["run-once", "--mode", "record_only"])
            self.assertEqual(window.command_status.objectName(), "StatusNeutral")
            self.assertIn("Recording: running since", window.command_status.text())
            for _ in range(600):
                if window.process.state() == QProcess.NotRunning and shown:
                    break
                window.process.waitForFinished(100)
                self.app.processEvents()

            self.assertEqual(len(shown), 1, window.console.toPlainText())
            self.assertEqual(shown[0].title, "Data drive is nearly full")
            self.assertIn("GB is free", shown[0].reason)
            self.assertNotIn("Run failed to start", shown[0].reason)
            self.assertEqual(window.command_status.objectName(), "StatusFail")
            self.assertIn("Run failed to start", window.console.toPlainText())
            self.assertEqual(window.pages.currentIndex(), 1)

    def test_crash_and_failure_to_start_are_reported(self) -> None:
        with self.window() as window:
            shown = []
            window._show_failure_dialog = shown.append
            window._current_command = ["camera-check"]

            window._process_finished(0, QProcess.CrashExit)
            self.assertEqual(shown[-1].title, "Camera check stopped unexpectedly")

            window._process_error(QProcess.Crashed)
            self.assertEqual(len(shown), 1)
            window._process_error(QProcess.FailedToStart)
            self.assertEqual(shown[-1].title, "Camera check could not be started")
            self.assertEqual(window.command_status.objectName(), "StatusFail")

    def test_successful_command_replaces_the_failure_message(self) -> None:
        with self.window() as window:
            window._show_failure_dialog = lambda explanation: None
            window._current_command = ["run-once"]
            window._command_output = ["Run failed to start: nope\n"]
            window._process_finished(1, QProcess.NormalExit)
            self.assertEqual(window.command_status.objectName(), "StatusFail")

            window._command_output = ["Success: True\n"]
            window._process_finished(0, QProcess.NormalExit)
            self.assertEqual(window.command_status.objectName(), "StatusGood")
            self.assertEqual(window.command_status.text(), "Recording finished.")

    def test_schedule_is_reviewed_before_it_is_started(self) -> None:
        review = box_status.StartReview(
            problems=["Memory budget: Estimated frame buffer is 93 GiB."],
            suggestions=["Set capture.frame_buffer to 'stream'."],
        )
        with self.window() as window, mock.patch(
            "bumblebox_v2.qt_gui.review_schedule_start", return_value=review
        ), mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False):
            started = []
            window.run_bbx_commands = lambda commands, **kwargs: started.append(commands)
            asked = []

            window._confirm_start_despite_problems = lambda shown: asked.append(shown) or False
            window._start_automation()
            self.assertEqual(asked, [review])
            self.assertEqual(started, [])
            self.assertFalse(load_config(window.config_path)["scheduling"]["enabled"])

            window._confirm_start_despite_problems = lambda shown: True
            window._start_automation()
            self.assertEqual(started, [[["systemd-write"], ["systemd-install"]]])
            self.assertTrue(load_config(window.config_path)["scheduling"]["enabled"])

    def test_workable_schedule_starts_without_a_question(self) -> None:
        with self.window() as window, mock.patch(
            "bumblebox_v2.qt_gui.review_schedule_start", return_value=box_status.StartReview()
        ), mock.patch("bumblebox_v2.live_monitor.live_monitor_is_running", return_value=False):
            started = []
            window.run_bbx_commands = lambda commands, **kwargs: started.append(commands)
            window._confirm_start_despite_problems = lambda shown: self.fail("should not ask")
            window._start_automation()
            self.assertEqual(len(started), 1)

    def test_latest_recording_card_reports_success_and_failure(self) -> None:
        with self.window() as window:
            started = datetime.now() - timedelta(minutes=18)
            payload = {
                "session_name": "session-a",
                "started_at": started.isoformat(),
                "mode": "record_and_track",
                "success": True,
                "frames_captured": 100,
                "actual_fps": 5.0,
                "warnings": [],
                "errors": [],
            }
            window._run_history_loaded(window._run_history_generation, [payload], None)
            self.assertIn("(18 min ago) · Success", window.latest_card.value_label.text())
            self.assertEqual(window.latest_card.detail_label.text(), "100 frames at 5.0 fps, record and track")
            self.assertEqual(window.latest_card.property("tone"), "mint")
            self.assertIn("Success", window.latest_run_status.text())
            self.assertTrue(window.latest_depth_button.isHidden())

            failed = dict(payload, success=False, errors=["Tracking failed: boom"])
            window._run_history_loaded(window._run_history_generation, [failed], None)
            self.assertIn("FAILED", window.latest_card.value_label.text())
            self.assertEqual(window.latest_card.detail_label.text(), "Tracking failed: boom")
            self.assertEqual(window.latest_card.property("tone"), "fail")

            window._run_history_loaded(window._run_history_generation, [], None)
            self.assertEqual(window.latest_card.value_label.text(), "No recordings yet")
            self.assertEqual(window.latest_card.property("tone"), "neutral")

    def test_finished_scheduled_run_refreshes_the_results_once(self) -> None:
        with self.window() as window:
            refreshed = []
            window.refresh_results = lambda: refreshed.append(True)

            def status_with_last_run(moment):
                status = _status(box_status.LEVEL_OK, "Drive mounted", box_status.LEVEL_OK, "On")
                status.schedule.timers = [
                    box_status.TimerState("main", "t.timer", "t.service", last_started=moment, last_ok=True)
                ]
                return status

            first = datetime(2026, 11, 12, 14, 0)
            window._apply_box_status(status_with_last_run(first))
            window._apply_box_status(status_with_last_run(first))
            self.assertEqual(refreshed, [])
            window._apply_box_status(status_with_last_run(first + timedelta(minutes=30)))
            self.assertEqual(refreshed, [True])


if __name__ == "__main__":
    unittest.main()
