from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from bumblebox_v2.qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    from PyQt5.QtWidgets import QApplication

    from bumblebox_v2.config import load_defaults, save_config
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


if __name__ == "__main__":
    unittest.main()
