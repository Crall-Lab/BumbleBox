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
                self.assertEqual(window.pages.count(), 4)

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
            finally:
                wizard.close()
                window.close()


if __name__ == "__main__":
    unittest.main()
