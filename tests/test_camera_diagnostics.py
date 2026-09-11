from __future__ import annotations

import unittest

from bumblebox_v2.camera_setup import interpret_camera_connection_diagnostics


class CameraConnectionDiagnosticTests(unittest.TestCase):
    def test_owlsight_remote_io_failure_has_physical_connection_guidance(self) -> None:
        status, findings, actions = interpret_camera_connection_diagnostics(
            expected_sensor="ov64a40",
            detected_cameras=[],
            picamera2_available=True,
            diagnostic_text="ov64a40 11-0036: Failed to read chip id: error -121",
        )

        self.assertEqual(status, "connection_failure")
        self.assertTrue(any("I2C" in finding for finding in findings))
        self.assertTrue(any("Shut" in action and "down" in action for action in actions))
        self.assertTrue(any("ribbon" in action for action in actions))

    def test_detected_expected_sensor_is_ready(self) -> None:
        status, findings, actions = interpret_camera_connection_diagnostics(
            expected_sensor="ov64a40",
            detected_cameras=[{"Model": "ov64a40", "Id": "camera0"}],
            picamera2_available=True,
        )

        self.assertEqual(status, "ready")
        self.assertIn("visible", findings[0])
        self.assertEqual(actions, [])

    def test_different_detected_sensor_reports_profile_mismatch(self) -> None:
        status, _findings, actions = interpret_camera_connection_diagnostics(
            expected_sensor="ov64a40",
            detected_cameras=[{"Model": "imx477"}],
            picamera2_available=True,
        )

        self.assertEqual(status, "mismatch")
        self.assertTrue(any("profile" in action for action in actions))


if __name__ == "__main__":
    unittest.main()
