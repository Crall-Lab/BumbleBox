from __future__ import annotations

import unittest
from unittest.mock import patch

from bumblebox_v2.config import load_defaults
from bumblebox_v2.realsense_camera import (
    RealSenseCheckResult,
    RealSenseDevice,
    apply_detected_realsense_config,
    format_realsense_check_result,
    run_realsense_check,
)


class RealSenseCameraTests(unittest.TestCase):
    def test_missing_python_binding_is_reported_without_crashing(self) -> None:
        with patch(
            "bumblebox_v2.realsense_camera.discover_realsense_devices",
            side_effect=RuntimeError("pyrealsense2 is missing"),
        ):
            result = run_realsense_check(load_defaults(), probe=False)

        self.assertFalse(result.module_available)
        self.assertIn("pyrealsense2 is missing", result.errors[0])

    def test_check_format_includes_supported_stream_profiles(self) -> None:
        device = RealSenseDevice(
            serial="1234",
            name="Intel RealSense D405",
            product_line="D400",
            firmware_version="test",
            usb_type="3.2",
            physical_port="test-port",
            depth_profiles=["848x480@30 format.z16"],
            color_profiles=["1280x720@30 format.bgr8"],
        )
        result = RealSenseCheckResult(
            module_available=True,
            sdk_version="test",
            preferred_serial=None,
            selected_serial="1234",
            devices=[device],
            requested_depth_profile="848x480@30 z16",
            requested_color_profile="848x480@30 bgr8",
            align_to="none",
            probe_attempted=False,
            probe_succeeded=False,
            depth_shape=None,
            depth_dtype=None,
            color_shape=None,
            color_dtype=None,
            depth_scale_meters=None,
            device_timestamp_ms=None,
            warnings=[],
            errors=[],
        )

        report = format_realsense_check_result(result)

        self.assertIn("848x480@30", report)
        self.assertIn("1280x720@30", report)

    def test_apply_requires_a_successful_stream_probe(self) -> None:
        result = RealSenseCheckResult(
            module_available=True,
            sdk_version="test",
            preferred_serial=None,
            selected_serial="1234",
            devices=[],
            requested_depth_profile="848x480@30 z16",
            requested_color_profile="848x480@30 bgr8",
            align_to="none",
            probe_attempted=False,
            probe_succeeded=False,
            depth_shape=None,
            depth_dtype=None,
            color_shape=None,
            color_dtype=None,
            depth_scale_meters=None,
            device_timestamp_ms=None,
            warnings=[],
            errors=[],
        )

        with self.assertRaises(RuntimeError):
            apply_detected_realsense_config(load_defaults(), result)


if __name__ == "__main__":
    unittest.main()
