from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from bumblebox_v2.config import load_defaults
from bumblebox_v2.realsense_camera import (
    RealSenseCheckResult,
    RealSenseDevice,
    RealSenseRecordingSession,
    apply_detected_realsense_config,
    format_realsense_check_result,
    run_realsense_check,
)


class _FakeFrame:
    def __init__(self, data, frame_number: int, timestamp_ms: float) -> None:
        self._data = data
        self._frame_number = frame_number
        self._timestamp_ms = timestamp_ms

    def get_data(self):
        return self._data

    def get_timestamp(self) -> float:
        return self._timestamp_ms

    def get_frame_number(self) -> int:
        return self._frame_number

    def get_frame_timestamp_domain(self) -> str:
        return "hardware_clock"


class _FakeFrameset:
    def __init__(self, frame_number: int) -> None:
        depth = np.full((24, 32), 1000 + frame_number, dtype=np.uint16)
        color = np.full((24, 32, 3), frame_number % 255, dtype=np.uint8)
        self._depth = _FakeFrame(depth, frame_number, frame_number * 10.0)
        self._color = _FakeFrame(color, frame_number, frame_number * 10.0 + 0.2)

    def get_depth_frame(self):
        return self._depth

    def get_color_frame(self):
        return self._color


class _FakeDepthSensor:
    def get_depth_scale(self) -> float:
        return 0.001


class _FakeDeviceHandle:
    def first_depth_sensor(self):
        return _FakeDepthSensor()


class _FakePipelineProfile:
    def get_device(self):
        return _FakeDeviceHandle()


class _FakePipeline:
    def __init__(self) -> None:
        self.frame_number = 0

    def start(self, _config):
        return _FakePipelineProfile()

    def wait_for_frames(self, _timeout_ms: int):
        time.sleep(0.004)
        self.frame_number += 1
        return _FakeFrameset(self.frame_number)

    def stop(self) -> None:
        pass


class _FakePipelineConfig:
    def enable_device(self, _serial: str) -> None:
        pass

    def enable_stream(self, *_args) -> None:
        pass


class _FakeRealSense:
    class stream:
        depth = "depth"
        color = "color"

    class format:
        z16 = "z16"
        bgr8 = "bgr8"

    @staticmethod
    def pipeline():
        return _FakePipeline()

    @staticmethod
    def config():
        return _FakePipelineConfig()


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

    def test_unsupported_profiles_report_available_fps_without_probing(self) -> None:
        device = RealSenseDevice(
            serial="1234",
            name="Intel RealSense D405",
            product_line="D400",
            firmware_version="test",
            usb_type="2.1",
            physical_port="test-port",
            depth_profiles=[
                "848x480@5 format.z16",
                "848x480@10 format.z16",
            ],
            color_profiles=[
                "848x480@5 format.bgr8",
                "848x480@10 format.bgr8",
            ],
        )
        with patch(
            "bumblebox_v2.realsense_camera.discover_realsense_devices",
            return_value=(object(), [device]),
        ), patch("bumblebox_v2.realsense_camera._capture_frame_pair") as capture:
            result = run_realsense_check(load_defaults(), probe=True)

        capture.assert_not_called()
        self.assertFalse(result.probe_attempted)
        self.assertFalse(result.probe_succeeded)
        self.assertIn("Available FPS at this resolution/format: 5, 10", result.errors[0])
        self.assertTrue(any("USB 3 port" in warning for warning in result.warnings))

    def test_recording_streams_exact_depth_stack_and_timestamp_metadata(self) -> None:
        config = load_defaults()
        config["realsense"].update(
            {
                "enabled": True,
                "depth_width": 32,
                "depth_height": 24,
                "color_width": 32,
                "color_height": 24,
                "fps": 30,
                "warmup_frames": 1,
                "save_depth": True,
                "save_color": True,
            }
        )
        device = RealSenseDevice(
            serial="1234",
            name="Intel RealSense D405",
            product_line="D400",
            firmware_version="test",
            usb_type="3.2",
            physical_port="test-port",
            depth_profiles=["32x24@30 format.z16"],
            color_profiles=["32x24@30 format.bgr8"],
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "bumblebox_v2.realsense_camera.discover_realsense_devices",
            return_value=(_FakeRealSense(), [device]),
        ):
            with RealSenseRecordingSession(
                config,
                session_dir=tmp,
                session_name="test_run",
            ) as session:
                result = session.capture_for(duration=0.03)

            raw_depth = np.load(result.raw_depth_npy_path)
            timestamp_lines = Path(result.timestamp_path).read_text().splitlines()
            metadata = json.loads(Path(result.metadata_json_path).read_text())

            self.assertGreater(result.frames_captured, 1)
            self.assertEqual(raw_depth.shape, (result.frames_captured, 24, 32))
            self.assertEqual(len(timestamp_lines), result.frames_captured + 1)
            self.assertEqual(metadata["depth_frame_number_gaps"], 0)
            self.assertEqual(metadata["color_frame_number_gaps"], 0)
            self.assertTrue(Path(result.depth_preview_video_path).exists())
            self.assertTrue(Path(result.color_video_path).exists())


if __name__ == "__main__":
    unittest.main()
