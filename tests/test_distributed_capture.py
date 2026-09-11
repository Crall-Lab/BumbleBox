from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from bumblebox_v2.config import ConfigError, load_defaults, save_config, validate_config
from bumblebox_v2.distributed_capture import (
    ClockEstimate,
    check_distributed_capture,
    create_capture_plan,
    decode_capture_plan,
    encode_capture_plan,
    execute_capture_node,
    run_distributed_capture,
)


def distributed_mock_config(root: Path) -> dict:
    config = load_defaults()
    config["system"]["data_root"] = str(root)
    config["local_index"]["enabled"] = False
    config["runtime"]["use_mock_camera"] = True
    config["runtime"]["fps_report_on_each_recording"] = False
    config["runtime"]["render_tracking_video"] = False
    config["capture"]["recording_seconds"] = 0.2
    config["camera"].update(width=160, height=120, fps_target=5, codec="mjpeg")
    config["thermal"].update(enabled=True, width=32, height=24)
    config["realsense"]["enabled"] = False
    config["distributed_capture"].update(
        enabled=True,
        role="controller",
        controller_node="primary",
        start_lead_seconds=0.8,
        ready_timeout_seconds=5.0,
        require_same_revision=False,
        nodes=[
            {
                "name": "primary",
                "host": "localhost",
                "local": True,
                "enabled": True,
                "sensors": ["rgb"],
                "user": None,
                "port": 22,
                "repo_path": None,
                "config_path": None,
                "data_root": None,
            },
            {
                "name": "worker",
                "host": "worker.local",
                "local": False,
                "enabled": True,
                "sensors": ["thermal"],
                "user": "pi",
                "port": 22,
                "repo_path": "/home/pi/Desktop/BumbleBox",
                "config_path": "/home/pi/Desktop/BumbleBox/bumblebox_v2/config.yaml",
                "data_root": str(root),
            },
        ],
    )
    return config


class DistributedCaptureTests(unittest.TestCase):
    def test_plan_round_trip_preserves_changeable_sensor_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = distributed_mock_config(Path(directory))
            validate_config(config)
            plan = create_capture_plan(
                config,
                config_path=Path(directory) / "config.yaml",
                mode="record_only",
                start_unix_ns=time.time_ns() + 1_000_000_000,
            )
            decoded = decode_capture_plan(encode_capture_plan(plan))
            owners = {
                sensor: node["name"]
                for node in decoded["nodes"]
                for sensor in node["sensors"]
            }
            self.assertEqual(owners["rgb"], "primary")
            self.assertEqual(owners["thermal"], "worker")
            self.assertNotIn(
                "device_path", decoded["effective_config_sections"]["thermal"]
            )

    def test_capture_plan_rejects_track_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = distributed_mock_config(Path(directory))
            with self.assertRaises(ValueError):
                create_capture_plan(
                    config,
                    config_path=Path(directory) / "config.yaml",
                    mode="track_only",
                )

    def test_duplicate_sensor_assignment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = distributed_mock_config(Path(directory))
            config["distributed_capture"]["nodes"][0]["sensors"].append("thermal")
            with self.assertRaises(ConfigError):
                validate_config(config)

    def test_worker_role_can_load_without_becoming_a_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = distributed_mock_config(Path(directory))
            config["distributed_capture"]["role"] = "worker"
            config["distributed_capture"]["nodes"] = []
            validate_config(config)

    def test_sensor_only_worker_writes_provenance_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = distributed_mock_config(Path(directory))
            plan = create_capture_plan(
                config,
                config_path=Path(directory) / "config.yaml",
                mode="record_only",
                start_unix_ns=time.time_ns() + 300_000_000,
            )
            result = execute_capture_node(
                config,
                plan=plan,
                node_name="worker",
                emit_protocol_markers=False,
            )
            self.assertTrue(result["success"])
            thermal = result["artifacts"]["thermal"]
            with Path(thermal["timestamp_path"]).open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["node"], "worker")
            self.assertEqual(row["sensor"], "thermal")
            self.assertTrue(row["captured_unix_s"])

    def test_local_controller_run_writes_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = distributed_mock_config(root / "data")
            config["distributed_capture"]["nodes"] = [
                {
                    "name": "primary",
                    "host": "localhost",
                    "local": True,
                    "enabled": True,
                    "sensors": ["rgb", "thermal"],
                    "user": None,
                    "port": 22,
                    "repo_path": None,
                    "config_path": None,
                    "data_root": None,
                }
            ]
            config_path = root / "config.yaml"
            save_config(config_path, config)
            manifest = run_distributed_capture(
                config,
                config_path=config_path,
                mode="record_only",
            )
            self.assertTrue(manifest["success"])
            self.assertTrue(manifest["capture_success"])
            self.assertTrue(manifest["collection_success"])
            summary = Path(manifest["summary_path"])
            self.assertTrue(summary.is_file())
            self.assertTrue((summary.parent / "distributed_capture_manifest.json").is_file())

    def test_unarmed_node_leaves_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = distributed_mock_config(root / "data")
            config["distributed_capture"].update(
                start_lead_seconds=0.3,
                ready_timeout_seconds=1.0,
                nodes=[
                    {
                        "name": "primary",
                        "host": "localhost",
                        "local": True,
                        "enabled": True,
                        "sensors": ["rgb", "thermal"],
                        "user": None,
                        "port": 22,
                        "repo_path": None,
                        "config_path": None,
                        "data_root": None,
                    }
                ],
            )
            config_path = root / "config.yaml"
            save_config(config_path, config)
            stalled_command = [sys.executable, "-c", "import time; time.sleep(5)"]
            with patch(
                "bumblebox_v2.distributed_capture._capture_command",
                return_value=(stalled_command, root),
            ):
                manifest = run_distributed_capture(
                    config,
                    config_path=config_path,
                    mode="record_only",
                )
            self.assertFalse(manifest["success"])
            self.assertFalse(manifest["capture_success"])
            self.assertTrue(manifest["collection_success"])
            self.assertIn("did not arm", manifest["errors"][0])
            self.assertTrue(Path(manifest["manifest_path"]).is_file())
            self.assertTrue(Path(manifest["summary_path"]).is_file())

    def test_unlocked_network_clock_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = distributed_mock_config(root / "data")
            config["distributed_capture"]["nodes"] = [
                {
                    "name": "primary",
                    "host": "localhost",
                    "local": True,
                    "enabled": True,
                    "sensors": ["rgb", "thermal"],
                    "user": None,
                    "port": 22,
                    "repo_path": None,
                    "config_path": None,
                    "data_root": None,
                }
            ]
            config_path = root / "config.yaml"
            config_path.write_text("{}\n")
            unlocked = ClockEstimate(0.0, 0.0, 1, False, "test")
            with patch(
                "bumblebox_v2.distributed_capture.estimate_node_clock",
                return_value=unlocked,
            ):
                report = check_distributed_capture(
                    config,
                    config_path=config_path,
                )
            self.assertFalse(report.success)
            self.assertIn("not locked", report.nodes[0].errors[0])


if __name__ == "__main__":
    unittest.main()
