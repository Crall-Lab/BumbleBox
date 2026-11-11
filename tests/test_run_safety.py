"""Tests for the protections around unattended recording.

- storage guard: a run refuses to start when the data drive is missing or full
- capture lock: only one run uses the camera at a time
- run timeout: generated systemd units stop a hung run
"""

from __future__ import annotations

import contextlib
import io
import os
from collections import namedtuple
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from bumblebox_v2 import cli, run_engine, storage_guard
from bumblebox_v2.capture_lock import CaptureBusyError, capture_lock
from bumblebox_v2.config import ConfigError, load_defaults, save_config, validate_config
from bumblebox_v2.runtime_alerts import _storage_alert
from bumblebox_v2.storage_guard import (
    StorageNotReadyError,
    check_storage_ready,
    expected_mount_point,
    fstab_mount_points,
)
from bumblebox_v2.systemd_units import run_timeout_seconds, write_systemd_units

Usage = namedtuple("Usage", "total used free")
PLENTY = lambda _path: Usage(500e9, 100e9, 400e9)  # noqa: E731

FSTAB = """
# /etc/fstab: static file system information.
UUID=aaaa-bbbb  /               ext4  defaults,noatime  0  1
UUID=cccc-dddd  /boot/firmware  vfat  defaults          0  2
UUID=1234-ABCD  /mnt/bumblebox/data  ext4  defaults,nofail  0  2
none            swap            swap  sw                0  0
"""


def _config(data_root: str, **system: object) -> dict:
    config = load_defaults()
    config["system"]["data_root"] = data_root
    config["system"].update(system)
    return config


def _mounted_at(*mount_points: str):
    """An ``ismount`` stand-in: only "/" and the given folders are mount points."""
    mounted = {"/", *mount_points}
    return lambda path: path in mounted


class FstabParsingTests(unittest.TestCase):
    def test_comments_swap_and_root_are_ignored(self) -> None:
        self.assertEqual(fstab_mount_points(FSTAB), ["/boot/firmware", "/mnt/bumblebox/data"])

    def test_expected_mount_point_matches_the_folder_or_a_folder_above_it(self) -> None:
        self.assertEqual(expected_mount_point("/mnt/bumblebox/data", FSTAB), "/mnt/bumblebox/data")
        self.assertEqual(expected_mount_point("/mnt/bumblebox/data/2026-01-01", FSTAB), "/mnt/bumblebox/data")
        self.assertIsNone(expected_mount_point("/mnt/bumblebox/database", FSTAB))
        self.assertIsNone(expected_mount_point("/home/pi/data", FSTAB))


class StorageGuardTests(unittest.TestCase):
    def _check(self, config: dict, *, fstab: str = FSTAB, mounted: tuple[str, ...] = (), usage=PLENTY):
        return check_storage_ready(config, fstab_text=fstab, ismount=_mounted_at(*mounted), disk_usage=usage)

    def test_drive_listed_in_fstab_must_be_mounted(self) -> None:
        missing = self._check(_config("/mnt/bumblebox/data"))
        self.assertFalse(missing.ready)
        self.assertEqual(missing.requirement, "expected_by_fstab")
        self.assertIn("not mounted", missing.problem)
        self.assertIn("/mnt/bumblebox/data", missing.problem)

        present = self._check(_config("/mnt/bumblebox/data"), mounted=("/mnt/bumblebox/data",))
        self.assertTrue(present.ready)
        self.assertTrue(present.mounted)
        self.assertEqual(present.free_gb, 400.0)

    def test_folder_under_mnt_needs_a_drive_even_without_an_fstab_entry(self) -> None:
        missing = self._check(_config("/mnt/usbdrive/bees"), fstab="")
        self.assertFalse(missing.ready)
        self.assertEqual(missing.requirement, "expected_by_location")
        self.assertIn("system disk", missing.problem)

        # A drive mounted above the folder is enough.
        present = self._check(_config("/mnt/usbdrive/bees"), fstab="", mounted=("/mnt/usbdrive",))
        self.assertTrue(present.ready)

    def test_ordinary_folder_needs_no_drive_by_default(self) -> None:
        readiness = self._check(_config("/home/pi/bumblebox-data"), fstab="")
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.requirement, "not_required")

    def test_require_true_refuses_the_system_disk_anywhere(self) -> None:
        on_system_disk = self._check(_config("/home/pi/data", require_mounted_storage=True), fstab="")
        self.assertFalse(on_system_disk.ready)
        self.assertEqual(on_system_disk.requirement, "required")

        on_drive = self._check(
            _config("/home/pi/data", require_mounted_storage=True), fstab="", mounted=("/home/pi/data",)
        )
        self.assertTrue(on_drive.ready)

    def test_require_false_allows_the_system_disk_and_says_so(self) -> None:
        readiness = self._check(_config("/mnt/bumblebox/data", require_mounted_storage=False))
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.requirement, "not_required")
        self.assertIn("not mounted", " ".join(readiness.notes))

    def test_low_free_space_refuses_and_zero_disables_the_check(self) -> None:
        nearly_full = lambda _path: Usage(500e9, 499.6e9, 0.4e9)  # noqa: E731
        config = _config("/home/pi/data")
        refused = self._check(config, fstab="", usage=nearly_full)
        self.assertFalse(refused.ready)
        self.assertIn("0.4 GB", refused.problem)
        self.assertIn("min_free_space_gb", refused.problem)

        config["system"]["min_free_space_gb"] = 0
        self.assertTrue(self._check(config, fstab="", usage=nearly_full).ready)

    def test_missing_data_root_setting_is_refused(self) -> None:
        self.assertFalse(self._check(_config("")).ready)


class StorageGuardRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_root = Path(self._tmp.name) / "drive" / "data"
        # fstab says a drive belongs at <tmp>/drive, and nothing is mounted there.
        fstab = f"UUID=1234-ABCD  {Path(self._tmp.name) / 'drive'}  ext4  defaults,nofail  0  2\n"
        patcher = patch.object(storage_guard, "read_fstab_text", lambda: fstab)
        patcher.start()
        self.addCleanup(patcher.stop)
        lock = patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(Path(self._tmp.name) / "capture.lock")})
        lock.start()
        self.addCleanup(lock.stop)

    def _mock_config(self) -> dict:
        config = _config(str(self.data_root))
        config["local_index"]["enabled"] = False
        config["pipeline"]["mode"] = "record_only"
        config["camera"].update({"width": 64, "height": 48, "codec": "mjpeg"})
        config["capture"]["recording_seconds"] = 0.5
        config["runtime"].update({"use_mock_camera": True, "fps_report_on_each_recording": False})
        return config

    def test_run_is_refused_before_anything_is_written(self) -> None:
        with self.assertRaises(StorageNotReadyError) as raised:
            run_engine.run_once(config=self._mock_config())
        self.assertIn("not mounted", str(raised.exception))
        self.assertFalse(self.data_root.exists(), "no folder may be created on the system disk")
        self.assertFalse(self.data_root.parent.exists())

    def test_run_proceeds_when_the_check_is_turned_off(self) -> None:
        config = self._mock_config()
        config["system"]["require_mounted_storage"] = False
        summary = run_engine.run_once(config=config)
        self.assertTrue(summary.success, summary.errors)
        self.assertTrue(Path(summary.video_path).exists())

    def test_run_once_command_reports_the_refusal_and_fails(self) -> None:
        config_path = Path(self._tmp.name) / "config.yaml"
        save_config(config_path, self._mock_config())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = cli.main(["run-once", "--config", str(config_path)])
        self.assertEqual(exit_code, 1)
        self.assertIn("The data drive is not mounted", output.getvalue())
        self.assertFalse(self.data_root.exists())

    def test_storage_alert_fails_when_the_drive_is_missing(self) -> None:
        self.data_root.mkdir(parents=True)  # the folder exists on the system disk, as after an unmount
        alert = _storage_alert(self._mock_config())
        self.assertEqual(alert.severity, "FAIL")
        self.assertIn("not mounted", alert.message)


class CaptureLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.lock_path = Path(self._tmp.name) / "capture.lock"

    def test_second_holder_gives_up_after_the_wait(self) -> None:
        waits: list[str] = []
        with capture_lock(0, lock_path=self.lock_path):
            started = time.monotonic()
            with self.assertRaises(CaptureBusyError) as raised:
                with capture_lock(0.3, lock_path=self.lock_path, on_wait=waits.append, poll_seconds=0.05):
                    self.fail("the lock must not be granted twice")
            waited = time.monotonic() - started
        self.assertGreaterEqual(waited, 0.3)
        self.assertLess(waited, 3.0)
        self.assertEqual(len(waits), 1)
        self.assertIn(f"pid {os.getpid()}", waits[0])
        self.assertIn(f"pid {os.getpid()}", str(raised.exception))

    def test_waiting_holder_proceeds_once_the_lock_is_released(self) -> None:
        release = threading.Event()
        holding = threading.Event()

        def hold() -> None:
            with capture_lock(0, lock_path=self.lock_path):
                holding.set()
                release.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(holding.wait(5))
        threading.Timer(0.2, release.set).start()
        with capture_lock(5, lock_path=self.lock_path, poll_seconds=0.05):
            pass  # granted after the first holder let go
        holder.join(5)

    def test_lock_is_reusable_after_release(self) -> None:
        for _ in range(3):
            with capture_lock(0, lock_path=self.lock_path):
                pass

    def test_lock_file_is_not_followed_through_a_symlink(self) -> None:
        target = Path(self._tmp.name) / "victim.txt"
        target.write_text("do not touch")
        self.lock_path.symlink_to(target)
        with self.assertRaises(OSError):
            with capture_lock(0, lock_path=self.lock_path):
                pass
        self.assertEqual(target.read_text(), "do not touch")


class CaptureLockRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.lock_path = Path(self._tmp.name) / "capture.lock"
        lock = patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(self.lock_path)})
        lock.start()
        self.addCleanup(lock.stop)

    def _mock_config(self) -> dict:
        config = _config(str(Path(self._tmp.name) / "data"))
        config["local_index"]["enabled"] = False
        config["pipeline"]["mode"] = "record_only"
        config["camera"].update({"width": 64, "height": 48, "codec": "mjpeg"})
        config["capture"]["recording_seconds"] = 0.5
        config["runtime"].update(
            {"use_mock_camera": True, "fps_report_on_each_recording": False, "capture_lock_wait_seconds": 0.2}
        )
        return config

    def test_run_is_skipped_while_another_capture_holds_the_camera(self) -> None:
        with capture_lock(0, lock_path=self.lock_path):
            busy = run_engine.run_once(config=self._mock_config())
        self.assertFalse(busy.success)
        self.assertIn("Another BumbleBox capture is using the camera", " ".join(busy.errors))
        self.assertEqual(busy.frames_captured, 0)
        self.assertIsNone(busy.video_path)

        free = run_engine.run_once(config=self._mock_config())
        self.assertTrue(free.success, free.errors)

    def test_lock_is_released_after_capture_even_when_the_run_fails(self) -> None:
        def broken_frame(*_args):
            raise RuntimeError("camera unplugged")

        with patch.object(run_engine, "simulated_rgb_yuv420_frame", broken_frame):
            failed = run_engine.run_once(config=self._mock_config())
        self.assertFalse(failed.success)
        with capture_lock(0, lock_path=self.lock_path):
            pass  # would raise CaptureBusyError if the failed run still held the lock


class SessionFolderTests(unittest.TestCase):
    def test_runs_starting_in_the_same_second_get_separate_folders(self) -> None:
        from datetime import datetime

        class FixedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 1, 2, 3, 4, 5)

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            with patch.object(run_engine, "datetime", FixedClock):
                first_name, first_dir = run_engine._make_session_paths(config)
                second_name, second_dir = run_engine._make_session_paths(config)
                third_name, _third_dir = run_engine._make_session_paths(config)

            self.assertTrue(first_name.endswith("_2026-01-02_03_04_05"))
            self.assertEqual(second_name, f"{first_name}_2")
            self.assertEqual(third_name, f"{first_name}_3")
            self.assertNotEqual(first_dir, second_dir)
            self.assertTrue(first_dir.is_dir() and second_dir.is_dir())
            self.assertEqual(first_dir.parent.name, "2026-01-02")


class RunTimeoutTests(unittest.TestCase):
    def test_auto_allows_recording_time_plus_a_generous_margin(self) -> None:
        config = load_defaults()
        config["capture"]["recording_seconds"] = 20
        # 30-minute schedule: 20 s + two intervals.
        self.assertEqual(run_timeout_seconds(config, 30), 20 + 2 * 30 * 60)
        # Short schedules still get at least 30 minutes.
        self.assertEqual(run_timeout_seconds(config, 5), 20 + 30 * 60)

    def test_configured_minutes_override_auto_for_recording_jobs_only(self) -> None:
        config = load_defaults()
        config["scheduling"]["run_timeout_minutes"] = 90
        self.assertEqual(run_timeout_seconds(config, 30), 90 * 60)
        self.assertEqual(run_timeout_seconds(config, 30, pipeline_job=False), 2 * 30 * 60)

    def test_generated_service_has_a_start_timeout(self) -> None:
        config = load_defaults()
        with tempfile.TemporaryDirectory() as tmp:
            write_systemd_units(config, Path(tmp) / "config.yaml", Path(tmp) / "units")
            service = (Path(tmp) / "units" / "bumblebox-v2-main.service").read_text()
        expected = int(config["capture"]["recording_seconds"]) + 2 * int(config["capture"]["record_interval_minutes"]) * 60
        self.assertIn("Type=oneshot", service)
        self.assertIn(f"TimeoutStartSec={expected}", service)

    def test_generated_timers_do_not_catch_up_after_downtime(self) -> None:
        config = load_defaults()
        config["pipeline"]["mode"] = "mixed_schedule"
        with tempfile.TemporaryDirectory() as tmp:
            write_systemd_units(config, Path(tmp) / "config.yaml", Path(tmp) / "units")
            timers = sorted((Path(tmp) / "units").glob("*.timer"))
            self.assertEqual([timer.name for timer in timers], ["bumblebox-v2-record.timer", "bumblebox-v2-track.timer"])
            for timer in timers:
                text = timer.read_text()
                with self.subTest(timer=timer.name):
                    # A slot missed while the Pi was off is skipped, not run at boot.
                    self.assertIn("Persistent=false", text)
                    self.assertNotIn("Persistent=true", text)
                    self.assertIn("OnCalendar=", text)  # still clock-aligned


class RunSafetyConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = load_defaults()
        validate_config(config)
        self.assertEqual(config["system"]["require_mounted_storage"], "auto")
        self.assertEqual(config["scheduling"]["run_timeout_minutes"], "auto")

    def test_accepted_values(self) -> None:
        for section, key, value in (
            ("system", "require_mounted_storage", True),
            ("system", "require_mounted_storage", False),
            ("system", "require_mounted_storage", "AUTO"),
            ("system", "min_free_space_gb", 0),
            ("runtime", "capture_lock_wait_seconds", 0),
            ("scheduling", "run_timeout_minutes", 45),
        ):
            with self.subTest(key=f"{section}.{key}", value=value):
                config = load_defaults()
                config[section][key] = value
                validate_config(config)

    def test_bad_values_are_rejected(self) -> None:
        for section, key, value in (
            ("system", "require_mounted_storage", "sometimes"),
            ("system", "min_free_space_gb", -1),
            ("system", "min_free_space_gb", "lots"),
            ("runtime", "capture_lock_wait_seconds", -5),
            ("runtime", "capture_lock_wait_seconds", True),
            ("scheduling", "run_timeout_minutes", 0),
            ("scheduling", "run_timeout_minutes", "never"),
        ):
            with self.subTest(key=f"{section}.{key}", value=value):
                config = load_defaults()
                config[section][key] = value
                with self.assertRaises(ConfigError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
