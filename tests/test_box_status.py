from __future__ import annotations

from collections import namedtuple
import copy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from bumblebox_v2 import box_status
from bumblebox_v2.box_status import (
    LEVEL_FAIL,
    LEVEL_OFF,
    LEVEL_OK,
    LEVEL_WARN,
    SCHEDULE_OFF,
    SCHEDULE_ON,
    SCHEDULE_PARTIAL,
    SCHEDULE_UNKNOWN,
    describe_duration,
    describe_moment,
    describe_relative,
    describe_schedule_plan,
    explain_failure,
    format_start_review,
    get_schedule_status,
    installed_units_are_current,
    parse_systemctl_show,
    parse_systemd_timestamp,
    review_schedule_start,
    storage_status,
)
from bumblebox_v2.config import load_defaults
from bumblebox_v2.schedule_check import ScheduleCheckItem, ScheduleCheckReport
from bumblebox_v2.storage_guard import StorageReadiness
from bumblebox_v2.systemd_units import write_systemd_units

Usage = namedtuple("Usage", "total used free")
NOW = datetime(2026, 11, 12, 14, 18, 0)
DRIVE_REASON = (
    "The data drive is not mounted: /etc/fstab lists a drive for /mnt/bumblebox/data, "
    "but nothing is mounted there."
)


def unit_block(name: str, **properties: str) -> str:
    lines = [f"Id={name}"] + [f"{key}={value}" for key, value in properties.items()]
    return "\n".join(lines)


def timer_block(name: str = "bumblebox-v2-main.timer", *, active: bool = True, enabled: bool = True,
                installed: bool = True, next_run: str = "Thu 2026-11-12 14:30:00 CST") -> str:
    return unit_block(
        name,
        LoadState="loaded" if installed else "not-found",
        ActiveState="active" if active else "inactive",
        UnitFileState=("enabled" if enabled else "disabled") if installed else "",
        NextElapseUSecRealtime=next_run if active else "",
    )


def service_block(name: str = "bumblebox-v2-main.service", *, state: str = "inactive", result: str = "success",
                  exit_status: str = "0", started: str = "Thu 2026-11-12 14:00:00 CST") -> str:
    return unit_block(
        name,
        LoadState="loaded",
        ActiveState=state,
        Result=result,
        ExecMainStatus=exit_status,
        ExecMainStartTimestamp=started,
    )


class FakeSystem:
    """Stands in for systemctl, journalctl and loginctl."""

    def __init__(self, show: str = "", *, journal: str = "", linger: str = "yes", show_code=0) -> None:
        self.show = show
        self.journal = journal
        self.linger = linger
        self.show_code = show_code
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str]):
        self.calls.append(list(command))
        if command[0] == "systemctl":
            return self.show_code, self.show
        if command[0] == "journalctl":
            return 0, self.journal
        if command[0] == "loginctl":
            return 0, f"Linger={self.linger}\n"
        return None, "unexpected command"


def schedule_config(mode: str = "record_and_track"):
    config = load_defaults()
    config["scheduling"]["enabled"] = True
    config["scheduling"]["scope"] = "user"
    config["pipeline"]["mode"] = mode
    return config


class TextHelperTests(unittest.TestCase):
    def test_durations_read_naturally(self) -> None:
        self.assertEqual(describe_duration(20), "less than a minute")
        self.assertEqual(describe_duration(12 * 60), "12 min")
        self.assertEqual(describe_duration(2 * 3600 + 5 * 60), "2 h 5 min")
        self.assertEqual(describe_duration(3 * 3600), "3 h")
        self.assertEqual(describe_duration(3 * 86400 + 2 * 3600), "3 days 2 h")
        self.assertEqual(describe_duration(86400), "1 day")

    def test_moments_show_the_date_only_when_it_is_not_today(self) -> None:
        self.assertEqual(describe_moment(datetime(2026, 11, 12, 14, 30), NOW), "14:30")
        self.assertEqual(describe_moment(datetime(2026, 11, 13, 0, 0), NOW), "Fri 13 Nov 00:00")
        self.assertEqual(describe_relative(datetime(2026, 11, 12, 14, 30), NOW), "in 12 min")
        self.assertEqual(describe_relative(datetime(2026, 11, 12, 14, 0), NOW), "18 min ago")

    def test_systemd_timestamps(self) -> None:
        self.assertEqual(parse_systemd_timestamp("Thu 2026-11-12 17:50:00 CST"), datetime(2026, 11, 12, 17, 50))
        self.assertEqual(parse_systemd_timestamp("2026-11-12 17:50:00"), datetime(2026, 11, 12, 17, 50))
        for empty in ("", "n/a", None, "0"):
            self.assertIsNone(parse_systemd_timestamp(empty))

    def test_systemctl_show_is_split_per_unit(self) -> None:
        units = parse_systemctl_show(timer_block() + "\n\n" + service_block())
        self.assertEqual(set(units), {"bumblebox-v2-main.timer", "bumblebox-v2-main.service"})
        self.assertEqual(units["bumblebox-v2-main.timer"]["ActiveState"], "active")
        self.assertEqual(units["bumblebox-v2-main.service"]["Result"], "success")

    def test_plan_sentence_follows_the_mode(self) -> None:
        self.assertEqual(describe_schedule_plan(schedule_config()), "Records 20 s every 30 min, then tracks.")
        self.assertEqual(describe_schedule_plan(schedule_config("record_only")), "Records 20 s every 30 min.")
        self.assertEqual(
            describe_schedule_plan(schedule_config("mixed_schedule")),
            "Records 20 s every 30 min; tracks every 5 min.",
        )


class StorageStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_defaults()

    def status(self, readiness: StorageReadiness, *, free_gb: float = 212.0, mounted_at: str = "/mnt/bumblebox"):
        return storage_status(
            self.config,
            readiness=readiness,
            disk_usage=lambda _path: Usage(0, 0, int(free_gb * 1e9)),
            ismount=lambda path: path in {"/", mounted_at},
        )

    def test_missing_drive_is_a_failure_that_says_recording_is_refused(self) -> None:
        line = self.status(
            StorageReadiness(path="/mnt/bumblebox/data", ready=False, requirement="expected_by_fstab",
                             mounted=False, problem=DRIVE_REASON)
        )
        self.assertEqual(line.level, LEVEL_FAIL)
        self.assertEqual(line.headline, "Data drive not mounted")
        self.assertIn("Recording is refused", line.detail)

    def test_mounted_drive_reports_free_space(self) -> None:
        line = self.status(
            StorageReadiness(path="/mnt/bumblebox/data", ready=True, requirement="expected_by_fstab",
                             mounted=True, free_gb=212.4)
        )
        self.assertEqual((line.level, line.headline, line.detail), (LEVEL_OK, "Drive mounted · 212 GB free", "/mnt/bumblebox/data"))

    def test_nearly_full_drive_is_a_failure(self) -> None:
        line = self.status(
            StorageReadiness(path="/mnt/bumblebox/data", ready=False, requirement="expected_by_fstab",
                             mounted=True, free_gb=0.4, min_free_gb=1.0, problem="Only 0.4 GB is free")
        )
        self.assertEqual(line.level, LEVEL_FAIL)
        self.assertEqual(line.headline, "Drive nearly full · 0.4 GB free")

    def test_low_space_is_a_warning_before_it_is_a_failure(self) -> None:
        line = self.status(
            StorageReadiness(path="/mnt/bumblebox/data", ready=True, requirement="expected_by_fstab",
                             mounted=True, free_gb=6.0)
        )
        self.assertEqual(line.level, LEVEL_WARN)
        self.assertIn("6.0 GB free", line.headline)

    def test_system_disk_is_named_when_no_drive_is_expected(self) -> None:
        readiness = StorageReadiness(path="/home/pi/data", ready=True, requirement="not_required")
        line = self.status(readiness, free_gb=40.0)
        self.assertEqual((line.level, line.headline), (LEVEL_OK, "On the system disk · 40 GB free"))

    def test_free_space_is_looked_up_when_the_guard_did_not_measure_it(self) -> None:
        readiness = StorageReadiness(path="/mnt/bumblebox/data", ready=True, requirement="required", mounted=True)
        self.assertEqual(self.status(readiness, free_gb=88.0).headline, "Drive mounted · 88 GB free")

    def test_unset_data_folder(self) -> None:
        line = self.status(StorageReadiness(path="", ready=False, requirement="not_required", problem="not set"))
        self.assertEqual((line.level, line.headline), (LEVEL_FAIL, "No data folder set"))


class ExplainFailureTests(unittest.TestCase):
    def test_refused_run_names_the_missing_drive(self) -> None:
        explanation = explain_failure(f"Run failed to start: {DRIVE_REASON}\n", exit_code=1, command=["run-once"])
        self.assertEqual(explanation.title, "Data drive not mounted")
        self.assertEqual(explanation.reason, DRIVE_REASON)

    def test_busy_camera_and_full_drive_get_their_own_titles(self) -> None:
        busy = explain_failure(
            "Run failed to start: Another BumbleBox capture is using the camera (pid 4312, since "
            "2026-11-11T14:00:00) and did not finish within 60s, so this run was skipped.",
            command=["run-once"],
        )
        self.assertEqual(busy.title, "Camera is busy")
        full = explain_failure(
            "Run failed to start: Only 0.4 GB is free where /mnt/bumblebox/data is stored; at least 1 GB "
            "is required to start a recording (system.min_free_space_gb).",
            command=["run-once"],
        )
        self.assertEqual(full.title, "Data drive is nearly full")

    def test_run_summary_errors_are_used_when_the_run_started(self) -> None:
        output = "Run mode: record_and_track\nErrors: 2\nSuccess: False\n\nError details:\n- Tracking failed: boom\n- Metrics skipped\n"
        explanation = explain_failure(output, exit_code=1, command=["run-once"])
        self.assertEqual(explanation.title, "Recording failed")
        self.assertTrue(explanation.reason.startswith("Tracking failed: boom"))
        self.assertIn("1 more", explanation.reason)

    def test_summary_lines_that_mention_failure_are_not_mistaken_for_the_reason(self) -> None:
        output = (
            "Camera autofocus preflight final state: failed\nErrors: 1\nSuccess: False\n\n"
            "Error details:\n- No camera detected by picamera2/libcamera. Check ribbon cable.\n"
        )
        explanation = explain_failure(output, exit_code=1, command=["run-once"])
        self.assertEqual(explanation.title, "Camera not available")
        self.assertTrue(explanation.reason.startswith("No camera detected"))

    def test_check_reports_list_their_failed_checks(self) -> None:
        explanation = explain_failure(
            "[PASS] Python\n[FAIL] Camera: none found\n[WARN] Clock\n[FAIL] ffmpeg: missing\n",
            exit_code=1,
            command=["doctor"],
        )
        self.assertEqual(explanation.title, "Health check found problems")
        self.assertEqual(explanation.reason, "Camera: none found\nffmpeg: missing")

    def test_traceback_reports_its_last_line(self) -> None:
        output = 'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nKeyError: \'camera\'\n'
        explanation = explain_failure(output, exit_code=1, command=["camera-check"])
        self.assertEqual(explanation.title, "Camera check failed")
        self.assertIn("KeyError: 'camera'", explanation.reason)

    def test_failed_to_lines_are_kept_whole(self) -> None:
        output = "Action: install\nSuccess: False\n\nCommand output:\n$ systemctl --user daemon-reload\nFailed to connect to bus: No medium found\n"
        explanation = explain_failure(output, exit_code=1, command=["systemd-install"])
        self.assertEqual(explanation.title, "Automated recording was not started")
        self.assertEqual(explanation.reason, "Failed to connect to bus: No medium found")

    def test_silent_failure_still_says_something(self) -> None:
        explanation = explain_failure("", exit_code=3, command=["thermal-check"])
        self.assertEqual(explanation.title, "Thermal camera check failed")
        self.assertIn("exit code 3", explanation.reason)


class ScheduleStatusTests(unittest.TestCase):
    def status(self, system: FakeSystem, config=None, **kwargs):
        return get_schedule_status(config or schedule_config(), run=system, now=NOW, **kwargs)

    def test_running_schedule_reports_the_next_recording_and_the_last_run(self) -> None:
        system = FakeSystem(timer_block() + "\n\n" + service_block())
        status = self.status(system)
        self.assertEqual(status.state, SCHEDULE_ON)
        self.assertEqual(status.level, LEVEL_OK)
        self.assertEqual(status.headline, "On · next recording 14:30 (in 12 min)")
        self.assertEqual(status.next_run, datetime(2026, 11, 12, 14, 30))
        self.assertIn("Records 20 s every 30 min, then tracks.", status.detail)
        self.assertIn("Last scheduled run (14:00) completed.", status.detail)
        self.assertEqual(system.calls[0][:3], ["systemctl", "--user", "show"])

    def test_system_scope_does_not_ask_the_user_manager(self) -> None:
        config = schedule_config()
        config["scheduling"]["scope"] = "system"
        system = FakeSystem(timer_block() + "\n\n" + service_block(), linger="no")
        status = self.status(system, config)
        self.assertEqual(system.calls[0][:2], ["systemctl", "show"])
        self.assertEqual(status.level, LEVEL_OK)
        self.assertFalse(any(call[0] == "loginctl" for call in system.calls))

    def test_schedule_that_is_not_installed_is_off(self) -> None:
        config = schedule_config()
        config["scheduling"]["enabled"] = False
        system = FakeSystem(timer_block(installed=False, active=False) + "\n\n" + service_block(started=""))
        status = self.status(system, config)
        self.assertEqual((status.state, status.level, status.headline), (SCHEDULE_OFF, LEVEL_OFF, "Off"))
        self.assertIn("No recordings are scheduled.", status.detail)

    def test_off_schedule_warns_when_the_settings_say_on(self) -> None:
        system = FakeSystem(timer_block(installed=False, active=False) + "\n\n" + service_block(started=""))
        status = self.status(system)
        self.assertEqual((status.state, status.level), (SCHEDULE_OFF, LEVEL_WARN))
        self.assertIn("should be on", status.detail)

    def test_failed_run_carries_the_reason_from_the_journal(self) -> None:
        journal = "\n".join(
            [
                "Starting bumblebox-v2-main.service - BumbleBox V2 main job...",
                f"Run failed to start: {DRIVE_REASON}",
                "bumblebox-v2-main.service: Main process exited, code=exited, status=1/FAILURE",
                "bumblebox-v2-main.service: Failed with result 'exit-code'.",
                "Failed to start bumblebox-v2-main.service - BumbleBox V2 main job.",
            ]
        )
        system = FakeSystem(
            timer_block() + "\n\n" + service_block(state="failed", result="exit-code", exit_status="1"),
            journal=journal,
        )
        status = self.status(system)
        self.assertEqual(status.state, SCHEDULE_ON)
        self.assertEqual(status.level, LEVEL_FAIL)
        self.assertIn("Last scheduled run (14:00) FAILED: Data drive not mounted.", status.detail)
        self.assertIn(DRIVE_REASON, status.detail)
        journal_call = next(call for call in system.calls if call[0] == "journalctl")
        self.assertIn("2026-11-12 14:00:00", journal_call)

    def test_failed_run_without_journal_points_at_the_log(self) -> None:
        system = FakeSystem(timer_block() + "\n\n" + service_block(state="failed", result="exit-code", exit_status="1"))
        status = self.status(system)
        self.assertEqual(status.level, LEVEL_FAIL)
        self.assertIn("exit code 1", status.detail)
        self.assertIn("journalctl --user -u bumblebox-v2-main.service", status.detail)

    def test_run_stopped_at_the_time_limit(self) -> None:
        system = FakeSystem(timer_block() + "\n\n" + service_block(state="failed", result="timeout", exit_status="0"))
        status = self.status(system)
        self.assertEqual(status.level, LEVEL_FAIL)
        self.assertIn("Stopped at the time limit", status.detail)
        self.assertFalse(any(call[0] == "journalctl" for call in system.calls))

    def test_run_in_progress_is_reported_without_a_verdict(self) -> None:
        system = FakeSystem(timer_block() + "\n\n" + service_block(state="activating"))
        status = self.status(system)
        self.assertEqual(status.level, LEVEL_OK)
        self.assertIn("A scheduled run is in progress (started 14:00).", status.detail)
        self.assertNotIn("completed", status.detail)

    def test_no_run_yet(self) -> None:
        system = FakeSystem(timer_block() + "\n\n" + service_block(started=""))
        self.assertIn("No scheduled run has happened yet", self.status(system).detail)

    def test_timer_that_will_not_survive_a_restart_is_a_warning(self) -> None:
        system = FakeSystem(timer_block(enabled=False) + "\n\n" + service_block())
        status = self.status(system)
        self.assertEqual((status.state, status.level), (SCHEDULE_ON, LEVEL_WARN))
        self.assertIn("will not come back after a restart", status.detail)

    def test_user_schedule_without_linger_is_a_warning(self) -> None:
        system = FakeSystem(timer_block() + "\n\n" + service_block(), linger="no")
        status = self.status(system)
        self.assertEqual(status.level, LEVEL_WARN)
        self.assertIn("loginctl enable-linger", status.detail)

    def test_mixed_schedule_headline_names_the_recording_not_the_tracking_run(self) -> None:
        show = "\n\n".join(
            [
                timer_block("bumblebox-v2-record.timer", next_run="Thu 2026-11-12 14:30:00 CST"),
                service_block("bumblebox-v2-record.service"),
                timer_block("bumblebox-v2-track.timer", next_run="Thu 2026-11-12 14:20:00 CST"),
                service_block("bumblebox-v2-track.service", started="Thu 2026-11-12 14:15:00 CST"),
            ]
        )
        status = self.status(FakeSystem(show), schedule_config("mixed_schedule"))
        self.assertEqual(status.headline, "On · next recording 14:30 (in 12 min)")

    def test_half_running_schedule_is_a_failure(self) -> None:
        show = "\n\n".join(
            [
                timer_block("bumblebox-v2-record.timer"),
                service_block("bumblebox-v2-record.service"),
                timer_block("bumblebox-v2-track.timer", active=False),
                service_block("bumblebox-v2-track.service", started=""),
            ]
        )
        status = self.status(FakeSystem(show), schedule_config("mixed_schedule"))
        self.assertEqual((status.state, status.level), (SCHEDULE_PARTIAL, LEVEL_FAIL))
        self.assertIn("Not running: track", status.detail)

    def test_unanswered_question_is_unknown_not_off(self) -> None:
        system = FakeSystem("Failed to connect to bus: No medium found\n", show_code=1)
        status = self.status(system)
        self.assertEqual((status.state, status.level, status.headline), (SCHEDULE_UNKNOWN, LEVEL_WARN, "Unknown"))
        self.assertIn("Failed to connect to bus", status.detail)

        missing = get_schedule_status(schedule_config(), run=lambda command: (None, "systemctl is not installed"), now=NOW)
        self.assertEqual(missing.state, SCHEDULE_UNKNOWN)
        self.assertIn("not installed", missing.detail)

    def test_changed_settings_ask_for_the_schedule_to_be_started_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory)
            config_path = installed / "config.yaml"
            config = schedule_config()
            write_systemd_units(config=config, config_path=config_path, output_dir=installed)
            self.assertTrue(installed_units_are_current(config, config_path, installed_dir=installed))

            system = FakeSystem(timer_block() + "\n\n" + service_block())
            current = self.status(system, config, config_path=config_path, installed_dir=installed)
            self.assertEqual(current.level, LEVEL_OK)
            self.assertTrue(current.units_current)

            changed = copy.deepcopy(config)
            changed["capture"]["record_interval_minutes"] = 10
            self.assertFalse(installed_units_are_current(changed, config_path, installed_dir=installed))
            stale = self.status(system, changed, config_path=config_path, installed_dir=installed)
            self.assertEqual(stale.level, LEVEL_WARN)
            self.assertIn("settings have changed since the schedule was started", stale.detail)


class StartReviewTests(unittest.TestCase):
    READY = StorageReadiness(path="/mnt/bumblebox/data", ready=True, requirement="required", mounted=True, free_gb=200.0)

    @staticmethod
    def report(*items: ScheduleCheckItem, suggestions=()):
        return ScheduleCheckReport(items=list(items), suggestions=list(suggestions), metrics={}, benchmark_used=False)

    def test_workable_schedule_on_a_ready_drive_has_no_problems(self) -> None:
        review = review_schedule_start(
            load_defaults(),
            readiness=self.READY,
            schedule_check=lambda config: self.report(ScheduleCheckItem("Memory budget", "PASS", "fine")),
        )
        self.assertTrue(review.ok)

    def test_failed_checks_become_problems_with_suggestions(self) -> None:
        review = review_schedule_start(
            load_defaults(),
            readiness=self.READY,
            schedule_check=lambda config: self.report(
                ScheduleCheckItem("Memory budget", "FAIL", "Estimated frame buffer is 93 GiB."),
                ScheduleCheckItem("Frame buffer", "WARN", "heads up"),
                suggestions=["Set capture.frame_buffer to 'stream'.", "For higher confidence, rerun schedule-check with --benchmark-input."],
            ),
        )
        self.assertFalse(review.ok)
        self.assertEqual(review.problems, ["Memory budget: Estimated frame buffer is 93 GiB."])
        self.assertEqual(review.suggestions, ["Set capture.frame_buffer to 'stream'."])
        text = format_start_review(review)
        self.assertIn("• Memory budget", text)
        self.assertIn("What would help:", text)

    def test_real_schedule_check_flags_a_recording_too_long_for_memory(self) -> None:
        config = load_defaults()
        config["capture"]["recording_seconds"] = 3600
        review = review_schedule_start(config, readiness=self.READY)
        self.assertFalse(review.ok)
        self.assertTrue(any(problem.startswith("Memory budget") for problem in review.problems))

    def test_missing_drive_is_a_problem(self) -> None:
        missing = StorageReadiness(path="/mnt/bumblebox/data", ready=False, requirement="expected_by_fstab",
                                   mounted=False, problem=DRIVE_REASON)
        review = review_schedule_start(
            load_defaults(), readiness=missing, schedule_check=lambda config: self.report()
        )
        self.assertFalse(review.ok)
        self.assertTrue(review.problems[0].startswith("Data drive not mounted"))

    def test_a_check_that_crashes_is_reported_not_raised(self) -> None:
        def broken(config):
            raise KeyError("camera")

        review = review_schedule_start(load_defaults(), readiness=self.READY, schedule_check=broken)
        self.assertFalse(review.ok)
        self.assertIn("could not be checked", review.problems[0])


def benchmark_report(*, width=4056, height=3040, mono=False, source="video: /data/a.mp4", results=None):
    return {
        "width": width,
        "height": height,
        "frames": 24,
        "fps_target": 5.0,
        "monochrome_output": mono,
        "source": source,
        "work_dir": "/tmp",
        "results": results
        if results is not None
        else [
            {"label": "mjpeg", "codec": "mjpeg", "mp4_codec": None, "mp4_preset": None,
             "frames_per_second": 9.8, "megabytes_per_frame": 2.9, "error": None},
            {"label": "libx264:ultrafast", "codec": "mp4", "mp4_codec": "libx264", "mp4_preset": "ultrafast",
             "frames_per_second": 7.1, "megabytes_per_frame": 1.9, "error": None},
            {"label": "libx264:fast", "codec": "mp4", "mp4_codec": "libx264", "mp4_preset": "fast",
             "frames_per_second": 1.4, "megabytes_per_frame": 1.1, "error": None},
        ],
    }


class StorageEstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_defaults()  # 20 s at 5 fps every 30 min: 100 frames, 48 recordings a day
        self.signature = box_status.recording_signature(self.config)

    def run_record(self, *, size=150e6, frames=100, success=True, mode="record_and_track", signature=None):
        return {
            "success": success,
            "mode": mode,
            "frames_captured": frames,
            "_session_bytes": size,
            "_recording_signature": signature if signature is not None else self.signature,
        }

    def test_recordings_per_day_follows_the_timer_slots(self) -> None:
        self.assertEqual(box_status.recordings_per_day(self.config), 48)
        self.config["capture"]["record_interval_minutes"] = 7  # 0, 7, ..., 1435: 206 slots
        self.assertEqual(box_status.recordings_per_day(self.config), 206)
        self.config["pipeline"]["mode"] = "track_only"
        self.assertEqual(box_status.recordings_per_day(self.config), 0)

    def test_estimate_from_matching_recordings(self) -> None:
        estimate = box_status.estimate_storage(self.config, [self.run_record(), self.run_record(size=250e6)], free_gb=212)
        # 2 MB per frame x 100 frames x 48 recordings = 9.6 GB a day; (212 - 1) GB / 9.6 GB = 22 days
        self.assertAlmostEqual(estimate.bytes_per_day, 9.6e9)
        self.assertAlmostEqual(estimate.days_left, 211 / 9.6)
        self.assertEqual(estimate.level, LEVEL_OK)
        self.assertEqual(estimate.headline, "About 9.6 GB per day · about 22 days of space left")
        self.assertIn("200 MB per recording, 48 recordings a day", estimate.detail)
        self.assertIn("last 2 recordings", estimate.detail)

    def test_length_and_rate_scale_the_estimate(self) -> None:
        self.config["capture"]["recording_seconds"] = 40
        self.config["camera"]["fps_target"] = 10.0
        estimate = box_status.estimate_storage(self.config, [self.run_record()])
        self.assertAlmostEqual(estimate.bytes_per_recording, 1.5e6 * 400)
        self.assertIsNone(estimate.days_left)

    def test_recordings_with_other_video_settings_failures_and_tracking_runs_are_ignored(self) -> None:
        other = ("mjpeg", "-", False, 4056, 3040)
        runs = [
            self.run_record(signature=other),
            self.run_record(success=False),
            self.run_record(mode="track_only"),
            self.run_record(frames=0),
            {"success": True, "mode": "record_only", "frames_captured": 100},
        ]
        estimate = box_status.estimate_storage(self.config, runs, free_gb=212)
        self.assertEqual((estimate.level, estimate.headline), (LEVEL_OFF, "Storage per day not known yet"))
        self.assertIn("4056 × 3040, MP4 (fast), colour", estimate.detail)

    def test_little_space_left_is_a_warning(self) -> None:
        estimate = box_status.estimate_storage(self.config, [self.run_record()], free_gb=30)
        self.assertEqual(estimate.level, LEVEL_WARN)
        self.assertIn("about 4 days of space left", estimate.headline)

    def test_reserve_is_subtracted_from_free_space(self) -> None:
        self.config["system"]["min_free_space_gb"] = 50
        estimate = box_status.estimate_storage(self.config, [self.run_record()], free_gb=40)
        self.assertEqual(estimate.days_left, 0)
        self.assertIn("less than 2 hours", estimate.headline)

    def test_track_only_records_nothing(self) -> None:
        self.config["pipeline"]["mode"] = "track_only"
        estimate = box_status.estimate_storage(self.config, [self.run_record()], free_gb=212)
        self.assertEqual((estimate.level, estimate.headline), (LEVEL_OFF, "No new video"))

    def test_benchmark_on_real_footage_fills_in_when_no_recording_matches(self) -> None:
        self.config["capture"]["frame_buffer"] = "stream"  # auto preset becomes ultrafast
        estimate = box_status.estimate_storage(self.config, [self.run_record()], free_gb=212, benchmark=benchmark_report())
        self.assertAlmostEqual(estimate.bytes_per_recording, 1.9e6 * 100)
        self.assertIn("Video only, from the encode benchmark", estimate.detail)

        synthetic = benchmark_report(source="synthetic nest-like frames")
        unknown = box_status.estimate_storage(self.config, [], benchmark=synthetic)
        self.assertEqual(unknown.headline, "Storage per day not known yet")

        other_size = benchmark_report(width=2028, height=1520)
        self.assertEqual(box_status.estimate_storage(self.config, [], benchmark=other_size).level, LEVEL_OFF)

    def test_matching_recordings_win_over_the_benchmark(self) -> None:
        estimate = box_status.estimate_storage(self.config, [self.run_record()], benchmark=benchmark_report())
        self.assertIn("Measured from the last recording", estimate.detail)

    def test_text_helpers(self) -> None:
        self.assertEqual(box_status.format_bytes(7.25e9), "7.2 GB")
        self.assertEqual(box_status.format_bytes(274e9), "274 GB")
        self.assertEqual(box_status.format_bytes(512), "512 B")
        self.assertEqual(box_status.describe_days(0.8), "about 19 hours")
        self.assertEqual(box_status.describe_days(29.4), "about 29 days")
        self.assertEqual(box_status.describe_days(120), "about 4 months")
        self.assertEqual(box_status.describe_days(1000), "about 3 years")


class BenchmarkTests(unittest.TestCase):
    def test_selected_encoder_is_found_in_the_report(self) -> None:
        config = load_defaults()
        self.assertEqual(box_status.benchmark_result_for(config, benchmark_report())["mp4_preset"], "fast")
        config["camera"]["codec"] = "mjpeg"
        self.assertEqual(box_status.benchmark_result_for(config, benchmark_report())["codec"], "mjpeg")
        config["camera"]["codec"] = "mp4"
        config["camera"]["mp4_preset"] = "veryfast"
        self.assertIsNone(box_status.benchmark_result_for(config, benchmark_report()))
        config["camera"]["mp4_preset"] = "auto"
        config["camera"]["monochrome_output"] = True
        self.assertIsNone(box_status.benchmark_result_for(config, benchmark_report()))

    def test_failed_encoders_are_skipped(self) -> None:
        report = benchmark_report(results=[
            {"label": "libx264:fast", "codec": "mp4", "mp4_codec": "libx264", "mp4_preset": "fast",
             "frames_per_second": 0, "megabytes_per_frame": 0, "error": "ffmpeg not found"},
        ])
        self.assertIsNone(box_status.benchmark_result_for(load_defaults(), report))

    def test_keeps_up_uses_the_benchmark_headroom(self) -> None:
        self.assertEqual(box_status.keeps_up(7.5, 5), "yes")
        self.assertEqual(box_status.keeps_up(7.1, 5), "marginal")
        self.assertEqual(box_status.keeps_up(4.9, 5), "no")
        self.assertEqual(box_status.keeps_up(None, 5), "n/a")

    def test_saved_report_round_trip(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.json"
            self.assertIsNone(box_status.load_benchmark(path))
            path.write_text("not json")
            self.assertIsNone(box_status.load_benchmark(path))
            path.write_text(json.dumps(benchmark_report()))
            self.assertEqual(box_status.load_benchmark(path)["width"], 4056)


class SessionFilesTests(unittest.TestCase):
    def test_session_size_counts_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.mp4").write_bytes(b"x" * 1000)
            (root / "depth").mkdir()
            (root / "depth" / "b.npy").write_bytes(b"x" * 500)
            self.assertEqual(box_status.session_size_bytes(root), 1500)
            self.assertIsNone(box_status.session_size_bytes(root / "missing"))
            self.assertIsNone(box_status.session_size_bytes(root, max_files=1))

    def test_signature_is_read_from_the_config_snapshot(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snap.json"
            config = load_defaults()
            config["camera"]["monochrome_output"] = True
            snapshot.write_text(json.dumps(config))
            self.assertEqual(
                box_status.signature_from_snapshot(snapshot), ("mp4", "fast", True, 4056, 3040)
            )
            self.assertIsNone(box_status.signature_from_snapshot(Path(directory) / "missing.json"))
            self.assertIsNone(box_status.signature_from_snapshot(""))


class CollectTests(unittest.TestCase):
    def test_collect_never_raises(self) -> None:
        status = box_status.collect_box_status({})
        self.assertIn(status.storage.level, {LEVEL_FAIL, LEVEL_WARN})
        self.assertIn(status.schedule.level, {LEVEL_OFF, LEVEL_WARN, LEVEL_OK, LEVEL_FAIL})


if __name__ == "__main__":
    unittest.main()
