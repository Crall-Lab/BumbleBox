"""Time of day for the schedule, and lengths and intervals in people's units."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from bumblebox_v2 import time_units
from bumblebox_v2.box_status import describe_schedule_plan, installed_units_are_current
from bumblebox_v2.config import ConfigError, load_defaults, validate_config
from bumblebox_v2.multi_camera import copy_schedule, schedules_match
from bumblebox_v2.systemd_units import _calendar_specs_for_interval, write_systemd_units
from bumblebox_v2.time_units import DURATION_UNITS, INTERVAL_UNITS


class TimeOfDayTests(unittest.TestCase):
    def test_parse_and_format(self) -> None:
        self.assertEqual(time_units.parse_time_of_day("00:00"), 0)
        self.assertEqual(time_units.parse_time_of_day("10:30"), 630)
        self.assertEqual(time_units.parse_time_of_day(" 9:05 "), 545)
        self.assertEqual(time_units.parse_time_of_day("23:59"), 1439)
        self.assertEqual(time_units.format_time_of_day(630), "10:30")
        self.assertEqual(time_units.format_time_of_day(1440 + 5), "00:05")

    def test_things_that_are_not_times_are_refused(self) -> None:
        for bad in ("24:00", "10:60", "9", "0930", "ten", "", None, 9.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                time_units.parse_time_of_day(bad)

    def test_config_start_time(self) -> None:
        config = load_defaults()
        self.assertEqual(time_units.start_minutes(config), 0)
        config["scheduling"]["start_time"] = "10:00"
        validate_config(config)
        self.assertEqual(time_units.start_minutes(config), 600)
        config["scheduling"]["start_time"] = "25:00"
        self.assertEqual(time_units.start_minutes(config), 0)  # unreadable counts as midnight when read
        with self.assertRaises(ConfigError) as raised:
            validate_config(config)
        self.assertIn("scheduling.start_time", str(raised.exception))


class UnitTests(unittest.TestCase):
    def test_largest_unit_that_stays_a_whole_number(self) -> None:
        cases = ((20, (20, "s")), (90, (90, "s")), (600, (10, "min")), (5400, (90, "min")), (7200, (2, "h")), (0.5, (0.5, "s")))
        for seconds, expected in cases:
            self.assertEqual(time_units.best_unit(seconds, DURATION_UNITS), expected, seconds)
        for minutes, expected in ((30, (30, "min")), (90, (90, "min")), (120, (2, "h")), (1440, (1, "day")), (720, (12, "h"))):
            self.assertEqual(time_units.best_unit(minutes, INTERVAL_UNITS), expected, minutes)

    def test_number_and_unit_back_to_the_stored_value(self) -> None:
        self.assertEqual(time_units.to_base(2, "h", DURATION_UNITS), 7200)
        self.assertEqual(time_units.to_base(1.5, "min", DURATION_UNITS), 90)
        self.assertEqual(time_units.to_base(1, "day", INTERVAL_UNITS), 1440)
        self.assertEqual(time_units.to_base(0.5, "day", INTERVAL_UNITS), 720)

    def test_lengths_and_intervals_in_words(self) -> None:
        for seconds, text in ((20, "20 s"), (0.5, "0.5 s"), (90, "90 s"), (600, "10 min"), (7200, "2 h"), (5400, "1 h 30 min")):
            self.assertEqual(time_units.describe_seconds(seconds), text)
        self.assertEqual(time_units.describe_interval(30), "every 30 min")
        self.assertEqual(time_units.describe_interval(120), "every 2 h")
        self.assertEqual(time_units.describe_interval(90), "every 90 min")
        self.assertEqual(time_units.describe_interval(30, 10), "every 30 min, counted from 00:10")
        self.assertEqual(time_units.describe_interval(1440), "once a day at 00:00")
        self.assertEqual(time_units.describe_interval(1440, 600), "once a day at 10:00")


class TimerSlotTests(unittest.TestCase):
    def test_default_slots_are_unchanged(self) -> None:
        self.assertEqual(_calendar_specs_for_interval(30), ["*-*-* *:00,30:00"])
        self.assertEqual(_calendar_specs_for_interval(30, 0), ["*-*-* *:00,30:00"])
        self.assertEqual(_calendar_specs_for_interval(1440), ["*-*-* 00:00:00"])
        self.assertEqual(_calendar_specs_for_interval(720), ["*-*-* 00,12:00:00"])

    def test_daily_recording_starts_at_the_chosen_time(self) -> None:
        self.assertEqual(_calendar_specs_for_interval(1440, 600), ["*-*-* 10:00:00"])
        self.assertEqual(_calendar_specs_for_interval(1440, 14 * 60 + 45), ["*-*-* 14:45:00"])

    def test_shorter_intervals_are_counted_from_the_start_time(self) -> None:
        self.assertEqual(_calendar_specs_for_interval(30, 10), ["*-*-* *:10,40:00"])
        self.assertEqual(_calendar_specs_for_interval(720, 9 * 60), ["*-*-* 09,21:00:00"])
        # every 90 min from 08:00: 08:00, 09:30, 11:00, ... wrapping past midnight
        self.assertEqual(
            _calendar_specs_for_interval(90, 8 * 60),
            ["*-*-* 00,03,06,09,12,15,18,21:30:00", "*-*-* 02,05,08,11,14,17,20,23:00:00"],
        )

    def test_same_number_of_slots_whatever_the_start(self) -> None:
        def slots(specs: list[str]) -> int:
            total = 0
            for spec in specs:
                hours, minutes, _ = spec.split(" ")[1].split(":")
                total += (24 if hours == "*" else len(hours.split(","))) * len(minutes.split(","))
            return total

        for interval in (7, 30, 45, 90, 720, 1440):
            self.assertEqual(slots(_calendar_specs_for_interval(interval, 0)), slots(_calendar_specs_for_interval(interval, 613)))

    def test_written_timer_carries_the_start_time_and_counts_as_a_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_defaults()
            config["capture"].update({"recording_seconds": 7200, "record_interval_minutes": 1440})
            write_systemd_units(config=config, config_path=root / "config.yaml", output_dir=root)
            timer = root / "bumblebox-v2-main.timer"
            self.assertIn("OnCalendar=*-*-* 00:00:00", timer.read_text())
            self.assertTrue(installed_units_are_current(config, root / "config.yaml", installed_dir=root))

            config["scheduling"]["start_time"] = "10:00"
            self.assertFalse(installed_units_are_current(config, root / "config.yaml", installed_dir=root))
            write_systemd_units(config=config, config_path=root / "config.yaml", output_dir=root)
            self.assertIn("OnCalendar=*-*-* 10:00:00", timer.read_text())
            self.assertNotIn("00:00:00", timer.read_text())

    def test_mixed_schedule_moves_both_timers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_defaults()
            config["pipeline"]["mode"] = "mixed_schedule"
            config["scheduling"]["start_time"] = "00:10"
            write_systemd_units(config=config, config_path=root / "config.yaml", output_dir=root)
            self.assertIn("OnCalendar=*-*-* *:10,40:00", (root / "bumblebox-v2-record.timer").read_text())
            self.assertIn("OnCalendar=*-*-* *:00,05,10,15,20,25,30,35,40,45,50,55:00", (root / "bumblebox-v2-track.timer").read_text())


class PlanSentenceTests(unittest.TestCase):
    def plan(self, **changes) -> str:
        config = load_defaults()
        for dotted, value in changes.items():
            section, key = dotted.split("__")
            config[section][key] = value
        return describe_schedule_plan(config)

    def test_default_reads_as_before(self) -> None:
        self.assertEqual(self.plan(), "Records 20 s every 30 min, then tracks.")

    def test_two_hours_once_a_day(self) -> None:
        self.assertEqual(
            self.plan(capture__recording_seconds=7200, capture__record_interval_minutes=1440,
                      scheduling__start_time="10:00", pipeline__mode="record_only"),
            "Records 2 h once a day at 10:00.",
        )

    def test_offset_schedule_names_its_start(self) -> None:
        self.assertEqual(
            self.plan(capture__recording_seconds=600, capture__record_interval_minutes=120, scheduling__start_time="00:15"),
            "Records 10 min every 2 h, counted from 00:15, then tracks.",
        )

    def test_start_time_is_part_of_the_same_schedule_for_two_cameras(self) -> None:
        first, second = load_defaults(), load_defaults()
        first["scheduling"]["start_time"] = "10:00"
        self.assertFalse(schedules_match(first, second))
        self.assertTrue(copy_schedule(first, second))
        self.assertEqual(second["scheduling"]["start_time"], "10:00")


if __name__ == "__main__":
    unittest.main()
