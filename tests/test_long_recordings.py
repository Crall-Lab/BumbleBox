"""Long recordings: consecutive video files, runs that do not finish, and the restart watchdog."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2

from bumblebox_v2 import cli, heartbeat, recording_watchdog, run_engine
from bumblebox_v2.config import ConfigError, load_defaults, save_config, validate_config
from bumblebox_v2.recording_watchdog import describe_result, restart_enabled, supervise_recording
from bumblebox_v2.simulated_capture import simulated_rgb_yuv420_frame
from bumblebox_v2.stream_writer import StreamingVideoWriter, find_ffmpeg_binary, stream_chunk_frames
from bumblebox_v2.systemd_units import run_timeout_seconds

try:  # discovered with "-s tests", or run as tests.test_long_recordings
    from test_tracking_and_metrics import TAG_IDS, tagged_yuv420_frame
except ImportError:  # pragma: no cover
    from tests.test_tracking_and_metrics import TAG_IDS, tagged_yuv420_frame

HAVE_FFMPEG = find_ffmpeg_binary() is not None


def frames_in(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
class ChunkedWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def write(self, writer: StreamingVideoWriter, count: int, fps: float = 4.0) -> None:
        for index in range(count):
            writer.add(simulated_rgb_yuv420_frame(64, 48, index / fps, 10.0), SimpleNamespace(time_s=index / fps))

    def test_recording_is_split_into_consecutive_files_with_every_frame(self) -> None:
        closed = []
        writer = StreamingVideoWriter(
            output_dir=self.out, stem="s", fps=5.0, chunk_frames=10, on_chunk_closed=closed.append
        )
        self.write(writer, 25)
        video = writer.finish(actual_fps=4.0)
        self.assertEqual(video.frame_count, 25)
        self.assertEqual([path.name for path in video.video_paths], ["s_part001.mp4", "s_part002.mp4", "s_part003.mp4"])
        self.assertEqual(video.video_path.name, "s_part001.mp4")
        self.assertEqual([frames_in(path) for path in video.video_paths], [10, 10, 5])
        self.assertEqual([(c.index, c.first_frame, c.frame_count, len(c.records)) for c in closed],
                         [(1, 0, 10, 10), (2, 10, 10, 10), (3, 20, 5, 5)])
        # Each file carries the frame rate measured over its own frames (4 fps here, not the nominal 5).
        self.assertEqual([chunk.container_fps for chunk in video.chunks], [4.0, 4.0, 4.0])
        self.assertEqual(sorted(path.name for path in self.out.iterdir()), ["s_part001.mp4", "s_part002.mp4", "s_part003.mp4"])

    def test_mjpeg_recordings_are_split_too(self) -> None:
        writer = StreamingVideoWriter(output_dir=self.out, stem="m", fps=5.0, recording_codec="mjpeg", chunk_frames=6)
        self.write(writer, 13)
        video = writer.finish(actual_fps=4.0)
        self.assertEqual([path.name for path in video.video_paths], ["m_part001.avi", "m_part002.avi", "m_part003.avi"])
        self.assertEqual([frames_in(path) for path in video.video_paths], [6, 6, 1])

    def test_one_file_recording_is_unchanged(self) -> None:
        writer = StreamingVideoWriter(output_dir=self.out, stem="one", fps=5.0)
        self.write(writer, 12)
        video = writer.finish(actual_fps=5.0)
        self.assertEqual(video.video_path.name, "one.mp4")
        self.assertEqual(video.chunks, [])
        self.assertEqual([path.name for path in video.video_paths], ["one.mp4"])
        self.assertEqual(frames_in(video.video_path), 12)

    def test_stopping_early_keeps_every_file_including_the_one_in_progress(self) -> None:
        closed = []
        writer = StreamingVideoWriter(
            output_dir=self.out, stem="a", fps=5.0, chunk_frames=10, on_chunk_closed=closed.append
        )
        self.write(writer, 14)
        writer.abort()
        self.assertEqual([chunk.frame_count for chunk in writer.chunks], [10, 4])
        self.assertEqual([frames_in(chunk.video_path) for chunk in writer.chunks], [10, 4])
        self.assertEqual(len(closed), 2)
        self.assertEqual([path.suffix for path in self.out.iterdir()], [".mp4", ".mp4"])

    def test_a_failing_callback_does_not_stop_the_recording(self) -> None:
        def broken(_chunk):
            raise OSError("disk hiccup")

        writer = StreamingVideoWriter(output_dir=self.out, stem="c", fps=5.0, chunk_frames=5, on_chunk_closed=broken)
        self.write(writer, 8)
        video = writer.finish(actual_fps=4.0)
        self.assertEqual([frames_in(path) for path in video.video_paths], [5, 3])
        self.assertTrue(any("disk hiccup" in warning for warning in video.warnings))

    def test_chunk_length_from_the_config(self) -> None:
        config = load_defaults()  # 10 minutes at 5 fps
        self.assertEqual(stream_chunk_frames(config), 3000)
        config["capture"]["chunk_minutes"] = 0
        self.assertEqual(stream_chunk_frames(config), 0)
        config["capture"]["chunk_minutes"] = 2.5
        config["camera"]["fps_target"] = 4
        self.assertEqual(stream_chunk_frames(config), 600)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
class ChunkedRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        lock = patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(self.root / "capture.lock")})
        lock.start()
        self.addCleanup(lock.stop)

    def config(self, *, seconds: float = 12, frames_per_file: int = 20, mode: str = "record_only") -> dict:
        config = load_defaults()
        config["system"]["data_root"] = str(self.root / "data")
        config["local_index"]["enabled"] = False
        config["pipeline"]["mode"] = mode
        config["pipeline"]["calculate_behavior_metrics"] = False
        config["camera"].update({"width": 640, "height": 480})
        config["capture"].update(
            {"recording_seconds": seconds, "frame_buffer": "stream", "chunk_minutes": frames_per_file / 5 / 60}
        )
        config["runtime"].update({"use_mock_camera": True, "fps_report_on_each_recording": False})
        return config

    def test_long_run_lists_its_files_and_saves_timestamps_with_each(self) -> None:
        summary = run_engine.run_once(config=self.config())
        self.assertTrue(summary.success, summary.errors)
        self.assertEqual(summary.frames_captured, 60)
        names = [Path(part).name for part in summary.video_parts]
        self.assertEqual(names, [f"{summary.session_name}_part00{n}.mp4" for n in (1, 2, 3)])
        self.assertEqual(Path(summary.video_path).name, names[0])
        session = Path(summary.session_dir)
        for number, first in ((1, 0), (2, 20), (3, 40)):
            lines = (session / f"{summary.session_name}_part00{number}_frame_timestamps.csv").read_text().splitlines()
            self.assertEqual(len(lines), 21)
            self.assertTrue(lines[1].startswith(f"{first},"))
        whole = (session / f"{summary.session_name}_frame_timestamps.csv").read_text().splitlines()
        self.assertEqual(len(whole), 61)
        saved = json.loads((session / f"{summary.session_name}_run_summary.json").read_text())
        self.assertTrue(saved["success"])
        self.assertNotIn("unfinished", saved)

        frames = run_engine._VideoFrameSequence(summary.video_parts, 60)
        try:
            self.assertIsNotNone(frames[0])
            self.assertIsNotNone(frames[25])  # in the second file
            self.assertIsNotNone(frames[59])  # the last frame of the third
            with self.assertRaises(IndexError):
                frames[60]
        finally:
            frames.close()

    def test_recording_shorter_than_one_file_keeps_its_usual_name(self) -> None:
        summary = run_engine.run_once(config=self.config(seconds=2, frames_per_file=20))
        self.assertTrue(summary.success, summary.errors)
        self.assertEqual(summary.video_parts, [])
        self.assertEqual(Path(summary.video_path).name, f"{summary.session_name}.mp4")

    def test_tags_are_tracked_across_files_with_continuous_frame_numbers(self) -> None:
        import pandas as pd

        def tagged(width, height, time_s, _duration):
            return tagged_yuv420_frame(width, height, time_s % 1.0, 1.0)

        with patch.object(run_engine, "simulated_rgb_yuv420_frame", tagged):
            summary = run_engine.run_once(
                config=self.config(seconds=2, frames_per_file=4, mode="record_and_track")
            )
        self.assertTrue(summary.success, summary.errors)
        self.assertEqual(len(summary.video_parts), 3)  # 4 + 4 + 2 frames
        raw = pd.read_csv(summary.raw_csv_path)
        self.assertEqual(sorted(raw["frame"].unique()), list(range(10)))
        self.assertEqual(sorted(raw["ID"].unique()), sorted(TAG_IDS))
        self.assertEqual(set(raw["filename"]), {summary.session_name})
        leftovers = [path.name for path in Path(summary.session_dir).glob("*_part0*_raw.csv")]
        self.assertEqual(leftovers, [])

    def interrupt_after(self, frames: int, error: type[BaseException]):
        original = run_engine.simulated_rgb_yuv420_frame
        seen = {"count": 0}

        def frame(*args, **kwargs):
            seen["count"] += 1
            if seen["count"] > frames:
                raise error
            return original(*args, **kwargs)

        return patch.object(run_engine, "simulated_rgb_yuv420_frame", frame)

    def test_cancelled_long_run_keeps_every_file_written_so_far(self) -> None:
        with self.interrupt_after(45, KeyboardInterrupt):
            summary = run_engine.run_once(config=self.config())
        self.assertFalse(summary.success)
        self.assertIn("Cancelled before it finished", " ".join(summary.errors))
        self.assertEqual([frames_in(Path(part)) for part in summary.video_parts], [20, 20, 5])
        self.assertIn("3 video file(s) with 45 frames were kept", " ".join(summary.warnings))
        self.assertEqual(list(Path(summary.session_dir).glob("*.mkv")), [])

    def test_run_that_is_killed_still_appears_as_a_run_that_did_not_finish(self) -> None:
        class Killed(BaseException):
            """Stands in for a kill or power loss: nothing gets to tidy up the summary."""

        with self.interrupt_after(45, Killed), self.assertRaises(Killed):
            run_engine.run_once(config=self.config())
        summaries = list((self.root / "data").glob("*/*/*_run_summary.json"))
        self.assertEqual(len(summaries), 1)
        saved = json.loads(summaries[0].read_text())
        self.assertFalse(saved["success"])
        self.assertTrue(saved["unfinished"])
        self.assertTrue(saved["errors"][0].startswith("Did not finish: the recorder was stopped, crashed, or lost power."))
        from bumblebox_v2.status_history import list_recent_run_records

        records = list_recent_run_records(self.root / "data")
        self.assertEqual([(record.success, record.error_count) for record in records], [(False, 1)])


def child(code: str) -> list[str]:
    """A stand-in recorder: Python code that can use beat() and done()."""
    prelude = textwrap.dedent(
        """
        import os, sys, time
        _fd = int(os.environ["BUMBLEBOX_HEARTBEAT_FD"])
        def beat(): os.write(_fd, b".")
        def done(): os.write(_fd, b"C")
        """
    )
    return [sys.executable, "-c", prelude + textwrap.dedent(code)]


class WatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.marker = Path(self._tmp.name) / "attempts"
        self.said: list[str] = []
        self.requested: list[float] = []

    def supervise(self, code: str, **options):
        defaults = dict(total_seconds=60, stall_after=30, restart_delays=(0.05,), min_resume_seconds=1, say=self.said.append)
        defaults.update(options)

        def build(seconds: float, attempt: int):
            self.requested.append(seconds)
            return child(code.replace("MARKER", repr(str(self.marker))))

        return supervise_recording(build, **defaults)

    # Lets a stand-in recorder behave differently on its first and later attempts.
    COUNT = (
        "path = MARKER\n"
        "attempt = int(open(path).read()) + 1 if os.path.exists(path) else 1\n"
        "open(path, 'w').write(str(attempt))\n"
    )

    def test_clean_recording_is_one_attempt(self) -> None:
        result = self.supervise("beat(); done(); sys.exit(0)")
        self.assertEqual((len(result.attempts), result.restarts, result.exit_code), (1, 0, 0))
        self.assertIsNone(describe_result(result, 60))
        self.assertEqual(self.said, [])

    def test_crashed_recorder_is_started_again_for_the_time_left(self) -> None:
        result = self.supervise(self.COUNT + "beat()\nif attempt == 1: sys.exit(3)\ndone(); sys.exit(0)")
        self.assertEqual((len(result.attempts), result.restarts), (2, 1))
        self.assertEqual(result.attempts[0].exit_code, 3)
        self.assertTrue(result.attempts[1].ok)
        self.assertEqual(self.requested[0], 60)
        self.assertLess(self.requested[1], 60)
        self.assertGreater(self.requested[1], 50)
        # The session has a gap, so it is still reported as a failure.
        self.assertEqual(result.exit_code, 1)
        line = describe_result(result, 60)
        self.assertTrue(line.startswith("Recording failed part-way: the recorder stopped with exit code 3."))
        self.assertIn("restarted 1 time(s) and the rest of the 1 min session was recorded", line)
        self.assertIn("Starting again in", self.said[0])

    def test_recorder_that_goes_quiet_is_stopped_and_restarted(self) -> None:
        started = time.monotonic()
        result = self.supervise(
            self.COUNT + "beat()\nif attempt == 1: time.sleep(60)\ndone(); sys.exit(0)", stall_after=1.0
        )
        self.assertLess(time.monotonic() - started, 15)
        self.assertTrue(result.attempts[0].stalled)
        self.assertTrue(result.attempts[1].ok)
        self.assertIn("it has stalled and is being stopped", self.said[0])
        self.assertIn("the recorder stalled", describe_result(result, 60))

    def test_slow_work_after_capture_is_not_a_stall(self) -> None:
        result = self.supervise("beat(); done(); time.sleep(2.5); sys.exit(0)", stall_after=1.0)
        self.assertEqual((len(result.attempts), result.exit_code), (1, 0))
        self.assertFalse(result.attempts[0].stalled)

    def test_failure_after_capture_is_not_recorded_again(self) -> None:
        result = self.supervise("beat(); done(); sys.exit(1)")
        self.assertEqual((len(result.attempts), result.exit_code), (1, 1))
        self.assertTrue(result.attempts[0].capture_finished)
        self.assertIsNone(describe_result(result, 60))

    def test_broken_recorder_is_not_restarted_for_ever(self) -> None:
        result = self.supervise("sys.exit(2)", restarts_allowed=2)
        self.assertEqual((len(result.attempts), result.restarts, result.exit_code), (3, 2, 1))
        self.assertIn("already been restarted 2 time(s)", result.gave_up)
        self.assertIn("then left, because it had already been restarted 2 time(s)", describe_result(result, 60))

    def test_no_restart_when_too_little_of_the_session_is_left(self) -> None:
        result = self.supervise("sys.exit(2)", total_seconds=5, min_resume_seconds=30)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.gave_up, "too little of the session was left to record")

    def test_cancel_stops_the_recorder_and_does_not_restart_it(self) -> None:
        import signal
        import threading

        # Ctrl+C, or Cancel in the GUI, reaches the watchdog as SIGINT.
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        timer = threading.Timer(0.7, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.start()
        try:
            started = time.monotonic()
            result = self.supervise("beat()\ntime.sleep(60)")
        finally:
            timer.cancel()
            signal.signal(signal.SIGINT, previous)
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(result.interrupted)
        self.assertEqual((len(result.attempts), result.exit_code), (1, 130))
        self.assertIsNotNone(result.attempts[0].exit_code)  # the recorder was stopped, not left running
        self.assertIsNone(describe_result(result, 60))

    def test_killed_recorder_is_described_as_killed(self) -> None:
        result = self.supervise("import signal; os.kill(os.getpid(), signal.SIGKILL)", restarts_allowed=0)
        self.assertIn("was killed (signal 9)", describe_result(result, 60))

    def test_auto_applies_to_recordings_of_two_minutes_or_more(self) -> None:
        config = load_defaults()
        self.assertFalse(restart_enabled(config))  # 20 s
        config["capture"]["recording_seconds"] = 120
        self.assertTrue(restart_enabled(config))
        config["capture"]["restart_on_failure"] = False
        self.assertFalse(restart_enabled(config))
        config["capture"].update({"recording_seconds": 20, "restart_on_failure": True})
        self.assertTrue(restart_enabled(config))

    def test_recorder_signals_reach_the_watchdog(self) -> None:
        read_end, write_end = os.pipe()
        try:
            with patch.dict(os.environ, {heartbeat.HEARTBEAT_FD_ENV: str(write_end)}), patch.object(
                heartbeat, "_looked", False
            ), patch.object(heartbeat, "_fd", None), patch.object(heartbeat, "_last_beat", 0.0):
                heartbeat.beat()
                heartbeat.beat()  # too soon after the first: not sent
                heartbeat.capture_finished()
                self.assertEqual(os.read(read_end, 16), b".C")
        finally:
            os.close(read_end)
            os.close(write_end)

    def test_signals_do_nothing_without_a_watchdog(self) -> None:
        with patch.dict(os.environ, {}, clear=False), patch.object(heartbeat, "_looked", False), patch.object(
            heartbeat, "_fd", None
        ):
            os.environ.pop(heartbeat.HEARTBEAT_FD_ENV, None)
            heartbeat.beat()
            heartbeat.capture_finished()


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
class WatchedCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        lock = patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(self.root / "capture.lock")})
        lock.start()
        self.addCleanup(lock.stop)
        config = load_defaults()
        config["system"]["data_root"] = str(self.root / "data")
        config["local_index"]["enabled"] = False
        config["pipeline"]["mode"] = "record_only"
        config["camera"].update({"width": 64, "height": 48})
        config["capture"].update({"recording_seconds": 1, "frame_buffer": "stream", "restart_on_failure": True})
        config["runtime"].update({"use_mock_camera": True, "fps_report_on_each_recording": False})
        self.config = config
        self.path = self.root / "config.yaml"
        save_config(self.path, config)

    def run_cli(self, *arguments: str) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(list(arguments))
        return code, output.getvalue()

    def summaries(self) -> list[dict]:
        return [json.loads(path.read_text()) for path in sorted((self.root / "data").glob("*/*/*_run_summary.json"))]

    def test_recording_runs_under_the_watchdog_and_succeeds(self) -> None:
        with patch.object(cli, "supervise_recording", wraps=supervise_recording) as watched:
            code, _ = self.run_cli("run-once", "--config", str(self.path))
        self.assertEqual(code, 0)
        self.assertEqual(watched.call_count, 1)
        command = watched.call_args[0][0](1.0, 1)
        self.assertEqual(command[-1], "--watchdog-child")
        self.assertEqual([summary["success"] for summary in self.summaries()], [True])

    def test_no_restart_flag_and_short_default_runs_skip_the_watchdog(self) -> None:
        with patch.object(cli, "supervise_recording") as watched:
            code, _ = self.run_cli("run-once", "--config", str(self.path), "--no-restart")
            self.assertEqual(code, 0)
            self.config["capture"]["restart_on_failure"] = "auto"  # a 1 s recording
            save_config(self.path, self.config)
            code, _ = self.run_cli("run-once", "--config", str(self.path))
            self.assertEqual(code, 0)
        self.assertEqual(watched.call_count, 0)

    def test_failed_recording_ends_with_one_line_that_says_what_happened(self) -> None:
        self.config["system"]["min_free_space_gb"] = 10_000_000  # the storage guard refuses every attempt
        save_config(self.path, self.config)
        code, output = self.run_cli("run-once", "--config", str(self.path))
        self.assertEqual(code, 1)
        self.assertIn("Recording failed part-way: the recorder stopped with exit code 1.", output)
        self.assertIn("too little of the session was left to record", output)

        from bumblebox_v2.box_status import explain_failure

        explanation = explain_failure(output, exit_code=1, command=["run-once"])
        self.assertEqual(explanation.title, "Recording failed")
        self.assertTrue(explanation.reason.startswith("the recorder stopped with exit code 1."))

    def test_recording_length_can_be_overridden_for_one_run(self) -> None:
        code, _ = self.run_cli("run-once", "--config", str(self.path), "--no-restart", "--recording-seconds", "2")
        self.assertEqual(code, 0)
        self.assertEqual(self.summaries()[0]["frames_captured"], 10)


class LimitsAndSettingsTests(unittest.TestCase):
    def test_hung_daily_run_cannot_take_the_next_day_with_it(self) -> None:
        config = load_defaults()
        config["capture"]["recording_seconds"] = 7200
        self.assertEqual(run_timeout_seconds(config, 1440), 24 * 3600 - 300)
        self.assertEqual(run_timeout_seconds(config, 360), 6 * 3600 - 300)
        # Too tight an interval to fit the recording plus 30 minutes: the old rule stands.
        self.assertEqual(run_timeout_seconds(config, 132), 7200 + 2 * 132 * 60)

    def test_short_schedules_keep_their_limit(self) -> None:
        config = load_defaults()  # 20 s recordings
        self.assertEqual(run_timeout_seconds(config, 30), 20 + 2 * 30 * 60)
        self.assertEqual(run_timeout_seconds(config, 5), 20 + 30 * 60)
        self.assertEqual(run_timeout_seconds(config, 60), 55 * 60)
        config["scheduling"]["run_timeout_minutes"] = 90
        self.assertEqual(run_timeout_seconds(config, 1440), 90 * 60)

    def test_settings_are_validated(self) -> None:
        for key, bad in (("chunk_minutes", -1), ("chunk_minutes", "ten"), ("restart_on_failure", "sometimes"),
                         ("stall_seconds", 5), ("max_restarts", -1), ("max_restarts", 2.5)):
            config = load_defaults()
            config["capture"][key] = bad
            with self.assertRaises(ConfigError, msg=f"{key}={bad!r}"):
                validate_config(config)
        config = load_defaults()
        config["capture"].update({"chunk_minutes": 0, "restart_on_failure": True, "stall_seconds": 30, "max_restarts": 0})
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
