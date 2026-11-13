"""Plain-language status of a BumbleBox, independent of any GUI toolkit.

A person walking up to a box needs four answers without reading a log:

* Can it record right now (is the data drive there, is there room)?
* Is the recording schedule on, and when is the next recording?
* Did the last scheduled run work, and if not, why not?
* When a command they just ran failed, what was the reason?

This module produces those answers as short text plus a level (``ok``,
``warn``, ``fail`` or ``off``). It imports no GUI code, so the Qt window, a
command line report or a future web page can all show the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import getpass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .encode_benchmark import HEADROOM_FOR_YES
from .schedule_check import run_schedule_check
from .storage_guard import (
    DEFAULT_MIN_FREE_SPACE_GB,
    StorageReadiness,
    check_storage_ready,
    containing_mount_point,
)
from .stream_writer import FRAME_BUFFER_STREAM, frame_buffer_mode, normalize_mp4_codec, resolve_mp4_preset
from .systemd_units import _job_specs, write_systemd_units


LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_FAIL = "fail"
LEVEL_OFF = "off"

LOW_FREE_SPACE_WARNING_GB = 10.0
COMMAND_TIMEOUT_SECONDS = 5.0
JOURNAL_LINES = 80

SCHEDULE_ON = "on"
SCHEDULE_OFF = "off"
SCHEDULE_PARTIAL = "partial"
SCHEDULE_UNKNOWN = "unknown"

# (returncode, combined output). returncode is None when the command could not run.
CommandRunner = Callable[[List[str]], Tuple[Optional[int], str]]


@dataclass
class StatusLine:
    level: str
    headline: str
    detail: str = ""
    free_gb: Optional[float] = None


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def describe_duration(seconds: float) -> str:
    """Round a duration to the one or two units a person would say aloud."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return "less than a minute"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    days, hours = divmod(hours, 24)
    unit = "day" if days == 1 else "days"
    return f"{days} {unit} {hours} h" if hours else f"{days} {unit}"


def describe_moment(moment: datetime, now: datetime) -> str:
    """'14:30' for today, otherwise 'Fri 13 Nov 14:30'."""
    if moment.date() == now.date():
        return moment.strftime("%H:%M")
    return f"{moment.strftime('%a')} {moment.day} {moment.strftime('%b %H:%M')}"


def describe_relative(moment: datetime, now: datetime) -> str:
    delta = (moment - now).total_seconds()
    if delta >= 0:
        return f"in {describe_duration(delta)}"
    return f"{describe_duration(-delta)} ago"


def _format_gb(value: float) -> str:
    return f"{value:.0f} GB" if value >= 10 else f"{value:.1f} GB"


def _format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "?"
    return f"{seconds:g} s"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def storage_status(
    config: Dict[str, Any],
    *,
    readiness: Optional[StorageReadiness] = None,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
    ismount: Callable[[str], bool] = os.path.ismount,
) -> StatusLine:
    """Summarise whether a recording could be written to the data folder now."""
    if readiness is None:
        readiness = check_storage_ready(config)
    path = readiness.path

    if not path:
        return StatusLine(LEVEL_FAIL, "No data folder set", "Choose a data folder in the setup wizard.")

    if readiness.mounted is False:
        return StatusLine(
            LEVEL_FAIL,
            "Data drive not mounted",
            f"{path} is on the system disk. Recording is refused until the drive is connected.",
        )

    free_gb = readiness.free_gb
    if free_gb is None:
        try:
            probe = Path(path)
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            free_gb = round(int(disk_usage(str(probe)).free) / 1e9, 2)
        except Exception:
            free_gb = None

    if not readiness.ready:
        if free_gb is not None and free_gb < readiness.min_free_gb:
            return StatusLine(
                LEVEL_FAIL,
                f"Drive nearly full · {_format_gb(free_gb)} free",
                f"{path}. Recording is refused below {readiness.min_free_gb:g} GB free.",
                free_gb=free_gb,
            )
        return StatusLine(LEVEL_FAIL, "Recording is refused", readiness.problem or path, free_gb=free_gb)

    on_separate_drive = readiness.requirement != "not_required"
    if not on_separate_drive:
        try:
            on_separate_drive = containing_mount_point(path, ismount=ismount) != "/"
        except Exception:
            on_separate_drive = False
    where = "Drive mounted" if on_separate_drive else "On the system disk"
    headline = f"{where} · {_format_gb(free_gb)} free" if free_gb is not None else where
    if readiness.notes:
        return StatusLine(LEVEL_WARN, headline, f"{path}. {readiness.notes[0]}", free_gb=free_gb)
    if free_gb is not None and free_gb < LOW_FREE_SPACE_WARNING_GB:
        return StatusLine(LEVEL_WARN, headline, f"{path}. Space is getting low.", free_gb=free_gb)
    return StatusLine(LEVEL_OK, headline, path, free_gb=free_gb)


# ---------------------------------------------------------------------------
# Storage use per day
# ---------------------------------------------------------------------------

RECORDING_MODES = {"record_only", "record_and_track", "mixed_schedule"}
RECORDING_RUN_MODES = {"record_only", "record_and_track"}
ESTIMATE_SAMPLE_RUNS = 10
SPACE_RUNNING_LOW_DAYS = 7.0
SESSION_SIZE_MAX_FILES = 20_000


def format_bytes(value: float) -> str:
    """Decimal units, as drives are sold: 1 GB = 1e9 bytes."""
    value = max(0.0, float(value))
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if value >= size:
            scaled = value / size
            return f"{scaled:.0f} {unit}" if scaled >= 10 else f"{scaled:.1f} {unit}"
    return f"{value:.0f} B"


def describe_days(days: float) -> str:
    if days >= 365 * 2:
        return f"about {days / 365:.0f} years"
    if days >= 60:
        return f"about {days / 30.4:.0f} months"
    if days >= 2:
        return f"about {days:.0f} days"
    hours = days * 24
    if hours >= 2:
        return f"about {hours:.0f} hours"
    return "less than 2 hours"


def session_size_bytes(session_dir: str | Path, *, max_files: int = SESSION_SIZE_MAX_FILES) -> Optional[int]:
    """Total size of the files in a session folder, or ``None`` if it cannot be read."""
    root = Path(session_dir)
    if not root.is_dir():
        return None
    total = 0
    seen = 0
    stack = [root]
    try:
        while stack:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                        seen += 1
                        if seen > max_files:
                            return None
    except OSError:
        return None
    return total


def recording_signature(config: Dict[str, Any]) -> Tuple[str, str, bool, int, int]:
    """The settings that decide how many bytes a recorded frame takes on disk.

    Two recordings with the same signature can be compared per frame; the
    recording length, frame rate and interval then scale the total.
    """
    camera = config.get("camera", {}) if isinstance(config.get("camera"), dict) else {}
    codec = str(camera.get("codec", "mp4")).strip().lower()
    streaming = frame_buffer_mode(config) == FRAME_BUFFER_STREAM
    preset = resolve_mp4_preset(camera.get("mp4_preset"), streaming=streaming) if codec == "mp4" else "-"

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return (
        codec,
        preset,
        bool(camera.get("monochrome_output", False)),
        _int(camera.get("width")),
        _int(camera.get("height")),
    )


def describe_signature(signature: Sequence[Any]) -> str:
    codec, preset, mono, width, height = signature
    video = "MJPEG" if codec == "mjpeg" else f"MP4 ({preset})"
    colour = "grayscale" if mono else "colour"
    return f"{width} × {height}, {video}, {colour}"


def signature_from_snapshot(snapshot_path: Any) -> Optional[Tuple[str, str, bool, int, int]]:
    """Read the recording signature from a run's saved config snapshot."""
    text = str(snapshot_path or "").strip()
    if not text:
        return None
    try:
        snapshot = json.loads(Path(text).read_text())
    except (OSError, ValueError):
        return None
    return recording_signature(snapshot) if isinstance(snapshot, dict) else None


def recordings_per_day(config: Dict[str, Any]) -> int:
    """How many recordings the schedule starts in 24 hours (0 for track-only)."""
    mode = str((config.get("pipeline") or {}).get("mode", "record_and_track"))
    if mode not in RECORDING_MODES:
        return 0
    try:
        interval = max(1, int((config.get("capture") or {}).get("record_interval_minutes", 30)))
    except (TypeError, ValueError):
        return 0
    # Timers fire at minute 0, interval, 2 x interval, ... of each day.
    return math.ceil(24 * 60 / interval)


def frames_per_recording(config: Dict[str, Any]) -> float:
    capture = config.get("capture", {}) if isinstance(config.get("capture"), dict) else {}
    camera = config.get("camera", {}) if isinstance(config.get("camera"), dict) else {}
    try:
        return max(0.0, float(capture.get("recording_seconds", 0)) * float(camera.get("fps_target", 0)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Saved encode benchmark results
# ---------------------------------------------------------------------------

def load_benchmark(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read a report saved by ``bbx encode-benchmark --json-out``."""
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) and isinstance(report.get("results"), list) else None


def benchmark_used_real_video(benchmark: Dict[str, Any]) -> bool:
    """Synthetic frames compress differently from a nest, so only real footage gives file sizes."""
    return str(benchmark.get("source", "")).startswith("video:")


def benchmark_matches_camera(config: Dict[str, Any], benchmark: Dict[str, Any]) -> bool:
    _, _, mono, width, height = recording_signature(config)
    try:
        same_size = int(benchmark.get("width", 0)) == width and int(benchmark.get("height", 0)) == height
    except (TypeError, ValueError):
        return False
    return same_size and bool(benchmark.get("monochrome_output", False)) == mono


def benchmark_result_for(config: Dict[str, Any], benchmark: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The benchmark result for the encoder ``config`` would use, if it was measured."""
    if not isinstance(benchmark, dict) or not benchmark_matches_camera(config, benchmark):
        return None
    camera = config.get("camera", {}) if isinstance(config.get("camera"), dict) else {}
    codec, preset, _, _, _ = recording_signature(config)
    wanted_mp4_codec = normalize_mp4_codec(camera.get("mp4_codec", "libx264"))
    for result in benchmark.get("results") or []:
        if not isinstance(result, dict) or result.get("error"):
            continue
        if codec == "mjpeg" and result.get("codec") == "mjpeg":
            return result
        if (
            codec == "mp4"
            and result.get("codec") == "mp4"
            and str(result.get("mp4_preset")) == preset
            and normalize_mp4_codec(result.get("mp4_codec")) == wanted_mp4_codec
        ):
            return result
    return None


def keeps_up(frames_per_second: Any, fps_target: Any) -> str:
    """'yes', 'marginal' or 'no', by the same rule as the encode benchmark."""
    try:
        measured, target = float(frames_per_second), float(fps_target)
    except (TypeError, ValueError):
        return "n/a"
    if target <= 0:
        return "n/a"
    if measured >= HEADROOM_FOR_YES * target:
        return "yes"
    if measured >= target:
        return "marginal"
    return "no"


@dataclass
class StorageEstimate:
    level: str
    headline: str
    detail: str = ""
    bytes_per_recording: Optional[float] = None
    bytes_per_day: Optional[float] = None
    days_left: Optional[float] = None
    sample_runs: int = 0


def estimate_storage(
    config: Dict[str, Any],
    recordings: Sequence[Dict[str, Any]],
    *,
    free_gb: Optional[float] = None,
    benchmark: Optional[Dict[str, Any]] = None,
) -> StorageEstimate:
    """Estimate storage per day and days of space left from past recordings.

    ``recordings`` are run summaries, newest first, with two extra keys added
    by the caller: ``_session_bytes`` (size of the session folder) and
    ``_recording_signature`` (from the run's config snapshot). Only
    successful recordings made with the same video settings as ``config``
    are used, because the size of a frame depends on them. The whole session
    folder is counted, so tracking outputs saved next to the video are
    included.

    When no such recording exists yet, a saved encode benchmark that was run
    on real footage at the same resolution is used instead; that measures the
    video file only.
    """
    per_day = recordings_per_day(config)
    if per_day == 0:
        return StorageEstimate(LEVEL_OFF, "No new video", "Track-only mode does not record.")

    wanted = recording_signature(config)
    samples = []
    for run in recordings:
        if not bool(run.get("success", False)) or str(run.get("mode")) not in RECORDING_RUN_MODES:
            continue
        frames = int(run.get("frames_captured", 0) or 0)
        size = run.get("_session_bytes")
        if frames <= 0 or not isinstance(size, (int, float)) or size <= 0:
            continue
        signature = run.get("_recording_signature")
        if signature is None or tuple(signature) != wanted:
            continue
        samples.append((float(size), frames))
        if len(samples) >= ESTIMATE_SAMPLE_RUNS:
            break

    benchmark_result = (
        benchmark_result_for(config, benchmark)
        if not samples and isinstance(benchmark, dict) and benchmark_used_real_video(benchmark)
        else None
    )
    if samples:
        bytes_per_frame = sum(size for size, _ in samples) / sum(frames for _, frames in samples)
        basis = (
            "Measured from the last recording with these video settings."
            if len(samples) == 1
            else f"Measured from the last {len(samples)} recordings with these video settings."
        )
    elif benchmark_result is not None and float(benchmark_result.get("megabytes_per_frame") or 0) > 0:
        bytes_per_frame = float(benchmark_result["megabytes_per_frame"]) * 1e6
        basis = (
            "Video only, from the encode benchmark on one of your recordings; "
            "tracking files add to this."
        )
    else:
        return StorageEstimate(
            LEVEL_OFF,
            "Storage per day not known yet",
            f"Make one recording with the current video settings ({describe_signature(wanted)}) to measure it.",
        )

    per_recording = bytes_per_frame * frames_per_recording(config)
    bytes_per_day = per_recording * per_day
    estimate = StorageEstimate(
        LEVEL_OK,
        f"About {format_bytes(bytes_per_day)} per day",
        f"{format_bytes(per_recording)} per recording, {per_day} recordings a day. {basis}",
        bytes_per_recording=per_recording,
        bytes_per_day=bytes_per_day,
        sample_runs=len(samples),
    )
    if free_gb is None or bytes_per_day <= 0:
        return estimate

    system = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    try:
        reserve_gb = max(0.0, float(system.get("min_free_space_gb", DEFAULT_MIN_FREE_SPACE_GB)))
    except (TypeError, ValueError):
        reserve_gb = DEFAULT_MIN_FREE_SPACE_GB
    usable = max(0.0, (float(free_gb) - reserve_gb) * 1e9)
    estimate.days_left = usable / bytes_per_day
    estimate.headline += f" · {describe_days(estimate.days_left)} of space left"
    if estimate.days_left < SPACE_RUNNING_LOW_DAYS:
        estimate.level = LEVEL_WARN
    return estimate


# ---------------------------------------------------------------------------
# Explaining a failed command
# ---------------------------------------------------------------------------


@dataclass
class FailureExplanation:
    title: str
    reason: str


COMMAND_LABELS: Dict[str, str] = {
    "run-once": "Recording",
    "doctor": "Health check",
    "storage": "Storage check",
    "camera-check": "Camera check",
    "camera-preview": "Camera preview",
    "thermal-check": "Thermal camera check",
    "thermal-snapshot": "Thermal snapshot",
    "realsense-check": "RealSense check",
    "realsense-snapshot": "Depth snapshot",
    "distributed-capture": "Two-Pi check",
    "systemd-write": "Starting automated recording",
    "systemd-install": "Starting automated recording",
    "systemd-enable": "Starting automated recording",
    "systemd-disable": "Stopping automated recording",
    "schedule-check": "Schedule check",
    "live-monitor": "Live camera monitor",
}

# Checked in order against the reason text; the first match names the problem.
_REASON_TITLES: Tuple[Tuple[str, str], ...] = (
    (r"data drive is not mounted", "Data drive not mounted"),
    (r"GB is free where|is required to start a recording", "Data drive is nearly full"),
    (r"data_root is not set", "No data folder set"),
    (r"another bumblebox capture is using the camera", "Camera is busy"),
    (r"no camera detected|failed to open camera|picamera2/libcamera not available", "Camera not available"),
    (r"permission denied|needs elevated privileges", "Permission denied"),
    (r"no space left on device", "Disk is full"),
    (r"time limit", "Stopped at the time limit"),
)

_PREFIXED_FAILURE = re.compile(
    r"^(?P<what>(?:Failed to [^:]{2,80}|[A-Z][^:]{1,80}?"
    r"(?:failed|failed to start|error|needs elevated privileges)[^:]{0,40})): (?P<why>\S.*)$"
)
_TRACEBACK_LAST_LINE = re.compile(r"^(?:[A-Za-z_][\w.]*)(?:Error|Exception|Exit|Interrupt|Warning): (?P<why>\S.*)$")
_CHECK_FAIL_LINE = re.compile(r"^\[FAIL\]\s*(?P<why>\S.*)$")


def command_label(command: Sequence[str]) -> str:
    """A noun phrase for a ``bbx`` command, for example 'Recording' for run-once."""
    name = str(command[0]) if command else ""
    return COMMAND_LABELS.get(name, f"bbx {name}".strip() if name else "The command")


def _title_for(reason: str, command: Sequence[str]) -> str:
    lowered = reason.lower()
    for pattern, title in _REASON_TITLES:
        if re.search(pattern, lowered):
            return title
    name = str(command[0]) if command else ""
    if name == "doctor":
        return "Health check found problems"
    label = command_label(command)
    if name in {"systemd-write", "systemd-install", "systemd-enable"}:
        return "Automated recording was not started"
    if name == "systemd-disable":
        return "Automated recording was not stopped"
    return f"{label} failed"


def explain_failure(
    output: str,
    *,
    exit_code: Optional[int] = None,
    command: Sequence[str] = (),
) -> FailureExplanation:
    """Find the reason a ``bbx`` command failed in what it printed.

    The reason is taken, in order of preference, from: a line such as
    ``Run failed to start: ...``; the ``Error details:`` block of a run
    summary; ``[FAIL]`` lines of a check report; the last line of a Python
    traceback; the last line printed.
    """
    lines = [line.strip() for line in str(output or "").splitlines()]
    lines = [line for line in lines if line]
    reason: Optional[str] = None

    for line in reversed(lines):
        match = _PREFIXED_FAILURE.match(line)
        if match:
            # "Run failed to start: <why>" -> <why>; "Failed to connect to bus: <why>"
            # only makes sense whole.
            reason = line if match.group("what").startswith("Failed to ") else match.group("why").strip()
            break

    if reason is None and "Error details:" in lines:
        start = len(lines) - 1 - lines[::-1].index("Error details:")
        errors = [line[1:].strip() for line in lines[start + 1 :] if line.startswith("-")]
        if errors:
            reason = errors[0]
            if len(errors) > 1:
                reason += f" (and {len(errors) - 1} more; see the command output)"

    if reason is None:
        failed_checks = [m.group("why") for m in map(_CHECK_FAIL_LINE.match, lines) if m]
        if failed_checks:
            shown = failed_checks[:3]
            reason = "\n".join(shown)
            if len(failed_checks) > len(shown):
                reason += f"\n(and {len(failed_checks) - len(shown)} more; see the command output)"

    if reason is None and any(line.startswith("Traceback (most recent call last)") for line in lines):
        for line in reversed(lines):
            match = _TRACEBACK_LAST_LINE.match(line)
            if match:
                reason = f"{line} (an unexpected error; the full trace is in the command output)"
                break

    if reason is None:
        candidates = [
            line
            for line in lines
            if not line.startswith(("$ ", "Note:", "[finished with exit code"))
            and line not in {"Success: False", "Command output:"}
        ]
        if candidates:
            reason = candidates[-1]

    if reason is None:
        code = "an error" if exit_code is None else f"exit code {exit_code}"
        reason = f"The command stopped with {code} and printed no explanation."

    return FailureExplanation(title=_title_for(reason, command), reason=reason)


# ---------------------------------------------------------------------------
# Recording schedule (systemd timers)
# ---------------------------------------------------------------------------


@dataclass
class TimerState:
    label: str
    timer: str
    service: str
    installed: bool = False
    enabled: bool = False
    active: bool = False
    next_run: Optional[datetime] = None
    running: bool = False
    last_started: Optional[datetime] = None
    last_ok: Optional[bool] = None  # None: no run since the timer was loaded
    failure: Optional[FailureExplanation] = None


@dataclass
class ScheduleStatus:
    state: str
    level: str
    headline: str
    detail: str = ""
    next_run: Optional[datetime] = None
    timers: List[TimerState] = field(default_factory=list)
    units_current: Optional[bool] = None


_TIMESTAMP_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIMESTAMP_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def parse_systemd_timestamp(text: Any) -> Optional[datetime]:
    """Parse ``Thu 2026-11-12 17:50:00 CST`` (local time); ``None`` for empty or n/a."""
    tokens = str(text or "").split()
    for index in range(len(tokens) - 1):
        if _TIMESTAMP_DATE.match(tokens[index]) and _TIMESTAMP_TIME.match(tokens[index + 1]):
            try:
                return datetime.strptime(f"{tokens[index]} {tokens[index + 1]}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    return None


def parse_systemctl_show(text: str) -> Dict[str, Dict[str, str]]:
    """Split ``systemctl show unit1 unit2 ...`` output into one dict per unit Id."""
    units: Dict[str, Dict[str, str]] = {}
    for block in re.split(r"\n\s*\n", str(text or "")):
        properties: Dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                properties[key.strip()] = value.strip()
        if properties.get("Id"):
            units[properties["Id"]] = properties
    return units


def _default_runner(command: List[str]) -> Tuple[Optional[int], str]:
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return None, f"{command[0]} is not installed"
    except subprocess.TimeoutExpired:
        return None, f"{command[0]} did not answer within {COMMAND_TIMEOUT_SECONDS:g} s"
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)
    if proc.returncode == 0:
        return 0, proc.stdout or ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def describe_schedule_plan(config: Dict[str, Any]) -> str:
    """One sentence saying what the schedule does, from the config."""
    capture = config.get("capture", {}) if isinstance(config.get("capture"), dict) else {}
    mode = str((config.get("pipeline") or {}).get("mode", "record_and_track"))
    length = _format_seconds(capture.get("recording_seconds"))
    record_every = capture.get("record_interval_minutes", "?")
    track_every = capture.get("track_interval_minutes", "?")
    if mode == "track_only":
        return f"Tracks existing videos every {track_every} min."
    if mode == "mixed_schedule":
        return f"Records {length} every {record_every} min; tracks every {track_every} min."
    if mode == "record_only":
        return f"Records {length} every {record_every} min."
    return f"Records {length} every {record_every} min, then tracks."


def _unit_names(config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    scheduling = config.get("scheduling", {}) if isinstance(config.get("scheduling"), dict) else {}
    prefix = str(scheduling.get("unit_prefix", "bumblebox-v2")).strip() or "bumblebox-v2"
    return [(label, f"{prefix}-{label}.timer", f"{prefix}-{label}.service") for label, _, _, _ in _job_specs(config)]


def _installed_unit_dir(scope: str) -> Path:
    return Path("/etc/systemd/system") if scope == "system" else Path.home() / ".config" / "systemd" / "user"


def installed_units_are_current(
    config: Dict[str, Any],
    config_path: str | Path,
    *,
    installed_dir: Optional[Path] = None,
) -> Optional[bool]:
    """Whether the installed timer files match what the current config generates.

    Settings such as the interval and the run time limit are written into the
    unit files, so they only take effect after the schedule is started again.
    Returns ``None`` when this cannot be determined.
    """
    scheduling = config.get("scheduling", {}) if isinstance(config.get("scheduling"), dict) else {}
    scope = str(scheduling.get("scope", "system")).lower()
    target = installed_dir if installed_dir is not None else _installed_unit_dir(scope)
    try:
        with tempfile.TemporaryDirectory(prefix="bbx-units-") as scratch:
            result = write_systemd_units(config=config, config_path=config_path, output_dir=scratch)
            for generated in result.written_files:
                installed = target / generated.name
                if not installed.is_file() or installed.read_text() != generated.read_text():
                    return False
    except Exception:
        return None
    return True


def _failure_from_systemd(
    properties: Dict[str, str],
    service: str,
    scope_flag: List[str],
    started: Optional[datetime],
    run: CommandRunner,
) -> FailureExplanation:
    result = properties.get("Result", "")
    if result == "timeout":
        return FailureExplanation(
            "Stopped at the time limit",
            "The run was stopped because it reached its time limit (scheduling.run_timeout_minutes).",
        )
    if result == "oom-kill":
        return FailureExplanation("Out of memory", "The run was stopped because the system ran out of memory.")
    if result in {"signal", "core-dump"}:
        return FailureExplanation("Run was killed", "The run was killed before it finished.")

    exit_status = properties.get("ExecMainStatus", "")
    selectors = [[*scope_flag, "-u", service]]
    if scope_flag:
        selectors.append(["--user-unit", service])
    for selector in selectors:
        journal_command = ["journalctl", *selector, "-o", "cat", "--no-pager", "-n", str(JOURNAL_LINES)]
        if started is not None:
            journal_command += ["--since", started.strftime("%Y-%m-%d %H:%M:%S")]
        code, text = run(journal_command)
        if code != 0:
            continue
        # systemd adds its own lines ("Failed with result 'exit-code'.") around
        # the program's output; they explain nothing, so drop them.
        program_lines = [
            line
            for line in text.splitlines()
            if line.strip()
            and service not in line
            and not line.startswith("-- ")
            and not line.lstrip().startswith(("Started ", "Starting ", "Stopped ", "Failed to start "))
        ]
        if program_lines:
            return explain_failure(
                "\n".join(program_lines),
                exit_code=int(exit_status) if exit_status.lstrip("-").isdigit() else None,
                command=["run-once"],
            )
    code_text = f"exit code {exit_status}" if exit_status else "an error"
    return FailureExplanation(
        "Recording failed",
        f"The run ended with {code_text}. Its messages are in the system log: "
        f"journalctl {' '.join([*scope_flag, '-u', service])}",
    )


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "$(whoami)"


def _user_linger_enabled(run: CommandRunner) -> Optional[bool]:
    """Whether user-scope timers keep running without a login session."""
    code, text = run(["loginctl", "show-user", _current_user(), "-p", "Linger"])
    if code != 0:
        return None
    value = text.strip().partition("=")[2].strip().lower()
    return {"yes": True, "no": False}.get(value)


_RECORDING_LABELS = {"main", "record"}
_JOB_NOUNS = {"main": "recording", "record": "recording", "track": "tracking run"}

_SHOW_PROPERTIES = (
    "Id,LoadState,ActiveState,SubState,UnitFileState,NextElapseUSecRealtime,"
    "Result,ExecMainStatus,ExecMainStartTimestamp"
)


def get_schedule_status(
    config: Dict[str, Any],
    *,
    config_path: Optional[str | Path] = None,
    run: CommandRunner = _default_runner,
    now: Optional[datetime] = None,
    installed_dir: Optional[Path] = None,
) -> ScheduleStatus:
    """Ask systemd whether the recording schedule is on and how the last run went."""
    now = now or datetime.now()
    scheduling = config.get("scheduling", {}) if isinstance(config.get("scheduling"), dict) else {}
    wanted = bool(scheduling.get("enabled", False))
    backend = str(scheduling.get("backend", "systemd")).lower()
    plan = describe_schedule_plan(config)

    if backend != "systemd":
        return ScheduleStatus(
            SCHEDULE_UNKNOWN,
            LEVEL_WARN if wanted else LEVEL_OFF,
            "Not shown here",
            f"This box is scheduled with {backend}, which this window cannot inspect.",
        )

    try:
        names = _unit_names(config)
    except Exception as exc:
        return ScheduleStatus(SCHEDULE_UNKNOWN, LEVEL_WARN, "Unknown", f"The schedule settings could not be read: {exc}")
    if not names:
        return ScheduleStatus(SCHEDULE_OFF, LEVEL_OFF, "Off", "This box has no scheduled jobs in its settings.")

    scope = str(scheduling.get("scope", "system")).lower()
    scope_flag = ["--user"] if scope != "system" else []
    units = [unit for _, timer, service in names for unit in (timer, service)]
    code, text = run(["systemctl", *scope_flag, "show", *units, "-p", _SHOW_PROPERTIES])
    shown = parse_systemctl_show(text) if code is not None else {}
    if not shown:
        why = " ".join(text.split()) or "no answer"
        return ScheduleStatus(
            SCHEDULE_UNKNOWN,
            LEVEL_WARN,
            "Unknown",
            f"Could not ask the system scheduler: {why}",
        )

    timers: List[TimerState] = []
    for label, timer_name, service_name in names:
        timer_props = shown.get(timer_name, {})
        service_props = shown.get(service_name, {})
        state = TimerState(label=label, timer=timer_name, service=service_name)
        state.installed = timer_props.get("LoadState") == "loaded"
        state.enabled = timer_props.get("UnitFileState", "").startswith("enabled")
        state.active = state.installed and timer_props.get("ActiveState") == "active"
        state.next_run = parse_systemd_timestamp(timer_props.get("NextElapseUSecRealtime")) if state.active else None
        state.running = service_props.get("ActiveState") in {"activating", "active", "deactivating"}
        state.last_started = parse_systemd_timestamp(service_props.get("ExecMainStartTimestamp"))
        if state.last_started is not None and not state.running:
            state.last_ok = (
                service_props.get("Result", "success") == "success"
                and service_props.get("ExecMainStatus", "0") in {"0", ""}
                and service_props.get("ActiveState") != "failed"
            )
            if state.last_ok is False:
                state.failure = _failure_from_systemd(service_props, service_name, scope_flag, state.last_started, run)
        timers.append(state)

    active = [timer for timer in timers if timer.active]
    next_run: Optional[datetime] = None
    lines: List[str] = []
    units_current: Optional[bool] = None

    if not active:
        schedule_state, level, headline = SCHEDULE_OFF, LEVEL_OFF, "Off"
        if wanted:
            level = LEVEL_WARN
            lines.append("The settings say the schedule should be on, but it is not running. Press Start Automated Recording.")
        else:
            lines.append("No recordings are scheduled.")
    else:
        if len(active) < len(timers):
            schedule_state, level = SCHEDULE_PARTIAL, LEVEL_FAIL
            missing = ", ".join(timer.label for timer in timers if not timer.active)
            headline = "Partly on"
            lines.append(f"Not running: {missing}. Press Start Automated Recording to start all of it.")
        else:
            schedule_state, level = SCHEDULE_ON, LEVEL_OK
            headline = "On"
        scheduled = [timer for timer in active if timer.next_run is not None]
        if scheduled:
            recording = [timer for timer in scheduled if timer.label in _RECORDING_LABELS]
            first = min(recording or scheduled, key=lambda timer: timer.next_run)
            next_run = first.next_run
            what = _JOB_NOUNS.get(first.label, "run")
            headline += f" · next {what} {describe_moment(next_run, now)} ({describe_relative(next_run, now)})"
        lines.append(plan)
        if any(not timer.enabled for timer in active):
            level = LEVEL_WARN if level == LEVEL_OK else level
            lines.append("It will not come back after a restart. Press Start Automated Recording to fix this.")
        if scope != "system" and _user_linger_enabled(run) is False:
            level = LEVEL_WARN if level == LEVEL_OK else level
            lines.append(
                "Recording stops when this user logs out, and after a restart it waits for a login. "
                f"To keep it running, run once in a terminal: loginctl enable-linger {_current_user()}"
            )
        if config_path is not None:
            units_current = installed_units_are_current(config, config_path, installed_dir=installed_dir)
            if units_current is False:
                level = LEVEL_WARN if level == LEVEL_OK else level
                lines.append(
                    "The settings have changed since the schedule was started. "
                    "Press Start Automated Recording to apply them."
                )

    reported = active or timers
    running = [timer for timer in reported if timer.running]
    failed = [timer for timer in reported if timer.last_ok is False]
    finished = [timer for timer in reported if timer.last_started is not None and not timer.running]
    if running:
        started = running[0].last_started
        when = f" (started {describe_moment(started, now)})" if started is not None else ""
        lines.append(f"A scheduled run is in progress{when}.")
    if failed:
        worst = max(failed, key=lambda timer: timer.last_started or datetime.min)
        level = LEVEL_FAIL
        failure = worst.failure or FailureExplanation("Recording failed", "No reason was recorded.")
        when = describe_moment(worst.last_started, now) if worst.last_started else "unknown time"
        lines.append(f"Last scheduled run ({when}) FAILED: {failure.title}.\n{failure.reason}")
    elif finished:
        latest = max(finished, key=lambda timer: timer.last_started or datetime.min)
        lines.append(f"Last scheduled run ({describe_moment(latest.last_started, now)}) completed.")
    elif active and not running:
        lines.append("No scheduled run has happened yet since the schedule was started or this computer restarted.")

    return ScheduleStatus(
        state=schedule_state,
        level=level,
        headline=headline,
        detail="\n".join(lines),
        next_run=next_run,
        timers=timers,
        units_current=units_current,
    )


# ---------------------------------------------------------------------------
# Review before the schedule is started
# ---------------------------------------------------------------------------


@dataclass
class StartReview:
    problems: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def review_schedule_start(
    config: Dict[str, Any],
    *,
    readiness: Optional[StorageReadiness] = None,
    schedule_check: Callable[[Dict[str, Any]], Any] = run_schedule_check,
) -> StartReview:
    """List reasons the schedule as configured is unlikely to work.

    The estimates behind the schedule check are rules of thumb, so the caller
    should let the user start anyway after showing these.
    """
    review = StartReview()
    storage = storage_status(config, readiness=readiness)
    if storage.level == LEVEL_FAIL:
        review.problems.append(f"{storage.headline}. {storage.detail}")

    try:
        report = schedule_check(config)
    except Exception as exc:
        review.problems.append(f"The schedule could not be checked: {exc}")
        return review
    failures = [item for item in report.items if item.status == "FAIL"]
    for item in failures:
        review.problems.append(f"{item.name}: {item.message}")
    if failures:
        review.suggestions.extend(
            str(suggestion) for suggestion in report.suggestions if "rerun schedule-check" not in str(suggestion)
        )
    return review


def format_start_review(review: StartReview, *, max_suggestions: int = 3) -> str:
    lines = [f"• {problem}" for problem in review.problems]
    if review.suggestions:
        lines.append("")
        lines.append("What would help:")
        lines.extend(f"• {suggestion}" for suggestion in review.suggestions[:max_suggestions])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------


@dataclass
class BoxStatus:
    storage: StatusLine
    schedule: ScheduleStatus
    checked_at: datetime


def collect_box_status(config: Dict[str, Any], config_path: Optional[str | Path] = None) -> BoxStatus:
    """Gather the storage and schedule status. Safe to call from a worker thread."""
    try:
        storage = storage_status(config)
    except Exception as exc:
        storage = StatusLine(LEVEL_WARN, "Storage could not be checked", str(exc))
    try:
        schedule = get_schedule_status(config, config_path=config_path)
    except Exception as exc:
        schedule = ScheduleStatus(SCHEDULE_UNKNOWN, LEVEL_WARN, "Unknown", f"The schedule could not be checked: {exc}")
    return BoxStatus(storage=storage, schedule=schedule, checked_at=datetime.now())
