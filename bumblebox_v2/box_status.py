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
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .schedule_check import run_schedule_check
from .storage_guard import StorageReadiness, check_storage_ready, containing_mount_point
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
            )
        return StatusLine(LEVEL_FAIL, "Recording is refused", readiness.problem or path)

    on_separate_drive = readiness.requirement != "not_required"
    if not on_separate_drive:
        try:
            on_separate_drive = containing_mount_point(path, ismount=ismount) != "/"
        except Exception:
            on_separate_drive = False
    where = "Drive mounted" if on_separate_drive else "On the system disk"
    headline = f"{where} · {_format_gb(free_gb)} free" if free_gb is not None else where
    if readiness.notes:
        return StatusLine(LEVEL_WARN, headline, f"{path}. {readiness.notes[0]}")
    if free_gb is not None and free_gb < LOW_FREE_SPACE_WARNING_GB:
        return StatusLine(LEVEL_WARN, headline, f"{path}. Space is getting low.")
    return StatusLine(LEVEL_OK, headline, path)


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
