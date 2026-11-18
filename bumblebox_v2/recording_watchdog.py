"""Keeps a long recording going when the recorder crashes or stalls.

One long recording has a weakness that many short ones do not: when it fails,
nothing else is due to start for a long time. The watchdog closes that gap.
It runs the recorder as a child process and watches two things:

* the child exiting with an error before capture is finished;
* the child going quiet. The recorder sends a signal for every frame and
  every progress message (see ``heartbeat``); a camera that stops delivering
  frames, or an encoder that stops accepting them, shows up as silence.

In either case the child is stopped and a new recording is started for the
time left in the session, after a short pause. Each attempt is an ordinary
recording with its own session folder and summary, so a restart is visible
afterwards as a failed run followed by a resumed one.

Limits, so that a camera which is truly broken cannot spin the Pi in a loop:
a pause before each restart that grows, a maximum number of restarts, and no
restart when too little of the session is left to be worth it.

Once capture has finished the watchdog only waits. A failure after that
point (encoding, tracking) cannot be fixed by recording again, and tracking
may legitimately be slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import select
import signal
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .heartbeat import CAPTURE_FINISHED, HEARTBEAT_FD_ENV
from .time_units import describe_seconds

AUTO_MINIMUM_RECORDING_SECONDS = 120.0
DEFAULT_STALL_SECONDS = 120.0
DEFAULT_MAX_RESTARTS = 5
MIN_RESUME_SECONDS = 30.0
RESTART_DELAYS = (5.0, 15.0, 30.0, 60.0)
INTERRUPT_GRACE_SECONDS = 10.0
TERMINATE_GRACE_SECONDS = 5.0
POLL_SECONDS = 0.5


def restart_enabled(config: Dict[str, Any]) -> bool:
    """Whether this config's recording should run under the watchdog.

    ``capture.restart_on_failure``: ``true``, ``false`` or ``auto``. Auto means
    recordings of two minutes or more; a short burst has nothing left to
    resume by the time it could be restarted.
    """
    capture = config.get("capture", {}) if isinstance(config.get("capture"), dict) else {}
    setting = capture.get("restart_on_failure", "auto")
    if isinstance(setting, bool):
        return setting
    if str(setting).strip().lower() != "auto":
        return False
    try:
        return float(capture.get("recording_seconds", 0)) >= AUTO_MINIMUM_RECORDING_SECONDS
    except (TypeError, ValueError):
        return False


def stall_seconds(config: Dict[str, Any]) -> float:
    capture = config.get("capture", {}) if isinstance(config.get("capture"), dict) else {}
    try:
        return max(10.0, float(capture.get("stall_seconds", DEFAULT_STALL_SECONDS)))
    except (TypeError, ValueError):
        return DEFAULT_STALL_SECONDS


def max_restarts(config: Dict[str, Any]) -> int:
    capture = config.get("capture", {}) if isinstance(config.get("capture"), dict) else {}
    try:
        return max(0, int(capture.get("max_restarts", DEFAULT_MAX_RESTARTS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESTARTS


@dataclass
class Attempt:
    number: int
    seconds_requested: float
    exit_code: Optional[int] = None
    stalled: bool = False
    capture_finished: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.stalled


@dataclass
class WatchdogResult:
    attempts: List[Attempt] = field(default_factory=list)
    interrupted: bool = False
    gave_up: Optional[str] = None  # why no further restart was made, if one was wanted
    seconds_not_recorded: float = 0.0

    @property
    def restarts(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def exit_code(self) -> int:
        if self.interrupted:
            return 130
        # A session that needed a restart has a gap in it; report that as a
        # failure even when the resumed recording went well.
        return 0 if len(self.attempts) == 1 and self.attempts[0].ok else 1


def _how_it_ended(exit_code: Optional[int]) -> str:
    """'stopped with exit code 1', or 'was killed (signal 9)' for a negative code."""
    if exit_code is not None and exit_code < 0:
        return f"was killed (signal {-exit_code})"
    return f"stopped with exit code {exit_code}"


def _stop(process: Any, say: Callable[[str], None]) -> None:
    """Stop a child that has stalled: ask, then insist."""
    for signal_number, grace in ((signal.SIGINT, INTERRUPT_GRACE_SECONDS), (signal.SIGTERM, TERMINATE_GRACE_SECONDS)):
        try:
            process.send_signal(signal_number)
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
        except OSError:
            return
    say("The recorder did not stop when asked; killing it.")
    try:
        process.kill()
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except Exception:
        pass


def _run_attempt(
    command: Sequence[str],
    attempt: Attempt,
    stall_after: float,
    say: Callable[[str], None],
    popen: Callable[..., Any],
    clock: Callable[[], float],
) -> None:
    read_end, write_end = os.pipe()
    environment = dict(os.environ)
    environment[HEARTBEAT_FD_ENV] = str(write_end)
    try:
        process = popen(list(command), pass_fds=(write_end,), env=environment)
    finally:
        os.close(write_end)
    os.set_blocking(read_end, False)
    last_signal = clock()
    try:
        while True:
            readable, _, _ = select.select([read_end], [], [], POLL_SECONDS)
            if readable:
                try:
                    data = os.read(read_end, 4096)
                except BlockingIOError:
                    data = b""
                if data:
                    last_signal = clock()
                    if CAPTURE_FINISHED in data:
                        attempt.capture_finished = True
            if process.poll() is not None:
                break
            if not attempt.capture_finished and clock() - last_signal > stall_after:
                attempt.stalled = True
                say(
                    f"No progress from the recorder for {describe_seconds(round(stall_after))}; "
                    "it has stalled and is being stopped."
                )
                _stop(process, say)
                break
        attempt.exit_code = process.wait()
    except KeyboardInterrupt:
        # Ctrl+C, or Cancel in the GUI: pass it on so the recording stops
        # cleanly and writes its summary, then let the caller stop too.
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=INTERRUPT_GRACE_SECONDS + TERMINATE_GRACE_SECONDS)
        except Exception:
            process.kill()
        attempt.exit_code = process.poll()
        raise
    finally:
        os.close(read_end)


def supervise_recording(
    build_command: Callable[[float, int], Sequence[str]],
    *,
    total_seconds: float,
    stall_after: float = DEFAULT_STALL_SECONDS,
    restarts_allowed: int = DEFAULT_MAX_RESTARTS,
    min_resume_seconds: float = MIN_RESUME_SECONDS,
    restart_delays: Sequence[float] = RESTART_DELAYS,
    say: Optional[Callable[[str], None]] = None,
    popen: Callable[..., Any] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WatchdogResult:
    """Record for ``total_seconds``, starting again if the recorder fails or stalls.

    ``build_command(seconds, attempt_number)`` returns the command for a
    recording of ``seconds``; the first attempt gets the whole session.
    """
    if say is None:
        say = lambda message: print(f"[watchdog] {message}", flush=True)  # noqa: E731
    result = WatchdogResult()
    session_ends = clock() + float(total_seconds)
    seconds = float(total_seconds)

    while True:
        attempt = Attempt(number=len(result.attempts) + 1, seconds_requested=seconds)
        result.attempts.append(attempt)
        try:
            _run_attempt(build_command(seconds, attempt.number), attempt, stall_after, say, popen, clock)
        except KeyboardInterrupt:
            result.interrupted = True
            return result
        if attempt.ok:
            return result
        if attempt.capture_finished:
            # The recording itself is complete; what failed came after it.
            return result

        why = "stalled" if attempt.stalled else _how_it_ended(attempt.exit_code)
        delay = restart_delays[min(result.restarts, len(restart_delays) - 1)] if restart_delays else 0.0
        remaining = session_ends - clock() - delay
        if result.restarts >= restarts_allowed:
            result.gave_up = f"it had already been restarted {result.restarts} time(s)"
        elif remaining < min_resume_seconds:
            result.gave_up = "too little of the session was left to record"
        if result.gave_up:
            result.seconds_not_recorded = max(0.0, session_ends - clock())
            say(f"The recorder {why}. Not restarting: {result.gave_up}.")
            return result

        say(
            f"The recorder {why}. Starting again in {describe_seconds(delay)} to record the "
            f"{describe_seconds(round(remaining))} left of this session "
            f"(restart {result.restarts + 1} of at most {restarts_allowed})."
        )
        try:
            sleep(delay)
        except KeyboardInterrupt:
            result.interrupted = True
            return result
        seconds = max(1.0, session_ends - clock())


def describe_result(result: WatchdogResult, total_seconds: float) -> Optional[str]:
    """One line for the end of the output when the session did not go cleanly."""
    if result.interrupted or (len(result.attempts) == 1 and result.attempts[0].ok):
        return None
    first = result.attempts[0]
    if first.capture_finished and not first.stalled:
        return None  # the recorder's own output already says what failed after capture
    cause = "the recorder stalled" if first.stalled else f"the recorder {_how_it_ended(first.exit_code)}"
    last = result.attempts[-1]
    if result.gave_up:
        ending = f"It was restarted {result.restarts} time(s) and then left, because {result.gave_up}."
    elif last.ok:
        ending = (
            f"It was restarted {result.restarts} time(s) and the rest of the "
            f"{describe_seconds(total_seconds)} session was recorded as a separate run."
        )
    else:
        ending = f"It was restarted {result.restarts} time(s); the last attempt also failed."
    return f"Recording failed part-way: {cause}. {ending} See Results for each run and what it kept."
