"""Progress signals from a recording to the watchdog that supervises it.

The watchdog (``recording_watchdog``) starts the recorder as a child process
and hands it one end of a pipe. The recorder writes a byte whenever it makes
progress. No bytes for a while means it has stalled, however it got stuck.
A pipe is used rather than a file so that nothing is written to the SD card.

When no watchdog is running these functions do nothing.
"""

from __future__ import annotations

import os
import time
from typing import Optional

HEARTBEAT_FD_ENV = "BUMBLEBOX_HEARTBEAT_FD"
BEAT = b"."
CAPTURE_FINISHED = b"C"
MIN_SECONDS_BETWEEN_BEATS = 0.5

_fd: Optional[int] = None
_looked = False
_last_beat = 0.0


def _pipe() -> Optional[int]:
    global _fd, _looked
    if not _looked:
        _looked = True
        try:
            fd = int(os.environ.get(HEARTBEAT_FD_ENV, ""))
            os.set_blocking(fd, False)  # a full pipe must never hold up the recorder
            _fd = fd
        except (ValueError, OSError):
            _fd = None
    return _fd


def _send(payload: bytes) -> None:
    global _fd
    fd = _pipe()
    if fd is None:
        return
    try:
        os.write(fd, payload)
    except BlockingIOError:
        pass  # the watchdog has plenty of unread beats already
    except OSError:
        _fd = None  # the watchdog is gone; carry on recording without it


def beat() -> None:
    """Say "still making progress". Cheap enough to call for every frame."""
    global _last_beat
    now = time.monotonic()
    if now - _last_beat < MIN_SECONDS_BETWEEN_BEATS:
        return
    _last_beat = now
    _send(BEAT)


def capture_finished() -> None:
    """Say that capture is over. From here a failure cannot be fixed by recording again."""
    _send(CAPTURE_FINISHED)
