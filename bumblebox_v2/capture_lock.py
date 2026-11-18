"""One capture at a time per machine.

Scheduled runs can coincide (for example the record and track timers of a
mixed schedule both fire on the hour), and only one process can use the camera.
``capture_lock`` is held while a run is capturing. A second run waits for it
for a limited time and then gives up with a clear error, instead of failing
inside the camera driver.

The lock is an advisory file lock, so it is released automatically if the
process holding it exits or is killed; it cannot be left stale.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

try:  # not available on Windows, where the lock is skipped
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

LOCK_FILE_NAME = "bumblebox-capture.lock"
LOCK_PATH_ENV = "BUMBLEBOX_CAPTURE_LOCK"
DEFAULT_LOCK_WAIT_SECONDS = 60.0


class CaptureBusyError(RuntimeError):
    """Raised when another capture kept the camera for longer than the allowed wait."""


def default_lock_path(suffix: str = "") -> Path:
    """Where the lock file lives.

    It is deliberately not on the data drive: the lock must work when that
    drive is missing, and must be shared by every user that can start a run.

    ``suffix`` names a second camera (for example ``"cam1"``). Each camera has
    its own lock, so two cameras can record at once while two runs on one
    camera still cannot.
    """
    override = os.environ.get(LOCK_PATH_ENV, "").strip()
    if override:
        base = Path(override).expanduser()
    else:
        run_lock = Path("/run/lock")
        directory = run_lock if run_lock.is_dir() and os.access(run_lock, os.W_OK) else Path(tempfile.gettempdir())
        base = directory / LOCK_FILE_NAME
    if not suffix:
        return base
    return base.with_name(f"{base.stem}-{suffix}{base.suffix}")


def _open_lock_file(path: Path) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)  # never follow a symlink planted in a shared folder
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT | no_follow, 0o666)
        try:
            os.chmod(str(path), 0o666)  # let other users (system and user services) share the lock
        except OSError:
            pass
        return fd
    except PermissionError:
        # Created by another user with a restrictive mode: a read-only handle can still be locked.
        return os.open(str(path), os.O_RDONLY | no_follow)


def _read_holder(fd: int) -> str:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        text = os.read(fd, 200).decode("utf-8", errors="replace").strip()
    except OSError:
        text = ""
    return text or "holder unknown"


def _write_holder(fd: int) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}, since {datetime.now().isoformat(timespec='seconds')}\n".encode("utf-8"))
    except OSError:
        pass  # read-only handle; the lock itself still works


@contextmanager
def capture_lock(
    wait_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    *,
    lock_path: Optional[str | Path] = None,
    on_wait: Optional[Callable[[str], None]] = None,
    poll_seconds: float = 0.2,
) -> Iterator[None]:
    """Hold the machine-wide capture lock for the duration of the ``with`` block.

    Waits up to ``wait_seconds`` for another capture to finish, then raises
    ``CaptureBusyError``. ``on_wait`` is called once, with a description of the
    current holder, if the lock is not free straight away.
    """
    if fcntl is None:  # pragma: no cover - platform dependent
        yield
        return

    path = Path(lock_path).expanduser() if lock_path else default_lock_path()
    fd = _open_lock_file(path)
    try:
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        announced = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                holder = _read_holder(fd)
                if not announced and on_wait is not None:
                    on_wait(holder)
                announced = True
                if time.monotonic() >= deadline:
                    raise CaptureBusyError(
                        "Another BumbleBox capture is using the camera "
                        f"({holder}) and did not finish within {float(wait_seconds):g}s, so this run "
                        "was skipped. If runs often overlap, space the schedule out or shorten the "
                        "recording (see `bbx schedule-check`)."
                    ) from None
                time.sleep(max(0.01, float(poll_seconds)))
        _write_holder(fd)
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)
