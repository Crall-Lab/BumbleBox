"""Decide whether a folder is safe to record into, before any data is written.

A recording Pi normally writes to an external drive mounted at
``system.data_root``. If that drive is missing, the mount point is still an
ordinary folder on the SD card, and recording into it silently fills the
system disk while the operator believes data is going to the drive.

``check_storage_ready`` answers one question: may a recording start now?

- If ``/etc/fstab`` lists a drive for the data folder (or a folder above it),
  that drive must be mounted.
- If the data folder is under ``/mnt`` or ``/media``, where external drives
  are mounted, it must be on a mounted drive and not on the system disk.
- If ``system.require_mounted_storage`` is true, the data folder must be on a
  mounted drive wherever it is. If it is false, none of this is checked.
- The drive must have at least ``system.min_free_space_gb`` free.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REQUIRE_AUTO = "auto"
DEFAULT_MIN_FREE_SPACE_GB = 1.0
FSTAB_PATH = Path("/etc/fstab")
# Folders where external drives are conventionally mounted. A data folder
# under one of these that is on the system disk almost always means the drive
# is missing.
EXTERNAL_DRIVE_ROOTS = ("/mnt/", "/media/", "/run/media/")


class StorageNotReadyError(RuntimeError):
    """Raised when recording is refused because the data folder is not safe to write to."""


@dataclass
class StorageReadiness:
    path: str
    ready: bool
    # "required" (config says so), "expected_by_fstab", "expected_by_location"
    # (the folder is under /mnt or /media), or "not_required"
    requirement: str
    expected_mount_point: Optional[str] = None
    mounted: Optional[bool] = None
    free_gb: Optional[float] = None
    min_free_gb: float = DEFAULT_MIN_FREE_SPACE_GB
    problem: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def normalize_require_mounted_storage(value: Any) -> Any:
    """Return True, False or "auto" for a config value; anything else is returned unchanged."""
    if isinstance(value, bool):
        return value
    if value is None:
        return REQUIRE_AUTO
    text = str(value).strip().lower()
    if text in {"auto", ""}:
        return REQUIRE_AUTO
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0"}:
        return False
    return value


def read_fstab_text() -> str:
    try:
        return FSTAB_PATH.read_text()
    except Exception:
        return ""


def _normalize(path: Any) -> str:
    return os.path.abspath(os.path.expanduser(str(path).strip()))


def fstab_mount_points(fstab_text: str) -> List[str]:
    """Mount points listed in fstab, ignoring comments, swap and the root filesystem."""
    points: List[str] = []
    for raw in str(fstab_text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        mount_point = fields[1].replace("\\040", " ")
        if not mount_point.startswith("/") or mount_point == "/":
            continue
        points.append(_normalize(mount_point))
    return points


def expected_mount_point(path: Any, fstab_text: str) -> Optional[str]:
    """The fstab mount point that ``path`` lives on, if fstab lists one.

    Matches the path itself or any folder above it, and returns the deepest
    match. The root filesystem never counts.
    """
    target = _normalize(path)
    best: Optional[str] = None
    for mount_point in fstab_mount_points(fstab_text):
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if best is None or len(mount_point) > len(best):
                best = mount_point
    return best


def containing_mount_point(path: Any, *, ismount: Callable[[str], bool] = os.path.ismount) -> str:
    """The mount point of the filesystem that ``path`` is (or would be) stored on."""
    current = _normalize(path)
    while True:
        if ismount(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return current
        current = parent


def _nearest_existing(path: str) -> str:
    current = path
    while not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


def check_storage_ready(
    config: Dict[str, Any],
    *,
    path: Optional[str | Path] = None,
    fstab_text: Optional[str] = None,
    ismount: Callable[[str], bool] = os.path.ismount,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> StorageReadiness:
    """Check whether recording into ``path`` (default ``system.data_root``) may start."""
    system = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    raw_path = str(path if path is not None else system.get("data_root") or "").strip()
    setting = normalize_require_mounted_storage(system.get("require_mounted_storage", REQUIRE_AUTO))
    try:
        min_free_gb = max(0.0, float(system.get("min_free_space_gb", DEFAULT_MIN_FREE_SPACE_GB)))
    except (TypeError, ValueError):
        min_free_gb = DEFAULT_MIN_FREE_SPACE_GB

    if not raw_path:
        return StorageReadiness(
            path="",
            ready=False,
            requirement="not_required",
            min_free_gb=min_free_gb,
            problem="system.data_root is not set, so there is nowhere to record to.",
        )

    target = _normalize(raw_path)
    expected = expected_mount_point(target, read_fstab_text() if fstab_text is None else fstab_text)
    readiness = StorageReadiness(path=target, ready=True, requirement="not_required", min_free_gb=min_free_gb)
    readiness.expected_mount_point = expected

    if setting is False:
        if expected is not None and not ismount(expected):
            readiness.notes.append(
                f"/etc/fstab lists a drive for {expected} but it is not mounted; recording to the "
                "system disk is allowed because system.require_mounted_storage is false."
            )
    elif expected is not None:
        readiness.requirement = "required" if setting is True else "expected_by_fstab"
        readiness.mounted = bool(ismount(expected))
        if not readiness.mounted:
            readiness.problem = (
                f"The data drive is not mounted: /etc/fstab lists a drive for {expected}, but nothing "
                f"is mounted there. Recording was refused so that data is not written to the system "
                f"disk under {target}. Check that the drive is connected, then run "
                "`bbx storage status`. To record to the system disk anyway, set "
                "system.require_mounted_storage: false."
            )
    elif setting is True or target.startswith(EXTERNAL_DRIVE_ROOTS):
        readiness.requirement = "required" if setting is True else "expected_by_location"
        mount_point = containing_mount_point(target, ismount=ismount)
        readiness.mounted = mount_point != "/"
        if not readiness.mounted:
            why = (
                "system.require_mounted_storage is true"
                if setting is True
                else f"{target} is in a folder where external drives are mounted"
            )
            readiness.problem = (
                f"The data drive is not mounted: {why}, but {target} is on the system disk (no "
                "drive is mounted at that folder or above it). Recording was refused so that data "
                "is not written to the system disk. Connect and mount the data drive "
                "(`bbx storage setup`), or set system.require_mounted_storage: false to record to "
                "the system disk on purpose."
            )

    if readiness.problem is None and min_free_gb > 0:
        try:
            free_bytes = int(disk_usage(_nearest_existing(target)).free)
        except Exception:
            free_bytes = None
        if free_bytes is not None:
            readiness.free_gb = round(free_bytes / 1e9, 2)
            if free_bytes < min_free_gb * 1e9:
                readiness.problem = (
                    f"Only {readiness.free_gb:g} GB is free where {target} is stored; at least "
                    f"{min_free_gb:g} GB is required to start a recording (system.min_free_space_gb). "
                    "Free some space or move older recordings off the drive."
                )

    readiness.ready = readiness.problem is None
    return readiness


def ensure_storage_ready(config: Dict[str, Any], *, path: Optional[str | Path] = None) -> StorageReadiness:
    """Raise ``StorageNotReadyError`` unless recording into ``path`` may start."""
    readiness = check_storage_ready(config, path=path)
    if not readiness.ready:
        raise StorageNotReadyError(readiness.problem or "Storage is not ready for recording.")
    return readiness


def format_storage_readiness(readiness: StorageReadiness) -> str:
    if readiness.ready:
        detail = {
            "required": "a mounted drive is required and is present",
            "expected_by_fstab": f"the drive listed in /etc/fstab for {readiness.expected_mount_point} is mounted",
            "expected_by_location": "the folder is on a mounted drive",
            "not_required": "no external drive is expected for this folder",
        }.get(readiness.requirement, readiness.requirement)
        free = f"; {readiness.free_gb:g} GB free" if readiness.free_gb is not None else ""
        lines = [f"Recording allowed: yes ({detail}{free})"]
    else:
        lines = ["Recording allowed: NO", f"Reason: {readiness.problem}"]
    lines.extend(f"Note: {note}" for note in readiness.notes)
    return "\n".join(lines)
