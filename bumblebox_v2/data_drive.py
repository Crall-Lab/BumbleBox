"""What the Data Drive window needs to know and do, without any Qt.

- which connected drives could hold the recordings;
- what is mounted at the data folder now, and whether it comes back after a restart;
- how to run the storage commands, which need administrator rights;
- the two config values that record the operator's choice.

Mounting itself is done by ``bbx storage ...`` (see ``storage_manager``), run
as root through ``sudo`` or ``pkexec``. This module never changes the system.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .camera_select import capture_lock_suffix
from .capture_lock import default_lock_path
from .config import load_config, save_config
from .storage_guard import (
    EXTERNAL_DRIVE_ROOTS,
    StorageReadiness,
    check_storage_ready,
    containing_mount_point,
)
from .storage_manager import (
    StorageDevice,
    _fstab_entry_for_mount,
    _findmnt_target,
    _NON_DATA_FSTYPES,
    _read_fstab,
    discover_storage_devices,
    fstab_source_uuid,
)

try:  # not available on Windows
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
BBX_SCRIPT = REPO_ROOT / "bbx.py"

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_FAIL = "fail"

# The actions the window can ask for. The first two mount a drive; the rest
# act on the drive that is mounted now.
ACTION_MOUNT_KEEP = "mount_keep"
ACTION_MOUNT_ONCE = "mount_once"
ACTION_KEEP = "keep"
ACTION_FORGET = "forget"
ACTION_UNMOUNT = "unmount"

_EFI_SYSTEM_PARTITION = "c12a7328-f81f-11d2-ba4b-00a0c93ec93b"
_FILESYSTEM_NAMES = {
    "exfat": "exFAT",
    "vfat": "FAT32",
    "ntfs": "NTFS",
    "fuseblk": "NTFS or exFAT",
    "ext4": "ext4",
    "ext3": "ext3",
    "ext2": "ext2",
    "btrfs": "Btrfs",
    "xfs": "XFS",
    "f2fs": "F2FS",
}
_TRANSPORT_NAMES = {"usb": "USB", "nvme": "NVMe", "sata": "SATA", "mmc": "SD card"}
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*([KMGTP]?)", re.IGNORECASE)
# Folders a data drive may be mounted at, besides the operator's home folder.
_MOUNT_ROOTS = ("/mnt", "/media", "/run/media", "/srv")
_UNSAFE_FOLDER_CHARACTERS = set("\\#\"'`$\n\r\t")
HIDDEN_ITEMS_LIMIT = 1000


def _normalize(path: Any) -> str:
    return os.path.abspath(os.path.expanduser(str(path).strip()))


def _is_inside(path: str, parent: str) -> bool:
    """Whether ``path`` is ``parent`` or somewhere below it."""
    parent = parent.rstrip("/") or "/"
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def size_in_bytes(size_text: str) -> Optional[int]:
    """Bytes for a size as ``lsblk`` prints it (``931.5G``; its units are powers of 1024)."""
    match = _SIZE_RE.match(str(size_text or ""))
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    power = "KMGTP".find(match.group(2).upper()) + 1 if match.group(2) else 0
    return int(number * (1024**power))


def format_size(value: Optional[float]) -> str:
    """Decimal units, as drives are sold: 1 GB = 1e9 bytes."""
    if value is None:
        return "size unknown"
    value = max(0.0, float(value))
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if value >= size * 0.9995:  # 999.97 GB reads as 1.0 TB, not 1000 GB
            scaled = value / size
            return f"{scaled:.0f} {unit}" if scaled >= 10 else f"{scaled:.1f} {unit}"
    return f"{value:.0f} B"


def filesystem_name(fstype: str) -> str:
    return _FILESYSTEM_NAMES.get(str(fstype).lower(), str(fstype) or "unknown format")


def describe_drive(device: StorageDevice) -> str:
    """One line a person can match to the drive in their hand."""
    parts: List[str] = []
    model = device.model.strip()
    transport = _TRANSPORT_NAMES.get(device.transport, "")
    if model and transport:
        parts.append(f"{model} ({transport})")
    elif model or transport:
        parts.append(model or f"{transport} drive")
    parts.append(format_size(size_in_bytes(device.size)))
    parts.append(filesystem_name(device.fstype))
    parts.append(device.path)
    return " · ".join(parts)


@dataclass
class DriveChoice:
    device: StorageDevice
    title: str
    detail: str

    @property
    def path(self) -> str:
        return self.device.path


def _is_external(device: StorageDevice) -> bool:
    return device.hotplug or device.removable or device.transport == "usb"


def usable_drives(devices: Sequence[StorageDevice], *, mount_folder: str = "") -> List[DriveChoice]:
    """The connected drives that could be mounted as the data drive.

    Left out: anything without a filesystem, system partitions (the root and
    boot partitions, an EFI partition, swap), and drives mounted somewhere
    other than where a desktop puts plugged-in drives, since those are in use
    by the system.
    """
    target = _normalize(mount_folder) if mount_folder else ""
    with_partitions = {device.parent for device in devices if device.parent}
    choices: List[DriveChoice] = []
    for device in devices:
        if device.dev_type not in {"part", "disk"} or not device.path.startswith("/dev/"):
            continue
        if device.dev_type == "disk" and device.name in with_partitions:
            continue
        if not device.has_filesystem or device.fstype.lower() in _NON_DATA_FSTYPES:
            continue
        if device.part_type == _EFI_SYSTEM_PARTITION:
            continue
        mounted_at = device.mountpoint.strip()
        if mounted_at and mounted_at.startswith("["):  # [SWAP]
            continue
        if mounted_at:
            normalized = _normalize(mounted_at)
            plugged_in_place = (normalized + "/").startswith(EXTERNAL_DRIVE_ROOTS)
            if not plugged_in_place and normalized != target:
                continue
        title = f"“{device.label}”" if device.label else (device.model or device.name or device.path)
        detail = describe_drive(device)
        if mounted_at and _normalize(mounted_at) != target:
            detail += f" · now at {mounted_at}"
        choices.append(DriveChoice(device=device, title=title, detail=detail))
    choices.sort(key=lambda choice: (0 if _is_external(choice.device) else 1, choice.device.path))
    return choices


@dataclass
class DataDriveState:
    data_folder: str
    # Where the drive is mounted when one is; otherwise the data folder.
    mount_folder: str
    mounted: bool = False
    drive: Optional[StorageDevice] = None
    source: str = ""
    writable: bool = False
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    # The /etc/fstab line that mounts a drive at mount_folder at startup, if any.
    startup_entry: Optional[str] = None
    startup_uuid: Optional[str] = None
    choices: List[DriveChoice] = field(default_factory=list)
    readiness: Optional[StorageReadiness] = None

    @property
    def kept_at_startup(self) -> bool:
        return bool(self.startup_entry)

    @property
    def startup_is_this_drive(self) -> bool:
        """Startup is set, and for the drive that is mounted now."""
        if not self.startup_entry:
            return False
        if self.drive is None or not self.startup_uuid:
            return True  # cannot tell them apart; do not raise a false alarm
        return self.startup_uuid == self.drive.uuid

    @property
    def startup_drive_connected(self) -> bool:
        return bool(self.startup_uuid) and any(
            choice.device.uuid == self.startup_uuid for choice in self.choices
        )

    @property
    def preferred_choice(self) -> Optional[DriveChoice]:
        """The drive to preselect: the one startup expects, or the only one there is."""
        if self.startup_uuid:
            for choice in self.choices:
                if choice.device.uuid == self.startup_uuid:
                    return choice
        return self.choices[0] if len(self.choices) == 1 else None


def _drive_mount_folder(data_folder: str, ismount: Callable[[str], bool]) -> Optional[str]:
    """The folder a data drive is mounted at for ``data_folder``, if one is.

    A drive counts when it is mounted at the data folder itself, or at a
    folder above it in a place where drives are plugged in. A data folder in,
    say, a separate /home partition is on the system's own disk.
    """
    if ismount(data_folder) and data_folder != "/":
        return data_folder
    containing = containing_mount_point(data_folder, ismount=ismount)
    if containing != "/" and (containing + "/").startswith(EXTERNAL_DRIVE_ROOTS):
        return containing
    return None


def read_data_drive_state(
    config: Dict[str, Any],
    *,
    devices: Optional[Sequence[StorageDevice]] = None,
    fstab_text: Optional[str] = None,
    ismount: Callable[[str], bool] = os.path.ismount,
    mount_source: Callable[[Path], Optional[Tuple[str, str]]] = _findmnt_target,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
    can_write: Callable[[str], bool] = lambda folder: os.access(folder, os.W_OK),
) -> DataDriveState:
    """Look at the data folder: is a drive mounted there, which one, and will it return after a restart."""
    system = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    data_folder = _normalize(system.get("data_root") or "/mnt/bumblebox/data")
    if devices is None:
        devices = discover_storage_devices()
    if fstab_text is None:
        fstab_text = _read_fstab()

    mount_folder = _drive_mount_folder(data_folder, ismount)
    state = DataDriveState(data_folder=data_folder, mount_folder=mount_folder or data_folder)
    state.readiness = check_storage_ready(config, fstab_text=fstab_text, ismount=ismount, disk_usage=disk_usage)
    state.startup_entry = _fstab_entry_for_mount(state.mount_folder, fstab_text)
    state.startup_uuid = fstab_source_uuid(state.startup_entry)
    state.choices = usable_drives(devices, mount_folder=state.mount_folder)

    if mount_folder is None:
        return state

    state.mounted = True
    found = mount_source(Path(mount_folder))
    state.source = found[0] if found else ""
    for device in devices:
        if state.source and state.source in {device.path, f"UUID={device.uuid}", f"/dev/disk/by-uuid/{device.uuid}"}:
            state.drive = device
            break
    if state.drive is None:
        for device in devices:
            if device.mountpoint and _normalize(device.mountpoint) == mount_folder:
                state.drive = device
                break
    state.writable = bool(can_write(mount_folder))
    try:
        usage = disk_usage(mount_folder)
        state.free_bytes, state.total_bytes = int(usage.free), int(usage.total)
    except Exception:
        pass
    return state


@dataclass
class DriveSummary:
    level: str
    headline: str
    detail: str


def summarize(state: DataDriveState) -> DriveSummary:
    """The two lines at the top of the window."""
    if state.mounted:
        lines: List[str] = []
        if state.drive is not None:
            name = f"“{state.drive.label}” · " if state.drive.label else ""
            lines.append(name + describe_drive(state.drive))
        elif state.source:
            lines.append(state.source)
        if state.free_bytes is not None and state.total_bytes:
            lines.append(f"{format_size(state.free_bytes)} free of {format_size(state.total_bytes)}")
        if state.data_folder != state.mount_folder:
            lines.append(f"Recordings go to {state.data_folder} on this drive.")
        if not state.writable:
            lines.append(
                "This drive cannot be written to. It may be read-only or damaged; recordings would fail."
            )
            return DriveSummary(LEVEL_FAIL, f"Drive mounted at {state.mount_folder}, but not writable", "\n".join(lines))
        return DriveSummary(LEVEL_OK, f"Drive mounted at {state.mount_folder}", "\n".join(lines))

    readiness = state.readiness
    headline = f"No drive is mounted at {state.data_folder}"
    if readiness is not None and readiness.mounted is False:
        return DriveSummary(
            LEVEL_FAIL,
            headline,
            "Recording is refused until a drive is mounted here, so that nothing is written to the "
            "system disk by mistake.",
        )
    return DriveSummary(
        LEVEL_WARN,
        headline,
        "Recordings are saved on the system disk (the Pi's own SD card or SSD), which fills quickly.",
    )


# ---------------------------------------------------------------------------
# Choosing the folder
# ---------------------------------------------------------------------------


def mount_folder_problem(
    folder: str,
    *,
    home: Optional[Path] = None,
    protected: Iterable[Path] = (REPO_ROOT,),
    ismount: Callable[[str], bool] = os.path.ismount,
) -> Optional[str]:
    """Why a drive must not be mounted at ``folder``, or ``None`` when it is a sensible place.

    Mounting a drive hides whatever the folder holds, so the folder may not be
    one the system or BumbleBox itself lives in.
    """
    text = str(folder or "").strip()
    if not text:
        return "Enter the folder where the drive should appear."
    if not text.startswith("/"):
        return "The folder must be a full path starting with /, for example /mnt/bumblebox/data."
    if any(character in _UNSAFE_FOLDER_CHARACTERS for character in text):
        return "The folder name may not contain quotes, #, $, backslashes or line breaks."
    path = _normalize(text)
    home_path = _normalize(home if home is not None else Path.home())
    user = os.path.basename(home_path)
    if path in {"/media", "/run/media", f"/media/{user}", f"/run/media/{user}"}:
        return f"{path} is where the desktop puts plugged-in drives. Choose a folder inside it, or under /mnt."
    for item in [Path(home_path), *protected]:
        guarded = _normalize(item)
        if _is_inside(guarded, path):
            return f"A drive mounted at {path} would hide {guarded}. Choose a different folder."
        if guarded != home_path and _is_inside(path, guarded):
            return f"{path} is inside a folder BumbleBox itself uses ({guarded}). Choose a different folder."
    in_mount_root = any(_is_inside(path, root) for root in _MOUNT_ROOTS)
    if not in_mount_root and not _is_inside(path, home_path):
        return "Choose a folder under /mnt, /media or your home folder, for example /mnt/bumblebox/data."
    if ismount(path):
        return f"Another drive is already mounted at {path}. Remove it first, or choose a different folder."
    return None


def items_hidden_by_mount(folder: str, *, ismount: Callable[[str], bool] = os.path.ismount) -> int:
    """How many files and folders already in ``folder`` a drive mounted there would hide."""
    path = _normalize(folder)
    if not os.path.isdir(path) or ismount(path):
        return 0
    count = 0
    try:
        with os.scandir(path) as entries:
            for _entry in entries:
                count += 1
                if count >= HIDDEN_ITEMS_LIMIT:
                    break
    except OSError:
        return 0
    return count


def own_disk_folder(config: Dict[str, Any], *, home: Optional[Path] = None) -> str:
    """Where recordings go when the operator chooses the system disk instead of a drive.

    The current data folder is kept when it is an ordinary folder the operator
    can write to. A folder where drives are mounted (``/mnt/...``) is neither
    writable nor a sensible name for data on the system disk, so that becomes
    ``~/BumbleBoxData``.
    """
    system = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    current = _normalize(system.get("data_root") or "")
    fallback = str(Path(_normalize(home if home is not None else Path.home())) / "BumbleBoxData")
    if not current or current == "/" or (current + "/").startswith(EXTERNAL_DRIVE_ROOTS):
        return fallback
    probe = current
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return current if os.access(probe, os.W_OK) else fallback


# ---------------------------------------------------------------------------
# Administrator rights
# ---------------------------------------------------------------------------


@dataclass
class Elevation:
    # "root" (already root), "sudo" (no password needed), "pkexec" (asks for a password), or "none"
    method: str
    prefix: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.method != "none"

    @property
    def asks_for_password(self) -> bool:
        return self.method == "pkexec"


def find_elevation(
    *,
    which: Callable[[str], Optional[str]] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    euid: Optional[int] = None,
) -> Elevation:
    """How this user can run a command as root from a window, without a terminal.

    A Raspberry Pi OS user can normally use ``sudo`` without a password. Where
    a password is needed, ``pkexec`` asks for it in a dialog. With neither,
    the window shows the command to type in a terminal.
    """
    if euid is None:
        euid = os.geteuid() if hasattr(os, "geteuid") else -1
    if euid == 0:
        return Elevation("root", [])
    sudo = which("sudo")
    if sudo:
        try:
            probe = run([sudo, "-n", "true"], capture_output=True, text=True, check=False, timeout=10)
            if probe.returncode == 0:
                return Elevation("sudo", [sudo, "-n"])
        except Exception:
            pass
    pkexec = which("pkexec")
    if pkexec:
        return Elevation("pkexec", [pkexec])
    return Elevation("none", [])


def storage_arguments(
    action: str,
    *,
    config_path: str | Path,
    mount_folder: str,
    device_path: Optional[str] = None,
) -> List[str]:
    """The ``bbx`` arguments for one of the window's actions."""
    if action in {ACTION_MOUNT_KEEP, ACTION_KEEP}:
        arguments = ["storage", "setup"]
    elif action == ACTION_MOUNT_ONCE:
        arguments = ["storage", "mount", "--forget"]
    elif action == ACTION_FORGET:
        arguments = ["storage", "forget"]
    elif action == ACTION_UNMOUNT:
        arguments = ["storage", "unmount"]
    else:
        raise ValueError(f"Unknown data drive action: {action!r}")
    arguments.extend(["--config", str(config_path), "--mount-point", str(mount_folder)])
    if device_path and action in {ACTION_MOUNT_KEEP, ACTION_MOUNT_ONCE, ACTION_KEEP}:
        arguments.extend(["--device", str(device_path)])
    return arguments


def storage_command(
    action: str,
    elevation: Elevation,
    *,
    config_path: str | Path,
    mount_folder: str,
    device_path: Optional[str] = None,
    python: Optional[str] = None,
) -> List[str]:
    """The full command, with the way to become root in front.

    ``-B`` keeps root from leaving compiled files in the operator's copy of BumbleBox.
    """
    arguments = storage_arguments(
        action, config_path=config_path, mount_folder=mount_folder, device_path=device_path
    )
    return [*elevation.prefix, python or sys.executable, "-B", str(BBX_SCRIPT), *arguments]


def terminal_command(
    action: str,
    *,
    config_path: str | Path,
    mount_folder: str,
    device_path: Optional[str] = None,
    python: Optional[str] = None,
) -> str:
    """The same command for someone to paste into a terminal."""
    command = storage_command(
        action,
        Elevation("sudo", ["sudo"]),
        config_path=config_path,
        mount_folder=mount_folder,
        device_path=device_path,
        python=python,
    )
    return " ".join(shlex.quote(part) for part in command)


PKEXEC_NOT_AUTHORIZED = 126
PKEXEC_NO_AUTHORIZATION = 127


def explain_elevation_failure(elevation: Elevation, exit_code: int) -> Optional[str]:
    """A plain reason when the command never ran because root access was not granted."""
    if elevation.method != "pkexec":
        return None
    if exit_code == PKEXEC_NOT_AUTHORIZED:
        return "The administrator password was not given, so nothing was changed."
    if exit_code == PKEXEC_NO_AUTHORIZATION:
        return "This desktop could not ask for the administrator password, so nothing was changed."
    return None


# ---------------------------------------------------------------------------
# Recording the choice, and not interrupting a recording
# ---------------------------------------------------------------------------


def save_drive_choice(
    config_paths: Iterable[str | Path],
    *,
    require_drive: bool,
    data_folder: Optional[str] = None,
) -> List[Path]:
    """Write the choice into each config.

    ``require_drive`` true means the box is set up to record to a drive: a run
    is refused whenever no drive is mounted at the data folder, wherever that
    folder is. False means the operator chose the system disk on purpose.
    """
    saved: List[Path] = []
    for raw_path in config_paths:
        path = Path(raw_path)
        config = load_config(path)
        system = config.setdefault("system", {})
        if data_folder:
            system["data_root"] = str(data_folder)
        system["require_mounted_storage"] = bool(require_drive)
        save_config(path, config)
        saved.append(path)
    return saved


def capture_in_progress(configs: Iterable[Dict[str, Any]]) -> bool:
    """Whether a recording is capturing right now on any of these cameras.

    Uses the capture lock every run holds while it records, so it also sees a
    scheduled run or one started from a terminal.
    """
    if fcntl is None:  # pragma: no cover - platform dependent
        return False
    for config in configs:
        path = default_lock_path(capture_lock_suffix(config))
        if not path.exists():
            continue
        try:
            descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            continue
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return False
