from __future__ import annotations

import os
import pwd
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_LSBLK_COLUMNS = "NAME,PATH,TYPE,FSTYPE,UUID,LABEL,SIZE,MOUNTPOINT,RM,MODEL"
# Asked for as well when this lsblk knows them; they only make drives easier to recognise.
_LSBLK_EXTRA_COLUMNS = "PKNAME,TRAN,HOTPLUG,PARTTYPE"
_FSTAB_PATH = Path("/etc/fstab")
_MARKER_BEGIN = "# >>> BumbleBox storage >>>"
_MARKER_END = "# <<< BumbleBox storage <<<"
_KEYVAL_RE = re.compile(r'([A-Z0-9_]+)="([^"]*)"')
_WINDOWS_LIKE_FSTYPES = {"vfat", "exfat", "ntfs", "fuseblk"}
_NON_DATA_FSTYPES = {"swap", "linux_raid_member", "lvm2_member", "crypto_luks"}


@dataclass
class StorageDevice:
    name: str
    path: str
    dev_type: str
    fstype: str
    uuid: str
    label: str
    size: str
    mountpoint: str
    removable: bool
    model: str
    parent: str = ""  # kernel name of the disk a partition is on
    transport: str = ""  # usb, nvme, sata, mmc ...
    hotplug: bool = False
    part_type: str = ""

    @property
    def has_filesystem(self) -> bool:
        return bool(self.fstype and self.uuid)

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        if self.name:
            return self.name
        return self.path or "unknown-device"


@dataclass
class StorageStatusReport:
    mount_point: str
    configured_data_root: str
    mounted: bool
    writable: bool
    source: Optional[str]
    target: Optional[str]
    device_name: Optional[str]
    device_path: Optional[str]
    device_uuid: Optional[str]
    fstab_entry_present: bool
    fstab_entry: Optional[str]
    recommended_device_path: Optional[str]
    recommended_device_name: Optional[str]
    recommended_device_uuid: Optional[str]
    message: str
    notes: List[str]
    status: str


@dataclass
class StorageSetupResult:
    mount_point: str
    device_name: str
    device_path: str
    device_uuid: str
    fstype: str
    fstab_updated: bool
    mounted_now: bool
    fstab_backup_path: Optional[str]
    message: str
    notes: List[str]


@dataclass
class StorageMountResult:
    mount_point: str
    device_name: str
    device_path: str
    device_uuid: str
    fstype: str
    mounted_now: bool
    used_bind_mount: bool
    source_mountpoint: Optional[str]
    message: str
    notes: List[str]


@dataclass
class StorageUnmountResult:
    mount_point: str
    was_mounted: bool
    unmounted: bool
    source: Optional[str]
    message: str
    notes: List[str]


@dataclass
class StorageForgetResult:
    mount_point: str
    fstab_updated: bool
    removed_entries: List[str]
    fstab_backup_path: Optional[str]
    message: str
    notes: List[str]


# Looking at drives must not hang a window when a failing drive stops answering.
_QUERY_TIMEOUT_SECONDS = 10.0


def _run_command(args: List[str], timeout: Optional[float] = None) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False, timeout=timeout)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _parse_keyval_line(line: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for match in _KEYVAL_RE.finditer(line):
        payload[match.group(1)] = match.group(2)
    return payload


def parse_lsblk_output(output: str) -> List[StorageDevice]:
    """Devices from ``lsblk -P`` output. A partition takes the model of the disk it is on."""
    devices: List[StorageDevice] = []
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = _parse_keyval_line(line)
        rm_text = str(row.get("RM", "0")).strip()
        devices.append(
            StorageDevice(
                name=str(row.get("NAME", "")).strip(),
                path=str(row.get("PATH", "")).strip(),
                dev_type=str(row.get("TYPE", "")).strip(),
                fstype=str(row.get("FSTYPE", "")).strip(),
                uuid=str(row.get("UUID", "")).strip(),
                label=_unescape_pairs_value(row.get("LABEL", "")).strip(),
                size=str(row.get("SIZE", "")).strip(),
                mountpoint=_unescape_pairs_value(row.get("MOUNTPOINT", "")).strip(),
                removable=rm_text == "1",
                model=str(row.get("MODEL", "")).strip(),
                parent=str(row.get("PKNAME", "")).strip(),
                transport=str(row.get("TRAN", "")).strip().lower(),
                hotplug=str(row.get("HOTPLUG", "0")).strip() == "1",
                part_type=str(row.get("PARTTYPE", "")).strip().lower(),
            )
        )
    by_name = {device.name: device for device in devices if device.name}
    for device in devices:
        disk = by_name.get(device.parent)
        if disk is None:
            continue
        device.model = device.model or disk.model
        device.transport = device.transport or disk.transport
        device.hotplug = device.hotplug or disk.hotplug
        device.removable = device.removable or disk.removable
    return devices


def discover_storage_devices() -> List[StorageDevice]:
    proc = _run_command(
        ["lsblk", "-P", "-o", f"{_LSBLK_COLUMNS},{_LSBLK_EXTRA_COLUMNS}"], timeout=_QUERY_TIMEOUT_SECONDS
    )
    if proc is None or proc.returncode != 0:
        # An older lsblk refuses columns it does not know; the basic set is enough to work.
        proc = _run_command(["lsblk", "-P", "-o", _LSBLK_COLUMNS], timeout=_QUERY_TIMEOUT_SECONDS)
    if proc is None or proc.returncode != 0:
        return []
    return parse_lsblk_output(proc.stdout)


def _findmnt_target(path: Path) -> Optional[Tuple[str, str]]:
    proc = _run_command(
        ["findmnt", "--pairs", "--output", "SOURCE,TARGET", "--target", str(path)],
        timeout=_QUERY_TIMEOUT_SECONDS,
    )
    if proc is None or proc.returncode != 0:
        return None
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    # Pairs output keeps a folder name with spaces in one piece.
    row = _parse_keyval_line(lines[-1])
    source = _unescape_pairs_value(row.get("SOURCE", ""))
    target = _unescape_pairs_value(row.get("TARGET", ""))
    if not source or not target:
        return None
    return source, target


def _unescape_pairs_value(value: str) -> str:
    """Undo the ``\\xNN`` escapes util-linux uses in ``--pairs`` output."""
    return re.sub(r"\\x([0-9a-fA-F]{2})", lambda match: chr(int(match.group(1), 16)), str(value))


def _fstab_escape(path_text: str) -> str:
    """A path as fstab wants it: a space or tab inside a field is written as an octal escape."""
    return str(path_text).replace("\\", "\\134").replace(" ", "\\040").replace("\t", "\\011")


def _fstab_unescape(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), str(field))


def _same_mount_path(first: str, second: str) -> bool:
    return _normalize_mount_path(first) == _normalize_mount_path(second)


def _normalize_mount_path(path_text: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path_text).strip()))


def _is_exact_mount_target(resolved_target: Optional[str], expected_target: str) -> bool:
    if not resolved_target:
        return False
    return _normalize_mount_path(resolved_target) == _normalize_mount_path(expected_target)


def _read_fstab() -> str:
    fstab_path = _FSTAB_PATH
    if not fstab_path.exists():
        return ""
    try:
        return fstab_path.read_text()
    except Exception:
        return ""


def _fstab_entry_for_mount(mount_point: str, fstab_text: Optional[str] = None) -> Optional[str]:
    text = _read_fstab() if fstab_text is None else fstab_text
    if not text:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        if _same_mount_path(_fstab_unescape(fields[1]), mount_point):
            return line
    return None


def fstab_source_uuid(fstab_line: Optional[str]) -> Optional[str]:
    """The UUID an fstab line mounts, when it names its drive that way."""
    fields = str(fstab_line or "").split()
    if not fields:
        return None
    source = fields[0]
    if source.upper().startswith("UUID="):
        return source[5:].strip('"') or None
    prefix = "/dev/disk/by-uuid/"
    if source.startswith(prefix):
        return source[len(prefix):] or None
    return None


def _resolve_source_device(source: str, devices: List[StorageDevice]) -> Optional[StorageDevice]:
    text = str(source).strip()
    if not text:
        return None

    if text.startswith("UUID="):
        uuid = text[5:]
        for device in devices:
            if device.uuid == uuid:
                return device
        return None

    for device in devices:
        if device.path == text:
            return device
    return None


def _candidate_devices(devices: List[StorageDevice]) -> List[StorageDevice]:
    out: List[StorageDevice] = []
    for device in devices:
        if device.dev_type != "part":
            continue
        if not device.has_filesystem:
            continue
        if device.fstype.strip().lower() in _NON_DATA_FSTYPES:
            continue
        mountpoint = device.mountpoint.strip()
        if mountpoint in {"/", "/boot", "/boot/firmware"}:
            continue
        if not device.path.startswith("/dev/"):
            continue
        out.append(device)
    return out


def choose_storage_device(
    devices: List[StorageDevice],
    *,
    mount_point: Optional[str] = None,
    device_path: Optional[str] = None,
) -> Optional[StorageDevice]:
    chosen_path = str(device_path or "").strip()
    if chosen_path:
        for device in devices:
            if chosen_path in {device.path, device.name, f"/dev/{device.name}"}:
                return device
        return None

    if mount_point:
        point = str(mount_point).strip()
        for device in devices:
            if device.mountpoint == point:
                return device

    candidates = _candidate_devices(devices)
    if not candidates:
        return None

    def _sort_key(device: StorageDevice) -> Tuple[int, int, str]:
        return (
            0 if device.removable else 1,
            0 if not device.mountpoint else 1,
            device.path,
        )

    candidates.sort(key=_sort_key)
    return candidates[0]


def _is_root_user() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    try:
        return int(geteuid()) == 0
    except Exception:
        return False


def _effective_owner_ids(uid: Optional[int] = None, gid: Optional[int] = None) -> Tuple[int, int]:
    owner_uid = int(uid if uid is not None else os.getuid())
    owner_gid = int(gid if gid is not None else os.getgid())

    env_uid = os.environ.get("PKEXEC_UID") or os.environ.get("SUDO_UID")
    env_gid = os.environ.get("SUDO_GID")
    if uid is None and env_uid:
        try:
            owner_uid = int(env_uid)
        except Exception:
            pass
    if gid is None and env_gid:
        try:
            owner_gid = int(env_gid)
        except Exception:
            pass
    elif gid is None and env_uid:
        try:
            owner_gid = int(pwd.getpwuid(owner_uid).pw_gid)
        except Exception:
            pass
    return owner_uid, owner_gid


def _fstab_options(fstype: str, uid: int, gid: int) -> str:
    base = "defaults,nofail,x-systemd.device-timeout=10"
    if fstype.lower() in _WINDOWS_LIKE_FSTYPES:
        return f"{base},uid={uid},gid={gid},umask=0002"
    return base


def _runtime_mount_options(fstype: str, uid: int, gid: int) -> Optional[str]:
    if fstype.lower() in _WINDOWS_LIKE_FSTYPES:
        return f"uid={uid},gid={gid},umask=0002"
    return None


def _stable_mount_source(device: StorageDevice) -> str:
    uuid = str(device.uuid or "").strip()
    if uuid:
        by_uuid = Path("/dev/disk/by-uuid") / uuid
        return str(by_uuid)
    return device.path


def _release_existing_device_mount(
    device: StorageDevice,
    *,
    desired_mount_point: str,
    notes: List[str],
    dry_run: bool = False,
) -> Optional[str]:
    source_mountpoint = str(device.mountpoint or "").strip()
    if not source_mountpoint:
        return None
    if _normalize_mount_path(source_mountpoint) == _normalize_mount_path(desired_mount_point):
        return None

    notes.append(
        f"Device was already mounted at {source_mountpoint}; remounting it directly at {desired_mount_point}."
    )
    if dry_run:
        notes.append("Dry run: existing mount was not unmounted.")
        return source_mountpoint

    proc = _run_command(["umount", source_mountpoint])
    if proc is None:
        raise RuntimeError("Could not run umount to release the existing desktop auto-mount.")
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            stderr
            or stdout
            or f"Could not unmount existing source mount {source_mountpoint}. Close any open file-browser windows for that drive and try again."
        )
    return source_mountpoint


def _uuid_fstab_note_needed(source: Optional[str], fstab_line: Optional[str]) -> bool:
    if not source or not source.startswith("/dev/sd"):
        return False
    if fstab_line and "UUID=" in fstab_line:
        return False
    return True


def _ensure_mount_root_owned_by_user(
    mount_dir: Path,
    *,
    fstype: str,
    owner_uid: int,
    owner_gid: int,
    notes: List[str],
    dry_run: bool = False,
) -> None:
    if str(fstype).strip().lower() in _WINDOWS_LIKE_FSTYPES:
        return
    if dry_run:
        notes.append(
            f"Dry run: would set ownership on {mount_dir} to uid={owner_uid}, gid={owner_gid} for user write access."
        )
        return
    try:
        os.chown(mount_dir, owner_uid, owner_gid)
        notes.append(f"Set ownership on {mount_dir} to uid={owner_uid}, gid={owner_gid} for user write access.")
    except PermissionError as exc:
        notes.append(f"Could not update ownership on {mount_dir}: {exc}")
    except Exception as exc:
        notes.append(f"Could not update ownership on {mount_dir}: {exc}")


def build_fstab_entry(device: StorageDevice, mount_point: str, *, uid: int, gid: int) -> str:
    if not device.uuid:
        raise ValueError("Selected storage device has no UUID; cannot create stable fstab entry.")
    if not device.fstype:
        raise ValueError("Selected storage device has no filesystem type (FSTYPE).")

    if not str(mount_point).startswith("/") or any(ch in str(mount_point) for ch in "\n\r#"):
        # A line break or a comment sign in this field would corrupt the file.
        raise ValueError(f"This folder name cannot be written to /etc/fstab: {mount_point!r}")

    options = _fstab_options(device.fstype, uid, gid)
    passno = "0" if device.fstype.lower() in _WINDOWS_LIKE_FSTYPES else "2"
    return f"UUID={device.uuid} {_fstab_escape(mount_point)} {device.fstype} {options} 0 {passno}"


def _rewrite_fstab_text(
    current_text: str,
    *,
    mount_point: str,
    device_uuid: Optional[str] = None,
    new_entry: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Take the lines for ``mount_point`` (and for ``device_uuid``) out of fstab text.

    ``new_entry``, when given, replaces them. Entries BumbleBox wrote for other
    folders are kept, so two cameras recording to two drives each keep theirs;
    everything BumbleBox manages ends up in one marked block at the end.
    Returns the new text and the lines that were removed.
    """

    def is_target(stripped: str) -> bool:
        fields = stripped.split()
        if len(fields) < 2:
            return False
        if _same_mount_path(_fstab_unescape(fields[1]), mount_point):
            return True
        return bool(device_uuid) and fstab_source_uuid(stripped) == device_uuid

    output: List[str] = []
    managed: List[str] = []
    removed: List[str] = []
    in_marker_block = False
    for raw in current_text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped == _MARKER_BEGIN:
            in_marker_block = True
            continue
        if stripped == _MARKER_END:
            in_marker_block = False
            continue
        is_entry = bool(stripped) and not stripped.startswith("#")
        if is_entry and is_target(stripped):
            removed.append(stripped)
            continue
        if in_marker_block:
            if is_entry:
                managed.append(stripped)
            continue
        output.append(line)

    if new_entry:
        managed.append(new_entry)
    while output and not output[-1].strip():
        output.pop()
    if managed:
        if output:
            output.append("")
        output.extend([_MARKER_BEGIN, *managed, _MARKER_END])
    new_text = "\n".join(output) + "\n" if output else ""
    return new_text, removed


def _update_fstab_text(current_text: str, *, mount_point: str, device_uuid: str, new_entry: str) -> Tuple[str, bool]:
    new_text, _removed = _rewrite_fstab_text(
        current_text, mount_point=mount_point, device_uuid=device_uuid, new_entry=new_entry
    )
    return new_text, new_text != current_text


def _write_fstab(new_text: str) -> Optional[str]:
    """Replace /etc/fstab, keeping a dated copy of the old one. Returns the copy's path."""
    fstab_path = _FSTAB_PATH
    backup_path: Optional[str] = None
    if fstab_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = fstab_path.with_name(f"fstab.bumblebox.bak.{stamp}")
        shutil.copy2(fstab_path, backup)
        backup_path = str(backup)
    # Written beside the file and moved into place, so a power cut cannot leave half a file.
    temporary = fstab_path.with_name(f"{fstab_path.name}.bumblebox.tmp")
    temporary.write_text(new_text)
    if fstab_path.exists():
        shutil.copymode(fstab_path, temporary)
    os.replace(temporary, fstab_path)
    return backup_path


def _reload_systemd(notes: List[str]) -> None:
    """Tell systemd that fstab changed, so it does not act on the mounts it read at boot."""
    proc = _run_command(["systemctl", "daemon-reload"])
    if proc is None:
        return  # no systemd here
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or "unknown error"
        notes.append(f"systemctl daemon-reload did not succeed ({detail}); a restart will apply the change.")


def _device_name_for_message(device: Optional[StorageDevice], source: Optional[str]) -> str:
    if device is not None:
        if device.name:
            return device.name
        if device.path:
            return device.path
    if source:
        return str(source)
    return "external"


def get_storage_status(config: Dict[str, Any], *, mount_point: Optional[str] = None) -> StorageStatusReport:
    configured_data_root = str(config.get("system", {}).get("data_root", "")).strip()
    target_mount_point = str(mount_point or configured_data_root or "/mnt/bumblebox/data").strip()
    devices = discover_storage_devices()
    selected = choose_storage_device(devices, mount_point=target_mount_point)

    resolved = _findmnt_target(Path(target_mount_point))
    fstab_line = _fstab_entry_for_mount(target_mount_point)
    notes: List[str] = []
    mounted = False
    writable = os.access(target_mount_point, os.W_OK) if Path(target_mount_point).exists() else False

    if resolved is not None:
        source, target = resolved
        source_device = _resolve_source_device(source, devices)
        if _is_exact_mount_target(target, target_mount_point):
            mounted = True
            device_name = _device_name_for_message(source_device, source)
            message = f"Writing data to {device_name} storage, you can find it at {target_mount_point}"
            status = "PASS" if writable else "WARN"
            if _uuid_fstab_note_needed(source, fstab_line):
                notes.append(
                    "Mounted source uses /dev/sdX naming. UUID-based fstab is recommended to avoid boot-time device-name drift."
                )
        else:
            device_name = _device_name_for_message(source_device, source)
            message = (
                f"No dedicated storage is mounted at {target_mount_point}. "
                f"The path currently lives on {device_name} mounted at {target}."
            )
            status = "WARN"
            notes.append(
                f"{target_mount_point} is not its own active mount point yet. "
                "Saving the path in config does not mount external storage by itself."
            )
    else:
        source = None
        target = None
        source_device = None
        device_name = None
        message = f"No storage is mounted at {target_mount_point}. Use storage setup to configure auto-mount."
        status = "WARN"

    if not fstab_line:
        notes.append(f"No /etc/fstab entry currently targets {target_mount_point}.")

    if selected is not None and not mounted:
        notes.append(
            f"Detected candidate device {selected.path} (UUID={selected.uuid or 'missing'}) "
            f"for auto-mount setup."
        )

    return StorageStatusReport(
        mount_point=target_mount_point,
        configured_data_root=configured_data_root,
        mounted=mounted,
        writable=writable,
        source=source,
        target=target,
        device_name=device_name,
        device_path=source_device.path if source_device is not None else None,
        device_uuid=source_device.uuid if source_device is not None else None,
        fstab_entry_present=bool(fstab_line),
        fstab_entry=fstab_line,
        recommended_device_path=selected.path if selected is not None else None,
        recommended_device_name=selected.display_name if selected is not None else None,
        recommended_device_uuid=selected.uuid if selected is not None else None,
        message=message,
        notes=notes,
        status=status,
    )


def setup_storage_auto_mount(
    mount_point: str,
    *,
    device_path: Optional[str] = None,
    uid: Optional[int] = None,
    gid: Optional[int] = None,
    dry_run: bool = False,
) -> StorageSetupResult:
    mount_point = str(mount_point).strip() or "/mnt/bumblebox/data"
    devices = discover_storage_devices()
    selected = choose_storage_device(devices, mount_point=mount_point, device_path=device_path)
    if selected is None:
        raise RuntimeError("No suitable storage partition was detected for auto-mount setup.")
    if not selected.uuid:
        raise RuntimeError(f"Selected device {selected.path} has no UUID. Format/mount a filesystem first.")
    if not selected.fstype:
        raise RuntimeError(f"Selected device {selected.path} has no detected filesystem type.")

    owner_uid, owner_gid = _effective_owner_ids(uid=uid, gid=gid)
    entry = build_fstab_entry(selected, mount_point, uid=owner_uid, gid=owner_gid)
    notes: List[str] = []

    if not dry_run and not _is_root_user():
        raise PermissionError(
            "Storage setup needs root privileges to edit /etc/fstab and mount filesystems."
        )

    mount_dir = Path(mount_point)
    if not dry_run:
        mount_dir.mkdir(parents=True, exist_ok=True)

    old_text = _read_fstab()
    new_text, changed = _update_fstab_text(
        old_text,
        mount_point=mount_point,
        device_uuid=selected.uuid,
        new_entry=entry,
    )
    fstab_backup_path: Optional[str] = None
    if changed and not dry_run:
        fstab_backup_path = _write_fstab(new_text)
        _reload_systemd(notes)

    mounted_now = False
    current = _findmnt_target(mount_dir)
    current_exact = current is not None and _is_exact_mount_target(current[1], mount_point)
    if not current_exact and not dry_run:
        _release_existing_device_mount(
            selected,
            desired_mount_point=mount_point,
            notes=notes,
            dry_run=dry_run,
        )
        proc = _run_command(["mount", mount_point])
        if proc is None:
            notes.append("Failed to run mount command (mount utility missing).")
        elif proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            notes.append(f"mount returned non-zero exit status: {stderr or stdout or 'unknown error'}")
    resolved = _findmnt_target(mount_dir)
    if resolved is not None and _is_exact_mount_target(resolved[1], mount_point):
        mounted_now = True
        _ensure_mount_root_owned_by_user(
            mount_dir,
            fstype=selected.fstype,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            notes=notes,
            dry_run=dry_run,
        )

    message = (
        f"Writing data to {selected.name or selected.path} storage, you can find it at {mount_point}"
        if mounted_now or dry_run
        else f"Configured /etc/fstab for {selected.path}, but mount is not active yet at {mount_point}."
    )
    if dry_run:
        notes.append("Dry run: no files were changed.")

    return StorageSetupResult(
        mount_point=mount_point,
        device_name=selected.name or selected.display_name,
        device_path=selected.path,
        device_uuid=selected.uuid,
        fstype=selected.fstype,
        fstab_updated=changed and not dry_run,
        mounted_now=mounted_now,
        fstab_backup_path=fstab_backup_path,
        message=message,
        notes=notes,
    )


def mount_storage_device_now(
    mount_point: str,
    *,
    device_path: Optional[str] = None,
    uid: Optional[int] = None,
    gid: Optional[int] = None,
    dry_run: bool = False,
    forget_startup: bool = False,
) -> StorageMountResult:
    """Mount a drive at ``mount_point`` until the next restart.

    ``forget_startup`` also removes any /etc/fstab entry for the folder first.
    Without that, an entry naming a different drive makes systemd unmount the
    drive again straight away, because it still expects the other one there.
    """
    mount_point = str(mount_point).strip() or "/mnt/bumblebox/data"
    devices = discover_storage_devices()
    selected = choose_storage_device(devices, mount_point=mount_point, device_path=device_path)
    if selected is None:
        raise RuntimeError("No suitable storage partition was detected to mount.")
    if not selected.path:
        raise RuntimeError("Selected storage device has no device path.")
    if not selected.fstype:
        raise RuntimeError(f"Selected device {selected.path} has no detected filesystem type.")

    owner_uid, owner_gid = _effective_owner_ids(uid=uid, gid=gid)
    notes: List[str] = []

    if not dry_run and not _is_root_user():
        raise PermissionError("Mounting storage needs root privileges.")

    mount_dir = Path(mount_point)
    if forget_startup and _fstab_entry_for_mount(mount_point):
        if dry_run:
            notes.append(f"Dry run: would remove the /etc/fstab entry for {mount_point}.")
        else:
            forgotten = remove_storage_auto_mount(mount_point)
            notes.append(forgotten.message)
            notes.extend(forgotten.notes)
    if not dry_run:
        mount_dir.mkdir(parents=True, exist_ok=True)

    current = _findmnt_target(mount_dir)
    current_exact = current is not None and _is_exact_mount_target(current[1], mount_point)
    if current_exact:
        current_source = current[0]
        stable_source = _stable_mount_source(selected)
        if current_source in {selected.path, stable_source, f"UUID={selected.uuid}"}:
            return StorageMountResult(
                mount_point=mount_point,
                device_name=selected.name or selected.display_name,
                device_path=selected.path,
                device_uuid=selected.uuid,
                fstype=selected.fstype,
                mounted_now=True,
                used_bind_mount=False,
                source_mountpoint=selected.mountpoint or None,
                message=f"Storage device is already mounted at {mount_point}.",
                notes=notes,
            )
        raise RuntimeError(
            f"{mount_point} is already occupied by {current_source}. Unmount it first or choose a different mount point."
        )

    used_bind_mount = False
    source_mountpoint = _release_existing_device_mount(
        selected,
        desired_mount_point=mount_point,
        notes=notes,
        dry_run=dry_run,
    )

    args = ["mount"]
    mount_opts = _runtime_mount_options(selected.fstype, owner_uid, owner_gid)
    if mount_opts:
        args.extend(["-o", mount_opts])
    mount_source = _stable_mount_source(selected)
    args.extend([mount_source, mount_point])
    if not dry_run:
        proc = _run_command(args)
        if proc is None:
            notes.append("Failed to run mount command (mount utility missing).")
        elif proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RuntimeError(stderr or stdout or "mount failed")

    resolved = _findmnt_target(mount_dir)
    mounted_now = bool(dry_run) or (
        resolved is not None and _is_exact_mount_target(resolved[1], mount_point)
    )
    if mounted_now:
        _ensure_mount_root_owned_by_user(
            mount_dir,
            fstype=selected.fstype,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            notes=notes,
            dry_run=dry_run,
        )
    if source_mountpoint:
        message = (
            f"Unmounted the existing desktop mount at {source_mountpoint} and mounted {selected.path} "
            f"directly at {mount_point}."
        )
    else:
        message = f"Mounted {selected.path} at {mount_point}."
    if selected.uuid:
        notes.append(f"Used UUID-backed mount source for this session: {_stable_mount_source(selected)}")
    if dry_run:
        notes.append("Dry run: no mount command was executed.")

    return StorageMountResult(
        mount_point=mount_point,
        device_name=selected.name or selected.display_name,
        device_path=selected.path,
        device_uuid=selected.uuid,
        fstype=selected.fstype,
        mounted_now=mounted_now,
        used_bind_mount=used_bind_mount,
        source_mountpoint=source_mountpoint,
        message=message,
        notes=notes,
    )


def remove_storage_auto_mount(mount_point: str, *, dry_run: bool = False) -> StorageForgetResult:
    """Stop mounting a drive at ``mount_point`` at startup. A drive mounted there now stays mounted."""
    mount_point = str(mount_point).strip() or "/mnt/bumblebox/data"
    notes: List[str] = []
    old_text = _read_fstab()
    new_text, removed = _rewrite_fstab_text(old_text, mount_point=mount_point)
    if not removed:
        return StorageForgetResult(
            mount_point=mount_point,
            fstab_updated=False,
            removed_entries=[],
            fstab_backup_path=None,
            message=f"/etc/fstab has no entry for {mount_point}; nothing is mounted there at startup.",
            notes=notes,
        )
    if dry_run:
        notes.append("Dry run: no files were changed.")
        backup_path: Optional[str] = None
    else:
        if not _is_root_user():
            raise PermissionError("Changing what is mounted at startup needs root privileges to edit /etc/fstab.")
        backup_path = _write_fstab(new_text)
        _reload_systemd(notes)
    return StorageForgetResult(
        mount_point=mount_point,
        fstab_updated=not dry_run,
        removed_entries=removed,
        fstab_backup_path=backup_path,
        message=f"No drive will be mounted at {mount_point} at startup.",
        notes=notes,
    )


def unmount_storage(mount_point: str, *, dry_run: bool = False) -> StorageUnmountResult:
    """Unmount the drive at ``mount_point`` so that it can be unplugged."""
    mount_point = str(mount_point).strip() or "/mnt/bumblebox/data"
    notes: List[str] = []
    current = _findmnt_target(Path(mount_point))
    if current is None or not _is_exact_mount_target(current[1], mount_point):
        return StorageUnmountResult(
            mount_point=mount_point,
            was_mounted=False,
            unmounted=False,
            source=None,
            message=f"No drive is mounted at {mount_point}.",
            notes=notes,
        )
    source = current[0]
    if dry_run:
        notes.append("Dry run: nothing was unmounted.")
        return StorageUnmountResult(
            mount_point=mount_point,
            was_mounted=True,
            unmounted=False,
            source=source,
            message=f"Would unmount {source} from {mount_point}.",
            notes=notes,
        )
    if not _is_root_user():
        raise PermissionError("Unmounting storage needs root privileges.")
    proc = _run_command(["umount", mount_point])
    if proc is None:
        raise RuntimeError("Could not run umount.")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or "umount failed"
        if "busy" in detail.lower():
            raise RuntimeError(
                f"The drive at {mount_point} is in use, so it was not unmounted. Wait for any recording "
                "or tracking to finish, close file-manager windows showing the drive, and try again. "
                f"({detail})"
            )
        raise RuntimeError(detail)
    return StorageUnmountResult(
        mount_point=mount_point,
        was_mounted=True,
        unmounted=True,
        source=source,
        message=f"Unmounted {source} from {mount_point}. The drive can be unplugged.",
        notes=notes,
    )


def format_storage_forget_result(result: StorageForgetResult) -> str:
    lines = [
        "Storage At Startup",
        "------------------",
        f"Mount point: {result.mount_point}",
        f"/etc/fstab updated: {result.fstab_updated}",
    ]
    for entry in result.removed_entries:
        lines.append(f"Removed line: {entry}")
    if result.fstab_backup_path:
        lines.append(f"/etc/fstab backup: {result.fstab_backup_path}")
    lines.append(result.message)
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def format_storage_unmount_result(result: StorageUnmountResult) -> str:
    lines = [
        "Storage Unmount",
        "---------------",
        f"Mount point: {result.mount_point}",
        f"Unmounted: {result.unmounted}",
    ]
    if result.source:
        lines.append(f"Device: {result.source}")
    lines.append(result.message)
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def format_storage_status_report(report: StorageStatusReport) -> str:
    lines = [
        f"[{report.status}] {report.message}",
        f"Configured data_root: {report.configured_data_root or '(not set)'}",
        f"Mount point: {report.mount_point}",
        f"Mounted: {report.mounted}",
        f"Writable: {report.writable}",
        f"Mount source: {report.source or 'n/a'}",
        f"/etc/fstab entry present: {report.fstab_entry_present}",
    ]
    if report.fstab_entry:
        lines.append(f"/etc/fstab line: {report.fstab_entry}")
    if report.recommended_device_path:
        lines.append(
            "Detected candidate device: "
            f"{report.recommended_device_path} (UUID={report.recommended_device_uuid or 'missing'})"
        )
    for note in report.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def format_storage_setup_result(result: StorageSetupResult) -> str:
    lines = [
        "Storage Setup",
        "-------------",
        f"Mount point: {result.mount_point}",
        f"Device: {result.device_path} (name={result.device_name}, uuid={result.device_uuid}, fstype={result.fstype})",
        f"/etc/fstab updated: {result.fstab_updated}",
        f"Mounted now: {result.mounted_now}",
    ]
    if result.fstab_backup_path:
        lines.append(f"/etc/fstab backup: {result.fstab_backup_path}")
    lines.append(result.message)
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def format_storage_mount_result(result: StorageMountResult) -> str:
    lines = [
        "Storage Mount",
        "-------------",
        f"Mount point: {result.mount_point}",
        f"Device: {result.device_path} (name={result.device_name}, uuid={result.device_uuid or 'missing'}, fstype={result.fstype})",
        f"Mounted now: {result.mounted_now}",
    ]
    if result.source_mountpoint:
        lines.append(f"Previous source mount: {result.source_mountpoint}")
    if result.used_bind_mount:
        lines.append(f"Bind mount used: {result.used_bind_mount}")
    lines.append(result.message)
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_storage_setup_sudo_command(
    *,
    config_path: str,
    mount_point: str,
    device_path: Optional[str] = None,
    apply_config: bool = True,
    dry_run: bool = False,
) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    bbx_path = repo_root / "bbx.py"
    args = [
        "sudo",
        "python3",
        str(bbx_path),
        "storage",
        "setup",
        "--config",
        str(config_path),
        "--mount-point",
        str(mount_point),
    ]
    if device_path:
        args.extend(["--device", str(device_path)])
    if apply_config:
        args.append("--apply-config")
    if dry_run:
        args.append("--dry-run")
    return " ".join(shlex.quote(str(part)) for part in args)


def build_storage_mount_sudo_command(
    *,
    config_path: str,
    mount_point: str,
    device_path: Optional[str] = None,
    apply_config: bool = True,
    dry_run: bool = False,
) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    bbx_path = repo_root / "bbx.py"
    args = [
        "sudo",
        "python3",
        str(bbx_path),
        "storage",
        "mount",
        "--config",
        str(config_path),
        "--mount-point",
        str(mount_point),
    ]
    if device_path:
        args.extend(["--device", str(device_path)])
    if apply_config:
        args.append("--apply-config")
    if dry_run:
        args.append("--dry-run")
    return " ".join(shlex.quote(str(part)) for part in args)
