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


def _run_command(args: List[str]) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _parse_keyval_line(line: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for match in _KEYVAL_RE.finditer(line):
        payload[match.group(1)] = match.group(2)
    return payload


def discover_storage_devices() -> List[StorageDevice]:
    proc = _run_command(["lsblk", "-P", "-o", _LSBLK_COLUMNS])
    if proc is None or proc.returncode != 0:
        return []

    devices: List[StorageDevice] = []
    for raw_line in proc.stdout.splitlines():
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
                label=str(row.get("LABEL", "")).strip(),
                size=str(row.get("SIZE", "")).strip(),
                mountpoint=str(row.get("MOUNTPOINT", "")).strip(),
                removable=rm_text == "1",
                model=str(row.get("MODEL", "")).strip(),
            )
        )
    return devices


def _findmnt_target(path: Path) -> Optional[Tuple[str, str]]:
    proc = _run_command(
        ["findmnt", "--noheadings", "--output", "SOURCE,TARGET", "--target", str(path)]
    )
    if proc is None or proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    fields = text.split()
    if len(fields) < 2:
        return None
    return fields[0], fields[1]


def _normalize_mount_path(path_text: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path_text).strip()))


def _is_exact_mount_target(resolved_target: Optional[str], expected_target: str) -> bool:
    if not resolved_target:
        return False
    return _normalize_mount_path(resolved_target) == _normalize_mount_path(expected_target)


def _read_fstab() -> str:
    fstab_path = Path("/etc/fstab")
    if not fstab_path.exists():
        return ""
    try:
        return fstab_path.read_text()
    except Exception:
        return ""


def _fstab_entry_for_mount(mount_point: str) -> Optional[str]:
    text = _read_fstab()
    if not text:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        if fields[1] == mount_point:
            return line
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

    options = _fstab_options(device.fstype, uid, gid)
    passno = "0" if device.fstype.lower() in _WINDOWS_LIKE_FSTYPES else "2"
    return f"UUID={device.uuid} {mount_point} {device.fstype} {options} 0 {passno}"


def _update_fstab_text(current_text: str, *, mount_point: str, device_uuid: str, new_entry: str) -> Tuple[str, bool]:
    lines = current_text.splitlines()
    output: List[str] = []
    changed = False
    in_marker_block = False

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped == "# >>> BumbleBox storage >>>":
            in_marker_block = True
            changed = True
            continue
        if stripped == "# <<< BumbleBox storage <<<":
            in_marker_block = False
            continue
        if in_marker_block:
            continue
        if stripped and not stripped.startswith("#"):
            fields = stripped.split()
            if len(fields) >= 2:
                source = fields[0]
                target = fields[1]
                if target == mount_point or source == f"UUID={device_uuid}":
                    changed = True
                    continue
        output.append(line)

    marker_block = [
        "# >>> BumbleBox storage >>>",
        new_entry,
        "# <<< BumbleBox storage <<<",
    ]
    if output and output[-1].strip():
        output.append("")
    output.extend(marker_block)
    new_text = "\n".join(output).rstrip() + "\n"
    if new_text != current_text:
        changed = True
    return new_text, changed


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
        fstab_path = Path("/etc/fstab")
        if fstab_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = fstab_path.with_name(f"fstab.bumblebox.bak.{stamp}")
            shutil.copy2(fstab_path, backup)
            fstab_backup_path = str(backup)
        fstab_path.write_text(new_text)

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
) -> StorageMountResult:
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
