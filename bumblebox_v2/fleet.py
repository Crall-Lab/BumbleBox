from __future__ import annotations

import json
import re
import os
import shlex
import shutil
import socket
import subprocess
import time
from ipaddress import ip_address
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


VALID_FLEET_ROLES = {"standalone", "queen", "worker"}


@dataclass
class FleetWorker:
    name: str
    host: str
    user: str
    port: int
    enabled: bool
    data_root: str
    unit_prefix: str


@dataclass
class FleetInitResult:
    role: str
    queen_host: str
    identity_file: str
    public_key_file: str
    public_key: str
    key_created: bool
    key_install_command_example: str
    chrony_queen_hint: str
    chrony_worker_hint: str


@dataclass
class FleetEnrollResult:
    worker_name: str
    worker_host: str
    worker_user: str
    worker_port: int
    updated_existing: bool
    install_key_attempted: bool
    install_key_ok: bool
    install_key_message: str
    manual_key_install_command: str
    enabled_worker_count: int
    queen_media_max_videos_total: int


@dataclass
class FleetWorkerStatus:
    worker_name: str
    host: str
    user: str
    status: str
    reachable: bool
    time_offset_seconds: Optional[float]
    disk_free_gb_root: Optional[float]
    mem_available_gb: Optional[float]
    cpu_temp_c: Optional[float]
    load_1m: Optional[float]
    bumblebox_timer_count: int
    latest_run_summary: Optional[str]
    notes: list[str]
    error: Optional[str]


@dataclass
class FleetStatusReport:
    generated_at: str
    queen_host: str
    role: str
    worker_count: int
    pass_count: int
    warn_count: int
    fail_count: int
    workers: list[FleetWorkerStatus]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "queen_host": self.queen_host,
            "role": self.role,
            "worker_count": self.worker_count,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "workers": [asdict(item) for item in self.workers],
        }


@dataclass
class QueenTrackItem:
    worker_name: str
    worker_host: str
    remote_video_path: str
    local_video_path: Optional[str]
    raw_csv_path: Optional[str]
    tracked_video_path: Optional[str]
    status: str
    note: str
    latest_video_path: Optional[str] = None
    latest_video_pulled_at: Optional[str] = None
    latest_tracked_video_path: Optional[str] = None
    latest_tracked_at: Optional[str] = None


@dataclass
class QueenTrackReport:
    generated_at: str
    role: str
    queen_local_pipeline_enabled: bool
    output_root: str
    state_path: str
    workers_considered: int
    videos_considered: int
    videos_processed: int
    videos_skipped: int
    videos_failed: int
    skipped_due_to_guard: bool
    guard_reason: Optional[str]
    queen_load_1m: Optional[float]
    queen_mem_available_gb: Optional[float]
    items: list[QueenTrackItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "role": self.role,
            "queen_local_pipeline_enabled": self.queen_local_pipeline_enabled,
            "output_root": self.output_root,
            "state_path": self.state_path,
            "workers_considered": self.workers_considered,
            "videos_considered": self.videos_considered,
            "videos_processed": self.videos_processed,
            "videos_skipped": self.videos_skipped,
            "videos_failed": self.videos_failed,
            "skipped_due_to_guard": self.skipped_due_to_guard,
            "guard_reason": self.guard_reason,
            "queen_load_1m": self.queen_load_1m,
            "queen_mem_available_gb": self.queen_mem_available_gb,
            "items": [asdict(item) for item in self.items],
        }


@dataclass
class QueenLatestStatusItem:
    worker_name: str
    worker_host: str
    latest_video_path: Optional[str]
    latest_video_pulled_at: Optional[str]
    latest_tracked_video_path: Optional[str]
    latest_tracked_at: Optional[str]
    status: str
    online: Optional[bool]
    track_lag_minutes: Optional[float]
    note: str


@dataclass
class QueenLatestStatusReport:
    generated_at: str
    output_root: str
    workers_considered: int
    synced_count: int
    stale_count: int
    missing_latest_video_count: int
    missing_latest_tracked_count: int
    offline_count: int
    items: list[QueenLatestStatusItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "output_root": self.output_root,
            "workers_considered": self.workers_considered,
            "synced_count": self.synced_count,
            "stale_count": self.stale_count,
            "missing_latest_video_count": self.missing_latest_video_count,
            "missing_latest_tracked_count": self.missing_latest_tracked_count,
            "offline_count": self.offline_count,
            "items": [asdict(item) for item in self.items],
        }


@dataclass
class FleetDiscoveryItem:
    host: str
    from_config: bool
    worker_name: Optional[str]
    ping_ok: Optional[bool]
    ssh_port_open: Optional[bool]
    reachable: bool
    arp_mac: Optional[str]
    arp_state: Optional[str]


@dataclass
class FleetDiscoveryReport:
    generated_at: str
    configured_workers: int
    configured_workers_online: int
    configured_workers_offline: int
    discovered_hosts: int
    items: list[FleetDiscoveryItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "configured_workers": self.configured_workers,
            "configured_workers_online": self.configured_workers_online,
            "configured_workers_offline": self.configured_workers_offline,
            "discovered_hosts": self.discovered_hosts,
            "items": [asdict(item) for item in self.items],
        }


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_identity_file() -> str:
    return str(Path("~/.ssh/bbx_fleet_ed25519").expanduser())


def _default_queen_host() -> str:
    host = socket.gethostname()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 53))
            ip = s.getsockname()[0]
            if ip:
                return ip
        finally:
            s.close()
    except Exception:
        pass
    return host


def _fleet_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.setdefault("fleet", {})
    section.setdefault("role", "standalone")
    section.setdefault("queen_host", None)
    section.setdefault("queen_local_pipeline_enabled", True)
    ssh_cfg = section.setdefault("ssh", {})
    ssh_cfg.setdefault("user", "pi")
    ssh_cfg.setdefault("port", 22)
    ssh_cfg.setdefault("identity_file", _default_identity_file())
    ssh_cfg.setdefault("connect_timeout_seconds", 5)
    section.setdefault("workers", [])
    capture = config.get("capture", {}) if isinstance(config.get("capture", {}), dict) else {}
    try:
        default_pull_interval = int(capture.get("record_interval_minutes", 30))
    except Exception:
        default_pull_interval = 30
    if default_pull_interval <= 0:
        default_pull_interval = 30
    media = section.setdefault("queen_media_schedule", {})
    media.setdefault("enabled", False)
    media.setdefault("pull_interval_minutes", default_pull_interval)
    media.setdefault("track_interval_minutes", 60)
    media.setdefault("output_root", None)
    media.setdefault("max_videos_total", 200)
    media.setdefault("max_videos_per_worker", 1)
    media.setdefault("cooldown_minutes", 60)
    media.setdefault("max_queen_load_1m", 3.0)
    media.setdefault("min_queen_mem_gb", 0.8)
    media.setdefault("disable_visualization", False)
    media.setdefault("allow_when_queen_bbox_active", False)
    return section


def apply_queen_media_schedule_defaults(
    config: dict[str, Any],
    *,
    enable: Optional[bool] = None,
) -> dict[str, Any]:
    fleet = _fleet_section(config)
    media = fleet.setdefault("queen_media_schedule", {})

    capture = config.get("capture", {}) if isinstance(config.get("capture", {}), dict) else {}
    try:
        record_interval = int(capture.get("record_interval_minutes", 30))
    except Exception:
        record_interval = 30
    if record_interval <= 0:
        record_interval = 30

    try:
        pull_interval = int(media.get("pull_interval_minutes", record_interval))
    except Exception:
        pull_interval = record_interval
    media["pull_interval_minutes"] = max(1, pull_interval)

    try:
        track_interval = int(media.get("track_interval_minutes", 60))
    except Exception:
        track_interval = 60
    media["track_interval_minutes"] = max(1, track_interval)

    try:
        max_total = int(media.get("max_videos_total", 200))
    except Exception:
        max_total = 200
    media["max_videos_total"] = max(1, max_total)

    try:
        max_per_worker = int(media.get("max_videos_per_worker", 1))
    except Exception:
        max_per_worker = 1
    media["max_videos_per_worker"] = max(1, max_per_worker)

    try:
        cooldown = int(media.get("cooldown_minutes", 60))
    except Exception:
        cooldown = 60
    media["cooldown_minutes"] = max(0, cooldown)

    try:
        max_load = float(media.get("max_queen_load_1m", 3.0))
    except Exception:
        max_load = 3.0
    media["max_queen_load_1m"] = max_load

    try:
        min_mem = float(media.get("min_queen_mem_gb", 0.8))
    except Exception:
        min_mem = 0.8
    media["min_queen_mem_gb"] = max(0.0, min_mem)

    media["disable_visualization"] = bool(media.get("disable_visualization", False))
    media["allow_when_queen_bbox_active"] = bool(media.get("allow_when_queen_bbox_active", False))

    if enable is not None:
        media["enabled"] = bool(enable)
    else:
        media["enabled"] = bool(media.get("enabled", False))
    return media


def recommended_media_max_videos_total(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    minimum: int = 1,
) -> int:
    workers = _workers_from_config(config, include_disabled=include_disabled)
    return max(int(minimum), len(workers))


def sync_media_capacity_to_workers(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    minimum: int = 1,
) -> int:
    media = apply_queen_media_schedule_defaults(config, enable=None)
    target = recommended_media_max_videos_total(config, include_disabled=include_disabled, minimum=minimum)
    media["max_videos_total"] = target
    return target


def _resolve_identity_file(config: dict[str, Any], override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    fleet = _fleet_section(config)
    ssh_cfg = fleet.setdefault("ssh", {})
    return Path(str(ssh_cfg.get("identity_file") or _default_identity_file())).expanduser()


def _ensure_public_key(private_key: Path) -> Path:
    public_key = Path(str(private_key) + ".pub")
    if public_key.exists() and public_key.is_file():
        return public_key

    if not private_key.exists():
        raise FileNotFoundError(f"Private key not found: {private_key}")

    proc = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(private_key)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed generating public key from private key ({private_key}): {proc.stderr.strip()}"
        )
    public_key.write_text(proc.stdout.strip() + "\n")
    return public_key


def ensure_ssh_keypair(identity_file: str | Path) -> tuple[Path, Path, bool]:
    private_key = Path(identity_file).expanduser()
    public_key = Path(str(private_key) + ".pub")
    private_key.parent.mkdir(parents=True, exist_ok=True)

    if private_key.exists() and public_key.exists():
        return private_key, public_key, False

    if private_key.exists() and not public_key.exists():
        generated = _ensure_public_key(private_key)
        return private_key, generated, False

    proc = subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(private_key),
            "-C",
            "bbx-fleet",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to create SSH keypair with ssh-keygen: "
            + (proc.stderr.strip() or proc.stdout.strip())
        )
    if not public_key.exists():
        public_key = _ensure_public_key(private_key)
    return private_key, public_key, True


def _build_manual_key_install_command(public_key_file: Path, user: str, host: str, port: int) -> str:
    quoted_pub = shlex.quote(str(public_key_file))
    remote = shlex.quote(f"{user}@{host}")
    return (
        f"cat {quoted_pub} | ssh -p {int(port)} {remote} "
        "\"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        "cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys\""
    )


def initialize_queen_config(
    config: dict[str, Any],
    queen_host: Optional[str] = None,
    identity_file: Optional[str] = None,
    ssh_user: Optional[str] = None,
    skip_keygen: bool = False,
) -> tuple[dict[str, Any], FleetInitResult]:
    updated = deepcopy(config)
    fleet = _fleet_section(updated)
    fleet["role"] = "queen"
    fleet["queen_host"] = queen_host or str(fleet.get("queen_host") or _default_queen_host())
    apply_queen_media_schedule_defaults(updated, enable=None)
    sync_media_capacity_to_workers(updated, include_disabled=False, minimum=1)

    ssh_cfg = fleet.setdefault("ssh", {})
    if ssh_user:
        ssh_cfg["user"] = ssh_user
    private_key = _resolve_identity_file(updated, identity_file)
    ssh_cfg["identity_file"] = str(private_key)

    if skip_keygen:
        public_key = Path(str(private_key) + ".pub")
        if not public_key.exists():
            public_key = _ensure_public_key(private_key)
        key_created = False
    else:
        private_key, public_key, key_created = ensure_ssh_keypair(private_key)

    key_text = public_key.read_text().strip() if public_key.exists() else ""
    sample_user = str(ssh_cfg.get("user", "pi"))
    sample_port = int(ssh_cfg.get("port", 22))
    sample_host = "worker-host-or-ip"
    key_install = _build_manual_key_install_command(
        public_key_file=public_key,
        user=sample_user,
        host=sample_host,
        port=sample_port,
    )

    queen_hint = (
        "Queen chrony hint (run as root): add to /etc/chrony/chrony.conf -> "
        "'allow 192.168.0.0/16' (adjust subnet), then restart chronyd."
    )
    worker_hint = (
        f"Worker chrony hint (run as root): set 'server {fleet['queen_host']} iburst' "
        "in /etc/chrony/chrony.conf, then restart chronyd."
    )

    result = FleetInitResult(
        role="queen",
        queen_host=str(fleet["queen_host"]),
        identity_file=str(private_key),
        public_key_file=str(public_key),
        public_key=key_text,
        key_created=key_created,
        key_install_command_example=key_install,
        chrony_queen_hint=queen_hint,
        chrony_worker_hint=worker_hint,
    )
    return updated, result


def _worker_name_from_host(host: str) -> str:
    text = str(host).strip().replace(" ", "-")
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "worker"


def _upsert_worker(workers: list[dict[str, Any]], new_worker: dict[str, Any]) -> bool:
    for idx, row in enumerate(workers):
        if not isinstance(row, dict):
            continue
        same_name = str(row.get("name", "")).strip() == str(new_worker.get("name", "")).strip()
        same_host = str(row.get("host", "")).strip() == str(new_worker.get("host", "")).strip()
        if same_name or same_host:
            workers[idx] = new_worker
            return True
    workers.append(new_worker)
    return False


def enroll_worker_config(
    config: dict[str, Any],
    host: str,
    name: Optional[str] = None,
    user: Optional[str] = None,
    port: Optional[int] = None,
    data_root: str = "/mnt/bumblebox/data",
    unit_prefix: str = "bumblebox-v2",
    enabled: bool = True,
    identity_file: Optional[str] = None,
    install_key: bool = False,
) -> tuple[dict[str, Any], FleetEnrollResult]:
    updated = deepcopy(config)
    fleet = _fleet_section(updated)
    fleet["role"] = "queen"

    ssh_cfg = fleet.setdefault("ssh", {})
    resolved_user = user or str(ssh_cfg.get("user") or "pi")
    resolved_port = int(port if port is not None else ssh_cfg.get("port", 22))
    resolved_name = name or _worker_name_from_host(host)

    worker_row = {
        "name": resolved_name,
        "host": str(host).strip(),
        "user": resolved_user,
        "port": resolved_port,
        "enabled": bool(enabled),
        "data_root": data_root,
        "unit_prefix": unit_prefix,
    }
    workers = fleet.setdefault("workers", [])
    updated_existing = _upsert_worker(workers, worker_row)
    media_capacity = sync_media_capacity_to_workers(updated, include_disabled=False, minimum=1)
    enabled_worker_count = recommended_media_max_videos_total(updated, include_disabled=False, minimum=0)

    private_key = _resolve_identity_file(updated, identity_file)
    public_key = Path(str(private_key) + ".pub")
    if not public_key.exists():
        private_key, public_key, _created = ensure_ssh_keypair(private_key)
    fleet["ssh"]["identity_file"] = str(private_key)

    install_ok = False
    install_message = "Key install not attempted."
    if install_key:
        try:
            proc = subprocess.run(
                [
                    "ssh-copy-id",
                    "-i",
                    str(public_key),
                    "-p",
                    str(resolved_port),
                    f"{resolved_user}@{host}",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            install_ok = proc.returncode == 0
            install_message = (
                "ssh-copy-id succeeded."
                if install_ok
                else f"ssh-copy-id failed: {(proc.stderr or proc.stdout).strip()}"
            )
        except FileNotFoundError:
            install_message = "ssh-copy-id not found. Use manual key install command."
        except subprocess.TimeoutExpired:
            install_message = "ssh-copy-id timed out. Use manual key install command."
        except Exception as exc:
            install_message = f"ssh-copy-id error: {exc}"

    manual_command = _build_manual_key_install_command(
        public_key_file=public_key,
        user=resolved_user,
        host=host,
        port=resolved_port,
    )

    result = FleetEnrollResult(
        worker_name=resolved_name,
        worker_host=str(host),
        worker_user=resolved_user,
        worker_port=resolved_port,
        updated_existing=updated_existing,
        install_key_attempted=bool(install_key),
        install_key_ok=install_ok,
        install_key_message=install_message,
        manual_key_install_command=manual_command,
        enabled_worker_count=enabled_worker_count,
        queen_media_max_videos_total=media_capacity,
    )
    return updated, result


def _workers_from_config(config: dict[str, Any], include_disabled: bool) -> list[FleetWorker]:
    fleet = _fleet_section(config)
    rows = fleet.get("workers", [])
    if not isinstance(rows, list):
        return []

    out: list[FleetWorker] = []
    default_user = str(fleet.get("ssh", {}).get("user", "pi"))
    default_port = int(fleet.get("ssh", {}).get("port", 22))
    for item in rows:
        if not isinstance(item, dict):
            continue
        enabled = bool(item.get("enabled", True))
        if (not include_disabled) and (not enabled):
            continue
        host = str(item.get("host", "")).strip()
        if not host:
            continue
        name = str(item.get("name") or _worker_name_from_host(host))
        user = str(item.get("user") or default_user)
        port = int(item.get("port", default_port))
        data_root = str(item.get("data_root") or "/mnt/bumblebox/data")
        unit_prefix = str(item.get("unit_prefix") or "bumblebox-v2")
        out.append(
            FleetWorker(
                name=name,
                host=host,
                user=user,
                port=port,
                enabled=enabled,
                data_root=data_root,
                unit_prefix=unit_prefix,
            )
        )
    return out


def _ssh_base_cmd(identity_file: Path, timeout_seconds: int, user: str, host: str, port: int) -> list[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(timeout_seconds)}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(int(port)),
    ]
    if identity_file:
        cmd.extend(["-i", str(identity_file)])
    cmd.append(f"{user}@{host}")
    return cmd


def _ssh_run(
    identity_file: Path,
    timeout_seconds: int,
    user: str,
    host: str,
    port: int,
    remote_cmd: str,
) -> subprocess.CompletedProcess:
    base = _ssh_base_cmd(identity_file, timeout_seconds, user, host, port)
    cmd = base + [remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=max(3, int(timeout_seconds) + 3))


def _parse_df_free_gb(df_output: str) -> Optional[float]:
    lines = [line.strip() for line in df_output.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    parts = lines[1].split()
    if len(parts) < 4:
        return None
    try:
        kb_free = float(parts[3])
    except ValueError:
        return None
    return kb_free / (1024.0 * 1024.0)


def _parse_mem_available_gb(meminfo_line: str) -> Optional[float]:
    parts = meminfo_line.split()
    if len(parts) < 2:
        return None
    try:
        kb = float(parts[1])
    except ValueError:
        return None
    return kb / (1024.0 * 1024.0)


def _parse_load_1m(loadavg_text: str) -> Optional[float]:
    parts = loadavg_text.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _parse_cpu_temp_c(raw: str) -> Optional[float]:
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    # Typical thermal_zone format is milli-Celsius.
    if value > 1000:
        return value / 1000.0
    return value


def _local_mem_available_gb() -> Optional[float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        for line in meminfo.read_text().splitlines():
            if not line.startswith("MemAvailable:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            kb = float(parts[1])
            return kb / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def _local_load_1m() -> Optional[float]:
    try:
        return float(os.getloadavg()[0])
    except Exception:
        return None


def _load_queen_track_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed": {}}
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return {"processed": {}}
        processed = payload.get("processed", {})
        if not isinstance(processed, dict):
            payload["processed"] = {}
        return payload
    except Exception:
        return {"processed": {}}


def _save_queen_track_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _normalize_box_preset(box_preset: Any) -> Optional[str]:
    if box_preset is None:
        return None
    text = str(box_preset).strip().lower()
    if text in {"none", "null", ""}:
        return None
    return str(box_preset)


def _list_remote_latest_recordings(
    worker: FleetWorker,
    identity_file: Path,
    timeout_seconds: int,
    limit: int,
) -> list[tuple[float, str]]:
    data_root_q = shlex.quote(worker.data_root)
    cmd = (
        f"find {data_root_q} -type f \\( -name '*.mp4' -o -name '*.mjpeg' -o -name '*.avi' \\) "
        "! -name '*_tracked*' ! -name '*_thermal_preview*' ! -name '*_side_by_side*' "
        "! -name '*_realsense_*' -printf '%T@ %p\\n' 2>/dev/null "
        f"| sort -nr | head -n {int(max(1, limit))} || true"
    )
    proc = _ssh_run(
        identity_file=identity_file,
        timeout_seconds=timeout_seconds,
        user=worker.user,
        host=worker.host,
        port=worker.port,
        remote_cmd=cmd,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(message or "Failed to list remote videos.")

    out: list[tuple[float, str]] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            epoch = float(parts[0])
        except ValueError:
            continue
        remote_path = parts[1].strip()
        if not remote_path:
            continue
        out.append((epoch, remote_path))
    return out


def _scp_pull_file(
    *,
    worker: FleetWorker,
    identity_file: Path,
    timeout_seconds: int,
    remote_path: str,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(timeout_seconds)}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "LogLevel=ERROR",
        "-P",
        str(int(worker.port)),
    ]
    if identity_file:
        cmd.extend(["-i", str(identity_file)])
    cmd.extend(
        [
            f"{worker.user}@{worker.host}:{remote_path}",
            str(local_path),
        ]
    )
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(10, int(timeout_seconds) + 30),
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(message or "scp copy failed.")


def _worker_output_root(output_root: Path, worker: FleetWorker) -> Path:
    return output_root / worker.name


def _worker_latest_dir(output_root: Path, worker: FleetWorker) -> Path:
    return _worker_output_root(output_root, worker) / "latest"


def _worker_archive_dir(output_root: Path, worker: FleetWorker) -> Path:
    return _worker_output_root(output_root, worker) / datetime.now().strftime("%Y-%m-%d")


def _latest_video_path(output_root: Path, worker: FleetWorker, preferred_suffix: Optional[str] = None) -> Path:
    suffix = str(preferred_suffix or ".mp4").strip().lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix not in {".mp4", ".mjpeg", ".avi"}:
        suffix = ".mp4"
    return _worker_latest_dir(output_root, worker) / f"latest_video{suffix}"


def _resolve_latest_video_path(output_root: Path, worker: FleetWorker, latest_meta: Optional[dict[str, Any]] = None) -> Path:
    if latest_meta is not None:
        from_meta = str(latest_meta.get("latest_video_path", "")).strip()
        if from_meta:
            candidate = Path(from_meta).expanduser()
            if candidate.exists():
                return candidate
    candidate_mp4 = _latest_video_path(output_root, worker, ".mp4")
    if candidate_mp4.exists():
        return candidate_mp4
    candidate_mjpeg = _latest_video_path(output_root, worker, ".mjpeg")
    if candidate_mjpeg.exists():
        return candidate_mjpeg
    candidate_avi = _latest_video_path(output_root, worker, ".avi")
    if candidate_avi.exists():
        return candidate_avi
    return candidate_mp4


def _latest_tracked_video_path(output_root: Path, worker: FleetWorker) -> Path:
    return _worker_latest_dir(output_root, worker) / "latest_tracked.mp4"


def _latest_video_meta_path(output_root: Path, worker: FleetWorker) -> Path:
    return _worker_latest_dir(output_root, worker) / "latest_video.json"


def _latest_tracked_meta_path(output_root: Path, worker: FleetWorker) -> Path:
    return _worker_latest_dir(output_root, worker) / "latest_tracked.json"


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _try_parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    text = str(ts).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _minutes_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    left = _try_parse_iso(a)
    right = _try_parse_iso(b)
    if left is None or right is None:
        return None
    return (left - right).total_seconds() / 60.0


def _is_ip_address(text: str) -> bool:
    try:
        ip_address(text.strip())
        return True
    except Exception:
        return False


def _probe_tcp_port(host: str, port: int = 22, timeout_seconds: float = 0.6) -> Optional[bool]:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_seconds):
            return True
    except (ConnectionRefusedError, OSError):
        return False
    except Exception:
        return None


def _probe_ping(host: str, timeout_seconds: float = 0.6) -> Optional[bool]:
    timeout_ms = max(1, int(timeout_seconds * 1000))
    candidates = [
        ["ping", "-c", "1", "-W", str(max(1, int(timeout_seconds))), host],
        ["ping", "-c", "1", "-W", str(timeout_ms), host],
    ]
    for command in candidates:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=max(2.0, timeout_seconds + 1.0))
        except FileNotFoundError:
            return None
        except Exception:
            continue
        return proc.returncode == 0
    return None


def _collect_arp_neighbors() -> dict[str, tuple[Optional[str], Optional[str]]]:
    out: dict[str, tuple[Optional[str], Optional[str]]] = {}

    def add(ip_text: str, mac: Optional[str], state: Optional[str]) -> None:
        key = ip_text.strip()
        if not key:
            return
        out[key] = (mac, state)

    try:
        proc = subprocess.run(["ip", "neigh", "show"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        proc = None
    except Exception:
        proc = None

    if proc is not None and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) < 1:
                continue
            ip_text = parts[0]
            mac = None
            state = parts[-1] if parts else None
            if "lladdr" in parts:
                idx = parts.index("lladdr")
                if idx + 1 < len(parts):
                    mac = parts[idx + 1]
            add(ip_text, mac, state)

    try:
        proc = subprocess.run(["arp", "-an"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        proc = None
    except Exception:
        proc = None

    if proc is not None and proc.returncode == 0:
        regex = re.compile(r"\((?P<ip>[0-9.]+)\)\s+at\s+(?P<mac>[0-9a-f:]+|\(incomplete\))", re.IGNORECASE)
        for line in proc.stdout.splitlines():
            match = regex.search(line)
            if not match:
                continue
            ip_text = match.group("ip")
            mac = match.group("mac")
            if mac.lower() == "(incomplete)":
                mac = None
            add(ip_text, mac, out.get(ip_text, (None, None))[1])
    return out


def _severity_to_status(value: int) -> str:
    if value >= 2:
        return "FAIL"
    if value == 1:
        return "WARN"
    return "PASS"


def _collect_worker_status(
    worker: FleetWorker,
    identity_file: Path,
    timeout_seconds: int,
    queen_epoch: float,
) -> FleetWorkerStatus:
    notes: list[str] = []
    severity = 0

    try:
        probe = _ssh_run(
            identity_file=identity_file,
            timeout_seconds=timeout_seconds,
            user=worker.user,
            host=worker.host,
            port=worker.port,
            remote_cmd="echo ok",
        )
    except subprocess.TimeoutExpired:
        return FleetWorkerStatus(
            worker_name=worker.name,
            host=worker.host,
            user=worker.user,
            status="FAIL",
            reachable=False,
            time_offset_seconds=None,
            disk_free_gb_root=None,
            mem_available_gb=None,
            cpu_temp_c=None,
            load_1m=None,
            bumblebox_timer_count=0,
            latest_run_summary=None,
            notes=[],
            error="SSH timed out.",
        )
    except Exception as exc:
        return FleetWorkerStatus(
            worker_name=worker.name,
            host=worker.host,
            user=worker.user,
            status="FAIL",
            reachable=False,
            time_offset_seconds=None,
            disk_free_gb_root=None,
            mem_available_gb=None,
            cpu_temp_c=None,
            load_1m=None,
            bumblebox_timer_count=0,
            latest_run_summary=None,
            notes=[],
            error=f"SSH probe failed: {exc}",
        )

    if probe.returncode != 0:
        return FleetWorkerStatus(
            worker_name=worker.name,
            host=worker.host,
            user=worker.user,
            status="FAIL",
            reachable=False,
            time_offset_seconds=None,
            disk_free_gb_root=None,
            mem_available_gb=None,
            cpu_temp_c=None,
            load_1m=None,
            bumblebox_timer_count=0,
            latest_run_summary=None,
            notes=[],
            error=(probe.stderr or probe.stdout).strip() or "SSH probe failed.",
        )

    def safe_run(command: str) -> Optional[str]:
        try:
            proc = _ssh_run(
                identity_file=identity_file,
                timeout_seconds=timeout_seconds,
                user=worker.user,
                host=worker.host,
                port=worker.port,
                remote_cmd=command,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    remote_epoch_text = safe_run("date +%s")
    time_offset = None
    if remote_epoch_text:
        try:
            remote_epoch = float(remote_epoch_text.splitlines()[0].strip())
            time_offset = remote_epoch - queen_epoch
            if abs(time_offset) > 30:
                severity = max(severity, 2)
                notes.append(f"Clock offset is high ({time_offset:+.1f}s).")
            elif abs(time_offset) > 5:
                severity = max(severity, 1)
                notes.append(f"Clock offset is elevated ({time_offset:+.1f}s).")
        except Exception:
            notes.append("Could not parse remote clock.")
            severity = max(severity, 1)
    else:
        notes.append("Could not read remote clock.")
        severity = max(severity, 1)

    disk_output = safe_run("df -Pk /")
    disk_free_gb = _parse_df_free_gb(disk_output or "")
    if disk_free_gb is not None:
        if disk_free_gb < 1.0:
            severity = max(severity, 2)
            notes.append(f"Low disk space on '/': {disk_free_gb:.2f} GB free.")
        elif disk_free_gb < 3.0:
            severity = max(severity, 1)
            notes.append(f"Disk space getting low on '/': {disk_free_gb:.2f} GB free.")
    else:
        notes.append("Could not read disk free space.")
        severity = max(severity, 1)

    mem_line = safe_run("grep MemAvailable /proc/meminfo")
    mem_available_gb = _parse_mem_available_gb(mem_line or "")
    if mem_available_gb is not None:
        if mem_available_gb < 0.3:
            severity = max(severity, 2)
            notes.append(f"Very low available memory: {mem_available_gb:.2f} GB.")
        elif mem_available_gb < 0.8:
            severity = max(severity, 1)
            notes.append(f"Available memory is limited: {mem_available_gb:.2f} GB.")
    else:
        notes.append("Could not read available memory.")
        severity = max(severity, 1)

    temp_text = safe_run("cat /sys/class/thermal/thermal_zone0/temp")
    cpu_temp_c = _parse_cpu_temp_c(temp_text or "")
    if cpu_temp_c is not None:
        if cpu_temp_c >= 85.0:
            severity = max(severity, 2)
            notes.append(f"CPU temperature high: {cpu_temp_c:.1f}C.")
        elif cpu_temp_c >= 78.0:
            severity = max(severity, 1)
            notes.append(f"CPU temperature elevated: {cpu_temp_c:.1f}C.")

    load_text = safe_run("cat /proc/loadavg")
    load_1m = _parse_load_1m(load_text or "")
    if load_1m is not None:
        if load_1m >= 8.0:
            severity = max(severity, 2)
            notes.append(f"System load is very high: {load_1m:.2f}.")
        elif load_1m >= 4.0:
            severity = max(severity, 1)
            notes.append(f"System load is high: {load_1m:.2f}.")

    timers_text = safe_run("systemctl list-timers --all --no-pager 2>/dev/null | grep -i bumblebox || true")
    timer_lines = [line for line in (timers_text or "").splitlines() if line.strip()]
    timer_count = len(timer_lines)
    if timer_count == 0:
        severity = max(severity, 1)
        notes.append("No bumblebox timers found via systemctl list-timers.")

    data_root_q = shlex.quote(worker.data_root)
    latest_text = safe_run(
        f"find {data_root_q} -name '*_run_summary.json' -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -n1 || true"
    )
    latest_summary = None
    if latest_text:
        parts = latest_text.split(maxsplit=1)
        if len(parts) == 2:
            latest_summary = parts[1]
            try:
                age_sec = time.time() - float(parts[0])
                if age_sec > (3 * 24 * 3600):
                    severity = max(severity, 1)
                    notes.append("No run summary updated in the last 3 days.")
            except Exception:
                pass
    else:
        severity = max(severity, 1)
        notes.append(f"No run summary found under {worker.data_root}.")

    return FleetWorkerStatus(
        worker_name=worker.name,
        host=worker.host,
        user=worker.user,
        status=_severity_to_status(severity),
        reachable=True,
        time_offset_seconds=time_offset,
        disk_free_gb_root=disk_free_gb,
        mem_available_gb=mem_available_gb,
        cpu_temp_c=cpu_temp_c,
        load_1m=load_1m,
        bumblebox_timer_count=timer_count,
        latest_run_summary=latest_summary,
        notes=notes,
        error=None,
    )


def run_fleet_status(
    config: dict[str, Any],
    include_disabled: bool = False,
    worker_filter: Optional[str] = None,
    identity_file: Optional[str] = None,
) -> FleetStatusReport:
    fleet = _fleet_section(config)
    role = str(fleet.get("role", "standalone"))
    queen_host = str(fleet.get("queen_host") or _default_queen_host())
    workers = _workers_from_config(config, include_disabled=include_disabled)

    if worker_filter:
        needle = worker_filter.strip().lower()
        workers = [
            item
            for item in workers
            if needle in item.name.lower() or needle in item.host.lower()
        ]

    ssh_cfg = fleet.get("ssh", {})
    timeout = int(ssh_cfg.get("connect_timeout_seconds", 5))
    identity = _resolve_identity_file(config, identity_file)
    queen_epoch = time.time()

    statuses: list[FleetWorkerStatus] = []
    for worker in workers:
        statuses.append(
            _collect_worker_status(
                worker=worker,
                identity_file=identity,
                timeout_seconds=timeout,
                queen_epoch=queen_epoch,
            )
        )

    pass_count = sum(1 for item in statuses if item.status == "PASS")
    warn_count = sum(1 for item in statuses if item.status == "WARN")
    fail_count = sum(1 for item in statuses if item.status == "FAIL")

    return FleetStatusReport(
        generated_at=_now_iso(),
        queen_host=queen_host,
        role=role,
        worker_count=len(statuses),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        workers=statuses,
    )


def run_queen_pull_latest_videos(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    worker_filter: Optional[str] = None,
    identity_file: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    max_videos_total: int = 200,
    dry_run: bool = False,
) -> QueenTrackReport:
    if max_videos_total <= 0:
        raise ValueError("max_videos_total must be >= 1")

    fleet = _fleet_section(config)
    role = str(fleet.get("role", "standalone")).strip().lower()
    if role != "queen":
        raise ValueError("Queen pull-latest is only available when fleet.role='queen'.")

    queen_local_pipeline_enabled = bool(fleet.get("queen_local_pipeline_enabled", True))
    if output_root is None:
        output_root_path = Path(config["system"]["data_root"]).expanduser().resolve() / "queen_worker_tracking"
    else:
        output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    report_path = output_root_path / "queen_pull_latest_last_report.json"
    workers = _workers_from_config(config, include_disabled=include_disabled)
    if worker_filter:
        needle = worker_filter.strip().lower()
        workers = [
            item
            for item in workers
            if needle in item.name.lower() or needle in item.host.lower()
        ]

    report = QueenTrackReport(
        generated_at=_now_iso(),
        role=role,
        queen_local_pipeline_enabled=queen_local_pipeline_enabled,
        output_root=str(output_root_path),
        state_path=str(report_path),
        workers_considered=len(workers),
        videos_considered=0,
        videos_processed=0,
        videos_skipped=0,
        videos_failed=0,
        skipped_due_to_guard=False,
        guard_reason=None,
        queen_load_1m=_local_load_1m(),
        queen_mem_available_gb=_local_mem_available_gb(),
        items=[],
    )

    ssh_cfg = fleet.get("ssh", {})
    timeout = int(ssh_cfg.get("connect_timeout_seconds", 5))
    identity = _resolve_identity_file(config, identity_file)

    processed = 0
    for worker in workers:
        if processed >= max_videos_total:
            break

        report.videos_considered += 1
        latest_meta_path = _latest_video_meta_path(output_root_path, worker)
        old_meta = _read_json_if_exists(latest_meta_path)
        latest_video = _resolve_latest_video_path(output_root_path, worker, old_meta)
        old_pulled_at = str(old_meta.get("pulled_at", "")).strip() or None
        latest_tracked = _latest_tracked_video_path(output_root_path, worker)
        latest_tracked_meta_path = _latest_tracked_meta_path(output_root_path, worker)
        latest_tracked_meta = _read_json_if_exists(latest_tracked_meta_path)
        latest_tracked_at = str(latest_tracked_meta.get("tracked_at", "")).strip() or None

        try:
            candidates = _list_remote_latest_recordings(
                worker=worker,
                identity_file=identity,
                timeout_seconds=timeout,
                limit=1,
            )
        except Exception as exc:
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path="",
                    local_video_path=None,
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video) if latest_video.exists() else None,
                    latest_video_pulled_at=old_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="failed",
                    note=f"Could not list worker videos: {exc}",
                )
            )
            report.videos_failed += 1
            continue

        if not candidates:
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path="",
                    local_video_path=None,
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video) if latest_video.exists() else None,
                    latest_video_pulled_at=old_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="skipped_no_video",
                    note=f"No recording videos (.mp4/.mjpeg) found under {worker.data_root}.",
                )
            )
            report.videos_skipped += 1
            continue

        remote_epoch, remote_video = candidates[0]
        old_remote = str(old_meta.get("remote_video_path", "")).strip()

        remote_suffix = Path(remote_video).suffix
        latest_video_target = _latest_video_path(output_root_path, worker, preferred_suffix=remote_suffix)

        if old_remote == remote_video and latest_video_target.exists():
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=remote_video,
                    local_video_path=str(latest_video_target),
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video_target),
                    latest_video_pulled_at=old_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="up_to_date",
                    note="Latest video already pulled.",
                )
            )
            report.videos_skipped += 1
            continue

        archive_dir = _worker_archive_dir(output_root_path, worker)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / Path(remote_video).name
        if archive_path.exists():
            archive_path = archive_dir / f"{archive_path.stem}_{int(time.time())}{archive_path.suffix}"

        if dry_run:
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=remote_video,
                    local_video_path=str(archive_path),
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video),
                    latest_video_pulled_at=old_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="planned_pull",
                    note="Dry run only. No files copied.",
                )
            )
            processed += 1
            continue

        try:
            _scp_pull_file(
                worker=worker,
                identity_file=identity,
                timeout_seconds=timeout,
                remote_path=remote_video,
                local_path=archive_path,
            )
            latest_video_target.parent.mkdir(parents=True, exist_ok=True)
            for stale in (
                _latest_video_path(output_root_path, worker, ".mp4"),
                _latest_video_path(output_root_path, worker, ".mjpeg"),
                _latest_video_path(output_root_path, worker, ".avi"),
            ):
                if stale != latest_video_target and stale.exists():
                    try:
                        stale.unlink()
                    except Exception:
                        pass
            shutil.copy2(archive_path, latest_video_target)

            pulled_at = _now_iso()
            latest_meta = {
                "worker_name": worker.name,
                "worker_host": worker.host,
                "remote_video_path": remote_video,
                "remote_video_epoch": remote_epoch,
                "pulled_at": pulled_at,
                "local_archive_path": str(archive_path),
                "latest_video_path": str(latest_video_target),
            }
            _write_json(latest_meta_path, latest_meta)

            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=remote_video,
                    local_video_path=str(archive_path),
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video_target),
                    latest_video_pulled_at=pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="pulled",
                    note="Pulled latest worker video and updated latest_video pointer.",
                )
            )
            report.videos_processed += 1
            processed += 1
        except Exception as exc:
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=remote_video,
                    local_video_path=str(archive_path),
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video),
                    latest_video_pulled_at=old_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="failed",
                    note=f"Pull failed: {exc}",
                )
            )
            report.videos_failed += 1

    if not dry_run:
        _write_json(report_path, report.to_dict())
    return report


def run_queen_track_latest_videos(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    worker_filter: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    max_videos_total: int = 200,
    cooldown_minutes: int = 60,
    with_visualization: bool = True,
    dry_run: bool = False,
    allow_when_queen_bbox_active: bool = False,
    max_queen_load_1m: Optional[float] = 3.0,
    min_queen_mem_available_gb: Optional[float] = 0.8,
) -> QueenTrackReport:
    if max_videos_total <= 0:
        raise ValueError("max_videos_total must be >= 1")
    if cooldown_minutes < 0:
        raise ValueError("cooldown_minutes must be >= 0")

    fleet = _fleet_section(config)
    role = str(fleet.get("role", "standalone")).strip().lower()
    if role != "queen":
        raise ValueError("Queen track-latest is only available when fleet.role='queen'.")

    queen_local_pipeline_enabled = bool(fleet.get("queen_local_pipeline_enabled", True))
    if queen_local_pipeline_enabled and (not allow_when_queen_bbox_active):
        raise ValueError(
            "Queen local BumbleBox pipeline is active. "
            "Use --allow-when-queen-bbox-active or switch to --queen-interface-only."
        )

    queen_load_1m = _local_load_1m()
    queen_mem_available_gb = _local_mem_available_gb()
    guard_reasons: list[str] = []
    if max_queen_load_1m is not None and queen_load_1m is not None and queen_load_1m > float(max_queen_load_1m):
        guard_reasons.append(
            f"queen load 1m is {queen_load_1m:.2f}, above limit {float(max_queen_load_1m):.2f}"
        )
    if (
        min_queen_mem_available_gb is not None
        and queen_mem_available_gb is not None
        and queen_mem_available_gb < float(min_queen_mem_available_gb)
    ):
        guard_reasons.append(
            f"queen available memory is {queen_mem_available_gb:.2f} GB, below limit {float(min_queen_mem_available_gb):.2f} GB"
        )
    guard_reason = "; ".join(guard_reasons) if guard_reasons else None

    if output_root is None:
        output_root_path = Path(config["system"]["data_root"]).expanduser().resolve() / "queen_worker_tracking"
    else:
        output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)
    report_path = output_root_path / "queen_track_latest_last_report.json"

    workers = _workers_from_config(config, include_disabled=include_disabled)
    if worker_filter:
        needle = worker_filter.strip().lower()
        workers = [
            item
            for item in workers
            if needle in item.name.lower() or needle in item.host.lower()
        ]

    report = QueenTrackReport(
        generated_at=_now_iso(),
        role=role,
        queen_local_pipeline_enabled=queen_local_pipeline_enabled,
        output_root=str(output_root_path),
        state_path=str(report_path),
        workers_considered=len(workers),
        videos_considered=0,
        videos_processed=0,
        videos_skipped=0,
        videos_failed=0,
        skipped_due_to_guard=bool(guard_reason),
        guard_reason=guard_reason,
        queen_load_1m=queen_load_1m,
        queen_mem_available_gb=queen_mem_available_gb,
        items=[],
    )
    if guard_reason:
        return report

    if dry_run:
        track_tags_from_video = None
    else:
        try:
            from tag_tracking_utils import trackTagsFromVid as track_tags_from_video
        except Exception as exc:
            raise RuntimeError(f"Could not import trackTagsFromVid: {exc}") from exc

    render_tracking_video = None
    render_import_error: Optional[str] = None
    if with_visualization and (not dry_run):
        try:
            from bumblebox_desktop.visualization import render_tracking_video as _render_tracking_video

            render_tracking_video = _render_tracking_video
        except Exception as exc:
            render_import_error = str(exc)

    tag_dictionary = str(config.get("tracking", {}).get("tag_dictionary", "4X4_50"))
    box_preset = _normalize_box_preset(config.get("tracking", {}).get("box_preset"))
    colony_id = str(config.get("system", {}).get("colony_id", "queen"))
    aruco_params = config.get("tracking", {}).get("aruco_params")

    processed = 0
    for worker in workers:
        if processed >= max_videos_total:
            break

        report.videos_considered += 1
        latest_video_meta_path = _latest_video_meta_path(output_root_path, worker)
        latest_video_meta = _read_json_if_exists(latest_video_meta_path)
        latest_video = _resolve_latest_video_path(output_root_path, worker, latest_video_meta)
        latest_video_pulled_at = str(latest_video_meta.get("pulled_at", "")).strip() or None
        source_remote_video = str(latest_video_meta.get("remote_video_path", "")).strip() or ""
        latest_tracked = _latest_tracked_video_path(output_root_path, worker)
        latest_tracked_meta_path = _latest_tracked_meta_path(output_root_path, worker)
        latest_tracked_meta = _read_json_if_exists(latest_tracked_meta_path)
        latest_tracked_at = str(latest_tracked_meta.get("tracked_at", "")).strip() or None
        tracked_source_remote = str(latest_tracked_meta.get("source_remote_video_path", "")).strip() or ""
        tracked_source_latest = str(latest_tracked_meta.get("source_latest_video_path", "")).strip() or ""

        if not latest_video.exists():
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=source_remote_video,
                    local_video_path=None,
                    raw_csv_path=None,
                    tracked_video_path=None,
                    latest_video_path=str(latest_video),
                    latest_video_pulled_at=latest_video_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="skipped_missing_latest_video",
                    note="No latest pulled video found. Run fleet queen-pull-latest first.",
                )
            )
            report.videos_skipped += 1
            continue

        if cooldown_minutes > 0:
            same_source = False
            if source_remote_video and tracked_source_remote:
                same_source = source_remote_video == tracked_source_remote
            elif tracked_source_latest:
                same_source = str(latest_video) == tracked_source_latest

            tracked_epoch_raw = latest_tracked_meta.get("tracked_at_epoch")
            try:
                tracked_epoch = float(tracked_epoch_raw) if tracked_epoch_raw is not None else None
            except Exception:
                tracked_epoch = None
            if same_source and tracked_epoch is not None:
                age = time.time() - tracked_epoch
                if age < (cooldown_minutes * 60):
                    report.items.append(
                        QueenTrackItem(
                            worker_name=worker.name,
                            worker_host=worker.host,
                            remote_video_path=source_remote_video,
                            local_video_path=str(latest_video),
                            raw_csv_path=str(latest_tracked_meta.get("raw_csv_path", "")) or None,
                            tracked_video_path=str(latest_tracked_meta.get("tracked_video_archive_path", "")) or None,
                            latest_video_path=str(latest_video),
                            latest_video_pulled_at=latest_video_pulled_at,
                            latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                            latest_tracked_at=latest_tracked_at,
                            status="skipped_cooldown",
                            note=(
                                f"Skipped by cooldown for unchanged latest video "
                                f"({age / 60.0:.1f} minutes since last tracking, need {cooldown_minutes} minutes)."
                            ),
                        )
                    )
                    report.videos_skipped += 1
                    continue

        archive_dir = _worker_archive_dir(output_root_path, worker)
        archive_dir.mkdir(parents=True, exist_ok=True)
        session_base = f"{worker.name}_{latest_video.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_name = "".join(
            ch if (ch.isalnum() or ch in {"-", "_"}) else "_"
            for ch in session_base
        ).strip("_")
        if not session_name:
            session_name = f"{worker.name}_{int(time.time())}"

        if dry_run:
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=source_remote_video,
                    local_video_path=str(latest_video),
                    raw_csv_path=str(archive_dir / f"{session_name}_raw.csv"),
                    tracked_video_path=str(archive_dir / f"{session_name}_tracked.mp4"),
                    latest_video_path=str(latest_video),
                    latest_video_pulled_at=latest_video_pulled_at,
                    latest_tracked_video_path=str(latest_tracked),
                    latest_tracked_at=latest_tracked_at,
                    status="planned_track",
                    note="Dry run only. No tracking executed.",
                )
            )
            processed += 1
            continue

        raw_csv_path: Optional[str] = None
        tracked_archive_path: Optional[str] = None
        try:
            track_tags_from_video(
                str(latest_video),
                str(archive_dir),
                session_name,
                tag_dictionary,
                box_preset,
                _now_iso(),
                colony_id,
                aruco_params=aruco_params,
            )
            raw_csv_candidate = archive_dir / f"{session_name}_raw.csv"
            if raw_csv_candidate.exists():
                raw_csv_path = str(raw_csv_candidate)

            tracked_at = _now_iso()
            note = "Tracking completed."
            if with_visualization and raw_csv_candidate.exists() and render_tracking_video is not None:
                tracked_archive = archive_dir / f"{session_name}_tracked.mp4"
                render_tracking_video(
                    video_path=latest_video,
                    tracking_csv_path=raw_csv_candidate,
                    output_video_path=tracked_archive,
                )
                latest_tracked.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tracked_archive, latest_tracked)
                tracked_archive_path = str(tracked_archive)
            elif with_visualization and render_import_error:
                note += f" Visualization skipped (import failed: {render_import_error})."
            elif with_visualization and not raw_csv_candidate.exists():
                note += " Visualization skipped (raw CSV not found)."

            tracked_meta = {
                "worker_name": worker.name,
                "worker_host": worker.host,
                "tracked_at": tracked_at,
                "tracked_at_epoch": time.time(),
                "source_remote_video_path": source_remote_video,
                "source_latest_video_path": str(latest_video),
                "source_latest_video_pulled_at": latest_video_pulled_at,
                "raw_csv_path": raw_csv_path,
                "tracked_video_archive_path": tracked_archive_path,
                "latest_tracked_video_path": str(latest_tracked) if latest_tracked.exists() else None,
            }
            _write_json(latest_tracked_meta_path, tracked_meta)

            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=source_remote_video,
                    local_video_path=str(latest_video),
                    raw_csv_path=raw_csv_path,
                    tracked_video_path=tracked_archive_path,
                    latest_video_path=str(latest_video),
                    latest_video_pulled_at=latest_video_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=tracked_at,
                    status="tracked_latest",
                    note=note,
                )
            )
            report.videos_processed += 1
            processed += 1
        except Exception as exc:
            report.items.append(
                QueenTrackItem(
                    worker_name=worker.name,
                    worker_host=worker.host,
                    remote_video_path=source_remote_video,
                    local_video_path=str(latest_video),
                    raw_csv_path=raw_csv_path,
                    tracked_video_path=tracked_archive_path,
                    latest_video_path=str(latest_video),
                    latest_video_pulled_at=latest_video_pulled_at,
                    latest_tracked_video_path=str(latest_tracked) if latest_tracked.exists() else None,
                    latest_tracked_at=latest_tracked_at,
                    status="failed",
                    note=f"Track latest failed: {exc}",
                )
            )
            report.videos_failed += 1

    if not dry_run:
        _write_json(report_path, report.to_dict())
    return report


def run_queen_pull_track_latest(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    worker_filter: Optional[str] = None,
    identity_file: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    max_videos_total: int = 200,
    max_videos_per_worker: int = 1,
    cooldown_minutes: int = 60,
    with_visualization: bool = True,
    dry_run: bool = False,
    allow_when_queen_bbox_active: bool = False,
    max_queen_load_1m: Optional[float] = 3.0,
    min_queen_mem_available_gb: Optional[float] = 0.8,
) -> QueenTrackReport:
    pull_report = run_queen_pull_latest_videos(
        config=config,
        include_disabled=include_disabled,
        worker_filter=worker_filter,
        identity_file=identity_file,
        output_root=output_root,
        max_videos_total=max_videos_total,
        dry_run=dry_run,
    )
    track_report = run_queen_track_latest_videos(
        config=config,
        include_disabled=include_disabled,
        worker_filter=worker_filter,
        output_root=output_root,
        max_videos_total=max_videos_total,
        cooldown_minutes=cooldown_minutes,
        with_visualization=with_visualization,
        dry_run=dry_run,
        allow_when_queen_bbox_active=allow_when_queen_bbox_active,
        max_queen_load_1m=max_queen_load_1m,
        min_queen_mem_available_gb=min_queen_mem_available_gb,
    )

    combined = QueenTrackReport(
        generated_at=_now_iso(),
        role=track_report.role,
        queen_local_pipeline_enabled=track_report.queen_local_pipeline_enabled,
        output_root=track_report.output_root,
        state_path=track_report.state_path,
        workers_considered=max(pull_report.workers_considered, track_report.workers_considered),
        videos_considered=pull_report.videos_considered + track_report.videos_considered,
        videos_processed=pull_report.videos_processed + track_report.videos_processed,
        videos_skipped=pull_report.videos_skipped + track_report.videos_skipped,
        videos_failed=pull_report.videos_failed + track_report.videos_failed,
        skipped_due_to_guard=track_report.skipped_due_to_guard,
        guard_reason=track_report.guard_reason,
        queen_load_1m=track_report.queen_load_1m,
        queen_mem_available_gb=track_report.queen_mem_available_gb,
        items=pull_report.items + track_report.items,
    )
    return combined


def _resolve_queen_output_root(config: dict[str, Any], output_root: Optional[str | Path]) -> Path:
    if output_root is None:
        return Path(config["system"]["data_root"]).expanduser().resolve() / "queen_worker_tracking"
    return Path(output_root).expanduser().resolve()


def run_queen_latest_status(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    worker_filter: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    identity_file: Optional[str] = None,
    probe_reachability: bool = True,
) -> QueenLatestStatusReport:
    output_root_path = _resolve_queen_output_root(config, output_root)
    workers = _workers_from_config(config, include_disabled=include_disabled)
    if worker_filter:
        needle = worker_filter.strip().lower()
        workers = [
            item
            for item in workers
            if needle in item.name.lower() or needle in item.host.lower()
        ]

    online_map: dict[tuple[str, str], bool] = {}
    if probe_reachability:
        try:
            status = run_fleet_status(
                config=config,
                include_disabled=include_disabled,
                worker_filter=worker_filter,
                identity_file=identity_file,
            )
            for item in status.workers:
                online_map[(item.worker_name, item.host)] = bool(item.reachable)
        except Exception:
            online_map = {}

    items: list[QueenLatestStatusItem] = []
    synced_count = 0
    stale_count = 0
    missing_latest_video_count = 0
    missing_latest_tracked_count = 0
    offline_count = 0

    for worker in workers:
        latest_video_meta = _read_json_if_exists(_latest_video_meta_path(output_root_path, worker))
        latest_video = _resolve_latest_video_path(output_root_path, worker, latest_video_meta)
        latest_tracked = _latest_tracked_video_path(output_root_path, worker)
        latest_tracked_meta = _read_json_if_exists(_latest_tracked_meta_path(output_root_path, worker))

        latest_video_pulled_at = str(latest_video_meta.get("pulled_at", "")).strip() or None
        latest_tracked_at = str(latest_tracked_meta.get("tracked_at", "")).strip() or None
        latest_video_exists = latest_video.exists()
        latest_tracked_exists = latest_tracked.exists()
        source_remote_video = str(latest_video_meta.get("remote_video_path", "")).strip() or ""
        tracked_source_remote = str(latest_tracked_meta.get("source_remote_video_path", "")).strip() or ""
        tracked_source_latest = str(latest_tracked_meta.get("source_latest_video_path", "")).strip() or ""
        track_lag_minutes = _minutes_between(latest_video_pulled_at, latest_tracked_at)

        synced = False
        if latest_video_exists and latest_tracked_exists and latest_tracked_at:
            if source_remote_video and tracked_source_remote:
                synced = source_remote_video == tracked_source_remote
            elif tracked_source_latest:
                synced = tracked_source_latest == str(latest_video)

        if not latest_video_exists:
            status = "missing_latest_video"
            note = "No latest pulled video found for this worker."
            missing_latest_video_count += 1
        elif not latest_tracked_exists:
            status = "missing_latest_tracked"
            note = "No latest tracked video found yet."
            missing_latest_tracked_count += 1
        elif synced:
            status = "synced"
            note = "Latest tracked video is in sync with latest pulled video."
            synced_count += 1
        else:
            status = "stale_tracked"
            note = "Latest tracked video is older than the latest pulled video."
            stale_count += 1

        online = online_map.get((worker.name, worker.host))
        if online is False:
            offline_count += 1
            status = f"{status}_offline"
            note += " Worker is currently offline/unreachable."

        items.append(
            QueenLatestStatusItem(
                worker_name=worker.name,
                worker_host=worker.host,
                latest_video_path=str(latest_video) if latest_video_exists else None,
                latest_video_pulled_at=latest_video_pulled_at,
                latest_tracked_video_path=str(latest_tracked) if latest_tracked_exists else None,
                latest_tracked_at=latest_tracked_at,
                status=status,
                online=online,
                track_lag_minutes=track_lag_minutes,
                note=note,
            )
        )

    return QueenLatestStatusReport(
        generated_at=_now_iso(),
        output_root=str(output_root_path),
        workers_considered=len(workers),
        synced_count=synced_count,
        stale_count=stale_count,
        missing_latest_video_count=missing_latest_video_count,
        missing_latest_tracked_count=missing_latest_tracked_count,
        offline_count=offline_count,
        items=items,
    )


def run_fleet_discovery(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
    worker_filter: Optional[str] = None,
    ping_probe: bool = True,
    timeout_seconds: float = 0.6,
) -> FleetDiscoveryReport:
    workers = _workers_from_config(config, include_disabled=include_disabled)
    if worker_filter:
        needle = worker_filter.strip().lower()
        workers = [
            item
            for item in workers
            if needle in item.name.lower() or needle in item.host.lower()
        ]

    worker_by_host: dict[str, FleetWorker] = {item.host: item for item in workers}
    arp_map = _collect_arp_neighbors()

    candidates: set[str] = set(worker_by_host.keys())
    for host in arp_map:
        if _is_ip_address(host):
            candidates.add(host)

    def sort_key(host: str) -> tuple[int, str]:
        if _is_ip_address(host):
            try:
                return (0, str(ip_address(host)))
            except Exception:
                return (0, host)
        return (1, host)

    items: list[FleetDiscoveryItem] = []
    configured_workers_online = 0
    configured_workers_offline = 0
    default_port = int(config.get("fleet", {}).get("ssh", {}).get("port", 22))

    for host in sorted(candidates, key=sort_key):
        worker = worker_by_host.get(host)
        probe_port = int(worker.port) if worker is not None else default_port
        ping_ok = _probe_ping(host, timeout_seconds=timeout_seconds) if ping_probe else None
        ssh_port_open = _probe_tcp_port(host, port=probe_port, timeout_seconds=timeout_seconds)
        reachable = bool((ping_ok is True) or (ssh_port_open is True))
        arp_mac, arp_state = arp_map.get(host, (None, None))

        if worker is not None:
            if reachable:
                configured_workers_online += 1
            else:
                configured_workers_offline += 1

        items.append(
            FleetDiscoveryItem(
                host=host,
                from_config=worker is not None,
                worker_name=worker.name if worker is not None else None,
                ping_ok=ping_ok,
                ssh_port_open=ssh_port_open,
                reachable=reachable,
                arp_mac=arp_mac,
                arp_state=arp_state,
            )
        )

    return FleetDiscoveryReport(
        generated_at=_now_iso(),
        configured_workers=len(workers),
        configured_workers_online=configured_workers_online,
        configured_workers_offline=configured_workers_offline,
        discovered_hosts=len(items),
        items=items,
    )


def format_fleet_init_result(result: FleetInitResult, show_public_key: bool = False) -> str:
    lines = [
        "Fleet Init (Queen)",
        "------------------",
        f"Role: {result.role}",
        f"Queen host: {result.queen_host}",
        f"Identity file: {result.identity_file}",
        f"Public key file: {result.public_key_file}",
        f"SSH key created now: {result.key_created}",
        "",
        "Install key on workers (example):",
        result.key_install_command_example,
        "",
        result.chrony_queen_hint,
        result.chrony_worker_hint,
    ]
    if show_public_key:
        lines.extend(["", "Public key:", result.public_key])
    return "\n".join(lines)


def format_fleet_enroll_result(result: FleetEnrollResult) -> str:
    lines = [
        "Fleet Enroll Worker",
        "-------------------",
        f"Worker: {result.worker_name}",
        f"Host: {result.worker_host}",
        f"SSH: {result.worker_user}@{result.worker_host}:{result.worker_port}",
        f"Updated existing entry: {result.updated_existing}",
        f"Install key attempted: {result.install_key_attempted}",
        f"Install key success: {result.install_key_ok}",
        f"Install key message: {result.install_key_message}",
        f"Enabled workers in config: {result.enabled_worker_count}",
        f"Queen media max_videos_total set to: {result.queen_media_max_videos_total}",
        "",
        "Manual key-install command:",
        result.manual_key_install_command,
    ]
    return "\n".join(lines)


def format_fleet_status_report(report: FleetStatusReport) -> str:
    lines = [
        "Fleet Status",
        "------------",
        f"Generated: {report.generated_at}",
        f"Role: {report.role}",
        f"Queen host: {report.queen_host}",
        (
            f"Workers checked: {report.worker_count} "
            f"(pass={report.pass_count}, warn={report.warn_count}, fail={report.fail_count})"
        ),
    ]

    for item in report.workers:
        lines.append("")
        lines.append(f"[{item.status}] {item.worker_name} ({item.user}@{item.host})")
        lines.append(f"- reachable: {item.reachable}")
        lines.append(f"- time_offset_seconds: {item.time_offset_seconds}")
        lines.append(f"- disk_free_gb_root: {item.disk_free_gb_root}")
        lines.append(f"- mem_available_gb: {item.mem_available_gb}")
        lines.append(f"- cpu_temp_c: {item.cpu_temp_c}")
        lines.append(f"- load_1m: {item.load_1m}")
        lines.append(f"- bumblebox_timer_count: {item.bumblebox_timer_count}")
        lines.append(f"- latest_run_summary: {item.latest_run_summary}")
        if item.error:
            lines.append(f"- error: {item.error}")
        if item.notes:
            lines.append("- notes:")
            for note in item.notes:
                lines.append(f"  - {note}")

    return "\n".join(lines)


def format_queen_track_report(report: QueenTrackReport) -> str:
    lines = [
        "Queen Latest Media Report",
        "-------------------------",
        f"Generated: {report.generated_at}",
        f"Role: {report.role}",
        f"Queen local pipeline enabled: {report.queen_local_pipeline_enabled}",
        f"Output root: {report.output_root}",
        f"State file: {report.state_path}",
        f"Workers considered: {report.workers_considered}",
        (
            f"Videos: considered={report.videos_considered}, processed={report.videos_processed}, "
            f"skipped={report.videos_skipped}, failed={report.videos_failed}"
        ),
        f"Queen load (1m): {report.queen_load_1m}",
        f"Queen mem available (GB): {report.queen_mem_available_gb}",
    ]
    if report.skipped_due_to_guard:
        lines.append(f"Guard stop reason: {report.guard_reason}")

    for item in report.items:
        lines.append("")
        lines.append(f"[{item.status}] {item.worker_name} ({item.worker_host})")
        lines.append(f"- remote_video_path: {item.remote_video_path or 'n/a'}")
        lines.append(f"- local_video_path: {item.local_video_path or 'n/a'}")
        lines.append(f"- latest_video_path: {item.latest_video_path or 'n/a'}")
        lines.append(f"- latest_video_pulled_at: {item.latest_video_pulled_at or 'n/a'}")
        lines.append(f"- raw_csv_path: {item.raw_csv_path or 'n/a'}")
        lines.append(f"- tracked_video_path: {item.tracked_video_path or 'n/a'}")
        lines.append(f"- latest_tracked_video_path: {item.latest_tracked_video_path or 'n/a'}")
        lines.append(f"- latest_tracked_at: {item.latest_tracked_at or 'n/a'}")
        lines.append(f"- note: {item.note}")
    return "\n".join(lines)


def format_queen_latest_status_report(report: QueenLatestStatusReport) -> str:
    lines = [
        "Queen Latest Media Status",
        "-------------------------",
        f"Generated: {report.generated_at}",
        f"Output root: {report.output_root}",
        (
            f"Workers: {report.workers_considered} "
            f"(synced={report.synced_count}, stale={report.stale_count}, "
            f"missing_video={report.missing_latest_video_count}, "
            f"missing_tracked={report.missing_latest_tracked_count}, offline={report.offline_count})"
        ),
    ]
    for item in report.items:
        lines.append("")
        lines.append(f"[{item.status}] {item.worker_name} ({item.worker_host})")
        lines.append(f"- online: {item.online}")
        lines.append(f"- latest_video_pulled_at: {item.latest_video_pulled_at or 'n/a'}")
        lines.append(f"- latest_tracked_at: {item.latest_tracked_at or 'n/a'}")
        lines.append(f"- track_lag_minutes: {item.track_lag_minutes if item.track_lag_minutes is not None else 'n/a'}")
        lines.append(f"- latest_video_path: {item.latest_video_path or 'n/a'}")
        lines.append(f"- latest_tracked_video_path: {item.latest_tracked_video_path or 'n/a'}")
        lines.append(f"- note: {item.note}")
    return "\n".join(lines)


def format_fleet_discovery_report(report: FleetDiscoveryReport) -> str:
    lines = [
        "Fleet Discovery",
        "---------------",
        f"Generated: {report.generated_at}",
        (
            f"Configured workers: {report.configured_workers} "
            f"(online={report.configured_workers_online}, offline={report.configured_workers_offline})"
        ),
        f"Discovered hosts: {report.discovered_hosts}",
    ]
    for item in report.items:
        lines.append("")
        lines.append(
            f"[{'UP' if item.reachable else 'DOWN'}] {item.host}"
            + (f" ({item.worker_name})" if item.worker_name else "")
        )
        lines.append(f"- from_config: {item.from_config}")
        lines.append(f"- ping_ok: {item.ping_ok}")
        lines.append(f"- ssh_port_open: {item.ssh_port_open}")
        lines.append(f"- arp_mac: {item.arp_mac or 'n/a'}")
        lines.append(f"- arp_state: {item.arp_state or 'n/a'}")
    return "\n".join(lines)


def write_fleet_status_json(report: FleetStatusReport, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def write_queen_track_json(report: QueenTrackReport, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def write_queen_latest_status_json(report: QueenLatestStatusReport, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def write_fleet_discovery_json(report: FleetDiscoveryReport, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
