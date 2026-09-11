from __future__ import annotations

import base64
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Optional
import uuid


SENSORS = {"rgb", "thermal", "realsense"}
READY_MARKER = "BBX_NODE_READY "
RESULT_MARKER = "BBX_NODE_RESULT "
MANIFEST_NAME = "distributed_capture_manifest.json"
PLAN_NAME = "distributed_capture_plan.json"
LOCK_NAME = ".bumblebox_capture.lock"
CHECKSUM_LIMIT_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class CaptureNode:
    name: str
    host: str
    local: bool
    enabled: bool
    sensors: tuple[str, ...]
    user: str
    port: int
    repo_path: Optional[str]
    config_path: Optional[str]
    data_root: Optional[str]


@dataclass(frozen=True)
class ClockEstimate:
    offset_ms: Optional[float]
    round_trip_ms: Optional[float]
    samples: int
    synchronized: Optional[bool]
    detail: str


@dataclass
class NodeCheck:
    name: str
    host: str
    local: bool
    sensors: list[str]
    reachable: bool
    storage_writable: bool
    storage_free_gb: Optional[float]
    repo_revision: Optional[str]
    revision_matches: Optional[bool]
    clock_offset_ms: Optional[float]
    clock_round_trip_ms: Optional[float]
    clock_synchronized: Optional[bool]
    hardware_ok: Optional[bool]
    warnings: list[str]
    errors: list[str]


@dataclass
class DistributedCheckReport:
    generated_at: str
    controller_revision: Optional[str]
    maximum_clock_offset_ms: float
    nodes: list[NodeCheck]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "controller_revision": self.controller_revision,
            "maximum_clock_offset_ms": self.maximum_clock_offset_ms,
            "nodes": [asdict(node) for node in self.nodes],
            "success": self.success,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _timebase_snapshot() -> dict[str, Any]:
    monotonic_ns = time.perf_counter_ns()
    unix_ns = time.time_ns()
    return {
        "captured_at_utc": datetime.fromtimestamp(
            unix_ns / 1_000_000_000.0, tz=timezone.utc
        ).isoformat(timespec="microseconds"),
        "unix_ns": unix_ns,
        "monotonic_ns": monotonic_ns,
        "unix_minus_monotonic_ns": unix_ns - monotonic_ns,
    }


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_revision(repo_path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _fleet_ssh_defaults(config: dict[str, Any]) -> tuple[str, int, Path, int]:
    fleet = config.get("fleet", {}) if isinstance(config.get("fleet", {}), dict) else {}
    ssh = fleet.get("ssh", {}) if isinstance(fleet.get("ssh", {}), dict) else {}
    return (
        str(ssh.get("user") or os.environ.get("USER") or "pi"),
        int(ssh.get("port", 22)),
        Path(str(ssh.get("identity_file") or "~/.ssh/bbx_fleet_ed25519")).expanduser(),
        int(ssh.get("connect_timeout_seconds", 5)),
    )


def configured_nodes(config: dict[str, Any], *, active_only: bool = True) -> list[CaptureNode]:
    section = config.get("distributed_capture", {})
    default_user, default_port, _identity, _timeout = _fleet_ssh_defaults(config)
    nodes: list[CaptureNode] = []
    for raw in section.get("nodes", []):
        enabled = bool(raw.get("enabled", True))
        if active_only and not enabled:
            continue
        nodes.append(
            CaptureNode(
                name=str(raw.get("name") or "").strip(),
                host=str(raw.get("host") or "").strip(),
                local=bool(raw.get("local", False)),
                enabled=enabled,
                sensors=tuple(str(item).strip().lower() for item in raw.get("sensors", [])),
                user=str(raw.get("user") or default_user),
                port=int(raw.get("port", default_port)),
                repo_path=(str(raw["repo_path"]).strip() if raw.get("repo_path") else None),
                config_path=(str(raw["config_path"]).strip() if raw.get("config_path") else None),
                data_root=(str(raw["data_root"]).strip() if raw.get("data_root") else None),
            )
        )
    return nodes


def active_sensors(config: dict[str, Any]) -> set[str]:
    active = {"rgb"}
    if bool(config.get("thermal", {}).get("enabled", False)):
        active.add("thermal")
    if bool(config.get("realsense", {}).get("enabled", False)):
        active.add("realsense")
    return active


def participating_nodes(config: dict[str, Any]) -> list[CaptureNode]:
    required = active_sensors(config)
    return [node for node in configured_nodes(config) if required.intersection(node.sensors)]


def distributed_capture_enabled(config: dict[str, Any]) -> bool:
    section = config.get("distributed_capture", {})
    return bool(section.get("enabled", False)) and str(section.get("role", "")).lower() == "controller"


def _ssh_base(config: dict[str, Any], node: CaptureNode) -> list[str]:
    _user, _port, identity, timeout = _fleet_ssh_defaults(config)
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(node.port),
    ]
    if identity:
        command.extend(["-i", str(identity)])
    command.append(f"{node.user}@{node.host}")
    return command


def _remote_run(
    config: dict[str, Any],
    node: CaptureNode,
    remote_command: str,
    *,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    _user, _port, _identity, configured_timeout = _fleet_ssh_defaults(config)
    return subprocess.run(
        [*_ssh_base(config, node), remote_command],
        capture_output=True,
        text=True,
        timeout=timeout or max(8, configured_timeout + 5),
    )


def _node_repo_path(node: CaptureNode) -> str:
    if node.repo_path:
        return node.repo_path
    if node.local:
        return str(_repo_root())
    return f"/home/{node.user}/Desktop/BumbleBox"


def _node_config_path(node: CaptureNode, controller_config_path: Path) -> str:
    if node.config_path:
        return node.config_path
    if node.local:
        return str(controller_config_path.resolve())
    return f"{_node_repo_path(node)}/bumblebox_v2/config.yaml"


def _node_data_root(node: CaptureNode, config: dict[str, Any]) -> str:
    if node.data_root:
        return node.data_root
    return str(config.get("system", {}).get("data_root") or "/mnt/bumblebox/data")


def _clock_sync_state_local() -> tuple[Optional[bool], str]:
    try:
        result = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return None, str(exc)
    text = result.stdout.strip().lower()
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()
    if text in {"yes", "true", "1"}:
        return True, "NTP synchronized"
    if text in {"no", "false", "0"}:
        return False, "NTP not synchronized"
    return None, text or "NTP state unavailable"


def estimate_node_clock(
    config: dict[str, Any], node: CaptureNode, *, samples: int
) -> ClockEstimate:
    if node.local:
        synchronized, detail = _clock_sync_state_local()
        return ClockEstimate(0.0, 0.0, 1, synchronized, detail)

    measurements: list[tuple[int, int]] = []
    failures: list[str] = []
    for _ in range(max(1, int(samples))):
        before = time.time_ns()
        try:
            result = _remote_run(config, node, "date +%s%N", timeout=8)
        except Exception as exc:
            failures.append(str(exc))
            continue
        after = time.time_ns()
        if result.returncode != 0:
            failures.append((result.stderr or result.stdout).strip())
            continue
        try:
            remote_ns = int(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            failures.append("remote date did not return nanoseconds")
            continue
        midpoint_ns = before + (after - before) // 2
        measurements.append((after - before, remote_ns - midpoint_ns))

    if not measurements:
        return ClockEstimate(None, None, 0, None, "; ".join(failures) or "clock probe failed")
    best_rtt_ns, best_offset_ns = min(measurements, key=lambda item: item[0])
    try:
        sync_result = _remote_run(
            config,
            node,
            "timedatectl show -p NTPSynchronized --value 2>/dev/null || true",
            timeout=8,
        )
        sync_text = sync_result.stdout.strip().lower()
        synchronized = True if sync_text in {"yes", "true", "1"} else False if sync_text in {"no", "false", "0"} else None
    except Exception:
        synchronized = None
    return ClockEstimate(
        offset_ms=best_offset_ns / 1_000_000.0,
        round_trip_ms=best_rtt_ns / 1_000_000.0,
        samples=len(measurements),
        synchronized=synchronized,
        detail="minimum-RTT SSH midpoint estimate",
    )


def _local_storage_status(path: Path) -> tuple[bool, Optional[float], Optional[str]]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        writable = os.access(path, os.W_OK)
        free_gb = shutil.disk_usage(path).free / (1024.0**3)
        return writable, free_gb, None if writable else f"Data root is not writable: {path}"
    except Exception as exc:
        return False, None, str(exc)


def _probe_hardware_command(node: CaptureNode, config_path: str) -> str:
    commands = []
    for sensor in node.sensors:
        if sensor == "rgb":
            commands.append(f"./bbx camera-check --config {shlex.quote(config_path)}")
        elif sensor == "thermal":
            commands.append(f"./bbx thermal-check --config {shlex.quote(config_path)}")
        elif sensor == "realsense":
            commands.append(f"./bbx realsense-check --config {shlex.quote(config_path)}")
    return " && ".join(commands) if commands else "true"


def _probe_local_hardware(node: CaptureNode, config_path: str) -> tuple[bool, str]:
    command_by_sensor = {
        "rgb": "camera-check",
        "thermal": "thermal-check",
        "realsense": "realsense-check",
    }
    failures: list[str] = []
    for sensor in node.sensors:
        command_name = command_by_sensor.get(sensor)
        if command_name is None:
            continue
        result = subprocess.run(
            [
                str(_repo_root() / "bbx"),
                command_name,
                "--config",
                config_path,
            ],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "probe failed"
            failures.append(f"{sensor}: {detail}")
    return not failures, "\n".join(failures)


def check_distributed_capture(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    probe_hardware: bool = False,
) -> DistributedCheckReport:
    config_path = Path(config_path).expanduser()
    section = config.get("distributed_capture", {})
    clock_samples = int(section.get("clock_samples", 5))
    max_clock_ms = float(section.get("maximum_clock_offset_ms", 5.0))
    require_revision = bool(section.get("require_same_revision", True))
    controller_revision = _git_revision(_repo_root())
    checks: list[NodeCheck] = []

    participants = participating_nodes(config)
    controller_name = str(section.get("controller_node") or "")
    nodes_to_check = list(participants)
    controller = next(
        (node for node in configured_nodes(config) if node.name == controller_name), None
    )
    if controller is not None and all(node.name != controller.name for node in nodes_to_check):
        nodes_to_check.insert(0, controller)
    for node in nodes_to_check:
        warnings: list[str] = []
        errors: list[str] = []
        repo_revision: Optional[str] = None
        storage_writable = False
        storage_free_gb: Optional[float] = None
        reachable = False
        hardware_ok: Optional[bool] = None
        node_repo = _node_repo_path(node)
        node_config = _node_config_path(node, config_path)
        data_root = _node_data_root(node, config)

        if node.local:
            reachable = True
            repo_revision = _git_revision(Path(node_repo).expanduser())
            storage_writable, storage_free_gb, storage_error = _local_storage_status(
                Path(data_root).expanduser()
            )
            if storage_error:
                errors.append(storage_error)
            if not Path(node_config).expanduser().is_file():
                errors.append(f"Config file not found: {node_config}")
            if probe_hardware:
                hardware_ok, hardware_error = _probe_local_hardware(node, node_config)
                if not hardware_ok:
                    errors.append(hardware_error or "Hardware probe failed")
        else:
            command = (
                f"cd {shlex.quote(node_repo)} && "
                f"test -x ./bbx && test -r {shlex.quote(node_config)} && "
                f"mkdir -p {shlex.quote(data_root)} && test -w {shlex.quote(data_root)} && "
                f"printf 'REV=' && (git rev-parse HEAD 2>/dev/null || true) && "
                f"df -Pk {shlex.quote(data_root)} | tail -n1"
            )
            try:
                result = _remote_run(config, node, command, timeout=12)
                reachable = result.returncode == 0
                if reachable:
                    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                    revision_line = next((line for line in lines if line.startswith("REV=")), "")
                    repo_revision = revision_line.removeprefix("REV=") or None
                    df_line = lines[-1].split() if lines else []
                    if len(df_line) >= 4:
                        storage_free_gb = float(df_line[3]) / (1024.0 * 1024.0)
                    storage_writable = True
                else:
                    errors.append((result.stderr or result.stdout).strip() or "SSH preflight failed")
            except Exception as exc:
                errors.append(f"SSH preflight failed: {exc}")
            if reachable and probe_hardware:
                result = _remote_run(
                    config,
                    node,
                    f"cd {shlex.quote(node_repo)} && {_probe_hardware_command(node, node_config)}",
                    timeout=120,
                )
                hardware_ok = result.returncode == 0
                if not hardware_ok:
                    errors.append((result.stderr or result.stdout).strip() or "Hardware probe failed")

        clock = estimate_node_clock(config, node, samples=clock_samples) if reachable else ClockEstimate(None, None, 0, None, "unreachable")
        if clock.offset_ms is None:
            errors.append("Could not estimate clock offset")
        elif abs(clock.offset_ms) > max_clock_ms:
            errors.append(
                f"Clock offset {clock.offset_ms:+.3f} ms exceeds configured {max_clock_ms:.3f} ms"
            )
        if clock.synchronized is False:
            errors.append("The node reports that network time synchronization is not locked")
        elif clock.synchronized is None:
            warnings.append("Network time synchronization state could not be verified")
        revision_matches = (
            repo_revision == controller_revision
            if repo_revision is not None and controller_revision is not None
            else None
        )
        if require_revision and revision_matches is False:
            errors.append(
                f"Repository revision differs from controller ({repo_revision} != {controller_revision})"
            )
        if require_revision and revision_matches is None:
            errors.append("Could not verify repository revision")
        if storage_free_gb is not None and storage_free_gb < 2.0:
            warnings.append(f"Only {storage_free_gb:.2f} GB is free at the data root")

        checks.append(
            NodeCheck(
                name=node.name,
                host=node.host,
                local=node.local,
                sensors=list(node.sensors),
                reachable=reachable,
                storage_writable=storage_writable,
                storage_free_gb=storage_free_gb,
                repo_revision=repo_revision,
                revision_matches=revision_matches,
                clock_offset_ms=clock.offset_ms,
                clock_round_trip_ms=clock.round_trip_ms,
                clock_synchronized=clock.synchronized,
                hardware_ok=hardware_ok,
                warnings=warnings,
                errors=errors,
            )
        )

    success = bool(checks) and all(not check.errors for check in checks)
    return DistributedCheckReport(
        generated_at=_utc_now(),
        controller_revision=controller_revision,
        maximum_clock_offset_ms=max_clock_ms,
        nodes=checks,
        success=success,
    )


def format_distributed_check(report: DistributedCheckReport) -> str:
    lines = [
        "Distributed Capture Check",
        "-------------------------",
        f"Controller revision: {report.controller_revision or 'unknown'}",
        f"Clock offset limit: {report.maximum_clock_offset_ms:.3f} ms",
    ]
    for node in report.nodes:
        state = "PASS" if not node.errors else "FAIL"
        offset = f"{node.clock_offset_ms:+.3f} ms" if node.clock_offset_ms is not None else "unknown"
        rtt = f"{node.clock_round_trip_ms:.3f} ms" if node.clock_round_trip_ms is not None else "unknown"
        free = f"{node.storage_free_gb:.2f} GB" if node.storage_free_gb is not None else "unknown"
        lines.extend(
            [
                "",
                f"[{state}] {node.name} ({'local' if node.local else node.host})",
                f"  sensors: {', '.join(node.sensors) or 'none'}",
                f"  clock offset / RTT: {offset} / {rtt}",
                f"  network time synchronized: {node.clock_synchronized if node.clock_synchronized is not None else 'unknown'}",
                f"  storage writable / free: {node.storage_writable} / {free}",
                f"  repository revision: {node.repo_revision or 'unknown'}",
                f"  hardware probe: {node.hardware_ok if node.hardware_ok is not None else 'not requested'}",
            ]
        )
        lines.extend(f"  warning: {item}" for item in node.warnings)
        lines.extend(f"  error: {item}" for item in node.errors)
    lines.extend(["", f"Ready for distributed capture: {report.success}"])
    return "\n".join(lines)


def _effective_capture_sections(config: dict[str, Any]) -> dict[str, Any]:
    sections = {}
    for key in ("camera", "capture", "pipeline", "runtime", "tracking", "cleaning", "metrics", "calibration"):
        if isinstance(config.get(key), dict):
            sections[key] = deepcopy(config[key])
    thermal = deepcopy(config.get("thermal", {}))
    thermal.pop("device_path", None)
    sections["thermal"] = thermal
    realsense = deepcopy(config.get("realsense", {}))
    realsense.pop("device_serial", None)
    sections["realsense"] = realsense
    return sections


def create_capture_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    mode: str,
    start_unix_ns: Optional[int] = None,
) -> dict[str, Any]:
    if mode not in {"record_only", "record_and_track"}:
        raise ValueError(
            "Distributed capture supports record_only and record_and_track modes"
        )
    nodes = participating_nodes(config)
    if not nodes:
        raise ValueError("Distributed capture has no participating nodes")
    lead_seconds = float(config.get("distributed_capture", {}).get("start_lead_seconds", 8.0))
    if start_unix_ns is None:
        start_unix_ns = time.time_ns() + int(lead_seconds * 1_000_000_000)
    start_dt = datetime.fromtimestamp(start_unix_ns / 1_000_000_000.0).astimezone()
    controller = str(config.get("distributed_capture", {}).get("controller_node") or nodes[0].name)
    session_name = f"{socket.gethostname()}_{start_dt.strftime('%Y-%m-%d_%H_%M_%S')}_multi"
    return {
        "schema_version": 1,
        "plan_id": str(uuid.uuid4()),
        "created_at_utc": _utc_now(),
        "controller_node": controller,
        "controller_host": socket.gethostname(),
        "controller_revision": _git_revision(_repo_root()),
        "controller_config_path": str(Path(config_path).expanduser().resolve()),
        "controller_config_sha256": _config_hash(config),
        "session_name": session_name,
        "session_date": start_dt.strftime("%Y-%m-%d"),
        "mode": mode,
        "capture_start_unix_ns": int(start_unix_ns),
        "capture_start_utc": datetime.fromtimestamp(
            start_unix_ns / 1_000_000_000.0, tz=timezone.utc
        ).isoformat(timespec="microseconds"),
        "recording_seconds": float(config.get("capture", {}).get("recording_seconds", 0)),
        "failure_policy": str(config.get("distributed_capture", {}).get("failure_policy", "all_or_nothing")),
        "active_sensors": sorted(active_sensors(config)),
        "nodes": [asdict(node) for node in nodes],
        "effective_config_sections": _effective_capture_sections(config),
    }


def encode_capture_plan(plan: dict[str, Any]) -> str:
    raw = json.dumps(plan, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_capture_plan(value: str) -> dict[str, Any]:
    try:
        return json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid distributed capture plan: {exc}") from exc


def _merge_plan_sections(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    sections = plan.get("effective_config_sections", {})
    for key, value in sections.items():
        if not isinstance(value, dict):
            continue
        target = merged.setdefault(key, {})
        preserve = {}
        if key == "thermal":
            preserve["device_path"] = target.get("device_path")
        elif key == "realsense":
            preserve["device_serial"] = target.get("device_serial")
        target.update(deepcopy(value))
        target.update({name: item for name, item in preserve.items() if item is not None})
    return merged


def _node_from_plan(plan: dict[str, Any], node_name: str) -> dict[str, Any]:
    for node in plan.get("nodes", []):
        if str(node.get("name")) == node_name:
            return node
    raise ValueError(f"Capture plan does not contain node: {node_name}")


def _planned_start_monotonic(plan: dict[str, Any]) -> float:
    remaining = (int(plan["capture_start_unix_ns"]) - time.time_ns()) / 1_000_000_000.0
    return time.perf_counter() + max(0.0, remaining)


@contextmanager
def _capture_lock(data_root: Path, *, lock_name: str = LOCK_NAME) -> Iterator[None]:
    data_root.mkdir(parents=True, exist_ok=True)
    lock_path = data_root / lock_name
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another BumbleBox capture is already active on this node ({lock_path})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _artifact_inventory(session_dir: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in session_dir.rglob("*") if item.is_file()):
        size = path.stat().st_size
        digest = None
        if size <= CHECKSUM_LIMIT_BYTES:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        inventory.append(
            {
                "relative_path": str(path.relative_to(session_dir)),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return inventory


def _jsonable_artifacts(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable_artifacts(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_artifacts(item) for item in value]
    return value


def _sensor_only_capture(
    config: dict[str, Any],
    *,
    sensors: set[str],
    session_dir: Path,
    session_name: str,
    start_unix_ns: int,
    ready_callback: Callable[[], None],
    node_name: str,
) -> dict[str, Any]:
    from .realsense_camera import RealSenseRecordingSession, capture_simulated_realsense_recording
    from .run_engine import (
        FrameTimestampRecord,
        _ThermalCaptureSession,
        _flush_thermal_capture_queue,
        _write_thermal_recording_outputs,
    )
    from .simulated_capture import simulated_frame_times, simulated_thermal_frame

    duration = float(config.get("capture", {}).get("recording_seconds", 0))
    thermal_result = None
    realsense_result = None
    errors: list[str] = []
    start_monotonic = _planned_start_monotonic({"capture_start_unix_ns": start_unix_ns})

    if bool(config.get("runtime", {}).get("use_mock_camera", False)):
        ready_callback()
        while time.perf_counter() < start_monotonic:
            time.sleep(min(0.001, max(0.0, start_monotonic - time.perf_counter())))
        if "thermal" in sensors:
            fps = float(config.get("camera", {}).get("fps_target", 7.0))
            frame_times = simulated_frame_times(duration, fps)
            offset = time.time() - time.perf_counter()
            records = [
                FrameTimestampRecord(t, start_monotonic + t, start_monotonic + t + offset)
                for t in frame_times
            ]
            frames = [
                simulated_thermal_frame(
                    int(config.get("thermal", {}).get("width", 160)),
                    int(config.get("thermal", {}).get("height", 120)),
                    t,
                    duration,
                )
                for t in frame_times
            ]
            thermal_result = _write_thermal_recording_outputs(
                session_dir=session_dir,
                session_name=session_name,
                device_path="mock://thermal",
                frames=frames,
                timestamps=frame_times,
                actual_fps=fps,
                raw16_layout="uint16_mono16",
                timestamp_records=records,
                node_name=node_name,
            )
        if "realsense" in sensors:
            realsense_result = capture_simulated_realsense_recording(
                config,
                session_dir=session_dir,
                session_name=session_name,
                duration=duration,
                start_monotonic=start_monotonic,
                node_name=node_name,
            )
    else:
        thermal_capture: dict[str, Any] = {}
        realsense_capture: dict[str, Any] = {}
        with ExitStack() as stack:
            thermal_session = (
                stack.enter_context(_ThermalCaptureSession(config)) if "thermal" in sensors else None
            )
            realsense_session = (
                stack.enter_context(
                    RealSenseRecordingSession(
                        config,
                        session_dir=session_dir,
                        session_name=session_name,
                        node_name=node_name,
                    )
                )
                if "realsense" in sensors
                else None
            )
            if thermal_session is not None:
                _flush_thermal_capture_queue(thermal_session.capture, 4)
            ready_callback()

            def thermal_worker() -> None:
                try:
                    assert thermal_session is not None
                    frames, timestamps, fps, layout, records = thermal_session.capture_for(
                        fps=float(config.get("camera", {}).get("fps_target", 7.0)),
                        duration=duration,
                        start_monotonic=start_monotonic,
                        progress_label="Thermal capture",
                    )
                    thermal_capture.update(
                        frames=frames,
                        timestamps=timestamps,
                        actual_fps=fps,
                        raw16_layout=layout,
                        records=records,
                        device_path=thermal_session.device_path,
                    )
                except Exception as exc:
                    errors.append(f"Thermal capture failed: {exc}")

            def realsense_worker() -> None:
                try:
                    assert realsense_session is not None
                    realsense_capture["result"] = realsense_session.capture_for(
                        duration=duration,
                        start_monotonic=start_monotonic,
                    )
                except Exception as exc:
                    errors.append(f"RealSense capture failed: {exc}")

            threads = []
            if thermal_session is not None:
                threads.append(threading.Thread(target=thermal_worker, daemon=True))
            if realsense_session is not None:
                threads.append(threading.Thread(target=realsense_worker, daemon=True))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        if errors:
            raise RuntimeError(" | ".join(errors))
        if thermal_capture:
            thermal_result = _write_thermal_recording_outputs(
                session_dir=session_dir,
                session_name=session_name,
                device_path=str(thermal_capture["device_path"]),
                frames=thermal_capture["frames"],
                timestamps=thermal_capture["timestamps"],
                actual_fps=float(thermal_capture["actual_fps"]),
                raw16_layout=str(thermal_capture["raw16_layout"]),
                timestamp_records=thermal_capture["records"],
                node_name=node_name,
            )
        realsense_result = realsense_capture.get("result")

    return {
        "thermal": _jsonable_artifacts(thermal_result),
        "realsense": _jsonable_artifacts(realsense_result),
    }


def execute_capture_node(
    config: dict[str, Any],
    *,
    plan: dict[str, Any],
    node_name: str,
    emit_protocol_markers: bool = True,
) -> dict[str, Any]:
    from .run_engine import run_once

    node = _node_from_plan(plan, node_name)
    sensors = set(str(item).lower() for item in node.get("sensors", [])) & set(plan["active_sensors"])
    if not sensors:
        raise ValueError(f"Node {node_name} has no active sensor assignments in this plan")
    config = _merge_plan_sections(config, plan)
    config.setdefault("thermal", {})["enabled"] = "thermal" in sensors
    config.setdefault("realsense", {})["enabled"] = "realsense" in sensors
    data_root = Path(str(node.get("data_root") or config.get("system", {}).get("data_root"))).expanduser()
    session_dir = data_root / str(plan["session_date"]) / str(plan["session_name"])
    session_dir.mkdir(parents=True, exist_ok=True)
    result_path = session_dir / f"{plan['session_name']}_{node_name}_capture_result.json"
    ready_at: Optional[str] = None
    started_at = _utc_now()
    timebase_before = _timebase_snapshot()

    def ready() -> None:
        nonlocal ready_at
        remaining_ms = (int(plan["capture_start_unix_ns"]) - time.time_ns()) / 1_000_000.0
        if remaining_ms < 50.0:
            raise RuntimeError(
                f"Node became ready too late ({remaining_ms:.1f} ms before the planned start)"
            )
        ready_at = _utc_now()
        payload = {"plan_id": plan["plan_id"], "node": node_name, "ready_at_utc": ready_at}
        if emit_protocol_markers:
            print(READY_MARKER + json.dumps(payload, separators=(",", ":")), flush=True)

    payload: dict[str, Any]
    try:
        with _capture_lock(data_root):
            if "rgb" in sensors:
                summary = run_once(
                    config,
                    mode_override=str(plan["mode"]),
                    session_name_override=str(plan["session_name"]),
                    session_dir_override=session_dir,
                    capture_start_unix_ns=int(plan["capture_start_unix_ns"]),
                    capture_ready_callback=ready,
                    sensor_names_override=sorted(sensors),
                    capture_node_name=node_name,
                )
                run_summary_path = session_dir / f"{plan['session_name']}_run_summary.json"
                payload = {
                    "success": bool(summary.success),
                    "errors": list(summary.errors),
                    "warnings": list(summary.warnings),
                    "run_summary": asdict(summary),
                    "run_summary_path": str(run_summary_path),
                    "artifacts": {},
                }
            else:
                artifacts = _sensor_only_capture(
                    config,
                    sensors=sensors,
                    session_dir=session_dir,
                    session_name=str(plan["session_name"]),
                    start_unix_ns=int(plan["capture_start_unix_ns"]),
                    ready_callback=ready,
                    node_name=node_name,
                )
                payload = {
                    "success": True,
                    "errors": [],
                    "warnings": [],
                    "run_summary": None,
                    "run_summary_path": None,
                    "artifacts": artifacts,
                }
    except Exception as exc:
        payload = {
            "success": False,
            "errors": [str(exc)],
            "warnings": [],
            "run_summary": None,
            "run_summary_path": None,
            "artifacts": {},
        }

    payload.update(
        {
            "schema_version": 1,
            "plan_id": str(plan["plan_id"]),
            "node": node_name,
            "host": socket.gethostname(),
            "sensors": sorted(sensors),
            "session_name": str(plan["session_name"]),
            "session_dir": str(session_dir),
            "planned_start_utc": str(plan["capture_start_utc"]),
            "ready_at_utc": ready_at,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "config_sha256": _config_hash(config),
            "repo_revision": _git_revision(_repo_root()),
            "timebase_before": timebase_before,
            "timebase_after": _timebase_snapshot(),
        }
    )
    payload["artifact_inventory"] = _artifact_inventory(session_dir)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    payload["node_result_path"] = str(result_path)
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    if emit_protocol_markers:
        print(RESULT_MARKER + encoded, flush=True)
    return payload


def _capture_command(
    config: dict[str, Any],
    node: CaptureNode,
    *,
    config_path: Path,
    encoded_plan: str,
) -> tuple[list[str], Optional[Path]]:
    node_config = _node_config_path(node, config_path)
    arguments = [
        "capture-node",
        "run",
        "--config",
        node_config,
        "--node",
        node.name,
        "--plan-base64",
        encoded_plan,
    ]
    if node.local:
        return [sys.executable, str(_repo_root() / "bbx.py"), *arguments], _repo_root()
    quoted = " ".join(shlex.quote(item) for item in ["./bbx", *arguments])
    remote = f"cd {shlex.quote(_node_repo_path(node))} && {quoted}"
    return [*_ssh_base(config, node), remote], None


def _decode_result_line(line: str) -> Optional[dict[str, Any]]:
    if not line.startswith(RESULT_MARKER):
        return None
    try:
        raw = base64.urlsafe_b64decode(line[len(RESULT_MARKER) :].strip().encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _terminate_processes(processes: dict[str, subprocess.Popen[str]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    end = time.monotonic() + 3.0
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass


def _transfer_node_session(
    config: dict[str, Any],
    node: CaptureNode,
    *,
    remote_session_dir: str,
    canonical_session_dir: Path,
    plan_id: str,
) -> dict[str, Any]:
    if node.local:
        return {"status": "local", "path": remote_session_dir}
    if shutil.which("rsync") is None:
        return {"status": "failed", "error": "rsync is not installed on the controller"}
    final_dir = canonical_session_dir / "nodes" / node.name
    marker = final_dir / ".transfer_complete.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text()).get("plan_id") == plan_id:
                return {"status": "already_complete", "path": str(final_dir)}
        except Exception:
            pass
    partial_dir = canonical_session_dir / "nodes" / f".{node.name}.partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    source = f"{node.user}@{node.host}:{remote_session_dir.rstrip('/')}/"
    ssh_parts = _ssh_base(config, node)[:-1]
    ssh_command = " ".join(shlex.quote(item) for item in ssh_parts)
    result = subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "--checksum",
            "-e",
            ssh_command,
            source,
            f"{partial_dir}/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "status": "failed",
            "partial_path": str(partial_dir),
            "error": (result.stderr or result.stdout).strip() or "rsync failed",
        }
    if final_dir.exists():
        return {
            "status": "failed",
            "partial_path": str(partial_dir),
            "error": f"Transfer destination already exists without a matching marker: {final_dir}",
        }
    partial_dir.rename(final_dir)
    marker = final_dir / ".transfer_complete.json"
    marker.write_text(json.dumps({"plan_id": plan_id, "completed_at_utc": _utc_now()}, indent=2) + "\n")
    return {"status": "complete", "path": str(final_dir)}


def _aggregate_summary(
    config: dict[str, Any],
    plan: dict[str, Any],
    node_results: dict[str, dict[str, Any]],
    manifest_path: Path,
    transfers: dict[str, dict[str, Any]],
    coordinator_errors: Optional[list[str]] = None,
) -> Path:
    canonical_dir = manifest_path.parent
    rgb_result = next(
        (item for item in node_results.values() if "rgb" in item.get("sensors", [])),
        None,
    )
    source_summary = deepcopy(rgb_result.get("run_summary")) if rgb_result and rgb_result.get("run_summary") else {}

    def local_artifact_path(name: str, result: dict[str, Any], raw_path: Any) -> Any:
        if not raw_path:
            return raw_path
        transfer = transfers.get(name, {})
        transferred_root = transfer.get("path") if isinstance(transfer, dict) else None
        if not transferred_root:
            return raw_path
        try:
            relative = Path(str(raw_path)).relative_to(Path(str(result["session_dir"])))
        except Exception:
            return raw_path
        return str(Path(str(transferred_root)) / relative)

    if rgb_result is not None:
        rgb_name = str(rgb_result.get("node"))
        for key, value in list(source_summary.items()):
            if key.endswith("_path") and value:
                source_summary[key] = local_artifact_path(rgb_name, rgb_result, value)
    source_summary.update(
        {
            "session_name": plan["session_name"],
            "session_dir": str(canonical_dir),
            "started_at": source_summary.get("started_at") or plan["capture_start_utc"],
            "finished_at": source_summary.get("finished_at") or _utc_now(),
            "mode": plan["mode"],
            "success": all(bool(item.get("success")) for item in node_results.values())
            and not coordinator_errors,
            "distributed_capture": {
                "enabled": True,
                "plan_id": plan["plan_id"],
                "manifest_path": str(manifest_path),
                "planned_start_utc": plan["capture_start_utc"],
                "nodes": {
                    name: {
                        "success": bool(result.get("success")),
                        "host": result.get("host"),
                        "sensors": result.get("sensors", []),
                        "session_dir": result.get("session_dir"),
                        "transfer": transfers.get(name),
                    }
                    for name, result in node_results.items()
                },
            },
            "distributed_manifest_path": str(manifest_path),
        }
    )
    source_summary.setdefault("warnings", [])
    source_summary.setdefault("errors", [])
    source_summary["errors"].extend(
        f"distributed controller: {item}" for item in (coordinator_errors or [])
    )
    for name, result in node_results.items():
        if not (rgb_result is result and result.get("run_summary")):
            source_summary["warnings"].extend(f"{name}: {item}" for item in result.get("warnings", []))
            source_summary["errors"].extend(f"{name}: {item}" for item in result.get("errors", []))
        artifacts = result.get("artifacts", {})
        thermal = artifacts.get("thermal") if isinstance(artifacts, dict) else None
        depth = artifacts.get("realsense") if isinstance(artifacts, dict) else None
        if isinstance(thermal, dict):
            source_summary["thermal_enabled"] = True
            source_summary["thermal_frames_captured"] = int(thermal.get("frames_captured", 0))
            source_summary["thermal_actual_fps"] = thermal.get("actual_fps")
            source_summary["thermal_device_path"] = thermal.get("device_path")
            for source_key, summary_key in (
                ("timestamp_path", "thermal_timestamp_path"),
                ("raw_npy_path", "thermal_raw_npy_path"),
                ("preview_video_path", "thermal_preview_video_path"),
                ("preview_png_path", "thermal_preview_png_path"),
                ("metadata_json_path", "thermal_metadata_json_path"),
            ):
                source_summary[summary_key] = local_artifact_path(
                    name, result, thermal.get(source_key)
                )
        if isinstance(depth, dict):
            source_summary["realsense_enabled"] = True
            source_summary["realsense_frames_captured"] = int(depth.get("frames_captured", 0))
            source_summary["realsense_actual_fps"] = depth.get("actual_fps")
            source_summary["realsense_depth_scale_meters"] = depth.get("depth_scale_meters")
            source_summary["realsense_device_serial"] = depth.get("selected_serial")
            for source_key, summary_key in (
                ("timestamp_path", "realsense_timestamp_path"),
                ("raw_depth_npy_path", "realsense_raw_depth_npy_path"),
                ("depth_preview_video_path", "realsense_depth_preview_video_path"),
                ("depth_preview_png_path", "realsense_depth_preview_png_path"),
                ("color_video_path", "realsense_color_video_path"),
                ("color_preview_png_path", "realsense_color_preview_png_path"),
                ("metadata_json_path", "realsense_metadata_json_path"),
            ):
                source_summary[summary_key] = local_artifact_path(
                    name, result, depth.get(source_key)
                )
    summary_path = canonical_dir / f"{plan['session_name']}_run_summary.json"
    summary_path.write_text(json.dumps(source_summary, indent=2) + "\n")
    return summary_path


def run_distributed_capture(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    mode: str,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser()
    check = check_distributed_capture(config, config_path=config_path, probe_hardware=False)
    if not check.success:
        raise RuntimeError("Distributed preflight failed:\n" + format_distributed_check(check))

    plan = create_capture_plan(config, config_path=config_path, mode=mode)
    nodes = participating_nodes(config)
    controller_node = next(
        node for node in configured_nodes(config) if node.name == plan["controller_node"]
    )
    canonical_root = Path(_node_data_root(controller_node, config)).expanduser()
    canonical_dir = canonical_root / str(plan["session_date"]) / str(plan["session_name"])
    canonical_dir.mkdir(parents=True, exist_ok=True)
    (canonical_dir / PLAN_NAME).write_text(json.dumps(plan, indent=2) + "\n")
    encoded_plan = encode_capture_plan(plan)
    processes: dict[str, subprocess.Popen[str]] = {}
    lines: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
    node_results: dict[str, dict[str, Any]] = {}
    ready_nodes: set[str] = set()
    output_tail: dict[str, list[str]] = {node.name: [] for node in nodes}
    coordinator_errors: list[str] = []

    def read_output(name: str, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            lines.put((name, raw_line.rstrip("\n")))
        lines.put((name, None))

    with _capture_lock(canonical_root, lock_name=".bumblebox_distributed_controller.lock"):
        try:
            print(f"[distributed] Plan {plan['plan_id']} starts at {plan['capture_start_utc']}.", flush=True)
            for node in nodes:
                command, cwd = _capture_command(
                    config,
                    node,
                    config_path=config_path,
                    encoded_plan=encoded_plan,
                )
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                processes[node.name] = process
                threading.Thread(target=read_output, args=(node.name, process), daemon=True).start()

            ready_timeout = float(config.get("distributed_capture", {}).get("ready_timeout_seconds", 30.0))
            ready_deadline = min(
                time.monotonic() + ready_timeout,
                time.monotonic() + max(0.0, (int(plan["capture_start_unix_ns"]) - time.time_ns()) / 1e9 - 0.05),
            )
            while len(ready_nodes) < len(nodes) and time.monotonic() < ready_deadline:
                try:
                    name, line = lines.get(timeout=0.1)
                except queue.Empty:
                    if processes and all(
                        proc.poll() is not None for proc in processes.values()
                    ):
                        break
                    continue
                if line is None:
                    continue
                output_tail[name] = (output_tail[name] + [line])[-25:]
                if line.startswith(READY_MARKER):
                    ready_nodes.add(name)
                    print(f"[distributed] {name} is armed ({len(ready_nodes)}/{len(nodes)}).", flush=True)
                else:
                    result = _decode_result_line(line)
                    if result is not None:
                        node_results[name] = result
                    else:
                        print(f"[{name}] {line}", flush=True)

            missing = sorted(node.name for node in nodes if node.name not in ready_nodes)
            if missing and str(plan["failure_policy"]) == "all_or_nothing":
                _terminate_processes(processes)
                raise RuntimeError("Nodes did not arm before the shared deadline: " + ", ".join(missing))
            for name in missing:
                process = processes[name]
                if process.poll() is None:
                    process.terminate()
            if missing:
                print("[distributed] Continuing without unready nodes: " + ", ".join(missing), flush=True)

            while any(process.poll() is None for process in processes.values()):
                try:
                    name, line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    continue
                output_tail[name] = (output_tail[name] + [line])[-25:]
                result = _decode_result_line(line)
                if result is not None:
                    node_results[name] = result
                elif not line.startswith(READY_MARKER):
                    print(f"[{name}] {line}", flush=True)
            while True:
                try:
                    name, line = lines.get_nowait()
                except queue.Empty:
                    break
                if line:
                    result = _decode_result_line(line)
                    if result is not None:
                        node_results[name] = result
        except Exception as exc:
            _terminate_processes(processes)
            coordinator_errors.append(str(exc))
        finally:
            for process in processes.values():
                if process.stdout is not None:
                    process.stdout.close()

    for node in nodes:
        if node.name not in node_results:
            process = processes.get(node.name)
            returncode = process.returncode if process is not None else "not started"
            node_results[node.name] = {
                "success": False,
                "node": node.name,
                "host": node.host,
                "sensors": sorted(set(node.sensors) & set(plan["active_sensors"])),
                "errors": [
                    f"Capture process exited with code {returncode}; "
                    + (" | ".join(output_tail[node.name][-5:]) or "no result marker received")
                ],
                "warnings": [],
                "session_dir": None,
                "artifact_inventory": [],
            }

    transfers: dict[str, dict[str, Any]] = {}
    if bool(config.get("distributed_capture", {}).get("transfer_after_capture", False)):
        for node in nodes:
            result = node_results[node.name]
            if result.get("success") and result.get("session_dir"):
                print(f"[distributed] Collecting {node.name} artifacts.", flush=True)
                try:
                    transfers[node.name] = _transfer_node_session(
                        config,
                        node,
                        remote_session_dir=str(result["session_dir"]),
                        canonical_session_dir=canonical_dir,
                        plan_id=str(plan["plan_id"]),
                    )
                except Exception as exc:
                    transfers[node.name] = {"status": "failed", "error": str(exc)}

    transfer_errors = [
        f"{name}: {item.get('error') or 'artifact transfer failed'}"
        for name, item in transfers.items()
        if str(item.get("status")) == "failed"
    ]
    capture_success = all(
        bool(item.get("success")) for item in node_results.values()
    ) and not coordinator_errors
    collection_success = not transfer_errors

    manifest = {
        "schema_version": 1,
        "plan": plan,
        "preflight": check.to_dict(),
        "completed_at_utc": _utc_now(),
        "success": capture_success and collection_success,
        "capture_success": capture_success,
        "collection_success": collection_success,
        "errors": coordinator_errors,
        "transfer_errors": transfer_errors,
        "ready_nodes": sorted(ready_nodes),
        "node_results": node_results,
        "transfers": transfers,
    }
    manifest_path = canonical_dir / MANIFEST_NAME
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary_path = _aggregate_summary(
        config,
        plan,
        node_results,
        manifest_path,
        transfers,
        coordinator_errors + transfer_errors,
    )
    manifest["summary_path"] = str(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def collect_distributed_capture(
    config: dict[str, Any], *, manifest_path: str | Path
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    plan = manifest["plan"]
    configured = {node.name: node for node in configured_nodes(config, active_only=False)}
    transfers = dict(manifest.get("transfers", {}))
    for name, result in manifest.get("node_results", {}).items():
        node = configured.get(name)
        if node is None or node.local or not result.get("success") or not result.get("session_dir"):
            continue
        try:
            transfers[name] = _transfer_node_session(
                config,
                node,
                remote_session_dir=str(result["session_dir"]),
                canonical_session_dir=path.parent,
                plan_id=str(plan["plan_id"]),
            )
        except Exception as exc:
            transfers[name] = {"status": "failed", "error": str(exc)}
    transfer_errors = [
        f"{name}: {item.get('error') or 'artifact transfer failed'}"
        for name, item in transfers.items()
        if str(item.get("status")) == "failed"
    ]
    manifest["transfers"] = transfers
    manifest["transfer_errors"] = transfer_errors
    manifest["collection_success"] = not transfer_errors
    capture_success = bool(
        manifest.get(
            "capture_success",
            all(
                bool(item.get("success"))
                for item in manifest.get("node_results", {}).values()
            )
            and not manifest.get("errors"),
        )
    )
    manifest["capture_success"] = capture_success
    manifest["success"] = capture_success and not transfer_errors
    manifest["last_collection_attempt_utc"] = _utc_now()
    summary_path = _aggregate_summary(
        config,
        plan,
        manifest.get("node_results", {}),
        path,
        transfers,
        list(manifest.get("errors", [])) + transfer_errors,
    )
    manifest["summary_path"] = str(summary_path)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def setup_distributed_worker(
    config: dict[str, Any],
    *,
    node_name: str,
    install_key: bool = True,
    run_environment_setup: bool = False,
) -> dict[str, Any]:
    from .fleet import ensure_ssh_keypair

    nodes = {node.name: node for node in configured_nodes(config)}
    node = nodes.get(node_name)
    if node is None:
        raise ValueError(f"Unknown enabled distributed capture node: {node_name}")
    if node.local:
        raise ValueError("Worker setup applies only to a remote node")
    _default_user, _default_port, identity, _timeout = _fleet_ssh_defaults(config)
    private_key, public_key, key_created = ensure_ssh_keypair(identity)
    steps: list[dict[str, Any]] = []
    if install_key:
        if shutil.which("ssh-copy-id") is None:
            raise RuntimeError("ssh-copy-id is required to install the controller key")
        key_result = subprocess.run(
            [
                "ssh-copy-id",
                "-i",
                str(public_key),
                "-p",
                str(node.port),
                f"{node.user}@{node.host}",
            ]
        )
        steps.append({"step": "install_ssh_key", "returncode": key_result.returncode})
        if key_result.returncode != 0:
            raise RuntimeError("SSH key installation failed")

    repo = _node_repo_path(node)
    config_path = node.config_path or f"{repo}/bumblebox_v2/config.yaml"
    verify_command = (
        f"cd {shlex.quote(repo)} && test -x ./bbx && "
        f"test -r {shlex.quote(config_path)}"
    )
    verify = _remote_run(config, node, verify_command, timeout=15)
    steps.append({"step": "verify_repository", "returncode": verify.returncode})
    if verify.returncode != 0:
        raise RuntimeError(
            (verify.stderr or verify.stdout).strip()
            or f"BumbleBox repository/config was not found on {node.host}"
        )

    if run_environment_setup:
        setup_args = ["bash", "scripts/setup_venv.sh", "--skip-label-env", "--skip-gui-shortcut"]
        if "realsense" in node.sensors:
            setup_args.append("--install-realsense")
        setup_command = " ".join(shlex.quote(item) for item in setup_args)
        setup = subprocess.run([*_ssh_base(config, node), f"cd {shlex.quote(repo)} && {setup_command}"])
        steps.append({"step": "install_runtime", "returncode": setup.returncode})
        if setup.returncode != 0:
            raise RuntimeError("Remote BumbleBox environment setup failed")

    return {
        "node": node.name,
        "host": node.host,
        "user": node.user,
        "identity_file": str(private_key),
        "public_key_file": str(public_key),
        "key_created": key_created,
        "environment_setup_run": run_environment_setup,
        "steps": steps,
        "success": True,
    }


def format_distributed_result(manifest: dict[str, Any]) -> str:
    plan = manifest["plan"]
    lines = [
        "Distributed Capture Result",
        "--------------------------",
        f"Plan: {plan['plan_id']}",
        f"Session: {plan['session_name']}",
        f"Planned start: {plan['capture_start_utc']}",
        f"Manifest: {manifest.get('manifest_path', 'see session directory')}",
    ]
    for name, result in manifest.get("node_results", {}).items():
        lines.append(
            f"- {name}: {'success' if result.get('success') else 'failed'} | "
            f"sensors={','.join(result.get('sensors', []))} | "
            f"directory={result.get('session_dir') or 'none'}"
        )
        lines.extend(f"  error: {error}" for error in result.get("errors", []))
    lines.extend(f"Controller error: {error}" for error in manifest.get("errors", []))
    lines.extend(f"Transfer error: {error}" for error in manifest.get("transfer_errors", []))
    lines.extend(
        [
            f"Summary: {manifest.get('summary_path', 'see session directory')}",
            f"Success: {bool(manifest.get('success'))}",
        ]
    )
    return "\n".join(lines)
