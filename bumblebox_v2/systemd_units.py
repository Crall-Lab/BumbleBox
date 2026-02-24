from __future__ import annotations

import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import normalize_service_user_value


@dataclass
class SystemdWriteResult:
    output_dir: Path
    written_files: List[Path]
    install_commands: List[str]


@dataclass
class SystemdActionResult:
    action: str
    scope: str
    timer_names: List[str]
    commands: List[List[str]]
    command_outputs: List[str]
    success: bool
    note: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _service_text(description: str, working_dir: Path, exec_start: str, service_user: str | None) -> str:
    lines = [
        "[Unit]",
        f"Description={description}",
        "After=local-fs.target network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=oneshot",
        f"WorkingDirectory={working_dir}",
        "Environment=PYTHONUNBUFFERED=1",
    ]
    if service_user:
        lines.append(f"User={service_user}")
    lines.extend(
        [
            f"ExecStart={exec_start}",
            "Nice=5",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )
    return "\n".join(lines)


def _timer_text(description: str, service_name: str, interval_minutes: int) -> str:
    return "\n".join(
        [
            "[Unit]",
            f"Description={description}",
            "",
            "[Timer]",
            "OnBootSec=2min",
            f"OnUnitActiveSec={int(interval_minutes)}min",
            "AccuracySec=1s",
            "Persistent=true",
            f"Unit={service_name}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def _exec_start_for_args(config_path: Path, args: List[str]) -> str:
    python_path = Path(sys.executable).resolve()
    bbx_path = (_repo_root() / "bbx.py").resolve()
    cmd = [str(python_path), str(bbx_path), *args, "--config", str(config_path.resolve())]
    return " ".join(shlex.quote(part) for part in cmd)


def _exec_start(config_path: Path, mode_override: str | None) -> str:
    cmd_args = ["run-once"]
    if mode_override:
        cmd_args.extend(["--mode", mode_override])
    return _exec_start_for_args(config_path, cmd_args)


def _write_unit(path: Path, content: str, written: List[Path]) -> None:
    path.write_text(content)
    written.append(path)


def _coerce_int(value: Any, fallback: int, minimum: int = 1) -> int:
    try:
        out = int(value)
    except Exception:
        out = fallback
    if out < minimum:
        return minimum
    return out


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _fleet_media_jobs(config: Dict[str, Any]) -> List[Tuple[str, str | None, int, List[str] | None]]:
    fleet = config.get("fleet", {}) if isinstance(config.get("fleet", {}), dict) else {}
    role = str(fleet.get("role", "standalone")).strip().lower()
    if role != "queen":
        return []

    media = fleet.get("queen_media_schedule", {})
    if not isinstance(media, dict) or not bool(media.get("enabled", False)):
        return []

    capture = config.get("capture", {}) if isinstance(config.get("capture", {}), dict) else {}
    pull_interval = _coerce_int(
        media.get("pull_interval_minutes", capture.get("record_interval_minutes", 30)),
        fallback=30,
        minimum=1,
    )
    track_interval = _coerce_int(media.get("track_interval_minutes", 60), fallback=60, minimum=1)
    max_videos_total = _coerce_int(media.get("max_videos_total", 200), fallback=200, minimum=1)
    cooldown_minutes = _coerce_int(media.get("cooldown_minutes", 60), fallback=60, minimum=0)
    max_queen_load_raw = media.get("max_queen_load_1m", 3.0)
    max_queen_load = None if max_queen_load_raw is None else _coerce_float(max_queen_load_raw, 3.0)
    min_queen_mem_raw = media.get("min_queen_mem_gb", 0.8)
    min_queen_mem = None if min_queen_mem_raw is None else _coerce_float(min_queen_mem_raw, 0.8)
    disable_visualization = bool(media.get("disable_visualization", False))
    allow_when_active = bool(media.get("allow_when_queen_bbox_active", False))
    output_root = str(media.get("output_root", "") or "").strip()

    pull_args: List[str] = [
        "fleet",
        "queen-pull-latest",
        "--max-videos-total",
        str(max_videos_total),
    ]
    if output_root:
        pull_args.extend(["--output-root", output_root])

    track_args: List[str] = [
        "fleet",
        "queen-track-latest",
        "--max-videos-total",
        str(max_videos_total),
        "--cooldown-minutes",
        str(cooldown_minutes),
    ]
    if max_queen_load is not None:
        track_args.extend(["--max-queen-load-1m", str(max_queen_load)])
    if min_queen_mem is not None:
        track_args.extend(["--min-queen-mem-gb", str(min_queen_mem)])
    if output_root:
        track_args.extend(["--output-root", output_root])
    if disable_visualization:
        track_args.append("--no-visualization")
    if allow_when_active:
        track_args.append("--allow-when-queen-bbox-active")

    return [
        ("queen-pull-latest", None, pull_interval, pull_args),
        ("queen-track-latest", None, track_interval, track_args),
    ]


def _job_specs(config: Dict[str, Any]) -> List[Tuple[str, str | None, int, List[str] | None]]:
    jobs: List[Tuple[str, str | None, int, List[str] | None]]
    jobs = []

    fleet = config.get("fleet", {}) if isinstance(config.get("fleet", {}), dict) else {}
    role = str(fleet.get("role", "standalone")).strip().lower()
    queen_local_enabled = bool(fleet.get("queen_local_pipeline_enabled", True))
    pipeline_enabled = not (role == "queen" and not queen_local_enabled)

    if pipeline_enabled:
        mode = config["pipeline"]["mode"]
        record_interval = int(config["capture"]["record_interval_minutes"])
        track_interval = int(config["capture"]["track_interval_minutes"])
        if mode == "mixed_schedule":
            jobs.extend(
                [
                    ("record", "record_only", record_interval, None),
                    ("track", "track_only", track_interval, None),
                ]
            )
        elif mode == "track_only":
            jobs.append(("track", "track_only", track_interval, None))
        else:
            jobs.append(("main", None, record_interval, None))

    jobs.extend(_fleet_media_jobs(config))
    return jobs


def _timer_names(config: Dict[str, Any]) -> List[str]:
    unit_prefix = str(config["scheduling"].get("unit_prefix", "bumblebox-v2")).strip() or "bumblebox-v2"
    return [f"{unit_prefix}-{label}.timer" for label, _, _, _ in _job_specs(config)]


def _service_names(config: Dict[str, Any]) -> List[str]:
    unit_prefix = str(config["scheduling"].get("unit_prefix", "bumblebox-v2")).strip() or "bumblebox-v2"
    return [f"{unit_prefix}-{label}.service" for label, _, _, _ in _job_specs(config)]


def _run_command(command: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        return False, f"$ {' '.join(command)}\n{exc}"

    output = (proc.stdout or "") + (proc.stderr or "")
    prefixed = f"$ {' '.join(command)}\n{output.strip()}".strip()
    return proc.returncode == 0, prefixed


def _install_units(config: Dict[str, Any], output_dir: Path) -> Tuple[bool, List[List[str]], List[str], str]:
    scope = str(config["scheduling"].get("scope", "system")).lower()
    timer_names = _timer_names(config)
    service_names = _service_names(config)
    output_logs: List[str] = []
    commands: List[List[str]] = []

    if scope == "system":
        target_dir = Path("/etc/systemd/system")
    else:
        target_dir = Path.home() / ".config" / "systemd" / "user"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return False, commands, output_logs, f"Could not create target systemd directory {target_dir}: {exc}"

    for filename in service_names + timer_names:
        src = output_dir / filename
        if not src.exists():
            return False, commands, output_logs, f"Missing generated unit file: {src}"
        dst = target_dir / filename
        try:
            shutil.copy2(src, dst)
        except PermissionError as exc:
            return (
                False,
                commands,
                output_logs,
                (
                    f"Permission denied while copying to {dst}. "
                    "For scope=system, run this action as root or use `sudo` manually."
                ),
            )
        except Exception as exc:
            return False, commands, output_logs, f"Failed to copy {src} to {dst}: {exc}"

    if scope == "system":
        commands = [
            ["systemctl", "daemon-reload"],
            ["systemctl", "enable", "--now", *timer_names],
        ]
    else:
        commands = [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", *timer_names],
        ]

    ok = True
    for command in commands:
        cmd_ok, cmd_output = _run_command(command)
        output_logs.append(cmd_output)
        ok = ok and cmd_ok

    note = ""
    if scope == "user":
        note = "If timers should continue without an active login session, run: loginctl enable-linger $(whoami)"

    return ok, commands, output_logs, note


def run_systemd_action(
    config: Dict[str, Any],
    action: str,
    output_dir: str | Path,
    config_path: str | Path,
) -> SystemdActionResult:
    action = action.strip().lower()
    if action not in {"install", "enable", "disable", "status"}:
        raise ValueError(f"Unsupported systemd action: {action}")

    scheduling = config["scheduling"]
    backend = str(scheduling.get("backend", "systemd")).lower()
    if backend != "systemd":
        raise ValueError(f"Config scheduling.backend is '{backend}', expected 'systemd'.")

    scope = str(scheduling.get("scope", "system")).lower()
    timer_names = _timer_names(config)
    if not timer_names:
        raise ValueError(
            "No timers defined for current config. "
            "For interface-only queens, enable fleet.queen_media_schedule.enabled."
        )
    commands: List[List[str]] = []
    outputs: List[str] = []
    note = ""

    if action == "install":
        write_systemd_units(config=config, config_path=config_path, output_dir=output_dir)
        ok, commands, outputs, note = _install_units(config=config, output_dir=Path(output_dir))
        return SystemdActionResult(
            action=action,
            scope=scope,
            timer_names=timer_names,
            commands=commands,
            command_outputs=outputs,
            success=ok,
            note=note,
        )

    if scope == "system":
        if action == "enable":
            commands = [["systemctl", "enable", "--now", *timer_names]]
        elif action == "disable":
            commands = [["systemctl", "disable", "--now", *timer_names]]
        else:
            commands = [["systemctl", "status", "--no-pager", *timer_names]]
    else:
        if action == "enable":
            commands = [["systemctl", "--user", "enable", "--now", *timer_names]]
        elif action == "disable":
            commands = [["systemctl", "--user", "disable", "--now", *timer_names]]
        else:
            commands = [["systemctl", "--user", "status", "--no-pager", *timer_names]]

    ok = True
    for command in commands:
        cmd_ok, cmd_output = _run_command(command)
        outputs.append(cmd_output)
        ok = ok and cmd_ok

    if scope == "user" and action == "enable":
        note = "If timers should continue without an active login session, run: loginctl enable-linger $(whoami)"

    return SystemdActionResult(
        action=action,
        scope=scope,
        timer_names=timer_names,
        commands=commands,
        command_outputs=outputs,
        success=ok,
        note=note,
    )


def write_systemd_units(config: Dict[str, Any], config_path: str | Path, output_dir: str | Path) -> SystemdWriteResult:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduling = config["scheduling"]
    backend = str(scheduling.get("backend", "systemd")).lower()
    if backend != "systemd":
        raise ValueError(f"Config scheduling.backend is '{backend}', expected 'systemd' for systemd-write.")

    unit_prefix = str(scheduling.get("unit_prefix", "bumblebox-v2")).strip() or "bumblebox-v2"
    scope = str(scheduling.get("scope", "system")).lower()
    service_user = normalize_service_user_value(scheduling.get("service_user")) if scope == "system" else None

    written: List[Path] = []
    timers_to_enable: List[str] = []
    jobs = _job_specs(config)
    if not jobs:
        raise ValueError(
            "No systemd jobs to generate. "
            "For interface-only queens, enable fleet.queen_media_schedule.enabled "
            "or switch to --queen-bbox-active."
        )

    for label, override_mode, interval, command_args in jobs:
        service_name = f"{unit_prefix}-{label}.service"
        timer_name = f"{unit_prefix}-{label}.timer"
        service_path = output_dir / service_name
        timer_path = output_dir / timer_name

        if command_args is None:
            exec_start = _exec_start(config_path=config_path, mode_override=override_mode)
        else:
            exec_start = _exec_start_for_args(config_path=config_path, args=command_args)
        service_content = _service_text(
            description=f"BumbleBox V2 {label} job",
            working_dir=_repo_root(),
            exec_start=exec_start,
            service_user=service_user,
        )
        timer_content = _timer_text(
            description=f"BumbleBox V2 {label} schedule",
            service_name=service_name,
            interval_minutes=interval,
        )
        _write_unit(service_path, service_content, written)
        _write_unit(timer_path, timer_content, written)
        timers_to_enable.append(timer_name)

    install_commands: List[str] = []
    if scope == "system":
        install_commands.append(f"sudo cp {shlex.quote(str(output_dir))}/*.service /etc/systemd/system/")
        install_commands.append(f"sudo cp {shlex.quote(str(output_dir))}/*.timer /etc/systemd/system/")
        install_commands.append("sudo systemctl daemon-reload")
        install_commands.append(f"sudo systemctl enable --now {' '.join(timers_to_enable)}")
        install_commands.append(f"systemctl status {' '.join(timers_to_enable)}")
    else:
        install_commands.append("mkdir -p ~/.config/systemd/user")
        install_commands.append(f"cp {shlex.quote(str(output_dir))}/*.service ~/.config/systemd/user/")
        install_commands.append(f"cp {shlex.quote(str(output_dir))}/*.timer ~/.config/systemd/user/")
        install_commands.append("systemctl --user daemon-reload")
        install_commands.append(f"systemctl --user enable --now {' '.join(timers_to_enable)}")
        install_commands.append("loginctl enable-linger $(whoami)")
        install_commands.append(f"systemctl --user status {' '.join(timers_to_enable)}")

    return SystemdWriteResult(output_dir=output_dir, written_files=written, install_commands=install_commands)


def format_systemd_result(result: SystemdWriteResult) -> str:
    lines = [
        f"Wrote {len(result.written_files)} files to {result.output_dir}",
        "",
        "Files:",
    ]
    for path in result.written_files:
        lines.append(f"- {path}")

    lines.extend(["", "Suggested install commands:"])
    for command in result.install_commands:
        lines.append(command)
    return "\n".join(lines)


def format_systemd_action_result(result: SystemdActionResult) -> str:
    lines = [
        f"Action: {result.action}",
        f"Scope: {result.scope}",
        f"Timers: {', '.join(result.timer_names)}",
        f"Success: {result.success}",
        "",
        "Command output:",
    ]
    if not result.command_outputs:
        lines.append("(no command output)")
    else:
        for chunk in result.command_outputs:
            lines.append(chunk)
            lines.append("")
    if result.note:
        lines.append(f"Note: {result.note}")
    return "\n".join(lines).rstrip()
