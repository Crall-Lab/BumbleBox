from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .calibration import (
    apply_scale_to_config,
    calibrate_from_aruco_image,
    calibrate_from_points,
    format_calibration,
    parse_point,
)
from .camera_setup import (
    format_camera_preview_result,
    format_tracking_test_result,
    run_camera_preview,
    run_camera_tracking_test,
    write_tracking_test_json,
)
from .config import (
    DEFAULT_USER_CONFIG_PATH,
    ConfigError,
    load_config,
    load_defaults,
    save_config,
    write_default_config,
)
from .doctor import format_report as format_doctor_report
from .doctor import has_failures, run_doctor
from .fps_report import build_fps_report, format_report as format_fps_report, write_report_json
from .fps_sweep import (
    format_fps_sweep_report,
    fps_range,
    parse_fps_values,
    run_fps_sweep,
    write_fps_sweep_json,
)
from .fleet import (
    apply_queen_media_schedule_defaults,
    enroll_worker_config,
    format_fleet_discovery_report,
    format_fleet_enroll_result,
    format_fleet_init_result,
    format_fleet_status_report,
    format_queen_latest_status_report,
    format_queen_track_report,
    initialize_queen_config,
    run_fleet_discovery,
    run_queen_pull_latest_videos,
    run_queen_pull_track_latest,
    run_queen_latest_status,
    run_queen_track_latest_videos,
    run_fleet_status,
    sync_media_capacity_to_workers,
    write_fleet_discovery_json,
    write_fleet_status_json,
    write_queen_latest_status_json,
    write_queen_track_json,
)
from .gui_launcher import format_gui_shortcut_result, install_gui_shortcut
from .nest_labeling import (
    build_nest_labeling_command,
    check_nest_labeling_environment,
    default_script_path,
    format_nest_labeling_environment,
    launch_nest_labeling,
)
from .roadmap import render_roadmap
from .run_bundle import (
    export_run_bundle,
    find_latest_summary_path,
    format_bundle_export_result,
    resolve_summary_from_session_dir,
)
from .run_engine import format_run_summary, reset_camera_runtime, run_once
from .schedule_check import format_schedule_check_report, run_schedule_check
from .storage_manager import (
    build_storage_mount_sudo_command,
    build_storage_setup_sudo_command,
    format_storage_mount_result,
    format_storage_setup_result,
    format_storage_status_report,
    get_storage_status,
    mount_storage_device_now,
    setup_storage_auto_mount,
)
from .systemd_units import (
    format_systemd_action_result,
    format_systemd_result,
    run_systemd_action,
    write_systemd_units,
)
from .thermal_camera import (
    apply_detected_thermal_config,
    capture_thermal_snapshot,
    format_thermal_check_result,
    format_thermal_snapshot_result,
    run_thermal_check,
    write_thermal_check_json,
    write_thermal_snapshot_json,
)


class _LiveProgressPrinter:
    """Print progress normally, but redraw optimizer tables in-place on TTYs."""

    _LIVE_TABLE_MARKER = "top mean-detection candidates so far"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled and sys.stdout.isatty())
        self._live_lines = 0

    def __call__(self, message: str) -> None:
        text = str(message)
        is_live_table = self._LIVE_TABLE_MARKER in text
        if self.enabled and is_live_table:
            self._clear_live_block()
            sys.stdout.write(text.rstrip("\n") + "\n")
            sys.stdout.flush()
            self._live_lines = max(1, len(text.rstrip("\n").splitlines()))
            return

        self._live_lines = 0
        print(text, flush=True)

    def _clear_live_block(self) -> None:
        if self._live_lines <= 0:
            return
        # Move to the first line of the previous live block and clear downward.
        sys.stdout.write(f"\033[{self._live_lines}F\033[J")


def _load_or_defaults(config_path: Path):
    if config_path.exists():
        return load_config(config_path)
    return load_defaults()


def _fleet_media_section(config: dict) -> dict:
    fleet = config.get("fleet", {})
    if not isinstance(fleet, dict):
        return {}
    media = fleet.get("queen_media_schedule", {})
    return media if isinstance(media, dict) else {}


def _coerce_int(value: object, fallback: int, minimum: int = 0) -> int:
    try:
        out = int(value)
    except Exception:
        out = fallback
    if out < minimum:
        return minimum
    return out


def _coerce_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _parse_comma_numeric_values(raw: str, *, label: str, value_type: str) -> list[float | int]:
    text = str(raw or "").strip()
    if not text:
        return []
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{label} is empty.")

    out: list[float | int] = []
    for token in tokens:
        if value_type == "float":
            try:
                value = float(token)
            except Exception as exc:
                raise ValueError(f"{label} has invalid float value: {token}") from exc
            out.append(value)
            continue

        if value_type == "int":
            try:
                value_float = float(token)
            except Exception as exc:
                raise ValueError(f"{label} has invalid integer value: {token}") from exc
            if not value_float.is_integer():
                raise ValueError(f"{label} requires whole numbers, got: {token}")
            out.append(int(value_float))
            continue

        raise ValueError(f"Unsupported numeric parser type: {value_type}")

    unique = []
    for value in out:
        if value not in unique:
            unique.append(value)
    return unique


def _optimizer_sweep_overrides_from_args(args: argparse.Namespace) -> dict[str, list[float | int]]:
    sweep_overrides: dict[str, list[float | int]] = {}
    min_perimeter = _parse_comma_numeric_values(
        getattr(args, "sweep_min_marker_perimeter_rate", ""),
        label="--sweep-min-marker-perimeter-rate",
        value_type="float",
    )
    if min_perimeter:
        sweep_overrides["minMarkerPerimeterRate"] = min_perimeter

    max_perimeter = _parse_comma_numeric_values(
        getattr(args, "sweep_max_marker_perimeter_rate", ""),
        label="--sweep-max-marker-perimeter-rate",
        value_type="float",
    )
    if max_perimeter:
        sweep_overrides["maxMarkerPerimeterRate"] = max_perimeter

    win_min = _parse_comma_numeric_values(
        getattr(args, "sweep_adaptive_thresh_win_size_min", ""),
        label="--sweep-adaptive-thresh-win-size-min",
        value_type="int",
    )
    if win_min:
        sweep_overrides["adaptiveThreshWinSizeMin"] = win_min

    win_max = _parse_comma_numeric_values(
        getattr(args, "sweep_adaptive_thresh_win_size_max", ""),
        label="--sweep-adaptive-thresh-win-size-max",
        value_type="int",
    )
    if win_max:
        sweep_overrides["adaptiveThreshWinSizeMax"] = win_max

    win_step = _parse_comma_numeric_values(
        getattr(args, "sweep_adaptive_thresh_win_size_step", ""),
        label="--sweep-adaptive-thresh-win-size-step",
        value_type="int",
    )
    if win_step:
        sweep_overrides["adaptiveThreshWinSizeStep"] = win_step

    poly = _parse_comma_numeric_values(
        getattr(args, "sweep_polygonal_approx_accuracy_rate", ""),
        label="--sweep-polygonal-approx-accuracy-rate",
        value_type="float",
    )
    if poly:
        sweep_overrides["polygonalApproxAccuracyRate"] = poly

    thresh_constant = _parse_comma_numeric_values(
        getattr(args, "sweep_adaptive_thresh_constant", ""),
        label="--sweep-adaptive-thresh-constant",
        value_type="int",
    )
    if thresh_constant:
        sweep_overrides["adaptiveThreshConstant"] = thresh_constant

    return sweep_overrides


def _optimizer_allowed_tag_ids_from_args(args: argparse.Namespace, config: dict) -> set[int]:
    from .posthoc_tracking import load_tag_ids

    allowed_tag_ids: set[int] = set()
    tracking_filter_config = (
        config.get("tracking", {})
        if isinstance(config.get("tracking", {}), dict)
        else {}
    )
    raw_config_allowed = tracking_filter_config.get("allowed_tag_ids", [])
    if isinstance(raw_config_allowed, list):
        for raw_id in raw_config_allowed:
            try:
                allowed_tag_ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue
    raw_config_tag_list = str(tracking_filter_config.get("allowed_tag_ids_path") or "").strip()
    if raw_config_tag_list:
        allowed_tag_ids.update(load_tag_ids(raw_config_tag_list))

    cli_allowed_ids = _parse_comma_numeric_values(
        getattr(args, "allowed_tag_ids", ""),
        label="--allowed-tag-ids",
        value_type="int",
    )
    allowed_tag_ids.update(int(value) for value in cli_allowed_ids)
    tag_list = getattr(args, "tag_list", None)
    if tag_list:
        allowed_tag_ids.update(load_tag_ids(tag_list))
    return allowed_tag_ids


def _apply_camera_bool_override(
    config: dict,
    *,
    key: str,
    requested_value: bool | None,
    context_label: str,
) -> bool:
    if requested_value is None:
        return True

    config.setdefault("camera", {})
    saved_value = bool(config["camera"].get(key, False))
    requested_bool = bool(requested_value)
    if requested_bool != saved_value:
        prompt = (
            f"{context_label} is overriding saved camera.{key}={saved_value} "
            f"with {requested_bool} for this run only. Continue? [y/N]: "
        )
        if not sys.stdin.isatty():
            print(
                f"Refusing to override saved camera.{key} in non-interactive mode. "
                "Run interactively, change the config, or remove the override flag."
            )
            return False
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print("Cancelled.")
            return False
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return False

    config["camera"][key] = requested_bool
    return True


def _apply_runtime_bool_override(
    config: dict,
    *,
    key: str,
    requested_value: bool | None,
    context_label: str,
) -> bool:
    if requested_value is None:
        return True

    config.setdefault("runtime", {})
    saved_value = bool(config["runtime"].get(key, False))
    requested_bool = bool(requested_value)
    if requested_bool != saved_value:
        prompt = (
            f"{context_label} is overriding saved runtime.{key}={saved_value} "
            f"with {requested_bool} for this run only. Continue? [y/N]: "
        )
        if not sys.stdin.isatty():
            print(
                f"Refusing to override saved runtime.{key} in non-interactive mode. "
                "Run interactively, change the config, or remove the override flag."
            )
            return False
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print("Cancelled.")
            return False
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return False

    config["runtime"][key] = requested_bool
    return True


def _apply_infrared_override(
    config: dict,
    *,
    requested_infrared: bool | None,
    context_label: str,
) -> bool:
    if requested_infrared is None:
        return True

    config.setdefault("camera", {})
    explicit_tuning = config["camera"].get("tuning_file")
    if explicit_tuning not in (None, ""):
        print(
            "Note: camera.tuning_file is explicitly set to "
            f"{explicit_tuning!r}; that manual tuning file still overrides auto IR/NoIR selection."
        )

    return _apply_camera_bool_override(
        config,
        key="infrared",
        requested_value=requested_infrared,
        context_label=context_label,
    )


def _apply_monochrome_output_override(
    config: dict,
    *,
    requested_monochrome_output: bool | None,
    context_label: str,
) -> bool:
    return _apply_camera_bool_override(
        config,
        key="monochrome_output",
        requested_value=requested_monochrome_output,
        context_label=context_label,
    )


def _cmd_init(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        write_default_config(config_path, force=args.force)
        print(f"Created BumbleBox V2 config: {config_path}")
    except FileExistsError as exc:
        print(str(exc))
        return 1
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    results = run_doctor(config)
    print(format_doctor_report(results))
    return 1 if has_failures(results) else 0


def _cmd_storage_status(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    report = get_storage_status(
        config=config,
        mount_point=args.mount_point,
    )
    print(format_storage_status_report(report))
    return 0 if report.mounted and report.writable else 1


def _cmd_storage_set_mount_point(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    mount_point = str(args.mount_point or "").strip()
    if not mount_point:
        print("Invalid mount point: value is empty.")
        return 1

    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    config.setdefault("system", {})
    config["system"]["data_root"] = mount_point

    if args.dry_run:
        print(f"Dry run: would set system.data_root to {mount_point} in {config_path}")
        return 0

    try:
        save_config(config_path, config)
    except Exception as exc:
        print(f"Failed to save config: {exc}")
        return 1

    print(f"Updated system.data_root to {mount_point} in {config_path}")
    return 0


def _cmd_storage_setup(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    mount_point = str(args.mount_point or config.get("system", {}).get("data_root") or "").strip()
    if not mount_point:
        mount_point = "/mnt/bumblebox/data"

    status = get_storage_status(config=config, mount_point=mount_point)
    if status.mounted and status.writable:
        print(format_storage_status_report(status))
        return 0

    try:
        result = setup_storage_auto_mount(
            mount_point=mount_point,
            device_path=args.device,
            dry_run=bool(args.dry_run),
        )
    except PermissionError as exc:
        print(f"Storage setup needs elevated privileges: {exc}")
        sudo_cmd = build_storage_setup_sudo_command(
            config_path=str(config_path),
            mount_point=mount_point,
            device_path=args.device,
            apply_config=bool(args.apply_config),
            dry_run=bool(args.dry_run),
        )
        print("\nRun this command on the Pi:")
        print(sudo_cmd)
        return 1
    except Exception as exc:
        print(f"Storage setup failed: {exc}")
        return 1

    print(format_storage_setup_result(result))
    if args.apply_config and not args.dry_run:
        config.setdefault("system", {})
        config["system"]["data_root"] = mount_point
        try:
            save_config(config_path, config)
        except Exception as exc:
            print(f"\nStorage setup succeeded, but config save failed: {exc}")
            return 1
        print(f"\nUpdated config data_root: {config_path}")
    return 0 if result.mounted_now or args.dry_run else 1


def _cmd_storage_mount(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    mount_point = str(args.mount_point or config.get("system", {}).get("data_root") or "").strip()
    if not mount_point:
        mount_point = "/mnt/bumblebox/data"

    status = get_storage_status(config=config, mount_point=mount_point)
    if status.mounted and status.writable and (
        not args.device or status.device_path == args.device
    ):
        print(format_storage_status_report(status))
        if args.apply_config and not args.dry_run:
            config.setdefault("system", {})
            config["system"]["data_root"] = mount_point
            try:
                save_config(config_path, config)
            except Exception as exc:
                print(f"\nStorage is mounted, but config save failed: {exc}")
                return 1
            print(f"\nUpdated config data_root: {config_path}")
        return 0

    try:
        result = mount_storage_device_now(
            mount_point=mount_point,
            device_path=args.device,
            dry_run=bool(args.dry_run),
        )
    except PermissionError as exc:
        print(f"Storage mount needs elevated privileges: {exc}")
        sudo_cmd = build_storage_mount_sudo_command(
            config_path=str(config_path),
            mount_point=mount_point,
            device_path=args.device,
            apply_config=bool(args.apply_config),
            dry_run=bool(args.dry_run),
        )
        print("\nRun this command on the Pi:")
        print(sudo_cmd)
        return 1
    except Exception as exc:
        print(f"Storage mount failed: {exc}")
        return 1

    print(format_storage_mount_result(result))
    if args.apply_config and not args.dry_run:
        config.setdefault("system", {})
        config["system"]["data_root"] = mount_point
        try:
            save_config(config_path, config)
        except Exception as exc:
            print(f"\nStorage mount succeeded, but config save failed: {exc}")
            return 1
        print(f"\nUpdated config data_root: {config_path}")
    return 0 if result.mounted_now or args.dry_run else 1


def _cmd_camera_preview(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    if not _apply_infrared_override(
        config,
        requested_infrared=getattr(args, "infrared", None),
        context_label="camera-preview",
    ):
        return 1
    if not _apply_monochrome_output_override(
        config,
        requested_monochrome_output=getattr(args, "monochrome_output", None),
        context_label="camera-preview",
    ):
        return 1

    try:
        result = run_camera_preview(
            config=config,
            preview_seconds=args.seconds,
            window=args.window,
            width=args.width,
            height=args.height,
        )
    except Exception as exc:
        print(f"Camera preview failed: {exc}")
        return 1

    print(format_camera_preview_result(result))
    return 0


def _format_camera_reset_result(result) -> str:
    lines = [
        "Camera Reset",
        "------------",
        f"Probe size: {result.probe_width}x{result.probe_height}",
        f"Detected cameras before reset: {result.detected_cameras_before if result.detected_cameras_before is not None else 'unknown'}",
        f"Detected cameras after reset: {result.detected_cameras_after if result.detected_cameras_after is not None else 'unknown'}",
        f"Post-reset settle time: {result.settle_seconds:.2f}s",
    ]
    if result.note:
        lines.append(f"Note: {result.note}")
    return "\n".join(lines)


def _cmd_camera_reset(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        result = reset_camera_runtime(
            config=config,
            settle_seconds=float(args.settle_seconds),
        )
    except Exception as exc:
        print(f"Camera reset failed: {exc}")
        return 1

    print(_format_camera_reset_result(result))
    return 0 if result.detected_cameras_after != 0 else 1


def _cmd_camera_test_tracking(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    box_preset = args.box_preset
    if box_preset == "auto":
        box_preset = None
    elif box_preset == "none":
        box_preset = "none"

    try:
        result = run_camera_tracking_test(
            config=config,
            test_seconds=args.seconds,
            display_width=args.display_width,
            dictionary_name=args.dictionary,
            box_preset=box_preset,
            show_rejected=args.show_rejected,
            use_clahe=not args.no_clahe,
        )
    except Exception as exc:
        print(f"Camera tracking test failed: {exc}")
        return 1

    print(format_tracking_test_result(result))
    if args.json_out:
        try:
            path = write_tracking_test_json(result, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Tracking test completed, but failed to write JSON report: {exc}")
            return 1
    return 0


def _cmd_roadmap(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    print(render_roadmap(config, config_path))
    return 0


def _cmd_fps_report(args: argparse.Namespace) -> int:
    try:
        report = build_fps_report(
            args.video,
            timestamps_path=args.timestamps,
            recording_seconds=args.recording_seconds,
        )
    except Exception as exc:
        print(f"Failed to build FPS report: {exc}")
        return 1

    print(format_fps_report(report))

    if args.json_out:
        path = write_report_json(report, args.json_out)
        print(f"JSON report saved: {path}")
    return 0


def _cmd_fps_sweep(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        if args.fps_values:
            fps_values = parse_fps_values(args.fps_values)
        else:
            fps_values = fps_range(args.fps_start, args.fps_stop, args.fps_step)
    except Exception as exc:
        print(f"FPS sweep argument error: {exc}")
        return 1

    def _progress(done: int, total: int, target_fps: float) -> None:
        print(f"[{done}/{total}] probing target fps {target_fps:.3f}")

    try:
        report = run_fps_sweep(
            config=config,
            fps_values=fps_values,
            probe_seconds=args.probe_seconds,
            assume_ram_gb=args.assume_ram_gb,
            use_mock_camera=args.mock_camera if args.mock_camera else None,
            session_start_iso=args.session_start,
            progress_callback=_progress,
        )
    except Exception as exc:
        print(f"FPS sweep failed: {exc}")
        return 1

    print("")
    print(format_fps_sweep_report(report))

    if args.json_out:
        try:
            path = write_fps_sweep_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"FPS sweep completed, but failed to write JSON report: {exc}")
            return 1
    return 1 if report.has_errors else 0


def _cmd_thermal_check(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        result = run_thermal_check(
            config=config,
            device_override=args.device,
            width_override=args.width,
            height_override=args.height,
        )
    except Exception as exc:
        print(f"Thermal check failed: {exc}")
        return 1

    print(format_thermal_check_result(result))

    if args.apply:
        try:
            updated = apply_detected_thermal_config(config, result)
            save_config(config_path, updated)
            print(f"\nApplied detected thermal settings to: {config_path}")
            print(f"  thermal.enabled = true")
            print(f"  thermal.device_path = {updated.get('thermal', {}).get('device_path')}")
            print(f"  thermal.pixel_format = {updated.get('thermal', {}).get('pixel_format')}")
        except Exception as exc:
            print(f"\nThermal check completed, but apply failed: {exc}")
            return 1

    if args.json_out:
        try:
            path = write_thermal_check_json(result, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Thermal check completed, but failed to write JSON report: {exc}")
            return 1
    return 1 if result.errors else 0


def _cmd_thermal_snapshot(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        result = capture_thermal_snapshot(
            config=config,
            device_override=args.device,
            width_override=args.width,
            height_override=args.height,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Thermal snapshot failed: {exc}")
        return 1

    print(format_thermal_snapshot_result(result))

    if args.json_out:
        try:
            path = write_thermal_snapshot_json(result, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Thermal snapshot completed, but failed to write JSON report: {exc}")
            return 1
    return 0


def _cmd_calibrate_manual(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
        point_a = parse_point(args.point_a)
        point_b = parse_point(args.point_b)
        calibration = calibrate_from_points(point_a, point_b, args.distance_cm)
        updated = apply_scale_to_config(config, calibration)

        pixel_contact = updated.get("metrics", {}).get("pixel_contact_distance")
        print(format_calibration(calibration, pixel_contact_distance=pixel_contact))
        if not args.dry_run:
            save_config(config_path, updated)
            print(f"Updated config: {config_path}")
    except Exception as exc:
        print(f"Calibration failed: {exc}")
        return 1

    return 0


def _cmd_calibrate_aruco(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
        calibration = calibrate_from_aruco_image(
            image_path=args.image,
            marker_size_mm=args.marker_size_mm,
            dictionary_name=args.dictionary,
            marker_id=args.marker_id,
        )
        updated = apply_scale_to_config(config, calibration)

        pixel_contact = updated.get("metrics", {}).get("pixel_contact_distance")
        print(format_calibration(calibration, pixel_contact_distance=pixel_contact))
        if not args.dry_run:
            save_config(config_path, updated)
            print(f"Updated config: {config_path}")
    except Exception as exc:
        print(f"Calibration failed: {exc}")
        return 1

    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    del args
    try:
        from .gui_app import launch
    except Exception as exc:
        print(f"Failed to load GUI: {exc}")
        return 1

    launch()
    return 0


def _cmd_gui_install_shortcut(args: argparse.Namespace) -> int:
    try:
        result = install_gui_shortcut(
            name=args.name,
            comment=args.comment,
            repo_root=args.repo_root,
            desktop_dir=args.desktop_dir,
            applications_dir=args.applications_dir,
            bin_dir=args.bin_dir,
            icon_path=args.icon_path,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Failed to install GUI desktop shortcut: {exc}")
        return 1

    print(format_gui_shortcut_result(result))
    return 0


def _cmd_nest_label_check(args: argparse.Namespace) -> int:
    try:
        env = check_nest_labeling_environment(
            image_folder=args.folder,
            script_path=args.script,
            labelmerc_override=args.labelmerc,
            python_executable=args.python,
        )
    except Exception as exc:
        print(f"Nest labeling check failed: {exc}")
        return 1

    print(format_nest_labeling_environment(env))
    return 0 if env.ready else 1


def _cmd_nest_label_launch(args: argparse.Namespace) -> int:
    if not args.folder:
        print("Missing --folder. Provide the image folder to label.")
        return 1

    try:
        process = launch_nest_labeling(
            image_folder=args.folder,
            script_path=args.script,
            labelmerc_override=args.labelmerc,
            python_executable=args.python,
        )
        command = build_nest_labeling_command(
            image_folder=args.folder,
            script_path=args.script,
            labelmerc_override=args.labelmerc,
            python_executable=args.python,
        )
    except Exception as exc:
        print(f"Nest labeling launch failed: {exc}")
        return 1

    print(f"Launched nest labeling (pid {process.pid}).")
    print("Command:", " ".join(shlex.quote(part) for part in command))
    if args.wait:
        code = process.wait()
        print(f"Nest labeling process exited with code {code}.")
        return code
    return 0


def _cmd_run_once(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    if args.mock_camera:
        config.setdefault("runtime", {})
        config["runtime"]["use_mock_camera"] = True
    if args.codec:
        config.setdefault("camera", {})
        config["camera"]["codec"] = str(args.codec).strip().lower()
    if not _apply_infrared_override(
        config,
        requested_infrared=getattr(args, "infrared", None),
        context_label="run-once",
    ):
        return 1
    if not _apply_monochrome_output_override(
        config,
        requested_monochrome_output=getattr(args, "monochrome_output", None),
        context_label="run-once",
    ):
        return 1
    if not _apply_runtime_bool_override(
        config,
        key="render_tracking_video",
        requested_value=getattr(args, "visualization", None),
        context_label="run-once",
    ):
        return 1

    try:
        summary = run_once(config=config, mode_override=args.mode)
    except Exception as exc:
        print(f"Run failed to start: {exc}")
        return 1

    print(format_run_summary(summary))
    return 0 if summary.success else 1


def _cmd_fleet_init_queen(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        queen_pipeline_enabled = bool(args.queen_bbox_active)
        if args.queen_interface_only:
            queen_pipeline_enabled = False

        updated, result = initialize_queen_config(
            config=config,
            queen_host=args.queen_host,
            identity_file=args.identity_file,
            ssh_user=args.ssh_user,
            skip_keygen=args.skip_keygen,
        )
        updated.setdefault("fleet", {})
        updated["fleet"]["queen_local_pipeline_enabled"] = queen_pipeline_enabled
        media = apply_queen_media_schedule_defaults(
            updated,
            enable=(not queen_pipeline_enabled),
        )
        media["max_videos_total"] = sync_media_capacity_to_workers(updated, include_disabled=False, minimum=1)
        if not args.dry_run:
            save_config(config_path, updated)
    except Exception as exc:
        print(f"Fleet init failed: {exc}")
        return 1

    print(format_fleet_init_result(result, show_public_key=args.show_public_key))
    if args.dry_run:
        print("\nDry run: config was not written.")
    else:
        print(f"\nUpdated config: {config_path}")
        print(
            "Queen local pipeline is "
            + ("enabled (--queen-bbox-active)." if queen_pipeline_enabled else "disabled (--queen-interface-only).")
        )
        print(
            "Queen media schedule is "
            + (
                f"enabled (pull every {media.get('pull_interval_minutes')} min, track every {media.get('track_interval_minutes')} min)."
                if bool(media.get("enabled"))
                else "disabled."
            )
        )
    return 0


def _cmd_fleet_enroll_worker(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        updated, result = enroll_worker_config(
            config=config,
            host=args.host,
            name=args.name,
            user=args.user,
            port=args.port,
            data_root=args.data_root,
            unit_prefix=args.unit_prefix,
            enabled=not args.disabled,
            identity_file=args.identity_file,
            install_key=args.install_key,
        )
        sync_media_capacity_to_workers(updated, include_disabled=False, minimum=1)
        if not args.dry_run:
            save_config(config_path, updated)
    except Exception as exc:
        print(f"Fleet enroll failed: {exc}")
        return 1

    print(format_fleet_enroll_result(result))
    if args.dry_run:
        print("\nDry run: config was not written.")
    else:
        print(f"\nUpdated config: {config_path}")
    return 0


def _cmd_fleet_status(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        report = run_fleet_status(
            config=config,
            include_disabled=args.include_disabled,
            worker_filter=args.worker,
            identity_file=args.identity_file,
        )
    except Exception as exc:
        print(f"Fleet status failed: {exc}")
        return 1

    print(format_fleet_status_report(report))
    if args.json_out:
        try:
            path = write_fleet_status_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Failed to save fleet status JSON: {exc}")
            return 1

    return 1 if report.fail_count > 0 else 0


def _cmd_fleet_latest_status(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        report = run_queen_latest_status(
            config=config,
            include_disabled=args.include_disabled,
            worker_filter=args.worker,
            output_root=args.output_root,
            identity_file=args.identity_file,
            probe_reachability=not bool(args.no_reachability),
        )
    except Exception as exc:
        print(f"Queen latest-status failed: {exc}")
        return 1

    print(format_queen_latest_status_report(report))
    if args.json_out:
        try:
            path = write_queen_latest_status_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Failed to save queen latest-status JSON: {exc}")
            return 1
    return 1 if report.offline_count > 0 else 0


def _cmd_fleet_discover(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        report = run_fleet_discovery(
            config=config,
            include_disabled=args.include_disabled,
            worker_filter=args.worker,
            ping_probe=not bool(args.no_ping),
            timeout_seconds=float(args.timeout_seconds),
        )
    except Exception as exc:
        print(f"Fleet discovery failed: {exc}")
        return 1

    print(format_fleet_discovery_report(report))
    if args.json_out:
        try:
            path = write_fleet_discovery_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Failed to save fleet discovery JSON: {exc}")
            return 1
    return 1 if report.configured_workers_offline > 0 else 0


def _cmd_fleet_queen_pull_latest(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    media = _fleet_media_section(config)
    max_videos_total = args.max_videos_total
    if max_videos_total is None:
        max_videos_total = _coerce_int(media.get("max_videos_total", 200), 200, minimum=1)
    output_root = args.output_root or (str(media.get("output_root", "")).strip() or None)

    try:
        report = run_queen_pull_latest_videos(
            config=config,
            include_disabled=args.include_disabled,
            worker_filter=args.worker,
            identity_file=args.identity_file,
            output_root=output_root,
            max_videos_total=max_videos_total,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"Queen pull-latest failed: {exc}")
        return 1

    print(format_queen_track_report(report))
    if args.json_out:
        try:
            path = write_queen_track_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Failed to save queen pull-latest JSON: {exc}")
            return 1
    return 1 if report.videos_failed > 0 else 0


def _cmd_fleet_queen_track_latest(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    media = _fleet_media_section(config)
    max_videos_total = args.max_videos_total
    if max_videos_total is None:
        max_videos_total = _coerce_int(media.get("max_videos_total", 200), 200, minimum=1)
    cooldown_minutes = args.cooldown_minutes
    if cooldown_minutes is None:
        cooldown_minutes = _coerce_int(media.get("cooldown_minutes", 60), 60, minimum=0)
    max_queen_load_1m = args.max_queen_load_1m
    if max_queen_load_1m is None:
        raw_load = media.get("max_queen_load_1m", 3.0)
        max_queen_load_1m = None if raw_load is None else _coerce_float(raw_load, 3.0)
    min_queen_mem_gb = args.min_queen_mem_gb
    if min_queen_mem_gb is None:
        raw_mem = media.get("min_queen_mem_gb", 0.8)
        min_queen_mem_gb = None if raw_mem is None else _coerce_float(raw_mem, 0.8)
    output_root = args.output_root or (str(media.get("output_root", "")).strip() or None)
    disable_visualization = bool(media.get("disable_visualization", False)) or bool(args.no_visualization)
    allow_when_queen_bbox_active = bool(media.get("allow_when_queen_bbox_active", False)) or bool(
        args.allow_when_queen_bbox_active
    )

    try:
        report = run_queen_track_latest_videos(
            config=config,
            include_disabled=args.include_disabled,
            worker_filter=args.worker,
            output_root=output_root,
            max_videos_total=max_videos_total,
            cooldown_minutes=cooldown_minutes,
            with_visualization=not disable_visualization,
            dry_run=bool(args.dry_run),
            allow_when_queen_bbox_active=allow_when_queen_bbox_active,
            max_queen_load_1m=max_queen_load_1m,
            min_queen_mem_available_gb=min_queen_mem_gb,
        )
    except Exception as exc:
        print(f"Queen track-latest failed: {exc}")
        return 1

    print(format_queen_track_report(report))
    if args.json_out:
        try:
            path = write_queen_track_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Failed to save queen track-latest JSON: {exc}")
            return 1
    return 1 if report.videos_failed > 0 else 0


def _cmd_fleet_queen_pull_track(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    media = _fleet_media_section(config)
    max_videos_total = args.max_videos_total
    if max_videos_total is None:
        max_videos_total = _coerce_int(media.get("max_videos_total", 200), 200, minimum=1)
    cooldown_minutes = args.cooldown_minutes
    if cooldown_minutes is None:
        cooldown_minutes = _coerce_int(media.get("cooldown_minutes", 60), 60, minimum=0)
    max_queen_load_1m = args.max_queen_load_1m
    if max_queen_load_1m is None:
        raw_load = media.get("max_queen_load_1m", 3.0)
        max_queen_load_1m = None if raw_load is None else _coerce_float(raw_load, 3.0)
    min_queen_mem_gb = args.min_queen_mem_gb
    if min_queen_mem_gb is None:
        raw_mem = media.get("min_queen_mem_gb", 0.8)
        min_queen_mem_gb = None if raw_mem is None else _coerce_float(raw_mem, 0.8)
    output_root = args.output_root or (str(media.get("output_root", "")).strip() or None)
    disable_visualization = bool(media.get("disable_visualization", False)) or bool(args.no_visualization)
    allow_when_queen_bbox_active = bool(media.get("allow_when_queen_bbox_active", False)) or bool(
        args.allow_when_queen_bbox_active
    )

    try:
        report = run_queen_pull_track_latest(
            config=config,
            include_disabled=args.include_disabled,
            worker_filter=args.worker,
            identity_file=args.identity_file,
            output_root=output_root,
            max_videos_total=max_videos_total,
            max_videos_per_worker=args.max_videos_per_worker,
            cooldown_minutes=cooldown_minutes,
            with_visualization=not disable_visualization,
            dry_run=bool(args.dry_run),
            allow_when_queen_bbox_active=allow_when_queen_bbox_active,
            max_queen_load_1m=max_queen_load_1m,
            min_queen_mem_available_gb=min_queen_mem_gb,
        )
    except Exception as exc:
        print(f"Queen pull-track failed: {exc}")
        return 1

    print(format_queen_track_report(report))
    if args.json_out:
        try:
            path = write_queen_track_json(report, args.json_out)
            print(f"\nJSON report saved: {path}")
        except Exception as exc:
            print(f"Failed to save queen pull-track JSON: {exc}")
            return 1

    return 1 if report.videos_failed > 0 else 0


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    selector_count = int(bool(args.summary)) + int(bool(args.session_dir)) + int(bool(args.latest))
    if selector_count > 1:
        print("Use only one run selector: --summary, --session-dir, or --latest.")
        return 1

    summary_path: Optional[Path] = None
    if args.summary:
        summary_path = Path(args.summary).expanduser().resolve()
    elif args.session_dir:
        try:
            summary_path = resolve_summary_from_session_dir(args.session_dir)
        except Exception as exc:
            print(f"Failed to resolve run summary from session dir: {exc}")
            return 1
    else:
        data_root = args.data_root
        if not data_root:
            config_path = Path(args.config)
            try:
                config = _load_or_defaults(config_path)
            except (FileNotFoundError, ConfigError, RuntimeError) as exc:
                print(f"Config error: {exc}")
                print("Tip: pass --data-root when using --latest without a readable config.")
                return 1
            data_root = config.get("system", {}).get("data_root")
        if not data_root:
            print("Could not determine data root for --latest. Provide --data-root.")
            return 1
        try:
            summary_path = find_latest_summary_path(data_root)
        except Exception as exc:
            print(f"Failed to find latest run summary: {exc}")
            return 1

    if summary_path is None:
        print("Could not resolve run summary path.")
        return 1

    config_path_for_bundle: Optional[Path] = None
    if not args.no_config:
        candidate = Path(args.config).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            config_path_for_bundle = candidate

    try:
        result = export_run_bundle(
            summary_path=summary_path,
            output_dir=args.output_dir,
            bundle_name=args.bundle_name,
            config_path=config_path_for_bundle,
            include_video=not bool(args.skip_video),
            include_all_session_files=not bool(args.core_only),
            zip_bundle=bool(args.zip_bundle),
        )
    except Exception as exc:
        print(f"Run bundle export failed: {exc}")
        return 1

    print(format_bundle_export_result(result))
    return 0


def _cmd_optimize_tracking(args: argparse.Namespace) -> int:
    from .tracking_optimizer import (
        apply_best_params_to_config,
        format_optimization_report,
        optimize_tracking,
    )
    from .tracking_index import format_local_index_result, sync_optimization_result

    try:
        from .posthoc_tracking import load_tag_ids

        config_for_filters = _load_or_defaults(Path(args.config))
        sweep_overrides = {}
        min_perimeter = _parse_comma_numeric_values(
            args.sweep_min_marker_perimeter_rate,
            label="--sweep-min-marker-perimeter-rate",
            value_type="float",
        )
        if min_perimeter:
            sweep_overrides["minMarkerPerimeterRate"] = min_perimeter

        max_perimeter = _parse_comma_numeric_values(
            args.sweep_max_marker_perimeter_rate,
            label="--sweep-max-marker-perimeter-rate",
            value_type="float",
        )
        if max_perimeter:
            sweep_overrides["maxMarkerPerimeterRate"] = max_perimeter

        win_min = _parse_comma_numeric_values(
            args.sweep_adaptive_thresh_win_size_min,
            label="--sweep-adaptive-thresh-win-size-min",
            value_type="int",
        )
        if win_min:
            sweep_overrides["adaptiveThreshWinSizeMin"] = win_min

        win_max = _parse_comma_numeric_values(
            args.sweep_adaptive_thresh_win_size_max,
            label="--sweep-adaptive-thresh-win-size-max",
            value_type="int",
        )
        if win_max:
            sweep_overrides["adaptiveThreshWinSizeMax"] = win_max

        win_step = _parse_comma_numeric_values(
            args.sweep_adaptive_thresh_win_size_step,
            label="--sweep-adaptive-thresh-win-size-step",
            value_type="int",
        )
        if win_step:
            sweep_overrides["adaptiveThreshWinSizeStep"] = win_step

        poly = _parse_comma_numeric_values(
            args.sweep_polygonal_approx_accuracy_rate,
            label="--sweep-polygonal-approx-accuracy-rate",
            value_type="float",
        )
        if poly:
            sweep_overrides["polygonalApproxAccuracyRate"] = poly

        thresh_constant = _parse_comma_numeric_values(
            args.sweep_adaptive_thresh_constant,
            label="--sweep-adaptive-thresh-constant",
            value_type="int",
        )
        if thresh_constant:
            sweep_overrides["adaptiveThreshConstant"] = thresh_constant

        allowed_tag_ids = set()
        tracking_filter_config = (
            config_for_filters.get("tracking", {})
            if isinstance(config_for_filters.get("tracking", {}), dict)
            else {}
        )
        raw_config_allowed = tracking_filter_config.get("allowed_tag_ids", [])
        if isinstance(raw_config_allowed, list):
            for raw_id in raw_config_allowed:
                try:
                    allowed_tag_ids.add(int(raw_id))
                except (TypeError, ValueError):
                    continue
        raw_config_tag_list = str(tracking_filter_config.get("allowed_tag_ids_path") or "").strip()
        if raw_config_tag_list:
            allowed_tag_ids.update(load_tag_ids(raw_config_tag_list))
        cli_allowed_ids = _parse_comma_numeric_values(
            args.allowed_tag_ids,
            label="--allowed-tag-ids",
            value_type="int",
        )
        allowed_tag_ids.update(int(value) for value in cli_allowed_ids)
        if args.tag_list:
            allowed_tag_ids.update(load_tag_ids(args.tag_list))

        result = optimize_tracking(
            input_path=args.input,
            profile=args.profile,
            sample_frames=args.sample_frames,
            dictionary_name=args.dictionary,
            tag_size_mm=args.tag_size_mm,
            sweep_overrides=sweep_overrides or None,
            max_parameter_combinations=args.max_combinations,
            execution_target=args.execution_target,
            workers=args.workers,
            expected_tags=args.expected_tags,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_improvement=args.early_stop_min_improvement,
            output_dir=args.output_dir,
            write_preview=args.write_preview,
            preview_frames=args.preview_frames,
            top_k=args.top_k,
            valid_tag_ids=allowed_tag_ids or None,
        )
    except Exception as exc:
        print(f"Tracking optimization failed: {exc}")
        return 1

    print(format_optimization_report(result, top_k=args.top_k))

    try:
        index_config = _load_or_defaults(Path(args.config))
        index_result = sync_optimization_result(index_config, result)
        print("")
        print(format_local_index_result(index_result))
    except Exception as exc:
        print(f"\nLocal tracking index update failed: {exc}")

    if args.apply_best:
        config_path = Path(args.config)
        try:
            config = _load_or_defaults(config_path)
            updated = apply_best_params_to_config(config, result.best_params)
            save_config(config_path, updated)
            print(f"Applied best parameters to config: {config_path}")
            try:
                selected_index_result = sync_optimization_result(
                    updated,
                    result,
                    selected_params=result.best_params,
                    selected_label="apply_best",
                )
                print("")
                print(format_local_index_result(selected_index_result))
            except Exception as index_exc:
                print(f"Applied best parameters, but failed to update selected params in local index: {index_exc}")
        except Exception as exc:
            print(f"Optimization completed, but failed to apply config update: {exc}")
            return 1

    return 0


def _safe_path_component(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    return safe.strip("_") or "item"


def _parse_video_range_text(row: dict[str, str]) -> list[tuple[int, int]]:
    text = str(row.get("frame_ranges") or "").strip()
    if not text:
        start_text = str(row.get("first_start_frame") or "").strip()
        end_text = str(row.get("last_end_frame") or "").strip()
        if start_text and end_text:
            text = f"{start_text}-{end_text}"
    if not text:
        raise ValueError("missing frame_ranges")

    ranges: list[tuple[int, int]] = []
    for token in re.split(r"[;,]+", text):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(float(start_text.strip()))
            end = int(float(end_text.strip()))
        else:
            start = end = int(float(token))
        if start < 0 or end < 0:
            raise ValueError(f"negative frame index in {text!r}")
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    if not ranges:
        raise ValueError("no usable frame ranges")
    return ranges


def _frame_indices_from_ranges(ranges: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for start, end in ranges:
        for frame_index in range(int(start), int(end) + 1):
            if frame_index in seen:
                continue
            seen.add(frame_index)
            out.append(frame_index)
    return sorted(out)


def _is_aug_2019_video_id(video_id: str) -> bool:
    return bool(re.match(r"^\d{2}-Aug-2019_", str(video_id)))


def _is_2024_no_tag_video_id(video_id: str) -> bool:
    return "2024" in str(video_id)


def _resolve_manifest_video(source_root: Path, video_id: str) -> Path | None:
    from .tracking_optimizer import VIDEO_EXTENSIONS

    search_roots = [
        source_root / "input_data" / "val",
        source_root / "input_data",
        source_root,
    ]
    extensions = VIDEO_EXTENSIONS | {".mjpe"}
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for extension in sorted(extensions):
            path = root / f"{video_id}{extension}"
            if path.exists() and path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                candidates.append(path)
        for path in root.rglob(f"{video_id}.*"):
            if (
                path.is_file()
                and path.stem == video_id
                and path.suffix.lower() in extensions
                and path.resolve() not in seen
            ):
                seen.add(path.resolve())
                candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (0 if "input_data/val" in str(item) else 1, str(item)))[0]


def _frame_index_from_image_path(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _copy_or_extract_video_range_frames(
    *,
    video_id: str,
    video_path: Path,
    source_root: Path,
    frame_indices: list[int],
    output_dir: Path,
    force: bool,
) -> tuple[list[Path], list[str]]:
    import cv2

    from .tracking_optimizer import IMAGE_EXTENSIONS

    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    written: list[Path] = []
    source_frames_dir = source_root / "frames" / video_id
    source_images_by_index: dict[int, Path] = {}
    if source_frames_dir.exists():
        for image_path in sorted(source_frames_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            frame_index = _frame_index_from_image_path(image_path)
            if frame_index is not None:
                source_images_by_index[frame_index] = image_path

    fallback_indices: list[int] = []
    for frame_index in frame_indices:
        out_path = output_dir / f"frame_{frame_index:06d}.png"
        if out_path.exists() and not force:
            written.append(out_path)
            continue
        source_image = source_images_by_index.get(frame_index)
        if source_image is None:
            fallback_indices.append(frame_index)
            continue
        frame = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if frame is None:
            warnings.append(f"Could not read extracted frame {source_image}; falling back to video seek.")
            fallback_indices.append(frame_index)
            continue
        if not cv2.imwrite(str(out_path), frame):
            warnings.append(f"Could not write frame image {out_path}")
            continue
        written.append(out_path)

    if fallback_indices:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for frame extraction: {video_path}")
        try:
            for frame_index in fallback_indices:
                out_path = output_dir / f"frame_{frame_index:06d}.png"
                if out_path.exists() and not force:
                    written.append(out_path)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    warnings.append(f"Could not read frame {frame_index} from {video_path.name}")
                    continue
                if not cv2.imwrite(str(out_path), frame):
                    warnings.append(f"Could not write frame image {out_path}")
                    continue
                written.append(out_path)
        finally:
            cap.release()

    deduped = sorted(set(written), key=lambda path: _frame_index_from_image_path(path) or -1)
    if not deduped:
        raise RuntimeError(f"No requested frames could be extracted for {video_id}")
    return deduped, warnings


def _tag_list_pair_from_path(path: Path) -> tuple[int, int] | None:
    parent_match = re.search(r"mcs-(\d+)-and-(\d+)", path.parent.name.lower())
    if parent_match:
        return int(parent_match.group(1)), int(parent_match.group(2))
    name_match = re.search(r"mc(\d+)_mc(\d+)", path.name.lower())
    if name_match:
        return int(name_match.group(1)), int(name_match.group(2))
    return None


def _resolve_tag_list_for_video(video_id: str, tag_list_root: Path | None) -> Path | None:
    if tag_list_root is None or not tag_list_root.exists():
        return None
    match = re.match(r"^bumblebox-(\d+)_", video_id)
    if not match:
        return None
    box_number = int(match.group(1))
    candidates: list[Path] = []
    for path in sorted(tag_list_root.rglob("*tag*list*.txt")):
        pair = _tag_list_pair_from_path(path)
        if pair and box_number in pair:
            candidates.append(path)
    return candidates[0] if candidates else None


def _dictionary_candidates_for_video(video_id: str, dictionary: str, dictionary_candidates: str) -> list[str]:
    requested = str(dictionary or "auto").strip()
    if requested and requested.lower() != "auto":
        return [requested]
    if "2021" in video_id:
        return ["4X4_50"]
    if "2024" in video_id:
        values = [item.strip() for item in str(dictionary_candidates or "").split(",") if item.strip()]
        return values or ["4X4_50", "4X4_100"]
    if video_id.startswith("bumblebox-"):
        return ["4X4_100"]
    return ["4X4_50"]


def _load_video_range_bounds(path_text: str | None) -> dict[str, dict[str, Any]]:
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    if not isinstance(entries, dict):
        return {}
    return {str(key): dict(value) for key, value in entries.items() if isinstance(value, dict)}


def _bounds_for_video(video_id: str, bounds_by_video: dict[str, dict[str, Any]]) -> tuple[tuple[float, float] | None, dict[str, list[float]]]:
    entry = bounds_by_video.get(video_id)
    if not isinstance(entry, dict):
        return None, {}
    review_bounds_raw = entry.get("review_perimeter_bounds")
    review_bounds: tuple[float, float] | None = None
    if isinstance(review_bounds_raw, list) and len(review_bounds_raw) == 2:
        try:
            low = float(review_bounds_raw[0])
            high = float(review_bounds_raw[1])
            if low > 0 and high > low:
                review_bounds = (low, high)
        except (TypeError, ValueError):
            review_bounds = None

    overrides: dict[str, list[float]] = {}
    for source_key, param_key in (
        ("suggested_min_marker_perimeter_rate", "minMarkerPerimeterRate"),
        ("suggested_max_marker_perimeter_rate", "maxMarkerPerimeterRate"),
    ):
        values = entry.get(source_key)
        if isinstance(values, list):
            parsed: list[float] = []
            for value in values:
                try:
                    parsed.append(float(value))
                except (TypeError, ValueError):
                    continue
            if parsed:
                overrides[param_key] = parsed
    return review_bounds, overrides


def _selected_params_from_optimization(result: Any) -> tuple[dict[str, Any], str, float]:
    candidates = list(getattr(result, "top_detection_candidates", []) or [])
    if candidates:
        candidate = candidates[0]
        return (
            dict(getattr(candidate, "params", {}) or {}),
            "top_mean_detection",
            float(getattr(candidate, "mean_detected", 0.0) or 0.0),
        )
    return (
        dict(getattr(result, "best_params", {}) or {}),
        "best_score",
        float(getattr(result, "best_mean_detected", 0.0) or 0.0),
    )


def _write_detection_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    from .tracking_optimizer import IMAGE_DETECTION_CSV_FIELDS

    fieldnames = [
        "video_id",
        "source_video_path",
        "frame",
        *IMAGE_DETECTION_CSV_FIELDS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_annotated_frame_video(annotated_paths: list[Path], output_path: Path, *, fps: float = 2.0) -> Path | None:
    import cv2

    readable: list[tuple[Path, Any]] = []
    for path in annotated_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            readable.append((path, frame))
    if not readable:
        return None

    height, width = readable[0][1].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        return None
    try:
        for _path, frame in readable:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return output_path if output_path.exists() else None


def _cmd_optimize_video_ranges(args: argparse.Namespace) -> int:
    from .config import load_config
    from .posthoc_tracking import load_tag_ids
    from .tracking_optimizer import detect_markers_in_image, normalize_dictionary_name, optimize_tracking

    manifest_path = Path(args.manifest).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    if getattr(args, "open_gui", False):
        bounds_path = (
            Path(args.tag_bounds_json).expanduser()
            if args.tag_bounds_json
            else manifest_path.with_name(f"{manifest_path.stem}_tag_bounds.json")
        )
        print("[optimize-video-ranges] Opening GUI for smallest/largest tag bounds.")
        print(f"[optimize-video-ranges] Manifest: {manifest_path}")
        print(f"[optimize-video-ranges] Source root: {source_root}")
        print(f"[optimize-video-ranges] Bounds JSON: {bounds_path}")
        try:
            from .gui_app import launch
        except Exception as exc:
            print(f"Failed to load GUI: {exc}")
            return 1
        launch(
            open_video_range_bounds=True,
            video_range_manifest=str(manifest_path),
            video_range_source_root=str(source_root),
            video_range_bounds_file=str(bounds_path),
        )
        print(
            "[optimize-video-ranges] GUI closed. "
            "Rerun the command without --open-gui to optimize with the saved bounds."
        )
        return 0

    load_config(args.config)
    output_root = Path(args.output_root).expanduser().resolve()
    tag_list_root = Path(args.tag_list_root).expanduser().resolve() if args.tag_list_root else None
    bounds_by_video = _load_video_range_bounds(getattr(args, "tag_bounds_json", None))
    global_allowed_ids = {
        int(value)
        for value in _parse_comma_numeric_values(
            getattr(args, "allowed_tag_ids", ""),
            label="--allowed-tag-ids",
            value_type="int",
        )
    }
    global_excluded_ids = {
        int(value)
        for value in _parse_comma_numeric_values(
            getattr(args, "exclude_tag_ids", ""),
            label="--exclude-tag-ids",
            value_type="int",
        )
    }
    excluded_id_exempt_prefixes = tuple(
        prefix.strip()
        for prefix in str(getattr(args, "exclude_tag_ids_except_video_prefixes", "") or "").split(",")
        if prefix.strip()
    )
    sweep_overrides_base = _optimizer_sweep_overrides_from_args(args)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_path}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            manifest_rows.append(dict(row))
    if args.limit:
        manifest_rows = manifest_rows[: max(0, int(args.limit))]

    report_rows: list[dict[str, Any]] = []
    combined_detection_rows: list[dict[str, object]] = []
    skipped = 0
    processed = 0
    failed = 0

    print(f"[optimize-video-ranges] Manifest: {manifest_path}")
    print(f"[optimize-video-ranges] Source root: {source_root}")
    print(f"[optimize-video-ranges] Output root: {output_root}")
    print(f"[optimize-video-ranges] Rows loaded: {len(manifest_rows)}")
    if tag_list_root:
        print(f"[optimize-video-ranges] Tag-list root: {tag_list_root}")
    if args.tag_bounds_json:
        print(f"[optimize-video-ranges] Tag bounds JSON: {Path(args.tag_bounds_json).expanduser()}")
    if bounds_by_video:
        print(f"[optimize-video-ranges] Per-video tag bounds loaded: {len(bounds_by_video)}")
    elif args.tag_bounds_json:
        print("[optimize-video-ranges] WARNING: no per-video tag bounds were loaded from that JSON.")

    for row_index, row in enumerate(manifest_rows, start=1):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            skipped += 1
            report_rows.append({"row_index": row_index, "status": "skipped", "reason": "missing video_id"})
            continue
        if _is_aug_2019_video_id(video_id) and not args.include_aug_2019:
            skipped += 1
            print(f"\n[optimize-video-ranges] {row_index}/{len(manifest_rows)} {video_id}: skipped Aug-2019 row")
            report_rows.append({"video_id": video_id, "row_index": row_index, "status": "skipped", "reason": "Aug-2019 no ArUco tags"})
            continue
        if _is_2024_no_tag_video_id(video_id) and not args.include_2024:
            skipped += 1
            print(f"\n[optimize-video-ranges] {row_index}/{len(manifest_rows)} {video_id}: skipped 2024 no-tag row")
            report_rows.append({"video_id": video_id, "row_index": row_index, "status": "skipped", "reason": "2024 no ArUco tags"})
            continue

        safe_video_id = _safe_path_component(video_id)
        item_dir = output_root / f"{safe_video_id}_tracking"
        frames_dir = item_dir / "optimized_frames"
        annotated_dir = item_dir / "annotated_frames"
        optimization_root = item_dir / "optimization"
        detection_csv_path = item_dir / f"{safe_video_id}_detections.csv"
        selected_params_path = item_dir / "selected_tracking_params.json"
        summary_path = item_dir / "video_range_tracking_summary.json"
        item_warnings: list[str] = []
        item_errors: list[str] = []
        item_start = time.perf_counter()

        print(f"\n[optimize-video-ranges] {row_index}/{len(manifest_rows)} {video_id}: preparing")
        try:
            ranges = _parse_video_range_text(row)
            frame_indices = _frame_indices_from_ranges(ranges)
            video_path = _resolve_manifest_video(source_root, video_id)
            if video_path is None:
                raise FileNotFoundError(f"Could not find video for {video_id} under {source_root}")

            if args.replace_optimization_runs:
                if args.dry_run:
                    print(
                        f"[optimize-video-ranges] {video_id}: dry run; "
                        f"would remove previous optimization runs under {optimization_root}"
                    )
                elif optimization_root.exists():
                    shutil.rmtree(optimization_root)
                    print(
                        f"[optimize-video-ranges] {video_id}: removed previous optimization runs "
                        f"under {optimization_root}"
                    )

            copied_video_path = output_root / video_path.name
            if args.dry_run:
                print(f"[optimize-video-ranges] {video_id}: dry run; would copy {video_path} -> {copied_video_path}")
            elif args.force or not copied_video_path.exists():
                shutil.copy2(video_path, copied_video_path)
            else:
                source_stat = video_path.stat()
                dest_stat = copied_video_path.stat()
                if source_stat.st_size != dest_stat.st_size:
                    shutil.copy2(video_path, copied_video_path)

            tag_list_path = _resolve_tag_list_for_video(video_id, tag_list_root)
            allowed_tag_ids = set(global_allowed_ids)
            if tag_list_path is not None:
                allowed_tag_ids.update(load_tag_ids(tag_list_path))
            elif video_id.startswith("bumblebox-") and "2026" in video_id:
                item_warnings.append("No tag list found for 2026 BumbleBox video.")
            excluded_tags_exempt = any(
                video_id.startswith(prefix) for prefix in excluded_id_exempt_prefixes
            )
            excluded_tag_ids = set() if excluded_tags_exempt else set(global_excluded_ids)
            if excluded_tag_ids:
                allowed_tag_ids.difference_update(excluded_tag_ids)
            tag_filter_ids = None if args.no_tag_list_filter else (allowed_tag_ids or None)
            if args.no_tag_list_filter and allowed_tag_ids:
                item_warnings.append("Tag-list filtering disabled by --no-tag-list-filter.")

            review_bounds, bounds_sweep_overrides = _bounds_for_video(video_id, bounds_by_video)
            sweep_overrides = dict(sweep_overrides_base)
            if "minMarkerPerimeterRate" not in sweep_overrides and "minMarkerPerimeterRate" in bounds_sweep_overrides:
                sweep_overrides["minMarkerPerimeterRate"] = bounds_sweep_overrides["minMarkerPerimeterRate"]
            if "maxMarkerPerimeterRate" not in sweep_overrides and "maxMarkerPerimeterRate" in bounds_sweep_overrides:
                sweep_overrides["maxMarkerPerimeterRate"] = bounds_sweep_overrides["maxMarkerPerimeterRate"]

            if args.dry_run:
                extracted_paths: list[Path] = [frames_dir / f"frame_{idx:06d}.png" for idx in frame_indices]
            else:
                extracted_paths, extraction_warnings = _copy_or_extract_video_range_frames(
                    video_id=video_id,
                    video_path=video_path,
                    source_root=source_root,
                    frame_indices=frame_indices,
                    output_dir=frames_dir,
                    force=args.force,
                )
                item_warnings.extend(extraction_warnings)

            dictionaries = [
                normalize_dictionary_name(dictionary)
                for dictionary in _dictionary_candidates_for_video(video_id, args.dictionary, args.dictionary_candidates)
            ]
            print(
                f"[optimize-video-ranges] {video_id}: frames={len(extracted_paths)} "
                f"range={row.get('frame_ranges') or ranges}; dictionaries={','.join(dictionaries)}; "
                f"tag_list={tag_list_path or 'none'}"
            )
            if args.no_tag_list_filter:
                print(f"[optimize-video-ranges] {video_id}: tag-list filter disabled")
            if excluded_tag_ids:
                print(f"[optimize-video-ranges] {video_id}: excluding tag IDs {sorted(excluded_tag_ids)}")
            elif excluded_tags_exempt and global_excluded_ids:
                print(
                    f"[optimize-video-ranges] {video_id}: tag ID exclusions skipped "
                    f"because video ID matches {','.join(excluded_id_exempt_prefixes)}"
                )
            if review_bounds:
                print(f"[optimize-video-ranges] {video_id}: measured perimeter bounds {review_bounds[0]:.6f}-{review_bounds[1]:.6f}")
            elif args.tag_bounds_json:
                print(
                    f"[optimize-video-ranges] {video_id}: no saved smallest/largest tag bounds found; "
                    "using command sweep/profile perimeter settings"
                )

            best_item: dict[str, Any] | None = None
            if not args.dry_run:
                for dictionary in dictionaries:
                    dict_dir = optimization_root / dictionary
                    print(f"[optimize-video-ranges] {video_id}: optimizing dictionary {dictionary}")
                    last_emit = {"time": 0.0}

                    def filter_breakdown(stats: dict[str, object]) -> str:
                        filtered = float(stats.get("mean_filtered") or 0.0)
                        small = float(stats.get("mean_filtered_too_small") or 0.0)
                        large = float(stats.get("mean_filtered_too_large") or 0.0)
                        excluded = float(stats.get("mean_filtered_excluded_tag") or 0.0)
                        outside = float(stats.get("mean_filtered_outside_tag_list") or 0.0)
                        other = float(stats.get("mean_filtered_other") or 0.0)
                        return (
                            f"filtered={filtered:.2f} "
                            f"(small={small:.2f}, large={large:.2f}, excluded={excluded:.2f}, "
                            f"outside-list={outside:.2f}, other={other:.2f})"
                        )

                    def progress_callback(done: int, total: int, top_candidates: list[dict[str, object]], latest: dict[str, object]) -> None:
                        now = time.monotonic()
                        if done not in {1, total} and now - last_emit["time"] < 5.0:
                            return
                        last_emit["time"] = now
                        high = latest.get("highest_detection_candidate") or {}
                        print(
                            f"[optimize-video-ranges] {video_id} {dictionary}: evaluated {done}/{total}; "
                            f"latest detect={float(latest.get('mean_detected') or 0.0):.2f}, "
                            f"decoded={float(latest.get('mean_decoded') or 0.0):.2f}, "
                            f"{filter_breakdown(latest)}; "
                            f"highest detect={float(high.get('mean_detected') or 0.0):.2f}, "
                            f"{filter_breakdown(high)}"
                        )

                    result = optimize_tracking(
                        input_path=frames_dir,
                        profile=args.profile,
                        sample_frames=min(len(extracted_paths), max(1, int(args.sample_frames or len(extracted_paths)))),
                        dictionary_name=dictionary,
                        tag_size_mm=float(args.tag_size_mm),
                        sweep_overrides=sweep_overrides or None,
                        max_parameter_combinations=args.max_combinations,
                        execution_target=args.execution_target,
                        workers=args.workers,
                        expected_tags=args.expected_tags,
                        output_dir=dict_dir,
                        write_preview=False,
                        write_candidate_review=not args.no_candidate_review,
                        top_k=20,
                        review_perimeter_bounds=review_bounds,
                        valid_tag_ids=tag_filter_ids,
                        excluded_tag_ids=excluded_tag_ids or None,
                        progress_callback=None if args.no_live_progress else progress_callback,
                    )
                    params, selected_label, mean_detected = _selected_params_from_optimization(result)
                    candidate = {
                        "dictionary": dictionary,
                        "params": params,
                        "selected_label": selected_label,
                        "mean_detected": mean_detected,
                        "summary_json_path": getattr(result, "summary_json_path", None),
                        "candidates_csv_path": getattr(result, "candidates_csv_path", None),
                    }
                    if best_item is None or float(candidate["mean_detected"]) > float(best_item["mean_detected"]):
                        best_item = candidate

                if best_item is None:
                    raise RuntimeError("Optimization did not produce a selected parameter set.")
                selected_payload = {
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                    "video_id": video_id,
                    "frame_ranges": row.get("frame_ranges") or "",
                    "dictionary": best_item["dictionary"],
                    "selected_label": best_item["selected_label"],
                    "params": best_item["params"],
                    "mean_detected": best_item["mean_detected"],
                    "tag_list_path": str(tag_list_path) if tag_list_path else None,
                    "tag_list_filter_enabled": tag_filter_ids is not None,
                    "allowed_tag_count": len(allowed_tag_ids) if allowed_tag_ids else None,
                    "excluded_tag_ids_filter": sorted(excluded_tag_ids) if excluded_tag_ids else None,
                    "candidate_review_enabled": not args.no_candidate_review,
                    "review_perimeter_bounds": list(review_bounds) if review_bounds else None,
                    "source_optimization_summary": best_item["summary_json_path"],
                    "source_candidate_scores": best_item["candidates_csv_path"],
                }
                selected_params_path.write_text(json.dumps(selected_payload, indent=2, sort_keys=True))

                detection_rows: list[dict[str, object]] = []
                annotated_paths: list[Path] = []
                final_audit_totals = {
                    "decoded_count": 0,
                    "valid_detection_count": 0,
                    "filtered_count": 0,
                    "filtered_too_small_count": 0,
                    "filtered_too_large_count": 0,
                    "filtered_excluded_tag_count": 0,
                    "filtered_outside_tag_list_count": 0,
                    "filtered_other_count": 0,
                }
                final_filtered_ids: dict[str, set[int]] = {
                    "too_small": set(),
                    "too_large": set(),
                    "excluded_tag": set(),
                    "outside_tag_list": set(),
                    "other": set(),
                }
                for frame_path in extracted_paths:
                    frame_index = _frame_index_from_image_path(frame_path)
                    annotated_path = annotated_dir / f"{frame_path.stem}_annotated.png"
                    rows_for_frame, audit = detect_markers_in_image(
                        frame_path,
                        params=best_item["params"],
                        dictionary_name=best_item["dictionary"],
                        selected_label=best_item["selected_label"],
                        selected_params_path=selected_params_path,
                        relative_image_path=f"{video_id}/{frame_path.name}",
                        image_index=frame_index,
                        valid_tag_ids=tag_filter_ids,
                        excluded_tag_ids=excluded_tag_ids or None,
                        perimeter_filter_bounds=review_bounds,
                        annotated_output_path=annotated_path,
                    )
                    for key in final_audit_totals:
                        try:
                            final_audit_totals[key] += int(audit.get(key) or 0)
                        except (TypeError, ValueError):
                            pass
                    reason_ids = audit.get("filtered_reason_ids")
                    if isinstance(reason_ids, dict):
                        for reason, ids in reason_ids.items():
                            if reason not in final_filtered_ids or not isinstance(ids, list):
                                continue
                            for tag_id in ids:
                                try:
                                    final_filtered_ids[reason].add(int(tag_id))
                                except (TypeError, ValueError):
                                    continue
                    annotated_paths.append(annotated_path)
                    for detection_row in rows_for_frame:
                        detection_row = dict(detection_row)
                        detection_row["video_id"] = video_id
                        detection_row["source_video_path"] = str(video_path)
                        detection_row["frame"] = frame_index if frame_index is not None else ""
                        detection_rows.append(detection_row)
                _write_detection_rows_csv(detection_csv_path, detection_rows)
                combined_detection_rows.extend(detection_rows)

                final_detection_audit = {
                    **final_audit_totals,
                    "filtered_reason_ids": {
                        reason: sorted(ids) for reason, ids in final_filtered_ids.items()
                    },
                }
                if not detection_rows and final_audit_totals["decoded_count"] > 0:
                    reason_text = (
                        f"decoded {final_audit_totals['decoded_count']} tags but kept 0 after filters "
                        f"(too_small={final_audit_totals['filtered_too_small_count']}, "
                        f"too_large={final_audit_totals['filtered_too_large_count']}, "
                        f"excluded={final_audit_totals['filtered_excluded_tag_count']}, "
                        f"outside-list={final_audit_totals['filtered_outside_tag_list_count']}, "
                        f"other={final_audit_totals['filtered_other_count']})"
                    )
                    excluded_ids = final_detection_audit["filtered_reason_ids"].get("excluded_tag", [])
                    outside_ids = final_detection_audit["filtered_reason_ids"].get("outside_tag_list", [])
                    if excluded_ids:
                        reason_text += f"; excluded IDs={excluded_ids[:25]}"
                    if outside_ids:
                        reason_text += f"; outside-list IDs={outside_ids[:25]}"
                    item_warnings.append(reason_text)
                    print(f"[optimize-video-ranges] {video_id}: WARNING {reason_text}")

                annotated_video_path = None
                if args.write_annotated_video:
                    annotated_video_path = _write_annotated_frame_video(
                        annotated_paths,
                        item_dir / f"{safe_video_id}_optimized_frames_annotated.mp4",
                        fps=float(args.annotated_video_fps),
                    )

                summary = {
                    "video_id": video_id,
                    "status": "ok",
                    "source_video_path": str(video_path),
                    "copied_video_path": str(copied_video_path),
                    "frame_ranges": row.get("frame_ranges") or "",
                    "frame_indices": frame_indices,
                    "frames_dir": str(frames_dir),
                    "annotated_frames_dir": str(annotated_dir),
                    "annotated_video_path": str(annotated_video_path) if annotated_video_path else None,
                    "selected_params_path": str(selected_params_path),
                    "detection_csv_path": str(detection_csv_path),
                    "dictionary": best_item["dictionary"],
                    "tag_list_path": str(tag_list_path) if tag_list_path else None,
                    "tag_list_filter_enabled": tag_filter_ids is not None,
                    "allowed_tag_count": len(allowed_tag_ids) if allowed_tag_ids else None,
                    "excluded_tag_ids_filter": sorted(excluded_tag_ids) if excluded_tag_ids else None,
                    "candidate_review_enabled": not args.no_candidate_review,
                    "detections": len(detection_rows),
                    "detection_audit": final_detection_audit,
                    "warnings": item_warnings,
                    "errors": item_errors,
                    "elapsed_seconds": time.perf_counter() - item_start,
                }
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
                report_rows.append(summary)
                processed += 1
                print(
                    f"[optimize-video-ranges] {video_id}: done; dictionary={best_item['dictionary']}; "
                    f"detections={len(detection_rows)}; annotated PNGs={len(annotated_paths)}"
                )
            else:
                report_rows.append(
                    {
                        "video_id": video_id,
                        "status": "planned",
                        "source_video_path": str(video_path),
                        "copied_video_path": str(copied_video_path),
                        "frame_ranges": row.get("frame_ranges") or "",
                        "frame_indices": frame_indices,
                        "dictionary_candidates": dictionaries,
                        "tag_list_path": str(tag_list_path) if tag_list_path else None,
                        "allowed_tag_count": len(allowed_tag_ids) if allowed_tag_ids else None,
                        "excluded_tag_ids_filter": sorted(excluded_tag_ids) if excluded_tag_ids else None,
                        "candidate_review_enabled": not args.no_candidate_review,
                        "replace_optimization_runs": bool(args.replace_optimization_runs),
                        "warnings": item_warnings,
                    }
                )
                processed += 1
        except Exception as exc:
            failed += 1
            item_errors.append(str(exc))
            print(f"[optimize-video-ranges] {video_id}: failed: {exc}")
            report_rows.append(
                {
                    "video_id": video_id,
                    "row_index": row_index,
                    "status": "failed",
                    "warnings": item_warnings,
                    "errors": item_errors,
                }
            )

    combined_csv_path = output_root / "all_video_range_detections.csv"
    if combined_detection_rows:
        _write_detection_rows_csv(combined_csv_path, combined_detection_rows)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tag_list_root": str(tag_list_root) if tag_list_root else None,
        "tag_bounds_json": str(args.tag_bounds_json) if args.tag_bounds_json else None,
        "candidate_review_enabled": not args.no_candidate_review,
        "replace_optimization_runs": bool(args.replace_optimization_runs),
        "rows_loaded": len(manifest_rows),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "combined_detection_csv_path": str(combined_csv_path) if combined_detection_rows else None,
        "results": report_rows,
    }
    report_path = output_root / "optimize_video_ranges_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[optimize-video-ranges] Finished: processed={processed}, skipped={skipped}, failed={failed}")
    print(f"[optimize-video-ranges] Report: {report_path}")
    if combined_detection_rows:
        print(f"[optimize-video-ranges] Combined detections CSV: {combined_csv_path}")
    return 0 if failed == 0 else 1


def _cmd_optimize_image_folder(args: argparse.Namespace) -> int:
    from .tracking_optimizer import (
        IMAGE_EXTENSIONS,
        IMAGE_DETECTION_CSV_FIELDS,
        detect_markers_in_image,
        find_supported_image_paths,
        is_generated_optimizer_artifact_path,
        optimize_tracking,
    )

    def safe_path_component(text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
        return safe.strip("_")

    def per_image_output_dir(image_path: Path, input_root: Path, output_root: Path) -> Path:
        try:
            relative = image_path.relative_to(input_root)
        except ValueError:
            relative = Path(image_path.name)

        stem = safe_path_component(relative.stem) or safe_path_component(image_path.stem) or "image"
        return output_root / relative.parent / f"{stem}_tracking_optimization"

    def annotated_image_path(image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")

    def per_image_detection_csv_path(image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_detections.csv")

    def image_signature(image_path: Path) -> dict[str, object]:
        stat = image_path.stat()
        return {
            "path": str(image_path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def marker_matches(marker_path: Path, image_path: Path) -> tuple[bool, Optional[dict]]:
        if not marker_path.exists():
            return False, None
        try:
            payload = json.loads(marker_path.read_text())
        except Exception:
            return False, None
        return payload.get("source_signature") == image_signature(image_path), payload

    def load_saved_selected_params(row: dict[str, object], default_path: Path) -> tuple[dict[str, object], Path]:
        raw_path = str(row.get("selected_params_path") or "").strip()
        candidate_paths = []
        if raw_path:
            candidate_paths.append(Path(raw_path).expanduser())
        candidate_paths.append(default_path)

        for candidate_path in candidate_paths:
            try:
                if not candidate_path.exists():
                    continue
                payload = json.loads(candidate_path.read_text())
                params = payload.get("params") if isinstance(payload, dict) else None
                if isinstance(params, dict) and params:
                    return params, candidate_path
            except Exception:
                continue

        for key in ("top_detect_params_json", "best_score_params_json"):
            raw_params = str(row.get(key) or "").strip()
            if not raw_params:
                continue
            try:
                params = json.loads(raw_params)
            except json.JSONDecodeError:
                continue
            if isinstance(params, dict) and params:
                return params, default_path

        raise RuntimeError(f"Could not load saved selected tracking parameters from {default_path}")

    def candidate_row(prefix: str, candidate: object) -> dict[str, object]:
        params = getattr(candidate, "params", {}) or {}
        return {
            f"{prefix}_score": getattr(candidate, "score", ""),
            f"{prefix}_mean_detected": getattr(candidate, "mean_detected", ""),
            f"{prefix}_mean_decoded": getattr(candidate, "mean_decoded", ""),
            f"{prefix}_mean_filtered": getattr(candidate, "mean_filtered", ""),
            f"{prefix}_mean_filtered_too_small": getattr(candidate, "mean_filtered_too_small", ""),
            f"{prefix}_mean_filtered_too_large": getattr(candidate, "mean_filtered_too_large", ""),
            f"{prefix}_mean_filtered_outside_tag_list": getattr(
                candidate,
                "mean_filtered_outside_tag_list",
                "",
            ),
            f"{prefix}_mean_filtered_other": getattr(candidate, "mean_filtered_other", ""),
            f"{prefix}_std_detected": getattr(candidate, "std_detected", ""),
            f"{prefix}_mean_rejected": getattr(candidate, "mean_rejected", ""),
            f"{prefix}_stability": getattr(candidate, "stability", ""),
            f"{prefix}_eval_fps": getattr(candidate, "eval_fps", ""),
            f"{prefix}_runtime_seconds": getattr(candidate, "runtime_seconds", ""),
            f"{prefix}_params_json": json.dumps(params, sort_keys=True),
        }

    def shorten_table_text(text: object, width: int) -> str:
        value = str(text)
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return f"{value[: width - 3]}..."

    def selected_metric(metrics: object, key: str, *fallback_keys: str) -> object:
        if isinstance(metrics, dict):
            for candidate_key in (key, f"top_detect_{key}", *fallback_keys):
                value = metrics.get(candidate_key)
                if value not in ("", None):
                    return value
            return ""
        value = getattr(metrics, key, "")
        return value if value not in ("", None) else ""

    def format_table_float(value: object, width: int, precision: int) -> str:
        if value in ("", None):
            return f"{'':>{width}}"
        try:
            return f"{float(value):>{width}.{precision}f}"
        except (TypeError, ValueError):
            return f"{str(value):>{width}}"

    def format_table_int(value: object, width: int) -> str:
        if value in ("", None):
            return f"{'':>{width}}"
        try:
            return f"{int(float(value)):>{width}}"
        except (TypeError, ValueError):
            return f"{str(value):>{width}}"

    def selected_params_table_text(relative: Path, params: dict[str, object], metrics: object) -> str:
        header = (
            f"{'image':<48} "
            f"{'detect':>7} {'decoded':>7} {'filt':>6} "
            f"{'small':>6} {'large':>6} {'outside':>7} "
            f"{'score':>9} {'minPerim':>9} {'maxPerim':>9} "
            f"{'winMin':>6} {'winMax':>6} {'winStep':>7} {'poly':>7} {'const':>5}"
        )
        row = (
            f"{shorten_table_text(relative, 48):<48} "
            f"{format_table_float(selected_metric(metrics, 'mean_detected', 'detection_count'), 7, 2)} "
            f"{format_table_float(selected_metric(metrics, 'mean_decoded', 'decoded_count'), 7, 2)} "
            f"{format_table_float(selected_metric(metrics, 'mean_filtered', 'filtered_detection_count'), 6, 2)} "
            f"{format_table_float(selected_metric(metrics, 'mean_filtered_too_small', 'filtered_too_small_count'), 6, 2)} "
            f"{format_table_float(selected_metric(metrics, 'mean_filtered_too_large', 'filtered_too_large_count'), 6, 2)} "
            f"{format_table_float(selected_metric(metrics, 'mean_filtered_outside_tag_list', 'filtered_outside_tag_list_count'), 7, 2)} "
            f"{format_table_float(selected_metric(metrics, 'score'), 9, 4)} "
            f"{format_table_float(params.get('minMarkerPerimeterRate'), 9, 6)} "
            f"{format_table_float(params.get('maxMarkerPerimeterRate'), 9, 6)} "
            f"{format_table_int(params.get('adaptiveThreshWinSizeMin'), 6)} "
            f"{format_table_int(params.get('adaptiveThreshWinSizeMax'), 6)} "
            f"{format_table_int(params.get('adaptiveThreshWinSizeStep'), 7)} "
            f"{format_table_float(params.get('polygonalApproxAccuracyRate'), 7, 4)} "
            f"{format_table_int(params.get('adaptiveThreshConstant'), 5)}"
        )
        return "\n".join(
            [
                "[optimize-images] selected top-detection parameters:",
                header,
                "-" * len(header),
                row,
            ]
        )

    def candidate_params(metrics: object) -> dict[str, object]:
        if not isinstance(metrics, dict):
            params = getattr(metrics, "params", {})
            return params if isinstance(params, dict) else {}

        params = metrics.get("params")
        if isinstance(params, dict):
            return params

        for key in ("params_json", "top_detect_params_json", "best_score_params_json"):
            raw = str(metrics.get(key) or "").strip()
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
        return {}

    def parameter_sort_value(params: dict[str, object], key: str) -> tuple[int, object]:
        value = params.get(key)
        if value in ("", None):
            return (1, "")
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (0, str(value))

    def candidate_rows_from_csv(path: str | Path) -> list[dict[str, object]]:
        candidate_path = Path(path)
        if not candidate_path.exists():
            return []

        rows: list[dict[str, object]] = []
        with candidate_path.open(newline="") as f:
            for row in csv.DictReader(f):
                params = candidate_params(row)
                row["params"] = params
                rows.append(row)
        return rows

    def candidate_mean_detected(candidate: dict[str, object]) -> Optional[float]:
        try:
            return float(candidate.get("mean_detected") or 0.0)
        except (TypeError, ValueError):
            return None

    def candidate_score_rank(candidate: dict[str, object]) -> int:
        try:
            return int(float(candidate.get("rank") or 999999))
        except (TypeError, ValueError):
            return 999999

    def candidate_score_value(candidate: dict[str, object]) -> float:
        try:
            return float(candidate.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def sorted_by_parameter_values(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        sort_keys = (
            "minMarkerPerimeterRate",
            "maxMarkerPerimeterRate",
            "adaptiveThreshWinSizeMin",
            "adaptiveThreshWinSizeMax",
            "adaptiveThreshWinSizeStep",
            "polygonalApproxAccuracyRate",
            "adaptiveThreshConstant",
        )
        return sorted(
            candidates,
            key=lambda row: (
                tuple(parameter_sort_value(candidate_params(row), key) for key in sort_keys),
                candidate_score_value(row),
            )
        )

    def best_detection_ties_from_csv(path: str | Path, best_detect: object) -> list[dict[str, object]]:
        try:
            target_detect = float(best_detect)
        except (TypeError, ValueError):
            return []

        ties: list[dict[str, object]] = []
        for row in candidate_rows_from_csv(path):
            mean_detected = candidate_mean_detected(row)
            if mean_detected is None:
                continue
            if abs(mean_detected - target_detect) <= 1e-9:
                ties.append(row)
        return sorted_by_parameter_values(ties)

    def near_miss_detection_groups_from_csv(
        path: str | Path,
        best_detect: object,
        *,
        levels: int = 2,
        limit_per_level: int = 15,
    ) -> list[tuple[float, list[dict[str, object]], int]]:
        try:
            target_detect = float(best_detect)
        except (TypeError, ValueError):
            return []

        groups: dict[float, list[dict[str, object]]] = {}
        for row in candidate_rows_from_csv(path):
            mean_detected = candidate_mean_detected(row)
            if mean_detected is None or mean_detected >= target_detect - 1e-9:
                continue
            groups.setdefault(mean_detected, []).append(row)

        out: list[tuple[float, list[dict[str, object]], int]] = []
        for detect_value in sorted(groups.keys(), reverse=True)[: max(0, int(levels))]:
            candidates = sorted(
                groups[detect_value],
                key=lambda row: (candidate_score_rank(row), -candidate_score_value(row)),
            )
            out.append((detect_value, candidates[: max(1, int(limit_per_level))], len(candidates)))
        return out

    def best_detection_ties_table_text(
        relative: Path,
        candidates: list[dict[str, object]],
        best_detect: object,
    ) -> str:
        if not candidates:
            return f"[optimize-images] no best-detection tie rows found for {relative}"

        header = (
            f"{'tie':>4} {'score#':>6} "
            f"{'detect':>7} {'decoded':>7} {'filt':>6} "
            f"{'small':>6} {'large':>6} {'outside':>7} "
            f"{'score':>9} {'minPerim':>9} {'maxPerim':>9} "
            f"{'winMin':>6} {'winMax':>6} {'winStep':>7} {'poly':>7} {'const':>5}"
        )
        lines = [
            (
                f"[optimize-images] parameter sets tied for best detection "
                f"for {relative} (detect={format_table_float(best_detect, 0, 2).strip()}, "
                f"n={len(candidates)}):"
            ),
            header,
            "-" * len(header),
        ]

        for tie_index, candidate in enumerate(candidates, start=1):
            params = candidate_params(candidate)
            lines.append(
                " ".join(
                    [
                        f"{tie_index:>4}",
                        format_table_int(candidate.get("rank"), 6),
                        format_table_float(selected_metric(candidate, "mean_detected"), 7, 2),
                        format_table_float(selected_metric(candidate, "mean_decoded"), 7, 2),
                        format_table_float(selected_metric(candidate, "mean_filtered"), 6, 2),
                        format_table_float(selected_metric(candidate, "mean_filtered_too_small"), 6, 2),
                        format_table_float(selected_metric(candidate, "mean_filtered_too_large"), 6, 2),
                        format_table_float(
                            selected_metric(candidate, "mean_filtered_outside_tag_list"),
                            7,
                            2,
                        ),
                        format_table_float(selected_metric(candidate, "score"), 9, 4),
                        format_table_float(params.get("minMarkerPerimeterRate"), 9, 6),
                        format_table_float(params.get("maxMarkerPerimeterRate"), 9, 6),
                        format_table_int(params.get("adaptiveThreshWinSizeMin"), 6),
                        format_table_int(params.get("adaptiveThreshWinSizeMax"), 6),
                        format_table_int(params.get("adaptiveThreshWinSizeStep"), 7),
                        format_table_float(params.get("polygonalApproxAccuracyRate"), 7, 4),
                        format_table_int(params.get("adaptiveThreshConstant"), 5),
                    ]
                )
            )
        return "\n".join(lines)

    def near_miss_detection_groups_table_text(
        relative: Path,
        groups: list[tuple[float, list[dict[str, object]], int]],
        *,
        limit_per_level: int = 15,
    ) -> str:
        if not groups:
            return ""

        header = (
            f"{'row':>4} {'score#':>6} "
            f"{'detect':>7} {'decoded':>7} {'filt':>6} "
            f"{'small':>6} {'large':>6} {'outside':>7} "
            f"{'score':>9} {'minPerim':>9} {'maxPerim':>9} "
            f"{'winMin':>6} {'winMax':>6} {'winStep':>7} {'poly':>7} {'const':>5}"
        )
        lines = [
            (
                f"[optimize-images] near-miss parameter sets for {relative} "
                f"(top {limit_per_level} by score rank for each next-lower detection level):"
            )
        ]
        for detect_value, candidates, total_count in groups:
            lines.extend(
                [
                    "",
                    (
                        f"detect={format_table_float(detect_value, 0, 2).strip()} "
                        f"(showing {len(candidates)} of {total_count})"
                    ),
                    header,
                    "-" * len(header),
                ]
            )
            for row_index, candidate in enumerate(candidates, start=1):
                params = candidate_params(candidate)
                lines.append(
                    " ".join(
                        [
                            f"{row_index:>4}",
                            format_table_int(candidate.get("rank"), 6),
                            format_table_float(selected_metric(candidate, "mean_detected"), 7, 2),
                            format_table_float(selected_metric(candidate, "mean_decoded"), 7, 2),
                            format_table_float(selected_metric(candidate, "mean_filtered"), 6, 2),
                            format_table_float(selected_metric(candidate, "mean_filtered_too_small"), 6, 2),
                            format_table_float(selected_metric(candidate, "mean_filtered_too_large"), 6, 2),
                            format_table_float(
                                selected_metric(candidate, "mean_filtered_outside_tag_list"),
                                7,
                                2,
                            ),
                            format_table_float(selected_metric(candidate, "score"), 9, 4),
                            format_table_float(params.get("minMarkerPerimeterRate"), 9, 6),
                            format_table_float(params.get("maxMarkerPerimeterRate"), 9, 6),
                            format_table_int(params.get("adaptiveThreshWinSizeMin"), 6),
                            format_table_int(params.get("adaptiveThreshWinSizeMax"), 6),
                            format_table_int(params.get("adaptiveThreshWinSizeStep"), 7),
                            format_table_float(params.get("polygonalApproxAccuracyRate"), 7, 4),
                            format_table_int(params.get("adaptiveThreshConstant"), 5),
                        ]
                    )
                )
        return "\n".join(lines)

    fieldnames = [
        "image_index",
        "image_path",
        "relative_image_path",
        "status",
        "error",
        "runtime_seconds",
        "output_dir",
        "selected_params_path",
        "summary_json_path",
        "candidate_scores_path",
        "detection_csv_path",
        "annotated_image_path",
        "detection_count",
        "decoded_count",
        "filtered_detection_count",
        "filtered_too_small_count",
        "filtered_too_large_count",
        "filtered_outside_tag_list_count",
        "filtered_other_count",
        "rejected_candidate_count",
        "dictionary",
        "profile",
        "execution_target",
        "workers",
        "parameter_combinations_total",
        "combinations_evaluated",
        "sample_frames_used",
        "best_score",
        "best_mean_detected",
        "top_detect_score",
        "top_detect_mean_detected",
        "top_detect_mean_decoded",
        "top_detect_mean_filtered",
        "top_detect_mean_filtered_too_small",
        "top_detect_mean_filtered_too_large",
        "top_detect_mean_filtered_outside_tag_list",
        "top_detect_mean_filtered_other",
        "top_detect_std_detected",
        "top_detect_mean_rejected",
        "top_detect_stability",
        "top_detect_eval_fps",
        "top_detect_runtime_seconds",
        "top_detect_params_json",
        "best_score_params_json",
    ]

    def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def write_detection_csv(path: Path, detection_rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=IMAGE_DETECTION_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(detection_rows)

    def read_detection_csv(path: Path) -> list[dict[str, object]]:
        if not path.exists() or not path.is_file():
            return []
        with path.open(newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    def filtered_reason_summary_text(audit: dict[str, object]) -> str:
        reason_ids = audit.get("filtered_reason_ids") or {}
        if not isinstance(reason_ids, dict):
            reason_ids = {}

        def part(label: str, count_key: str, ids_key: str) -> str:
            count = int(audit.get(count_key) or 0)
            ids = reason_ids.get(ids_key) or []
            if ids:
                return f"{label}={count} ids={list(ids)}"
            return f"{label}={count}"

        pieces = [
            part("too_small", "filtered_too_small_count", "too_small"),
            part("too_large", "filtered_too_large_count", "too_large"),
            part("outside_tag_list", "filtered_outside_tag_list_count", "outside_tag_list"),
        ]
        other_count = int(audit.get("filtered_other_count") or 0)
        if other_count:
            pieces.append(part("other", "filtered_other_count", "other"))
        return "; ".join(pieces)

    def candidate_filtered_reason_summary_text(candidate: dict[str, object]) -> str:
        def mean_part(label: str, key: str) -> str:
            try:
                value = float(candidate.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            return f"{label}={value:.2f}"

        pieces = [
            mean_part("small", "mean_filtered_too_small"),
            mean_part("large", "mean_filtered_too_large"),
            mean_part("outside", "mean_filtered_outside_tag_list"),
        ]
        try:
            other = float(candidate.get("mean_filtered_other") or 0.0)
        except (TypeError, ValueError):
            other = 0.0
        if other:
            pieces.append(f"other={other:.2f}")
        return ", ".join(pieces)

    def row_int(row: dict[str, object], key: str) -> int:
        try:
            return int(float(row.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
        sweep_overrides = _optimizer_sweep_overrides_from_args(args)
        allowed_tag_ids = _optimizer_allowed_tag_ids_from_args(args, config)
    except Exception as exc:
        print(f"Image-folder optimization setup failed: {exc}")
        return 1

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"Input image folder does not exist or is not a directory: {input_root}")
        return 1

    if args.no_recursive:
        images = sorted(
            path
            for path in input_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.name.startswith("._")
            and path.name not in {".DS_Store", "Thumbs.db"}
            and not is_generated_optimizer_artifact_path(path, root=input_root)
        )
    else:
        images = find_supported_image_paths(input_root)

    if args.limit is not None:
        if int(args.limit) <= 0:
            print("--limit must be >= 1 when provided.")
            return 1
        images = images[: int(args.limit)]

    if not images:
        print(f"No supported PNG/image files found in: {input_root}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_root / "per_image_optimization_summary.csv"
    combined_detection_csv_path = output_root / "per_image_tag_detections.csv"
    report_json_path = output_root / "per_image_optimization_report.json"
    started_at = datetime.now().isoformat(timespec="seconds")
    run_started = time.perf_counter()
    rows: list[dict[str, object]] = []
    all_detection_rows: list[dict[str, object]] = []
    completed = 0
    skipped = 0
    backfilled_annotations = 0
    failed = 0

    print(f"[optimize-images] Input: {input_root}")
    print(f"[optimize-images] Images found: {len(images)}")
    print(f"[optimize-images] Output root: {output_root}")
    print(f"[optimize-images] Resume completed images: {'no' if args.force else 'yes'}")
    print(f"[optimize-images] Allowed tag IDs: {len(allowed_tag_ids) if allowed_tag_ids else 'none'}")

    for image_index, image_path in enumerate(images, start=1):
        if image_index > 1:
            print()

        relative = image_path.relative_to(input_root)
        image_output_dir = per_image_output_dir(image_path, input_root, output_root)
        marker_path = image_output_dir / "image_optimization_complete.json"
        selected_params_path = image_output_dir / "selected_tracking_params.json"
        detection_csv_path = per_image_detection_csv_path(image_path)
        annotated_path = annotated_image_path(image_path)

        matches, marker_payload = marker_matches(marker_path, image_path)
        if matches and not args.force and detection_csv_path.exists():
            marker_row = dict((marker_payload or {}).get("summary_row") or {})
            if not annotated_path.exists():
                try:
                    saved_params, saved_params_path = load_saved_selected_params(
                        marker_row,
                        selected_params_path,
                    )
                    saved_dictionary = str(marker_row.get("dictionary") or args.dictionary)
                    detection_rows, detection_audit = detect_markers_in_image(
                        image_path,
                        params=saved_params,
                        dictionary_name=saved_dictionary,
                        selected_label="top_mean_detection",
                        selected_params_path=saved_params_path,
                        relative_image_path=str(relative),
                        image_index=image_index,
                        valid_tag_ids=allowed_tag_ids or None,
                        annotated_output_path=annotated_path,
                    )
                    write_detection_csv(detection_csv_path, detection_rows)
                    all_detection_rows.extend(detection_rows)
                    marker_row.update(
                        {
                            "status": "annotated_backfilled",
                            "error": "",
                            "image_index": image_index,
                            "image_path": str(image_path),
                            "relative_image_path": str(relative),
                            "output_dir": str(image_output_dir),
                            "selected_params_path": str(saved_params_path),
                            "detection_csv_path": str(detection_csv_path),
                            "annotated_image_path": detection_audit["annotated_image_path"],
                            "detection_count": detection_audit["valid_detection_count"],
                            "decoded_count": detection_audit["decoded_count"],
                            "filtered_detection_count": detection_audit["filtered_count"],
                            "filtered_too_small_count": detection_audit["filtered_too_small_count"],
                            "filtered_too_large_count": detection_audit["filtered_too_large_count"],
                            "filtered_outside_tag_list_count": detection_audit[
                                "filtered_outside_tag_list_count"
                            ],
                            "filtered_other_count": detection_audit["filtered_other_count"],
                            "rejected_candidate_count": detection_audit["rejected_count"],
                            "dictionary": detection_audit["dictionary"],
                        }
                    )
                    if marker_payload is None:
                        marker_payload = {}
                    marker_payload["schema_version"] = marker_payload.get("schema_version", 1)
                    marker_payload["source_signature"] = image_signature(image_path)
                    marker_payload["summary_row"] = marker_row
                    marker_path.write_text(json.dumps(marker_payload, indent=2))
                    rows.append(marker_row)
                    skipped += 1
                    backfilled_annotations += 1
                    print(
                        f"[optimize-images] {image_index}/{len(images)} {relative}: "
                        "backfilled annotated image from existing result"
                    )
                    print()
                    print(selected_params_table_text(relative, saved_params, marker_row))
                    print()
                    candidate_scores_path = marker_row.get("candidate_scores_path") or ""
                    best_detect_for_tables = (
                        marker_row.get("top_detect_mean_detected")
                        or marker_row.get("detection_count")
                    )
                    tie_rows = best_detection_ties_from_csv(
                        candidate_scores_path,
                        best_detect_for_tables,
                    )
                    if tie_rows:
                        print(
                            best_detection_ties_table_text(
                                relative,
                                tie_rows,
                                best_detect_for_tables,
                            )
                        )
                        print()
                    near_miss_groups = near_miss_detection_groups_from_csv(
                        candidate_scores_path,
                        best_detect_for_tables,
                    )
                    near_miss_text = near_miss_detection_groups_table_text(relative, near_miss_groups)
                    if near_miss_text:
                        print(near_miss_text)
                        print()
                    write_summary_csv(summary_csv_path, rows)
                    write_detection_csv(combined_detection_csv_path, all_detection_rows)
                    continue
                except Exception as exc:
                    failed += 1
                    marker_row.update(
                        {
                            "status": "annotation_backfill_failed",
                            "error": str(exc),
                            "image_index": image_index,
                            "image_path": str(image_path),
                            "relative_image_path": str(relative),
                            "output_dir": str(image_output_dir),
                            "detection_csv_path": str(detection_csv_path),
                            "annotated_image_path": str(annotated_path),
                        }
                    )
                    rows.append(marker_row)
                    print(
                        f"[optimize-images] {image_index}/{len(images)} {relative}: "
                        f"failed to backfill annotated image: {exc}"
                    )
                    write_summary_csv(summary_csv_path, rows)
                    write_detection_csv(combined_detection_csv_path, all_detection_rows)
                    continue

            marker_row["status"] = "skipped_completed"
            marker_row["image_index"] = image_index
            marker_row["detection_csv_path"] = str(detection_csv_path)
            marker_row["annotated_image_path"] = marker_row.get("annotated_image_path") or str(annotated_path)
            rows.append(marker_row)
            all_detection_rows.extend(read_detection_csv(detection_csv_path))
            skipped += 1
            print(f"[optimize-images] {image_index}/{len(images)} {relative}: skipped existing result")
            write_summary_csv(summary_csv_path, rows)
            write_detection_csv(combined_detection_csv_path, all_detection_rows)
            continue

        print(f"[optimize-images] {image_index}/{len(images)} {relative}: optimizing")
        image_started = time.perf_counter()
        last_progress = {"time": 0.0}

        def progress_callback(done: int, total: int, top_candidates: list[dict], latest: dict) -> None:
            now = time.perf_counter()
            if done not in {1, total} and now - last_progress["time"] < 5.0:
                return
            last_progress["time"] = now
            high = latest.get("highest_detection_candidate") or {}
            high_detect = float(high.get("mean_detected") or 0.0)
            latest_detect = float(latest.get("mean_detected") or 0.0)
            latest_decoded = float(latest.get("mean_decoded") or 0.0)
            latest_filtered = float(latest.get("mean_filtered") or 0.0)
            print(
                f"[optimize-images] {image_index}/{len(images)} {relative}: "
                f"evaluated {done}/{total}; latest detect={latest_detect:.2f}, "
                f"decoded={latest_decoded:.2f}; "
                f"filtered={latest_filtered:.2f} "
                f"({candidate_filtered_reason_summary_text(latest)}); "
                f"highest detect={high_detect:.2f}"
            )

        try:
            result = optimize_tracking(
                input_path=image_path,
                profile=args.profile,
                sample_frames=1,
                dictionary_name=args.dictionary,
                tag_size_mm=args.tag_size_mm,
                sweep_overrides=sweep_overrides or None,
                max_parameter_combinations=args.max_combinations,
                execution_target=args.execution_target,
                workers=args.workers,
                expected_tags=args.expected_tags,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_improvement=args.early_stop_min_improvement,
                output_dir=image_output_dir,
                write_preview=False,
                top_k=args.top_k,
                valid_tag_ids=allowed_tag_ids or None,
                progress_callback=progress_callback,
            )
            top_detect = (
                result.top_detection_candidates[0]
                if result.top_detection_candidates
                else result.highest_detection_candidate
            )
            selected_params = dict(getattr(top_detect, "params", {}) or {})
            selected_payload = {
                "selected_at": datetime.now().isoformat(timespec="seconds"),
                "selected_label": "top_mean_detection",
                "source_image": str(image_path),
                "source_optimization_summary": result.summary_json_path,
                "source_candidate_scores": result.candidates_csv_path,
                "params": selected_params,
            }
            selected_params_path.write_text(json.dumps(selected_payload, indent=2))
            detection_rows, detection_audit = detect_markers_in_image(
                image_path,
                params=selected_params,
                dictionary_name=result.dictionary,
                selected_label="top_mean_detection",
                selected_params_path=selected_params_path,
                relative_image_path=str(relative),
                image_index=image_index,
                valid_tag_ids=allowed_tag_ids or None,
                annotated_output_path=annotated_path,
            )
            write_detection_csv(detection_csv_path, detection_rows)
            all_detection_rows.extend(detection_rows)

            runtime_seconds = time.perf_counter() - image_started
            row: dict[str, object] = {
                "image_index": image_index,
                "image_path": str(image_path),
                "relative_image_path": str(relative),
                "status": "completed",
                "error": "",
                "runtime_seconds": f"{runtime_seconds:.3f}",
                "output_dir": str(result.output_dir),
                "selected_params_path": str(selected_params_path),
                "summary_json_path": result.summary_json_path,
                "candidate_scores_path": result.candidates_csv_path,
                "detection_csv_path": str(detection_csv_path),
                "annotated_image_path": detection_audit["annotated_image_path"],
                "detection_count": detection_audit["valid_detection_count"],
                "decoded_count": detection_audit["decoded_count"],
                "filtered_detection_count": detection_audit["filtered_count"],
                "filtered_too_small_count": detection_audit["filtered_too_small_count"],
                "filtered_too_large_count": detection_audit["filtered_too_large_count"],
                "filtered_outside_tag_list_count": detection_audit["filtered_outside_tag_list_count"],
                "filtered_other_count": detection_audit["filtered_other_count"],
                "rejected_candidate_count": detection_audit["rejected_count"],
                "dictionary": result.dictionary,
                "profile": result.profile,
                "execution_target": result.execution_target,
                "workers": result.workers,
                "parameter_combinations_total": result.parameter_combinations_total,
                "combinations_evaluated": result.combinations_evaluated,
                "sample_frames_used": result.sample_frames_used,
                "best_score": result.best_score,
                "best_mean_detected": result.best_mean_detected,
                "best_score_params_json": json.dumps(result.best_params, sort_keys=True),
            }
            row.update(candidate_row("top_detect", top_detect))
            image_output_dir.mkdir(parents=True, exist_ok=True)
            marker_payload = {
                "schema_version": 1,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "source_signature": image_signature(image_path),
                "summary_row": row,
            }
            marker_path.write_text(json.dumps(marker_payload, indent=2))
            rows.append(row)
            completed += 1
            print(
                f"[optimize-images] {image_index}/{len(images)} {relative}: "
                f"done in {runtime_seconds:.1f}s; top_detect={row['top_detect_mean_detected']}; "
                f"detections={row['detection_count']}; decoded={row['decoded_count']}; "
                f"filtered={row['filtered_detection_count']}"
            )
            print()
            print(selected_params_table_text(relative, selected_params, top_detect))
            print()
            tie_rows = best_detection_ties_from_csv(
                result.candidates_csv_path,
                row["top_detect_mean_detected"],
            )
            print(best_detection_ties_table_text(relative, tie_rows, row["top_detect_mean_detected"]))
            print()
            near_miss_groups = near_miss_detection_groups_from_csv(
                result.candidates_csv_path,
                row["top_detect_mean_detected"],
            )
            near_miss_text = near_miss_detection_groups_table_text(relative, near_miss_groups)
            if near_miss_text:
                print(near_miss_text)
                print()
            print(
                f"[optimize-images] {image_index}/{len(images)} {relative}: "
                f"filtered decoded tags: {filtered_reason_summary_text(detection_audit)}"
            )
        except KeyboardInterrupt:
            print("\n[optimize-images] Interrupted by user; writing partial summary.")
            write_summary_csv(summary_csv_path, rows)
            write_detection_csv(combined_detection_csv_path, all_detection_rows)
            return 130
        except Exception as exc:
            runtime_seconds = time.perf_counter() - image_started
            failed += 1
            row = {
                "image_index": image_index,
                "image_path": str(image_path),
                "relative_image_path": str(relative),
                "status": "failed",
                "error": str(exc),
                "runtime_seconds": f"{runtime_seconds:.3f}",
                "output_dir": str(image_output_dir),
                "detection_csv_path": str(detection_csv_path),
            }
            rows.append(row)
            print(f"[optimize-images] {image_index}/{len(images)} {relative}: failed: {exc}")

        write_summary_csv(summary_csv_path, rows)
        write_detection_csv(combined_detection_csv_path, all_detection_rows)

    finished_at = datetime.now().isoformat(timespec="seconds")
    elapsed = time.perf_counter() - run_started
    write_detection_csv(combined_detection_csv_path, all_detection_rows)
    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "input": str(input_root),
        "output_root": str(output_root),
        "images_found": len(images),
        "completed": completed,
        "skipped": skipped,
        "backfilled_annotations": backfilled_annotations,
        "failed": failed,
        "summary_csv": str(summary_csv_path),
        "combined_detection_csv": str(combined_detection_csv_path),
        "dictionary": args.dictionary,
        "profile": args.profile,
        "execution_target": args.execution_target,
        "max_combinations": args.max_combinations,
        "allowed_tag_ids": sorted(allowed_tag_ids) if allowed_tag_ids else None,
    }
    report_json_path.write_text(json.dumps(report, indent=2))

    print("")
    print("Per-Image Tracking Optimization")
    print("--------------------------------")
    print(f"Images found: {len(images)}")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Annotated images backfilled: {backfilled_annotations}")
    print(f"Failed: {failed}")
    print("Filtered decoded tags:")
    print(f"  too_small: {sum(row_int(row, 'filtered_too_small_count') for row in rows)}")
    print(f"  too_large: {sum(row_int(row, 'filtered_too_large_count') for row in rows)}")
    print(
        "  outside_tag_list: "
        f"{sum(row_int(row, 'filtered_outside_tag_list_count') for row in rows)}"
    )
    other_total = sum(row_int(row, "filtered_other_count") for row in rows)
    if other_total:
        print(f"  other: {other_total}")
    print(f"Elapsed (s): {elapsed:.1f}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Combined detection CSV: {combined_detection_csv_path}")
    print(f"Report JSON: {report_json_path}")
    return 1 if failed else 0


def _cmd_track_videos(args: argparse.Namespace) -> int:
    from .posthoc_tracking import format_posthoc_tracking_report, run_posthoc_tracking

    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        allowed_ids = _parse_comma_numeric_values(
            args.allowed_tag_ids,
            label="--allowed-tag-ids",
            value_type="int",
        )
        extensions = [
            token.strip()
            for token in str(args.extensions or "").split(",")
            if token.strip()
        ]
        if bool(args.optimize_per_date) and args.optimization_sample_frames is None:
            print(
                "Argument error: --optimization-sample-frames is required when "
                "--optimize-per-date is used. Recommended: --optimization-sample-frames 40"
            )
            return 2
        if bool(args.optimization_only) and not bool(args.optimize_per_date):
            print("Argument error: --optimization-only requires --optimize-per-date.")
            return 2

        _posthoc_progress = _LiveProgressPrinter(enabled=not bool(args.no_live_progress))

        report = run_posthoc_tracking(
            config,
            input_path=args.input,
            output_root=args.output_root,
            params_path=args.params,
            allowed_ids=[int(value) for value in allowed_ids] if allowed_ids else None,
            tag_list_path=args.tag_list,
            render_tracked_video=args.render_tracked_video,
            run_cleaning=not bool(args.no_cleaning),
            run_metrics=args.metrics,
            recursive=not bool(args.no_recursive),
            extensions=extensions,
            dry_run=bool(args.dry_run),
            optimize_per_date=bool(args.optimize_per_date),
            force_optimize_per_date=bool(args.force_optimize_per_date),
            optimization_profile=args.optimization_profile,
            optimization_sample_frames=args.optimization_sample_frames or 40,
            optimization_tag_size_mm=args.optimization_tag_size_mm,
            optimization_expected_tags=args.optimization_expected_tags,
            optimization_max_combinations=args.optimization_max_combinations,
            optimization_execution_target=args.optimization_execution_target,
            optimization_workers=args.optimization_workers,
            optimization_selection=args.optimization_selection,
            optimization_strategy=args.optimization_strategy,
            optimization_initial_sample_frames=args.optimization_initial_sample_frames,
            optimization_middle_sample_frames=args.optimization_middle_sample_frames,
            optimization_only=bool(args.optimization_only),
            resume_tracking=not bool(args.force_retrack),
            legacy_skip_existing_tracking=bool(args.legacy_skip_existing_tracking),
            progress_callback=_posthoc_progress,
        )
    except Exception as exc:
        print(f"Post-hoc tracking failed: {exc}")
        return 1

    print(format_posthoc_tracking_report(report))
    if not bool(args.dry_run):
        try:
            from .tracking_index import format_local_index_result, sync_posthoc_tracking_report

            print("")
            print(format_local_index_result(sync_posthoc_tracking_report(config, report)))
        except Exception as exc:
            print(f"\nLocal tracking index update failed: {exc}")
    return 1 if report.videos_failed > 0 else 0


def _cmd_schedule_check(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        report = run_schedule_check(
            config=config,
            benchmark_input=args.benchmark_input,
            benchmark_frames=args.benchmark_frames,
            assume_ram_gb=args.assume_ram_gb,
        )
    except Exception as exc:
        print(f"Schedule check failed: {exc}")
        return 1

    print(format_schedule_check_report(report))
    return 1 if report.has_failures else 0


def _cmd_systemd_write(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        result = write_systemd_units(
            config=config,
            config_path=config_path,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Failed to write systemd units: {exc}")
        return 1

    print(format_systemd_result(result))
    return 0


def _cmd_systemd_action(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        result = run_systemd_action(
            config=config,
            action=args.action,
            output_dir=args.output_dir,
            config_path=config_path,
        )
    except Exception as exc:
        print(f"Failed to run systemd action: {exc}")
        return 1

    print(format_systemd_action_result(result))
    return 0 if result.success else 1


def _add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_USER_CONFIG_PATH),
        help=f"Path to BumbleBox V2 config (default: {DEFAULT_USER_CONFIG_PATH})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bbx",
        description="BumbleBox V2 command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a default BumbleBox V2 config file.")
    init_parser.add_argument(
        "--config",
        default=str(DEFAULT_USER_CONFIG_PATH),
        help=f"Path for generated config (default: {DEFAULT_USER_CONFIG_PATH})",
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config.")
    init_parser.set_defaults(func=_cmd_init)

    doctor_parser = subparsers.add_parser("doctor", help="Run hardware and dependency checks.")
    _add_common_config_arg(doctor_parser)
    doctor_parser.set_defaults(func=_cmd_doctor)

    storage_parser = subparsers.add_parser(
        "storage",
        help="Inspect and configure external storage auto-mount.",
    )
    storage_sub = storage_parser.add_subparsers(dest="storage_command", required=True)

    storage_status = storage_sub.add_parser(
        "status",
        help="Show current storage mount status for BumbleBox data root.",
    )
    _add_common_config_arg(storage_status)
    storage_status.add_argument(
        "--mount-point",
        help="Override mount point to inspect (default: system.data_root from config).",
    )
    storage_status.set_defaults(func=_cmd_storage_status)

    storage_set_mount = storage_sub.add_parser(
        "set-mount-point",
        help="Update system.data_root mount point in BumbleBox config.",
    )
    _add_common_config_arg(storage_set_mount)
    storage_set_mount.add_argument("--mount-point", required=True, help="New mount point path.")
    storage_set_mount.add_argument("--dry-run", action="store_true", help="Print change without writing config.")
    storage_set_mount.set_defaults(func=_cmd_storage_set_mount_point)

    storage_mount = storage_sub.add_parser(
        "mount",
        help="Mount a storage device at the chosen mount point without editing /etc/fstab.",
    )
    _add_common_config_arg(storage_mount)
    storage_mount.add_argument(
        "--mount-point",
        help="Mount point to use now (default: system.data_root from config).",
    )
    storage_mount.add_argument(
        "--device",
        help="Optional specific device path/name (example /dev/sda1).",
    )
    storage_mount.add_argument(
        "--apply-config",
        action="store_true",
        help="Also save mount point into system.data_root in config.",
    )
    storage_mount.add_argument("--dry-run", action="store_true", help="Show actions without running mount.")
    storage_mount.set_defaults(func=_cmd_storage_mount)

    storage_setup = storage_sub.add_parser(
        "setup",
        help="Configure UUID-based /etc/fstab auto-mount and mount storage now.",
    )
    _add_common_config_arg(storage_setup)
    storage_setup.add_argument(
        "--mount-point",
        help="Mount point to configure (default: system.data_root from config).",
    )
    storage_setup.add_argument(
        "--device",
        help="Optional specific device path/name (example /dev/sda1).",
    )
    storage_setup.add_argument(
        "--apply-config",
        action="store_true",
        help="Also save mount point into system.data_root in config.",
    )
    storage_setup.add_argument("--dry-run", action="store_true", help="Show actions without editing /etc/fstab.")
    storage_setup.set_defaults(func=_cmd_storage_setup)

    camera_preview_parser = subparsers.add_parser(
        "camera-preview",
        help="Open a timed live camera preview window (focus/exposure/framing check).",
    )
    _add_common_config_arg(camera_preview_parser)
    camera_preview_parser.add_argument("--seconds", type=float, default=20.0, help="Preview duration in seconds.")
    camera_preview_parser.add_argument(
        "--window",
        choices=["QTGL", "QT", "DRM"],
        default="QT",
        help="Picamera2 preview backend window type.",
    )
    camera_preview_parser.add_argument("--width", type=int, help="Optional preview width override in pixels.")
    camera_preview_parser.add_argument("--height", type=int, help="Optional preview height override in pixels.")
    preview_infrared_group = camera_preview_parser.add_mutually_exclusive_group()
    preview_infrared_group.add_argument(
        "--infrared",
        dest="infrared",
        action="store_true",
        help="Force IR/NoIR camera tuning selection for this preview when no manual camera.tuning_file override is set.",
    )
    preview_infrared_group.add_argument(
        "--no-infrared",
        dest="infrared",
        action="store_false",
        help="Force standard non-IR camera tuning selection for this preview when no manual camera.tuning_file override is set.",
    )
    camera_preview_parser.set_defaults(infrared=None)
    preview_monochrome_group = camera_preview_parser.add_mutually_exclusive_group()
    preview_monochrome_group.add_argument(
        "--monochrome-output",
        dest="monochrome_output",
        action="store_true",
        help="Force grayscale RGB output for this preview when comparing IR footage or reducing the usual NoIR color cast.",
    )
    preview_monochrome_group.add_argument(
        "--no-monochrome-output",
        dest="monochrome_output",
        action="store_false",
        help="Force normal color RGB output for this preview.",
    )
    camera_preview_parser.set_defaults(monochrome_output=None)
    camera_preview_parser.set_defaults(func=_cmd_camera_preview)

    camera_reset_parser = subparsers.add_parser(
        "camera-reset",
        help="Run a conservative camera open/close reset probe in a fresh process.",
    )
    _add_common_config_arg(camera_reset_parser)
    camera_reset_parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.5,
        help="Seconds to wait after releasing the camera before reporting status (default 1.5).",
    )
    camera_reset_parser.set_defaults(func=_cmd_camera_reset)

    camera_test_parser = subparsers.add_parser(
        "camera-test-tracking",
        help="Run live ArUco detection from camera feed and summarize tag-detection performance.",
    )
    _add_common_config_arg(camera_test_parser)
    camera_test_parser.add_argument("--seconds", type=float, default=20.0, help="Live tracking test duration in seconds.")
    camera_test_parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Display width for preview window (for readability on desktop/Pi monitors).",
    )
    camera_test_parser.add_argument(
        "--dictionary",
        help="Optional ArUco dictionary override (for example 4X4_50). Defaults to tracking.tag_dictionary in config.",
    )
    camera_test_parser.add_argument(
        "--box-preset",
        choices=["auto", "custom", "koppert", "none"],
        default="auto",
        help="Use config preset (auto), force custom/koppert, or none.",
    )
    camera_test_parser.add_argument("--show-rejected", action="store_true", help="Draw rejected candidate quads in orange.")
    camera_test_parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE pre-processing.")
    camera_test_parser.add_argument("--json-out", help="Optional output path for JSON summary report.")
    camera_test_parser.set_defaults(func=_cmd_camera_test_tracking)

    roadmap_parser = subparsers.add_parser("roadmap", help="Show user roadmap based on current config.")
    _add_common_config_arg(roadmap_parser)
    roadmap_parser.set_defaults(func=_cmd_roadmap)

    fps_parser = subparsers.add_parser("fps-report", help="Build FPS quality report for a recorded video.")
    fps_parser.add_argument("--video", required=True, help="Path to recorded video (.mp4/.mjpeg).")
    fps_parser.add_argument("--timestamps", help="Optional path to frame timestamp sidecar file.")
    fps_parser.add_argument("--recording-seconds", type=float, help="Expected recording duration in seconds.")
    fps_parser.add_argument("--json-out", help="Optional path to save JSON report.")
    fps_parser.set_defaults(func=_cmd_fps_report)

    fps_sweep_parser = subparsers.add_parser(
        "fps-sweep",
        help=(
            "Probe target FPS values and estimate max recording durations using the current workflow "
            "(empirical RAM profiling for MP4 or RAM tracking on the current machine, "
            "disk-backed profiling for MJPEG video-backed capture). "
            "If recent tracking exists, include estimated tracking time."
        ),
    )
    _add_common_config_arg(fps_sweep_parser)
    fps_sweep_parser.add_argument(
        "--fps-values",
        help="Comma-separated target FPS list (example: 2,4,6,8). Overrides --fps-start/--fps-stop/--fps-step.",
    )
    fps_sweep_parser.add_argument("--fps-start", type=float, default=2.0, help="FPS range start (default 2.0).")
    fps_sweep_parser.add_argument("--fps-stop", type=float, default=20.0, help="FPS range stop (default 20.0).")
    fps_sweep_parser.add_argument("--fps-step", type=float, default=2.0, help="FPS range step (default 2.0).")
    fps_sweep_parser.add_argument(
        "--probe-seconds",
        type=float,
        default=20.0,
        help="Capture duration for each FPS probe (default 20 seconds).",
    )
    fps_sweep_parser.add_argument(
        "--assume-ram-gb",
        type=float,
        help=(
            "Optional RAM size (GiB) to simulate target hardware during RAM-backed duration estimation. "
            "When set, BumbleBox uses the heuristic RAM model instead of empirical on-machine RAM profiling. "
            "Ignored for disk-backed MJPEG sweeps."
        ),
    )
    fps_sweep_parser.add_argument(
        "--session-start",
        help=(
            "Optional ISO timestamp to restrict tracking benchmark lookup "
            "(example: 2026-02-17T10:30:00)."
        ),
    )
    fps_sweep_parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Use synthetic frames instead of picamera2 capture for probe/testing.",
    )
    fps_sweep_parser.add_argument("--json-out", help="Optional path to save JSON sweep report.")
    fps_sweep_parser.set_defaults(func=_cmd_fps_sweep)

    thermal_parser = subparsers.add_parser(
        "thermal-check",
        help=(
            "Discover USB/V4L2 thermal camera devices, identify likely PureThermal/Lepton candidates, "
            "and run a basic OpenCV probe without touching the Pi HQ camera stack."
        ),
    )
    _add_common_config_arg(thermal_parser)
    thermal_parser.add_argument(
        "--device",
        help="Optional explicit thermal device path (example: /dev/video2). Overrides thermal.device_path.",
    )
    thermal_parser.add_argument("--width", type=int, help="Optional probe width override.")
    thermal_parser.add_argument("--height", type=int, help="Optional probe height override.")
    thermal_parser.add_argument(
        "--apply",
        action="store_true",
        help="If the thermal check fully passes, write the detected stable path and Y16 settings into config.",
    )
    thermal_parser.add_argument("--json-out", help="Optional path to save JSON report.")
    thermal_parser.set_defaults(func=_cmd_thermal_check)

    thermal_snapshot_parser = subparsers.add_parser(
        "thermal-snapshot",
        help=(
            "Capture one raw thermal frame from the selected PureThermal/Lepton device, save the raw 16-bit data, "
            "a 16-bit PNG, a preview PNG, and metadata."
        ),
    )
    _add_common_config_arg(thermal_snapshot_parser)
    thermal_snapshot_parser.add_argument(
        "--device",
        help="Optional explicit thermal device path (example: /dev/video8). Overrides thermal.device_path.",
    )
    thermal_snapshot_parser.add_argument("--width", type=int, help="Optional capture width override.")
    thermal_snapshot_parser.add_argument("--height", type=int, help="Optional capture height override.")
    thermal_snapshot_parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to <system.data_root>/<date>/thermal.",
    )
    thermal_snapshot_parser.add_argument("--json-out", help="Optional path to save JSON report.")
    thermal_snapshot_parser.set_defaults(func=_cmd_thermal_snapshot)

    run_once_parser = subparsers.add_parser(
        "run-once",
        help="Execute one BumbleBox V2 run (record, track, or record+track).",
    )
    _add_common_config_arg(run_once_parser)
    run_once_parser.add_argument(
        "--mode",
        choices=["record_only", "track_only", "record_and_track"],
        help="Optional mode override. Useful when pipeline.mode is mixed_schedule.",
    )
    run_once_parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Use synthetic frames instead of picamera2 capture (for testing).",
    )
    run_once_parser.add_argument(
        "--codec",
        choices=["mp4", "mjpeg"],
        help="Optional one-run recording codec override (default from camera.codec in config).",
    )
    infrared_group = run_once_parser.add_mutually_exclusive_group()
    infrared_group.add_argument(
        "--infrared",
        dest="infrared",
        action="store_true",
        help="Force IR/NoIR camera tuning selection for this run when no manual camera.tuning_file override is set.",
    )
    infrared_group.add_argument(
        "--no-infrared",
        dest="infrared",
        action="store_false",
        help="Force standard non-IR camera tuning selection for this run when no manual camera.tuning_file override is set.",
    )
    run_once_parser.set_defaults(infrared=None)
    monochrome_group = run_once_parser.add_mutually_exclusive_group()
    monochrome_group.add_argument(
        "--monochrome-output",
        dest="monochrome_output",
        action="store_true",
        help="Force grayscale RGB output for this run.",
    )
    monochrome_group.add_argument(
        "--no-monochrome-output",
        dest="monochrome_output",
        action="store_false",
        help="Force normal color RGB output for this run.",
    )
    run_once_parser.set_defaults(monochrome_output=None)
    visualization_group = run_once_parser.add_mutually_exclusive_group()
    visualization_group.add_argument(
        "--visualization",
        dest="visualization",
        action="store_true",
        help="Render a tracked overlay video after tracking. If thermal recording is enabled, also render a tracked RGB+thermal side-by-side video.",
    )
    visualization_group.add_argument(
        "--no-visualization",
        dest="visualization",
        action="store_false",
        help="Do not render tracked overlay video artifacts for this run.",
    )
    run_once_parser.set_defaults(visualization=None)
    run_once_parser.set_defaults(func=_cmd_run_once)

    track_videos_parser = subparsers.add_parser(
        "track-videos",
        help="Run post-hoc ArUco tracking on existing videos without opening the live camera.",
    )
    _add_common_config_arg(track_videos_parser)
    track_videos_parser.add_argument(
        "--input",
        required=True,
        help="Video file, date folder, or colony parent folder containing recorded videos.",
    )
    track_videos_parser.add_argument(
        "--output-root",
        help=(
            "Optional output root. If omitted, each date folder gets a tracking/ subfolder. "
            "If set, outputs go to <output-root>/<date>/tracking/."
        ),
    )
    track_videos_parser.add_argument(
        "--params",
        help=(
            "Optional selected tracking parameter JSON. If omitted, BumbleBox looks for "
            "<date>/optimization/selected_tracking_params.json, then top_mean_detection_params.json, "
            "then best_score_params.json, then falls back to tracking.aruco_params in config."
        ),
    )
    track_videos_parser.add_argument(
        "--tag-list",
        help=(
            "Optional colony allowlist file. Supports JSON list/object, CSV-ish text, or newline-separated IDs. "
            "Detections outside the allowlist are removed after ArUco detection."
        ),
    )
    track_videos_parser.add_argument(
        "--allowed-tag-ids",
        default="",
        help="Optional comma-separated colony allowlist IDs, combined with --tag-list and config tracking.allowed_tag_ids.",
    )
    track_videos_parser.add_argument(
        "--extensions",
        default="mp4,mjpeg,mjpe",
        help="Comma-separated video extensions to track (default: mp4,mjpeg,mjpe).",
    )
    track_videos_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the input directory itself, not nested date/session folders.",
    )
    visualization_group = track_videos_parser.add_mutually_exclusive_group()
    visualization_group.add_argument(
        "--render-tracked-video",
        dest="render_tracked_video",
        action="store_true",
        help="Write annotated tracked MP4 videos next to the tracking CSVs.",
    )
    visualization_group.add_argument(
        "--no-render-tracked-video",
        dest="render_tracked_video",
        action="store_false",
        help="Do not write annotated tracked MP4 videos.",
    )
    track_videos_parser.set_defaults(render_tracked_video=None)
    metrics_group = track_videos_parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--metrics",
        dest="metrics",
        action="store_true",
        help="Run behavior metrics after tracking.",
    )
    metrics_group.add_argument(
        "--no-metrics",
        dest="metrics",
        action="store_false",
        help="Skip behavior metrics after tracking.",
    )
    track_videos_parser.set_defaults(metrics=None)
    track_videos_parser.add_argument(
        "--no-cleaning",
        action="store_true",
        help="Skip data-cleaning outputs and only write raw/noID tracking CSVs.",
    )
    track_videos_parser.add_argument(
        "--force-retrack",
        action="store_true",
        help="Ignore completed per-video tracking markers and regenerate tracking outputs.",
    )
    track_videos_parser.add_argument(
        "--legacy-skip-existing-tracking",
        action="store_true",
        help=(
            "Temporary bridge for pre-marker runs: skip a video when existing raw/noID tracking CSVs "
            "are present, then write a completion marker for future safe resume."
        ),
    )
    track_videos_parser.add_argument(
        "--no-live-progress",
        action="store_true",
        help="Disable in-place terminal updates and print every optimization progress table separately.",
    )
    track_videos_parser.add_argument(
        "--optimization-only",
        action="store_true",
        help="Run per-date optimization and write selected params, then stop before tracking videos.",
    )
    track_videos_parser.add_argument(
        "--optimize-per-date",
        action="store_true",
        help=(
            "Before tracking, optimize ArUco parameters once per date folder, write "
            "<date>/optimization/selected_tracking_params.json, then track that date with those params."
        ),
    )
    track_videos_parser.add_argument(
        "--force-optimize-per-date",
        action="store_true",
        help="Regenerate per-date selected_tracking_params.json even if one already exists.",
    )
    track_videos_parser.add_argument(
        "--optimization-profile",
        choices=["quick", "balanced", "deep", "daily"],
        default="daily",
        help="Optimization sweep profile used by --optimize-per-date (default: daily, 360 combinations before caps).",
    )
    track_videos_parser.add_argument(
        "--optimization-sample-frames",
        type=int,
        default=None,
        help=(
            "Required with --optimize-per-date. Representative frames sampled across each "
            "date's videos for per-date optimization. Recommended: 40."
        ),
    )
    track_videos_parser.add_argument(
        "--optimization-tag-size-mm",
        type=float,
        default=2.5,
        help="Physical tag size passed to the optimizer when size thresholds are not fixed by existing params.",
    )
    track_videos_parser.add_argument(
        "--optimization-expected-tags",
        type=float,
        help="Optional expected average visible tag count per optimization sample frame.",
    )
    track_videos_parser.add_argument(
        "--optimization-max-combinations",
        type=int,
        default=750,
        help="Maximum parameter combinations per date optimization (default: 750).",
    )
    track_videos_parser.add_argument(
        "--optimization-execution-target",
        choices=["pi_safe", "desktop"],
        default="pi_safe",
        help="Worker-count defaults for per-date optimization.",
    )
    track_videos_parser.add_argument("--optimization-workers", type=int, help="Explicit worker count for per-date optimization.")
    track_videos_parser.add_argument(
        "--optimization-selection",
        choices=["mean_detection", "best_score"],
        default="mean_detection",
        help="Which optimizer winner to write as selected_tracking_params.json (default: mean_detection).",
    )
    track_videos_parser.add_argument(
        "--optimization-strategy",
        choices=["successive_halving", "exhaustive"],
        default="successive_halving",
        help=(
            "Per-date optimization strategy. successive_halving tests the broad grid on a small "
            "initial frame set, then validates survivors; exhaustive tests every candidate on all sample frames."
        ),
    )
    track_videos_parser.add_argument(
        "--optimization-initial-sample-frames",
        type=int,
        default=5,
        help="Initial frame count for --optimization-strategy successive_halving (default: 5).",
    )
    track_videos_parser.add_argument(
        "--optimization-middle-sample-frames",
        type=int,
        help=(
            "Optional middle-stage frame count for --optimization-strategy successive_halving. "
            "If omitted, BumbleBox uses about half of --optimization-sample-frames."
        ),
    )
    track_videos_parser.add_argument("--dry-run", action="store_true", help="Show planned videos without tracking them.")
    track_videos_parser.set_defaults(func=_cmd_track_videos)

    fleet_parser = subparsers.add_parser(
        "fleet",
        help="Manage queen/worker fleet orchestration and worker health checks.",
    )
    fleet_sub = fleet_parser.add_subparsers(dest="fleet_command", required=True)

    fleet_init = fleet_sub.add_parser(
        "init-queen",
        help="Configure this node as fleet queen and prepare SSH/time-sync hints.",
        description=(
            "Initialize fleet queen settings. "
            "Use '--queen-interface-only' for a controller-only queen (no local recording/tracking), "
            "or '--queen-bbox-active' to keep local recording/tracking enabled on the queen."
        ),
    )
    _add_common_config_arg(fleet_init)
    fleet_init.add_argument("--queen-host", help="Host/IP workers should use for queen time sync.")
    fleet_init.add_argument("--ssh-user", help="Default SSH user for workers (default from config).")
    fleet_init.add_argument("--identity-file", help="SSH identity file path (default ~/.ssh/bbx_fleet_ed25519).")
    fleet_init.add_argument("--skip-keygen", action="store_true", help="Do not create keypair if missing.")
    queen_mode_group = fleet_init.add_mutually_exclusive_group()
    queen_mode_group.add_argument(
        "--queen-interface-only",
        action="store_true",
        help="Controller-only queen: disable local recording/tracking jobs on this queen (default behavior).",
    )
    queen_mode_group.add_argument(
        "--queen-bbox-active",
        action="store_true",
        help="Active queen: keep local recording/tracking jobs enabled on this queen.",
    )
    fleet_init.add_argument("--show-public-key", action="store_true", help="Print public key in output.")
    fleet_init.add_argument("--dry-run", action="store_true", help="Print changes without writing config.")
    fleet_init.set_defaults(func=_cmd_fleet_init_queen)

    fleet_enroll = fleet_sub.add_parser(
        "enroll-worker",
        help="Add or update a worker host in fleet config.",
    )
    _add_common_config_arg(fleet_enroll)
    fleet_enroll.add_argument("--host", required=True, help="Worker hostname or IP.")
    fleet_enroll.add_argument("--name", help="Optional worker display name.")
    fleet_enroll.add_argument("--user", help="SSH username for worker.")
    fleet_enroll.add_argument("--port", type=int, default=22, help="SSH port (default 22).")
    fleet_enroll.add_argument(
        "--data-root",
        default="/mnt/bumblebox/data",
        help="Worker data root for health checks (default /mnt/bumblebox/data).",
    )
    fleet_enroll.add_argument(
        "--unit-prefix",
        default="bumblebox-v2",
        help="Worker systemd unit prefix for health checks (default bumblebox-v2).",
    )
    fleet_enroll.add_argument("--identity-file", help="SSH identity file override.")
    fleet_enroll.add_argument("--install-key", action="store_true", help="Attempt ssh-copy-id immediately.")
    fleet_enroll.add_argument("--disabled", action="store_true", help="Enroll worker as disabled.")
    fleet_enroll.add_argument("--dry-run", action="store_true", help="Print changes without writing config.")
    fleet_enroll.set_defaults(func=_cmd_fleet_enroll_worker)

    fleet_status = fleet_sub.add_parser(
        "status",
        help="Run SSH-based fleet health checks from queen to workers.",
    )
    _add_common_config_arg(fleet_status)
    fleet_status.add_argument("--worker", help="Filter worker by partial name or host.")
    fleet_status.add_argument("--include-disabled", action="store_true", help="Include disabled workers.")
    fleet_status.add_argument("--identity-file", help="SSH identity file override.")
    fleet_status.add_argument("--json-out", help="Optional output path for JSON report.")
    fleet_status.set_defaults(func=_cmd_fleet_status)

    fleet_latest_status = fleet_sub.add_parser(
        "latest-status",
        help="Show side-by-side latest pulled vs latest tracked status per worker.",
    )
    _add_common_config_arg(fleet_latest_status)
    fleet_latest_status.add_argument("--worker", help="Filter worker by partial name or host.")
    fleet_latest_status.add_argument("--include-disabled", action="store_true", help="Include disabled workers.")
    fleet_latest_status.add_argument("--identity-file", help="SSH identity file override for reachability checks.")
    fleet_latest_status.add_argument(
        "--output-root",
        help="Output root for queen latest media pointers (default: <data_root>/queen_worker_tracking).",
    )
    fleet_latest_status.add_argument(
        "--no-reachability",
        action="store_true",
        help="Skip worker SSH reachability checks and only report latest media pointer state.",
    )
    fleet_latest_status.add_argument("--json-out", help="Optional output path for JSON report.")
    fleet_latest_status.set_defaults(func=_cmd_fleet_latest_status)

    fleet_discover = fleet_sub.add_parser(
        "discover",
        help="Scan LAN neighbors and probe configured workers for online/offline warnings.",
    )
    _add_common_config_arg(fleet_discover)
    fleet_discover.add_argument("--worker", help="Filter configured workers by partial name or host.")
    fleet_discover.add_argument("--include-disabled", action="store_true", help="Include disabled configured workers.")
    fleet_discover.add_argument(
        "--no-ping",
        action="store_true",
        help="Skip ping probe and only test TCP/22 reachability.",
    )
    fleet_discover.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.6,
        help="Network probe timeout in seconds (default 0.6).",
    )
    fleet_discover.add_argument("--json-out", help="Optional output path for JSON report.")
    fleet_discover.set_defaults(func=_cmd_fleet_discover)

    fleet_pull_latest = fleet_sub.add_parser(
        "queen-pull-latest",
        help="Queen pulls newest worker video and updates per-worker latest_video pointer.",
    )
    _add_common_config_arg(fleet_pull_latest)
    fleet_pull_latest.add_argument("--worker", help="Filter worker by partial name or host.")
    fleet_pull_latest.add_argument("--include-disabled", action="store_true", help="Include disabled workers.")
    fleet_pull_latest.add_argument("--identity-file", help="SSH identity file override.")
    fleet_pull_latest.add_argument(
        "--output-root",
        help="Output root for pulled videos/latest pointers (default: <data_root>/queen_worker_tracking).",
    )
    fleet_pull_latest.add_argument(
        "--max-videos-total",
        type=int,
        default=None,
        help="Maximum workers/videos to process in this run (default from fleet.queen_media_schedule.max_videos_total).",
    )
    fleet_pull_latest.add_argument("--dry-run", action="store_true", help="Plan actions without SSH copy.")
    fleet_pull_latest.add_argument("--json-out", help="Optional output path for JSON report.")
    fleet_pull_latest.set_defaults(func=_cmd_fleet_queen_pull_latest)

    fleet_track_latest = fleet_sub.add_parser(
        "queen-track-latest",
        help="Queen tracks the per-worker latest pulled video and updates latest_tracked pointer.",
    )
    _add_common_config_arg(fleet_track_latest)
    fleet_track_latest.add_argument("--worker", help="Filter worker by partial name or host.")
    fleet_track_latest.add_argument("--include-disabled", action="store_true", help="Include disabled workers.")
    fleet_track_latest.add_argument(
        "--output-root",
        help="Output root for pulled/tracked media (default: <data_root>/queen_worker_tracking).",
    )
    fleet_track_latest.add_argument(
        "--max-videos-total",
        type=int,
        default=None,
        help="Maximum worker latest-videos to track in this run (default from fleet.queen_media_schedule.max_videos_total).",
    )
    fleet_track_latest.add_argument(
        "--cooldown-minutes",
        type=int,
        default=None,
        help="Minimum minutes between tracking runs for the same latest video (default from fleet.queen_media_schedule.cooldown_minutes).",
    )
    fleet_track_latest.add_argument(
        "--max-queen-load-1m",
        type=float,
        default=None,
        help="Guardrail: skip run if queen load(1m) is above this value (default from fleet.queen_media_schedule.max_queen_load_1m).",
    )
    fleet_track_latest.add_argument(
        "--min-queen-mem-gb",
        type=float,
        default=None,
        help="Guardrail: skip run if queen available memory is below this value (default from fleet.queen_media_schedule.min_queen_mem_gb).",
    )
    fleet_track_latest.add_argument(
        "--allow-when-queen-bbox-active",
        action="store_true",
        help="Allow tracking even when queen local recording/tracking pipeline is active.",
    )
    fleet_track_latest.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip tracked-video rendering step; output tracking CSV only.",
    )
    fleet_track_latest.add_argument("--dry-run", action="store_true", help="Plan actions without tracking.")
    fleet_track_latest.add_argument("--json-out", help="Optional output path for JSON report.")
    fleet_track_latest.set_defaults(func=_cmd_fleet_queen_track_latest)

    fleet_pull_track = fleet_sub.add_parser(
        "queen-pull-track",
        help="Combined convenience: run queen-pull-latest then queen-track-latest.",
    )
    _add_common_config_arg(fleet_pull_track)
    fleet_pull_track.add_argument("--worker", help="Filter worker by partial name or host.")
    fleet_pull_track.add_argument("--include-disabled", action="store_true", help="Include disabled workers.")
    fleet_pull_track.add_argument("--identity-file", help="SSH identity file override.")
    fleet_pull_track.add_argument(
        "--output-root",
        help="Output root for pulled videos/tracking results (default: <data_root>/queen_worker_tracking).",
    )
    fleet_pull_track.add_argument(
        "--max-videos-total",
        type=int,
        default=None,
        help="Maximum worker videos to process in this run (default from fleet.queen_media_schedule.max_videos_total).",
    )
    fleet_pull_track.add_argument(
        "--max-videos-per-worker",
        type=int,
        default=1,
        help="Reserved for compatibility (current flow processes latest one per worker).",
    )
    fleet_pull_track.add_argument(
        "--cooldown-minutes",
        type=int,
        default=None,
        help="Minimum minutes between tracking runs for the same latest video (default from fleet.queen_media_schedule.cooldown_minutes).",
    )
    fleet_pull_track.add_argument(
        "--max-queen-load-1m",
        type=float,
        default=None,
        help="Guardrail: skip run if queen load(1m) is above this value (default from fleet.queen_media_schedule.max_queen_load_1m).",
    )
    fleet_pull_track.add_argument(
        "--min-queen-mem-gb",
        type=float,
        default=None,
        help="Guardrail: skip run if queen available memory is below this value (default from fleet.queen_media_schedule.min_queen_mem_gb).",
    )
    fleet_pull_track.add_argument(
        "--allow-when-queen-bbox-active",
        action="store_true",
        help="Allow pull-track even when queen local recording/tracking pipeline is active.",
    )
    fleet_pull_track.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip tracked-video rendering step; do tracking CSV output only.",
    )
    fleet_pull_track.add_argument("--dry-run", action="store_true", help="Plan actions without SSH copy or tracking.")
    fleet_pull_track.add_argument("--json-out", help="Optional output path for JSON report.")
    fleet_pull_track.set_defaults(func=_cmd_fleet_queen_pull_track)

    export_parser = subparsers.add_parser(
        "export-bundle",
        help="Create a portable run bundle for downstream desktop analysis.",
    )
    _add_common_config_arg(export_parser)
    selector = export_parser.add_mutually_exclusive_group()
    selector.add_argument("--summary", help="Path to a specific *_run_summary.json file.")
    selector.add_argument("--session-dir", help="Path to session directory containing *_run_summary.json.")
    selector.add_argument(
        "--latest",
        action="store_true",
        help="Export the latest run found under data root (default behavior when no selector is provided).",
    )
    export_parser.add_argument(
        "--data-root",
        help="Data root used with --latest. Defaults to system.data_root from config.",
    )
    export_parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "bundles"),
        help="Directory where bundle folder and optional zip are written.",
    )
    export_parser.add_argument("--bundle-name", help="Optional custom bundle folder name.")
    export_parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Exclude session recording video (.mp4/.mjpeg) from bundle (smaller transfer size).",
    )
    export_parser.add_argument(
        "--core-only",
        action="store_true",
        help="Include only core expected artifacts, excluding extra session files.",
    )
    export_parser.add_argument(
        "--no-config",
        action="store_true",
        help="Do not include current config file snapshot in bundle.",
    )
    zip_group = export_parser.add_mutually_exclusive_group()
    zip_group.add_argument("--zip", dest="zip_bundle", action="store_true", help="Also create .zip archive (default).")
    zip_group.add_argument("--no-zip", dest="zip_bundle", action="store_false", help="Skip .zip archive creation.")
    export_parser.set_defaults(func=_cmd_export_bundle, zip_bundle=True)

    systemd_parser = subparsers.add_parser(
        "systemd-write",
        help="Generate systemd service/timer units from config.",
    )
    _add_common_config_arg(systemd_parser)
    systemd_parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "systemd"),
        help="Directory to write generated .service/.timer files.",
    )
    systemd_parser.set_defaults(func=_cmd_systemd_write)

    for action in ("install", "enable", "disable", "status"):
        action_parser = subparsers.add_parser(
            f"systemd-{action}",
            help=f"Run systemd action: {action}.",
        )
        _add_common_config_arg(action_parser)
        action_parser.add_argument(
            "--output-dir",
            default=str(Path.cwd() / "systemd"),
            help="Directory containing generated .service/.timer files (used by install).",
        )
        action_parser.set_defaults(func=_cmd_systemd_action, action=action)

    calibrate_parser = subparsers.add_parser(
        "calibrate-scale",
        help="Calibrate pixel-to-distance conversion and update config.",
    )
    calibrate_sub = calibrate_parser.add_subparsers(dest="calibrate_mode", required=True)

    manual_parser = calibrate_sub.add_parser(
        "manual",
        help="Calibrate using two points with known real distance (recommended baseline >=5 cm).",
    )
    _add_common_config_arg(manual_parser)
    manual_parser.add_argument("--point-a", required=True, help="First point in x,y format.")
    manual_parser.add_argument("--point-b", required=True, help="Second point in x,y format.")
    manual_parser.add_argument(
        "--distance-cm",
        required=True,
        type=float,
        help="Real distance between points in centimeters (recommend using 5-15 cm).",
    )
    manual_parser.add_argument("--dry-run", action="store_true", help="Print calibration but do not update config.")
    manual_parser.set_defaults(func=_cmd_calibrate_manual)

    aruco_parser = calibrate_sub.add_parser("aruco", help="Calibrate from ArUco marker in an image.")
    _add_common_config_arg(aruco_parser)
    aruco_parser.add_argument("--image", required=True, help="Path to calibration image.")
    aruco_parser.add_argument("--marker-size-mm", required=True, type=float, help="Marker side length in millimeters.")
    aruco_parser.add_argument("--dictionary", default="4X4_50", help="ArUco dictionary name (default 4X4_50).")
    aruco_parser.add_argument("--marker-id", type=int, help="Optional marker ID to use.")
    aruco_parser.add_argument("--dry-run", action="store_true", help="Print calibration but do not update config.")
    aruco_parser.set_defaults(func=_cmd_calibrate_aruco)

    gui_parser = subparsers.add_parser("gui", help="Launch the BumbleBox V2 desktop GUI.")
    gui_parser.set_defaults(func=_cmd_gui)

    gui_shortcut_parser = subparsers.add_parser(
        "gui-install-shortcut",
        help="Create a desktop icon/launcher for BumbleBox GUI.",
        description=(
            "Create a Desktop launcher (.desktop), app-menu entry, launcher script, and icon for BumbleBox GUI. "
            "The launcher script prefers repo .venvs/bbx-runtime Python, then repo .venv, then system python3."
        ),
    )
    gui_shortcut_parser.add_argument("--name", default="BumbleBox GUI", help="Launcher display name.")
    gui_shortcut_parser.add_argument("--comment", default="Launch BumbleBox V2 GUI", help="Launcher description text.")
    gui_shortcut_parser.add_argument("--repo-root", help="Override BumbleBox repo root path.")
    gui_shortcut_parser.add_argument("--desktop-dir", help="Override Desktop directory for .desktop file.")
    gui_shortcut_parser.add_argument("--applications-dir", help="Override app-menu directory for .desktop file.")
    gui_shortcut_parser.add_argument("--bin-dir", help="Override directory for launcher script.")
    gui_shortcut_parser.add_argument(
        "--icon-path",
        help=(
            "Use an existing icon file path. If omitted, BumbleBox first checks repo assets "
            "(for example assets/bumblebox.png) then falls back to a generated default SVG icon."
        ),
    )
    gui_shortcut_parser.add_argument("--dry-run", action="store_true", help="Show target paths without writing files.")
    gui_shortcut_parser.set_defaults(func=_cmd_gui_install_shortcut)

    nest_parser = subparsers.add_parser(
        "nest-label",
        help="Check or launch the nest labeling interface.",
    )
    nest_sub = nest_parser.add_subparsers(dest="nest_command", required=True)

    nest_check_parser = nest_sub.add_parser(
        "check",
        help="Check PyQt/LabelMe/script readiness for nest labeling.",
    )
    nest_check_parser.add_argument("--folder", help="Path to image folder you plan to label.")
    nest_check_parser.add_argument(
        "--script",
        default=str(default_script_path()),
        help="Path to LabelNests GUI script.",
    )
    nest_check_parser.add_argument("--labelmerc", help="Optional path to labelmerc config.")
    nest_check_parser.add_argument(
        "--python",
        help=(
            "Optional Python executable for labeling environment. "
            "If omitted, BumbleBox auto-selects a dedicated label env."
        ),
    )
    nest_check_parser.set_defaults(func=_cmd_nest_label_check)

    nest_launch_parser = nest_sub.add_parser(
        "launch",
        help="Launch the nest labeling GUI.",
    )
    nest_launch_parser.add_argument("--folder", required=True, help="Path to image folder to label.")
    nest_launch_parser.add_argument(
        "--script",
        default=str(default_script_path()),
        help="Path to LabelNests GUI script.",
    )
    nest_launch_parser.add_argument("--labelmerc", help="Optional path to labelmerc config.")
    nest_launch_parser.add_argument(
        "--python",
        help=(
            "Optional Python executable for labeling environment. "
            "If omitted, BumbleBox auto-selects a dedicated label env."
        ),
    )
    nest_launch_parser.add_argument("--wait", action="store_true", help="Wait for process exit.")
    nest_launch_parser.set_defaults(func=_cmd_nest_label_launch)

    optimize_parser = subparsers.add_parser(
        "optimize-tracking",
        help="Optimize ArUco tracking parameters from a video or image folder.",
    )
    optimize_parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input video (.mp4/.mjpeg/...) or image directory. "
            "Image directories are searched recursively, excluding generated optimizer output folders."
        ),
    )
    optimize_parser.add_argument(
        "--profile",
        choices=["quick", "balanced", "deep", "daily"],
        default="quick",
        help="Grid profile size (quick is fastest).",
    )
    optimize_parser.add_argument(
        "--sample-frames",
        type=int,
        required=True,
        help="How many sampled frames/images to evaluate. Recommended: 40.",
    )
    optimize_parser.add_argument(
        "--dictionary",
        default="4X4_50",
        help="ArUco dictionary name (for example 4X4_50 or DICT_4X4_50).",
    )
    optimize_parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=2.5,
        help="Physical ArUco tag size in millimeters (default 2.5).",
    )
    optimize_parser.add_argument(
        "--sweep-min-marker-perimeter-rate",
        default="",
        help=(
            "Optional comma-separated override values for minMarkerPerimeterRate "
            "(for example 0.008,0.012,0.02)."
        ),
    )
    optimize_parser.add_argument(
        "--sweep-max-marker-perimeter-rate",
        default="",
        help=(
            "Optional comma-separated override values for maxMarkerPerimeterRate "
            "(for example 0.12,0.16,0.20)."
        ),
    )
    optimize_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-min",
        default="",
        help=(
            "Optional comma-separated override values for adaptiveThreshWinSizeMin "
            "(for example 3,5,7)."
        ),
    )
    optimize_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-max",
        default="",
        help=(
            "Optional comma-separated override values for adaptiveThreshWinSizeMax "
            "(for example 21,31,41)."
        ),
    )
    optimize_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-step",
        default="",
        help=(
            "Optional comma-separated override values for adaptiveThreshWinSizeStep "
            "(for example 2,4)."
        ),
    )
    optimize_parser.add_argument(
        "--sweep-polygonal-approx-accuracy-rate",
        default="",
        help=(
            "Optional comma-separated override values for polygonalApproxAccuracyRate "
            "(for example 0.06,0.08)."
        ),
    )
    optimize_parser.add_argument(
        "--sweep-adaptive-thresh-constant",
        default="",
        help=(
            "Optional comma-separated override values for adaptiveThreshConstant "
            "(for example 1,3,5,7,9,11)."
        ),
    )
    optimize_parser.add_argument(
        "--execution-target",
        choices=["pi_safe", "desktop"],
        default="pi_safe",
        help="pi_safe uses conservative worker defaults. desktop uses more cores.",
    )
    optimize_parser.add_argument(
        "--max-combinations",
        type=int,
        help="Optional cap on parameter combinations to evaluate after building the sweep.",
    )
    optimize_parser.add_argument("--workers", type=int, help="Optional explicit worker count override.")
    optimize_parser.add_argument(
        "--expected-tags",
        type=float,
        help="Optional expected average visible tag count per frame to guide scoring.",
    )
    optimize_parser.add_argument(
        "--tag-list",
        help=(
            "Optional colony allowlist file. Supports JSON list/object, CSV-ish text, or newline-separated IDs. "
            "Optimizer mean detections count only decoded tags inside this list."
        ),
    )
    optimize_parser.add_argument(
        "--allowed-tag-ids",
        default="",
        help="Optional comma-separated colony allowlist IDs for optimizer scoring.",
    )
    optimize_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving evaluations (0 disables; default 0).",
    )
    optimize_parser.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=0.0,
        help="Minimum score increase considered an improvement for early stop.",
    )
    optimize_parser.add_argument("--output-dir", help="Optional output root directory for optimization runs.")
    optimize_parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write a preview MP4 with best-parameter detections drawn.",
    )
    optimize_parser.add_argument(
        "--preview-frames",
        type=int,
        default=240,
        help="Max frame count for preview video when --write-preview is enabled.",
    )
    optimize_parser.add_argument("--top-k", type=int, default=5, help="How many top candidates to print.")
    optimize_parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best parameters into tracking.aruco_params in config.",
    )
    _add_common_config_arg(optimize_parser)
    optimize_parser.set_defaults(func=_cmd_optimize_tracking)

    optimize_video_ranges_parser = subparsers.add_parser(
        "optimize-video-ranges",
        help="Optimize and annotate ArUco detections on frame ranges listed in a video manifest CSV.",
    )
    _add_common_config_arg(optimize_video_ranges_parser)
    optimize_video_ranges_parser.add_argument(
        "--manifest",
        required=True,
        help="CSV containing video_id and frame_ranges columns.",
    )
    optimize_video_ranges_parser.add_argument(
        "--source-root",
        required=True,
        help="Project/source root containing input_data videos and, when available, frames/<video_id> images.",
    )
    optimize_video_ranges_parser.add_argument(
        "--output-root",
        required=True,
        help="Output folder for copied videos, extracted range frames, optimization results, and annotations.",
    )
    optimize_video_ranges_parser.add_argument(
        "--tag-list-root",
        help="Root folder containing MC tag-list files such as tag_list_mc7_mc8.txt.",
    )
    optimize_video_ranges_parser.add_argument(
        "--tag-bounds-json",
        help="Optional per-video smallest/largest tag bounds JSON saved from the GUI.",
    )
    optimize_video_ranges_parser.add_argument(
        "--open-gui",
        action="store_true",
        help=(
            "Open the GUI prefilled to the video-range smallest/largest tag-bounds workflow, "
            "then exit without optimizing. Rerun without this flag after saving bounds."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--profile",
        choices=["quick", "balanced", "deep", "daily"],
        default="daily",
        help="Grid profile size (default: daily).",
    )
    optimize_video_ranges_parser.add_argument(
        "--dictionary",
        default="auto",
        help=(
            "ArUco dictionary, or 'auto'. Auto skips Aug-2019 and 2024 no-tag rows by default, "
            "uses 4X4_50 for 2021, and uses 4X4_100 for 2026 BumbleBox videos."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--dictionary-candidates",
        default="4X4_50,4X4_100",
        help="Comma-separated dictionaries tested when --dictionary auto cannot decide uniquely, especially 2024 rows.",
    )
    optimize_video_ranges_parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=2.5,
        help="Physical ArUco tag size in millimeters (default 2.5).",
    )
    optimize_video_ranges_parser.add_argument(
        "--sample-frames",
        type=int,
        help="Frames sampled from the extracted range for optimization. Defaults to all requested frames.",
    )
    optimize_video_ranges_parser.add_argument(
        "--expected-tags",
        type=float,
        help="Optional expected visible tag count for each optimized frame to guide scoring.",
    )
    optimize_video_ranges_parser.add_argument(
        "--max-combinations",
        type=int,
        default=750,
        help="Optional cap on parameter combinations per dictionary/video (default: 750).",
    )
    optimize_video_ranges_parser.add_argument(
        "--execution-target",
        choices=["pi_safe", "desktop"],
        default="desktop",
        help="pi_safe uses conservative worker defaults. desktop uses more cores.",
    )
    optimize_video_ranges_parser.add_argument("--workers", type=int, help="Optional explicit worker count override.")
    optimize_video_ranges_parser.add_argument(
        "--sweep-min-marker-perimeter-rate",
        default="",
        help="Optional comma-separated override values for minMarkerPerimeterRate.",
    )
    optimize_video_ranges_parser.add_argument(
        "--sweep-max-marker-perimeter-rate",
        default="",
        help="Optional comma-separated override values for maxMarkerPerimeterRate.",
    )
    optimize_video_ranges_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-min",
        default="",
        help="Optional comma-separated override values for adaptiveThreshWinSizeMin.",
    )
    optimize_video_ranges_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-max",
        default="",
        help="Optional comma-separated override values for adaptiveThreshWinSizeMax.",
    )
    optimize_video_ranges_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-step",
        default="",
        help="Optional comma-separated override values for adaptiveThreshWinSizeStep.",
    )
    optimize_video_ranges_parser.add_argument(
        "--sweep-polygonal-approx-accuracy-rate",
        default="",
        help="Optional comma-separated override values for polygonalApproxAccuracyRate.",
    )
    optimize_video_ranges_parser.add_argument(
        "--sweep-adaptive-thresh-constant",
        default="",
        help="Optional comma-separated override values for adaptiveThreshConstant.",
    )
    optimize_video_ranges_parser.add_argument(
        "--allowed-tag-ids",
        default="",
        help="Optional comma-separated colony allowlist IDs, combined with auto-resolved --tag-list-root files.",
    )
    optimize_video_ranges_parser.add_argument(
        "--exclude-tag-ids",
        default="",
        help=(
            "Optional comma-separated marker IDs to ignore even when they decode. "
            "Ignored IDs do not count during optimization or final detection."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--exclude-tag-ids-except-video-prefixes",
        default="",
        help=(
            "Optional comma-separated video_id prefixes where --exclude-tag-ids should not apply, "
            "for example col_40."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--no-tag-list-filter",
        action="store_true",
        help=(
            "Decode and size-filter tags without requiring IDs to appear in --allowed-tag-ids "
            "or the auto-resolved --tag-list-root file. Useful for diagnosing tag-list mismatches."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--include-aug-2019",
        action="store_true",
        help="Do not skip Aug-2019 rows. By default those rows are skipped because they do not have ArUco tags.",
    )
    optimize_video_ranges_parser.add_argument(
        "--include-2024",
        action="store_true",
        help="Do not skip 2024 rows. By default those rows are skipped because this manifest's 2024 video has no ArUco tags.",
    )
    optimize_video_ranges_parser.add_argument(
        "--write-annotated-video",
        action="store_true",
        help="Also stitch optimized-frame annotated PNGs into a short MP4 review video for each row.",
    )
    optimize_video_ranges_parser.add_argument(
        "--no-candidate-review",
        action="store_true",
        help=(
            "Do not write per-candidate top_candidate_review PNGs during optimization. "
            "This greatly reduces file counts for Dropbox-synced batch runs."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--annotated-video-fps",
        type=float,
        default=2.0,
        help="FPS for --write-annotated-video review clips (default: 2).",
    )
    optimize_video_ranges_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite copied videos and extracted frames if they already exist.",
    )
    optimize_video_ranges_parser.add_argument(
        "--replace-optimization-runs",
        action="store_true",
        help=(
            "Remove each video's existing optimization/ folder before running, "
            "so old timestamped optimize_tracking_* runs are not kept."
        ),
    )
    optimize_video_ranges_parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N manifest rows after reading the CSV.",
    )
    optimize_video_ranges_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve videos/tag lists/dictionaries and print the plan without copying or optimizing.",
    )
    optimize_video_ranges_parser.add_argument(
        "--no-live-progress",
        action="store_true",
        help="Suppress live per-candidate optimizer progress lines.",
    )
    optimize_video_ranges_parser.set_defaults(func=_cmd_optimize_video_ranges)

    optimize_images_parser = subparsers.add_parser(
        "optimize-image-folder",
        help="Run a separate ArUco optimization for each PNG/image in a folder.",
    )
    optimize_images_parser.add_argument(
        "--input",
        required=True,
        help="Folder containing PNG/image files to optimize one image at a time.",
    )
    optimize_images_parser.add_argument(
        "--output-root",
        required=True,
        help="Output folder for per-image optimizer runs and the summary CSV.",
    )
    optimize_images_parser.add_argument(
        "--profile",
        choices=["quick", "balanced", "deep", "daily"],
        default="daily",
        help="Grid profile size (default: daily).",
    )
    optimize_images_parser.add_argument(
        "--dictionary",
        default="4X4_50",
        help="ArUco dictionary name (for example 4X4_50 or DICT_4X4_50).",
    )
    optimize_images_parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=2.5,
        help="Physical ArUco tag size in millimeters (default 2.5).",
    )
    optimize_images_parser.add_argument(
        "--sweep-min-marker-perimeter-rate",
        default="",
        help="Optional comma-separated override values for minMarkerPerimeterRate.",
    )
    optimize_images_parser.add_argument(
        "--sweep-max-marker-perimeter-rate",
        default="",
        help="Optional comma-separated override values for maxMarkerPerimeterRate.",
    )
    optimize_images_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-min",
        default="",
        help="Optional comma-separated override values for adaptiveThreshWinSizeMin.",
    )
    optimize_images_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-max",
        default="",
        help="Optional comma-separated override values for adaptiveThreshWinSizeMax.",
    )
    optimize_images_parser.add_argument(
        "--sweep-adaptive-thresh-win-size-step",
        default="",
        help="Optional comma-separated override values for adaptiveThreshWinSizeStep.",
    )
    optimize_images_parser.add_argument(
        "--sweep-polygonal-approx-accuracy-rate",
        default="",
        help="Optional comma-separated override values for polygonalApproxAccuracyRate.",
    )
    optimize_images_parser.add_argument(
        "--sweep-adaptive-thresh-constant",
        default="",
        help="Optional comma-separated override values for adaptiveThreshConstant.",
    )
    optimize_images_parser.add_argument(
        "--execution-target",
        choices=["pi_safe", "desktop"],
        default="pi_safe",
        help="pi_safe uses conservative worker defaults. desktop uses more cores.",
    )
    optimize_images_parser.add_argument(
        "--max-combinations",
        type=int,
        default=750,
        help="Optional cap on parameter combinations per image (default: 750).",
    )
    optimize_images_parser.add_argument("--workers", type=int, help="Optional explicit worker count override.")
    optimize_images_parser.add_argument(
        "--expected-tags",
        type=float,
        help="Optional expected visible tag count for each image to guide scoring.",
    )
    optimize_images_parser.add_argument(
        "--tag-list",
        help=(
            "Optional colony allowlist file. Supports JSON list/object, CSV-ish text, or newline-separated IDs. "
            "Optimizer mean detections count only decoded tags inside this list."
        ),
    )
    optimize_images_parser.add_argument(
        "--allowed-tag-ids",
        default="",
        help="Optional comma-separated colony allowlist IDs for optimizer scoring.",
    )
    optimize_images_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop each image after this many non-improving evaluations (0 disables; default 0).",
    )
    optimize_images_parser.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=0.0,
        help="Minimum score increase considered an improvement for early stop.",
    )
    optimize_images_parser.add_argument("--top-k", type=int, default=5, help="How many top candidates to retain.")
    optimize_images_parser.add_argument(
        "--limit",
        type=int,
        help="Only optimize the first N sorted images. Useful for timing pilots.",
    )
    optimize_images_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the input directory itself, not nested folders.",
    )
    optimize_images_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun images even if an image_optimization_complete.json marker already exists.",
    )
    _add_common_config_arg(optimize_images_parser)
    optimize_images_parser.set_defaults(func=_cmd_optimize_image_folder)

    schedule_check_parser = subparsers.add_parser(
        "schedule-check",
        help="Estimate whether recording/tracking schedule fits available hardware resources.",
    )
    _add_common_config_arg(schedule_check_parser)
    schedule_check_parser.add_argument(
        "--benchmark-input",
        help="Optional representative video or image folder for empirical tracking-speed benchmark.",
    )
    schedule_check_parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=80,
        help="Frame/image count used when --benchmark-input is provided.",
    )
    schedule_check_parser.add_argument(
        "--assume-ram-gb",
        type=float,
        help=(
            "Optional RAM size (GiB) to simulate target hardware. "
            "Example: --assume-ram-gb 2 for a Pi 4B 2GB setup."
        ),
    )
    schedule_check_parser.set_defaults(func=_cmd_schedule_check)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
