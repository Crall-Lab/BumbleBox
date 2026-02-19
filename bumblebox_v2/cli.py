from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Optional

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
from .run_engine import format_run_summary, run_once
from .schedule_check import format_schedule_check_report, run_schedule_check
from .storage_manager import (
    build_storage_setup_sudo_command,
    format_storage_setup_result,
    format_storage_status_report,
    get_storage_status,
    setup_storage_auto_mount,
)
from .systemd_units import (
    format_systemd_action_result,
    format_systemd_result,
    run_systemd_action,
    write_systemd_units,
)


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


def _cmd_camera_preview(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = _load_or_defaults(config_path)
    except (FileNotFoundError, ConfigError, RuntimeError) as exc:
        print(f"Config error: {exc}")
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

    try:
        result = optimize_tracking(
            input_path=args.input,
            profile=args.profile,
            sample_frames=args.sample_frames,
            dictionary_name=args.dictionary,
            tag_size_mm=args.tag_size_mm,
            execution_target=args.execution_target,
            workers=args.workers,
            expected_tags=args.expected_tags,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_improvement=args.early_stop_min_improvement,
            output_dir=args.output_dir,
            write_preview=args.write_preview,
            preview_frames=args.preview_frames,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Tracking optimization failed: {exc}")
        return 1

    print(format_optimization_report(result, top_k=args.top_k))

    if args.apply_best:
        config_path = Path(args.config)
        try:
            config = _load_or_defaults(config_path)
            updated = apply_best_params_to_config(config, result.best_params)
            save_config(config_path, updated)
            print(f"Applied best parameters to config: {config_path}")
        except Exception as exc:
            print(f"Optimization completed, but failed to apply config update: {exc}")
            return 1

    return 0


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
        default="QTGL",
        help="Picamera2 preview backend window type.",
    )
    camera_preview_parser.add_argument("--width", type=int, help="Optional preview width override in pixels.")
    camera_preview_parser.add_argument("--height", type=int, help="Optional preview height override in pixels.")
    camera_preview_parser.set_defaults(func=_cmd_camera_preview)

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

    fps_parser = subparsers.add_parser("fps-report", help="Build FPS quality report for an MP4 recording.")
    fps_parser.add_argument("--video", required=True, help="Path to recorded video (.mp4/.mjpeg).")
    fps_parser.add_argument("--timestamps", help="Optional path to frame timestamp sidecar file.")
    fps_parser.add_argument("--recording-seconds", type=float, help="Expected recording duration in seconds.")
    fps_parser.add_argument("--json-out", help="Optional path to save JSON report.")
    fps_parser.set_defaults(func=_cmd_fps_report)

    fps_sweep_parser = subparsers.add_parser(
        "fps-sweep",
        help=(
            "Probe target FPS values and estimate max recording durations at safe/warn/high-risk RAM levels. "
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
        help="Optional RAM size (GiB) to simulate target hardware during duration estimation.",
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
    run_once_parser.set_defaults(func=_cmd_run_once)

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
    fleet_init.add_argument("--ssh-user", help="Default SSH user for workers (default from config or 'pi').")
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
        help="Exclude session MP4 from bundle (smaller transfer size).",
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
            "The launcher script prefers repo .venv Python when available, then falls back to system python3."
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
        help="Use an existing icon file path. If omitted, a default BumbleBox SVG icon is created if missing.",
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
        help="Path to input video (.mp4/.mjpeg/...) or image directory.",
    )
    optimize_parser.add_argument(
        "--profile",
        choices=["quick", "balanced", "deep"],
        default="quick",
        help="Grid profile size (quick is fastest).",
    )
    optimize_parser.add_argument(
        "--sample-frames",
        type=int,
        default=80,
        help="How many sampled frames/images to evaluate.",
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
        "--execution-target",
        choices=["pi_safe", "desktop"],
        default="pi_safe",
        help="pi_safe uses conservative worker defaults. desktop uses more cores.",
    )
    optimize_parser.add_argument("--workers", type=int, help="Optional explicit worker count override.")
    optimize_parser.add_argument(
        "--expected-tags",
        type=float,
        help="Optional expected average visible tag count per frame to guide scoring.",
    )
    optimize_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=40,
        help="Stop after this many non-improving evaluations (0 disables).",
    )
    optimize_parser.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=0.002,
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
