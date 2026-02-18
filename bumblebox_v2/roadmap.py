from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _status(done: bool) -> str:
    return "DONE" if done else "TODO"


def build_roadmap(config: Dict[str, Any], config_path: str | Path) -> List[Tuple[str, str]]:
    config_path = Path(config_path)
    items: List[Tuple[str, str]] = []

    items.append(
        (
            _status(config_path.exists()),
            f"Create or verify config file at {config_path}",
        )
    )

    items.append(
        (
            "TODO",
            "Run camera bring-up checks: 'bbx camera-preview --seconds 20' then 'bbx camera-test-tracking --seconds 20'.",
        )
    )

    calibration = config.get("calibration", {})
    pixels_per_cm = calibration.get("pixels_per_cm", 0)
    method = str(calibration.get("method", "")).strip().lower()
    last_updated = calibration.get("last_updated")
    calibration_ready = (
        isinstance(pixels_per_cm, (int, float))
        and pixels_per_cm > 0
        and method in {"manual_points", "aruco_marker"}
        and bool(last_updated)
    )
    items.append(
        (
            _status(calibration_ready),
            "Run scale calibration with a larger baseline (recommended >=5 cm) and verify px/cm conversion.",
        )
    )

    mode = config["pipeline"]["mode"]
    items.append(
        (
            "INFO",
            f"Pipeline mode is '{mode}'. Supported modes: record_only, track_only, record_and_track, mixed_schedule.",
        )
    )

    defer_tracking = config["pipeline"]["defer_tracking_until_after_recording"]
    items.append(
        (
            "INFO",
            (
                "Deferred tracking is ON (maximizes recording availability by running tracking after capture)."
                if defer_tracking
                else "Deferred tracking is OFF (tracking can compete with recording time)."
            ),
        )
    )

    fps_reporting_on = config.get("runtime", {}).get("fps_report_on_each_recording", False)
    items.append(
        (
            _status(bool(fps_reporting_on)),
            "Enable MP4 FPS reporting per recording and review drift after each run.",
        )
    )
    items.append(
        (
            "INFO",
            (
                "Optional: run 'bbx fps-sweep' to probe increasing target FPS values and estimate "
                "max recording duration (plus tracking-time estimates when recent tracking exists)."
            ),
        )
    )

    items.append(
        (
            "INFO",
            (
                "Optional: tune ArUco parameters with "
                "'bbx optimize-tracking --input <video_or_folder> --execution-target pi_safe|desktop'."
            ),
        )
    )
    items.append(
        (
            "INFO",
            (
                "Run 'bbx schedule-check' before deployment to verify RAM and timing margins; "
                "use --benchmark-input for hardware-specific estimates."
            ),
        )
    )
    items.append(
        (
            "INFO",
            (
                "For downstream desktop analysis, package runs with "
                "'bbx export-bundle --latest' (or --summary <run_summary.json>)."
            ),
        )
    )
    items.append(
        (
            "INFO",
            "Optional: run 'bbx gui-install-shortcut' to create a clickable Desktop icon for the GUI.",
        )
    )

    fleet = config.get("fleet", {}) if isinstance(config.get("fleet", {}), dict) else {}
    fleet_role = str(fleet.get("role", "standalone")).strip().lower()
    if fleet_role == "queen":
        workers = fleet.get("workers", []) if isinstance(fleet.get("workers", []), list) else []
        local_pipeline = bool(fleet.get("queen_local_pipeline_enabled", True))
        items.append(
            (
                "INFO",
                f"Fleet role is queen with {len(workers)} worker(s). "
                + ("Queen local pipeline is enabled." if local_pipeline else "Queen is interface-only (local pipeline disabled)."),
            )
        )
        items.append(
            (
                "INFO",
                "Run 'bbx fleet status' to check worker health and clock offsets.",
            )
        )
        items.append(
            (
                "INFO",
                (
                    "Optional: run 'bbx fleet queen-pull-latest' after worker recording cycles, "
                    "and 'bbx fleet queen-track-latest --cooldown-minutes 60' hourly "
                    "to maintain separate latest_video and latest_tracked views per worker."
                ),
            )
        )
    elif fleet_role == "worker":
        items.append(
            (
                "INFO",
                "Fleet role is worker. Ensure chrony points to queen and SSH trust is configured.",
            )
        )
    else:
        items.append(
            (
                "INFO",
                "Optional: use 'bbx fleet init-queen' and 'bbx fleet enroll-worker' for multi-box orchestration.",
            )
        )

    nest_script = Path(__file__).resolve().parent.parent / "LabelNests_GUI.1.16.py"
    items.append(
        (
            _status(nest_script.exists()),
            (
                "Nest labeling tool ready. Use 'bbx nest-label check --folder <composite_images>' "
                "then 'bbx nest-label launch --folder <composite_images>'."
                if nest_script.exists()
                else f"Nest labeling script missing at {nest_script}."
            ),
        )
    )

    schedule_enabled = config["scheduling"]["enabled"]
    schedule_backend = config["scheduling"]["backend"]
    items.append(
        (
            _status(bool(schedule_enabled)),
            f"Install and enable scheduled runs using {schedule_backend}.",
        )
    )

    items.append(
        (
            "TODO",
            "Run one 24-hour validation cycle and inspect recording uptime, tag yield, and FPS drift reports.",
        )
    )

    return items


def render_roadmap(config: Dict[str, Any], config_path: str | Path) -> str:
    lines = ["BumbleBox V2 Roadmap", "==================="]
    for index, (state, text) in enumerate(build_roadmap(config, config_path), start=1):
        lines.append(f"{index}. [{state}] {text}")
    return "\n".join(lines)
