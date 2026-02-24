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

    mode = config["pipeline"]["mode"]
    items.append(
        (
            "OPTIONAL",
            (
                f"Pipeline mode is set to '{mode}'. "
                "Optional: adjust in Config Editor if your experiment design has changed."
            ),
        )
    )

    defer_tracking = config["pipeline"]["defer_tracking_until_after_recording"]
    items.append(
        (
            "OPTIONAL",
            (
                "Deferred tracking is currently ON. Optional: keep this ON if maximizing recording time is your priority."
                if defer_tracking
                else "Deferred tracking is currently OFF. Optional: turn this ON to reduce recording interruptions."
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

    items.append(
        (
            "OPTIONAL",
            (
                "Optional: run 'bbx fps-sweep' to probe increasing target FPS values and estimate "
                "max recording duration (plus tracking-time estimates when recent tracking exists)."
            ),
        )
    )

    items.append(
        (
            "OPTIONAL",
            (
                "Optional: tune ArUco parameters with "
                "'bbx optimize-tracking --input <video_or_folder> --execution-target pi_safe|desktop'."
            ),
        )
    )

    items.append(
        (
            "TODO",
            (
                "Run 'bbx schedule-check' before deployment to verify RAM and timing margins; "
                "use --benchmark-input for hardware-specific estimates."
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
            "OPTIONAL",
            "Optional: run one 24-hour validation cycle and inspect recording uptime, tag yield, and FPS drift reports.",
        )
    )

    return items


def render_roadmap(config: Dict[str, Any], config_path: str | Path) -> str:
    lines = ["BumbleBox V2 Roadmap", "==================="]
    for index, (state, text) in enumerate(build_roadmap(config, config_path), start=1):
        lines.append(f"{index}. [{state}] {text}")
    return "\n".join(lines)
