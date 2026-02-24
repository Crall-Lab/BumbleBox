from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _status(done: bool) -> str:
    return "DONE" if done else "TODO"


def _optional_status(done: bool) -> str:
    return "OPTIONAL_DONE" if done else "OPTIONAL_TODO"


def build_roadmap(config: Dict[str, Any], config_path: str | Path) -> List[Tuple[str, str]]:
    config_path = Path(config_path)
    items: List[Tuple[str, str]] = []

    items.append(
        (
            _status(config_path.exists()),
            f"Tab: Config Editor. Create or verify the config file at {config_path}.",
        )
    )

    codec = str(config.get("camera", {}).get("codec", "mp4")).strip().lower()
    items.append(
        (
            _status(codec == "mp4"),
            (
                "Tab: Config Editor. Choose recording codec in 'Codec' "
                "(default: MP4; alternative: MJPEG)."
            ),
        )
    )

    mode = config["pipeline"]["mode"]
    mode_is_default = str(mode).strip().lower() == "record_and_track"
    items.append(
        (
            _optional_status(mode_is_default),
            (
                f"Tab: Config Editor. Pipeline mode is '{mode}'. "
                "Recommended default is 'record_and_track'; change this in Pipeline mode if needed."
            ),
        )
    )

    defer_tracking = config["pipeline"]["defer_tracking_until_after_recording"]
    items.append(
        (
            _optional_status(bool(defer_tracking)),
            (
                "Tab: Config Editor. Deferred tracking is ON. Keep this ON when maximizing recording time is a priority."
                if defer_tracking
                else "Tab: Config Editor. Deferred tracking is OFF. Turn this ON to reduce recording interruptions."
            ),
        )
    )

    fps_reporting_on = config.get("runtime", {}).get("fps_report_on_each_recording", False)
    if codec != "mp4":
        mp4_step_status = "DISABLED"
    else:
        mp4_step_status = _status(bool(fps_reporting_on))
    items.append(
        (
            mp4_step_status,
            (
                "Tab: Config Editor. For MP4 recording, keep 'FPS report each recording' enabled "
                "to track real framerate per recording."
            ),
        )
    )

    items.append(
        (
            "TODO",
            "Tab: Camera Setup. Check Camera by running preview and live tracking to confirm focus, exposure, and tag visibility.",
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
            "Tab: Calibration. Run scale calibration with a larger baseline (recommended >=5 cm) and verify px/cm conversion.",
        )
    )

    items.append(
        (
            "TODO",
            "Tab: Schedule Check. Run schedule validation to confirm RAM and timing margins for your hardware and plan.",
        )
    )

    schedule_enabled = config["scheduling"]["enabled"]
    schedule_backend = config["scheduling"]["backend"]
    items.append(
        (
            _status(bool(schedule_enabled)),
            f"Tab: Schedule and Run. Install and enable scheduled runs using {schedule_backend}.",
        )
    )

    items.append(
        (
            "OPTIONAL_TODO",
            (
                "Tab: FPS Report. Run FPS Sweep to probe increasing target FPS values and estimate "
                "max recording duration (and tracking-time estimates when available)."
            ),
        )
    )

    items.append(
        (
            "OPTIONAL_TODO",
            (
                "Tab: Test Tracking. Tune ArUco parameters for your setup and apply the best values to config if needed."
            ),
        )
    )

    items.append(
        (
            "OPTIONAL_TODO",
            "Tab: Schedule and Run. Run one 24-hour validation cycle and inspect recording uptime, tag yield, and FPS drift reports.",
        )
    )

    return items


def render_roadmap(config: Dict[str, Any], config_path: str | Path) -> str:
    lines = ["BumbleBox V2 Roadmap", "==================="]
    for index, (state, text) in enumerate(build_roadmap(config, config_path), start=1):
        lines.append(f"{index}. [{state}] {text}")
    return "\n".join(lines)
