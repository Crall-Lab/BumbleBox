from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .status_history import list_recent_run_records, load_run_summary


RECORDING_MODES = {"record_only", "record_and_track", "mixed_schedule"}


@dataclass
class RuntimeAlert:
    severity: str
    title: str
    message: str


def _findmnt_target(path: Path) -> Optional[tuple[str, str]]:
    try:
        proc = subprocess.run(
            ["findmnt", "--noheadings", "--output", "SOURCE,TARGET", "--target", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _latest_recording_video(data_root: Path) -> Optional[Path]:
    if not data_root.exists():
        return None

    newest_path: Optional[Path] = None
    newest_mtime = -1.0
    for suffix in ("*.mp4", "*.mjpeg"):
        for path in data_root.rglob(suffix):
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = path
    return newest_path


def _latest_recorded_video_from_run_history(data_root: Path, limit: int = 40) -> Optional[Path]:
    records = list_recent_run_records(data_root, limit=limit)
    for record in records:
        try:
            summary = load_run_summary(record.summary_path)
        except Exception:
            continue
        video_raw = summary.get("video_path")
        if not video_raw:
            continue
        try:
            video_path = Path(str(video_raw))
        except Exception:
            continue
        if video_path.exists():
            return video_path
    return None


def _storage_alert(config: Dict[str, Any]) -> RuntimeAlert:
    raw = str(config.get("system", {}).get("data_root", "") or "").strip()
    if not raw:
        return RuntimeAlert("FAIL", "Data root", "system.data_root is not configured.")
    data_root = Path(raw).expanduser()

    if not data_root.exists():
        return RuntimeAlert("FAIL", "Data root", f"Data root does not exist: {data_root}")
    if not os.access(data_root, os.W_OK):
        return RuntimeAlert("FAIL", "Data root", f"Data root is not writable: {data_root}")

    mount = _findmnt_target(data_root)
    if mount is not None:
        source, target = mount
        if source.startswith("/dev/sd"):
            return RuntimeAlert(
                "WARN",
                "Storage mount source",
                (
                    f"Mounted at {target} from {source}. Device names like sda1/sdb1 can change across boots; "
                    "prefer UUID-based /etc/fstab entries for stable mounting."
                ),
            )
        return RuntimeAlert("PASS", "Storage mount source", f"Data root resolves to mount target {target} ({source}).")

    if str(data_root).startswith("/mnt/"):
        return RuntimeAlert(
            "WARN",
            "Storage mount source",
            (
                f"{data_root} is under /mnt but no active mount was detected via findmnt. "
                "Recording may be writing to SD root filesystem instead of external storage."
            ),
        )
    return RuntimeAlert("PASS", "Data root", f"Writable data root: {data_root}")


def _recording_freshness_alert(config: Dict[str, Any]) -> RuntimeAlert:
    pipeline_mode = str(config.get("pipeline", {}).get("mode", "")).strip().lower()
    scheduling_enabled = bool(config.get("scheduling", {}).get("enabled", False))
    fleet = config.get("fleet", {})
    if not isinstance(fleet, dict):
        fleet = {}

    if pipeline_mode not in RECORDING_MODES:
        return RuntimeAlert("INFO", "Recording freshness", f"Mode '{pipeline_mode}' does not schedule recordings.")
    if not scheduling_enabled:
        return RuntimeAlert("INFO", "Recording freshness", "Scheduling is disabled; skipping recording freshness warning.")
    if str(fleet.get("role", "standalone")).strip().lower() == "queen" and not bool(
        fleet.get("queen_local_pipeline_enabled", True)
    ):
        return RuntimeAlert(
            "INFO",
            "Recording freshness",
            "Queen is interface-only; local recording freshness check is skipped by design.",
        )

    data_root = Path(str(config.get("system", {}).get("data_root", ""))).expanduser()
    if not data_root.exists():
        return RuntimeAlert("WARN", "Recording freshness", f"No data root found at {data_root}.")

    latest_recorded_video = _latest_recorded_video_from_run_history(data_root)
    used_fallback_scan = False
    if latest_recorded_video is None:
        latest_recorded_video = _latest_recording_video(data_root)
        used_fallback_scan = latest_recorded_video is not None

    if latest_recorded_video is None:
        return RuntimeAlert(
            "WARN",
            "Recording freshness",
            f"No recording videos (.mp4 or .mjpeg) found under {data_root} while scheduling is enabled.",
        )

    try:
        age_minutes = (time.time() - latest_recorded_video.stat().st_mtime) / 60.0
    except Exception as exc:
        return RuntimeAlert(
            "WARN",
            "Recording freshness",
            f"Could not stat latest recording video ({latest_recorded_video}): {exc}",
        )

    capture = config.get("capture", {})
    if not isinstance(capture, dict):
        capture = {}
    record_interval_min = max(1.0, float(capture.get("record_interval_minutes", 30)))
    recording_seconds = max(1.0, float(capture.get("recording_seconds", 20)))
    expected_gap = max(record_interval_min * 2.0, record_interval_min + (recording_seconds / 60.0) + 2.0)

    if age_minutes > expected_gap:
        source_note = " (fallback filesystem scan)" if used_fallback_scan else ""
        return RuntimeAlert(
            "WARN",
            "Recording freshness",
            (
                f"Latest recording video is {age_minutes:.1f} minutes old ({latest_recorded_video.name}){source_note}, "
                f"which exceeds expected gap {expected_gap:.1f} minutes. "
                "Check camera, storage mount, and scheduler."
            ),
        )
    source_note = " (fallback filesystem scan)" if used_fallback_scan else ""
    return RuntimeAlert(
        "PASS",
        "Recording freshness",
        (
            f"Latest recording video age is {age_minutes:.1f} minutes "
            f"({latest_recorded_video.name}){source_note}, within expected schedule window."
        ),
    )


def build_runtime_alerts(config: Dict[str, Any]) -> List[RuntimeAlert]:
    alerts: List[RuntimeAlert] = []
    alerts.append(_storage_alert(config))
    alerts.append(_recording_freshness_alert(config))
    return alerts


def format_runtime_alerts(alerts: List[RuntimeAlert]) -> str:
    lines: List[str] = []
    for alert in alerts:
        lines.append(f"[{alert.severity}] {alert.title}: {alert.message}")
    warn_count = sum(1 for item in alerts if item.severity == "WARN")
    fail_count = sum(1 for item in alerts if item.severity == "FAIL")
    lines.append("")
    lines.append(f"Summary: {fail_count} fail, {warn_count} warn, {len(alerts)} total checks")
    return "\n".join(lines)
