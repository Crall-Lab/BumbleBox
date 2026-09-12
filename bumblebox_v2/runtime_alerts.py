from __future__ import annotations

from datetime import datetime
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
    for suffix in ("*.mp4", "*.mjpeg", "*.avi"):
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
        if record.mode not in RECORDING_MODES:
            continue
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


def _record_timestamp_iso(iso_text: str) -> Optional[float]:
    raw = str(iso_text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _latest_recording_attempt(data_root: Path, limit: int = 40) -> Optional[Dict[str, Any]]:
    records = list_recent_run_records(data_root, limit=limit)
    for record in records:
        if record.mode not in RECORDING_MODES:
            continue
        try:
            summary = load_run_summary(record.summary_path)
        except Exception:
            continue

        attempt_ts = (
            _record_timestamp_iso(record.finished_at)
            or _record_timestamp_iso(record.started_at)
        )
        if attempt_ts is None:
            try:
                attempt_ts = record.summary_path.stat().st_mtime
            except Exception:
                attempt_ts = time.time()

        errors = [str(item).strip() for item in (summary.get("errors") or []) if str(item).strip()]
        warnings = [str(item).strip() for item in (summary.get("warnings") or []) if str(item).strip()]
        return {
            "record": record,
            "summary": summary,
            "timestamp": attempt_ts,
            "first_error": errors[0] if errors else None,
            "first_warning": warnings[0] if warnings else None,
        }
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

    latest_attempt = _latest_recording_attempt(data_root)
    latest_recorded_video = _latest_recorded_video_from_run_history(data_root)
    used_fallback_scan = False
    if latest_recorded_video is None:
        latest_recorded_video = _latest_recording_video(data_root)
        used_fallback_scan = latest_recorded_video is not None

    capture = config.get("capture", {})
    if not isinstance(capture, dict):
        capture = {}
    record_interval_min = max(1.0, float(capture.get("record_interval_minutes", 30)))
    recording_seconds = max(1.0, float(capture.get("recording_seconds", 20)))
    expected_gap = max(record_interval_min * 2.0, record_interval_min + (recording_seconds / 60.0) + 2.0)
    now_ts = time.time()

    latest_attempt_age_minutes: Optional[float] = None
    if latest_attempt is not None:
        latest_attempt_age_minutes = max(0.0, (now_ts - float(latest_attempt["timestamp"])) / 60.0)

    if latest_recorded_video is None:
        if latest_attempt is not None:
            session_name = str(latest_attempt["record"].session_name)
            error_text = str(latest_attempt.get("first_error") or "").strip()
            if latest_attempt["record"].success:
                return RuntimeAlert(
                    "WARN",
                    "Recording freshness",
                    (
                        f"Latest scheduled recording attempt was {latest_attempt_age_minutes:.1f} minutes ago "
                        f"({session_name}), but no recording video was found under {data_root}. "
                        "Check output path and cleanup settings."
                    ),
                )
            detail = f" First error: {error_text}" if error_text else ""
            return RuntimeAlert(
                "WARN",
                "Recording freshness",
                (
                    f"Latest scheduled recording attempt was {latest_attempt_age_minutes:.1f} minutes ago "
                    f"({session_name}), and it failed before producing a video.{detail}"
                ),
            )
        return RuntimeAlert(
            "WARN",
            "Recording freshness",
            f"No recording videos (.mp4, .mjpeg, or .avi) found under {data_root} while scheduling is enabled.",
        )

    try:
        latest_video_mtime = latest_recorded_video.stat().st_mtime
        age_minutes = (now_ts - latest_video_mtime) / 60.0
    except Exception as exc:
        return RuntimeAlert(
            "WARN",
            "Recording freshness",
            f"Could not stat latest recording video ({latest_recorded_video}): {exc}",
        )

    if (
        latest_attempt is not None
        and latest_attempt_age_minutes is not None
        and latest_attempt["timestamp"] > latest_video_mtime
        and not latest_attempt["record"].success
    ):
        session_name = str(latest_attempt["record"].session_name)
        error_text = str(latest_attempt.get("first_error") or "").strip()
        detail = f" First error: {error_text}" if error_text else ""
        return RuntimeAlert(
            "WARN",
            "Recording freshness",
            (
                f"Latest successful recording video is {age_minutes:.1f} minutes old ({latest_recorded_video.name}), "
                f"but the most recent scheduled recording attempt was {latest_attempt_age_minutes:.1f} minutes ago "
                f"({session_name}) and failed before video output.{detail}"
            ),
        )

    if age_minutes > expected_gap:
        source_note = " (fallback filesystem scan)" if used_fallback_scan else ""
        if latest_attempt is not None and latest_attempt_age_minutes is not None:
            session_name = str(latest_attempt["record"].session_name)
            if not latest_attempt["record"].success:
                error_text = str(latest_attempt.get("first_error") or "").strip()
                detail = f" First error: {error_text}" if error_text else ""
                return RuntimeAlert(
                    "WARN",
                    "Recording freshness",
                    (
                        f"Latest recording video is {age_minutes:.1f} minutes old ({latest_recorded_video.name}){source_note}, "
                        f"but the scheduler did run {latest_attempt_age_minutes:.1f} minutes ago ({session_name}) and that "
                        f"recording attempt failed before producing a video.{detail}"
                    ),
                )
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
