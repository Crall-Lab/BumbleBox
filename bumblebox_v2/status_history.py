from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunRecord:
    summary_path: Path
    session_name: str
    started_at: str
    finished_at: str
    mode: str
    actual_fps: float
    success: bool
    warning_count: int
    error_count: int
    # "" for the first camera; "cam2" for a second camera on the same Pi.
    camera_session_tag: str = ""


def _parse_summary(path: Path) -> Optional[RunRecord]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None

    try:
        return RunRecord(
            summary_path=path,
            session_name=str(payload.get("session_name", path.stem)),
            started_at=str(payload.get("started_at", "")),
            finished_at=str(payload.get("finished_at", "")),
            mode=str(payload.get("mode", "")),
            actual_fps=float(payload.get("actual_fps", 0.0)),
            success=bool(payload.get("success", False)),
            warning_count=len(payload.get("warnings", []) or []),
            error_count=len(payload.get("errors", []) or []),
            camera_session_tag=str(payload.get("camera_session_tag") or ""),
        )
    except Exception:
        return None


def list_recent_run_records(
    data_root: str | Path,
    limit: int = 30,
    *,
    session_tag: Optional[str] = None,
) -> List[RunRecord]:
    """Newest runs under ``data_root``.

    Two cameras on one Pi can share a data folder. Pass ``session_tag`` (``""``
    for the first camera) to get one camera's runs only; ``None`` returns all.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        return []

    records: List[RunRecord] = []
    for path in data_root.rglob("*_run_summary.json"):
        record = _parse_summary(path)
        if record is not None and (session_tag is None or record.camera_session_tag == session_tag):
            records.append(record)

    def _sort_key(record: RunRecord):
        for candidate in (record.started_at, record.finished_at):
            if candidate:
                try:
                    return datetime.fromisoformat(candidate)
                except ValueError:
                    continue
        try:
            return datetime.fromtimestamp(record.summary_path.stat().st_mtime)
        except Exception:
            return datetime.min

    records.sort(key=_sort_key, reverse=True)
    return records[: max(1, int(limit))]


def load_run_summary(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text())

