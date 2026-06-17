from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_INDEX_PATH = REPO_ROOT / "LocalTrackingIndex"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class LocalIndexArtifact:
    kind: str
    source_path: str
    index_path: str
    size_bytes: int


@dataclass
class LocalIndexResult:
    enabled: bool
    index_root: str
    manifest_path: Optional[str]
    artifacts: list[LocalIndexArtifact]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def local_index_enabled(config: dict[str, Any]) -> bool:
    raw = config.get("local_index", {})
    if not isinstance(raw, dict):
        return True
    return bool(raw.get("enabled", True))


def local_index_root(config: dict[str, Any]) -> Path:
    raw = config.get("local_index", {})
    configured = ""
    if isinstance(raw, dict):
        configured = str(raw.get("path") or "").strip()
    path = Path(configured).expanduser() if configured else DEFAULT_TRACKING_INDEX_PATH
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def ensure_local_index_root(config: dict[str, Any]) -> Path:
    root = local_index_root(config)
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "BumbleBox Tracking Index",
                    "========================",
                    "",
                    "This folder is a lightweight local index of run, tracking, and optimization artifacts.",
                    "It intentionally avoids copying large raw videos.",
                    "",
                    "Expected contents:",
                    "- runs/YYYY-MM-DD/<session_name>/",
                    "- tracking/YYYY-MM-DD/<session_name>/",
                    "- optimization/YYYY-MM-DD/<optimization_run>/",
                    "",
                ]
            )
        )
    return root


def _copy_file(source: Path, destination: Path, kind: str) -> LocalIndexArtifact:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return LocalIndexArtifact(
        kind=kind,
        source_path=str(source),
        index_path=str(destination),
        size_bytes=int(destination.stat().st_size),
    )


def _copy_existing_file(
    artifacts: list[LocalIndexArtifact],
    warnings: list[str],
    source_raw: object,
    destination: Path,
    kind: str,
) -> None:
    source_text = str(source_raw or "").strip()
    if not source_text:
        return
    source = Path(source_text).expanduser()
    if not source.exists() or not source.is_file():
        warnings.append(f"Skipped missing {kind}: {source}")
        return
    try:
        artifacts.append(_copy_file(source, destination, kind))
    except Exception as exc:
        warnings.append(f"Could not copy {kind} to local index: {exc}")


def _date_from_text(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) >= 10 and DATE_RE.match(text[:10]):
        return text[:10]
    return None


def _date_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.expanduser().parts):
        if DATE_RE.match(part):
            return part
    return None


def _safe_session_name(summary_payload: dict[str, Any], summary_path: Path) -> str:
    raw = str(summary_payload.get("session_name") or "").strip()
    if raw:
        return raw
    return summary_path.stem.replace("_run_summary", "")


def _write_manifest(
    manifest_path: Path,
    *,
    kind: str,
    source_path: Path,
    index_root: Path,
    artifacts: list[LocalIndexArtifact],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "kind": kind,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": str(source_path),
        "index_root": str(index_root),
        "artifacts": [asdict(item) for item in artifacts],
    }
    if extra:
        payload.update(extra)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def sync_run_summary_file(
    config: dict[str, Any],
    summary_path: str | Path,
    *,
    summary_payload: Optional[dict[str, Any]] = None,
) -> LocalIndexResult:
    root = local_index_root(config)
    if not local_index_enabled(config):
        return LocalIndexResult(enabled=False, index_root=str(root), manifest_path=None, artifacts=[], warnings=[])

    warnings: list[str] = []
    artifacts: list[LocalIndexArtifact] = []
    summary = Path(summary_path).expanduser()
    if summary_payload is None:
        with summary.open() as f:
            summary_payload = json.load(f)

    root = ensure_local_index_root(config)
    session_name = _safe_session_name(summary_payload, summary)
    run_date = (
        _date_from_text(summary_payload.get("started_at"))
        or _date_from_path(summary)
        or datetime.now().strftime("%Y-%m-%d")
    )

    run_dir = root / "runs" / run_date / session_name
    tracking_dir = root / "tracking" / run_date / session_name

    copy_config = bool(config.get("local_index", {}).get("copy_config_snapshots", True))
    copy_fps = bool(config.get("local_index", {}).get("copy_fps_reports", True))
    copy_tracking = bool(config.get("local_index", {}).get("copy_tracking_csvs", True))
    copy_summary = bool(config.get("local_index", {}).get("copy_run_summaries", True))

    if copy_summary:
        _copy_existing_file(artifacts, warnings, summary, run_dir / f"{session_name}_run_summary.json", "run_summary_json")
    if copy_config:
        _copy_existing_file(
            artifacts,
            warnings,
            summary_payload.get("config_snapshot_path"),
            run_dir / f"{session_name}_config_snapshot.json",
            "config_snapshot_json",
        )
    if copy_fps:
        _copy_existing_file(
            artifacts,
            warnings,
            summary_payload.get("fps_report_json"),
            run_dir / f"{session_name}_fps_report.json",
            "fps_report_json",
        )
    if copy_tracking:
        for key, suffix, kind in [
            ("raw_csv_path", "raw.csv", "tracking_raw_csv"),
            ("noid_csv_path", "noID.csv", "tracking_noid_csv"),
            ("cleaned_csv_path", "cleaned.csv", "tracking_cleaned_csv"),
        ]:
            _copy_existing_file(
                artifacts,
                warnings,
                summary_payload.get(key),
                tracking_dir / f"{session_name}_{suffix}",
                kind,
            )

    manifest_path = run_dir / "index_manifest.json"
    _write_manifest(
        manifest_path,
        kind="run",
        source_path=summary,
        index_root=root,
        artifacts=artifacts,
        extra={
            "run_date": run_date,
            "session_name": session_name,
            "source_session_dir": summary_payload.get("session_dir"),
        },
    )
    (root / "latest_run_index.json").write_text(manifest_path.read_text())
    return LocalIndexResult(
        enabled=True,
        index_root=str(root),
        manifest_path=str(manifest_path),
        artifacts=artifacts,
        warnings=warnings,
    )


def _candidate_params(candidate: object) -> dict[str, Any]:
    if isinstance(candidate, dict):
        params = candidate.get("params", {})
    else:
        params = getattr(candidate, "params", {})
    return dict(params) if isinstance(params, dict) else {}


def _optimization_run_name(result: object, summary_path: Path) -> str:
    output_dir = str(getattr(result, "output_dir", "") or "").strip()
    if output_dir:
        return Path(output_dir).name
    return summary_path.parent.name or summary_path.stem


def sync_optimization_result(
    config: dict[str, Any],
    result: object,
    *,
    selected_params: Optional[dict[str, Any]] = None,
    selected_label: str = "selected",
) -> LocalIndexResult:
    root = local_index_root(config)
    if not local_index_enabled(config):
        return LocalIndexResult(enabled=False, index_root=str(root), manifest_path=None, artifacts=[], warnings=[])
    if not bool(config.get("local_index", {}).get("copy_optimization_results", True)):
        return LocalIndexResult(enabled=False, index_root=str(root), manifest_path=None, artifacts=[], warnings=[])

    warnings: list[str] = []
    artifacts: list[LocalIndexArtifact] = []
    summary_path = Path(str(getattr(result, "summary_json_path", ""))).expanduser()
    if not summary_path.exists():
        raise FileNotFoundError(f"Optimization summary not found: {summary_path}")

    root = ensure_local_index_root(config)
    input_path = Path(str(getattr(result, "input_path", ""))).expanduser()
    run_date = (
        _date_from_path(input_path)
        or _date_from_text(getattr(result, "created_at", None))
        or _date_from_path(summary_path)
        or datetime.now().strftime("%Y-%m-%d")
    )
    run_name = _optimization_run_name(result, summary_path)
    opt_dir = root / "optimization" / run_date / run_name

    _copy_existing_file(
        artifacts,
        warnings,
        summary_path,
        opt_dir / "optimization_summary.json",
        "optimization_summary_json",
    )
    _copy_existing_file(
        artifacts,
        warnings,
        getattr(result, "candidates_csv_path", None),
        opt_dir / "candidate_scores.csv",
        "candidate_scores_csv",
    )
    _copy_existing_file(
        artifacts,
        warnings,
        getattr(result, "review_manifest_json_path", None),
        opt_dir / "review_manifest.json",
        "optimization_review_manifest_json",
    )

    best_params = _candidate_params(getattr(result, "top_candidates", [None])[0]) if getattr(result, "top_candidates", None) else {}
    if best_params:
        best_path = opt_dir / "best_score_params.json"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(json.dumps(best_params, indent=2, sort_keys=True))
        artifacts.append(
            LocalIndexArtifact(
                kind="best_score_params_json",
                source_path="derived from optimization result",
                index_path=str(best_path),
                size_bytes=int(best_path.stat().st_size),
            )
        )

    top_detection = getattr(result, "top_detection_candidates", None) or []
    top_detection_params = _candidate_params(top_detection[0]) if top_detection else {}
    if top_detection_params:
        detection_path = opt_dir / "top_mean_detection_params.json"
        detection_path.parent.mkdir(parents=True, exist_ok=True)
        detection_path.write_text(json.dumps(top_detection_params, indent=2, sort_keys=True))
        artifacts.append(
            LocalIndexArtifact(
                kind="top_mean_detection_params_json",
                source_path="derived from optimization result",
                index_path=str(detection_path),
                size_bytes=int(detection_path.stat().st_size),
            )
        )

    if selected_params:
        selected_path = opt_dir / "selected_tracking_params.json"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_payload = {
            "selected_at": datetime.now().isoformat(timespec="seconds"),
            "selected_label": selected_label,
            "params": dict(selected_params),
            "source_optimization_summary": str(summary_path),
        }
        selected_path.write_text(json.dumps(selected_payload, indent=2, sort_keys=True))
        artifacts.append(
            LocalIndexArtifact(
                kind="selected_tracking_params_json",
                source_path="selected in GUI",
                index_path=str(selected_path),
                size_bytes=int(selected_path.stat().st_size),
            )
        )

    manifest_path = opt_dir / "index_manifest.json"
    _write_manifest(
        manifest_path,
        kind="optimization",
        source_path=summary_path,
        index_root=root,
        artifacts=artifacts,
        extra={
            "run_date": run_date,
            "optimization_run": run_name,
            "input_path": str(input_path) if str(input_path) else None,
            "profile": getattr(result, "profile", None),
            "dictionary": getattr(result, "dictionary", None),
        },
    )
    (root / "latest_optimization_index.json").write_text(manifest_path.read_text())
    return LocalIndexResult(
        enabled=True,
        index_root=str(root),
        manifest_path=str(manifest_path),
        artifacts=artifacts,
        warnings=warnings,
    )


def sync_posthoc_tracking_report(config: dict[str, Any], report: object) -> LocalIndexResult:
    root = local_index_root(config)
    if not local_index_enabled(config):
        return LocalIndexResult(enabled=False, index_root=str(root), manifest_path=None, artifacts=[], warnings=[])
    if not bool(config.get("local_index", {}).get("copy_tracking_csvs", True)):
        return LocalIndexResult(enabled=False, index_root=str(root), manifest_path=None, artifacts=[], warnings=[])

    warnings: list[str] = []
    artifacts: list[LocalIndexArtifact] = []
    root = ensure_local_index_root(config)
    report_path_text = str(getattr(report, "report_path", "") or "").strip()
    report_path = Path(report_path_text).expanduser() if report_path_text else None
    report_name = datetime.now().strftime("posthoc_%Y%m%d_%H%M%S")
    input_path = Path(str(getattr(report, "input_path", "") or ".")).expanduser()
    report_date = _date_from_path(input_path) or datetime.now().strftime("%Y-%m-%d")
    posthoc_dir = root / "posthoc" / report_date / report_name

    if report_path and report_path.exists():
        _copy_existing_file(
            artifacts,
            warnings,
            report_path,
            posthoc_dir / "posthoc_tracking_report.json",
            "posthoc_tracking_report_json",
        )

    for item in list(getattr(report, "results", []) or []):
        video_path = Path(str(getattr(item, "video_path", "") or "")).expanduser()
        run_date = _date_from_path(video_path) or report_date
        session_name = str(getattr(item, "session_name", "") or "").strip()
        if not session_name:
            session_name = video_path.stem or f"posthoc_{datetime.now().strftime('%H%M%S')}"
        tracking_dir = root / "tracking" / run_date / session_name
        for attr, suffix, kind in [
            ("raw_csv_path", "raw.csv", "tracking_raw_csv"),
            ("noid_csv_path", "noID.csv", "tracking_noid_csv"),
            ("cleaned_csv_path", "cleaned.csv", "tracking_cleaned_csv"),
        ]:
            _copy_existing_file(
                artifacts,
                warnings,
                getattr(item, attr, None),
                tracking_dir / f"{session_name}_{suffix}",
                kind,
            )

    if bool(config.get("local_index", {}).get("copy_optimization_results", True)):
        for item in list(getattr(report, "optimizations", []) or []):
            date_dir = Path(str(getattr(item, "date_dir", "") or "")).expanduser()
            run_date = _date_from_path(date_dir) or report_date
            summary_path = Path(str(getattr(item, "optimization_summary_path", "") or "")).expanduser()
            run_name = summary_path.parent.name if summary_path and str(summary_path) != "." else ""
            if not run_name:
                run_name = f"posthoc_tracking_{run_date}"
            opt_dir = root / "optimization" / run_date / run_name
            _copy_existing_file(
                artifacts,
                warnings,
                getattr(item, "selected_params_path", None),
                opt_dir / "selected_tracking_params.json",
                "selected_tracking_params_json",
            )
            _copy_existing_file(
                artifacts,
                warnings,
                getattr(item, "optimization_summary_path", None),
                opt_dir / "optimization_summary.json",
                "optimization_summary_json",
            )
            _copy_existing_file(
                artifacts,
                warnings,
                getattr(item, "candidate_scores_path", None),
                opt_dir / "candidate_scores.csv",
                "candidate_scores_csv",
            )

    source_path = report_path if report_path is not None else input_path
    manifest_path = posthoc_dir / "index_manifest.json"
    _write_manifest(
        manifest_path,
        kind="posthoc_tracking",
        source_path=source_path,
        index_root=root,
        artifacts=artifacts,
        extra={
            "run_date": report_date,
            "input_path": str(input_path),
            "videos_found": getattr(report, "videos_found", None),
            "videos_processed": getattr(report, "videos_processed", None),
            "videos_failed": getattr(report, "videos_failed", None),
        },
    )
    (root / "latest_posthoc_tracking_index.json").write_text(manifest_path.read_text())
    return LocalIndexResult(
        enabled=True,
        index_root=str(root),
        manifest_path=str(manifest_path),
        artifacts=artifacts,
        warnings=warnings,
    )


def format_local_index_result(result: LocalIndexResult) -> str:
    if not result.enabled:
        return f"Local tracking index is disabled. Index path: {result.index_root}"
    lines = [
        f"Local tracking index: {result.index_root}",
        f"Manifest: {result.manifest_path or 'none'}",
        f"Artifacts copied: {len(result.artifacts)}",
    ]
    for artifact in result.artifacts:
        lines.append(f"- {artifact.kind}: {artifact.index_path}")
    if result.warnings:
        lines.append("")
        lines.append("Index warnings:")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def summarize_index_results(results: Iterable[LocalIndexResult]) -> str:
    items = list(results)
    enabled_items = [item for item in items if item.enabled]
    artifact_count = sum(len(item.artifacts) for item in enabled_items)
    warnings = [warning for item in enabled_items for warning in item.warnings]
    root = enabled_items[0].index_root if enabled_items else (items[0].index_root if items else str(DEFAULT_TRACKING_INDEX_PATH))
    lines = [
        f"Local tracking index: {root}",
        f"Indexed records: {len(enabled_items)}",
        f"Artifacts copied: {artifact_count}",
    ]
    if warnings:
        lines.append("")
        lines.append("Index warnings:")
        for warning in warnings[:20]:
            lines.append(f"- {warning}")
        if len(warnings) > 20:
            lines.append(f"- ... {len(warnings) - 20} more warnings")
    return "\n".join(lines)
