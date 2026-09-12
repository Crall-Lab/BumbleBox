from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .status_history import list_recent_run_records


BUNDLE_SCHEMA_VERSION = "bbx.run_bundle.v1"


@dataclass
class BundleArtifact:
    kind: str
    required: bool
    source_path: str
    bundle_relpath: str
    size_bytes: int
    sha256: str


@dataclass
class RunBundleExportResult:
    bundle_name: str
    bundle_dir: str
    manifest_path: str
    zip_path: Optional[str]
    artifacts_included: int
    missing_expected: int
    preferred_tracking_csv: Optional[str]


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize_name(value: str) -> str:
    out = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    text = "".join(out).strip("._")
    return text or "bbx_bundle"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def resolve_summary_from_session_dir(session_dir: str | Path) -> Path:
    session_dir = Path(session_dir).expanduser().resolve()
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if not session_dir.is_dir():
        raise ValueError(f"Session directory must be a folder: {session_dir}")

    candidates = sorted(
        (path for path in session_dir.glob("*_run_summary.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No '*_run_summary.json' file found in: {session_dir}")
    return candidates[0]


def find_latest_summary_path(data_root: str | Path) -> Path:
    records = list_recent_run_records(data_root, limit=1)
    if not records:
        raise FileNotFoundError(f"No run summary files found under data root: {data_root}")
    return records[0].summary_path.resolve()


def _resolve_session_dir(summary_payload: dict[str, Any], summary_path: Path) -> Path:
    # A run summary copied to another machine may have a stale absolute session_dir.
    recorded = str(summary_payload.get("session_dir", "")).strip()
    if recorded:
        candidate = Path(recorded).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return summary_path.parent.resolve()


def _expected_artifacts(
    session_name: str,
    session_dir: Path,
    summary_payload: dict[str, Any],
) -> list[tuple[str, Path, bool]]:
    items: list[tuple[str, Path, bool]] = [
        ("run_config_snapshot_json", session_dir / f"{session_name}_config_snapshot.json", False),
        ("recording_midframe_png", session_dir / f"{session_name}_midframe.png", False),
        ("frame_timestamps_csv", session_dir / f"{session_name}_frame_timestamps.csv", False),
        ("actual_fps_txt", session_dir / f"{session_name}_actual_fps.txt", False),
        ("thermal_frame_timestamps_csv", session_dir / f"{session_name}_thermal_frame_timestamps.csv", False),
        ("thermal_raw_npy", session_dir / f"{session_name}_thermal_raw16.npy", False),
        ("thermal_preview_video", session_dir / f"{session_name}_thermal_preview.avi", False),
        ("thermal_midframe_png", session_dir / f"{session_name}_thermal_midframe.png", False),
        ("thermal_metadata_json", session_dir / f"{session_name}_thermal_metadata.json", False),
        ("rgb_thermal_side_by_side_video", session_dir / f"{session_name}_rgb_thermal_side_by_side.avi", False),
        ("rgb_thermal_side_by_side_png", session_dir / f"{session_name}_rgb_thermal_side_by_side_midframe.png", False),
        ("realsense_frame_timestamps_csv", session_dir / f"{session_name}_realsense_frame_timestamps.csv", False),
        ("realsense_depth_raw_npy", session_dir / f"{session_name}_realsense_depth_raw16.npy", False),
        ("realsense_depth_preview_video", session_dir / f"{session_name}_realsense_depth_preview.avi", False),
        ("realsense_depth_midframe_png", session_dir / f"{session_name}_realsense_depth_midframe.png", False),
        ("realsense_depth_raw_midframe_png", session_dir / f"{session_name}_realsense_depth_midframe_raw16.png", False),
        ("realsense_color_video", session_dir / f"{session_name}_realsense_color.avi", False),
        ("realsense_color_midframe_png", session_dir / f"{session_name}_realsense_color_midframe.png", False),
        ("realsense_metadata_json", session_dir / f"{session_name}_realsense_metadata.json", False),
        ("tracked_video_mp4", session_dir / f"{session_name}_tracked.mp4", False),
        (
            "tracked_rgb_thermal_side_by_side_video",
            session_dir / f"{session_name}_tracked_rgb_thermal_side_by_side.avi",
            False,
        ),
        (
            "tracked_rgb_thermal_side_by_side_png",
            session_dir / f"{session_name}_tracked_rgb_thermal_side_by_side_midframe.png",
            False,
        ),
        ("tracking_raw_csv", session_dir / f"{session_name}_raw.csv", False),
        ("tracking_noid_csv", session_dir / f"{session_name}_noID.csv", False),
        ("tracking_cleaned_csv", session_dir / f"{session_name}_cleaned.csv", False),
        ("fps_report_json", session_dir / f"{session_name}_fps_report.json", False),
    ]

    seen: set[Path] = set()
    video_raw = str(summary_payload.get("video_path", "")).strip()
    if video_raw:
        candidate = Path(video_raw).expanduser()
        if not candidate.is_absolute():
            candidate = (session_dir / candidate).resolve()
        seen.add(candidate)
        items.insert(1, ("video_mp4", candidate, False))

    for fallback in (
        session_dir / f"{session_name}.mp4",
        session_dir / f"{session_name}.mjpeg",
        session_dir / f"{session_name}.avi",
    ):
        resolved = fallback.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        items.insert(1, ("video_mp4", fallback, False))
    return items


def _preferred_tracking_relpath(included: list[BundleArtifact]) -> Optional[str]:
    for preferred_kind in ("tracking_cleaned_csv", "tracking_raw_csv"):
        for artifact in included:
            if artifact.kind == preferred_kind:
                return artifact.bundle_relpath
    return None


def _copy_with_unique_name(source: Path, artifacts_dir: Path, preferred_name: str) -> Path:
    target = artifacts_dir / preferred_name
    if not target.exists():
        shutil.copy2(source, target)
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 2
    while True:
        candidate = artifacts_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        counter += 1


def export_run_bundle(
    summary_path: str | Path,
    output_dir: str | Path,
    bundle_name: Optional[str] = None,
    config_path: Optional[str | Path] = None,
    include_video: bool = True,
    include_all_session_files: bool = True,
    zip_bundle: bool = True,
) -> RunBundleExportResult:
    summary_path = Path(summary_path).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")
    if not summary_path.is_file():
        raise ValueError(f"Run summary must be a file: {summary_path}")

    summary_payload = _read_json(summary_path)
    session_name = str(summary_payload.get("session_name", "")).strip()
    if not session_name:
        raise ValueError(f"Run summary is missing session_name: {summary_path}")

    session_dir = _resolve_session_dir(summary_payload, summary_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_bundle_name = f"bbx_bundle_{_sanitize_name(session_name)}_{timestamp}"
    resolved_bundle_name = _sanitize_name(bundle_name or default_bundle_name)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / resolved_bundle_name
    if bundle_dir.exists():
        raise FileExistsError(f"Bundle output already exists: {bundle_dir}")

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    included: list[BundleArtifact] = []
    missing_expected: list[dict[str, Any]] = []
    included_sources: set[Path] = set()

    # Required core artifact.
    copied_summary = _copy_with_unique_name(summary_path, artifacts_dir, f"{session_name}_run_summary.json")
    included_sources.add(summary_path)
    included.append(
        BundleArtifact(
            kind="run_summary_json",
            required=True,
            source_path=str(summary_path),
            bundle_relpath=str(copied_summary.relative_to(bundle_dir).as_posix()),
            size_bytes=int(copied_summary.stat().st_size),
            sha256=_sha256(copied_summary),
        )
    )

    if config_path:
        config = Path(config_path).expanduser().resolve()
        if config.exists() and config.is_file():
            copied_config = _copy_with_unique_name(config, artifacts_dir, "config_snapshot_input" + config.suffix)
            included_sources.add(config)
            included.append(
                BundleArtifact(
                    kind="config_snapshot_input",
                    required=False,
                    source_path=str(config),
                    bundle_relpath=str(copied_config.relative_to(bundle_dir).as_posix()),
                    size_bytes=int(copied_config.stat().st_size),
                    sha256=_sha256(copied_config),
                )
            )
        else:
            missing_expected.append(
                {
                    "kind": "config_snapshot_input",
                    "required": False,
                    "expected_source_path": str(config),
                    "reason": "Config file path does not exist.",
                }
            )

    for kind, candidate, required in _expected_artifacts(session_name, session_dir, summary_payload):
        if kind == "video_mp4" and not include_video:
            continue
        if candidate.exists() and candidate.is_file():
            copied = _copy_with_unique_name(candidate, artifacts_dir, candidate.name)
            included_sources.add(candidate.resolve())
            included.append(
                BundleArtifact(
                    kind=kind,
                    required=required,
                    source_path=str(candidate),
                    bundle_relpath=str(copied.relative_to(bundle_dir).as_posix()),
                    size_bytes=int(copied.stat().st_size),
                    sha256=_sha256(copied),
                )
            )
        else:
            missing_expected.append(
                {
                    "kind": kind,
                    "required": required,
                    "expected_source_path": str(candidate),
                    "reason": "File not found.",
                }
            )

    if include_all_session_files and session_dir.exists():
        for path in sorted(session_dir.iterdir()):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in included_sources:
                continue
            if (path.suffix.lower() in {".mp4", ".mjpeg", ".avi"}) and not include_video:
                continue
            copied = _copy_with_unique_name(path, artifacts_dir, path.name)
            included_sources.add(resolved)
            included.append(
                BundleArtifact(
                    kind="session_extra",
                    required=False,
                    source_path=str(path),
                    bundle_relpath=str(copied.relative_to(bundle_dir).as_posix()),
                    size_bytes=int(copied.stat().st_size),
                    sha256=_sha256(copied),
                )
            )

    preferred_tracking = _preferred_tracking_relpath(included)

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": _iso_now(),
        "bundle_name": resolved_bundle_name,
        "session_name": session_name,
        "source_summary_path": str(summary_path),
        "source_session_dir": str(session_dir),
        "run_started_at": summary_payload.get("started_at"),
        "run_finished_at": summary_payload.get("finished_at"),
        "run_mode": summary_payload.get("mode"),
        "run_success": summary_payload.get("success"),
        "preferred_tracking_csv": preferred_tracking,
        "artifact_count": len(included),
        "missing_expected_count": len(missing_expected),
        "artifacts": [asdict(item) for item in included],
        "missing_expected": missing_expected,
    }

    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    zip_path: Optional[str] = None
    if zip_bundle:
        zip_path = shutil.make_archive(
            base_name=str(bundle_dir),
            format="zip",
            root_dir=str(bundle_dir.parent),
            base_dir=bundle_dir.name,
        )

    return RunBundleExportResult(
        bundle_name=resolved_bundle_name,
        bundle_dir=str(bundle_dir),
        manifest_path=str(manifest_path),
        zip_path=zip_path,
        artifacts_included=len(included),
        missing_expected=len(missing_expected),
        preferred_tracking_csv=preferred_tracking,
    )


def format_bundle_export_result(result: RunBundleExportResult) -> str:
    lines = [
        "Run Bundle Export",
        "-----------------",
        f"Bundle name: {result.bundle_name}",
        f"Bundle directory: {result.bundle_dir}",
        f"Manifest: {result.manifest_path}",
        f"Artifacts included: {result.artifacts_included}",
        f"Missing expected artifacts: {result.missing_expected}",
    ]
    if result.preferred_tracking_csv:
        lines.append(f"Preferred tracking CSV: {result.preferred_tracking_csv}")
    if result.zip_path:
        lines.append(f"Zip archive: {result.zip_path}")
    return "\n".join(lines)
