from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .camera_profiles import apply_camera_profile, configured_profile_name


SCHEMA_VERSION = "bbx.multimodal_calibration.v1"
MANIFEST_NAME = "calibration_project.json"
CAPTURE_ROLES = {"calibration", "validation"}


@dataclass(frozen=True)
class CalibrationProjectResult:
    project_dir: str
    manifest_path: str
    readme_path: str
    project_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSessionResult:
    project_dir: str
    manifest_path: str
    capture_id: str
    added: bool
    available_streams: list[str]
    missing_streams: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationProjectStatus:
    project_dir: str
    manifest_path: str
    project_name: str
    capture_sets: int
    calibration_capture_sets: int
    validation_capture_sets: int
    rgb_thermal_pairs: int
    rgb_realsense_pairs: int
    complete_three_camera_sets: int
    distinct_depth_layers: int
    profile_depth_layers: dict[str, int]
    stages: dict[str, str]
    next_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _manifest_path(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser().resolve() / MANIFEST_NAME


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_manifest(project_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration project manifest not found: {path}. "
            "Create it with `./bbx calibration-project init`."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Calibration project manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration project schema in {path}.")
    return path, payload


def _sensor_manifest(config: dict[str, Any]) -> dict[str, Any]:
    resolved = apply_camera_profile(config)
    camera = resolved.get("camera", {})
    thermal = resolved.get("thermal", {})
    realsense = resolved.get("realsense", {})
    return {
        "rgb": {
            "enabled": True,
            "profile": configured_profile_name(resolved),
            "model": str(camera.get("model", "auto")),
            "width": int(camera.get("width", 4056)),
            "height": int(camera.get("height", 3040)),
            "fps_target": float(camera.get("fps_target", 5.0)),
            "timestamp_domains": ["host_wall_time", "host_monotonic", "camera_sensor"],
        },
        "thermal": {
            "enabled": bool(thermal.get("enabled", False)),
            "device_path": str(thermal.get("device_path", "auto")),
            "width": int(thermal.get("width", 160)),
            "height": int(thermal.get("height", 120)),
            "fps_target": float(thermal.get("fps_target", 8.7)),
            "timestamp_domains": ["host_wall_time", "host_monotonic"],
        },
        "realsense": {
            "enabled": bool(realsense.get("enabled", False)),
            "device_serial": str(realsense.get("device_serial", "auto")),
            "depth_width": int(realsense.get("depth_width", 848)),
            "depth_height": int(realsense.get("depth_height", 480)),
            "color_width": int(realsense.get("color_width", 848)),
            "color_height": int(realsense.get("color_height", 480)),
            "fps": int(realsense.get("fps", 30)),
            "align_to": str(realsense.get("align_to", "none")),
            "timestamp_domains": ["host_wall_time", "host_monotonic", "device"],
        },
    }


def _workflow_manifest() -> dict[str, Any]:
    return {
        "reference_stream": "rgb",
        "coordinate_frame_goal": "realsense_depth",
        "temporal": {
            "method": "shared_mechanical_occlusion_cue_and_timestamp_cross_correlation",
            "clock_model": "rgb_time = scale * sensor_time + offset_seconds",
            "corrections_path": "temporal/temporal_corrections.json",
            "notes": (
                "Estimate initial offset and drift separately for thermal and RealSense. "
                "Preserve original timestamps; apply corrections as derived data."
            ),
        },
        "spatial": {
            "method": "depth_aware_multicamera_registration",
            "rgb_thermal_transform_path": "spatial/rgb_thermal_registration.json",
            "rgb_realsense_transform_path": "spatial/rgb_realsense_calibration.json",
            "common_frame_transform_path": "spatial/common_frame_calibration.json",
            "notes": (
                "Collect correspondences at multiple nest depths. A single planar homography is retained "
                "only as a diagnostic baseline, not the final 3D mapping."
            ),
        },
        "validation": {
            "report_path": "validation/validation_report.json",
            "notes": (
                "Hold out at least one synchronized capture and report temporal error, RGB reprojection "
                "error, and thermal sampling uncertainty by depth."
            ),
        },
    }


def _project_readme(project_name: str) -> str:
    return f"""# {project_name}

This directory stores reproducible RGB, PureThermal, and RealSense calibration inputs and results.

## Workflow

1. Record synchronized sessions with a single mechanical occlusion or heat/visibility cue seen by all cameras.
2. Repeat calibration captures with targets or landmarks at at least three nest-depth layers.
3. Register each run summary with `./bbx calibration-project add-session` and a descriptive `--depth-layer`.
4. Estimate thermal and RealSense time offset plus clock drift relative to RGB; write `temporal/temporal_corrections.json`.
5. Solve RGB-to-RealSense geometry, then map thermal observations through the measured depth geometry. Keep a planar RGB/thermal homography only as a baseline.
6. Validate on held-out recordings and write `validation/validation_report.json` with error distributions, not only mean error.

Run `./bbx calibration-project status --project <this-directory>` for readiness and next actions.
"""


def initialize_calibration_project(
    config: dict[str, Any],
    project_dir: str | Path,
    *,
    project_name: str | None = None,
    force: bool = False,
) -> CalibrationProjectResult:
    root = Path(project_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Calibration project already exists: {manifest_path}. Use --force to refresh its manifest."
        )
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("captures", "temporal", "spatial", "validation"):
        (root / directory).mkdir(exist_ok=True)

    now = _now_iso()
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        _, existing = _load_manifest(root)
    name = str(
        project_name or existing.get("project_name") or root.name or "BumbleBox calibration"
    ).strip()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project_name": name,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "sensors": _sensor_manifest(config),
        "workflow": existing.get("workflow", _workflow_manifest()),
        "capture_sets": existing.get("capture_sets", []),
    }
    _write_json(manifest_path, payload)
    readme_path = root / "README.md"
    readme_path.write_text(_project_readme(name))
    return CalibrationProjectResult(
        project_dir=str(root),
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
        project_name=name,
    )


def _stream_payload(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    if name == "rgb":
        enabled = True
        frames_key = "frames_captured"
        fps_key = "actual_fps"
        timestamps_key = "timestamp_path"
        artifacts = {
            "video_path": summary.get("video_path"),
            "preview_png_path": summary.get("recording_preview_png_path"),
        }
    elif name == "thermal":
        enabled = bool(summary.get("thermal_enabled", False))
        frames_key = "thermal_frames_captured"
        fps_key = "thermal_actual_fps"
        timestamps_key = "thermal_timestamp_path"
        artifacts = {
            "raw_npy_path": summary.get("thermal_raw_npy_path"),
            "preview_video_path": summary.get("thermal_preview_video_path"),
            "preview_png_path": summary.get("thermal_preview_png_path"),
            "metadata_json_path": summary.get("thermal_metadata_json_path"),
        }
    elif name == "realsense":
        enabled = bool(summary.get("realsense_enabled", False))
        frames_key = "realsense_frames_captured"
        fps_key = "realsense_actual_fps"
        timestamps_key = "realsense_timestamp_path"
        artifacts = {
            "raw_depth_npy_path": summary.get("realsense_raw_depth_npy_path"),
            "depth_preview_video_path": summary.get("realsense_depth_preview_video_path"),
            "depth_preview_png_path": summary.get("realsense_depth_preview_png_path"),
            "color_video_path": summary.get("realsense_color_video_path"),
            "color_preview_png_path": summary.get("realsense_color_preview_png_path"),
            "metadata_json_path": summary.get("realsense_metadata_json_path"),
        }
    else:
        raise ValueError(f"Unknown stream: {name}")

    frames = int(summary.get(frames_key, 0) or 0)
    if not enabled or frames <= 0:
        return None
    return {
        "frames_captured": frames,
        "actual_fps": summary.get(fps_key),
        "timestamps_path": summary.get(timestamps_key),
        "artifacts": {key: value for key, value in artifacts.items() if value},
    }


def add_calibration_session(
    project_dir: str | Path,
    summary_path: str | Path,
    *,
    role: str = "calibration",
    depth_layer: str | None = None,
    notes: str | None = None,
) -> CalibrationSessionResult:
    role = str(role).strip().lower()
    if role not in CAPTURE_ROLES:
        raise ValueError(f"role must be one of: {', '.join(sorted(CAPTURE_ROLES))}")
    manifest_path, manifest = _load_manifest(project_dir)
    summary_file = Path(summary_path).expanduser().resolve()
    if not summary_file.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_file}")
    try:
        summary = json.loads(summary_file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run summary is not valid JSON: {summary_file}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Run summary must contain a JSON object: {summary_file}")

    captures = manifest.setdefault("capture_sets", [])
    resolved_summary = str(summary_file)
    for capture in captures:
        if str(capture.get("summary_path", "")) == resolved_summary:
            streams = capture.get("streams", {})
            available = sorted(streams)
            return CalibrationSessionResult(
                project_dir=str(manifest_path.parent),
                manifest_path=str(manifest_path),
                capture_id=str(capture.get("capture_id", summary_file.stem)),
                added=False,
                available_streams=available,
                missing_streams=sorted({"rgb", "thermal", "realsense"} - set(available)),
            )

    streams = {
        name: stream
        for name in ("rgb", "thermal", "realsense")
        if (stream := _stream_payload(summary, name)) is not None
    }
    session_name = str(summary.get("session_name") or summary_file.stem)
    capture_id = f"{role}-{len(captures) + 1:03d}-{session_name}"
    captures.append(
        {
            "capture_id": capture_id,
            "role": role,
            "depth_layer": str(depth_layer).strip() if depth_layer else None,
            "notes": str(notes).strip() if notes else None,
            "registered_at": _now_iso(),
            "summary_path": resolved_summary,
            "session_name": session_name,
            "started_at": summary.get("started_at"),
            "success": bool(summary.get("success", False)),
            "camera_profile": summary.get("camera_profile", "unknown"),
            "camera_model": summary.get("camera_model", "unknown"),
            "config_snapshot_path": summary.get("config_snapshot_path"),
            "streams": streams,
            "warnings": list(summary.get("warnings", []) or []),
            "errors": list(summary.get("errors", []) or []),
        }
    )
    manifest["updated_at"] = _now_iso()
    _write_json(manifest_path, manifest)
    available = sorted(streams)
    return CalibrationSessionResult(
        project_dir=str(manifest_path.parent),
        manifest_path=str(manifest_path),
        capture_id=capture_id,
        added=True,
        available_streams=available,
        missing_streams=sorted({"rgb", "thermal", "realsense"} - set(available)),
    )


def _artifact_complete(root: Path, relative_path: str) -> bool:
    if not relative_path:
        return False
    path = root / relative_path
    if not path.is_file():
        return False
    try:
        return isinstance(json.loads(path.read_text()), dict)
    except (OSError, json.JSONDecodeError):
        return False


def get_calibration_project_status(project_dir: str | Path) -> CalibrationProjectStatus:
    manifest_path, manifest = _load_manifest(project_dir)
    root = manifest_path.parent
    captures = [item for item in manifest.get("capture_sets", []) if isinstance(item, dict)]
    calibration = [item for item in captures if item.get("role") == "calibration"]
    validation = [item for item in captures if item.get("role") == "validation"]

    def has_streams(item: dict[str, Any], *names: str) -> bool:
        streams = item.get("streams", {})
        return bool(item.get("success", False)) and all(name in streams for name in names)

    rgb_thermal_pairs = sum(has_streams(item, "rgb", "thermal") for item in calibration)
    rgb_realsense_pairs = sum(has_streams(item, "rgb", "realsense") for item in calibration)
    complete_sets = sum(has_streams(item, "rgb", "thermal", "realsense") for item in calibration)
    profile_layers: dict[str, set[str]] = {}
    profile_complete_counts: dict[str, int] = {}
    for item in calibration:
        if not has_streams(item, "rgb", "thermal", "realsense"):
            continue
        profile = str(item.get("camera_profile") or "unknown")
        profile_complete_counts[profile] = profile_complete_counts.get(profile, 0) + 1
        if item.get("depth_layer"):
            profile_layers.setdefault(profile, set()).add(str(item["depth_layer"]).strip())
    all_depth_layers = {layer for layers in profile_layers.values() for layer in layers}

    workflow = manifest.get("workflow", {})
    temporal = workflow.get("temporal", {})
    spatial = workflow.get("spatial", {})
    validation_workflow = workflow.get("validation", {})
    temporal_complete = _artifact_complete(root, str(temporal.get("corrections_path", "")))
    spatial_paths = (
        str(spatial.get("rgb_thermal_transform_path", "")),
        str(spatial.get("rgb_realsense_transform_path", "")),
        str(spatial.get("common_frame_transform_path", "")),
    )
    spatial_complete = all(_artifact_complete(root, path) for path in spatial_paths)
    validation_complete = _artifact_complete(
        root, str(validation_workflow.get("report_path", ""))
    )

    temporal_ready = any(
        has_streams(item, "rgb", "thermal", "realsense")
        and all(item["streams"][name].get("timestamps_path") for name in ("rgb", "thermal", "realsense"))
        for item in calibration
    )
    spatial_ready = any(
        profile_complete_counts.get(profile, 0) >= 3 and len(layers) >= 3
        for profile, layers in profile_layers.items()
    )
    validation_ready = temporal_complete and spatial_complete and any(
        has_streams(item, "rgb", "thermal", "realsense") for item in validation
    )
    stages = {
        "capture": "ready" if spatial_ready else "collecting",
        "temporal": "complete" if temporal_complete else ("ready" if temporal_ready else "blocked"),
        "spatial": "complete" if spatial_complete else ("ready" if spatial_ready else "blocked"),
        "validation": "complete" if validation_complete else ("ready" if validation_ready else "blocked"),
    }
    next_actions: list[str] = []
    if not temporal_ready:
        next_actions.append(
            "Register at least one successful capture containing RGB, thermal, and RealSense streams with a shared timing cue."
        )
    if not spatial_ready:
        next_actions.append(
            "Register successful three-camera calibration captures at three or more labeled nest-depth layers for each RGB camera profile being calibrated."
        )
    if temporal_ready and not temporal_complete:
        next_actions.append("Estimate temporal offset/drift and write temporal/temporal_corrections.json.")
    if spatial_ready and not spatial_complete:
        next_actions.append(
            "Solve RGB/RealSense and RGB/thermal geometry, then write all three spatial calibration JSON artifacts."
        )
    if temporal_complete and spatial_complete and not validation:
        next_actions.append("Register at least one held-out three-camera session with role=validation.")
    elif validation_ready and not validation_complete:
        next_actions.append("Evaluate the held-out session and write validation/validation_report.json.")
    if validation_complete:
        next_actions.append("Calibration artifacts are complete; review validation error before production use.")

    return CalibrationProjectStatus(
        project_dir=str(root),
        manifest_path=str(manifest_path),
        project_name=str(manifest.get("project_name", root.name)),
        capture_sets=len(captures),
        calibration_capture_sets=len(calibration),
        validation_capture_sets=len(validation),
        rgb_thermal_pairs=rgb_thermal_pairs,
        rgb_realsense_pairs=rgb_realsense_pairs,
        complete_three_camera_sets=complete_sets,
        distinct_depth_layers=len(all_depth_layers),
        profile_depth_layers={profile: len(layers) for profile, layers in sorted(profile_layers.items())},
        stages=stages,
        next_actions=next_actions,
    )


def format_calibration_project_result(result: CalibrationProjectResult) -> str:
    return "\n".join(
        (
            "Multimodal Calibration Project",
            "------------------------------",
            f"Project: {result.project_name}",
            f"Directory: {result.project_dir}",
            f"Manifest: {result.manifest_path}",
            f"Workflow guide: {result.readme_path}",
        )
    )


def format_calibration_session_result(result: CalibrationSessionResult) -> str:
    return "\n".join(
        (
            "Calibration Session",
            "-------------------",
            f"Capture ID: {result.capture_id}",
            f"Added: {result.added}",
            f"Available streams: {', '.join(result.available_streams) or 'none'}",
            f"Missing streams: {', '.join(result.missing_streams) or 'none'}",
            f"Manifest: {result.manifest_path}",
        )
    )


def format_calibration_project_status(status: CalibrationProjectStatus) -> str:
    lines = [
        "Multimodal Calibration Status",
        "-----------------------------",
        f"Project: {status.project_name}",
        f"Directory: {status.project_dir}",
        f"Capture sets: {status.capture_sets}",
        f"Calibration / validation sets: {status.calibration_capture_sets} / {status.validation_capture_sets}",
        f"RGB+thermal pairs: {status.rgb_thermal_pairs}",
        f"RGB+RealSense pairs: {status.rgb_realsense_pairs}",
        f"Complete three-camera sets: {status.complete_three_camera_sets}",
        f"Distinct depth layers: {status.distinct_depth_layers}",
        "Depth layers by RGB profile: "
        + (
            ", ".join(
                f"{profile}={count}" for profile, count in status.profile_depth_layers.items()
            )
            or "none"
        ),
        "",
        "Stages",
        "------",
    ]
    lines.extend(f"- {name}: {state}" for name, state in status.stages.items())
    if status.next_actions:
        lines.extend(["", "Next Actions", "------------"])
        lines.extend(f"{index}. {action}" for index, action in enumerate(status.next_actions, start=1))
    return "\n".join(lines)
