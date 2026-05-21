from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .analysis import build_tracking_overview, write_tracking_overview
from .bundle import (
    cleanup_resolved_bundle,
    load_manifest,
    preferred_video_mp4_path,
    preferred_tracking_csv_path,
    resolve_bundle_input,
    validate_manifest,
)
from .visualization import render_tracking_video


@dataclass
class PipelineOptions:
    verify_hashes: bool = True
    include_heading_angle: bool = True
    run_segmentation_pipeline: bool = False
    run_behavior_classifier: bool = False
    render_tracked_video: bool = False
    tracked_video_max_frames: Optional[int] = None


@dataclass
class PipelineResult:
    bundle_input: str
    bundle_dir: str
    manifest_path: str
    tracking_csv_used: str
    output_dir: str
    report_json: str
    validation_errors: int
    validation_warnings: int
    segmentation_status: str
    classifier_status: str
    tracked_video_status: str
    tracked_video_path: Optional[str]


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_pipeline(
    bundle_input: str | Path,
    output_dir: str | Path,
    options: Optional[PipelineOptions] = None,
) -> PipelineResult:
    options = options or PipelineOptions()
    resolved = resolve_bundle_input(bundle_input)
    try:
        manifest = load_manifest(resolved.manifest_path)
        validation = validate_manifest(
            manifest=manifest,
            bundle_dir=resolved.bundle_dir,
            verify_hashes=options.verify_hashes,
        )
        if validation.error_count > 0:
            raise ValueError(
                "Bundle validation failed:\n" + "\n".join(f"- {item}" for item in validation.errors)
            )

        tracking_path = preferred_tracking_csv_path(manifest, resolved.bundle_dir)
        if tracking_path is None:
            raise FileNotFoundError(
                "Could not determine tracking CSV from manifest. "
                "Need preferred_tracking_csv or tracking_* artifact entries."
            )

        tracking_df = pd.read_csv(tracking_path)
        overview = build_tracking_overview(
            tracking_df,
            include_heading=options.include_heading_angle,
        )

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = write_tracking_overview(overview, output_dir=output_dir)

        tracked_video_status = "disabled"
        tracked_video_path: Optional[str] = None
        if options.render_tracked_video:
            source_video = preferred_video_mp4_path(manifest, resolved.bundle_dir)
            if source_video is None:
                tracked_video_status = "requested_but_video_missing"
            else:
                try:
                    render = render_tracking_video(
                        video_path=source_video,
                        tracking_csv_path=tracking_path,
                        output_video_path=output_dir / "tracking_overlay.mp4",
                        max_frames=options.tracked_video_max_frames,
                    )
                    tracked_video_status = "ok"
                    tracked_video_path = render.output_video
                    artifact_paths["tracked_video_mp4"] = render.output_video
                except Exception as exc:
                    tracked_video_status = f"failed: {exc}"

        segmentation_status = (
            "requested_but_not_implemented"
            if options.run_segmentation_pipeline
            else "disabled"
        )
        classifier_status = (
            "requested_but_not_implemented"
            if options.run_behavior_classifier
            else "disabled"
        )

        report = {
            "created_at": _iso_now(),
            "bundle_input": str(bundle_input),
            "bundle_dir": str(resolved.bundle_dir),
            "manifest_path": str(resolved.manifest_path),
            "tracking_csv_used": str(tracking_path),
            "pipeline_options": asdict(options),
            "validation": {
                "errors": validation.errors,
                "warnings": validation.warnings,
                "error_count": validation.error_count,
                "warning_count": validation.warning_count,
            },
            "overview_summary": overview.summary,
            "artifacts": artifact_paths,
            "tracked_video": {
                "status": tracked_video_status,
                "path": tracked_video_path,
            },
            "segmentation_pipeline": {
                "status": segmentation_status,
                "note": (
                    "Segmentation integration is scaffolded but not implemented yet. "
                    "Implement this stage in downstream repo."
                ),
            },
            "behavior_classifier": {
                "status": classifier_status,
                "note": (
                    "Classifier integration is scaffolded but not implemented yet. "
                    "Implement model loading/inference in downstream repo."
                ),
            },
        }
        report_path = output_dir / "pipeline_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        return PipelineResult(
            bundle_input=str(bundle_input),
            bundle_dir=str(resolved.bundle_dir),
            manifest_path=str(resolved.manifest_path),
            tracking_csv_used=str(tracking_path),
            output_dir=str(output_dir),
            report_json=str(report_path),
            validation_errors=validation.error_count,
            validation_warnings=validation.warning_count,
            segmentation_status=segmentation_status,
            classifier_status=classifier_status,
            tracked_video_status=tracked_video_status,
            tracked_video_path=tracked_video_path,
        )
    finally:
        cleanup_resolved_bundle(resolved)


def format_pipeline_result(result: PipelineResult) -> str:
    lines = [
        "Desktop Analysis Pipeline",
        "-------------------------",
        f"Bundle input: {result.bundle_input}",
        f"Bundle directory: {result.bundle_dir}",
        f"Manifest: {result.manifest_path}",
        f"Tracking CSV used: {result.tracking_csv_used}",
        f"Output directory: {result.output_dir}",
        f"Pipeline report: {result.report_json}",
        f"Validation: {result.validation_errors} error(s), {result.validation_warnings} warning(s)",
        f"Tracked video stage: {result.tracked_video_status}",
        f"Tracked video output: {result.tracked_video_path or 'none'}",
        f"Segmentation stage: {result.segmentation_status}",
        f"Classifier stage: {result.classifier_status}",
    ]
    return "\n".join(lines)
