from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .bundle import (
    cleanup_resolved_bundle,
    format_validation_result,
    load_manifest,
    preferred_tracking_csv_path,
    preferred_video_mp4_path,
    resolve_bundle_input,
    validate_manifest,
)
from .pipeline import PipelineOptions, format_pipeline_result, run_pipeline
from .visualization import format_tracking_video_render_result, render_tracking_video


def _cmd_validate_bundle(args: argparse.Namespace) -> int:
    resolved = resolve_bundle_input(args.bundle)
    try:
        manifest = load_manifest(resolved.manifest_path)
        result = validate_manifest(
            manifest=manifest,
            bundle_dir=resolved.bundle_dir,
            verify_hashes=not bool(args.skip_hash),
        )
    finally:
        cleanup_resolved_bundle(resolved)

    print(format_validation_result(result))
    return 0 if result.ok else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    options = PipelineOptions(
        verify_hashes=not bool(args.skip_hash),
        include_heading_angle=not bool(args.no_heading),
        run_segmentation_pipeline=bool(args.with_segmentation),
        run_behavior_classifier=bool(args.with_classifier),
        render_tracked_video=bool(args.with_tracked_video),
        tracked_video_max_frames=args.tracked_video_max_frames,
    )
    try:
        result = run_pipeline(
            bundle_input=args.bundle,
            output_dir=output_dir,
            options=options,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return 1

    print(format_pipeline_result(result))
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    if args.bundle:
        resolved = resolve_bundle_input(args.bundle)
        try:
            manifest = load_manifest(resolved.manifest_path)
            validation = validate_manifest(
                manifest=manifest,
                bundle_dir=resolved.bundle_dir,
                verify_hashes=not bool(args.skip_hash),
            )
            if validation.error_count > 0:
                print(format_validation_result(validation))
                return 1

            tracking_csv = preferred_tracking_csv_path(manifest, resolved.bundle_dir)
            video_path = preferred_video_mp4_path(manifest, resolved.bundle_dir)
            if tracking_csv is None:
                print("Could not find tracking CSV in bundle.")
                return 1
            if video_path is None:
                print("Could not find video_mp4 in bundle.")
                return 1

            if output_path is None:
                output_path = Path.cwd() / f"{manifest.session_name}_tracking_overlay.mp4"
            result = render_tracking_video(
                video_path=video_path,
                tracking_csv_path=tracking_csv,
                output_video_path=output_path,
                max_frames=args.max_frames,
            )
        except Exception as exc:
            print(f"Tracking video render failed: {exc}")
            return 1
        finally:
            cleanup_resolved_bundle(resolved)
    else:
        if not args.video or not args.tracking_csv:
            print("Provide either --bundle, or both --video and --tracking-csv.")
            return 1
        if output_path is None:
            video_input = Path(args.video).expanduser().resolve()
            output_path = video_input.with_name(f"{video_input.stem}_tracked.mp4")
        try:
            result = render_tracking_video(
                video_path=args.video,
                tracking_csv_path=args.tracking_csv,
                output_video_path=output_path,
                max_frames=args.max_frames,
            )
        except Exception as exc:
            print(f"Tracking video render failed: {exc}")
            return 1

    print(format_tracking_video_render_result(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bbx-desktop",
        description="BumbleBox downstream desktop scaffold (bundle ingestion + analysis).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-bundle",
        help="Validate bundle manifest structure and artifact checksums.",
    )
    validate_parser.add_argument("--bundle", required=True, help="Path to bundle directory or .zip.")
    validate_parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA256 verification (faster, lower integrity assurance).",
    )
    validate_parser.set_defaults(func=_cmd_validate_bundle)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run first downstream analysis pipeline on a run bundle.",
    )
    analyze_parser.add_argument("--bundle", required=True, help="Path to bundle directory or .zip.")
    analyze_parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "desktop_analysis_output"),
        help="Output directory for pipeline artifacts.",
    )
    analyze_parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA256 verification (faster, lower integrity assurance).",
    )
    analyze_parser.add_argument(
        "--no-heading",
        action="store_true",
        help="Do not compute heading-angle columns in downstream output.",
    )
    analyze_parser.add_argument(
        "--with-segmentation",
        action="store_true",
        help="Flag segmentation stage on (currently scaffolded placeholder).",
    )
    analyze_parser.add_argument(
        "--with-classifier",
        action="store_true",
        help="Flag behavior classifier stage on (currently scaffolded placeholder).",
    )
    analyze_parser.add_argument(
        "--with-tracked-video",
        action="store_true",
        help="Render tracking overlay MP4 (create_tracked_videos-style output).",
    )
    analyze_parser.add_argument(
        "--tracked-video-max-frames",
        type=int,
        help="Optional frame cap for faster debug/test renders.",
    )
    analyze_parser.set_defaults(func=_cmd_analyze)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Render tracking overlay MP4 from a bundle or explicit video+tracking CSV.",
    )
    visualize_parser.add_argument("--bundle", help="Path to bundle directory or .zip.")
    visualize_parser.add_argument("--video", help="Source video path when not using --bundle.")
    visualize_parser.add_argument("--tracking-csv", help="Tracking CSV path when not using --bundle.")
    visualize_parser.add_argument("--output", help="Output MP4 path.")
    visualize_parser.add_argument("--max-frames", type=int, help="Optional frame cap for faster rendering.")
    visualize_parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip hash checks when bundle input is used.",
    )
    visualize_parser.set_defaults(func=_cmd_visualize)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
