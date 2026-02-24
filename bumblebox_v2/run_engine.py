from __future__ import annotations

from copy import deepcopy
import json
import socket
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunSummary:
    started_at: str
    finished_at: str
    mode: str
    session_name: str
    session_dir: str
    hostname: str
    frames_captured: int
    actual_fps: float
    tracking_elapsed_seconds: Optional[float]
    tracking_frames_processed: int
    tracking_processing_fps: Optional[float]
    video_codec: Optional[str]
    video_path: Optional[str]
    recording_preview_png_path: Optional[str]
    timestamp_path: Optional[str]
    raw_csv_path: Optional[str]
    noid_csv_path: Optional[str]
    cleaned_csv_path: Optional[str]
    fps_report_json: Optional[str]
    config_snapshot_path: Optional[str]
    warnings: List[str]
    errors: List[str]
    success: bool


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _guard_fleet_queen_local_pipeline(config: Dict[str, Any]) -> None:
    fleet = config.get("fleet", {})
    if not isinstance(fleet, dict):
        return
    role = str(fleet.get("role", "standalone")).strip().lower()
    local_enabled = bool(fleet.get("queen_local_pipeline_enabled", True))
    if role == "queen" and not local_enabled:
        raise ValueError(
            "This node is configured as fleet queen with local pipeline disabled "
            "(fleet.queen_local_pipeline_enabled=false). "
            "Enable it in config to run capture/tracking on queen."
        )


def _normalize_box_preset(box_preset: Any) -> Optional[str]:
    if box_preset is None:
        return None
    text = str(box_preset).strip().lower()
    if text in {"none", "null", ""}:
        return None
    return str(box_preset)


def _normalize_recording_codec(config: Dict[str, Any]) -> str:
    camera = config.get("camera", {})
    if not isinstance(camera, dict):
        return "mp4"
    raw = str(camera.get("codec", "mp4")).strip().lower()
    if raw in {"mjpeg", "mjpg"}:
        return "mjpeg"
    return "mp4"


def _make_session_paths(config: Dict[str, Any]) -> Tuple[str, Path]:
    hostname = socket.gethostname()
    dt = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    session_name = f"{hostname}_{dt}"

    data_root = Path(config["system"]["data_root"])
    day_dir = data_root / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return session_name, day_dir


def _write_timestamps(timestamps: List[float], session_dir: Path, session_name: str) -> Path:
    path = session_dir / f"{session_name}_frame_timestamps.csv"
    with path.open("w") as f:
        f.write("frame,time_s\n")
        for i, value in enumerate(timestamps):
            f.write(f"{i},{value:.6f}\n")
    return path


def _mock_capture_frames(config: Dict[str, Any]) -> Tuple[List[Any], List[float], float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("numpy is required for mock capture mode") from exc

    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])

    frame_count = max(1, int(round(duration * fps)))
    frames = []
    timestamps = []
    for i in range(frame_count):
        frame = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        frame[:height, :] = (i * 17) % 255
        frames.append(frame)
        timestamps.append(i / fps)

    actual_fps = frame_count / duration if duration > 0 else fps
    return frames, timestamps, actual_fps


def _capture_frames_picamera(config: Dict[str, Any]) -> Tuple[List[Any], List[float], float]:
    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError(
            "picamera2/libcamera not available. Install on Raspberry Pi OS, or set runtime.use_mock_camera=true."
        ) from exc

    fps = float(config["camera"]["fps_target"])
    duration = float(config["capture"]["recording_seconds"])
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    shutter_us = int(config["camera"]["shutter_us"])
    digital_zoom = config["camera"].get("digital_zoom")
    noise_reduction = config["camera"].get("noise_reduction", "Auto")

    picam2 = Picamera2()
    preview = picam2.create_preview_configuration({"format": "YUV420", "size": (width, height)})
    picam2.align_configuration(preview)
    picam2.configure(preview)
    picam2.set_controls({"ExposureTime": shutter_us})

    if noise_reduction != "Auto":
        try:
            mode = getattr(controls.draft.NoiseReductionModeEnum, str(noise_reduction))
            picam2.set_controls({"NoiseReductionMode": mode})
        except Exception:
            pass

    if isinstance(digital_zoom, (list, tuple)) and len(digital_zoom) == 4:
        picam2.set_controls({"ScalerCrop": tuple(digital_zoom)})

    warmup_s = float(config["runtime"].get("camera_warmup_seconds", 2.0))
    picam2.start()
    time.sleep(max(0.0, warmup_s))

    frames: List[Any] = []
    timestamps: List[float] = []

    start = time.perf_counter()
    frame_index = 0
    target_interval = 1.0 / fps
    while (time.perf_counter() - start) < duration:
        now = time.perf_counter()
        expected = start + frame_index * target_interval
        if now >= expected:
            yuv420 = picam2.capture_array()
            frames.append(yuv420)
            timestamps.append(now - start)
            frame_index += 1

    picam2.stop()

    elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else duration
    actual_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 and len(timestamps) > 1 else float(len(frames)) / max(duration, 1e-6)
    return frames, timestamps, actual_fps


def _capture_frames(config: Dict[str, Any]) -> Tuple[List[Any], List[float], float]:
    if bool(config["runtime"].get("use_mock_camera", False)):
        return _mock_capture_frames(config)
    return _capture_frames_picamera(config)


def capture_probe(
    config: Dict[str, Any],
    *,
    fps_target: Optional[float] = None,
    recording_seconds: Optional[float] = None,
    use_mock_camera: Optional[bool] = None,
) -> Tuple[int, float, float]:
    probe_config = deepcopy(config)
    probe_config.setdefault("camera", {})
    probe_config.setdefault("capture", {})
    probe_config.setdefault("runtime", {})

    if fps_target is not None:
        probe_config["camera"]["fps_target"] = float(fps_target)
    if recording_seconds is not None:
        probe_config["capture"]["recording_seconds"] = float(recording_seconds)
    if use_mock_camera is not None:
        probe_config["runtime"]["use_mock_camera"] = bool(use_mock_camera)

    start = time.perf_counter()
    frames, _timestamps, actual_fps = _capture_frames(probe_config)
    elapsed = time.perf_counter() - start
    frame_count = len(frames)
    del frames
    return frame_count, float(actual_fps), float(elapsed)


def _write_recording_video(
    frames: List[Any],
    session_dir: Path,
    session_name: str,
    fps: float,
    width: int,
    height: int,
    recording_codec: str,
    mp4_codec: str,
) -> Path:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("OpenCV is required to write recording output.") from exc

    codec_name = str(recording_codec).strip().lower()
    if codec_name == "mjpeg":
        output = session_dir / f"{session_name}.mjpeg"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        output = session_dir / f"{session_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*str(mp4_codec or "mp4v")[:4])
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output}")

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        writer.write(bgr)

    writer.release()
    return output


def _write_midpoint_preview_png(
    frames: List[Any],
    session_dir: Path,
    session_name: str,
) -> Optional[Path]:
    if not frames:
        return None
    try:
        import cv2
    except ImportError:
        return None

    mid_idx = max(0, min(len(frames) - 1, len(frames) // 2))
    frame = frames[mid_idx]
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    out = session_dir / f"{session_name}_midframe.png"
    ok = cv2.imwrite(str(out), bgr)
    if not ok:
        return None
    return out


def _track_from_ram(
    config: Dict[str, Any],
    session_name: str,
    session_dir: Path,
    frames: List[Any],
) -> Tuple[Any, Any]:
    try:
        from tag_tracking_utils import trackTagsFromRAM, trackTagsFromRAM_parallel
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import tag_tracking_utils for RAM tracking.") from exc

    wrapped_frames = [[frame] for frame in frames]
    box_preset = _normalize_box_preset(config["tracking"].get("box_preset"))
    now_str = _now_iso()
    colony_id = config["system"]["colony_id"]
    aruco_params = config["tracking"].get("aruco_params")
    parallel = bool(config["pipeline"].get("parallel_tracking", False))

    if parallel:
        df, df2, _ = trackTagsFromRAM_parallel(
            session_name,
            str(session_dir),
            wrapped_frames,
            config["tracking"]["tag_dictionary"],
            box_preset,
            now_str,
            colony_id,
            aruco_params=aruco_params,
        )
    else:
        df, df2, _ = trackTagsFromRAM(
            session_name,
            str(session_dir),
            wrapped_frames,
            config["tracking"]["tag_dictionary"],
            box_preset,
            now_str,
            socket.gethostname(),
            colony_id,
            aruco_params=aruco_params,
        )
    return df, df2


def _track_from_video(
    config: Dict[str, Any],
    session_name: str,
    session_dir: Path,
    video_path: Path,
) -> Tuple[Any, Any]:
    try:
        from tag_tracking_utils import trackTagsFromVid
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import tag_tracking_utils for video tracking.") from exc

    box_preset = _normalize_box_preset(config["tracking"].get("box_preset"))
    now_str = _now_iso()
    colony_id = config["system"]["colony_id"]
    aruco_params = config["tracking"].get("aruco_params")

    df, df2, _ = trackTagsFromVid(
        str(video_path),
        str(session_dir),
        session_name,
        config["tracking"]["tag_dictionary"],
        box_preset,
        now_str,
        colony_id,
        aruco_params=aruco_params,
    )
    return df, df2


def _run_cleaning(config: Dict[str, Any], df: Any, actual_fps: float) -> Any:
    try:
        import data_cleaning
    except Exception as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Could not import data_cleaning module.") from exc

    cleaning = config["cleaning"]
    out = df
    if bool(cleaning.get("remove_jumps", False)) and not out.empty:
        out = data_cleaning.remove_jumps(out)
    if bool(cleaning.get("interpolate_data", False)) and not out.empty:
        out = data_cleaning.interpolate(out, float(cleaning.get("max_seconds_gap", 3)), float(actual_fps))
    if bool(cleaning.get("compute_heading_angle", False)) and not out.empty:
        out = data_cleaning.compute_heading_angle(out)
    return out


def _run_metrics(config: Dict[str, Any], df: Any, actual_fps: float, session_dir: Path, session_name: str, warnings: List[str]) -> None:
    if not bool(config["pipeline"].get("calculate_behavior_metrics", False)):
        return
    if df.empty:
        return

    try:
        import behavioral_metrics as bm
    except Exception as exc:  # pragma: no cover - dependency/runtime
        warnings.append(f"Metrics skipped: could not import behavioral_metrics ({exc})")
        return

    enabled = {str(name).strip().lower() for name in config["metrics"].get("enabled", [])}
    moving_threshold = float(config["metrics"].get("moving_threshold_px_per_frame", 3.16))
    speed_cutoff_seconds = int(config["runtime"].get("speed_cutoff_seconds", 4))
    pixel_contact_distance = float(config["metrics"].get("pixel_contact_distance", 0))

    try:
        if "speed" in enabled:
            df = bm.compute_speed(df, actual_fps, speed_cutoff_seconds, moving_threshold, str(session_dir), session_name)
        if "activity" in enabled:
            df = bm.compute_activity(df, actual_fps, speed_cutoff_seconds, moving_threshold, str(session_dir), session_name)
        if "distance_from_center" in enabled:
            df = bm.compute_social_center_distance(df, str(session_dir), session_name)

        pw_df = None
        if "pairwise_distance" in enabled or "contacts" in enabled:
            pw_df = bm.pairwise_distance(df, str(session_dir), session_name)

        if "contacts" in enabled and pw_df is not None:
            bm.contact_matrix(pw_df, str(session_dir), pixel_contact_distance, session_name)

        if "video_averages" in enabled:
            bm.compute_video_averages(df, str(session_dir), session_name)
    except Exception as exc:
        warnings.append(f"Metric calculation encountered an error: {exc}")


def _run_fps_report_if_needed(
    config: Dict[str, Any],
    session_dir: Path,
    session_name: str,
    video_path: Optional[Path],
    timestamp_path: Optional[Path],
) -> Optional[Path]:
    if video_path is None:
        return None
    if video_path.suffix.lower() != ".mp4":
        return None
    if not bool(config["runtime"].get("fps_report_on_each_recording", False)):
        return None

    try:
        from .fps_report import build_fps_report, write_report_json
    except Exception:
        return None

    try:
        report = build_fps_report(
            video_path=video_path,
            timestamps_path=timestamp_path,
            recording_seconds=float(config["capture"]["recording_seconds"]),
        )
        out = session_dir / f"{session_name}_fps_report.json"
        write_report_json(report, out)
        return out
    except Exception:
        return None


def run_once(config: Dict[str, Any], mode_override: Optional[str] = None) -> RunSummary:
    started_at = _now_iso()
    warnings: List[str] = []
    errors: List[str] = []

    _guard_fleet_queen_local_pipeline(config)

    configured_mode = str(config["pipeline"]["mode"])
    mode = mode_override or configured_mode

    if configured_mode == "mixed_schedule" and mode_override is None:
        raise ValueError(
            "pipeline.mode is mixed_schedule. Use mode override 'record_only' or 'track_only' when running run-once."
        )

    if mode not in {"record_only", "track_only", "record_and_track"}:
        raise ValueError(f"Unsupported run mode: {mode}")

    session_name, session_dir = _make_session_paths(config)
    hostname = socket.gethostname()

    frames: List[Any] = []
    timestamps: List[float] = []
    actual_fps = 0.0
    tracking_elapsed_seconds: Optional[float] = None
    tracking_frames_processed = 0
    tracking_processing_fps: Optional[float] = None
    video_codec: Optional[str] = None
    video_path: Optional[Path] = None
    recording_preview_png_path: Optional[Path] = None
    timestamp_path: Optional[Path] = None
    raw_csv: Optional[Path] = None
    noid_csv: Optional[Path] = None
    cleaned_csv: Optional[Path] = None
    fps_report_json: Optional[Path] = None
    config_snapshot_path: Optional[Path] = None

    try:
        should_record = mode in {"record_only", "record_and_track"}
        should_track = mode in {"track_only", "record_and_track"}

        if should_record or should_track:
            frames, timestamps, actual_fps = _capture_frames(config)

        if bool(config["runtime"].get("save_frame_timestamps", True)) and timestamps:
            timestamp_path = _write_timestamps(timestamps, session_dir, session_name)

        if should_record:
            video_codec = _normalize_recording_codec(config)
            video_path = _write_recording_video(
                frames,
                session_dir,
                session_name,
                fps=float(config["camera"]["fps_target"]),
                width=int(config["camera"]["width"]),
                height=int(config["camera"]["height"]),
                recording_codec=video_codec,
                mp4_codec=str(config["camera"].get("mp4_codec", "mp4v")),
            )

            if video_codec == "mjpeg":
                recording_preview_png_path = _write_midpoint_preview_png(frames, session_dir, session_name)
                if recording_preview_png_path is None:
                    warnings.append("Could not write MJPEG midpoint preview PNG.")

            if video_codec == "mp4" and bool(config["runtime"].get("save_mp4_sidecar_fps_txt", True)):
                sidecar = session_dir / f"{session_name}_actual_fps.txt"
                sidecar.write_text(f"{actual_fps:.6f}\n")

        if should_track:
            source = str(config["pipeline"].get("tracking_source", "ram"))
            if not bool(config["pipeline"].get("defer_tracking_until_after_recording", True)):
                warnings.append("Non-deferred tracking is not implemented yet; using deferred tracking.")

            tracking_started = time.perf_counter()
            if source == "video":
                if video_path is None:
                    warnings.append("tracking_source=video requires recording; falling back to RAM tracking.")
                    df, df2 = _track_from_ram(config, session_name, session_dir, frames)
                else:
                    df, df2 = _track_from_video(config, session_name, session_dir, video_path)
            else:
                df, df2 = _track_from_ram(config, session_name, session_dir, frames)
            tracking_elapsed_seconds = time.perf_counter() - tracking_started
            tracking_frames_processed = len(frames)
            if tracking_elapsed_seconds > 0 and tracking_frames_processed > 0:
                tracking_processing_fps = tracking_frames_processed / tracking_elapsed_seconds

            raw_csv = session_dir / f"{session_name}_raw.csv"
            noid_csv = session_dir / f"{session_name}_noID.csv"

            if not df.empty:
                df_clean = _run_cleaning(config, df, actual_fps if actual_fps > 0 else float(config["camera"]["fps_target"]))
                cleaned_csv = session_dir / f"{session_name}_cleaned.csv"
                df_clean.to_csv(cleaned_csv, index=False)
                _run_metrics(
                    config=config,
                    df=df_clean,
                    actual_fps=actual_fps if actual_fps > 0 else float(config["camera"]["fps_target"]),
                    session_dir=session_dir,
                    session_name=session_name,
                    warnings=warnings,
                )

        fps_report_json = _run_fps_report_if_needed(config, session_dir, session_name, video_path, timestamp_path)

        try:
            config_snapshot_path = session_dir / f"{session_name}_config_snapshot.json"
            config_snapshot_path.write_text(json.dumps(config, indent=2))
        except Exception as exc:
            warnings.append(f"Config snapshot write failed: {exc}")

    except Exception as exc:
        errors.append(str(exc))

    finished_at = _now_iso()
    summary = RunSummary(
        started_at=started_at,
        finished_at=finished_at,
        mode=mode,
        session_name=session_name,
        session_dir=str(session_dir),
        hostname=hostname,
        frames_captured=len(frames),
        actual_fps=round(actual_fps, 6),
        tracking_elapsed_seconds=round(tracking_elapsed_seconds, 6) if tracking_elapsed_seconds is not None else None,
        tracking_frames_processed=tracking_frames_processed,
        tracking_processing_fps=(
            round(tracking_processing_fps, 6) if tracking_processing_fps is not None else None
        ),
        video_codec=video_codec,
        video_path=str(video_path) if video_path else None,
        recording_preview_png_path=str(recording_preview_png_path) if recording_preview_png_path else None,
        timestamp_path=str(timestamp_path) if timestamp_path else None,
        raw_csv_path=str(raw_csv) if raw_csv else None,
        noid_csv_path=str(noid_csv) if noid_csv else None,
        cleaned_csv_path=str(cleaned_csv) if cleaned_csv else None,
        fps_report_json=str(fps_report_json) if fps_report_json else None,
        config_snapshot_path=str(config_snapshot_path) if config_snapshot_path else None,
        warnings=warnings,
        errors=errors,
        success=(len(errors) == 0),
    )

    summary_path = Path(summary.session_dir) / f"{summary.session_name}_run_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    return summary


def format_run_summary(summary: RunSummary) -> str:
    lines = [
        f"Run mode: {summary.mode}",
        f"Session: {summary.session_name}",
        f"Directory: {summary.session_dir}",
        f"Frames captured: {summary.frames_captured}",
        f"Actual FPS: {summary.actual_fps}",
        f"Tracking elapsed (s): {summary.tracking_elapsed_seconds if summary.tracking_elapsed_seconds is not None else 'n/a'}",
        f"Tracking frames processed: {summary.tracking_frames_processed}",
        f"Tracking processing FPS: {summary.tracking_processing_fps if summary.tracking_processing_fps is not None else 'n/a'}",
        f"Recording codec: {summary.video_codec or 'n/a'}",
        f"Video: {summary.video_path or 'none'}",
        f"MJPEG midpoint PNG: {summary.recording_preview_png_path or 'none'}",
        f"Timestamps: {summary.timestamp_path or 'none'}",
        f"Raw CSV: {summary.raw_csv_path or 'none'}",
        f"NoID CSV: {summary.noid_csv_path or 'none'}",
        f"Cleaned CSV: {summary.cleaned_csv_path or 'none'}",
        f"FPS report: {summary.fps_report_json or 'none'}",
        f"Config snapshot: {summary.config_snapshot_path or 'none'}",
        f"Warnings: {len(summary.warnings)}",
        f"Errors: {len(summary.errors)}",
        f"Success: {summary.success}",
    ]
    if summary.warnings:
        lines.append("")
        lines.append("Warning details:")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
    if summary.errors:
        lines.append("")
        lines.append("Error details:")
        for error in summary.errors:
            lines.append(f"- {error}")
    return "\n".join(lines)
