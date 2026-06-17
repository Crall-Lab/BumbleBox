from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mjpeg", ".mjpe", ".avi", ".mov", ".mkv")
DEFAULT_TRACKING_EXTENSIONS = (".mp4", ".mjpeg", ".mjpe")
TRACKING_COMPLETION_SCHEMA_VERSION = 1
PosthocProgressCallback = Callable[[str], None]
ARUCO_PARAM_KEYS = {
    "adaptiveThreshConstant",
    "adaptiveThreshWinSizeMax",
    "adaptiveThreshWinSizeMin",
    "adaptiveThreshWinSizeStep",
    "maxMarkerPerimeterRate",
    "minMarkerPerimeterRate",
    "polygonalApproxAccuracyRate",
}
GENERATED_VIDEO_STEM_MARKERS = (
    "_tracked",
    "_thermal_preview",
    "_rgb_thermal_side_by_side",
    "_thermal_side_by_side",
    "_side_by_side",
)
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BUMBLEBOX_VIDEO_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})$"
)


def _emit_progress(progress_callback: Optional[PosthocProgressCallback], message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(message)
    except Exception:
        pass


@dataclass
class PosthocVideoResult:
    video_path: str
    session_name: str
    output_dir: str
    params_path: Optional[str]
    raw_csv_path: Optional[str]
    noid_csv_path: Optional[str]
    cleaned_csv_path: Optional[str]
    tracked_video_path: Optional[str]
    frame_count: int
    fps: float
    raw_rows_before_filters: int
    raw_rows_after_filters: int
    removed_excluded_id_rows: int
    removed_disallowed_id_rows: int
    allowed_tag_count: Optional[int]
    elapsed_seconds: float
    success: bool
    skipped: bool
    completion_marker_path: Optional[str]
    warnings: list[str]
    errors: list[str]


@dataclass
class PosthocDateOptimizationResult:
    date_dir: str
    output_dir: str
    sample_dir: Optional[str]
    selected_params_path: Optional[str]
    selected_label: Optional[str]
    optimization_summary_path: Optional[str]
    candidate_scores_path: Optional[str]
    parameter_combinations_total: int
    combinations_evaluated: int
    sample_frames_used: int
    status: str
    warnings: list[str]
    errors: list[str]


@dataclass
class PosthocTrackingReport:
    started_at: str
    finished_at: str
    input_path: str
    output_root: Optional[str]
    videos_found: int
    videos_processed: int
    videos_skipped: int
    videos_failed: int
    dictionary: str
    render_tracked_video: bool
    run_cleaning: bool
    run_metrics: bool
    resume_tracking: bool
    report_path: Optional[str]
    optimizations: list[PosthocDateOptimizationResult]
    results: list[PosthocVideoResult]
    warnings: list[str]
    errors: list[str]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _normalize_aruco_params(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("params"), dict):
        raw = raw["params"]
    if isinstance(raw.get("best_params"), dict):
        raw = raw["best_params"]
    return {key: value for key, value in raw.items() if key in ARUCO_PARAM_KEYS}


def load_aruco_params_file(path: str | Path) -> dict[str, Any]:
    params_path = Path(path).expanduser().resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Tracking parameter JSON not found: {params_path}")
    params = _normalize_aruco_params(_read_json(params_path))
    if not params:
        raise ValueError(f"No recognized ArUco parameters found in: {params_path}")
    return params


def _candidate_param_paths(date_dir: Path) -> list[Path]:
    optimization_dir = date_dir / "optimization"
    if not optimization_dir.exists():
        return []

    preferred_names = (
        "selected_tracking_params.json",
        "top_mean_detection_params.json",
        "best_score_params.json",
    )
    direct = [optimization_dir / name for name in preferred_names if (optimization_dir / name).exists()]
    recursive: list[Path] = []
    for name in preferred_names:
        recursive.extend(sorted(optimization_dir.glob(f"**/{name}"), key=lambda path: path.stat().st_mtime, reverse=True))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in direct + recursive:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _auto_params_for_date(date_dir: Path) -> tuple[Optional[dict[str, Any]], Optional[Path], list[str]]:
    warnings: list[str] = []
    for candidate in _candidate_param_paths(date_dir):
        try:
            params = load_aruco_params_file(candidate)
        except Exception as exc:
            warnings.append(f"Could not load parameter file {candidate}: {exc}")
            continue
        return params, candidate, warnings
    return None, None, warnings


def _parse_ints_from_text(text: str) -> set[int]:
    out: set[int] = set()
    for token in re.split(r"[\s,;]+", text):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(float(token)))
        except ValueError:
            continue
    return out


def load_tag_ids(path: str | Path) -> set[int]:
    tag_path = Path(path).expanduser().resolve()
    if not tag_path.exists():
        raise FileNotFoundError(f"Tag list not found: {tag_path}")
    if tag_path.suffix.lower() == ".json":
        payload = _read_json(tag_path)
        if isinstance(payload, dict):
            for key in ("tag_ids", "allowed_tag_ids", "ids", "tags"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if isinstance(payload, list):
            out: set[int] = set()
            for value in payload:
                try:
                    out.add(int(value))
                except (TypeError, ValueError):
                    continue
            return out
    return _parse_ints_from_text(tag_path.read_text())


def _config_allowed_tag_ids(config: dict[str, Any]) -> set[int]:
    tracking = config.get("tracking", {}) if isinstance(config.get("tracking", {}), dict) else {}
    out: set[int] = set()
    raw_ids = tracking.get("allowed_tag_ids", [])
    if isinstance(raw_ids, list):
        for raw_id in raw_ids:
            try:
                out.add(int(raw_id))
            except (TypeError, ValueError):
                continue
    raw_path = tracking.get("allowed_tag_ids_path")
    if raw_path:
        out.update(load_tag_ids(str(raw_path)))
    return out


def _allowed_tag_ids(
    config: dict[str, Any],
    *,
    allowed_ids: Optional[Iterable[int]] = None,
    tag_list_path: Optional[str | Path] = None,
) -> set[int]:
    out = _config_allowed_tag_ids(config)
    if allowed_ids:
        for raw_id in allowed_ids:
            out.add(int(raw_id))
    if tag_list_path:
        out.update(load_tag_ids(tag_list_path))
    return out


def _apply_allowed_tag_filter(df: Any, allowed_ids: set[int]) -> tuple[Any, int]:
    if not allowed_ids:
        return df, 0
    if getattr(df, "empty", True):
        return df, 0
    if "ID" not in getattr(df, "columns", []):
        return df, 0

    import pandas as pd

    work = df.copy()
    numeric_ids = pd.to_numeric(work["ID"], errors="coerce")
    keep_mask = numeric_ids.isin(list(allowed_ids))
    filtered = work.loc[keep_mask].copy()
    return filtered, int(len(work) - len(filtered))


def _find_date_dir(video_path: Path, input_root: Path) -> Path:
    for parent in [video_path.parent, *video_path.parents]:
        if DATE_DIR_RE.match(parent.name):
            return parent
        if parent == input_root:
            break
    return video_path.parent


def _is_generated_video(path: Path) -> bool:
    stem = path.stem.lower()
    return any(marker in stem for marker in GENERATED_VIDEO_STEM_MARKERS)


def _is_system_sidecar_file(path: Path) -> bool:
    name = path.name
    if name.startswith("._"):
        return True
    if name in {".DS_Store", "Thumbs.db"}:
        return True
    return any(part.startswith("._") for part in path.parts)


def discover_tracking_videos(
    input_path: str | Path,
    *,
    extensions: Iterable[str] = DEFAULT_TRACKING_EXTENSIONS,
    recursive: bool = True,
) -> list[Path]:
    root = Path(input_path).expanduser().resolve()
    normalized_exts = tuple(ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in extensions)
    if root.is_file():
        if (
            root.suffix.lower() in normalized_exts
            and not _is_generated_video(root)
            and not _is_system_sidecar_file(root)
        ):
            return [root]
        return []
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    pattern = "**/*" if recursive else "*"
    videos = [
        path
        for path in root.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in normalized_exts
        and not _is_generated_video(path)
        and not _is_system_sidecar_file(path)
    ]
    return sorted(videos)


def _video_datetime_and_colony(video_path: Path, config: dict[str, Any]) -> tuple[str, str]:
    match = BUMBLEBOX_VIDEO_RE.match(video_path.stem)
    fallback_colony = str(config.get("system", {}).get("colony_id", "unknown"))
    if not match:
        return _now_iso(), fallback_colony

    date_text = match.group("date")
    now_text = (
        f"{date_text}T{match.group('hour')}:{match.group('minute')}:{match.group('second')}"
    )
    prefix = match.group("prefix")
    colony = fallback_colony
    if "-" in prefix:
        colony = prefix.rsplit("-", 1)[-1] or colony
    return now_text, colony


def _video_fps(video_path: Path, config: dict[str, Any]) -> float:
    try:
        from tag_tracking_utils import load_actual_fps

        fps = load_actual_fps(str(video_path))
        if fps:
            return float(fps)
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps > 0:
            return fps
    except Exception:
        pass

    try:
        return float(config.get("camera", {}).get("fps_target", 5.0))
    except Exception:
        return 5.0


def _config_with_params(config: dict[str, Any], params: Optional[dict[str, Any]]) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(config)
    if params:
        out.setdefault("tracking", {})["aruco_params"] = dict(params)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _file_signature(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _json_hash(payload: dict[str, Any]) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _tracking_resume_signature_payload(
    *,
    video_path: Path,
    params: Optional[dict[str, Any]],
    dictionary: str,
    box_preset: Any,
    render: bool,
    run_cleaning: bool,
    metrics: bool,
    allowed_tag_ids: set[int],
    excluded_tag_ids: set[int],
) -> dict[str, Any]:
    return {
        "schema_version": TRACKING_COMPLETION_SCHEMA_VERSION,
        "source_video": _file_signature(video_path),
        "dictionary": dictionary,
        "box_preset": box_preset,
        "aruco_params": dict(params or {}),
        "render_tracked_video": bool(render),
        "run_cleaning": bool(run_cleaning),
        "run_metrics": bool(metrics),
        "allowed_tag_ids": sorted(int(tag_id) for tag_id in allowed_tag_ids),
        "excluded_tag_ids": sorted(int(tag_id) for tag_id in excluded_tag_ids),
    }


def _same_resolved_path(left: object, right: Path) -> bool:
    text = str(left or "").strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _existing_file_from_payload(payload: dict[str, Any], key: str) -> Optional[str]:
    text = str(payload.get(key) or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return str(path) if path.exists() and path.is_file() else None


def _load_valid_tracking_completion(
    marker_path: Path,
    *,
    expected_signature_hash: str,
    raw_csv_path: Path,
    noid_csv_path: Path,
    run_cleaning: bool,
    render: bool,
) -> tuple[Optional[dict[str, Any]], str]:
    if not marker_path.exists():
        return None, "no completion marker"
    try:
        payload = _read_json(marker_path)
    except Exception as exc:
        return None, f"could not read completion marker: {exc}"
    if not isinstance(payload, dict):
        return None, "completion marker is not a JSON object"
    if payload.get("schema_version") != TRACKING_COMPLETION_SCHEMA_VERSION:
        return None, "completion marker version does not match"
    if payload.get("success") is not True:
        return None, "completion marker is not successful"
    if payload.get("resume_signature_hash") != expected_signature_hash:
        return None, "tracking settings or source video changed"
    if not _same_resolved_path(payload.get("raw_csv_path"), raw_csv_path):
        return None, "completion marker raw CSV path does not match current output"
    if not _same_resolved_path(payload.get("noid_csv_path"), noid_csv_path):
        return None, "completion marker noID CSV path does not match current output"
    if not _existing_file_from_payload(payload, "raw_csv_path"):
        return None, "raw CSV is missing"
    if not _existing_file_from_payload(payload, "noid_csv_path"):
        return None, "noID CSV is missing"

    raw_after = int(payload.get("raw_rows_after_filters") or 0)
    if run_cleaning and raw_after > 0 and not _existing_file_from_payload(payload, "cleaned_csv_path"):
        return None, "cleaned CSV is missing"
    if render and raw_after > 0 and not _existing_file_from_payload(payload, "tracked_video_path"):
        return None, "tracked video is missing"
    return payload, ""


def _date_output_base(date_dir: Path, output_root_path: Optional[Path]) -> Path:
    if output_root_path is not None:
        return output_root_path / date_dir.name
    return date_dir


def _sample_indices(total_count: int, sample_count: int) -> list[int]:
    if total_count <= 0 or sample_count <= 0:
        return []
    if sample_count >= total_count:
        return list(range(total_count))
    if sample_count == 1:
        return [0]
    span = total_count - 1
    indices = sorted({int(round((idx * span) / float(sample_count - 1))) for idx in range(sample_count)})
    cursor = 0
    while len(indices) < sample_count and cursor < total_count:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return sorted(indices[:sample_count])


def _allocate_video_samples(videos: list[Path], sample_count: int) -> dict[Path, int]:
    if not videos or sample_count <= 0:
        return {}
    if sample_count < len(videos):
        selected_indices = _sample_indices(len(videos), sample_count)
        return {videos[index]: 1 for index in selected_indices}

    base = sample_count // len(videos)
    remainder = sample_count % len(videos)
    allocations: dict[Path, int] = {}
    for index, video_path in enumerate(videos):
        allocations[video_path] = base + (1 if index < remainder else 0)
    return allocations


def _write_daily_optimization_samples(
    videos: list[Path],
    sample_dir: Path,
    *,
    sample_count: int,
) -> int:
    import cv2

    sample_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in sample_dir.glob("*.png"):
        try:
            old_sample.unlink()
        except OSError:
            pass

    written = 0
    allocations = _allocate_video_samples(videos, sample_count)
    for video_path, video_sample_count in allocations.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_indices = _sample_indices(frame_count, video_sample_count) if frame_count > 0 else list(range(video_sample_count))
            for frame_index in frame_indices:
                if frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out_path = sample_dir / f"sample_{written:05d}_{video_path.stem}_frame_{frame_index:06d}.png"
                if cv2.imwrite(str(out_path), gray):
                    written += 1
                    if written >= sample_count:
                        return written
        finally:
            cap.release()
    return written


def _daily_sweep_overrides(seed_params: Optional[dict[str, Any]]) -> dict[str, list[float | int]]:
    overrides: dict[str, list[float | int]] = {}
    if not isinstance(seed_params, dict):
        return overrides
    if "minMarkerPerimeterRate" not in seed_params or "maxMarkerPerimeterRate" not in seed_params:
        return overrides
    for key in ("minMarkerPerimeterRate", "maxMarkerPerimeterRate"):
        try:
            value = round(float(seed_params[key]), 6)
        except (TypeError, ValueError):
            return {}
        if value > 0:
            overrides[key] = [value]
    return overrides if len(overrides) == 2 else {}


def _selected_candidate_params(result: Any, selection: str) -> tuple[dict[str, Any], str]:
    selection_key = str(selection or "mean_detection").strip().lower()
    if selection_key in {"mean_detection", "detection", "top_detection"}:
        candidates = getattr(result, "top_detection_candidates", None) or []
        if candidates:
            return dict(candidates[0].params), "top_mean_detection"
    return dict(getattr(result, "best_params", {}) or {}), "best_score"


def _write_selected_tracking_params(
    path: Path,
    *,
    params: dict[str, Any],
    selected_label: str,
    result: Any,
) -> None:
    payload = {
        "selected_at": _now_iso(),
        "selected_label": selected_label,
        "params": dict(params),
        "source_optimization_summary": str(getattr(result, "summary_json_path", "") or ""),
        "source_candidate_scores": str(getattr(result, "candidates_csv_path", "") or ""),
        "selection_note": (
            "Generated by bbx track-videos --optimize-per-date before post-hoc tracking."
        ),
    }
    _write_json(path, payload)


def _group_videos_by_date(videos: list[Path], input_root: Path) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    root = input_root if input_root.is_dir() else input_root.parent
    for video_path in videos:
        date_dir = _find_date_dir(video_path, root)
        grouped.setdefault(date_dir, []).append(video_path)
    return {date_dir: sorted(items) for date_dir, items in sorted(grouped.items(), key=lambda item: str(item[0]))}


def _optimize_tracking_per_date(
    config: dict[str, Any],
    *,
    videos: list[Path],
    input_root: Path,
    output_root_path: Optional[Path],
    seed_params: Optional[dict[str, Any]],
    profile: str,
    sample_frames: int,
    dictionary: str,
    tag_size_mm: float,
    expected_tags: Optional[float],
    max_combinations: int,
    execution_target: str,
    workers: Optional[int],
    selection: str,
    valid_tag_ids: Optional[Iterable[int]],
    force: bool,
    dry_run: bool,
    progress_callback: Optional[PosthocProgressCallback] = None,
) -> tuple[dict[Path, tuple[dict[str, Any], Path]], list[PosthocDateOptimizationResult]]:
    per_date_params: dict[Path, tuple[dict[str, Any], Path]] = {}
    optimization_results: list[PosthocDateOptimizationResult] = []
    grouped = _group_videos_by_date(videos, input_root)
    _emit_progress(progress_callback, f"[optimize] Dates to optimize: {len(grouped)}")

    for date_dir, date_videos in grouped.items():
        item_warnings: list[str] = []
        item_errors: list[str] = []
        base_dir = _date_output_base(date_dir, output_root_path)
        optimization_dir = base_dir / "optimization"
        selected_params_path = optimization_dir / "selected_tracking_params.json"
        sample_dir = optimization_dir / "daily_frame_samples"
        date_label = date_dir.name
        _emit_progress(progress_callback, f"[optimize] {date_label}: {len(date_videos)} video(s)")

        if selected_params_path.exists() and not force:
            try:
                params = load_aruco_params_file(selected_params_path)
                per_date_params[date_dir] = (params, selected_params_path)
                _emit_progress(progress_callback, f"[optimize] {date_label}: reusing {selected_params_path}")
                optimization_results.append(
                    PosthocDateOptimizationResult(
                        date_dir=str(date_dir),
                        output_dir=str(optimization_dir),
                        sample_dir=None,
                        selected_params_path=str(selected_params_path),
                        selected_label="existing_selected_tracking_params",
                        optimization_summary_path=None,
                        candidate_scores_path=None,
                        parameter_combinations_total=0,
                        combinations_evaluated=0,
                        sample_frames_used=0,
                        status="reused_existing",
                        warnings=item_warnings,
                        errors=item_errors,
                    )
                )
                continue
            except Exception as exc:
                item_warnings.append(f"Existing selected params could not be loaded and will be regenerated: {exc}")

        if dry_run:
            _emit_progress(progress_callback, f"[optimize] {date_label}: planned only; dry run")
            optimization_results.append(
                PosthocDateOptimizationResult(
                    date_dir=str(date_dir),
                    output_dir=str(optimization_dir),
                    sample_dir=str(sample_dir),
                    selected_params_path=str(selected_params_path),
                    selected_label=selection,
                    optimization_summary_path=None,
                    candidate_scores_path=None,
                    parameter_combinations_total=0,
                    combinations_evaluated=0,
                    sample_frames_used=0,
                    status="planned",
                    warnings=["Dry run: per-date optimization was not executed."],
                    errors=[],
                )
            )
            continue

        try:
            _emit_progress(
                progress_callback,
                f"[optimize] {date_label}: sampling {sample_frames} frame(s) into {sample_dir}",
            )
            written_samples = _write_daily_optimization_samples(
                date_videos,
                sample_dir,
                sample_count=sample_frames,
            )
            if written_samples <= 0:
                raise RuntimeError("No readable video frames could be sampled for optimization.")
            _emit_progress(progress_callback, f"[optimize] {date_label}: sampled {written_samples} frame(s)")

            from .tracking_optimizer import optimize_tracking

            last_emit = {"time": 0.0}

            def optimization_progress(
                done: int,
                total: int,
                top_candidates: list[dict[str, Any]],
                latest: dict[str, Any],
            ) -> None:
                now = time.monotonic()
                if done not in {1, total} and now - last_emit["time"] < 5.0:
                    return
                last_emit["time"] = now
                best = top_candidates[0] if top_candidates else {}
                latest_detect = float(latest.get("mean_detected") or 0.0)
                latest_decoded = float(latest.get("mean_decoded") or 0.0)
                latest_filtered = float(latest.get("mean_filtered") or 0.0)
                best_detect = float(best.get("mean_detected") or 0.0)
                best_score = float(best.get("score") or 0.0)
                high = latest.get("highest_detection_candidate") or {}
                high_detect = float(high.get("mean_detected") or 0.0)
                _emit_progress(
                    progress_callback,
                    (
                        f"[optimize] {date_label}: evaluated {done}/{total}; "
                        f"latest detect={latest_detect:.2f}, decoded={latest_decoded:.2f}, filtered={latest_filtered:.2f}; "
                        f"best detect={best_detect:.2f}, score={best_score:.3f}; "
                        f"highest detect={high_detect:.2f}"
                    ),
                )

            result = optimize_tracking(
                input_path=sample_dir,
                profile=profile,
                sample_frames=sample_frames,
                dictionary_name=dictionary,
                tag_size_mm=tag_size_mm,
                sweep_overrides=_daily_sweep_overrides(seed_params),
                max_parameter_combinations=max_combinations,
                execution_target=execution_target,
                workers=workers,
                expected_tags=expected_tags,
                output_dir=optimization_dir,
                write_preview=False,
                top_k=5,
                valid_tag_ids=valid_tag_ids,
                progress_callback=optimization_progress,
            )
            selected_params, selected_label = _selected_candidate_params(result, selection)
            if not selected_params:
                raise RuntimeError("Optimization completed but did not produce selected parameters.")
            _write_selected_tracking_params(
                selected_params_path,
                params=selected_params,
                selected_label=selected_label,
                result=result,
            )
            per_date_params[date_dir] = (selected_params, selected_params_path)
            _emit_progress(
                progress_callback,
                f"[optimize] {date_label}: selected {selected_label}; wrote {selected_params_path}",
            )
            optimization_results.append(
                PosthocDateOptimizationResult(
                    date_dir=str(date_dir),
                    output_dir=str(optimization_dir),
                    sample_dir=str(sample_dir),
                    selected_params_path=str(selected_params_path),
                    selected_label=selected_label,
                    optimization_summary_path=str(getattr(result, "summary_json_path", "") or "") or None,
                    candidate_scores_path=str(getattr(result, "candidates_csv_path", "") or "") or None,
                    parameter_combinations_total=int(getattr(result, "parameter_combinations_total", 0) or 0),
                    combinations_evaluated=int(getattr(result, "combinations_evaluated", 0) or 0),
                    sample_frames_used=int(getattr(result, "sample_frames_used", 0) or 0),
                    status="optimized",
                    warnings=item_warnings,
                    errors=item_errors,
                )
            )
        except Exception as exc:
            item_errors.append(str(exc))
            _emit_progress(progress_callback, f"[optimize] {date_label}: failed: {exc}")
            optimization_results.append(
                PosthocDateOptimizationResult(
                    date_dir=str(date_dir),
                    output_dir=str(optimization_dir),
                    sample_dir=str(sample_dir),
                    selected_params_path=None,
                    selected_label=None,
                    optimization_summary_path=None,
                    candidate_scores_path=None,
                    parameter_combinations_total=0,
                    combinations_evaluated=0,
                    sample_frames_used=0,
                    status="failed",
                    warnings=item_warnings,
                    errors=item_errors,
                )
            )
    return per_date_params, optimization_results


def run_posthoc_tracking(
    config: dict[str, Any],
    *,
    input_path: str | Path,
    output_root: Optional[str | Path] = None,
    params_path: Optional[str | Path] = None,
    allowed_ids: Optional[Iterable[int]] = None,
    tag_list_path: Optional[str | Path] = None,
    render_tracked_video: Optional[bool] = None,
    run_cleaning: bool = True,
    run_metrics: Optional[bool] = None,
    recursive: bool = True,
    extensions: Iterable[str] = DEFAULT_TRACKING_EXTENSIONS,
    dry_run: bool = False,
    optimize_per_date: bool = False,
    force_optimize_per_date: bool = False,
    optimization_profile: str = "daily",
    optimization_sample_frames: int = 40,
    optimization_tag_size_mm: float = 2.5,
    optimization_expected_tags: Optional[float] = None,
    optimization_max_combinations: int = 750,
    optimization_execution_target: str = "pi_safe",
    optimization_workers: Optional[int] = None,
    optimization_selection: str = "mean_detection",
    resume_tracking: bool = True,
    progress_callback: Optional[PosthocProgressCallback] = None,
) -> PosthocTrackingReport:
    from .run_engine import (
        _apply_tracking_id_exclusions,
        _excluded_tracking_tag_ids,
        _normalize_box_preset,
        _run_cleaning,
        _run_metrics,
    )

    started_at = _now_iso()
    run_start = time.perf_counter()
    root = Path(input_path).expanduser().resolve()
    videos = discover_tracking_videos(root, extensions=extensions, recursive=recursive)
    output_root_path = Path(output_root).expanduser().resolve() if output_root else None
    allowed_tag_ids = _allowed_tag_ids(config, allowed_ids=allowed_ids, tag_list_path=tag_list_path)
    excluded_tag_ids = _excluded_tracking_tag_ids(config)
    render = bool(config.get("runtime", {}).get("render_tracking_video", False)) if render_tracked_video is None else bool(render_tracked_video)
    metrics = bool(config.get("pipeline", {}).get("calculate_behavior_metrics", False)) if run_metrics is None else bool(run_metrics)
    global_params: Optional[dict[str, Any]] = None
    global_params_resolved_path: Optional[Path] = None
    warnings: list[str] = []
    errors: list[str] = []

    _emit_progress(progress_callback, f"[posthoc] Input: {root}")
    _emit_progress(progress_callback, f"[posthoc] Videos found: {len(videos)}")
    _emit_progress(progress_callback, f"[posthoc] Output root: {output_root_path or 'date-local tracking folders'}")
    _emit_progress(progress_callback, f"[posthoc] Resume completed videos: {'yes' if resume_tracking else 'no'}")
    _emit_progress(
        progress_callback,
        f"[posthoc] Allowed tag IDs: {len(allowed_tag_ids) if allowed_tag_ids else 'none'}",
    )
    if videos:
        date_count = len(_group_videos_by_date(videos, root if root.is_dir() else root.parent))
        _emit_progress(progress_callback, f"[posthoc] Date groups: {date_count}")
    else:
        _emit_progress(progress_callback, "[posthoc] No matching videos found; nothing to track.")

    if params_path:
        global_params_resolved_path = Path(params_path).expanduser().resolve()
        global_params = load_aruco_params_file(global_params_resolved_path)
        _emit_progress(progress_callback, f"[posthoc] Global params: {global_params_resolved_path}")

    seed_params = global_params
    if seed_params is None:
        raw_seed_params = config.get("tracking", {}).get("aruco_params")
        seed_params = dict(raw_seed_params) if isinstance(raw_seed_params, dict) else None

    per_date_params: dict[Path, tuple[dict[str, Any], Path]] = {}
    optimization_results: list[PosthocDateOptimizationResult] = []
    if optimize_per_date:
        per_date_params, optimization_results = _optimize_tracking_per_date(
            config,
            videos=videos,
            input_root=root,
            output_root_path=output_root_path,
            seed_params=seed_params,
            profile=optimization_profile,
            sample_frames=optimization_sample_frames,
            dictionary=str(config.get("tracking", {}).get("tag_dictionary", "4X4_50")),
            tag_size_mm=optimization_tag_size_mm,
            expected_tags=optimization_expected_tags,
            max_combinations=optimization_max_combinations,
            execution_target=optimization_execution_target,
            workers=optimization_workers,
            selection=optimization_selection,
            valid_tag_ids=allowed_tag_ids or None,
            force=force_optimize_per_date,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )

    results: list[PosthocVideoResult] = []
    dictionary = str(config.get("tracking", {}).get("tag_dictionary", "4X4_50"))
    box_preset = _normalize_box_preset(config.get("tracking", {}).get("box_preset"))

    for video_index, video_path in enumerate(videos, start=1):
        item_warnings: list[str] = []
        item_errors: list[str] = []
        item_start = time.perf_counter()
        date_dir = _find_date_dir(video_path, root if root.is_dir() else video_path.parent)
        output_dir = _date_output_base(date_dir, output_root_path) / "tracking"
        session_name = video_path.stem
        _emit_progress(progress_callback, f"[track] {video_index}/{len(videos)}: {video_path}")
        _emit_progress(progress_callback, f"[track] {session_name}: output {output_dir}")
        params: Optional[dict[str, Any]] = None
        used_params_path: Optional[Path] = None
        if date_dir in per_date_params:
            params, used_params_path = per_date_params[date_dir]
        elif optimize_per_date:
            item_warnings.append("Per-date optimization did not produce params for this date; checking fallback params.")

        if params is None and global_params is not None:
            params = global_params
            used_params_path = global_params_resolved_path

        if params is None:
            params, used_params_path, auto_warnings = _auto_params_for_date(date_dir)
            item_warnings.extend(auto_warnings)
        if params is None:
            raw_params = config.get("tracking", {}).get("aruco_params")
            params = dict(raw_params) if isinstance(raw_params, dict) else None
            item_warnings.append("No date optimization params found; using tracking.aruco_params from config.")
        _emit_progress(
            progress_callback,
            f"[track] {session_name}: params {used_params_path or 'config tracking.aruco_params'}",
        )

        raw_csv_path = output_dir / f"{session_name}_raw.csv"
        noid_csv_path = output_dir / f"{session_name}_noID.csv"
        completion_marker_path = output_dir / f"{session_name}_tracking_complete.json"
        cleaned_csv_path: Optional[Path] = None
        tracked_video_path: Optional[Path] = None
        frame_count = 0
        try:
            fallback_fps = float(config.get("camera", {}).get("fps_target", 5.0))
        except Exception:
            fallback_fps = 5.0
        fps = fallback_fps if dry_run else _video_fps(video_path, config)
        raw_before = 0
        raw_after = 0
        removed_excluded = 0
        removed_disallowed = 0
        success = False
        skipped = False

        resume_signature: Optional[dict[str, Any]] = None
        resume_signature_hash: Optional[str] = None
        if not dry_run:
            try:
                resume_signature = _tracking_resume_signature_payload(
                    video_path=video_path,
                    params=params,
                    dictionary=dictionary,
                    box_preset=box_preset,
                    render=render,
                    run_cleaning=run_cleaning,
                    metrics=metrics,
                    allowed_tag_ids=allowed_tag_ids,
                    excluded_tag_ids=excluded_tag_ids,
                )
                resume_signature_hash = _json_hash(resume_signature)
            except Exception as exc:
                item_warnings.append(f"Could not build resume signature; this video will be reprocessed: {exc}")

        if resume_tracking and resume_signature_hash and not dry_run:
            completed_payload, resume_reason = _load_valid_tracking_completion(
                completion_marker_path,
                expected_signature_hash=resume_signature_hash,
                raw_csv_path=raw_csv_path,
                noid_csv_path=noid_csv_path,
                run_cleaning=run_cleaning,
                render=render,
            )
            if completed_payload is not None:
                skipped = True
                success = True
                item_warnings.append("Skipped existing completed tracking result. Use --force-retrack to regenerate.")
                _emit_progress(progress_callback, f"[track] {session_name}: skipping completed result ({completion_marker_path})")
                results.append(
                    PosthocVideoResult(
                        video_path=str(video_path),
                        session_name=session_name,
                        output_dir=str(output_dir),
                        params_path=str(completed_payload.get("params_path") or (used_params_path if used_params_path else "")) or None,
                        raw_csv_path=_existing_file_from_payload(completed_payload, "raw_csv_path"),
                        noid_csv_path=_existing_file_from_payload(completed_payload, "noid_csv_path"),
                        cleaned_csv_path=_existing_file_from_payload(completed_payload, "cleaned_csv_path"),
                        tracked_video_path=_existing_file_from_payload(completed_payload, "tracked_video_path"),
                        frame_count=int(completed_payload.get("frame_count") or 0),
                        fps=float(completed_payload.get("fps") or fps),
                        raw_rows_before_filters=int(completed_payload.get("raw_rows_before_filters") or 0),
                        raw_rows_after_filters=int(completed_payload.get("raw_rows_after_filters") or 0),
                        removed_excluded_id_rows=int(completed_payload.get("removed_excluded_id_rows") or 0),
                        removed_disallowed_id_rows=int(completed_payload.get("removed_disallowed_id_rows") or 0),
                        allowed_tag_count=(
                            int(completed_payload["allowed_tag_count"])
                            if completed_payload.get("allowed_tag_count") is not None
                            else (len(allowed_tag_ids) if allowed_tag_ids else None)
                        ),
                        elapsed_seconds=0.0,
                        success=success,
                        skipped=skipped,
                        completion_marker_path=str(completion_marker_path),
                        warnings=item_warnings,
                        errors=item_errors,
                    )
                )
                continue
            if completion_marker_path.exists():
                _emit_progress(progress_callback, f"[track] {session_name}: not resuming existing marker: {resume_reason}")

        try:
            if dry_run:
                success = True
                item_warnings.append("Dry run: tracking was not executed.")
                _emit_progress(progress_callback, f"[track] {session_name}: planned only; dry run")
            else:
                from tag_tracking_utils import trackTagsFromVid

                output_dir.mkdir(parents=True, exist_ok=True)
                item_config = _config_with_params(config, params)
                now_value, colony_number = _video_datetime_and_colony(video_path, config)
                _emit_progress(progress_callback, f"[track] {session_name}: detecting tags...")
                df, df2, frame_count = trackTagsFromVid(
                    str(video_path),
                    str(output_dir),
                    session_name,
                    dictionary,
                    box_preset,
                    now_value,
                    colony_number,
                    aruco_params=params,
                )
                raw_before = int(len(df))
                df, removed_excluded = _apply_tracking_id_exclusions(df, excluded_tag_ids)
                df, removed_disallowed = _apply_allowed_tag_filter(df, allowed_tag_ids)
                raw_after = int(len(df))
                df.to_csv(raw_csv_path, index=False)
                df2.to_csv(noid_csv_path, index=False)
                _emit_progress(
                    progress_callback,
                    (
                        f"[track] {session_name}: wrote raw CSV; "
                        f"rows {raw_before}->{raw_after}, excluded={removed_excluded}, disallowed={removed_disallowed}"
                    ),
                )

                if run_cleaning and not getattr(df, "empty", True):
                    _emit_progress(progress_callback, f"[track] {session_name}: cleaning tracks...")
                    df_clean = _run_cleaning(item_config, df, fps)
                    cleaned_csv_path = output_dir / f"{session_name}_cleaned.csv"
                    df_clean.to_csv(cleaned_csv_path, index=False)
                    if metrics:
                        _emit_progress(progress_callback, f"[track] {session_name}: calculating metrics...")
                        _run_metrics(item_config, df_clean, fps, output_dir, session_name, item_warnings)

                if render and raw_csv_path.exists() and not getattr(df, "empty", True):
                    from bumblebox_desktop.visualization import render_tracking_video

                    tracked_video_path = output_dir / f"{session_name}_tracked.mp4"
                    _emit_progress(progress_callback, f"[track] {session_name}: rendering tracked video...")
                    render_tracking_video(
                        video_path=video_path,
                        tracking_csv_path=raw_csv_path,
                        output_video_path=tracked_video_path,
                    )
                elif render and getattr(df, "empty", True):
                    item_warnings.append("Tracked video visualization skipped because no allowed detections remained.")

                success = True
                _emit_progress(
                    progress_callback,
                    f"[track] {session_name}: done in {time.perf_counter() - item_start:.1f}s; frames={frame_count}",
                )
        except Exception as exc:
            item_errors.append(str(exc))
            errors.append(f"{video_path}: {exc}")
            _emit_progress(progress_callback, f"[track] {session_name}: failed: {exc}")

        video_result = PosthocVideoResult(
            video_path=str(video_path),
            session_name=session_name,
            output_dir=str(output_dir),
            params_path=str(used_params_path) if used_params_path else None,
            raw_csv_path=str(raw_csv_path) if raw_csv_path.exists() else None,
            noid_csv_path=str(noid_csv_path) if noid_csv_path.exists() else None,
            cleaned_csv_path=str(cleaned_csv_path) if cleaned_csv_path and cleaned_csv_path.exists() else None,
            tracked_video_path=str(tracked_video_path) if tracked_video_path and tracked_video_path.exists() else None,
            frame_count=int(frame_count),
            fps=float(fps),
            raw_rows_before_filters=raw_before,
            raw_rows_after_filters=raw_after,
            removed_excluded_id_rows=int(removed_excluded),
            removed_disallowed_id_rows=int(removed_disallowed),
            allowed_tag_count=len(allowed_tag_ids) if allowed_tag_ids else None,
            elapsed_seconds=float(time.perf_counter() - item_start),
            success=success,
            skipped=skipped,
            completion_marker_path=str(completion_marker_path) if completion_marker_path.exists() else None,
            warnings=item_warnings,
            errors=item_errors,
        )
        if success and not skipped and not dry_run and resume_signature is not None and resume_signature_hash is not None:
            marker_payload = {
                **asdict(video_result),
                "schema_version": TRACKING_COMPLETION_SCHEMA_VERSION,
                "completed_at": _now_iso(),
                "resume_signature": resume_signature,
                "resume_signature_hash": resume_signature_hash,
                "completion_marker_path": str(completion_marker_path),
            }
            _write_json(completion_marker_path, marker_payload)
            video_result.completion_marker_path = str(completion_marker_path)
            _emit_progress(progress_callback, f"[track] {session_name}: wrote completion marker {completion_marker_path}")
        results.append(video_result)

    report_path: Optional[Path] = None
    if output_root_path is not None:
        report_path = output_root_path / "posthoc_tracking_report.json"
    elif root.is_dir():
        report_path = root / "posthoc_tracking_report.json"
    elif results:
        report_path = Path(results[0].output_dir) / "posthoc_tracking_report.json"

    report = PosthocTrackingReport(
        started_at=started_at,
        finished_at=_now_iso(),
        input_path=str(root),
        output_root=str(output_root_path) if output_root_path else None,
        videos_found=len(videos),
        videos_processed=sum(1 for item in results if item.success),
        videos_skipped=sum(1 for item in results if item.skipped),
        videos_failed=sum(1 for item in results if not item.success),
        dictionary=dictionary,
        render_tracked_video=render,
        run_cleaning=run_cleaning,
        run_metrics=metrics,
        resume_tracking=resume_tracking,
        report_path=str(report_path) if report_path else None,
        optimizations=optimization_results,
        results=results,
        warnings=warnings,
        errors=errors,
    )
    if report_path and not dry_run:
        _write_json(report_path, asdict(report))
        _emit_progress(progress_callback, f"[posthoc] Report JSON: {report_path}")
    _emit_progress(
        progress_callback,
        (
            f"[posthoc] Finished: processed={report.videos_processed}, "
            f"skipped={report.videos_skipped}, failed={report.videos_failed}, "
            f"elapsed={time.perf_counter() - run_start:.1f}s"
        ),
    )
    return report


def format_posthoc_tracking_report(report: PosthocTrackingReport) -> str:
    lines = [
        "Post-hoc Tracking",
        "-----------------",
        f"Input: {report.input_path}",
        f"Videos found: {report.videos_found}",
        f"Videos processed: {report.videos_processed}",
        f"Videos skipped by resume: {report.videos_skipped}",
        f"Videos failed: {report.videos_failed}",
        f"Dictionary: {report.dictionary}",
        f"Render tracked video: {report.render_tracked_video}",
        f"Run cleaning: {report.run_cleaning}",
        f"Run metrics: {report.run_metrics}",
        f"Resume completed videos: {report.resume_tracking}",
        f"Report JSON: {report.report_path or 'none'}",
    ]

    if report.results:
        if report.optimizations:
            lines.extend(["", "Per-Date Optimization"])
            for item in report.optimizations:
                lines.append(
                    f"- {Path(item.date_dir).name} | {item.status} | "
                    f"evaluated={item.combinations_evaluated}/{item.parameter_combinations_total} | "
                    f"samples={item.sample_frames_used}"
                )
                lines.append(f"  selected params: {item.selected_params_path or 'none'}")
                lines.append(f"  summary: {item.optimization_summary_path or 'none'}")
                if item.warnings:
                    lines.append("  warnings: " + "; ".join(item.warnings[:3]))
                if item.errors:
                    lines.append("  errors: " + "; ".join(item.errors[:3]))

        lines.extend(["", "Videos"])
        for item in report.results:
            status = "skipped" if item.skipped else "ok" if item.success else "failed"
            lines.append(
                f"- {item.session_name} | {status} | frames={item.frame_count} | "
                f"detections={item.raw_rows_before_filters}->{item.raw_rows_after_filters} | "
                f"excluded={item.removed_excluded_id_rows} | disallowed={item.removed_disallowed_id_rows}"
            )
            lines.append(f"  raw CSV: {item.raw_csv_path or 'none'}")
            lines.append(f"  cleaned CSV: {item.cleaned_csv_path or 'none'}")
            lines.append(f"  tracked video: {item.tracked_video_path or 'none'}")
            if item.params_path:
                lines.append(f"  params: {item.params_path}")
            if item.completion_marker_path:
                lines.append(f"  completion marker: {item.completion_marker_path}")
            if item.warnings:
                lines.append("  warnings: " + "; ".join(item.warnings[:3]))
            if item.errors:
                lines.append("  errors: " + "; ".join(item.errors[:3]))

    if report.warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.errors:
        lines.extend(["", "Errors"])
        lines.extend(f"- {error}" for error in report.errors)
    return "\n".join(lines)
