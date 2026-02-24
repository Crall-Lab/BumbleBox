from __future__ import annotations

import concurrent.futures
import csv
import itertools
import json
import os
import statistics
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2


DEFAULT_DICTIONARY = "4X4_50"
DEFAULT_PROFILE = "quick"
DEFAULT_EXECUTION_TARGET = "pi_safe"
DEFAULT_TAG_SIZE_MM = 2.5
DEFAULT_EARLY_STOP_PATIENCE = 40
DEFAULT_EARLY_STOP_MIN_IMPROVEMENT = 0.002
VALID_SWEEP_OVERRIDE_KEYS = {
    "minMarkerPerimeterRate",
    "adaptiveThreshWinSizeMin",
    "adaptiveThreshWinSizeMax",
    "adaptiveThreshWinSizeStep",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mjpeg", ".avi", ".mov", ".mkv"}
VALID_PROFILES = {"quick", "balanced", "deep"}
VALID_EXECUTION_TARGETS = {"pi_safe", "desktop"}

PROFILE_PARAMETER_SPACE = {
    "quick": {
        "minMarkerPerimeterRate": [0.015, 0.02, 0.03],
        "adaptiveThreshWinSizeMin": [3, 5],
        "adaptiveThreshWinSizeMax": [23, 31],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.05, 0.08],
        "adaptiveThreshConstant": [7],
    },
    "balanced": {
        "minMarkerPerimeterRate": [0.01, 0.015, 0.02, 0.03],
        "adaptiveThreshWinSizeMin": [3, 5, 7],
        "adaptiveThreshWinSizeMax": [21, 31, 41],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.04, 0.06, 0.08],
        "adaptiveThreshConstant": [7],
    },
    "deep": {
        "minMarkerPerimeterRate": [0.008, 0.012, 0.016, 0.02, 0.03],
        "adaptiveThreshWinSizeMin": [3, 5, 7],
        "adaptiveThreshWinSizeMax": [21, 31, 41],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.04, 0.05, 0.06, 0.08],
        "adaptiveThreshConstant": [5, 7],
    },
}


ProgressCallback = Callable[[int, int], None]


@dataclass
class OptimizationCandidate:
    rank: int
    score: float
    params: dict[str, float | int]
    mean_detected: float
    std_detected: float
    mean_rejected: float
    stability: float
    unique_ids: int
    eval_fps: float
    runtime_seconds: float


@dataclass
class TrackingOptimizationResult:
    created_at: str
    input_path: str
    input_type: str
    dictionary: str
    tag_size_mm: float
    profile: str
    execution_target: str
    workers: int
    sample_frames_requested: int
    sample_frames_used: int
    combinations_evaluated: int
    early_stop_patience: int
    early_stop_min_improvement: float
    early_stopped: bool
    sweep_overrides: dict[str, list[float | int]]
    output_dir: str
    summary_json_path: str
    candidates_csv_path: str
    preview_video_path: Optional[str]
    best_params: dict[str, float | int]
    best_score: float
    best_mean_detected: float
    top_candidates: list[OptimizationCandidate]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["top_candidates"] = [asdict(item) for item in self.top_candidates]
        return payload


def _require_aruco() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("OpenCV ArUcoDetector API is missing. Use OpenCV 4.7+.")


def normalize_dictionary_name(dictionary_name: str) -> str:
    name = str(dictionary_name or "").strip().upper()
    if not name:
        name = f"DICT_{DEFAULT_DICTIONARY}"
    if not name.startswith("DICT_"):
        name = f"DICT_{name}"
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return name


def recommended_worker_count(
    execution_target: str,
    requested_workers: Optional[int] = None,
) -> int:
    cpu_count = os.cpu_count() or 1
    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError("workers must be >= 1")
        return requested_workers

    if execution_target == "pi_safe":
        return max(1, min(2, cpu_count))
    if execution_target == "desktop":
        if cpu_count <= 2:
            return cpu_count
        return max(1, min(12, cpu_count - 1))
    raise ValueError(f"Unknown execution target: {execution_target}")


def _min_marker_rates_for_tag_size(
    tag_size_mm: float,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    # Defaults target small ArUco tags (2.5 mm) and high-resolution BumbleBox cameras.
    if tag_size_mm <= 2.5:
        base = [0.003, 0.005, 0.008, 0.012, 0.016, 0.02]
    elif tag_size_mm <= 3.5:
        base = [0.005, 0.008, 0.012, 0.016, 0.02, 0.03]
    else:
        base = [0.008, 0.012, 0.016, 0.02, 0.03, 0.04]

    # Small normalization based on image area relative to HQ default.
    hq_area = float(4056 * 3040)
    frame_area = float(max(1, frame_width * frame_height))
    area_scale = (frame_area / hq_area) ** 0.5
    scaled = [max(0.001, min(0.08, value * area_scale)) for value in base]
    return sorted({round(value, 4) for value in scaled})


def build_parameter_grid(
    profile: str,
    tag_size_mm: float,
    frame_width: int,
    frame_height: int,
    sweep_overrides: Optional[dict[str, Sequence[float | int]]] = None,
) -> list[dict[str, float | int]]:
    profile_key = str(profile).strip().lower()
    if profile_key not in VALID_PROFILES:
        raise ValueError(f"profile must be one of {sorted(VALID_PROFILES)}, got: {profile}")
    if tag_size_mm <= 0:
        raise ValueError("tag_size_mm must be > 0")

    space = PROFILE_PARAMETER_SPACE[profile_key]
    space = dict(space)
    space["minMarkerPerimeterRate"] = _min_marker_rates_for_tag_size(
        tag_size_mm=tag_size_mm,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if sweep_overrides:
        for key, values in sweep_overrides.items():
            if key not in VALID_SWEEP_OVERRIDE_KEYS:
                raise ValueError(
                    "sweep override key must be one of "
                    f"{sorted(VALID_SWEEP_OVERRIDE_KEYS)}, got: {key}"
                )
            if not values:
                raise ValueError(f"sweep override list for '{key}' cannot be empty")

            if key == "minMarkerPerimeterRate":
                parsed = []
                for value in values:
                    parsed_value = float(value)
                    if parsed_value <= 0:
                        raise ValueError(f"{key} override values must be > 0")
                    parsed.append(round(parsed_value, 6))
                space[key] = sorted(set(parsed))
            else:
                parsed = []
                for value in values:
                    parsed_float = float(value)
                    if not parsed_float.is_integer():
                        raise ValueError(f"{key} override values must be whole numbers")
                    parsed_value = int(parsed_float)
                    if parsed_value <= 0:
                        raise ValueError(f"{key} override values must be >= 1")
                    parsed.append(parsed_value)
                space[key] = sorted(set(parsed))
    keys = list(space.keys())
    combinations = []
    for values in itertools.product(*(space[key] for key in keys)):
        params = dict(zip(keys, values))
        win_min = int(params["adaptiveThreshWinSizeMin"])
        win_max = int(params["adaptiveThreshWinSizeMax"])
        step = int(params["adaptiveThreshWinSizeStep"])
        if win_min >= win_max:
            continue
        if step <= 0:
            continue
        if (win_max - win_min) < step:
            continue
        combinations.append(params)
    if not combinations:
        raise RuntimeError("Parameter grid is empty after validation.")
    return combinations


def _classify_input_path(path: Path) -> str:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension for {path}.")
        return "video"
    if path.is_dir():
        return "image_dir"
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _sample_indices(total_count: int, sample_count: int) -> list[int]:
    sample_count = max(1, sample_count)
    if total_count <= sample_count:
        return list(range(total_count))
    if sample_count == 1:
        return [total_count // 2]

    step = (total_count - 1) / float(sample_count - 1)
    indices = sorted({min(total_count - 1, int(round(i * step))) for i in range(sample_count)})
    if len(indices) == sample_count:
        return indices

    used = set(indices)
    cursor = 0
    while len(indices) < sample_count and cursor < total_count:
        if cursor not in used:
            indices.append(cursor)
            used.add(cursor)
        cursor += 1
    return sorted(indices[:sample_count])


def _load_sample_frames_from_video(video_path: Path, sample_count: int) -> tuple[list, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for optimization: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        indices = _sample_indices(total_frames, sample_count)
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    else:
        while len(frames) < sample_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

    cap.release()
    if not frames:
        raise RuntimeError(f"No readable frames found in video: {video_path}")
    return frames, max(total_frames, len(frames))


def _load_sample_frames_from_image_dir(image_dir: Path, sample_count: int) -> tuple[list, int]:
    all_images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        raise RuntimeError(f"No supported image files found in directory: {image_dir}")

    indices = _sample_indices(len(all_images), sample_count)
    frames = []
    for idx in indices:
        image = cv2.imread(str(all_images[idx]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        frames.append(image)

    if not frames:
        raise RuntimeError(f"All sampled images failed to load from: {image_dir}")
    return frames, len(all_images)


def load_sample_frames(input_path: str | Path, sample_count: int) -> tuple[list, str, int]:
    path = Path(input_path).expanduser().resolve()
    input_type = _classify_input_path(path)
    if input_type == "video":
        frames, total_count = _load_sample_frames_from_video(path, sample_count)
    else:
        frames, total_count = _load_sample_frames_from_image_dir(path, sample_count)
    return frames, input_type, total_count


def _score_candidate(
    mean_detected: float,
    mean_rejected: float,
    std_detected: float,
    stability: float,
    eval_fps: float,
    expected_tags: Optional[float],
) -> float:
    score = mean_detected
    score += 0.35 * stability
    score += 0.10 * min(eval_fps, 90.0) / 90.0
    score -= 0.08 * mean_rejected
    score -= 0.05 * std_detected
    if expected_tags is not None and expected_tags > 0:
        expected_error = abs(mean_detected - expected_tags) / expected_tags
        score -= 0.40 * expected_error
    return score


def _evaluate_candidate(
    params: dict[str, float | int],
    frames: Sequence,
    dictionary_name: str,
    expected_tags: Optional[float],
) -> OptimizationCandidate:
    detector_params = cv2.aruco.DetectorParameters()
    for param_name, param_value in params.items():
        if not hasattr(detector_params, param_name):
            continue
        setattr(detector_params, param_name, param_value)

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    detected_counts = []
    rejected_counts = []
    stability_scores = []
    unique_ids = set()
    previous_ids = set()

    start = time.perf_counter()
    for frame in frames:
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)
        ids_set = set()
        if ids is not None and len(ids) > 0:
            ids_set = {int(value) for value in ids.flatten().tolist()}
            unique_ids.update(ids_set)

        detected = len(ids_set)
        rejected_count = len(rejected) if rejected is not None else 0
        detected_counts.append(detected)
        rejected_counts.append(rejected_count)

        union = previous_ids | ids_set
        if union:
            stability_scores.append(len(previous_ids & ids_set) / len(union))
        else:
            stability_scores.append(1.0)
        previous_ids = ids_set

    runtime = max(1e-9, time.perf_counter() - start)
    frame_count = len(frames)
    mean_detected = statistics.fmean(detected_counts) if detected_counts else 0.0
    mean_rejected = statistics.fmean(rejected_counts) if rejected_counts else 0.0
    std_detected = statistics.pstdev(detected_counts) if len(detected_counts) > 1 else 0.0
    stability = statistics.fmean(stability_scores) if stability_scores else 0.0
    eval_fps = frame_count / runtime
    score = _score_candidate(
        mean_detected=mean_detected,
        mean_rejected=mean_rejected,
        std_detected=std_detected,
        stability=stability,
        eval_fps=eval_fps,
        expected_tags=expected_tags,
    )
    return OptimizationCandidate(
        rank=0,
        score=score,
        params=params,
        mean_detected=mean_detected,
        std_detected=std_detected,
        mean_rejected=mean_rejected,
        stability=stability,
        unique_ids=len(unique_ids),
        eval_fps=eval_fps,
        runtime_seconds=runtime,
    )


def _default_output_root(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "tracking_optimization"
    return input_path / "tracking_optimization"


def _write_candidates_csv(candidates: Sequence[OptimizationCandidate], csv_path: Path) -> None:
    fields = [
        "rank",
        "score",
        "mean_detected",
        "std_detected",
        "mean_rejected",
        "stability",
        "unique_ids",
        "eval_fps",
        "runtime_seconds",
        "params_json",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "rank": item.rank,
                    "score": f"{item.score:.6f}",
                    "mean_detected": f"{item.mean_detected:.6f}",
                    "std_detected": f"{item.std_detected:.6f}",
                    "mean_rejected": f"{item.mean_rejected:.6f}",
                    "stability": f"{item.stability:.6f}",
                    "unique_ids": item.unique_ids,
                    "eval_fps": f"{item.eval_fps:.6f}",
                    "runtime_seconds": f"{item.runtime_seconds:.6f}",
                    "params_json": json.dumps(item.params, sort_keys=True),
                }
            )


def _iter_input_frames(input_path: Path):
    input_type = _classify_input_path(input_path)
    if input_type == "video":
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video for preview: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 6.0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame, fps
        cap.release()
        return

    image_paths = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield frame, 6.0


def write_preview_video(
    input_path: str | Path,
    dictionary_name: str,
    best_params: dict[str, float | int],
    output_path: str | Path,
    max_frames: int = 240,
) -> Optional[Path]:
    path = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()

    detector_params = cv2.aruco.DetectorParameters()
    for key, value in best_params.items():
        if hasattr(detector_params, key):
            setattr(detector_params, key, value)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    writer = None
    frames_written = 0
    try:
        for frame, fps in _iter_input_frames(path):
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to create preview video: {output}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rejected = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            writer.write(frame)
            frames_written += 1
            if frames_written >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        return None
    return output


def optimize_tracking(
    input_path: str | Path,
    profile: str = DEFAULT_PROFILE,
    sample_frames: int = 80,
    dictionary_name: str = DEFAULT_DICTIONARY,
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM,
    sweep_overrides: Optional[dict[str, Sequence[float | int]]] = None,
    execution_target: str = DEFAULT_EXECUTION_TARGET,
    workers: Optional[int] = None,
    expected_tags: Optional[float] = None,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    early_stop_min_improvement: float = DEFAULT_EARLY_STOP_MIN_IMPROVEMENT,
    output_dir: Optional[str | Path] = None,
    write_preview: bool = False,
    preview_frames: int = 240,
    top_k: int = 10,
    progress_callback: Optional[ProgressCallback] = None,
) -> TrackingOptimizationResult:
    _require_aruco()

    profile_key = str(profile).strip().lower()
    if profile_key not in VALID_PROFILES:
        raise ValueError(f"profile must be one of {sorted(VALID_PROFILES)}, got: {profile}")

    target_key = str(execution_target).strip().lower()
    if target_key not in VALID_EXECUTION_TARGETS:
        raise ValueError(
            f"execution_target must be one of {sorted(VALID_EXECUTION_TARGETS)}, got: {execution_target}"
        )
    if sample_frames <= 0:
        raise ValueError("sample_frames must be >= 1")
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    if tag_size_mm <= 0:
        raise ValueError("tag_size_mm must be > 0")
    if expected_tags is not None and expected_tags <= 0:
        raise ValueError("expected_tags must be > 0 when provided")
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be >= 0")
    if early_stop_min_improvement < 0:
        raise ValueError("early_stop_min_improvement must be >= 0")

    normalized_dictionary = normalize_dictionary_name(dictionary_name)
    resolved_input = Path(input_path).expanduser().resolve()
    sampled_frames, input_type, total_input_frames = load_sample_frames(resolved_input, sample_frames)
    frame_height, frame_width = sampled_frames[0].shape[:2]
    param_grid = build_parameter_grid(
        profile_key,
        tag_size_mm=tag_size_mm,
        frame_width=frame_width,
        frame_height=frame_height,
        sweep_overrides=sweep_overrides,
    )
    resolved_workers = recommended_worker_count(target_key, workers)

    evaluated: list[OptimizationCandidate] = []
    total = len(param_grid)
    done = 0
    early_stop_enabled = early_stop_patience > 0
    best_seen_score = float("-inf")
    since_improvement = 0
    early_stopped = False

    def register_candidate(candidate: OptimizationCandidate) -> None:
        nonlocal done, best_seen_score, since_improvement, early_stopped
        evaluated.append(candidate)
        done += 1
        if progress_callback:
            progress_callback(done, total)
        if candidate.score > (best_seen_score + early_stop_min_improvement):
            best_seen_score = candidate.score
            since_improvement = 0
        else:
            since_improvement += 1
        if early_stop_enabled and since_improvement >= early_stop_patience:
            early_stopped = True

    if resolved_workers == 1:
        for params in param_grid:
            candidate = _evaluate_candidate(params, sampled_frames, normalized_dictionary, expected_tags)
            register_candidate(candidate)
            if early_stopped:
                break
    else:
        cursor = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            while cursor < total and not early_stopped:
                batch = param_grid[cursor : cursor + resolved_workers]
                futures = [
                    executor.submit(
                        _evaluate_candidate,
                        params,
                        sampled_frames,
                        normalized_dictionary,
                        expected_tags,
                    )
                    for params in batch
                ]
                for future in concurrent.futures.as_completed(futures):
                    register_candidate(future.result())
                cursor += len(batch)

    evaluated.sort(
        key=lambda item: (item.score, item.mean_detected, -item.mean_rejected, item.eval_fps),
        reverse=True,
    )
    for rank, item in enumerate(evaluated, start=1):
        item.rank = rank

    best = evaluated[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _default_output_root(resolved_input).resolve()
    )
    run_dir = output_root / f"optimize_tracking_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "candidate_scores.csv"
    _write_candidates_csv(evaluated, csv_path)

    preview_video_path: Optional[Path] = None
    if write_preview:
        preview_video_path = write_preview_video(
            input_path=resolved_input,
            dictionary_name=normalized_dictionary,
            best_params=best.params,
            output_path=run_dir / "best_params_preview.mp4",
            max_frames=preview_frames,
        )

    top_candidates = evaluated[: max(top_k, 1)]
    result = TrackingOptimizationResult(
        created_at=datetime.now().isoformat(timespec="seconds"),
        input_path=str(resolved_input),
        input_type=input_type,
        dictionary=normalized_dictionary,
        tag_size_mm=tag_size_mm,
        profile=profile_key,
        execution_target=target_key,
        workers=resolved_workers,
        sample_frames_requested=sample_frames,
        sample_frames_used=len(sampled_frames),
        combinations_evaluated=len(evaluated),
        early_stop_patience=early_stop_patience,
        early_stop_min_improvement=early_stop_min_improvement,
        early_stopped=early_stopped,
        sweep_overrides={
            key: list(values)
            for key, values in (sweep_overrides or {}).items()
        },
        output_dir=str(run_dir),
        summary_json_path=str(run_dir / "optimization_summary.json"),
        candidates_csv_path=str(csv_path),
        preview_video_path=str(preview_video_path) if preview_video_path else None,
        best_params=best.params,
        best_score=best.score,
        best_mean_detected=best.mean_detected,
        top_candidates=top_candidates,
    )

    summary = result.to_dict()
    summary["total_input_frames"] = total_input_frames
    summary["all_candidates_count"] = len(evaluated)
    summary["all_candidates_csv"] = str(csv_path)

    with Path(result.summary_json_path).open("w") as f:
        json.dump(summary, f, indent=2)

    return result


def format_optimization_report(result: TrackingOptimizationResult, top_k: int = 5) -> str:
    lines = [
        "Tracking Optimization Report",
        "----------------------------",
        f"Created: {result.created_at}",
        f"Input: {result.input_path} ({result.input_type})",
        f"Dictionary: {result.dictionary}",
        f"Tag size: {result.tag_size_mm:.3f} mm",
        f"Profile: {result.profile}",
        f"Execution target: {result.execution_target}",
        f"Workers used: {result.workers}",
        f"Sample frames: {result.sample_frames_used}/{result.sample_frames_requested}",
        f"Parameter sets evaluated: {result.combinations_evaluated}",
        (
            "Early stop: disabled"
            if result.early_stop_patience <= 0
            else (
                f"Early stop: {'triggered' if result.early_stopped else 'not triggered'} "
                f"(patience={result.early_stop_patience}, min_improvement={result.early_stop_min_improvement:.6f})"
            )
        ),
        (
            "Sweep overrides: none"
            if not result.sweep_overrides
            else f"Sweep overrides: {json.dumps(result.sweep_overrides, sort_keys=True)}"
        ),
        f"Best score: {result.best_score:.4f}",
        f"Best mean detections/frame: {result.best_mean_detected:.3f}",
        f"Best params: {json.dumps(result.best_params, sort_keys=True)}",
        f"Output dir: {result.output_dir}",
        f"Summary JSON: {result.summary_json_path}",
        f"Candidates CSV: {result.candidates_csv_path}",
    ]
    if result.preview_video_path:
        lines.append(f"Preview video: {result.preview_video_path}")

    lines.append("")
    lines.append("Top candidates:")
    for candidate in result.top_candidates[: max(1, top_k)]:
        lines.append(
            (
                f"{candidate.rank}. score={candidate.score:.4f}, detected={candidate.mean_detected:.3f}, "
                f"rejected={candidate.mean_rejected:.3f}, stability={candidate.stability:.3f}, "
                f"fps={candidate.eval_fps:.2f}, params={json.dumps(candidate.params, sort_keys=True)}"
            )
        )
    return "\n".join(lines)


def apply_best_params_to_config(config: dict, best_params: dict[str, float | int]) -> dict:
    updated = deepcopy(config)
    tracking = updated.setdefault("tracking", {})
    tracking["aruco_params"] = dict(best_params)
    return updated


def legacy_entrypoint(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    def _parse_csv_values(raw: str, label: str, value_type: str) -> list[float | int]:
        text = str(raw or "").strip()
        if not text:
            return []
        tokens = [token.strip() for token in text.split(",") if token.strip()]
        if not tokens:
            raise ValueError(f"{label} is empty.")
        out: list[float | int] = []
        for token in tokens:
            if value_type == "float":
                out.append(float(token))
            elif value_type == "int":
                value_float = float(token)
                if not value_float.is_integer():
                    raise ValueError(f"{label} requires whole numbers, got: {token}")
                out.append(int(value_float))
            else:
                raise ValueError(f"Unsupported parse type: {value_type}")
        return out

    parser = argparse.ArgumentParser(
        description=(
            "Deprecated entry point. Use 'python3 bbx.py optimize-tracking' instead."
        )
    )
    parser.add_argument("--input", required=True, help="Video file or image directory for optimization.")
    parser.add_argument(
        "--profile",
        choices=sorted(VALID_PROFILES),
        default=DEFAULT_PROFILE,
        help="Grid profile size.",
    )
    parser.add_argument("--sample-frames", type=int, default=80, help="Number of frames/images to sample.")
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY, help="ArUco dictionary (for example 4X4_50).")
    parser.add_argument("--tag-size-mm", type=float, default=DEFAULT_TAG_SIZE_MM, help="Physical tag size in mm.")
    parser.add_argument(
        "--sweep-min-marker-perimeter-rate",
        default="",
        help="Optional comma-separated minMarkerPerimeterRate override values.",
    )
    parser.add_argument(
        "--sweep-adaptive-thresh-win-size-min",
        default="",
        help="Optional comma-separated adaptiveThreshWinSizeMin override values.",
    )
    parser.add_argument(
        "--sweep-adaptive-thresh-win-size-max",
        default="",
        help="Optional comma-separated adaptiveThreshWinSizeMax override values.",
    )
    parser.add_argument(
        "--sweep-adaptive-thresh-win-size-step",
        default="",
        help="Optional comma-separated adaptiveThreshWinSizeStep override values.",
    )
    parser.add_argument(
        "--execution-target",
        choices=sorted(VALID_EXECUTION_TARGETS),
        default=DEFAULT_EXECUTION_TARGET,
        help="Run mode for worker defaults.",
    )
    parser.add_argument("--workers", type=int, help="Override worker count.")
    parser.add_argument("--expected-tags", type=float, help="Optional expected tag count per frame.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop after this many non-improving evaluations (0 disables early stop).",
    )
    parser.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=DEFAULT_EARLY_STOP_MIN_IMPROVEMENT,
        help="Minimum score increase considered an improvement for early stop.",
    )
    parser.add_argument("--output-dir", help="Optional output root directory.")
    parser.add_argument("--write-preview", action="store_true", help="Write a short preview video for best params.")
    parser.add_argument("--preview-frames", type=int, default=240, help="Max frames in preview video.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to print.")
    args = parser.parse_args(argv)

    print(
        "DEPRECATED: tracking-optimization.0.6.py is deprecated. "
        "Use 'python3 bbx.py optimize-tracking ...' instead."
    )
    try:
        sweep_overrides = {}
        min_perimeter = _parse_csv_values(
            args.sweep_min_marker_perimeter_rate,
            "--sweep-min-marker-perimeter-rate",
            "float",
        )
        if min_perimeter:
            sweep_overrides["minMarkerPerimeterRate"] = min_perimeter

        win_min = _parse_csv_values(
            args.sweep_adaptive_thresh_win_size_min,
            "--sweep-adaptive-thresh-win-size-min",
            "int",
        )
        if win_min:
            sweep_overrides["adaptiveThreshWinSizeMin"] = win_min

        win_max = _parse_csv_values(
            args.sweep_adaptive_thresh_win_size_max,
            "--sweep-adaptive-thresh-win-size-max",
            "int",
        )
        if win_max:
            sweep_overrides["adaptiveThreshWinSizeMax"] = win_max

        win_step = _parse_csv_values(
            args.sweep_adaptive_thresh_win_size_step,
            "--sweep-adaptive-thresh-win-size-step",
            "int",
        )
        if win_step:
            sweep_overrides["adaptiveThreshWinSizeStep"] = win_step

        result = optimize_tracking(
            input_path=args.input,
            profile=args.profile,
            sample_frames=args.sample_frames,
            dictionary_name=args.dictionary,
            tag_size_mm=args.tag_size_mm,
            sweep_overrides=sweep_overrides or None,
            execution_target=args.execution_target,
            workers=args.workers,
            expected_tags=args.expected_tags,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_improvement=args.early_stop_min_improvement,
            output_dir=args.output_dir,
            write_preview=args.write_preview,
            preview_frames=args.preview_frames,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Optimization failed: {exc}")
        return 1

    print(format_optimization_report(result, top_k=args.top_k))
    return 0
