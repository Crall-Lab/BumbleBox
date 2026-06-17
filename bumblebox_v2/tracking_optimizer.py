from __future__ import annotations

import concurrent.futures
import csv
import itertools
import json
import math
import os
import statistics
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2


DEFAULT_DICTIONARY = "4X4_50"
DEFAULT_PROFILE = "quick"
DEFAULT_EXECUTION_TARGET = "pi_safe"
DEFAULT_TAG_SIZE_MM = 2.5
DEFAULT_EARLY_STOP_PATIENCE = 0
DEFAULT_EARLY_STOP_MIN_IMPROVEMENT = 0.0
DEFAULT_MAX_MARKER_PERIMETER_RATE = 4.0
SCORE_DETECTION_WEIGHT = 1.0
SCORE_STABILITY_WEIGHT = 0.10
SCORE_FPS_WEIGHT = 0.10
SCORE_REJECTED_WEIGHT = 0.005
SCORE_STD_DETECTED_WEIGHT = 0.01
SCORE_EXPECTED_ERROR_WEIGHT = 0.10

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mjpeg", ".avi", ".mov", ".mkv"}
GENERATED_OPTIMIZER_DIR_NAMES = {"tracking_optimization", "top_candidate_review"}
VALID_PROFILES = {"quick", "balanced", "deep", "daily"}
VALID_EXECUTION_TARGETS = {"pi_safe", "desktop"}

PROFILE_PARAMETER_SPACE = {
    "quick": {
        "minMarkerPerimeterRate": [0.015, 0.02, 0.03],
        "maxMarkerPerimeterRate": [DEFAULT_MAX_MARKER_PERIMETER_RATE],
        "adaptiveThreshWinSizeMin": [3, 5],
        "adaptiveThreshWinSizeMax": [23, 31],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.05, 0.08],
        "adaptiveThreshConstant": [7],
    },
    "balanced": {
        "minMarkerPerimeterRate": [0.01, 0.015, 0.02, 0.03],
        "maxMarkerPerimeterRate": [DEFAULT_MAX_MARKER_PERIMETER_RATE],
        "adaptiveThreshWinSizeMin": [3, 5, 7],
        "adaptiveThreshWinSizeMax": [21, 31, 41],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.04, 0.06, 0.08],
        "adaptiveThreshConstant": [7],
    },
    "deep": {
        "minMarkerPerimeterRate": [0.008, 0.012, 0.016, 0.02, 0.03],
        "maxMarkerPerimeterRate": [DEFAULT_MAX_MARKER_PERIMETER_RATE],
        "adaptiveThreshWinSizeMin": [3, 5, 7],
        "adaptiveThreshWinSizeMax": [21, 31, 41],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.04, 0.05, 0.06, 0.08],
        "adaptiveThreshConstant": [5, 7],
    },
    "daily": {
        "minMarkerPerimeterRate": [0.019153],
        "maxMarkerPerimeterRate": [0.052808],
        "adaptiveThreshWinSizeMin": [3, 5],
        "adaptiveThreshWinSizeMax": [29, 36, 41, 57, 73, 81, 105, 127, 151],
        "adaptiveThreshWinSizeStep": [2, 3],
        "polygonalApproxAccuracyRate": [0.06, 0.08],
        "adaptiveThreshConstant": [1, 3, 5, 7, 9],
    },
}
VALID_SWEEP_OVERRIDE_KEYS = {
    key
    for space in PROFILE_PARAMETER_SPACE.values()
    for key in space
}
FLOAT_SWEEP_OVERRIDE_KEYS = {
    "minMarkerPerimeterRate",
    "maxMarkerPerimeterRate",
    "polygonalApproxAccuracyRate",
}
OPTIMIZED_ARUCO_PARAM_KEYS = (
    "minMarkerPerimeterRate",
    "maxMarkerPerimeterRate",
    "adaptiveThreshWinSizeMin",
    "adaptiveThreshWinSizeMax",
    "adaptiveThreshWinSizeStep",
    "polygonalApproxAccuracyRate",
    "adaptiveThreshConstant",
)
CANDIDATE_TABLE_PARAM_COLUMNS = (
    ("minMarkerPerimeterRate", "minPerim", 8, "float"),
    ("maxMarkerPerimeterRate", "maxPerim", 8, "float"),
    ("adaptiveThreshWinSizeMin", "winMin", 6, "int"),
    ("adaptiveThreshWinSizeMax", "winMax", 6, "int"),
    ("adaptiveThreshWinSizeStep", "winStep", 7, "int"),
    ("polygonalApproxAccuracyRate", "poly", 7, "float"),
    ("adaptiveThreshConstant", "const", 5, "int"),
)


ProgressCandidateSnapshot = dict[str, Any]
ProgressCallback = Callable[
    [int, int, list[ProgressCandidateSnapshot], ProgressCandidateSnapshot],
    None,
]
StopRequestedCallback = Callable[[], bool]


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
    mean_decoded: float = 0.0
    mean_filtered: float = 0.0


@dataclass
class TrackingOptimizationResult:
    created_at: str
    input_path: str
    input_type: str
    dictionary: str
    tag_size_mm: float
    profile: str
    parameter_source: str
    execution_target: str
    workers: int
    sample_frames_requested: int
    sample_frames_used: int
    parameter_combinations_total: int
    combinations_evaluated: int
    early_stop_patience: int
    early_stop_min_improvement: float
    early_stopped: bool
    stopped_by_user: bool
    sweep_overrides: dict[str, list[float | int]]
    output_dir: str
    summary_json_path: str
    candidates_csv_path: str
    preview_video_path: Optional[str]
    review_manifest_json_path: Optional[str]
    best_params: dict[str, float | int]
    best_score: float
    best_mean_detected: float
    highest_detection_candidate: OptimizationCandidate
    top_detection_candidates: list[OptimizationCandidate]
    top_candidates: list[OptimizationCandidate]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["highest_detection_candidate"] = asdict(self.highest_detection_candidate)
        payload["top_detection_candidates"] = [asdict(item) for item in self.top_detection_candidates]
        payload["top_candidates"] = [asdict(item) for item in self.top_candidates]
        return payload


@dataclass
class IterativeTrackingRefinementResult:
    created_at: str
    input_path: str
    output_dir: str
    rounds_requested: int
    rounds_completed: int
    seed_candidate_count: int
    sample_frames: int
    validation_sample_frames: int
    summary_json_path: str
    round_results: list[TrackingOptimizationResult]
    validation_result: Optional[TrackingOptimizationResult]
    final_result: TrackingOptimizationResult
    stopped_by_user: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["round_results"] = [item.to_dict() for item in self.round_results]
        payload["validation_result"] = (
            self.validation_result.to_dict() if self.validation_result else None
        )
        payload["final_result"] = self.final_result.to_dict()
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


def _resolve_user_path(path_value: str | Path) -> Path:
    text = str(path_value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return Path(text).expanduser().resolve()


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


def resolve_profile_parameter_space(
    profile: str,
    tag_size_mm: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, list[float | int]]:
    profile_key = str(profile).strip().lower()
    if profile_key not in VALID_PROFILES:
        raise ValueError(f"profile must be one of {sorted(VALID_PROFILES)}, got: {profile}")
    if tag_size_mm <= 0:
        raise ValueError("tag_size_mm must be > 0")

    space = deepcopy(PROFILE_PARAMETER_SPACE[profile_key])
    if profile_key != "daily":
        space["minMarkerPerimeterRate"] = _min_marker_rates_for_tag_size(
            tag_size_mm=tag_size_mm,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    return space


def build_parameter_grid(
    profile: str,
    tag_size_mm: float,
    frame_width: int,
    frame_height: int,
    sweep_overrides: Optional[dict[str, Sequence[float | int]]] = None,
) -> list[dict[str, float | int]]:
    space = resolve_profile_parameter_space(
        profile=profile,
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

            if key in FLOAT_SWEEP_OVERRIDE_KEYS:
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
        if float(params.get("maxMarkerPerimeterRate", DEFAULT_MAX_MARKER_PERIMETER_RATE)) <= float(params["minMarkerPerimeterRate"]):
            continue
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


def limit_parameter_grid(
    param_grid: Sequence[dict[str, float | int]],
    max_combinations: Optional[int],
) -> list[dict[str, float | int]]:
    grid = list(param_grid)
    if max_combinations is None or int(max_combinations) <= 0:
        return grid
    max_count = int(max_combinations)
    if len(grid) <= max_count:
        return grid
    if max_count == 1:
        return [grid[0]]

    # Evenly thin the deterministic product grid so broad parameter coverage is preserved.
    selected_indices: list[int] = []
    used: set[int] = set()
    span = len(grid) - 1
    for idx in range(max_count):
        selected = int(round((idx * span) / float(max_count - 1)))
        while selected in used and selected < len(grid) - 1:
            selected += 1
        while selected in used and selected > 0:
            selected -= 1
        if selected in used:
            continue
        used.add(selected)
        selected_indices.append(selected)

    cursor = 0
    while len(selected_indices) < max_count and cursor < len(grid):
        if cursor not in used:
            selected_indices.append(cursor)
            used.add(cursor)
        cursor += 1
    return [grid[index] for index in sorted(selected_indices[:max_count])]


def _candidate_param_key(params: dict[str, float | int]) -> tuple[float | int, ...]:
    return tuple(params[key] for key in OPTIMIZED_ARUCO_PARAM_KEYS)


def _normalize_candidate_params(params: dict[str, Any]) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {}
    if "maxMarkerPerimeterRate" not in params:
        params["maxMarkerPerimeterRate"] = DEFAULT_MAX_MARKER_PERIMETER_RATE
    missing = [key for key in OPTIMIZED_ARUCO_PARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"candidate params missing required keys: {', '.join(missing)}")

    for key in OPTIMIZED_ARUCO_PARAM_KEYS:
        value = params[key]
        if key in FLOAT_SWEEP_OVERRIDE_KEYS:
            parsed = round(float(value), 6)
            if parsed <= 0:
                raise ValueError(f"{key} must be > 0")
            normalized[key] = parsed
            continue

        parsed_float = float(value)
        if not parsed_float.is_integer():
            raise ValueError(f"{key} must be a whole number")
        parsed_int = int(parsed_float)
        if parsed_int <= 0:
            raise ValueError(f"{key} must be >= 1")
        normalized[key] = parsed_int

    win_min = int(normalized["adaptiveThreshWinSizeMin"])
    win_max = int(normalized["adaptiveThreshWinSizeMax"])
    step = int(normalized["adaptiveThreshWinSizeStep"])
    if float(normalized["maxMarkerPerimeterRate"]) <= float(normalized["minMarkerPerimeterRate"]):
        raise ValueError("maxMarkerPerimeterRate must be greater than minMarkerPerimeterRate")
    if win_min < 3 or win_max < 3:
        raise ValueError("adaptive threshold window min/max values must be >= 3")
    if win_min >= win_max:
        raise ValueError("adaptiveThreshWinSizeMin must be less than adaptiveThreshWinSizeMax")
    if (win_max - win_min) < step:
        raise ValueError("adaptive threshold window range must be at least one step wide")
    return normalized


def normalize_candidate_param_grid(
    candidate_param_grid: Sequence[dict[str, Any]],
) -> list[dict[str, float | int]]:
    normalized_grid: list[dict[str, float | int]] = []
    seen: set[tuple[float | int, ...]] = set()
    for params in candidate_param_grid:
        normalized = _normalize_candidate_params(dict(params))
        key = _candidate_param_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized_grid.append(normalized)
    if not normalized_grid:
        raise RuntimeError("Candidate parameter grid is empty after validation.")
    return normalized_grid


def _round_float_values(values: Sequence[float]) -> list[float]:
    return sorted({round(max(0.000001, float(value)), 6) for value in values})


def _round_int_values(values: Sequence[int]) -> list[int]:
    return sorted({max(1, int(value)) for value in values})


def _refinement_float_values(key: str, value: float, round_index: int) -> list[float]:
    if key == "minMarkerPerimeterRate":
        if round_index <= 1:
            multipliers = (0.5, 0.75, 1.0, 1.25, 1.5)
        elif round_index == 2:
            multipliers = (0.8, 0.9, 1.0, 1.1, 1.2)
        else:
            multipliers = (0.9, 0.95, 1.0, 1.05, 1.1)
        return _round_float_values(min(0.08, max(0.0005, value * factor)) for factor in multipliers)

    if key == "maxMarkerPerimeterRate":
        if round_index <= 1:
            multipliers = (0.75, 0.9, 1.0, 1.15, 1.35)
        elif round_index == 2:
            multipliers = (0.9, 0.97, 1.0, 1.03, 1.1)
        else:
            multipliers = (0.95, 1.0, 1.05)
        return _round_float_values(
            min(DEFAULT_MAX_MARKER_PERIMETER_RATE, max(0.001, value * factor))
            for factor in multipliers
        )

    if key == "polygonalApproxAccuracyRate":
        if round_index <= 1:
            offsets = (-0.02, -0.01, 0.0, 0.01, 0.02)
        elif round_index == 2:
            offsets = (-0.008, -0.004, 0.0, 0.004, 0.008)
        else:
            offsets = (-0.004, -0.002, 0.0, 0.002, 0.004)
        return _round_float_values(min(0.20, max(0.001, value + offset)) for offset in offsets)

    raise ValueError(f"Unsupported refinement float key: {key}")


def _refinement_int_values(key: str, value: int, round_index: int) -> list[int]:
    if key == "adaptiveThreshWinSizeMin":
        offsets = (-4, -2, 0, 2, 4) if round_index <= 1 else (-2, 0, 2)
    elif key == "adaptiveThreshWinSizeMax":
        offsets = (-8, -4, 0, 4, 8) if round_index <= 1 else (-4, 0, 4)
    elif key == "adaptiveThreshWinSizeStep":
        offsets = (-2, -1, 0, 1, 2) if round_index <= 1 else (-1, 0, 1)
    elif key == "adaptiveThreshConstant":
        offsets = (-4, -2, 0, 2, 4) if round_index <= 1 else (-2, -1, 0, 1, 2)
    else:
        raise ValueError(f"Unsupported refinement integer key: {key}")
    return _round_int_values(value + offset for offset in offsets)


def build_refinement_parameter_grid(
    seed_params: Sequence[dict[str, Any]],
    *,
    round_index: int = 1,
) -> list[dict[str, float | int]]:
    if round_index <= 0:
        raise ValueError("round_index must be >= 1")
    if not seed_params:
        raise ValueError("At least one seed candidate is required for refinement.")

    candidates: list[dict[str, float | int]] = []
    seen: set[tuple[float | int, ...]] = set()

    def add_variant(raw_params: dict[str, Any]) -> None:
        try:
            normalized = _normalize_candidate_params(raw_params)
        except ValueError:
            return
        key = _candidate_param_key(normalized)
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    for raw_seed in seed_params:
        seed = _normalize_candidate_params(dict(raw_seed))
        add_variant(seed)

        min_rate_values = _refinement_float_values(
            "minMarkerPerimeterRate",
            float(seed["minMarkerPerimeterRate"]),
            round_index,
        )
        poly_values = _refinement_float_values(
            "polygonalApproxAccuracyRate",
            float(seed["polygonalApproxAccuracyRate"]),
            round_index,
        )
        for min_rate in min_rate_values:
            for poly in poly_values:
                variant = dict(seed)
                variant["minMarkerPerimeterRate"] = min_rate
                variant["polygonalApproxAccuracyRate"] = poly
                add_variant(variant)

        max_rate_values = _refinement_float_values(
            "maxMarkerPerimeterRate",
            float(seed["maxMarkerPerimeterRate"]),
            round_index,
        )
        for max_rate in max_rate_values:
            variant = dict(seed)
            variant["maxMarkerPerimeterRate"] = max_rate
            add_variant(variant)

        win_min_values = _refinement_int_values(
            "adaptiveThreshWinSizeMin",
            int(seed["adaptiveThreshWinSizeMin"]),
            round_index,
        )
        win_max_values = _refinement_int_values(
            "adaptiveThreshWinSizeMax",
            int(seed["adaptiveThreshWinSizeMax"]),
            round_index,
        )
        for win_min in win_min_values:
            for win_max in win_max_values:
                variant = dict(seed)
                variant["adaptiveThreshWinSizeMin"] = win_min
                variant["adaptiveThreshWinSizeMax"] = win_max
                add_variant(variant)

        for key in ("adaptiveThreshWinSizeStep", "adaptiveThreshConstant"):
            for value in _refinement_int_values(key, int(seed[key]), round_index):
                variant = dict(seed)
                variant[key] = value
                add_variant(variant)

    if not candidates:
        raise RuntimeError("Refinement parameter grid is empty after validation.")
    return candidates


def suggest_min_marker_perimeter_rates_from_measurement(perimeter_rate: float) -> list[float]:
    measured = float(perimeter_rate)
    if measured <= 0:
        raise ValueError("perimeter_rate must be > 0")
    # Measured tags should be the smallest real tags users care about detecting.
    # Keep the guard just below that measured size without sweeping much smaller false positives.
    return [round(min(0.08, max(0.0005, measured * 0.90)), 6)]


def suggest_marker_perimeter_rate_sweeps_from_measurements(
    smallest_perimeter_rate: float,
    largest_perimeter_rate: float,
) -> tuple[list[float], list[float]]:
    smallest = float(smallest_perimeter_rate)
    largest = float(largest_perimeter_rate)
    if smallest <= 0 or largest <= 0:
        raise ValueError("smallest and largest perimeter rates must be > 0")
    if largest < smallest:
        smallest, largest = largest, smallest

    min_values = suggest_min_marker_perimeter_rates_from_measurement(smallest)
    max_values = [round(min(DEFAULT_MAX_MARKER_PERIMETER_RATE, max(0.001, largest * 1.10)), 6)]
    max_values = [value for value in max_values if value > min_values[-1]]
    if not max_values:
        max_values = [round(min(DEFAULT_MAX_MARKER_PERIMETER_RATE, max(min_values[-1] * 1.5, largest * 1.25)), 6)]
    return min_values, max_values


def load_candidate_params_from_scores(path: str | Path, *, limit: int = 5) -> list[dict[str, float | int]]:
    resolved = _resolve_user_path(path)
    if resolved.is_dir():
        resolved = resolved / "candidate_scores.csv"
    if not resolved.exists():
        raise FileNotFoundError(f"Candidate score file does not exist: {resolved}")

    if resolved.suffix.lower() == ".json":
        with resolved.open() as f:
            payload = json.load(f)
        raw_candidates = payload.get("top_candidates") or []
        params = [candidate.get("params", {}) for candidate in raw_candidates[:limit]]
        return normalize_candidate_param_grid(params)

    candidates: list[tuple[int, float, dict[str, Any]]] = []
    with resolved.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            params_json = row.get("params_json") or "{}"
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid params_json in {resolved}: {params_json}") from exc
            rank_text = row.get("rank") or "999999"
            score_text = row.get("score") or "0"
            candidates.append((int(float(rank_text)), float(score_text), params))

    if not candidates:
        raise RuntimeError(f"No candidates found in {resolved}")
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return normalize_candidate_param_grid([params for _rank, _score, params in candidates[:limit]])


def _classify_input_path(path: Path) -> str:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension for {path}.")
        return "video"
    if path.is_dir():
        return "image_dir"
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _is_generated_optimizer_artifact_path(path: Path, *, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    root_name = root.name
    return any(
        part in GENERATED_OPTIMIZER_DIR_NAMES
        or part.startswith("optimize_tracking_")
        or part.startswith("iterative_tracking_refinement_")
        for part in (root_name, *parts)
    )


def find_supported_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not _is_generated_optimizer_artifact_path(path, root=image_dir)
    )


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


def _load_sample_frames_from_video(video_path: Path, sample_count: int) -> tuple[list, int, list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for optimization: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    used_indices: list[int] = []

    if total_frames > 0:
        indices = _sample_indices(total_frames, sample_count)
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            used_indices.append(frame_idx)
    else:
        next_index = 0
        while len(frames) < sample_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            used_indices.append(next_index)
            next_index += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No readable frames found in video: {video_path}")
    return frames, max(total_frames, len(frames)), used_indices


def _load_sample_frames_from_image_dir(image_dir: Path, sample_count: int) -> tuple[list, int, list[int]]:
    all_images = find_supported_image_paths(image_dir)
    if not all_images:
        raise RuntimeError(f"No supported image files found recursively in directory: {image_dir}")

    indices = _sample_indices(len(all_images), sample_count)
    frames = []
    used_indices: list[int] = []
    for idx in indices:
        image = cv2.imread(str(all_images[idx]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        frames.append(image)
        used_indices.append(idx)

    if not frames:
        raise RuntimeError(f"All sampled images failed to load from: {image_dir}")
    return frames, len(all_images), used_indices


def load_sample_frames_with_indices(input_path: str | Path, sample_count: int) -> tuple[list, str, int, list[int]]:
    path = _resolve_user_path(input_path)
    input_type = _classify_input_path(path)
    if input_type == "video":
        frames, total_count, sample_indices = _load_sample_frames_from_video(path, sample_count)
    else:
        frames, total_count, sample_indices = _load_sample_frames_from_image_dir(path, sample_count)
    return frames, input_type, total_count, sample_indices


def load_sample_frames(input_path: str | Path, sample_count: int) -> tuple[list, str, int]:
    frames, input_type, total_count, _sample_indices = load_sample_frames_with_indices(input_path, sample_count)
    return frames, input_type, total_count


def _score_candidate(
    mean_detected: float,
    mean_rejected: float,
    std_detected: float,
    stability: float,
    eval_fps: float,
    expected_tags: Optional[float],
) -> float:
    score = SCORE_DETECTION_WEIGHT * mean_detected
    score += SCORE_STABILITY_WEIGHT * stability
    score += SCORE_FPS_WEIGHT * min(eval_fps, 90.0) / 90.0
    score -= SCORE_REJECTED_WEIGHT * mean_rejected
    score -= SCORE_STD_DETECTED_WEIGHT * std_detected
    if expected_tags is not None and expected_tags > 0:
        expected_error = abs(mean_detected - expected_tags) / expected_tags
        score -= SCORE_EXPECTED_ERROR_WEIGHT * expected_error
    return score


def _normalized_valid_tag_ids(valid_tag_ids: Optional[Iterable[int]]) -> Optional[set[int]]:
    if valid_tag_ids is None:
        return None
    out: set[int] = set()
    for raw_id in valid_tag_ids:
        try:
            out.add(int(raw_id))
        except (TypeError, ValueError):
            continue
    return out if out else None


def _perimeter_filter_bounds(
    params: dict[str, float | int],
    explicit_bounds: Optional[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    if explicit_bounds is not None:
        min_rate, max_rate = explicit_bounds
        return float(min_rate), float(max_rate)
    try:
        min_rate = float(params.get("minMarkerPerimeterRate", 0.0) or 0.0)
        max_rate = float(params.get("maxMarkerPerimeterRate", DEFAULT_MAX_MARKER_PERIMETER_RATE) or 0.0)
    except (TypeError, ValueError):
        return None
    if min_rate <= 0 and max_rate <= 0:
        return None
    return min_rate, max_rate


def _valid_decoded_marker_ids(
    *,
    corners: Any,
    ids: Any,
    frame_width: int,
    frame_height: int,
    valid_tag_ids: Optional[set[int]],
    perimeter_bounds: Optional[tuple[float, float]],
) -> tuple[set[int], set[int], int]:
    decoded_ids: set[int] = set()
    valid_ids: set[int] = set()
    filtered_count = 0
    if ids is None or len(ids) <= 0:
        return decoded_ids, valid_ids, filtered_count

    for corner, marker_id_raw in zip(corners, ids.flatten().tolist()):
        try:
            marker_id = int(marker_id_raw)
        except (TypeError, ValueError):
            filtered_count += 1
            continue
        decoded_ids.add(marker_id)

        if valid_tag_ids is not None and marker_id not in valid_tag_ids:
            filtered_count += 1
            continue

        if perimeter_bounds is not None:
            min_rate, max_rate = perimeter_bounds
            perimeter_rate = _corner_perimeter_rate(
                corner,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if min_rate > 0 and perimeter_rate < min_rate:
                filtered_count += 1
                continue
            if max_rate > 0 and perimeter_rate > max_rate:
                filtered_count += 1
                continue

        valid_ids.add(marker_id)
    return decoded_ids, valid_ids, filtered_count


def _evaluate_candidate(
    params: dict[str, float | int],
    frames: Sequence,
    dictionary_name: str,
    expected_tags: Optional[float],
    valid_tag_ids: Optional[set[int]] = None,
    perimeter_filter_bounds: Optional[tuple[float, float]] = None,
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
    decoded_counts = []
    filtered_counts = []
    rejected_counts = []
    stability_scores = []
    unique_ids = set()
    previous_ids = set()
    bounds = _perimeter_filter_bounds(params, perimeter_filter_bounds)

    start = time.perf_counter()
    for frame in frames:
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = gray.shape[:2]

        corners, ids, rejected = detector.detectMarkers(gray)
        decoded_ids, ids_set, filtered_count = _valid_decoded_marker_ids(
            corners=corners,
            ids=ids,
            frame_width=frame_width,
            frame_height=frame_height,
            valid_tag_ids=valid_tag_ids,
            perimeter_bounds=bounds,
        )
        unique_ids.update(ids_set)

        detected = len(ids_set)
        decoded_count = len(decoded_ids)
        rejected_count = len(rejected) if rejected is not None else 0
        detected_counts.append(detected)
        decoded_counts.append(decoded_count)
        filtered_counts.append(filtered_count)
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
    mean_decoded = statistics.fmean(decoded_counts) if decoded_counts else 0.0
    mean_filtered = statistics.fmean(filtered_counts) if filtered_counts else 0.0
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
        mean_decoded=mean_decoded,
        mean_filtered=mean_filtered,
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
        "mean_decoded",
        "mean_filtered",
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
                    "mean_decoded": f"{item.mean_decoded:.6f}",
                    "mean_filtered": f"{item.mean_filtered:.6f}",
                    "std_detected": f"{item.std_detected:.6f}",
                    "mean_rejected": f"{item.mean_rejected:.6f}",
                    "stability": f"{item.stability:.6f}",
                    "unique_ids": item.unique_ids,
                    "eval_fps": f"{item.eval_fps:.6f}",
                    "runtime_seconds": f"{item.runtime_seconds:.6f}",
                    "params_json": json.dumps(item.params, sort_keys=True),
                }
            )


def _top_candidate_snapshots(
    candidates: Sequence[OptimizationCandidate],
    *,
    limit: int = 5,
) -> list[ProgressCandidateSnapshot]:
    ranked = sorted(
        candidates,
        key=lambda item: (item.score, item.mean_detected, -item.mean_rejected, item.eval_fps),
        reverse=True,
    )
    snapshots = []
    for rank, candidate in enumerate(ranked[: max(1, limit)], start=1):
        snapshots.append(
            {
                "rank": rank,
                "score": candidate.score,
                "mean_detected": candidate.mean_detected,
                "mean_decoded": candidate.mean_decoded,
                "mean_filtered": candidate.mean_filtered,
                "std_detected": candidate.std_detected,
                "mean_rejected": candidate.mean_rejected,
                "stability": candidate.stability,
                "unique_ids": candidate.unique_ids,
                "eval_fps": candidate.eval_fps,
                "runtime_seconds": candidate.runtime_seconds,
                "params": dict(candidate.params),
            }
        )
    return snapshots


def _score_rank_lookup(candidates: Sequence[OptimizationCandidate]) -> dict[int, int]:
    ranked_by_score = sorted(
        candidates,
        key=lambda item: (item.score, item.mean_detected, -item.mean_rejected, item.eval_fps),
        reverse=True,
    )
    return {id(candidate): rank for rank, candidate in enumerate(ranked_by_score, start=1)}


def _top_detection_candidates(
    candidates: Sequence[OptimizationCandidate],
    *,
    limit: int = 5,
) -> list[OptimizationCandidate]:
    if not candidates:
        return []
    # Detection ties are resolved by score, fewer rejected candidates, stability, then speed.
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.mean_detected,
            item.score,
            -item.mean_rejected,
            item.stability,
            item.eval_fps,
        ),
        reverse=True,
    )
    return ranked[: max(1, limit)]


def _top_detection_candidate_snapshots(
    candidates: Sequence[OptimizationCandidate],
) -> list[ProgressCandidateSnapshot]:
    if not candidates:
        return []
    score_ranks = _score_rank_lookup(candidates)
    snapshots = []
    for detection_rank, candidate in enumerate(_top_detection_candidates(candidates, limit=5), start=1):
        snapshot = _candidate_progress_snapshot(candidate, rank=score_ranks.get(id(candidate), 0))
        snapshot["detection_rank"] = detection_rank
        snapshots.append(snapshot)
    return snapshots


def _highest_detection_candidate_snapshot(
    candidates: Sequence[OptimizationCandidate],
) -> Optional[ProgressCandidateSnapshot]:
    snapshots = _top_detection_candidate_snapshots(candidates)
    return snapshots[0] if snapshots else None


def _candidate_progress_snapshot(
    candidate: OptimizationCandidate,
    *,
    rank: int,
) -> ProgressCandidateSnapshot:
    return {
        "rank": rank,
        "score": candidate.score,
        "mean_detected": candidate.mean_detected,
        "mean_decoded": candidate.mean_decoded,
        "mean_filtered": candidate.mean_filtered,
        "std_detected": candidate.std_detected,
        "mean_rejected": candidate.mean_rejected,
        "stability": candidate.stability,
        "unique_ids": candidate.unique_ids,
        "eval_fps": candidate.eval_fps,
        "runtime_seconds": candidate.runtime_seconds,
        "params": dict(candidate.params),
    }


def _candidate_table_value(candidate: object, key: str, default: object = "") -> object:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _candidate_table_params(candidate: object) -> dict[str, object]:
    params = _candidate_table_value(candidate, "params", {})
    return params if isinstance(params, dict) else {}


def _format_candidate_table_float(value: object, width: int, precision: int) -> str:
    if value in ("", None):
        return f"{'':>{width}}"
    try:
        return f"{float(value):>{width}.{precision}f}"
    except Exception:
        return f"{str(value):>{width}}"


def _format_candidate_table_param(value: object, width: int, kind: str) -> str:
    if value in ("", None):
        return f"{'':>{width}}"
    try:
        if kind == "int":
            return f"{int(float(value)):>{width}}"
        return f"{float(value):>{width}.4g}"
    except Exception:
        return f"{str(value):>{width}}"


def format_candidate_results_table(
    candidates: Sequence[object],
    *,
    ranking: str = "score",
    max_rows: int = 5,
) -> str:
    rows = list(candidates[: max(1, int(max_rows))])
    if not rows:
        return "No candidates have finished yet."

    detection_table = ranking == "detection"
    header_parts = (
        [
            f"{'det#':>4}",
            f"{'score#':>6}",
        ]
        if detection_table
        else [f"{'#':>4}"]
    )
    header_parts.extend(
        [
            f"{'score':>9}",
            f"{'detect':>7}",
            f"{'decoded':>7}",
            f"{'filt':>6}",
            f"{'std':>6}",
            f"{'reject':>8}",
            f"{'stable':>7}",
            f"{'ms/frame':>8}",
            f"{'test_s':>7}",
        ]
    )
    for _key, label, width, _kind in CANDIDATE_TABLE_PARAM_COLUMNS:
        header_parts.append(f"{label:>{width}}")

    header = "  ".join(header_parts)
    lines = [header, "-" * len(header)]

    for row_index, candidate in enumerate(rows, start=1):
        params = _candidate_table_params(candidate)
        eval_fps = _candidate_table_value(candidate, "eval_fps", 0.0)
        try:
            avg_frame_ms = (1000.0 / float(eval_fps)) if float(eval_fps) > 0 else 0.0
        except Exception:
            avg_frame_ms = 0.0

        if detection_table:
            detection_rank = _candidate_table_value(candidate, "detection_rank", row_index)
            score_rank = _candidate_table_value(candidate, "rank", 0)
            row_parts = [
                f"{int(detection_rank or row_index):>4}",
                f"{int(score_rank or 0):>6}",
            ]
        else:
            rank = _candidate_table_value(candidate, "rank", row_index)
            row_parts = [f"{int(rank or row_index):>4}"]

        row_parts.extend(
            [
                _format_candidate_table_float(_candidate_table_value(candidate, "score", 0.0), 9, 4),
                _format_candidate_table_float(_candidate_table_value(candidate, "mean_detected", 0.0), 7, 3),
                _format_candidate_table_float(_candidate_table_value(candidate, "mean_decoded", 0.0), 7, 3),
                _format_candidate_table_float(_candidate_table_value(candidate, "mean_filtered", 0.0), 6, 3),
                _format_candidate_table_float(_candidate_table_value(candidate, "std_detected", ""), 6, 3),
                _format_candidate_table_float(_candidate_table_value(candidate, "mean_rejected", 0.0), 8, 3),
                _format_candidate_table_float(_candidate_table_value(candidate, "stability", 0.0), 7, 3),
                f"{avg_frame_ms:>8.2f}",
                _format_candidate_table_float(_candidate_table_value(candidate, "runtime_seconds", 0.0), 7, 2),
            ]
        )
        for key, _label, width, kind in CANDIDATE_TABLE_PARAM_COLUMNS:
            row_parts.append(_format_candidate_table_param(params.get(key, ""), width, kind))
        lines.append("  ".join(row_parts))

    return "\n".join(lines)


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

    image_paths = find_supported_image_paths(input_path)
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield frame, 6.0


def _corner_perimeter_rate(corner: Any, *, frame_width: int, frame_height: int) -> float:
    points = corner.reshape(-1, 2)
    if len(points) < 4:
        return 0.0
    perimeter = 0.0
    for idx in range(4):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % 4]
        perimeter += math.hypot(float(x2) - float(x1), float(y2) - float(y1))
    return perimeter / float(max(1, max(frame_width, frame_height)))


def _draw_perimeter_flag(
    frame_bgr: Any,
    corner: Any,
    *,
    marker_id: int,
    perimeter_rate: float,
    label: str,
    color: tuple[int, int, int],
) -> None:
    points = corner.reshape(-1, 2).astype("int32")
    if len(points) < 4:
        return
    cv2.polylines(frame_bgr, [points], True, color, 5, cv2.LINE_AA)
    center_x = int(round(float(points[:, 0].mean())))
    center_y = int(round(float(points[:, 1].mean())))
    cv2.circle(frame_bgr, (center_x, center_y), 8, color, -1, cv2.LINE_AA)
    text = f"{marker_id} {label} {perimeter_rate:.4f}"
    text_origin = (int(points[:, 0].min()), max(18, int(points[:, 1].min()) - 8))
    cv2.putText(
        frame_bgr,
        text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_large_marker_ids(frame_bgr: Any, corners: Any, ids: Any) -> None:
    if ids is None or len(ids) <= 0:
        return
    frame_height, frame_width = frame_bgr.shape[:2]
    base = max(1.0, min(frame_width, frame_height) / 1800.0)
    font_scale = max(1.25, min(3.2, base * 1.7))
    thickness = max(3, int(round(base * 2.8)))
    id_values = ids.flatten().tolist()
    for corner, marker_id_raw in zip(corners, id_values):
        points = corner.reshape(-1, 2)
        if len(points) <= 0:
            continue
        marker_id = int(marker_id_raw)
        label = str(marker_id)
        center_x = int(round(float(points[:, 0].mean())))
        top_y = int(round(float(points[:, 1].min())))
        (text_w, text_h), _baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_x = max(0, min(frame_width - text_w, center_x - (text_w // 2)))
        text_y = max(text_h + 6, top_y - 10)
        cv2.putText(
            frame_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def write_preview_video(
    input_path: str | Path,
    dictionary_name: str,
    best_params: dict[str, float | int],
    output_path: str | Path,
    max_frames: int = 240,
) -> Optional[Path]:
    path = _resolve_user_path(input_path)
    output = _resolve_user_path(output_path)

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
                _draw_large_marker_ids(frame, corners, ids)
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


def _annotate_review_frame(
    frame_gray: Any,
    detector: Any,
    candidate: OptimizationCandidate,
    *,
    sample_position: int,
    source_index: int,
    perimeter_flag_bounds: Optional[tuple[float, float]] = None,
    valid_tag_ids: Optional[set[int]] = None,
) -> tuple[Any, int, int, int, int, int, int]:
    corners, ids, rejected = detector.detectMarkers(frame_gray)
    frame_height, frame_width = frame_gray.shape[:2]
    if perimeter_flag_bounds is not None:
        min_rate, max_rate = perimeter_flag_bounds
    else:
        min_rate = float(candidate.params.get("minMarkerPerimeterRate", 0.0) or 0.0)
        max_rate = float(
            candidate.params.get("maxMarkerPerimeterRate", DEFAULT_MAX_MARKER_PERIMETER_RATE)
            or DEFAULT_MAX_MARKER_PERIMETER_RATE
        )
    decoded_ids, valid_ids, filtered_count = _valid_decoded_marker_ids(
        corners=corners,
        ids=ids,
        frame_width=frame_width,
        frame_height=frame_height,
        valid_tag_ids=valid_tag_ids,
        perimeter_bounds=(min_rate, max_rate),
    )
    detected_count = len(valid_ids)
    decoded_count = len(decoded_ids)
    rejected_count = 0 if rejected is None else int(len(rejected))
    below_min_count = 0
    above_max_count = 0

    annotated = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        _draw_large_marker_ids(annotated, corners, ids)
        for corner, marker_id_raw in zip(corners, ids.flatten().tolist()):
            perimeter_rate = _corner_perimeter_rate(
                corner,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            marker_id = int(marker_id_raw)
            if min_rate > 0 and perimeter_rate < min_rate:
                below_min_count += 1
                _draw_perimeter_flag(
                    annotated,
                    corner,
                    marker_id=marker_id,
                    perimeter_rate=perimeter_rate,
                    label="small",
                    color=(255, 0, 255),
                )
            elif max_rate > 0 and perimeter_rate > max_rate:
                above_max_count += 1
                _draw_perimeter_flag(
                    annotated,
                    corner,
                    marker_id=marker_id,
                    perimeter_rate=perimeter_rate,
                    label="large",
                    color=(255, 255, 0),
                )

    overlay_lines = [
        f"Candidate #{candidate.rank}  score={candidate.score:.3f}",
        (
            f"sample={sample_position} source={source_index} detected={detected_count} "
            f"decoded={decoded_count} filtered={filtered_count} rejected={rejected_count}"
        ),
        f"perimeter flags: magenta small={below_min_count}  cyan large={above_max_count}",
    ]
    for idx, line in enumerate(overlay_lines):
        y = 34 + (idx * 34)
        cv2.putText(
            annotated,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated, detected_count, decoded_count, filtered_count, rejected_count, below_min_count, above_max_count


def _resize_review_image(frame_bgr: Any, *, max_width: int = 1280, max_height: int = 900) -> Any:
    height, width = frame_bgr.shape[:2]
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    if scale >= 1.0:
        return frame_bgr
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def write_top_candidate_review_artifacts(
    *,
    sampled_frames: Sequence,
    sample_indices: Sequence[int],
    dictionary_name: str,
    candidates: Sequence[OptimizationCandidate],
    output_dir: str | Path,
    max_candidates: int = 5,
    max_frames: int = 12,
    extra_candidates: Optional[Sequence[tuple[str, OptimizationCandidate]]] = None,
    perimeter_flag_bounds: Optional[tuple[float, float]] = None,
    valid_tag_ids: Optional[set[int]] = None,
) -> Optional[Path]:
    if not sampled_frames or not candidates:
        return None

    selected_candidate_count = max(1, min(int(max_candidates), len(candidates)))
    selected_candidates: list[tuple[str, OptimizationCandidate]] = [
        ("Top score candidate", candidate)
        for candidate in candidates[:selected_candidate_count]
    ]
    seen_ranks = {candidate.rank for _label, candidate in selected_candidates}
    for label, candidate in extra_candidates or []:
        if candidate.rank in seen_ranks:
            selected_candidates = [
                (
                    f"{existing_label} + {label}"
                    if existing_candidate.rank == candidate.rank and label not in existing_label
                    else existing_label,
                    existing_candidate,
                )
                for existing_label, existing_candidate in selected_candidates
            ]
            continue
        selected_candidates.append((label, candidate))
        seen_ranks.add(candidate.rank)
    selected_frame_positions = _sample_indices(len(sampled_frames), min(max_frames, len(sampled_frames)))

    review_dir = _resolve_user_path(output_dir) / "top_candidate_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    manifest: dict[str, object] = {
        "candidate_count": len(selected_candidates),
        "frame_count": len(selected_frame_positions),
        "perimeter_flag_bounds": (
            {
                "min_perimeter_rate": perimeter_flag_bounds[0],
                "max_perimeter_rate": perimeter_flag_bounds[1],
                "source": "measured_tag_bounds",
            }
            if perimeter_flag_bounds is not None
            else {
                "source": "candidate_min_max_marker_perimeter_rate",
            }
        ),
        "candidates": [],
    }

    for review_label, candidate in selected_candidates:
        detector_params = cv2.aruco.DetectorParameters()
        for key, value in candidate.params.items():
            if hasattr(detector_params, key):
                setattr(detector_params, key, value)
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        candidate_dir = review_dir / f"candidate_{candidate.rank:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        frame_entries = []
        total_below_min_count = 0
        total_above_max_count = 0
        for review_frame_idx, sample_position in enumerate(selected_frame_positions, start=1):
            source_index = (
                int(sample_indices[sample_position])
                if sample_position < len(sample_indices)
                else int(sample_position)
            )
            (
                annotated,
                detected_count,
                decoded_count,
                filtered_count,
                rejected_count,
                below_min_count,
                above_max_count,
            ) = _annotate_review_frame(
                sampled_frames[sample_position],
                detector,
                candidate,
                sample_position=sample_position,
                source_index=source_index,
                perimeter_flag_bounds=perimeter_flag_bounds,
                valid_tag_ids=valid_tag_ids,
            )
            total_below_min_count += below_min_count
            total_above_max_count += above_max_count
            preview_frame = _resize_review_image(annotated)
            output_path = candidate_dir / f"frame_{review_frame_idx:03d}.png"
            if not cv2.imwrite(str(output_path), preview_frame):
                raise RuntimeError(f"Failed to write review preview image: {output_path}")
            frame_entries.append(
                {
                    "review_frame_index": review_frame_idx - 1,
                    "sample_position": sample_position,
                    "source_index": source_index,
                    "detected_count": detected_count,
                    "decoded_count": decoded_count,
                    "filtered_count": filtered_count,
                    "rejected_count": rejected_count,
                    "below_min_perimeter_count": below_min_count,
                    "above_max_perimeter_count": above_max_count,
                    "path": str(output_path),
                }
            )

        manifest["candidates"].append(
            {
                "rank": candidate.rank,
                "review_label": review_label,
                "score": candidate.score,
                "mean_detected": candidate.mean_detected,
                "std_detected": candidate.std_detected,
                "mean_rejected": candidate.mean_rejected,
                "stability": candidate.stability,
                "unique_ids": candidate.unique_ids,
                "eval_fps": candidate.eval_fps,
                "runtime_seconds": candidate.runtime_seconds,
                "average_frame_ms": ((candidate.runtime_seconds / len(sampled_frames)) * 1000.0),
                "below_min_perimeter_count": total_below_min_count,
                "above_max_perimeter_count": total_above_max_count,
                "params": candidate.params,
                "frames": frame_entries,
            }
        )

    manifest_path = review_dir / "review_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def optimize_tracking(
    input_path: str | Path,
    profile: str = DEFAULT_PROFILE,
    sample_frames: int = 80,
    dictionary_name: str = DEFAULT_DICTIONARY,
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM,
    sweep_overrides: Optional[dict[str, Sequence[float | int]]] = None,
    candidate_param_grid: Optional[Sequence[dict[str, Any]]] = None,
    max_parameter_combinations: Optional[int] = None,
    execution_target: str = DEFAULT_EXECUTION_TARGET,
    workers: Optional[int] = None,
    expected_tags: Optional[float] = None,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    early_stop_min_improvement: float = DEFAULT_EARLY_STOP_MIN_IMPROVEMENT,
    output_dir: Optional[str | Path] = None,
    write_preview: bool = False,
    preview_frames: int = 240,
    top_k: int = 10,
    review_perimeter_bounds: Optional[tuple[float, float]] = None,
    valid_tag_ids: Optional[Iterable[int]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stop_requested: Optional[StopRequestedCallback] = None,
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
    if max_parameter_combinations is not None and int(max_parameter_combinations) <= 0:
        raise ValueError("max_parameter_combinations must be >= 1 when provided")
    if expected_tags is not None and expected_tags <= 0:
        raise ValueError("expected_tags must be > 0 when provided")
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be >= 0")
    if early_stop_min_improvement < 0:
        raise ValueError("early_stop_min_improvement must be >= 0")
    if review_perimeter_bounds is not None:
        min_review_rate, max_review_rate = review_perimeter_bounds
        if min_review_rate <= 0 or max_review_rate <= 0:
            raise ValueError("review_perimeter_bounds values must be > 0")
        if max_review_rate <= min_review_rate:
            raise ValueError("review_perimeter_bounds max must be greater than min")

    normalized_dictionary = normalize_dictionary_name(dictionary_name)
    normalized_valid_tag_ids = _normalized_valid_tag_ids(valid_tag_ids)
    resolved_input = _resolve_user_path(input_path)
    sampled_frames, input_type, total_input_frames, sample_indices = load_sample_frames_with_indices(
        resolved_input,
        sample_frames,
    )
    frame_height, frame_width = sampled_frames[0].shape[:2]
    if candidate_param_grid is not None:
        param_grid = normalize_candidate_param_grid(candidate_param_grid)
        parameter_source = "explicit candidate grid"
    else:
        param_grid = build_parameter_grid(
            profile_key,
            tag_size_mm=tag_size_mm,
            frame_width=frame_width,
            frame_height=frame_height,
            sweep_overrides=sweep_overrides,
        )
        parameter_source = "profile sweep grid"
    parameter_combinations_uncapped = len(param_grid)
    if max_parameter_combinations is not None:
        param_grid = limit_parameter_grid(param_grid, int(max_parameter_combinations))
        if len(param_grid) < parameter_combinations_uncapped:
            parameter_source += f" (capped from {parameter_combinations_uncapped})"
    resolved_workers = recommended_worker_count(target_key, workers)

    evaluated: list[OptimizationCandidate] = []
    total = len(param_grid)
    done = 0
    early_stop_enabled = early_stop_patience > 0
    best_seen_score = float("-inf")
    since_improvement = 0
    early_stopped = False
    stopped_by_user = False

    def should_stop() -> bool:
        try:
            return bool(stop_requested and stop_requested())
        except Exception:
            return False

    def register_candidate(candidate: OptimizationCandidate) -> None:
        nonlocal done, best_seen_score, since_improvement, early_stopped, stopped_by_user
        evaluated.append(candidate)
        done += 1
        if progress_callback:
            progress_callback(
                done,
                total,
                _top_candidate_snapshots(evaluated, limit=5),
                {
                    **_candidate_progress_snapshot(candidate, rank=done),
                    "highest_detection_candidate": _highest_detection_candidate_snapshot(evaluated),
                    "top_detection_candidates": _top_detection_candidate_snapshots(evaluated),
                },
            )
        if candidate.score > (best_seen_score + early_stop_min_improvement):
            best_seen_score = candidate.score
            since_improvement = 0
        else:
            since_improvement += 1
        if early_stop_enabled and since_improvement >= early_stop_patience:
            early_stopped = True
        if should_stop():
            stopped_by_user = True

    if resolved_workers == 1:
        for params in param_grid:
            candidate = _evaluate_candidate(
                params,
                sampled_frames,
                normalized_dictionary,
                expected_tags,
                valid_tag_ids=normalized_valid_tag_ids,
                perimeter_filter_bounds=review_perimeter_bounds,
            )
            register_candidate(candidate)
            if early_stopped or stopped_by_user:
                break
    else:
        cursor = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            while cursor < total and not early_stopped and not stopped_by_user:
                batch = param_grid[cursor : cursor + resolved_workers]
                futures = [
                    executor.submit(
                        _evaluate_candidate,
                        params,
                        sampled_frames,
                        normalized_dictionary,
                        expected_tags,
                        normalized_valid_tag_ids,
                        review_perimeter_bounds,
                    )
                    for params in batch
                ]
                for future in concurrent.futures.as_completed(futures):
                    register_candidate(future.result())
                    if stopped_by_user:
                        for pending in futures:
                            pending.cancel()
                        break
                cursor += len(batch)

    if not evaluated:
        raise RuntimeError("Optimization stopped before any parameter combinations completed.")

    evaluated.sort(
        key=lambda item: (item.score, item.mean_detected, -item.mean_rejected, item.eval_fps),
        reverse=True,
    )
    for rank, item in enumerate(evaluated, start=1):
        item.rank = rank

    best = evaluated[0]
    top_detection_candidates = _top_detection_candidates(evaluated, limit=5)
    highest_detection_candidate = top_detection_candidates[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        _resolve_user_path(output_dir)
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
    review_manifest_json_path = write_top_candidate_review_artifacts(
        sampled_frames=sampled_frames,
        sample_indices=sample_indices,
        dictionary_name=normalized_dictionary,
        candidates=evaluated,
        output_dir=run_dir,
        max_candidates=5,
        max_frames=24,
        extra_candidates=[
            (f"Top mean detections #{idx}", candidate)
            for idx, candidate in enumerate(top_detection_candidates, start=1)
        ],
        perimeter_flag_bounds=review_perimeter_bounds,
        valid_tag_ids=normalized_valid_tag_ids,
    )

    top_candidates = evaluated[: max(top_k, 1)]
    result = TrackingOptimizationResult(
        created_at=datetime.now().isoformat(timespec="seconds"),
        input_path=str(resolved_input),
        input_type=input_type,
        dictionary=normalized_dictionary,
        tag_size_mm=tag_size_mm,
        profile=profile_key,
        parameter_source=parameter_source,
        execution_target=target_key,
        workers=resolved_workers,
        sample_frames_requested=sample_frames,
        sample_frames_used=len(sampled_frames),
        parameter_combinations_total=total,
        combinations_evaluated=len(evaluated),
        early_stop_patience=early_stop_patience,
        early_stop_min_improvement=early_stop_min_improvement,
        early_stopped=early_stopped,
        stopped_by_user=stopped_by_user,
        sweep_overrides={
            key: list(values)
            for key, values in (sweep_overrides or {}).items()
        },
        output_dir=str(run_dir),
        summary_json_path=str(run_dir / "optimization_summary.json"),
        candidates_csv_path=str(csv_path),
        preview_video_path=str(preview_video_path) if preview_video_path else None,
        review_manifest_json_path=(
            str(review_manifest_json_path) if review_manifest_json_path else None
        ),
        best_params=best.params,
        best_score=best.score,
        best_mean_detected=best.mean_detected,
        highest_detection_candidate=highest_detection_candidate,
        top_detection_candidates=top_detection_candidates,
        top_candidates=top_candidates,
    )

    summary = result.to_dict()
    summary["total_input_frames"] = total_input_frames
    summary["all_candidates_count"] = len(evaluated)
    summary["parameter_combinations_total"] = total
    summary["parameter_combinations_uncapped"] = parameter_combinations_uncapped
    summary["max_parameter_combinations"] = max_parameter_combinations
    summary["all_candidates_csv"] = str(csv_path)
    summary["review_perimeter_bounds"] = (
        {
            "min_perimeter_rate": review_perimeter_bounds[0],
            "max_perimeter_rate": review_perimeter_bounds[1],
            "source": "measured_tag_bounds",
        }
        if review_perimeter_bounds is not None
        else None
    )
    summary["valid_tag_ids_filter"] = sorted(normalized_valid_tag_ids) if normalized_valid_tag_ids else None
    summary["mean_detected_definition"] = (
        "valid decoded tags after perimeter and allowed-ID filtering; "
        "see mean_decoded and mean_filtered for raw decoded audit counts"
    )

    with Path(result.summary_json_path).open("w") as f:
        json.dump(summary, f, indent=2)

    return result


def optimize_tracking_iterative_refinement(
    *,
    input_path: str | Path,
    seed_params: Sequence[dict[str, Any]],
    rounds: int = 2,
    seed_candidate_count: int = 5,
    sample_frames: int = 80,
    validation_sample_frames: Optional[int] = None,
    profile: str = DEFAULT_PROFILE,
    dictionary_name: str = DEFAULT_DICTIONARY,
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM,
    execution_target: str = DEFAULT_EXECUTION_TARGET,
    workers: Optional[int] = None,
    expected_tags: Optional[float] = None,
    output_dir: Optional[str | Path] = None,
    write_preview: bool = False,
    preview_frames: int = 240,
    top_k: int = 10,
    review_perimeter_bounds: Optional[tuple[float, float]] = None,
    valid_tag_ids: Optional[Iterable[int]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stop_requested: Optional[StopRequestedCallback] = None,
) -> IterativeTrackingRefinementResult:
    if rounds <= 0:
        raise ValueError("rounds must be >= 1")
    if seed_candidate_count <= 0:
        raise ValueError("seed_candidate_count must be >= 1")
    if sample_frames <= 0:
        raise ValueError("sample_frames must be >= 1")

    normalized_seed_params = normalize_candidate_param_grid(seed_params)[:seed_candidate_count]
    if not normalized_seed_params:
        raise ValueError("At least one valid seed candidate is required.")

    validation_frames = (
        int(validation_sample_frames)
        if validation_sample_frames is not None
        else max(sample_frames, sample_frames * 2)
    )
    if validation_frames <= 0:
        raise ValueError("validation_sample_frames must be >= 1")

    resolved_input = _resolve_user_path(input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        _resolve_user_path(output_dir)
        if output_dir
        else _default_output_root(resolved_input).resolve()
    )
    run_root = output_root / f"iterative_tracking_refinement_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    round_results: list[TrackingOptimizationResult] = []
    current_seeds = normalized_seed_params

    def stage_callback(stage: str) -> Optional[ProgressCallback]:
        if progress_callback is None:
            return None

        def callback(
            done: int,
            total: int,
            top_candidates: list[ProgressCandidateSnapshot],
            latest_candidate: ProgressCandidateSnapshot,
        ) -> None:
            latest = dict(latest_candidate)
            latest["stage"] = stage
            progress_callback(done, total, top_candidates, latest)

        return callback

    for round_index in range(1, rounds + 1):
        grid = build_refinement_parameter_grid(current_seeds, round_index=round_index)
        result = optimize_tracking(
            input_path=resolved_input,
            profile=profile,
            sample_frames=sample_frames,
            dictionary_name=dictionary_name,
            tag_size_mm=tag_size_mm,
            candidate_param_grid=grid,
            execution_target=execution_target,
            workers=workers,
            expected_tags=expected_tags,
            early_stop_patience=0,
            early_stop_min_improvement=0.0,
            output_dir=run_root / f"round_{round_index:02d}",
            write_preview=False,
            preview_frames=preview_frames,
            top_k=max(top_k, seed_candidate_count),
            review_perimeter_bounds=review_perimeter_bounds,
            valid_tag_ids=valid_tag_ids,
            progress_callback=stage_callback(f"Refinement round {round_index}/{rounds}"),
            stop_requested=stop_requested,
        )
        round_results.append(result)
        current_seeds = [
            dict(candidate.params)
            for candidate in result.top_candidates[:seed_candidate_count]
        ]
        if result.stopped_by_user:
            break

    if not round_results:
        raise RuntimeError("Iterative refinement ended before any round completed.")

    validation_result: Optional[TrackingOptimizationResult] = None
    final_result = round_results[-1]
    stopped_by_user = any(result.stopped_by_user for result in round_results)

    if not stopped_by_user:
        validation_grid = normalize_candidate_param_grid(current_seeds)
        validation_result = optimize_tracking(
            input_path=resolved_input,
            profile=profile,
            sample_frames=validation_frames,
            dictionary_name=dictionary_name,
            tag_size_mm=tag_size_mm,
            candidate_param_grid=validation_grid,
            execution_target=execution_target,
            workers=workers,
            expected_tags=expected_tags,
            early_stop_patience=0,
            early_stop_min_improvement=0.0,
            output_dir=run_root / "validation",
            write_preview=write_preview,
            preview_frames=preview_frames,
            top_k=max(top_k, seed_candidate_count),
            review_perimeter_bounds=review_perimeter_bounds,
            valid_tag_ids=valid_tag_ids,
            progress_callback=stage_callback("Validation pass"),
            stop_requested=stop_requested,
        )
        final_result = validation_result
        stopped_by_user = validation_result.stopped_by_user

    summary_json_path = run_root / "iterative_refinement_summary.json"
    result = IterativeTrackingRefinementResult(
        created_at=datetime.now().isoformat(timespec="seconds"),
        input_path=str(resolved_input),
        output_dir=str(run_root),
        rounds_requested=rounds,
        rounds_completed=len(round_results),
        seed_candidate_count=len(normalized_seed_params),
        sample_frames=sample_frames,
        validation_sample_frames=validation_frames,
        summary_json_path=str(summary_json_path),
        round_results=round_results,
        validation_result=validation_result,
        final_result=final_result,
        stopped_by_user=stopped_by_user,
    )
    with summary_json_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)
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
        f"Parameter source: {result.parameter_source}",
        f"Execution target: {result.execution_target}",
        f"Workers used: {result.workers}",
        f"Sample frames: {result.sample_frames_used}/{result.sample_frames_requested}",
        f"Parameter sets evaluated: {result.combinations_evaluated}/{result.parameter_combinations_total}",
        (
            "Score weights: "
            f"detect=+{SCORE_DETECTION_WEIGHT:g}, "
            f"stability=+{SCORE_STABILITY_WEIGHT:g}, "
            f"fps=+{SCORE_FPS_WEIGHT:g}, "
            f"rejected=-{SCORE_REJECTED_WEIGHT:g}, "
            f"std_detected=-{SCORE_STD_DETECTED_WEIGHT:g}, "
            f"expected_error=-{SCORE_EXPECTED_ERROR_WEIGHT:g}"
        ),
        (
            "Early stop: disabled"
            if result.early_stop_patience <= 0
            else (
                f"Early stop: {'triggered' if result.early_stopped else 'not triggered'} "
                f"(patience={result.early_stop_patience}, min_improvement={result.early_stop_min_improvement:.6f})"
            )
        ),
        (
            "Completion reason: early stop ended the sweep before testing every combination"
            if result.early_stopped
            else (
                "Completion reason: user ended optimization early"
                if result.stopped_by_user
                else "Completion reason: all parameter combinations were evaluated"
            )
        ),
        f"Stopped by user: {'yes' if result.stopped_by_user else 'no'}",
        (
            "Sweep overrides: not used for explicit candidate grid"
            if result.parameter_source == "explicit candidate grid"
            else "Sweep overrides: none"
            if not result.sweep_overrides
            else f"Sweep overrides: {json.dumps(result.sweep_overrides, sort_keys=True)}"
        ),
        f"Best score: {result.best_score:.4f}",
        f"Best-score candidate mean detections/frame: {result.best_mean_detected:.3f}",
        f"Best params: {json.dumps(result.best_params, sort_keys=True)}",
        f"Output dir: {result.output_dir}",
        f"Summary JSON: {result.summary_json_path}",
        f"Candidates CSV: {result.candidates_csv_path}",
    ]
    summary_path = Path(result.summary_json_path)
    valid_filter_note = ""
    try:
        summary_payload = json.loads(summary_path.read_text())
        if summary_payload.get("valid_tag_ids_filter"):
            valid_filter_note = f" Allowed-ID filter: {len(summary_payload['valid_tag_ids_filter'])} IDs."
    except Exception:
        valid_filter_note = ""
    lines.append(
        "Mean detections are valid decoded tags after perimeter and allowed-ID filters."
        + valid_filter_note
    )
    if result.preview_video_path:
        lines.append(f"Preview video: {result.preview_video_path}")
    if result.review_manifest_json_path:
        lines.append(f"Review manifest: {result.review_manifest_json_path}")

    lines.append("")
    lines.append("Top score candidates:")
    lines.append(format_candidate_results_table(result.top_candidates, ranking="score", max_rows=top_k))
    lines.append("")
    lines.append("Top mean-detection candidates:")
    detection_rows = []
    for detection_rank, candidate in enumerate(result.top_detection_candidates[: max(1, top_k)], start=1):
        row = asdict(candidate)
        row["detection_rank"] = detection_rank
        detection_rows.append(row)
    lines.append(format_candidate_results_table(detection_rows, ranking="detection", max_rows=top_k))
    lines.extend(
        [
            "",
            "Table notes: detect is valid decoded tags/frame; decoded is raw ArUco decoded tags/frame; "
            "filt is decoded tags/frame removed by perimeter or allowed-ID filters; reject/std are average counts per sampled frame; "
            "test_s is wall time for that parameter set. minPerim/maxPerim are marker perimeter-rate bounds; "
            "poly=polygonalApproxAccuracyRate; const=adaptiveThreshConstant.",
        ]
    )
    return "\n".join(lines)


def format_iterative_refinement_report(result: IterativeTrackingRefinementResult) -> str:
    lines = [
        "Iterative Refinement Report",
        "---------------------------",
        f"Created: {result.created_at}",
        f"Input: {result.input_path}",
        f"Output dir: {result.output_dir}",
        f"Seed candidates: {result.seed_candidate_count}",
        f"Refinement rounds completed: {result.rounds_completed}/{result.rounds_requested}",
        f"Sample frames per refinement round: {result.sample_frames}",
        f"Validation sample frames: {result.validation_sample_frames}",
        f"Stopped by user: {'yes' if result.stopped_by_user else 'no'}",
        f"Summary JSON: {result.summary_json_path}",
        "",
        "Round outputs:",
    ]
    for index, round_result in enumerate(result.round_results, start=1):
        lines.append(
            (
                f"- Round {index}: evaluated "
                f"{round_result.combinations_evaluated}/{round_result.parameter_combinations_total}, "
                f"best score={round_result.best_score:.4f}, "
                f"dir={round_result.output_dir}"
            )
        )
    if result.validation_result:
        lines.append(
            (
                "- Validation: evaluated "
                f"{result.validation_result.combinations_evaluated}/"
                f"{result.validation_result.parameter_combinations_total}, "
                f"best score={result.validation_result.best_score:.4f}, "
                f"dir={result.validation_result.output_dir}"
            )
        )
    lines.extend(
        [
            "",
            "Final selected candidate source:",
            f"- {result.final_result.output_dir}",
        ]
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
        "--sweep-max-marker-perimeter-rate",
        default="",
        help="Optional comma-separated maxMarkerPerimeterRate override values.",
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
    parser.add_argument("--max-combinations", type=int, help="Optional cap on parameter combinations to evaluate.")
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

        max_perimeter = _parse_csv_values(
            args.sweep_max_marker_perimeter_rate,
            "--sweep-max-marker-perimeter-rate",
            "float",
        )
        if max_perimeter:
            sweep_overrides["maxMarkerPerimeterRate"] = max_perimeter

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
            max_parameter_combinations=args.max_combinations,
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
