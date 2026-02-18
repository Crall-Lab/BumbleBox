# Legacy Function Review

This review focuses on legacy core scripts that were partially migrated and had correctness risks.

## Files Reviewed

- `/Users/aec/Desktop/BumbleBox/legacy/record_video.py`
- `/Users/aec/Desktop/BumbleBox/legacy/track_prerecorded_videos.py`
- `/Users/aec/Desktop/BumbleBox/data_cleaning.py`
- `/Users/aec/Desktop/BumbleBox/behavioral_metrics.py`
- `/Users/aec/Desktop/BumbleBox/tag_tracking_utils.py`

## Fixes Applied

1. `data_cleaning.remove_jumps` now evaluates jumps per-bee correctly and no longer only processes the last bee group.
2. `data_cleaning.remove_jumps` now accepts configurable threshold (`jump_threshold_pixels`) instead of hardcoded `500`.
3. `data_cleaning.return_duplicate_bees` now writes both `duplicate_in_frame` and legacy `duplicate` columns for compatibility with downstream cleaning.
4. `data_cleaning.drop_duplicates_clean` now accepts `duplicate_in_frame` as fallback when `duplicate` is missing.
5. `data_cleaning.interpolate` now safely returns input when no interpolation groups are available.
6. `behavioral_metrics.compute_speed` now uses provided `moving_threshold` instead of hardcoded `3.16`.
7. `behavioral_metrics.compute_activity` threshold logic fixed to `1` when moving and `0` when not moving.
8. `behavioral_metrics.calculate_behavior_metrics` activity branch indentation fixed to only run when requested.
9. `behavioral_metrics.pairwise_distance` now computes avg/min/max metrics across full concatenated video dataframe.
10. `behavioral_metrics` imported `subprocess` (missing import used in helper).
11. `tag_tracking_utils` fixed typo `adaptiveThreshWinSizMax` -> `adaptiveThreshWinSizeMax` in all tracking paths.
12. `tag_tracking_utils.load_actual_fps` now supports video-path input by resolving sidecar `*_actual_fps.txt`.
13. `track_prerecorded_videos.py` syntax and call-signature errors fixed (datetime construction, argument mismatch, invalid f-string).
14. `record_video.py` indentation and key-lookup issues fixed enough to compile and run with safer config fallback behavior.

## Remaining Risks in Legacy Path

1. Legacy modules still mix `setup.py` and `config.yaml` assumptions in multiple places.
2. Several scripts retain broad bare `except:` blocks, which can hide silent data-quality failures.
3. `record_video.py` remains tightly coupled to legacy config structure and ad hoc key fallback logic.
4. `behavioral_metrics.py` still performs repeated CSV writes inside many compute steps, increasing I/O overhead.
5. `track_prerecorded_videos.py` still uses filename parsing assumptions that can break if naming format changes.
6. Legacy duplicate-resolution functions rely on old column names (`video path`, `bee ID`, `frame number`) and can fail on newer schemas without adapters.

## Alternatives Worth Considering

### Alternative A: Full V2 Runtime Migration (Recommended)

- Keep legacy scripts read-only.
- Move all production execution to `/Users/aec/Desktop/BumbleBox/bumblebox_v2/run_engine.py`.
- Build compatibility adapters that map legacy CSV column names to V2 schema when needed.

Why:

- Reduces maintenance burden from parallel code paths.
- Avoids ongoing setup/config drift.
- Enables better testability and GUI-driven operation.

### Alternative B: Detector Backend Interface

- Introduce detector abstraction (`OpenCV ArUco`, optionally `AprilTag`) behind a single API.
- Keep tracking output schema stable regardless of backend.

Why:

- Lets you benchmark robustness/performance per camera/lighting profile.
- Future-proofs against detector-specific regressions.

### Alternative C: Metrics Pipeline Refactor

- Split metrics into pure functions that do not write files.
- Write files only once in an orchestrator layer.
- Add optional `polars` backend for larger datasets.

Why:

- Lower I/O overhead and cleaner test boundaries.
- Easier to add or remove metrics without side effects.

### Alternative D: Queue-Based Capture/Tracking Separation

- Capture process writes video + timestamps.
- Tracking worker runs from queue or post-capture batches.

Why:

- Better supports the priority of maximizing recording availability.
- Prevents tracking spikes from stealing capture time.

## Recommendation

Use legacy scripts only for transitional compatibility.  
For production and future features, continue consolidating into V2 and treat legacy modules as fallback adapters until fully retired.

