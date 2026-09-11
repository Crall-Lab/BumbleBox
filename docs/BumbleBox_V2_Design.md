# BumbleBox V2 Design

## Goals

BumbleBox V2 is designed to preserve the original BumbleBox capabilities while making setup and operation substantially easier for new users.

Priority goals:

1. Run on Raspberry Pi 4 and Raspberry Pi 5.
2. Support Raspberry Pi HQ camera and Raspberry Pi Camera Module 3 variants.
3. Keep operation simple for non-programmers.
4. Keep flexible data modes (`record_only`, `track_only`, `record_and_track`, mixed scheduling).
5. Maximize recording uptime while still enabling tag tracking.
6. Provide transparent MP4 framerate quality reporting.
7. Provide a clear, repeatable pixel-to-distance calibration workflow.
8. Provide a clear operator roadmap.

## High-Level Architecture

BumbleBox V2 is split into a backend core and operator interfaces.

Backend core (`/Users/aec/Desktop/BumbleBox/bumblebox_v2`):

- `config.py`: config schema, defaults, validation, and load/save.
- `doctor.py`: hardware and dependency checks.
- `camera_setup.py`: live camera preview and live ArUco tracking test for setup.
- `fps_report.py`: MP4 framerate quality report generation.
- `fps_sweep.py`: increasing-FPS probe with recording-capacity and tracking-time estimates.
- `calibration.py`: pixel-distance calibration (manual and ArUco workflows).
- `roadmap.py`: generated operator roadmap.
- `run_engine.py`: one-shot execution engine for recording/tracking pipelines.
- `systemd_units.py`: systemd service/timer generation from config.
- `nest_labeling.py`: launch/readiness checks for the PyQt nest-labeling tool.
- `tracking_optimizer.py`: ArUco parameter search and scoring with Pi/Desktop execution modes.
- `schedule_check.py`: schedule feasibility analysis (RAM + timing) with actionable fixes.
- `fleet.py`: optional queen/worker orchestration, SSH worker enrollment, and fleet health checks.
- `run_bundle.py`: portable run-package export for downstream desktop analysis tools.

Operator interfaces:

- CLI: `/Users/aec/Desktop/BumbleBox/bbx.py`
- GUI app: `/Users/aec/Desktop/BumbleBox/bbx_gui.py`
- Downstream desktop scaffold CLI: `/Users/aec/Desktop/BumbleBox/bbx_desktop.py`

## Why CLI + GUI

The CLI is scriptable and robust for automation.  
The GUI is for setup, validation, and day-to-day operation without terminal commands.

Both are backed by the same core logic, so behavior remains consistent.

## Camera Compatibility Strategy

V2 config explicitly separates camera model from capture policy.

- `camera.model` can be `auto`, `hq`, `hq_noir`, `module3`, `module3_wide`, `module3_standard`, `module3_noir`.
- Compatibility validation is done by `bbx doctor` with `libcamera`.
- Capture code should map camera model to tuned defaults (resolution, FPS target, exposure/noise settings) while allowing overrides.

## Flexible Pipeline Modes

Pipeline mode is declared in config:

- `record_only`: prioritize recording throughput.
- `track_only`: fast periodic tag snapshots with no video output.
- `record_and_track`: each recording can be tracked.
- `mixed_schedule`: separate record and track jobs with independent intervals.

Tracking source:

- `ram`: track from in-memory frame arrays (faster post-recording analysis).
- `video`: track from saved video files (less RAM pressure).

## Recording-Time Maximization Strategy

To preserve recording availability:

1. Set `defer_tracking_until_after_recording: true`.
2. Keep capture loop lightweight.
3. Run tracking in a separate phase or worker process.
4. In mixed mode, stagger intervals to avoid overlap.

This avoids tracking CPU spikes from reducing effective capture FPS or causing dropped capture windows.

## MP4 Framerate Verification

V2 FPS reporting supports:

1. Metadata checks (`frame_count`, metadata FPS, metadata duration).
2. Timestamp-sidecar checks (actual elapsed capture time and interval jitter).
3. Drift calculations against metadata.
4. JSON report export for archival QA.

Command:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-report --video /path/to/file.mp4 --json-out /path/to/report.json
```

For capacity testing across increasing framerates:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep --fps-start 2 --fps-stop 20 --fps-step 2 --probe-seconds 20
```

This reports target vs measured FPS and estimates max recording duration under multiple RAM-risk budgets.
If recent tracking runs exist, it also estimates tracking time for those durations.

## Pixel-to-Distance Calibration

Two calibration methods are supported:

1. Manual two-point calibration (recommended): pick two known points and provide real distance.
2. ArUco marker calibration (optional advanced): detect marker side in an image and convert marker size to px/cm.

Recommended practice:

- Use two points that are at least 5 cm apart (often 8-15 cm is better) to reduce relative measurement noise.
- Ensure both points are on the same plane where bees are moving.
- Re-run calibration if camera position, focus, zoom, or mounting changes.

Both methods update:

- `calibration.pixels_per_cm`
- `metrics.pixel_contact_distance` (derived from `metrics.contact_distance_cm`)
- `calibration.last_updated`

## Scheduling Recommendation

Default scheduler target in config is `systemd` for reliability on Pi.

Rationale:

- Better startup/restart behavior than manual cron editing.
- Cleaner logs.
- Stronger production behavior for 24/7 acquisition.
- Explicit dependency ordering (for example, wait for mounted data drives).
- Native retry and restart policies for transient failures.
- Better environment consistency than cron (path/user/shell differences are a common failure source).

`cron` remains supported for compatibility.

## GUI Scope in V2

The primary operator interface is now PyQt. A conditional first-run wizard selects a hardware profile and omits irrelevant thermal or RealSense setup pages. Existing Tk tools remain available through `bbx gui --legacy` while specialized pages are migrated without changing the underlying capture and analysis services.

Current GUI covers:

- Doctor checks
- Roadmap view
- Config editing and validation
- FPS reporting
- Scale calibration
- Schedule viability checks
- Tracking optimization (Pi-safe or Desktop)
- Nest-labeling readiness and launch
- One-shot run and scheduling actions
- Run-bundle export from selected recent runs

Next GUI milestones:

1. Mode/schedule editor with validation.
2. One-click scheduler install/remove.
3. Live run status and last-N run summaries.
4. Preview camera tuning panel.

## Next-Phase Runtime (Implemented)

The V2 "next phase" runtime now includes:

1. `bbx run-once`:
   - Executes one job using current config.
   - Supports mode override (`record_only`, `track_only`, `record_and_track`).
   - Supports deferred tracking behavior.
   - Writes per-run summary JSON, per-run config snapshot JSON, and optional FPS report JSON.

2. `bbx systemd-write`:
   - Generates `.service` and `.timer` units from config.
   - Supports mixed schedule by generating separate record/track timers.
   - Outputs install commands for `system` or `user` scope.

3. `bbx systemd-install|enable|disable|status`:
   - Executes lifecycle operations directly from CLI.
   - Handles `system` and `user` scopes.
   - Returns command output for troubleshooting.

4. GUI "Run & Schedule" tab:
   - Run one cycle now.
   - Optionally use mock camera for non-Pi testing.
   - Generate and manage systemd schedules from the current config.
   - Show recent run history from saved run summary JSON files.

5. GUI "Config Editor" tab:
   - Edit common settings directly.
   - Validate configuration before save.
   - Save to the active config file.

6. Nest-labeling integration:
   - `bbx nest-label check|launch` commands.
   - Tkinter `Nest Labeling` tab to run environment checks and launch the PyQt labeling app.
   - Portable `labelmerc` path resolution with fallback to LabelMe defaults.

7. Tracking optimization integration:
   - `bbx optimize-tracking` command.
   - Tkinter `Optimize Tracking` tab.
   - Mutually exclusive execution target modes: `pi_safe` and `desktop`.
   - Default tag-size assumptions tuned for `2.5 mm` tags, with user override.
   - Early-stop support to reduce long runs.
   - Optional direct application of best parameters into `tracking.aruco_params`.

8. Schedule feasibility integration:
   - `bbx schedule-check` command.
   - Tkinter `Schedule Check` tab.
   - Detects RAM/timing risks and suggests specific mitigations (FPS, intervals, resolution, tracking source).

9. Downstream handoff packaging:
   - `bbx export-bundle` command.
   - Builds a portable bundle with manifest + checksums for drag-and-drop desktop analysis.
   - Supports exporting latest run automatically or targeting specific run summaries.

10. Fleet orchestration integration:
   - `bbx fleet init-queen|enroll-worker|status`.
   - Optional interface-only queen mode (`fleet.queen_local_pipeline_enabled=false`).
   - SSH-based worker reachability and health checks (clock offset, disk, memory, load, timers, latest run summary).

11. Desktop GUI launcher integration:
   - `bbx gui-install-shortcut`.
   - Creates Desktop/app-menu `.desktop` launchers, icon, and a launcher script that prefers `.venv` Python.

12. Camera setup integration:
   - `bbx camera-preview` command.
   - `bbx camera-test-tracking` command.
   - Tkinter `Camera Setup` tab with a guided "Preview then Live Tracking Test" workflow.
