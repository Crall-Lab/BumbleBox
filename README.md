# BumbleBox V2 (New Architecture)

This repository now includes a V2 foundation that keeps BumbleBox flexibility while simplifying setup and operation.

## Entry Points

- Start-here setup: `/Users/aec/Desktop/BumbleBox/start_bumblebox.sh`
- CLI launcher (no manual activation): `/Users/aec/Desktop/BumbleBox/bbx`
- GUI launcher (no manual activation): `/Users/aec/Desktop/BumbleBox/bbx-gui`
- Python CLI entry point: `/Users/aec/Desktop/BumbleBox/bbx.py`
- Python GUI entry point: `/Users/aec/Desktop/BumbleBox/bbx_gui.py`
- V2 package: `/Users/aec/Desktop/BumbleBox/bumblebox_v2`
- Desktop scaffold CLI: `/Users/aec/Desktop/BumbleBox/bbx_desktop.py`

## Quick Start

```bash
bash /Users/aec/Desktop/BumbleBox/start_bumblebox.sh
/Users/aec/Desktop/BumbleBox/bbx init
/Users/aec/Desktop/BumbleBox/bbx doctor
/Users/aec/Desktop/BumbleBox/bbx thermal-check
/Users/aec/Desktop/BumbleBox/bbx thermal-check --apply
/Users/aec/Desktop/BumbleBox/bbx thermal-snapshot
/Users/aec/Desktop/BumbleBox/bbx realsense-check --no-probe
/Users/aec/Desktop/BumbleBox/bbx realsense-check --apply
/Users/aec/Desktop/BumbleBox/bbx realsense-snapshot
/Users/aec/Desktop/BumbleBox/bbx simulate-capture
/Users/aec/Desktop/BumbleBox/bbx camera-preview --seconds 20
/Users/aec/Desktop/BumbleBox/bbx camera-test-tracking --seconds 20
/Users/aec/Desktop/BumbleBox/bbx roadmap
/Users/aec/Desktop/BumbleBox/bbx gui
```

## V2 Dependencies

Preferred one-command setup:

```bash
bash /Users/aec/Desktop/BumbleBox/start_bumblebox.sh
```

This installs core V2 packages, including the primary PyQt GUI, into `.venvs/bbx-runtime`.
It also creates a dedicated nest-label environment at `/Users/aec/Desktop/BumbleBox/.venvs/bbx-label` and configures BumbleBox to auto-use it for nest-label check/launch flows.
On non-Pi hosts it also attempts optional `picamera2` and `labelme` installs.
On Raspberry Pi hosts, camera and Qt dependencies use apt packages where possible.
It installs the Desktop GUI icon and prints launch instructions.

Skip flags (only if needed for debugging/non-Pi hosts):

```bash
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh --skip-nest-label
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh --skip-picamera2
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh --skip-label-env
bash /Users/aec/Desktop/BumbleBox/start_bumblebox.sh --install-realsense
```

Recommended Raspberry Pi setup (more reliable than pip for camera/Qt stack):

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 libcamera-apps python3-pyqt5 ffmpeg
python3 -m venv --copies --system-site-packages /Users/aec/Desktop/BumbleBox/.venv
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh --system-site-packages --skip-picamera2 --skip-nest-label
```

Note for Python 3.13 on Pi:
- `pip install pyqt5 labelme` may fail because PyQt wheels/tooling are often unavailable for that combination and pip falls back to source builds (`qmake` errors).
- Prefer apt for Qt-related dependencies on Pi (`python3-pyqt5`), and run nest-labeling on a desktop machine if `labelme` is not available in your apt repositories.

Minimum Python packages:

```bash
pip3 install pyyaml opencv-contrib-python pandas numpy
```

On Raspberry Pi, install and enable the camera stack (`rpicam`/`libcamera` + `picamera2`) using Raspberry Pi OS package sources. MP4 recording also requires `ffmpeg`; thermal camera diagnostics work better with `v4l-utils`. `scripts/setup_venv.sh` now tries to install both automatically on Pi via `apt`, and also enables `loginctl linger` for the BumbleBox user so GUI-started user timers can continue after logout.

For nest labeling on Debian/Pi, prefer distro packages for Qt compatibility:

```bash
sudo apt update
sudo apt install python3-pyqt5 labelme
```

## Priority Features Implemented

- Pi 4/5 compatibility checks (`doctor`)
- camera stack checks for HQ/Module3 workflows
- USB thermal camera discovery/probe/snapshot path for PureThermal/Lepton-style devices (`thermal-check`, `thermal-snapshot`)
- RealSense discovery, stream-profile probe, serial pinning, raw depth/color snapshots, and synchronized incremental recording (`realsense-check`, `realsense-snapshot`, `run-once`)
- hardware-free RGB + thermal + RealSense integration recordings with a shared moving synchronization cue (`simulate-capture`)
- actionable OwlSight/OV64A40 connection diagnostics for chip-ID and CSI/I2C failures (`camera-check`)
- versioned RGB + thermal + RealSense calibration projects and capture readiness checks (`calibration-project`)
- conditional hardware profiles for RGB-only, RGB+thermal, RGB+depth, and full multimodal systems
- First-pass synchronized RGB + thermal recording when `thermal.enabled=true`
- Automatic RGB + thermal side-by-side inspection video for synchronized thermal runs
- camera setup tools for focus/framing and live tag-detection validation (`camera-preview`, `camera-test-tracking`)
- explicit camera tuning selection: auto-resolve by camera model + IR/NoIR flag (Pi4 `vc4` / Pi5 `pisp`) with manual override via `camera.tuning_file`
- flexible mode model (`record_only`, `track_only`, `record_and_track`, `mixed_schedule`)
- MP4 framerate quality reporting (`fps-report`)
- FPS sweep capacity test with target-vs-real FPS and recording/tracking duration estimates (`fps-sweep`)
- explicit pixel-to-distance calibration (`calibrate-scale manual|aruco`)
- generated user roadmap (`roadmap`)
- one-shot run execution with mode override (`run-once`)
- systemd unit/timer generation from config (`systemd-write`)
- systemd lifecycle actions (`systemd-install`, `systemd-enable`, `systemd-disable`, `systemd-status`)
- ArUco parameter optimization with Pi/Desktop execution targets, tag-size defaults, and early stop (`optimize-tracking`)
- schedule viability + RAM/timing risk analysis with fix suggestions (`schedule-check`)
- UUID-based storage status/setup helpers for boot-time auto-mount at `system.data_root` (`storage status|setup|set-mount-point`)
- optional queen/worker fleet orchestration + SSH worker health checks (`fleet`)
- queen-side split latest-media workflow: pull latest worker videos separately from hourly latest tracked videos (`fleet queen-pull-latest`, `fleet queen-track-latest`)
- queen-side latest-state matrix + LAN discovery (`fleet latest-status`, `fleet discover`) for offline worker warnings
- queen media timers generated automatically by `systemd-write` when `fleet.queen_media_schedule.enabled=true`
- one-command Desktop GUI launcher/icon installer (`gui-install-shortcut`)
- portable run-bundle export for downstream desktop pipelines (`export-bundle`)
- downstream desktop ingestion scaffold (`bbx_desktop.py`)
- tracked-video rendering in downstream pipeline (`bbx_desktop.py analyze --with-tracked-video`, `bbx_desktop.py visualize`)
- nest labeling launcher and dependency checks (`nest-label check|launch`)
- primary PyQt operator GUI with a conditional first-run setup wizard and single-instance protection
- legacy Tk advanced-tools bridge during the staged GUI migration

## Primary PyQt GUI

`bbx gui` opens the new PyQt interface. On first launch, a profile-based wizard collects the primary camera, optional thermal/depth hardware, data location, run mode, and recording cadence. Thermal and RealSense setup pages are skipped when those devices are not selected.

The current Qt pages are:

- `Overview`: profile, camera, optional-device, and storage status
- `Run`: one-shot runs and start/stop controls for automated recordings
- `Results`: asynchronous run history with RGB, thermal, and RealSense status plus direct artifact access
- `Hardware`: only the checks relevant to the selected hardware profile
- `Advanced`: multimodal calibration-project controls and access to existing specialized tools while their Qt pages are migrated

Only one primary Qt GUI instance is allowed per user. During migration, run `bbx gui --legacy` to open the previous Tk advanced interface.

## Legacy GUI Tabs

- Global header controls:
  - `View mode`: `Basic` (default) hides rarely used tuning/maintenance controls, `Advanced` shows them.
  - `Setup order`: one-click navigation strip for `Doctor -> Camera Setup -> Calibration -> FPS Report -> Schedule Check -> Run & Schedule`.
- `Doctor`: dependency and hardware checks.
  - includes Storage Setup panel (mount-point edit/save, status refresh, and auto-mount setup button).
- `Camera Setup`: run preview and live tracking test as camera setup steps.
- `Roadmap`: next-step checklist based on current config.
- `Config Editor`: edit core settings, validate, and save.
- `FPS Report`: includes single-video MP4 FPS analysis plus FPS sweep capacity testing.
- `Calibration`: compute/update pixels-per-cm.
- `Schedule Check`: estimate if your recording/tracking cadence fits hardware limits.
- `Optimize Tracking`: tune ArUco parameters and optionally apply best values to config.
- `Nest Labeling`: check dependencies and launch the LabelNests GUI.
  - auto-selects dedicated labeling interpreter (`.venvs/bbx-label`) when available; optional override field remains available.
- `Fleet`: set up queen/worker mode, enroll workers, and run fleet status checks.
  - advanced options include separate `Pull Latest` and `Track Latest`, schedule fields (default hourly tracking), and a side-by-side latest media matrix.
- `Run & Schedule`: run now, generate/install/status timers, inspect recent run summaries, and export run bundles.
  - includes runtime alerts for storage mount health and recording freshness.
  - includes `Install GUI Desktop Icon` for one-click Desktop launcher setup.

## Legacy Scripts

- Standalone legacy scripts are now grouped in `/Users/aec/Desktop/BumbleBox/legacy`.
- Legacy notebook/data artifacts moved:
  - `/Users/aec/Desktop/BumbleBox/legacy/notebooks/create_tracked_videos.0.12.ipynb`
  - `/Users/aec/Desktop/BumbleBox/legacy/samples/cumulative_averages.csv`
- V2 still imports these legacy core modules from repository root:
  - `/Users/aec/Desktop/BumbleBox/tag_tracking_utils.py`
  - `/Users/aec/Desktop/BumbleBox/data_cleaning.py`
  - `/Users/aec/Desktop/BumbleBox/behavioral_metrics.py`

## Design and Workflow Docs

- `/Users/aec/Desktop/BumbleBox/docs/Development_Backlog.md`
- `/Users/aec/Desktop/BumbleBox/docs/BumbleBox_V2_Design.md`
- `/Users/aec/Desktop/BumbleBox/docs/BumbleBox_V2_Roadmap.md`
- `/Users/aec/Desktop/BumbleBox/docs/Camera_Setup.md`
- `/Users/aec/Desktop/BumbleBox/docs/Schedule_Check.md`
- `/Users/aec/Desktop/BumbleBox/docs/FPS_Sweep.md`
- `/Users/aec/Desktop/BumbleBox/docs/Storage_Setup.md`
- `/Users/aec/Desktop/BumbleBox/docs/Tracking_Optimization.md`
- `/Users/aec/Desktop/BumbleBox/docs/Run_Bundle.md`
- `/Users/aec/Desktop/BumbleBox/docs/Desktop_Downstream_Scaffold.md`
- `/Users/aec/Desktop/BumbleBox/docs/Fleet_Management.md`
- `/Users/aec/Desktop/BumbleBox/docs/GUI_Desktop_Launcher.md`
- `/Users/aec/Desktop/BumbleBox/docs/Nest_Labeling_On_Debian_Pi.md`
- `/Users/aec/Desktop/BumbleBox/docs/Thermal_Camera.md`
- `/Users/aec/Desktop/BumbleBox/docs/RealSense_Camera.md`
- `/Users/aec/Desktop/BumbleBox/docs/Multimodal_Calibration.md`
- `/Users/aec/Desktop/BumbleBox/docs/Legacy_Function_Review.md`
