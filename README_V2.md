# BumbleBox V2 (New Architecture)

This repository now includes a V2 foundation that keeps BumbleBox flexibility while simplifying setup and operation.

## Entry Points

- CLI: `/Users/aec/Desktop/BumbleBox/bbx.py`
- GUI: `/Users/aec/Desktop/BumbleBox/bbx_gui.py`
- V2 package: `/Users/aec/Desktop/BumbleBox/bumblebox_v2`
- Desktop scaffold CLI: `/Users/aec/Desktop/BumbleBox/bbx_desktop.py`

## Quick Start

```bash
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py init
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py doctor
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py storage status
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py storage setup --apply-config
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py camera-preview --seconds 20
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py camera-test-tracking --seconds 20
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py roadmap
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep --fps-start 2 --fps-stop 20 --fps-step 2 --probe-seconds 20 --assume-ram-gb 2
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py run-once --mock-camera
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py systemd-write
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py systemd-install
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py systemd-status
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py optimize-tracking --input /path/to/video_or_images --execution-target pi_safe --tag-size-mm 2.5 --early-stop-patience 40
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py schedule-check --benchmark-input /path/to/video_or_images --assume-ram-gb 2
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet init-queen --queen-interface-only
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet enroll-worker --host 192.168.1.21 --name worker-1
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet status
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet latest-status
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet discover
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet queen-pull-latest
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py fleet queen-track-latest --cooldown-minutes 60
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py gui-install-shortcut
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py export-bundle --latest
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx_desktop.py validate-bundle --bundle /path/to/bundle_or_zip
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx_desktop.py analyze --bundle /path/to/bundle_or_zip --output-dir /path/to/desktop_output --with-tracked-video
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx_desktop.py visualize --bundle /path/to/bundle_or_zip --output /path/to/tracking_overlay.mp4
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py nest-label check --folder /path/to/composite_images
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx.py nest-label launch --folder /path/to/composite_images
/Users/aec/Desktop/BumbleBox/.venv/bin/python /Users/aec/Desktop/BumbleBox/bbx_gui.py
```

## V2 Dependencies

Preferred one-command setup:

```bash
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh
```

This installs core V2 packages plus `picamera2`, `PyQt5`, and `labelme` into `.venv`.

Skip flags (only if needed for debugging/non-Pi hosts):

```bash
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh --skip-nest-label
bash /Users/aec/Desktop/BumbleBox/scripts/setup_venv.sh --skip-picamera2
```

Minimum Python packages:

```bash
pip3 install pyyaml opencv-contrib-python pandas numpy
```

On Raspberry Pi, install and enable the camera stack (`libcamera` + `picamera2`) using Raspberry Pi OS package sources.

For nest labeling on Debian/Pi, prefer distro packages for Qt compatibility:

```bash
sudo apt update
sudo apt install python3-pyqt5 labelme
```

## Priority Features Implemented

- Pi 4/5 compatibility checks (`doctor`)
- camera stack checks for HQ/Module3 workflows
- camera bring-up tools for focus/framing and live tag-detection validation (`camera-preview`, `camera-test-tracking`)
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
- operator GUI shell

## GUI Tabs

- Global header controls:
  - `View mode`: `Basic` (default) hides rarely used tuning/maintenance controls, `Advanced` shows them.
  - `Setup order`: one-click navigation strip for `Doctor -> Camera Setup -> Calibration -> FPS Report -> Schedule Check -> Run & Schedule`.
- `Doctor`: dependency and hardware checks.
  - includes Storage Setup panel (mount-point edit/save, status refresh, and auto-mount setup button).
- `Camera Setup`: run preview and live tracking test as camera bring-up steps.
- `Roadmap`: next-step checklist based on current config.
- `Config Editor`: edit core settings, validate, and save.
- `FPS Report`: includes single-video MP4 FPS analysis plus FPS sweep capacity testing.
- `Calibration`: compute/update pixels-per-cm.
- `Schedule Check`: estimate if your recording/tracking cadence fits hardware limits.
- `Optimize Tracking`: tune ArUco parameters and optionally apply best values to config.
- `Nest Labeling`: check dependencies and launch the LabelNests GUI.
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
- `/Users/aec/Desktop/BumbleBox/docs/Legacy_Function_Review.md`
