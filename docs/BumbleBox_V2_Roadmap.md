# BumbleBox V2 Operator Roadmap

This is the recommended user flow after downloading BumbleBox.

## Step 1: Create V2 Config

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py init
```

Creates:

- `/Users/aec/Desktop/BumbleBox/bumblebox_v2/config.yaml`

## Step 2: Run Health Checks

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py doctor
python3 /Users/aec/Desktop/BumbleBox/bbx.py storage status
```

Confirm:

- Raspberry Pi model recognized (4 or 5)
- camera stack detected
- required Python packages installed
- data storage path writable

If storage is not mounted where BumbleBox expects it, run:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py storage setup --apply-config
```

Then run camera setup checks:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py camera-preview --seconds 20
python3 /Users/aec/Desktop/BumbleBox/bbx.py camera-test-tracking --seconds 20
```

This verifies focus/exposure/framing and confirms that live ArUco detection is working before calibration and long runs.

## Step 3: Set Camera + Pipeline Mode

Edit config:

- `camera.model`
- `pipeline.mode`
- `pipeline.tracking_source`
- `pipeline.defer_tracking_until_after_recording`

Recommended defaults for maximizing recording availability:

- `pipeline.mode: record_and_track`
- `pipeline.tracking_source: ram`
- `pipeline.defer_tracking_until_after_recording: true`

## Step 4: Calibrate Pixel Distance

### Option A: Manual two-point

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py calibrate-scale manual \
  --point-a 100,200 \
  --point-b 980,200 \
  --distance-cm 10.0
```

Use points on the same imaging plane and prefer a larger known distance (at least 5 cm, often 8-15 cm) to reduce noise.

### Option B: ArUco marker image

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py calibrate-scale aruco \
  --image /path/to/calibration_image.png \
  --marker-size-mm 5.0 \
  --dictionary 4X4_50
```

`--marker-size-mm` is required because the tool needs a known real-world marker size to convert pixels to centimeters.

Re-run `bbx roadmap` and ensure calibration is no longer TODO.

## Step 5: Validate MP4 Framerate

After a recording, run:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-report \
  --video /path/to/session.mp4 \
  --json-out /path/to/session_fps_report.json
```

Review:

- metadata FPS
- actual FPS (if timestamp sidecar exists)
- frame interval jitter
- drift percentage

To test increasing framerates and see estimated max recording duration:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep \
  --fps-start 2 \
  --fps-stop 20 \
  --fps-step 2 \
  --probe-seconds 20 \
  --assume-ram-gb 2
```

If recent tracking runs are available, this also estimates how long those recording durations would take to track.

## Step 6: Optimize Tracking (recommended)

Tune ArUco parameters on representative footage before long experiments:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py optimize-tracking \
  --input /path/to/video_or_images \
  --execution-target pi_safe \
  --tag-size-mm 2.5 \
  --early-stop-patience 40 \
  --apply-best
```

For faster tuning on a workstation:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py optimize-tracking \
  --input /path/to/video_or_images \
  --execution-target desktop \
  --profile balanced \
  --tag-size-mm 2.5 \
  --apply-best
```

The best parameters are written to `tracking.aruco_params` when `--apply-best` is set.

## Step 7: Validate Schedule Fit

Before long unattended runs, verify that timing and RAM fit your hardware:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py schedule-check \
  --benchmark-input /path/to/video_or_images \
  --assume-ram-gb 2
```

If this reports failures/warnings, follow the suggested fixes (lower FPS, lower resolution, longer intervals, etc).

## Step 8 (Optional): Configure Queen/Worker Fleet

If you run multiple BumbleBoxes over Ethernet and want one queen interface box:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet init-queen --queen-interface-only
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet enroll-worker --host 192.168.1.21 --name worker-1
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet status
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet latest-status
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet discover
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet queen-pull-latest
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet queen-track-latest --cooldown-minutes 60
```

Notes:

- `--queen-interface-only` sets a controller-only queen (no local recording/tracking).
- If you want queen to also run experiments, use `--queen-bbox-active`.
- Prefer `chrony` for stable time sync; use SSH for fleet orchestration and health checks.
- `fleet queen-pull-latest` refreshes the per-worker `latest_video` pointer.
- `fleet queen-track-latest` updates per-worker `latest_tracked` (hourly recommended), with load/memory guardrails.
- `fleet latest-status` shows side-by-side latest pulled vs latest tracked state and lag per worker.
- `fleet discover` scans LAN neighbors and flags configured workers that look offline.

## Step 9: Label Nests (if needed)

If your experiment requires nest-component labeling on composite images:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py nest-label check --folder /path/to/composite_images
python3 /Users/aec/Desktop/BumbleBox/bbx.py nest-label launch --folder /path/to/composite_images
```

Or from GUI: open `Nest Labeling`, run `Check Environment`, then `Launch Nest Labeling`.

## Step 10: Install Scheduling

When ready for unattended operation:

1. Choose `scheduling.backend` (`systemd` recommended).
2. Enable scheduling in config.
3. Generate units:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py systemd-write
```

If `fleet.role=queen` and `fleet.queen_media_schedule.enabled=true`, this also generates:

- `<unit_prefix>-queen-pull-latest.timer`
- `<unit_prefix>-queen-track-latest.timer`

4. Install/enable from CLI:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py systemd-install
python3 /Users/aec/Desktop/BumbleBox/bbx.py systemd-status
```

## Step 11: GUI Workflow (optional)

Create a Desktop icon (optional but recommended for non-technical users):

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py gui-install-shortcut
```

Run:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_gui.py
```

Use tabs in this order:

1. `Doctor`
  - in `Storage Setup`, confirm mounted status or run `Setup Storage Auto-Mount`.
2. `Camera Setup` (Preview + Live Tracking Test)
3. `Calibration`
4. `FPS Report`
5. `Schedule Check`
6. `Optimize Tracking` (choose `Pi-safe` or `Desktop`)
7. `Nest Labeling` (only if labeling is required)
8. `Roadmap`
9. `Fleet` (optional, if using queen/worker orchestration)
10. `Run & Schedule` (for one-click bundle export from Recent Runs)

`Run & Schedule` includes runtime alerts for:

- storage mount health (including unstable `/dev/sdX` mount-source warning)
- recording freshness (warn if scheduled recording appears stale)

## Step 12: 24-Hour Acceptance Test

Before production experiments:

1. Dry-run one cycle first:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py run-once --mock-camera
```

2. Run 24 hours with target mode.
3. Confirm no missed recording windows.
4. Confirm tag tracking output quality.
5. Confirm FPS drift is acceptable.
6. Save resulting config snapshot as your deployment baseline.

## Step 13: Export Portable Run Bundle

To move data into a downstream desktop tool, export a portable run bundle:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py export-bundle --latest
```

Target a specific run if needed:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py export-bundle \
  --summary /path/to/your_run_summary.json
```

Output includes a bundle manifest with file hashes, plus a `.zip` archive by default for drag-and-drop transfer.

Then on desktop:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py validate-bundle --bundle /path/to/bundle_or_zip
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py analyze --bundle /path/to/bundle_or_zip --output-dir /path/to/desktop_output --with-tracked-video
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py visualize --bundle /path/to/bundle_or_zip --output /path/to/tracking_overlay.mp4
```
