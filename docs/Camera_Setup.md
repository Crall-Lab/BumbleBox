# Camera Setup (Preview + Live Tracking Test)

Use these steps right after `bbx doctor` to confirm camera behavior before long recordings.

## Step A: Preview

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py camera-preview --seconds 20
```

What this checks:

- camera opens successfully
- focus and framing are acceptable
- exposure/shutter behavior looks correct

Optional overrides:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py camera-preview \
  --seconds 30 \
  --window QTGL \
  --width 1920 \
  --height 1080
```

## Step B: Live Tag Tracking Test

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py camera-test-tracking --seconds 20
```

What this checks:

- live ArUco detections from camera feed
- detection rate across frames
- average and max tags per frame
- unique IDs seen during the test

Useful options:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py camera-test-tracking \
  --seconds 30 \
  --display-width 1280 \
  --dictionary 4X4_50 \
  --box-preset custom \
  --json-out /path/to/camera_tracking_test.json
```

Press `ESC` in the OpenCV window to stop early.

## GUI workflow

Open `/Users/aec/Desktop/BumbleBox/bbx_gui.py`, then use the `Camera Setup` tab:

1. `Run Camera Preview`
2. `Run Live Tracking Test`

Or use `Run Full Setup Check (A then B)` for a single guided flow.

## Camera Connection Diagnostics

`./bbx camera-check` now compares the configured camera model with the sensor reported by Picamera2/libcamera. For an OwlSight profile, an OV64A40 chip-ID failure with Linux error `-121` is reported as a CSI/I2C connection or sensor-power problem, with shutdown and cable-reseat steps. It is not reported as an autofocus, resolution, or tuning-file problem. A different detected sensor is reported as a camera-profile mismatch.
