# Thermal Camera Integration

This document tracks the staged PureThermal3 + Lepton 3.5 integration path for BumbleBox V2.

## Current Status

Implemented now:

- config placeholders for a thermal device (`thermal.*`)
- USB/V4L2 thermal discovery and probe command:

```bash
/Users/aec/Desktop/BumbleBox/bbx thermal-check
```

This path is intentionally separate from the Pi HQ / Module 3 camera path. The RGB camera remains on
`picamera2/libcamera`, while the thermal camera is expected to appear as a USB/V4L2 device.

## Immediate Pi Bring-Up

Recommended first checks on the Raspberry Pi:

```bash
lsusb
v4l2-ctl --list-devices
/Users/aec/Desktop/BumbleBox/bbx thermal-check
```

If `v4l2-ctl` is missing:

```bash
sudo apt update
sudo apt install v4l-utils
```

## Thermal Config Fields

The default config now includes:

```yaml
thermal:
  enabled: false
  device_path: "auto"
  width: 160
  height: 120
  fps_target: 8.7
  pixel_format: "auto"
  expected_name: "PureThermal"
```

Notes:

- `device_path` can later be pinned to a stable V4L2 path such as `/dev/video2`
- `width` and `height` match the Lepton 3.5 sensor grid
- `pixel_format` is a future-facing field for Y16 vs converted RGB/gray capture handling

## Planned Phases

### Phase 1: discovery and stability

- detect PureThermal device automatically
- identify which `/dev/video*` node is the correct stream
- report available V4L2 formats
- confirm whether Y16 is exposed

### Phase 2: separate access alongside RGB camera

- keep thermal capture on the USB/V4L2 path
- keep HQ / Module 3 capture on `picamera2`
- make sure the two camera stacks do not interfere with each other

### Phase 3: synchronized recording

- dual-camera session runner
- shared timestamps for RGB and thermal frames
- saved session bundle with both outputs

### Phase 4: pre-colony alignment

- user workflow for mounting/alignment checks
- fixed reference target
- approximate overlay / offset guidance

### Phase 5: GUI integration

- thermal detect/probe tab
- thermal preview/snapshot
- dual-record settings
- alignment helper workflow

### Phase 6: post-hoc geometric calibration

- model thermal-to-RGB mapping after the nest is built
- account for nest depth/occlusion
- support downstream correction/registration workflows

## Design Constraints

- Thermal capture should not depend on `picamera2`
- RGB capture should not depend on V4L2 USB probing
- synchronized recording will likely need its own coordinator layer instead of forcing either
  device model into the other's API

## Command Summary

```bash
/Users/aec/Desktop/BumbleBox/bbx thermal-check
/Users/aec/Desktop/BumbleBox/bbx thermal-check --device /dev/video2
/Users/aec/Desktop/BumbleBox/bbx thermal-check --json-out /tmp/thermal_check.json
```
