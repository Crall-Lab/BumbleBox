# RealSense D405 Integration

## Current Scope

Phase 1 establishes a safe hardware-validation path before RealSense is added to continuous BumbleBox recordings.

Implemented:

- profile-based enable/disable behavior in the setup wizard
- RealSense device discovery and serial selection
- reporting of advertised depth and color stream profiles
- an exact configured-stream probe
- raw 16-bit depth, NumPy depth, colorized depth, color image, and metadata snapshots
- host-arrival and RealSense device timestamps for later synchronization analysis
- health-check integration and explicit `run-once` warnings

Not yet implemented:

- continuous RealSense capture in `run-once`
- a shared RGB, thermal, and depth acquisition coordinator
- calibrated temporal or spatial alignment between the three cameras

`run-once` deliberately warns when `realsense.enabled` is true so depth capture is never silently omitted.

## Installation

From the repository on the Raspberry Pi:

```bash
cd /home/auggie_ec/Desktop/BumbleBox
./start_bumblebox.sh --install-realsense
```

The setup script installs the primary PyQt GUI and attempts to install `pyrealsense2`. Availability of a compatible prebuilt `pyrealsense2` package varies by Python version and ARM platform. If that install fails, use the official [librealsense Raspberry Pi guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md) and [Python wrapper instructions](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md), then rerun the check below.

## First Hardware Test

Connect the D405 directly to a Raspberry Pi 5 USB 3 port, then run:

```bash
./bbx realsense-check --no-probe
./bbx realsense-check --apply
./bbx realsense-snapshot
```

The first command lists the camera and every advertised depth/color profile without opening streams. The second starts the configured streams and pins the selected serial only if the probe succeeds. The third writes a coherent depth/color snapshot set.

The initial D405 test profile uses matching `848x480@30` depth and color streams. D405 stream types should use the same resolution and frame rate. The configured profile must still be confirmed against the connected camera's report before production use.

## Configuration

```yaml
setup:
  completed: false
  hardware_profile: custom

realsense:
  enabled: false
  device_serial: auto
  depth_width: 848
  depth_height: 480
  color_width: 848
  color_height: 480
  fps: 30
  align_to: none
  warmup_frames: 15
  save_depth: true
  save_color: true
```

Use the `RGB + thermal + RealSense` hardware profile in the setup wizard for the full test system. Device-specific pages and buttons remain hidden for profiles that do not use those devices.

## Snapshot Outputs

By default, snapshots are written below:

```text
<system.data_root>/<date>/realsense/
```

Each snapshot includes:

- `*_realsense_depth_raw.npy`
- `*_realsense_depth_raw16.png`
- `*_realsense_depth_colorized.png`
- `*_realsense_color.png`
- `*_realsense_snapshot.json`

The metadata records both depth and color frame numbers, both device timestamps and timestamp domains, a host monotonic receipt time, a Unix receipt time, and the depth scale in meters per raw unit. These clocks are recorded separately because camera timestamps and host-arrival timestamps are not interchangeable.

## Phase 2 Plan

1. Validate D405 profiles, firmware, USB mode, depth scale, and snapshot quality on the actual Pi.
2. Measure CPU, RAM, USB bandwidth, and storage throughput with OwlSight, PureThermal, and D405 active.
3. Introduce one acquisition coordinator that timestamps every received frame against the same host monotonic clock.
4. Preserve each camera's native device timestamp and frame number where available.
5. Write depth data incrementally instead of retaining full recordings in RAM.
6. Add explicit degradation policy: fail the run, continue without depth, or retry when one camera drops.
7. Calibrate temporal offset using a shared physical cue, then estimate spatial transforms at multiple nest depths.
8. Add synchronized inspection outputs and acceptance tests before enabling production automation.

