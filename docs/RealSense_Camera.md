# RealSense D405 Integration

## Current Scope

Phase 2 adds synchronized, incremental RealSense recording to the Phase 1 hardware-validation path.

Implemented:

- profile-based enable/disable behavior in the setup wizard
- RealSense device discovery and serial selection
- reporting of advertised depth and color stream profiles
- an exact configured-stream probe
- raw 16-bit depth, NumPy depth, colorized depth, color image, and metadata snapshots
- host-arrival and RealSense device timestamps for later synchronization analysis
- health-check integration
- synchronized `run-once` capture from one shared host start deadline
- incremental raw-depth and color writing that does not retain the full D405 recording in RAM
- frame-number gap reporting and separate host/device timestamp columns

Not yet implemented:

- calibrated temporal offset correction between the three cameras
- calibrated spatial alignment between OwlSight/HQ, thermal, and D405 frames
- combined multimodal inspection video

## Installation

From the repository on the Raspberry Pi:

```bash
cd /home/auggie_ec/Desktop/BumbleBox
./start_bumblebox.sh --install-realsense
```

The setup script installs the primary PyQt GUI and attempts to install `pyrealsense2`. Availability of a compatible prebuilt package varies by Python version and ARM platform. On a 64-bit Raspberry Pi, the installer automatically falls back to a cached Librealsense v2.58.1 source build with Python bindings when no wheel is available. The first source build can take several minutes; later runs reuse the source and compiled objects.

The source fallback disables optional ROS2 bag support because it is not required by BumbleBox and can fail to compile with the GCC version in current Raspberry Pi OS. Override the pinned SDK version only when testing a newer release:

```bash
BUMBLEBOX_REALSENSE_VERSION=2.58.1 ./start_bumblebox.sh --install-realsense
```

## First Hardware Test

Connect the D405 directly to a Raspberry Pi 5 USB 3 port, then run:

```bash
./bbx realsense-check --no-probe
./bbx realsense-check --apply
./bbx realsense-snapshot
```

The first command lists the camera and every advertised depth/color profile without opening streams. The second starts the configured streams and pins the selected serial only if the probe succeeds. The third writes a coherent depth/color snapshot set.

The initial D405 test profile uses matching `848x480@30` depth and color streams. D405 stream types should use the same resolution and frame rate. The configured profile must still be confirmed against the connected camera's report before production use.

If the report says `USB 2.1`, the D405 is running through a USB 2 path or cable. At 848x480 it may then advertise only 5 or 10 FPS, so the reference 30 FPS configuration will not start. Move it to a blue USB 3 port and use a USB 3-capable cable; `realsense-check` reports the available FPS values without changing the saved configuration.

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

## Recording Outputs

When `realsense.enabled: true`, each recording session includes:

- `*_realsense_depth_raw16.npy`: exact raw `uint16` depth stack, written through a disk-backed array
- `*_realsense_depth_preview.avi` and `*_realsense_depth_midframe.png`: colorized inspection views
- `*_realsense_depth_midframe_raw16.png`: raw midpoint depth image
- `*_realsense_color.avi` and `*_realsense_color_midframe.png`: the D405 color stream used to register depth
- `*_realsense_frame_timestamps.csv`: host arrival clocks, device clocks, timestamp domains, and frame numbers
- `*_realsense_metadata.json`: profiles, depth scale, observed FPS, ranges, paths, and frame-number gaps

On one Pi, OwlSight/HQ, thermal, and RealSense workers wait on the same host monotonic start deadline. In optional two-Pi mode they prepare independently and wait for one shared UTC deadline, after a clock-offset preflight. Timestamp rows identify their node and source clock. Neither mode implies simultaneous exposure: each camera and driver still has its own buffering and transport latency. See `docs/Distributed_Capture.md` for the distributed protocol and shared-cue validation workflow.

## Hardware-Free Integration Test

Run the complete recording and output path without opening any cameras:

```bash
./bbx simulate-capture
```

This writes a short RGB, thermal, and RealSense session below the operating system temporary directory and prints the exact session path. Use `--output-root PATH` to retain it elsewhere. The generated streams share one deterministic clock and one moving target: a bright RGB marker, thermal hotspot, and nearer depth patch occupy the same normalized image position. This makes the output useful for checking timestamp handling, frame pairing, output names, and visualization behavior.

The default simulation is intentionally smaller than production capture. Resolution, duration, and frame-rate controls are available through `./bbx simulate-capture --help`. For lower-level tests, `run-once --mock-camera` now simulates every optional sensor enabled in the supplied config, but it retains that config's full production dimensions and duration.

## Remaining Integration Plan

1. Validate the `848x480@30` profile after moving the D405 to USB 3.
2. Measure CPU, RAM, USB bandwidth, and storage throughput with OwlSight, PureThermal, and D405 active.
3. Add an explicit degradation policy: fail the run, continue without depth, or retry when one camera drops.
4. Create a `calibration-project`, register shared-cue captures at multiple nest depths, and implement the solvers described in `docs/Multimodal_Calibration.md`.
5. Add synchronized multimodal inspection outputs and acceptance tests before enabling production automation.
