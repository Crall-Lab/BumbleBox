# BumbleBox Development Backlog

This document tracks research and engineering work that is not part of the
operator setup roadmap. Check off subtasks as they are implemented and
validated.

## RealSense and Multimodal Acquisition

**Status:** Single- and optional two-Pi capture implemented; physical USB 3 and sustained-load validation required

**Goal:** Integrate D405 depth with interchangeable HQ/OwlSight RGB capture and PureThermal data without obscuring timestamp provenance or exceeding Pi resources.

- [x] Add conditional hardware profiles and first-run setup wizard pages.
- [x] Add RealSense discovery, advertised-profile reporting, exact stream probe, and serial pinning.
- [x] Save raw depth/color snapshots with device and host timing metadata.
- [x] Add health-check coverage and explicit configured-profile diagnostics.
- [ ] Run `realsense-check --apply` on the Raspberry Pi 5 and retain the full profile report.
- [ ] Validate depth scale, near-field nest coverage, invalid-pixel rate, and thermal/RGB interference.
- [ ] Benchmark OwlSight + PureThermal + D405 USB, CPU, RAM, and storage load together.
- [x] Build a shared acquisition coordinator with host monotonic receipt timestamps for all streams.
- [x] Preserve native RealSense device timestamps/frame numbers alongside host clocks.
- [x] Write raw depth incrementally and report frame-number gaps/device loss as run failures.
- [x] Add multimodal run summaries and timestamp tables.
- [x] Add configurable two-Pi sensor assignment, capture plans, prepare-and-arm coordination, preflight, manifests, local-first recording, and optional resumable collection.
- [x] Integrate two-Pi setup, status, hardware checks, manifests, and timing analysis into the PyQt workflow.
- [ ] Validate distributed capture on two physical Pi 5 units with OwlSight, PureThermal, and D405 hardware.
- [ ] Add a detached worker service if experiments must survive controller loss during an active capture.
- [x] Add a hardware-free three-camera recording simulation with shared timing and visual cues.
- [x] Add a bounded local live monitor for OwlSight/RGB, thermal, RealSense color, and RealSense depth.
- [x] Show only the feeds a setup uses in the live monitor, and let any feed fill the screen.
- [ ] Validate sustained live-monitor load and displayed frame rates on the Raspberry Pi 5.
- [ ] Add frame-by-frame three-camera inspection output.
- [x] Define a versioned multimodal calibration project, capture registry, artifact contract, and readiness checks.
- [ ] Implement the temporal offset/drift solver against registered calibration captures.
- [ ] Implement and validate multi-depth RGB/thermal/RealSense spatial registration.

See `docs/RealSense_Camera.md` for hardware validation and `docs/Multimodal_Calibration.md` for the calibration protocol and file contract.

## PyQt GUI Migration

**Status:** Primary shell and setup wizard implemented

- [x] Make PyQt the default GUI and retain `bbx gui --legacy` as a migration bridge.
- [x] Add single-instance protection to reduce conflicting operator sessions.
- [x] Add profile-aware setup wizard, overview, run controls, and conditional hardware checks.
- [ ] Port storage setup and mount-status interactions from Tk.
- [ ] Port the full config editor with Basic and Advanced sections.
- [ ] Port calibration and thermal registration interfaces.
- [ ] Port tracking optimization and candidate-review interfaces.
- [x] Add asynchronous run history plus RGB, thermal, and RealSense result/artifact controls.
- [ ] Port fleet management, alerts, and bundle export.
- [ ] Remove the Tk bridge only after feature parity and Pi operator testing.
- [x] Show data drive, schedule, and latest-recording status on Overview; explain
      failed commands in plain words; review the schedule before starting it.
      See `docs/Unattended_Safety.md`.
- [ ] Validate the status cards against real BumbleBox timers on a Raspberry Pi,
      including a failed scheduled run whose reason is read from the system log.
- [x] Add a way to cancel a running command from the GUI. A cancelled recording
      writes a summary and keeps any video already on disk.
- [x] Ask for confirmation before Stop Automated Recording.
- [x] Hide thermal, depth, and two-Pi result buttons, table columns, and detail
      lines on boxes without that hardware (older recordings that used it keep them).
- [x] Add a recording settings page: run mode, length, interval, frame rate,
      grayscale, in-memory or encode-while-recording, format, encoder speed, and
      the free-space limit, with an encode benchmark button. See
      `docs/Streaming_Capture.md`.
- [ ] Decide the default frame handling and encoder speed from
      `bbx encode-benchmark` results on a Raspberry Pi 5. The defaults are
      unchanged (`ram`, `auto`).
- [x] State what the settings imply: storage used per day and days until the
      drive is full, on the Overview storage card and the Settings page.
- [ ] Check the storage estimate against a week of real recordings on a Pi.
- [x] Let lengths and intervals be typed in seconds, minutes, hours or days on
      the Settings page and in the setup wizard (the config still stores
      seconds and minutes).
- [x] Add `scheduling.start_time`, the time of day recordings are counted from,
      so a daily recording can start at a chosen time.
- [ ] Make `scheduling.active_hours` do something: record only between two
      times of day. It is in the config but unused.
- [x] Split a long recording into consecutive files without stopping the
      camera, so a two-hour session is a row of short videos with no gaps
      (`capture.chunk_minutes`). See `docs/Long_Recordings.md`.
- [x] Restart a failed or stalled recording for the time left in its session
      (`capture.restart_on_failure`), and record a run that is killed as
      "Did not finish" instead of leaving no trace.
- [x] Stop a hung run before the next slot on long intervals (the automatic
      service time limit was 50 hours for a daily schedule).
- [ ] Run a two-hour session on a Pi, on one camera and on two: frame rate
      held, temperature, file sizes, and a pulled camera cable mid-session.
- [ ] Decide whether a session interrupted by a reboot should resume when the
      Pi comes back inside its window. Off by design at present.
- [ ] Save the timestamps of the file in progress as it is written, so a
      recorder that is killed outright loses nothing but unflushed frames.
- [ ] Make the tracking overlay video cover every file of a split recording.
- [ ] Test Cancel during two-Pi capture; the other Pi may keep recording.

## Two Cameras on One Pi

**Status:** Implemented with one config per camera; not yet run on a Pi with two cameras

**Goal:** Record from two Raspberry Pi cameras on one Pi 5 at the same moments.

- [x] Choose a camera per config, by number or pinned to one physical camera
      (`camera.camera_num`, `camera.camera_id`); refuse to record when a pinned
      camera is missing instead of using the other one.
- [x] One capture lock per camera, tagged session names for the second camera,
      and separate timer names, so both schedules run together.
- [x] `bbx second-camera add|status|pin|remove` and `bbx run-once --all-cameras`.
- [x] GUI: add and remove the second camera, camera switch, both-camera status,
      start/stop/record both, schedule kept in step on request, memory and
      storage judged for both together. See `docs/Two_Cameras.md`.
- [ ] Run on a Pi 5 with two HQ cameras: confirm the libcamera Ids are stable
      across reboots and that pinning picks the right connector.
- [ ] Load-test both cameras at full resolution in memory mode and in
      encode-while-recording mode; record the frame rates each camera reaches.
- [ ] Make `encode-benchmark` able to test two streams at once.
- [ ] Decide whether frames need to be synchronised between cameras; if so,
      record both in one run with a shared clock (the larger redesign).
- [ ] Make `export-bundle` and `fps-sweep` pick the newest run of one camera
      when two share a data folder.
- [ ] Let the live monitor show both cameras side by side.

## View-Only Status Page (Optional Add-On)

**Status:** Not started; deferred on 2026-11-12

**Goal:** Let an operator check a box from a laptop or phone browser, without a
monitor, remote desktop, or the GUI being open.

**Decision:** The PyQt GUI stays the main interface so that a box works with no
network at all. A status page would be an add-on that is off unless enabled,
and it would only show information.

### What It Would Show

- The same checks as the Overview cards in the GUI: data drive, recording
  schedule, and the last scheduled run with its failure reason.
- The most recent recordings and whether each succeeded.
- The midpoint frame of the newest recording, which every run already saves, so
  the operator can see that the nest is in frame and lit.

### What It Would Not Do

It would have no control that changes anything. Starting and stopping the
schedule, setup, and the live camera monitor stay in the PyQt GUI. A page that
can stop a recording needs a login. A view-only page does not, although anyone
on the same network could open it.

### How It Would Work

- [ ] Add a small always-on command (for example `bbx status-serve`) that serves
      one page and reloads it every 30 seconds.
- [ ] Reuse `bumblebox_v2/box_status.py`, which has no GUI code, so the page and
      the Overview cards cannot disagree.
- [ ] Use Python's built-in web server so that no new package is installed on
      the Pi.
- [ ] Run it as its own systemd service, separate from the recording timers, so
      a fault in the page cannot affect recording.
- [ ] Serve the same status as JSON at a second address, so one page can later
      list several boxes.
- [ ] Keep it off unless enabled in the config.

### To Solve First

- [ ] Make the recent-recordings scan cheap. `status_history.list_recent_run_records`
      reads every `*_run_summary.json` under the data folder, which is too slow
      to repeat every 30 seconds on a full drive.
- [ ] Confirm the page stays light while a recording is in progress. It should
      read small files only and never serve video.
- [ ] Decide whether the nest frame may be visible to anyone on the network.

### Network Requirements

The Pi and the viewing device must be on the same network, and that network
must let devices talk to each other. Nothing is installed on the viewing device.

| Network | Expected to work |
|---|---|
| Home router, or a router owned by the lab | Yes |
| University Wi-Fi such as eduroam, or a guest network | Often not; these commonly block device-to-device traffic, and joining a Pi can need IT's help |
| Wired Ethernet to a lab switch | Usually, though some networks require each device to be registered |

Quick test: if SSH or remote desktop from the laptop to the Pi already works,
the page will work on that network.

The page would be opened at `http://<pi-name>.local:<port>`. Raspberry Pi OS
announces that name by default. Macs, iPhones, Windows, and Linux resolve it;
some Android phones do not, and the Pi's IP address is the fallback.

If the network blocks device-to-device traffic:

- Use a small dedicated router for the boxes. Keep an internet uplink, because
  the Pis set their clocks from the network and recording times depend on that.
- Run the Pi as its own Wi-Fi hotspot and join it when standing near the box.
  This reaches one box at a time.
- Use a private network service such as Tailscale. It also works off site, but
  needs an account and an install on every device.

None of this has been built or tested.

## RGB and Thermal Camera Synchronization

**Status:** First-pass capture and shared-cue analysis implemented; physical validation and correction remain

**Goal:** Measure and correct the effective sensor-to-sensor time offset,
clock drift, and spatial mapping between the RGB and PureThermal cameras.

### Temporal Calibration Cue

- [ ] Build a multimodal synchronization paddle.
- [ ] Place a stable, matte warm target in the shared RGB/thermal field of view.
- [ ] Use a room-temperature opaque paddle with a high-contrast checkerboard
      or ArUco marker to cover and reveal the warm target.
- [ ] Prefer mechanical occlusion over an LED or heating element so the RGB and
      thermal changes happen physically at the same instant.
- [ ] If practical, drive the paddle with a servo using irregular or
      pseudo-random open/closed intervals.
- [ ] Record at least 15-30 transitions, with calibration events near the
      beginning, middle, and end of a longer recording.

### Temporal Offset Analysis

- [x] Extract a one-dimensional motion-energy cue signal from each stream.
  - RGB candidates: frame-difference energy, paddle-edge position, or marker
    visibility.
  - Thermal candidates: frame-difference energy or mean temperature in the
    warm-target region.
- [x] Cross-correlate the signals over a configurable lag window, initially
      `-2` to `+2` seconds.
- [x] Refine the correlation peak to estimate a fractional-frame offset.
- [x] Detect individual cover/reveal transitions and use their median time
      difference as a robust offset estimate.
- [ ] Fit an affine clock mapping:

  ```text
  rgb_time = scale * thermal_time + offset
  ```

- [ ] Treat `offset` as the initial pipeline delay and `scale` as clock drift.
- [ ] Detect discontinuities that indicate dropped or buffered frames rather
      than gradual clock drift.
- [x] Report first-pass offset, equivalent frame lag, matched transitions,
      residual jitter, and correlation strength.
- [ ] Extend the report with:
  - thermal lead or lag in seconds
  - equivalent lead or lag in frames
  - uncertainty
  - number of transitions evaluated
  - fitted drift
  - residual timing error
  - suspected dropped-frame discontinuities
- [ ] Apply the fitted correction to thermal timestamps before nearest-time RGB
      and thermal frame pairing.

### Timestamp Caveat

The capture pipeline records host-side monotonic and UTC arrival timestamps for
every stream. RGB also records `SensorTimestamp` when Picamera2 exposes it, and
RealSense records native frame timestamps and frame numbers. Thermal currently
has only host-arrival timing. These clocks are not a shared hardware exposure
clock. Camera, USB, driver, and buffering latency can therefore produce
visually misaligned frames even when their recorded host timestamps are close.

### Spatial Calibration

- [ ] Build a visible/thermal calibration board using a regular array of holes
      over a warm background.
- [ ] Confirm that the same control points are clearly detectable as visible
      edges in RGB and warm circles in thermal.
- [ ] Record the board at multiple depths representative of the nest floor,
      mid-height structures, and upper structures.
- [ ] Estimate RGB and thermal camera intrinsics and their relative pose where
      the available image quality permits.
- [ ] Compare a single-plane homography against a multi-depth or depth-aware
      mapping.
- [ ] Use calibrated depth information, rather than a single homography, for
      bees and nest components substantially above or below the reference
      plane.

### Validation and Acceptance Criteria

- [ ] Reserve several paddle transitions as holdout events not used to fit the
      timing correction.
- [ ] Validate spatial mapping using points and depths not used during fitting.
- [ ] Generate a diagnostic overlay video showing corrected RGB/thermal pairing,
      frame timestamps, matched frame numbers, and residual time difference.
- [ ] Confirm whether the currently observed thermal lead is constant,
      gradually drifting, or caused by intermittent buffering.
- [ ] Document repeatability across the HQ camera and OwlSight camera profiles.
- [ ] Consider the feature ready when repeated calibration recordings produce
      stable offset estimates and holdout transitions align within the expected
      uncertainty imposed by the recording frame rates.
