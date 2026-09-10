# BumbleBox Development Backlog

This document tracks research and engineering work that is not part of the
operator setup roadmap. Check off subtasks as they are implemented and
validated.

## RGB and Thermal Camera Synchronization

**Status:** Planned

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

- [ ] Extract a one-dimensional cue signal from each stream.
  - RGB candidates: frame-difference energy, paddle-edge position, or marker
    visibility.
  - Thermal candidates: frame-difference energy or mean temperature in the
    warm-target region.
- [ ] Cross-correlate the signals over a configurable lag window, initially
      `-2` to `+2` seconds.
- [ ] Refine the correlation peak to estimate a fractional-frame offset.
- [ ] Detect individual cover/reveal transitions and use their median time
      difference as a robust offset estimate.
- [ ] Fit an affine clock mapping:

  ```text
  rgb_time = scale * thermal_time + offset
  ```

- [ ] Treat `offset` as the initial pipeline delay and `scale` as clock drift.
- [ ] Detect discontinuities that indicate dropped or buffered frames rather
      than gradual clock drift.
- [ ] Report:
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

The current capture pipeline timestamps frames after the RGB
`capture_array()` or thermal `read()` call returns. These are host-side frame
arrival timestamps, not timestamps from a shared hardware exposure clock.
Camera, USB, driver, and buffering latency can therefore produce visually
misaligned frames even when their recorded host timestamps are close.

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

