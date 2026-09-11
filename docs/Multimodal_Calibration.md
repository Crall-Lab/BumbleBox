# RGB, Thermal, and Depth Calibration

## Objective

BumbleBox must associate an RGB observation with the correct thermal measurement and RealSense depth sample. This requires two independent calibrations:

1. **Temporal calibration** estimates when each sensor exposed or delivered a frame.
2. **Spatial calibration** estimates where the same three-dimensional point appears in each sensor.

A shared host start deadline is useful, but it does not make exposures simultaneous. A single RGB-to-thermal homography is also insufficient for a nest containing bees, pots, brood, and other structures at different heights.

## Project Structure

Create a versioned calibration project:

```bash
./bbx calibration-project init \
  --project /path/to/BumbleBoxCalibration \
  --name "BumbleBox three-camera calibration"
```

The command creates:

```text
BumbleBoxCalibration/
  calibration_project.json
  README.md
  captures/
  temporal/
  spatial/
  validation/
```

The manifest records the configured RGB profile, thermal settings, RealSense profiles, calibration captures, coordinate-frame goal, expected output artifacts, and stage readiness. It stores references to the original recording files; it does not duplicate large videos or depth arrays. The `captures/` directory is reserved for lightweight extracted cue traces, correspondence annotations, and other calibration-only derivatives.

Register a successful synchronized recording:

```bash
./bbx calibration-project add-session \
  --project /path/to/BumbleBoxCalibration \
  --summary /path/to/session/session_run_summary.json \
  --role calibration \
  --depth-layer floor \
  --notes "Occlusion paddle and heated target; target on nest floor"
```

Use distinct labels such as `floor`, `mid`, and `upper`. Register held-out recordings with `--role validation`. Re-registering the same summary is idempotent.

Inspect readiness at any time:

```bash
./bbx calibration-project status --project /path/to/BumbleBoxCalibration
```

## Capture Protocol

### Timing

Use one physical event visible to all three sensors. A matte, room-temperature paddle that occludes a warm target is preferable to unrelated LEDs and heaters: the visible edge, depth change, and thermal transition occur at the same physical instant.

- Place the cue inside the common field of view.
- Record 15 to 30 irregular cover/reveal transitions.
- Include transitions near the beginning, middle, and end of a longer capture.
- Do not overwrite source timestamps when applying a correction.
- Repeat the timing test for HQ and OwlSight profiles because their pipelines may have different latency.

Extract a one-dimensional transition signal from each stream and fit both offset and drift:

```text
rgb_time = scale * sensor_time + offset_seconds
```

Use host monotonic clocks to compare processes on the Pi. Preserve RealSense device timestamps and RGB sensor timestamps as separate domains. Cross-correlation gives a starting offset; robust matching of individual transitions should produce the final estimate and uncertainty.

Write the result to `temporal/temporal_corrections.json`. It should include the source timestamp domain, reference domain, offset, scale, residual distribution, transition count, fit interval, and discontinuities that may indicate buffered or dropped frames.

### Geometry

Use a calibration target observable in visible RGB, thermal, and depth. A rigid board with a visible fiducial pattern and thermally distinct holes or inserts is a practical candidate.

1. Calibrate RGB intrinsics for each interchangeable primary-camera profile and resolution.
2. Retain the RealSense intrinsics, depth scale, and depth-to-color extrinsics reported by its SDK.
3. Estimate RGB-to-RealSense pose using corresponding visible target features.
4. Estimate RGB-to-thermal geometry from corresponding thermal/visible features.
5. Repeat at three or more nest-depth layers and across the usable field of view.
6. Compare a planar homography baseline against the depth-aware mapping.

The preferred common coordinate frame is the RealSense depth frame. For an RGB point, infer or sample its depth, back-project it to 3D, transform the point between camera coordinate frames, and project it into thermal pixels. If full camera calibration is unreliable at the thermal resolution, use a depth-conditioned piecewise mapping and report interpolation uncertainty.

Expected spatial artifacts are:

- `spatial/rgb_thermal_registration.json`
- `spatial/rgb_realsense_calibration.json`
- `spatial/common_frame_calibration.json`

Each artifact should record image dimensions, camera profile, intrinsic/extrinsic matrices or fitted model, point correspondences, depth layers, residuals, and software version.

## Validation

Do not validate using the same frames or point correspondences used to fit the transforms. A held-out report should include:

- temporal residual in milliseconds and equivalent frames for each camera pair
- reprojection error in RGB, thermal, and RealSense pixels
- mapping error versus physical depth
- invalid-depth and out-of-bounds rates
- temperature sampling uncertainty within bee and nest-component masks
- separate results for HQ and OwlSight profiles

Write these results to `validation/validation_report.json`. Production temperature measurements should retain both the sampled value and calibration/registration quality flags.

## Implementation Boundary

The current calibration-project commands implement reproducible capture registration, stage readiness, and the artifact contract. They do not yet solve clock correction or camera geometry. This is intentional: those solvers need real synchronized calibration recordings and acceptance thresholds derived from hardware tests.
