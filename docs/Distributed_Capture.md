# Distributed Multi-Pi Capture

BumbleBox can optionally coordinate RGB, thermal, and RealSense capture across
two Raspberry Pi 5 computers. The feature is disabled by default, so existing
single-Pi profiles and commands retain their previous behavior.

## Recommended Topology

- Primary Pi: GUI, automated schedule, OwlSight or HQ RGB camera, and
  PureThermal camera
- Second Pi: RealSense D405
- Both Pis: local data storage, synchronized system clocks, the same BumbleBox
  Git revision, and a passwordless SSH connection from the primary Pi

The assignment is configurable. RGB, thermal, and RealSense can each be moved
between the two Pis in the setup wizard without changing the capture protocol.
Camera data is recorded locally on each Pi; it is not streamed over Wi-Fi while
recording. SSH carries only coordination messages, and optional `rsync`
collection happens after capture.

## Why This Is Clock-Coordinated

The D405 does not provide the D457-style external hardware-sync connector, and
the OwlSight camera does not expose a documented external trigger through the
current Raspberry Pi camera stack. BumbleBox therefore uses:

1. Network time synchronization on both Pis.
2. A UTC start deadline sent in a versioned capture plan.
3. A prepare-and-arm barrier before that deadline.
4. Per-frame host monotonic and UTC arrival timestamps.
5. Native device timestamps where a camera exposes them.
6. A shared physical cue to measure residual offset and jitter.

This coordinates acquisition but does not claim simultaneous exposure. USB,
CSI, driver, buffering, and rolling-shutter latency remain measurable effects.

## First-Time Setup

1. Install Raspberry Pi OS and BumbleBox on both Pis.
2. Run the normal setup script on both Pis. The script installs `chrony`,
   `openssh-client`, and `rsync` on Raspberry Pi hosts.
3. Open the BumbleBox setup wizard on the primary Pi.
4. Enable **Use a second Raspberry Pi for simultaneous capture**.
5. Enter the worker host, SSH user, repository path, config path, and data root.
6. Assign each enabled sensor to the primary or second Pi.
7. Save the profile, then enroll and validate the worker:

```bash
./bbx distributed-capture setup-worker --node worker --install-runtime
./bbx distributed-capture check --probe-hardware
```

`setup-worker` creates or reuses the BumbleBox fleet SSH key, installs its
public key on the worker, verifies the repository/config paths, and optionally
runs the environment installer remotely. The first key installation may ask
for the worker password. Later operation is noninteractive.

Both clocks should report network synchronization. `chrony` is recommended,
but BumbleBox accepts any service that keeps the host clock synchronized. The
preflight uses several SSH clock samples, selects the minimum-round-trip
estimate, rejects a node that reports an unlocked network clock, and rejects
offsets above `maximum_clock_offset_ms`.

## Configuration

The setup wizard writes this section. It can also be edited directly:

```yaml
distributed_capture:
  enabled: true
  role: controller
  controller_node: primary
  start_lead_seconds: 8.0
  ready_timeout_seconds: 30.0
  maximum_clock_offset_ms: 5.0
  clock_samples: 5
  failure_policy: all_or_nothing
  transfer_after_capture: false
  require_same_revision: true
  nodes:
    - name: primary
      host: localhost
      local: true
      enabled: true
      sensors: [rgb, thermal]
      user: null
      port: 22
      repo_path: null
      config_path: null
      data_root: null
    - name: worker
      host: bumblebox-02.local
      local: false
      enabled: true
      sensors: [realsense]
      user: auggie_ec
      port: 22
      repo_path: /home/auggie_ec/Desktop/BumbleBox
      config_path: /home/auggie_ec/Desktop/BumbleBox/bumblebox_v2/config.yaml
      data_root: /home/auggie_ec/Desktop/BumbleBoxData
```

Important settings:

- `failure_policy: all_or_nothing` cancels the session if any node fails to arm.
- `failure_policy: continue_available` records with nodes that did arm and marks
  missing nodes as failures in the manifest.
- `transfer_after_capture: true` copies worker artifacts into the primary
  session after recording with resumable `rsync` staging.
- `require_same_revision: true` prevents a run when the Pis have different
  Git revisions.
- Each active sensor must have exactly one enabled node owner.

The controller sends effective capture settings in the plan but deliberately
does not overwrite the worker's device-specific thermal path or RealSense
serial number.

## Running

Once enabled, the normal command automatically uses the distributed
coordinator:

```bash
./bbx run-once --mode record_only
./bbx run-once --mode record_and_track
```

Use `--no-distributed` only for a deliberate local diagnostic run. Existing
systemd automation also calls `run-once`, so scheduled recordings use the same
distributed path without separate timers on the worker.

Useful diagnostics:

```bash
./bbx distributed-capture check
./bbx distributed-capture check --probe-hardware
./bbx distributed-capture plan --mode record_only
```

The GUI exposes the same readiness checks on the Run and Hardware pages.

## Session Records

Each distributed session has one plan ID and session name. The primary session
directory contains:

- `distributed_capture_plan.json`: immutable requested start, assignments,
  config hash, revision, and effective settings
- `distributed_capture_manifest.json`: readiness, preflight, node results,
  artifact inventories, separate capture/collection status, and errors
- `*_run_summary.json`: aggregate result used by the GUI
- `nodes/<node>/`: copied worker artifacts when post-capture transfer is enabled

Each node writes a local `node_capture_result.json`, overlap lock, camera
artifacts, and timestamp files. Timestamp rows identify the node, sensor,
timestamp source, host monotonic arrival, host UTC arrival, and native sensor
timestamp when available.

Large media files are inventoried by size. Smaller metadata files also receive
a SHA-256 checksum. This avoids rereading multi-gigabyte recordings solely to
build the manifest.

## Recovery

- Node and controller lock files prevent overlapping captures on one data root.
- A missed arm deadline or interrupted process is retained as a failed manifest
  when a session directory has already been created.
- Media already written to a worker remains there if the network drops.
- The manifest distinguishes `capture_success` from `collection_success`, so a
  failed copy does not imply that acquisition data was lost.
- Interrupted collection uses a hidden partial directory and can be resumed:

```bash
./bbx distributed-capture collect \
  --manifest /path/to/distributed_capture_manifest.json
```

The current remote capture is launched through the controller's SSH session.
Local-first writes protect completed frames, but a fully detached worker
systemd job is a worthwhile later hardening step for surviving a controller
power loss during acquisition.

## Measuring Residual Synchronization

Record a shared visual and thermal cue, such as a room-temperature paddle that
reveals and covers a matte warm target. Keep the cue within a fixed normalized
region if possible, then run:

```bash
./bbx distributed-capture analyze-sync \
  --summary /path/to/session_run_summary.json \
  --roi 0.2,0.2,0.5,0.5 \
  --max-lag-seconds 2
```

The analyzer extracts motion-energy traces, estimates fractional-frame offset
by cross-correlation, matches cue transitions, reports residual jitter, and
writes `sync_validation_report.json`. Its sign convention is:

```text
reference_time = sensor_time + offset_seconds
```

A positive offset moves the sensor timestamps later; a negative offset moves
them earlier. Use multiple irregular transitions near the beginning, middle,
and end of a longer run. Affine drift fitting and automatic correction of frame
pairing remain future calibration work.

## Hardware Validation Still Required

The controller/worker protocol and hardware-free capture path are covered by
automated tests. Production acceptance still requires two physical Pi 5 units
with the selected camera assignment. Measure clock offset, arm reliability,
frame loss, storage throughput, CPU/RAM load, fixed sensor delay, and timing
jitter before unattended experiments.
