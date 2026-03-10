# FPS Sweep Capacity Test

`fps-sweep` probes increasing target framerates and reports:

- target FPS vs measured real FPS
- estimated maximum recording duration at safe/warn/high-risk capacity budgets
- estimated tracking time (when recent tracking benchmark data exists)

This is intended to answer: "How long can I record at higher FPS on this hardware?"

The capacity model depends on the current BumbleBox workflow:

- `MP4` recording uses a RAM-backed model because frames are captured into memory first and encoded afterward.
- `record_and_track` with `tracking_source=ram` also uses a RAM-backed model, even if the recording codec is `MJPEG`, because frames still need to stay in memory for tracking.
- `MJPEG` recording without RAM-backed tracking uses a disk-backed model. The sweep writes short MJPEG probe clips under `system.data_root`, measures file growth rate, and estimates how long recording can continue before free space is exhausted.

## CLI

Range mode:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep \
  --fps-start 2 \
  --fps-stop 20 \
  --fps-step 2 \
  --probe-seconds 20 \
  --assume-ram-gb 2
```

Explicit list mode:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep \
  --fps-values 2,4,6,8,10,12 \
  --probe-seconds 20
```

Use mock camera for dry testing:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep \
  --fps-values 2,4,6 \
  --probe-seconds 5 \
  --mock-camera
```

Session-scoped tracking benchmark (optional):

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fps-sweep \
  --fps-start 2 --fps-stop 20 --fps-step 2 \
  --session-start 2026-02-17T10:30:00
```

## GUI

Open `FPS Report` tab:

1. Use `FPS Sweep Capacity Test`.
2. Set either `FPS list` or `Start/Stop/Step`.
3. Set `Probe seconds`.
4. Optionally set `Assume RAM GiB` if you want to simulate a Pi target for a RAM-backed workflow.
5. Click `Run FPS Sweep`.

The GUI defaults to using tracking benchmark data from the current app session only.

## Tracking-time estimates

If a recent successful tracking run is found, the report includes estimated tracking time for:

- configured recording duration
- safe/warn/high-risk recording-duration budgets

Benchmark confidence is shown:

- `high`: direct tracking timing fields from recent run summaries
- `low`: inferred from wall-clock run duration minus recording duration
