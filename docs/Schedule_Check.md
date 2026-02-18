# Schedule Check

Use `schedule-check` to estimate whether your recording/tracking cadence is likely to work on your hardware.

## CLI

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py schedule-check \
  --benchmark-input /path/to/video_or_images \
  --assume-ram-gb 2
```

`--benchmark-input` is optional but recommended. It measures detection throughput on representative footage instead of relying only on heuristics.
`--assume-ram-gb` is optional and lets you simulate target hardware RAM (for example `2` for a Pi 4B 2GB target) even when running the check on a desktop.

## What it checks

- estimated frame-buffer RAM footprint for your configured resolution/FPS/duration
- record interval fit
- track interval fit
- mixed schedule utilization
- camera resolution sanity for HQ (`4056x3040`) and Module 3 (`4608x2592`)

## Output

The report includes:

- PASS/WARN/FAIL checks
- key computed metrics
- RAM source used for estimation (`detected_host`, config override, or CLI assumption)
- concrete suggestions (for example reducing FPS, increasing intervals, using video tracking source)
