# Tracking Optimization (ArUco)

BumbleBox V2 now includes an ArUco parameter optimizer designed to tune tracking parameters from representative footage.

## Entry points

- CLI: `python3 /Users/aec/Desktop/BumbleBox/bbx.py optimize-tracking ...`
- GUI: `Optimize Tracking` tab in `bbx_gui.py`

## Core command

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py optimize-tracking \
  --input /path/to/video_or_images \
  --execution-target pi_safe \
  --tag-size-mm 2.5 \
  --early-stop-patience 40 \
  --apply-best
```

Defaults are tuned for small BumbleBox tags:

- default tag size: `2.5 mm`
- HQ camera reference resolution: `4056x3040`
- Module 3 reference resolution: `4608x2592`

## Execution target modes

- `pi_safe`: conservative worker defaults for Raspberry Pi.
- `desktop`: higher worker defaults for faster runs on workstations.

These are mutually exclusive in the GUI via radio buttons.

## Useful options

- `--profile quick|balanced|deep`
- `--sample-frames N`
- `--tag-size-mm N` (default 2.5)
- `--workers N` (manual override)
- `--expected-tags N` (optional scoring guidance)
- `--early-stop-patience N` (0 disables early stop)
- `--early-stop-min-improvement X`
- `--write-preview --preview-frames N`
- `--apply-best` (updates `tracking.aruco_params` in config)

## Outputs

Each run writes to a timestamped folder:

- `optimization_summary.json`
- `candidate_scores.csv`
- optional `best_params_preview.mp4`

## Deprecated script

`tracking-optimization.0.6.py` is deprecated and now forwards to the new optimizer.
