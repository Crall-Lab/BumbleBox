# Desktop Downstream Scaffold

This scaffold is the first step toward moving advanced behavior analytics off Raspberry Pi and onto desktop hardware.

## Why this split

- Pi side remains focused on acquisition reliability.
- Desktop side handles heavier metrics/modeling workloads.
- Data exchange is standardized through `bbx.run_bundle.v1`.

## Current scope

Path:

- `/Users/aec/Desktop/BumbleBox/bumblebox_desktop`
- entrypoint: `/Users/aec/Desktop/BumbleBox/bbx_desktop.py`

Commands:

1. Validate bundle:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py validate-bundle \
  --bundle /path/to/bbx_bundle.zip
```

2. Analyze bundle:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py analyze \
  --bundle /path/to/bbx_bundle_or_folder \
  --output-dir /path/to/output \
  --with-tracked-video
```

3. Render tracked video only:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py visualize \
  --bundle /path/to/bbx_bundle_or_folder \
  --output /path/to/tracking_overlay.mp4
```

## What analyze currently does

- Loads and validates manifest schema/version.
- Verifies file presence and optionally SHA256 hashes.
- Selects tracking CSV from contract (`preferred_tracking_csv` fallback to cleaned/raw artifact kinds).
- Builds basic summary outputs (`tracking_overview.json`, `frame_summary.csv`, `id_summary.csv`).
- Computes heading angle downstream by default (`tracking_with_heading.csv`).
- Can render create_tracked_videos-style overlay output (`tracking_overlay.mp4`).
- Writes `pipeline_report.json` with stage statuses.

## Planned next phases

1. Segmentation-assisted re-tracking module.
2. ID repair and occlusion handling.
3. Behavior-classifier model stage.
4. Paper-ready figure/report generation.
5. GUI for drag-and-drop bundle analysis.
