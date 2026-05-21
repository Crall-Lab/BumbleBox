# BumbleBox Desktop Scaffold

This folder is a separate-style downstream tool scaffold for desktop analysis.

It ingests BumbleBox run bundles (`bbx.run_bundle.v1`) and performs:

- manifest validation (including optional SHA256 file hash checks)
- tracking CSV ingestion from the bundle contract
- first-pass analysis outputs for publication pipeline development

## Entry point

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py --help
```

## Validate a bundle

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py validate-bundle \
  --bundle /path/to/bbx_bundle.zip
```

## Run analysis

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py analyze \
  --bundle /path/to/bbx_bundle_or_zip \
  --output-dir /path/to/output \
  --with-tracked-video
```

Outputs include:

- `tracking_overview.json`
- `frame_summary.csv`
- `id_summary.csv`
- `tracking_with_heading.csv`
- `tracking_overlay.mp4` (when `--with-tracked-video` is used and bundle includes MP4)
- `pipeline_report.json`

## Render tracked video only

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx_desktop.py visualize \
  --bundle /path/to/bbx_bundle_or_zip \
  --output /path/to/tracking_overlay.mp4
```

## Notes

- `--with-segmentation` and `--with-classifier` are scaffold flags only for now.
- Heading angle is computed downstream by default (`--no-heading` disables it).
- This scaffold is designed to split into its own repository cleanly.
