# Run Bundle Export

`export-bundle` packages one BumbleBox run into a portable folder (and zip) for downstream desktop analysis.

## Command

Latest run under configured data root:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py export-bundle --latest
```

Specific run summary:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py export-bundle \
  --summary /path/to/session_run_summary.json
```

By session directory:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py export-bundle \
  --session-dir /path/to/session_folder
```

GUI path:

1. Open `Run & Schedule`.
2. Select a run in `Recent Runs`.
3. Set bundle options.
4. Click `Export Selected Run Bundle`.

## What gets included

Core run artifacts (when present):

- `*_run_summary.json`
- `*_config_snapshot.json` (exact config captured at runtime when available)
- `*.mp4` (unless `--skip-video`)
- `*_frame_timestamps.csv`
- `*_actual_fps.txt`
- `*_raw.csv`
- `*_noID.csv`
- `*_cleaned.csv`
- `*_fps_report.json`

Plus:

- current config file snapshot (unless `--no-config`)
- extra files in session directory that match the session prefix (unless `--core-only`)

## Output structure

```
<bundle_name>/
  bundle_manifest.json
  artifacts/
    ...
<bundle_name>.zip
```

## Manifest schema (`bbx.run_bundle.v1`)

`bundle_manifest.json` contains:

- bundle metadata (`schema_version`, `created_at`, `bundle_name`)
- run metadata (`session_name`, run start/finish, mode, success)
- `preferred_tracking_csv` (cleaned CSV when available, else raw CSV)
- artifact list with:
  - `kind`
  - `required`
  - `source_path`
  - `bundle_relpath`
  - `size_bytes`
  - `sha256`
- `missing_expected` list for absent optional/required files

This manifest is intended as the contract for downstream ingestion pipelines.
