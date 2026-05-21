# Legacy Scripts

This folder contains legacy BumbleBox scripts that are kept for reference and transitional compatibility.

These are not the recommended production path for new experiments.
Use V2 entry points instead:

- `/Users/aec/Desktop/BumbleBox/bbx.py`
- `/Users/aec/Desktop/BumbleBox/bbx_gui.py`

## Moved Scripts

- `config_loader.py`
- `generate_nest_images.py`
- `python-recording-functions.py`
- `ram_capture_tag_tracking.py`
- `record_video.py`
- `rpi4_preview.py`
- `setup.py`
- `start_automated_recording.py`
- `stop_automated_recording.py`
- `test_tracking.py`
- `track_prerecorded_videos.py`

## Moved Legacy Artifacts

- `notebooks/create_tracked_videos.0.12.ipynb`
- `samples/cumulative_averages.csv`

## Notes

- V2 still imports legacy core modules at repository root:
  - `/Users/aec/Desktop/BumbleBox/tag_tracking_utils.py`
  - `/Users/aec/Desktop/BumbleBox/data_cleaning.py`
  - `/Users/aec/Desktop/BumbleBox/behavioral_metrics.py`
- Nest labeling script remains at root because V2 launcher expects it there:
  - `/Users/aec/Desktop/BumbleBox/LabelNests_GUI.1.16.py`
