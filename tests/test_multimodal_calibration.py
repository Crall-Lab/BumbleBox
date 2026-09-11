from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from bumblebox_v2.config import load_defaults
from bumblebox_v2.multimodal_calibration import (
    MANIFEST_NAME,
    add_calibration_session,
    get_calibration_project_status,
    initialize_calibration_project,
)


def _summary_payload(name: str, *, role_index: int = 1) -> dict:
    return {
        "session_name": name,
        "started_at": f"2026-09-{role_index:02d}T12:00:00-05:00",
        "success": True,
        "frames_captured": 10,
        "actual_fps": 7.0,
        "timestamp_path": f"/data/{name}_rgb_timestamps.csv",
        "video_path": f"/data/{name}.mp4",
        "thermal_enabled": True,
        "thermal_frames_captured": 10,
        "thermal_actual_fps": 7.0,
        "thermal_timestamp_path": f"/data/{name}_thermal_timestamps.csv",
        "thermal_raw_npy_path": f"/data/{name}_thermal.npy",
        "realsense_enabled": True,
        "realsense_frames_captured": 40,
        "realsense_actual_fps": 30.0,
        "realsense_timestamp_path": f"/data/{name}_depth_timestamps.csv",
        "realsense_raw_depth_npy_path": f"/data/{name}_depth.npy",
        "realsense_color_video_path": f"/data/{name}_depth_color.avi",
        "warnings": [],
        "errors": [],
    }


class MultimodalCalibrationTests(unittest.TestCase):
    def test_initialize_and_preserve_sessions_on_force_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calibration"
            config = load_defaults()
            config["thermal"]["enabled"] = True
            config["realsense"]["enabled"] = True
            result = initialize_calibration_project(config, root, project_name="Three camera test")
            self.assertTrue(Path(result.manifest_path).is_file())
            self.assertTrue(Path(result.readme_path).is_file())
            self.assertTrue((root / "temporal").is_dir())

            summary_path = root / "run_summary.json"
            summary_path.write_text(json.dumps(_summary_payload("session-a")))
            first = add_calibration_session(root, summary_path, depth_layer="floor")
            duplicate = add_calibration_session(root, summary_path, depth_layer="floor")
            self.assertTrue(first.added)
            self.assertFalse(duplicate.added)

            initialize_calibration_project(config, root, force=True)
            manifest = json.loads((root / MANIFEST_NAME).read_text())
            self.assertEqual(len(manifest["capture_sets"]), 1)
            self.assertEqual(manifest["project_name"], "Three camera test")

    def test_status_tracks_multidepth_and_validation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calibration"
            config = load_defaults()
            config["thermal"]["enabled"] = True
            config["realsense"]["enabled"] = True
            initialize_calibration_project(config, root)

            for index, layer in enumerate(("floor", "mid", "upper"), start=1):
                summary = root / f"calibration_{index}_run_summary.json"
                summary.write_text(json.dumps(_summary_payload(f"calibration-{index}", role_index=index)))
                add_calibration_session(root, summary, depth_layer=layer)
            validation = root / "validation_run_summary.json"
            validation.write_text(json.dumps(_summary_payload("validation", role_index=4)))
            add_calibration_session(root, validation, role="validation", depth_layer="mid")

            status = get_calibration_project_status(root)
            self.assertEqual(status.complete_three_camera_sets, 3)
            self.assertEqual(status.distinct_depth_layers, 3)
            self.assertEqual(status.stages["temporal"], "ready")
            self.assertEqual(status.stages["spatial"], "ready")

            for relative_path in (
                "temporal/temporal_corrections.json",
                "spatial/rgb_thermal_registration.json",
                "spatial/rgb_realsense_calibration.json",
                "spatial/common_frame_calibration.json",
            ):
                path = root / relative_path
                path.write_text("{}")
            status = get_calibration_project_status(root)
            self.assertEqual(status.stages["temporal"], "complete")
            self.assertEqual(status.stages["spatial"], "complete")
            self.assertEqual(status.stages["validation"], "ready")

            (root / "validation" / "validation_report.json").write_text("{}")
            status = get_calibration_project_status(root)
            self.assertEqual(status.stages["validation"], "complete")

    def test_spatial_readiness_does_not_mix_rgb_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calibration"
            initialize_calibration_project(load_defaults(), root)
            for index, (layer, profile) in enumerate(
                (("floor", "hq_reference"), ("mid", "owlsight_reference"), ("upper", "custom")),
                start=1,
            ):
                summary = root / f"mixed_{index}_run_summary.json"
                payload = _summary_payload(f"mixed-{index}", role_index=index)
                payload["camera_profile"] = profile
                summary.write_text(json.dumps(payload))
                add_calibration_session(root, summary, depth_layer=layer)

            status = get_calibration_project_status(root)
            self.assertEqual(status.complete_three_camera_sets, 3)
            self.assertEqual(status.distinct_depth_layers, 3)
            self.assertEqual(status.stages["spatial"], "blocked")


if __name__ == "__main__":
    unittest.main()
