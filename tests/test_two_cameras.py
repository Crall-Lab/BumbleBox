"""Two Raspberry Pi cameras on one Pi: camera selection, the second config, recording together."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bumblebox_v2 import cli, run_engine
from bumblebox_v2.camera_select import (
    CameraSelectionError,
    DetectedCamera,
    capture_lock_suffix,
    parse_camera_info,
    picamera2_camera_kwargs,
    resolve_camera_num,
    session_tag,
)
from bumblebox_v2.capture_lock import capture_lock, default_lock_path
from bumblebox_v2.config import ConfigError, load_config, load_defaults, save_config, validate_config
from bumblebox_v2.multi_camera import (
    CameraEntry,
    check_camera_set,
    combined_memory,
    copy_schedule,
    create_second_camera,
    has_second_camera,
    load_cameras,
    pin_cameras,
    remove_second_camera,
    schedules_match,
    second_camera_config_path,
    suggest_second_colony_id,
)
from bumblebox_v2.status_history import list_recent_run_records

CAM_A = "/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a"
CAM_B = "/base/axi/pcie@120000/rp1/i2c@80000/imx477@1a"
TWO_CAMERAS = [DetectedCamera(0, "imx477", CAM_A), DetectedCamera(1, "imx477", CAM_B)]


def camera_config(**camera):
    config = load_defaults()
    config["camera"].update(camera)
    return config


class CameraSelectionTests(unittest.TestCase):
    def test_camera_info_is_parsed_with_or_without_numbers(self) -> None:
        cameras = parse_camera_info([{"Model": "imx477", "Id": CAM_A, "Num": 0}, {"Model": "imx477", "Id": CAM_B}])
        self.assertEqual([(camera.num, camera.camera_id) for camera in cameras], [(0, CAM_A), (1, CAM_B)])
        self.assertEqual(cameras[0].description, "imx477 on connector i2c@88000")
        self.assertEqual(parse_camera_info(None), [])

    def test_default_config_uses_the_first_camera(self) -> None:
        self.assertEqual(resolve_camera_num(load_defaults(), TWO_CAMERAS), 0)
        self.assertEqual(resolve_camera_num(load_defaults(), None), 0)

    def test_pinned_camera_is_found_wherever_it_is_numbered(self) -> None:
        config = camera_config(camera_num=1, camera_id=CAM_B)
        self.assertEqual(resolve_camera_num(config, TWO_CAMERAS), 1)
        renumbered = [DetectedCamera(0, "imx477", CAM_B), DetectedCamera(1, "imx477", CAM_A)]
        self.assertEqual(resolve_camera_num(config, renumbered), 0)

    def test_unplugged_pinned_camera_is_refused_not_replaced(self) -> None:
        config = camera_config(camera_num=1, camera_id=CAM_B)
        with self.assertRaises(CameraSelectionError) as raised:
            resolve_camera_num(config, TWO_CAMERAS[:1])
        self.assertIn("not connected", str(raised.exception))
        self.assertIn("rather than using a different camera", str(raised.exception))

    def test_camera_number_beyond_what_is_connected_is_refused(self) -> None:
        with self.assertRaises(CameraSelectionError) as raised:
            resolve_camera_num(camera_config(camera_num=1), TWO_CAMERAS[:1])
        self.assertIn("only 1 camera(s) are connected", str(raised.exception))

    def test_single_camera_box_opens_its_camera_exactly_as_before(self) -> None:
        class NeverAsked:
            @staticmethod
            def global_camera_info():
                raise AssertionError("a default config must not need camera detection")

        self.assertEqual(picamera2_camera_kwargs(load_defaults(), NeverAsked), {})

    def test_second_camera_is_opened_by_number(self) -> None:
        class Picamera2:
            @staticmethod
            def global_camera_info():
                return [{"Model": "imx477", "Id": CAM_A, "Num": 0}, {"Model": "imx477", "Id": CAM_B, "Num": 1}]

        self.assertEqual(picamera2_camera_kwargs(camera_config(camera_num=1, camera_id=CAM_B), Picamera2), {"camera_num": 1})
        self.assertEqual(picamera2_camera_kwargs(camera_config(camera_id=CAM_A), Picamera2), {})

    def test_config_validation(self) -> None:
        validate_config(camera_config(camera_num=1, camera_id=CAM_B, session_tag="cam2"))
        for bad in ({"camera_num": -1}, {"camera_num": "1"}, {"camera_id": ""}, {"session_tag": "cam 2"}, {"session_tag": "x" * 17}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                validate_config(camera_config(**bad))

    def test_tags_and_lock_names(self) -> None:
        self.assertEqual(session_tag(load_defaults()), "")
        self.assertEqual(session_tag(camera_config(session_tag="cam2")), "cam2")
        self.assertEqual(capture_lock_suffix(load_defaults()), "")
        self.assertEqual(capture_lock_suffix(camera_config(camera_num=1)), "cam1")
        with patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": "/tmp/x/capture.lock"}):
            self.assertEqual(default_lock_path(), Path("/tmp/x/capture.lock"))
            self.assertEqual(default_lock_path("cam1"), Path("/tmp/x/capture-cam1.lock"))


class SecondCameraSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.primary = self.root / "config.yaml"
        config = load_defaults()
        config["system"]["data_root"] = str(self.root / "data")
        config["thermal"]["enabled"] = True
        config["scheduling"]["enabled"] = True
        save_config(self.primary, config)

    def test_second_config_sits_next_to_the_first(self) -> None:
        self.assertEqual(second_camera_config_path(self.primary), self.root / "config.camera2.yaml")
        self.assertFalse(has_second_camera(self.primary))
        self.assertEqual(len(load_cameras(self.primary)), 1)

    def test_second_camera_copies_settings_but_not_devices_or_the_running_schedule(self) -> None:
        result = create_second_camera(self.primary, cameras=[])
        self.assertFalse(result.pinned)
        self.assertIn("not pinned", " ".join(result.notes))
        first, second = load_cameras(self.primary)
        self.assertEqual((second.index, second.label, second.long_label), (2, "Camera 2", "Camera 2 (colony 02)"))
        self.assertEqual(second.config["camera"]["camera_num"], 1)
        self.assertEqual(second.config["camera"]["session_tag"], "cam2")
        self.assertEqual(second.config["scheduling"]["unit_prefix"], "bumblebox-v2-cam2")
        self.assertFalse(second.config["scheduling"]["enabled"])
        self.assertFalse(second.config["thermal"]["enabled"])
        self.assertEqual(second.config["system"]["data_root"], first.config["system"]["data_root"])
        self.assertEqual(second.config["capture"], first.config["capture"])
        # The first camera is untouched when nothing could be pinned.
        self.assertTrue(first.config["thermal"]["enabled"])
        self.assertIsNone(first.config["camera"]["camera_id"])
        self.assertEqual(check_camera_set([first, second]), [])
        with self.assertRaises(FileExistsError):
            create_second_camera(self.primary, cameras=[])

    def test_two_connected_cameras_are_pinned_one_each(self) -> None:
        result = create_second_camera(self.primary, colony_id="B7", data_root=str(self.root / "other"), cameras=TWO_CAMERAS)
        self.assertTrue(result.pinned)
        first, second = load_cameras(self.primary)
        self.assertEqual((first.config["camera"]["camera_num"], first.config["camera"]["camera_id"]), (0, CAM_A))
        self.assertEqual((second.config["camera"]["camera_num"], second.config["camera"]["camera_id"]), (1, CAM_B))
        self.assertEqual(second.config["system"]["colony_id"], "B7")
        self.assertEqual(second.config["system"]["data_root"], str(self.root / "other"))
        self.assertTrue(first.pinned and second.pinned)

    def test_cameras_can_be_pinned_later(self) -> None:
        create_second_camera(self.primary, cameras=TWO_CAMERAS[:1])
        self.assertFalse(pin_cameras(self.primary, cameras=TWO_CAMERAS[:1]))
        self.assertTrue(pin_cameras(self.primary, cameras=TWO_CAMERAS))
        first, second = load_cameras(self.primary)
        self.assertEqual({first.config["camera"]["camera_id"], second.config["camera"]["camera_id"]}, {CAM_A, CAM_B})

    def test_removing_keeps_the_settings_under_another_name(self) -> None:
        create_second_camera(self.primary, cameras=[])
        kept = remove_second_camera(self.primary)
        self.assertTrue(kept.is_file())
        self.assertIn(".removed-", kept.name)
        self.assertFalse(has_second_camera(self.primary))
        with self.assertRaises(FileNotFoundError):
            remove_second_camera(self.primary)

    def test_colony_suggestions(self) -> None:
        for current, expected in (("01", "02"), ("9", "10"), ("A", "A-2"), ("", "02")):
            config = load_defaults()
            config["system"]["colony_id"] = current
            self.assertEqual(suggest_second_colony_id(config), expected)

    def test_conflicting_pair_is_reported(self) -> None:
        first = CameraEntry(1, self.primary, load_defaults())
        clash = CameraEntry(2, self.root / "x.yaml", load_defaults())
        problems = " ".join(check_camera_set([first, clash]))
        self.assertIn("same timer names", problems)
        self.assertIn("same camera", problems)
        self.assertIn("name their sessions the same way", problems)

    def test_schedules_can_be_kept_in_step(self) -> None:
        first, second = load_defaults(), load_defaults()
        self.assertTrue(schedules_match(first, second))
        first["capture"]["record_interval_minutes"] = 15
        first["pipeline"]["mode"] = "record_only"
        self.assertFalse(schedules_match(first, second))
        self.assertTrue(copy_schedule(first, second))
        self.assertTrue(schedules_match(first, second))
        self.assertFalse(copy_schedule(first, second))

    def test_memory_is_judged_for_both_cameras_together(self) -> None:
        class Report:
            def __init__(self, used):
                self.metrics = {"estimated_frame_buffer_bytes": used, "total_ram_bytes": 8 * 2**30}

        entries = [CameraEntry(1, self.primary, {"use": 2}), CameraEntry(2, self.primary, {"use": 2})]
        check = lambda config: Report(config["use"] * 2**30)  # noqa: E731
        self.assertIsNone(combined_memory(entries[:1], schedule_check=check))
        self.assertEqual(combined_memory(entries, schedule_check=check).level, "ok")  # 4 of 8 GiB
        entries[1].config["use"] = 3
        self.assertEqual(combined_memory(entries, schedule_check=check).level, "warn")  # 5 of 8 GiB
        entries[0].config["use"] = 4
        load = combined_memory(entries, schedule_check=check)  # 7 of 8 GiB
        self.assertEqual(load.level, "fail")
        self.assertIn("7.0 GiB of 8.0 GiB", load.message)


class RecordingTogetherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.lock_path = self.root / "capture.lock"
        lock = patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(self.lock_path)})
        lock.start()
        self.addCleanup(lock.stop)
        self.primary = self.root / "config.yaml"
        config = load_defaults()
        config["system"]["data_root"] = str(self.root / "data")
        config["local_index"]["enabled"] = False
        config["pipeline"]["mode"] = "record_only"
        config["camera"].update({"width": 64, "height": 48, "codec": "mjpeg"})
        config["capture"]["recording_seconds"] = 0.5
        config["runtime"].update(
            {"use_mock_camera": True, "fps_report_on_each_recording": False, "capture_lock_wait_seconds": 0.2}
        )
        save_config(self.primary, config)
        create_second_camera(self.primary, cameras=[])
        self.first, self.second = load_cameras(self.primary)

    def test_second_camera_records_while_the_first_holds_its_lock(self) -> None:
        with capture_lock(0, lock_path=default_lock_path()):
            blocked = run_engine.run_once(config=self.first.config)
            together = run_engine.run_once(config=self.second.config)
        self.assertFalse(blocked.success)
        self.assertIn("Another BumbleBox capture is using the camera", " ".join(blocked.errors))
        self.assertTrue(together.success, together.errors)

    def test_second_camera_sessions_are_tagged_and_filterable(self) -> None:
        one = run_engine.run_once(config=self.first.config)
        two = run_engine.run_once(config=self.second.config)
        self.assertNotIn("_cam2_", one.session_name)
        self.assertIn("_cam2_", two.session_name)
        self.assertEqual((one.camera_num, one.camera_session_tag), (0, ""))
        self.assertEqual((two.camera_num, two.camera_session_tag), (1, "cam2"))
        self.assertTrue(Path(two.video_path).name.startswith(two.session_name))

        data_root = self.first.config["system"]["data_root"]
        self.assertEqual(len(list_recent_run_records(data_root)), 2)
        self.assertEqual([r.session_name for r in list_recent_run_records(data_root, session_tag="")], [one.session_name])
        self.assertEqual(
            [r.session_name for r in list_recent_run_records(data_root, session_tag="cam2")], [two.session_name]
        )

    def test_older_summaries_without_a_tag_belong_to_the_first_camera(self) -> None:
        session = self.root / "data" / "2026-01-01" / "old"
        session.mkdir(parents=True)
        (session / "old_run_summary.json").write_text(json.dumps({"session_name": "old", "success": True}))
        records = list_recent_run_records(self.root / "data", session_tag="")
        self.assertEqual([record.session_name for record in records], ["old"])

    def run_cli(self, *arguments: str) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(list(arguments))
        return code, output.getvalue()

    def test_record_both_cameras_command(self) -> None:
        code, output = self.run_cli("run-once", "--all-cameras", "--config", str(self.primary))
        self.assertEqual(code, 0, output)
        self.assertIn("[camera 1] ", output)
        self.assertIn("[camera 2] ", output)
        self.assertIn("Camera 1: recorded.", output)
        self.assertIn("Camera 2: recorded.", output)
        tags = sorted(record.camera_session_tag for record in list_recent_run_records(self.root / "data"))
        self.assertEqual(tags, ["", "cam2"])

    def test_record_both_reports_which_camera_failed_and_why(self) -> None:
        second = load_config(self.second.path)
        second["system"]["min_free_space_gb"] = 10_000_000  # the storage guard refuses this camera's run
        save_config(self.second.path, second)
        code, output = self.run_cli("run-once", "--all-cameras", "--config", str(self.primary))
        self.assertEqual(code, 1, output)
        self.assertIn("Camera 1: recorded.", output)
        self.assertIn("Camera 2 failed: Only ", output)

    def test_second_camera_commands(self) -> None:
        code, output = self.run_cli("second-camera", "status", "--config", str(self.primary))
        self.assertEqual(code, 0, output)
        self.assertIn("Camera 2 (colony 02)", output)
        self.assertIn("not pinned", output)
        self.assertIn("Schedules: the same", output)

        code, output = self.run_cli("second-camera", "add", "--config", str(self.primary))
        self.assertEqual(code, 1)
        self.assertIn("already set up", output)

        with patch.object(cli, "run_systemd_action", side_effect=RuntimeError("no systemd here")):
            code, output = self.run_cli("second-camera", "remove", "--config", str(self.primary))
        self.assertEqual(code, 0, output)
        self.assertIn("recordings were not touched", output)
        self.assertFalse(has_second_camera(self.primary))


if __name__ == "__main__":
    unittest.main()
