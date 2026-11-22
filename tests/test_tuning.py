from __future__ import annotations

from copy import deepcopy
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from bumblebox_v2 import tuning
from bumblebox_v2.camera_profiles import CAMERA_MODELS, infrared_filter_note
from bumblebox_v2.camera_select import selected_camera_model
from bumblebox_v2.config import load_defaults, validate_config
from bumblebox_v2.tuning import (
    TUNING_ENV,
    describe_tuning,
    infrared_enabled,
    inspect_camera_tuning_resolution,
    open_picamera2,
    release_camera_manager,
    resolve_camera_tuning_file,
)


def config_with(**camera) -> dict:
    config = load_defaults()
    config["camera"].update(camera)
    return config


class TuningFolders:
    """A stand-in for the folders libcamera keeps its tuning files in."""

    def __init__(self, test: unittest.TestCase, *, pipeline: str | None, files: dict[str, list[str]]) -> None:
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for folder, names in files.items():
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            for name in names:
                (self.root / folder / name).write_text("{}")
        for patcher in (
            mock.patch.object(tuning, "TUNING_ROOTS", (self.root,)),
            mock.patch.object(tuning, "pi_pipeline", lambda: pipeline),
        ):
            patcher.start()
            test.addCleanup(patcher.stop)

    def path(self, folder: str, name: str) -> str:
        return str((self.root / folder / name).resolve())


class InfraredSettingTests(unittest.TestCase):
    """camera.infrared alone decides NoIR or standard, whatever the camera model."""

    def test_every_camera_model_follows_the_infrared_setting(self) -> None:
        for model in CAMERA_MODELS:
            with self.subTest(model=model):
                self.assertTrue(infrared_enabled(config_with(model=model, infrared=True)))
                self.assertFalse(infrared_enabled(config_with(model=model, infrared=False)))

    def test_an_ordinary_hq_camera_with_its_filter_removed_gets_the_noir_file(self) -> None:
        self.assertEqual(resolve_camera_tuning_file(config_with(model="hq", infrared=True)), "imx477_noir.json")
        self.assertEqual(resolve_camera_tuning_file(config_with(model="hq", infrared=False)), "imx477.json")

    def test_a_camera_sold_as_noir_can_still_use_standard_tuning(self) -> None:
        self.assertEqual(resolve_camera_tuning_file(config_with(model="hq_noir", infrared=False)), "imx477.json")

    def test_a_model_sold_with_a_filter_is_no_longer_refused(self) -> None:
        for model in ("module3_standard", "module3_wide", "owlsight_64mp"):
            with self.subTest(model=model):
                config = config_with(model=model, infrared=True)
                validate_config(config)  # used to raise "marked as not IR-capable"
                self.assertTrue(infrared_enabled(config))
                self.assertIn("infrared filter", infrared_filter_note(config))
        self.assertEqual(
            resolve_camera_tuning_file(config_with(model="module3_wide", infrared=True)), "imx708_noir.json"
        )
        self.assertIsNone(infrared_filter_note(config_with(model="hq", infrared=True)))
        self.assertIsNone(infrared_filter_note(config_with(model="owlsight_64mp", infrared=False)))

    def test_a_config_without_the_setting_goes_by_the_model_name(self) -> None:
        config = load_defaults()
        del config["camera"]["infrared"]
        config["camera"]["model"] = "hq_noir"
        self.assertTrue(infrared_enabled(config))
        config["camera"]["model"] = "hq"
        self.assertFalse(infrared_enabled(config))

    def test_a_file_named_by_hand_wins(self) -> None:
        config = config_with(model="hq", infrared=True, tuning_file="my_tuning.json")
        self.assertEqual(resolve_camera_tuning_file(config, detected_sensor="imx477"), "my_tuning.json")
        self.assertEqual(inspect_camera_tuning_resolution(config)["source"], "explicit")


class ConnectedCameraTests(unittest.TestCase):
    """The connected camera says which sensor it is, so the model and size no longer matter."""

    def test_model_on_auto_at_any_size_uses_the_connected_cameras_file(self) -> None:
        # The live monitor's size, and a size no camera is known by.
        for width, height in ((1920, 1440), (2028, 1520), (4056, 3040)):
            with self.subTest(size=(width, height)):
                config = config_with(model="auto", infrared=True, width=width, height=height)
                self.assertEqual(resolve_camera_tuning_file(config, detected_sensor="imx477"), "imx477_noir.json")
                config["camera"]["infrared"] = False
                self.assertEqual(resolve_camera_tuning_file(config, detected_sensor="imx477"), "imx477.json")

    def test_without_a_camera_to_ask_the_size_is_still_a_hint(self) -> None:
        self.assertEqual(resolve_camera_tuning_file(config_with(model="auto", infrared=True)), "imx477_noir.json")
        self.assertIsNone(resolve_camera_tuning_file(config_with(model="auto", infrared=True, width=1920, height=1440)))

    def test_the_connected_camera_decides_over_the_model_in_the_config(self) -> None:
        config = config_with(model="hq", infrared=True)
        self.assertEqual(resolve_camera_tuning_file(config, detected_sensor="imx708_wide"), "imx708_wide_noir.json")
        info = inspect_camera_tuning_resolution(config, detected_sensor="imx708_wide")
        self.assertEqual((info["sensor"], info["sensor_source"]), ("imx708_wide", "connected camera"))

    def test_a_camera_that_calls_itself_noir_follows_the_setting_too(self) -> None:
        config = config_with(model="auto", infrared=True)
        self.assertEqual(resolve_camera_tuning_file(config, detected_sensor="imx708_noir"), "imx708_noir.json")
        config["camera"]["infrared"] = False
        self.assertEqual(resolve_camera_tuning_file(config, detected_sensor="imx708_noir"), "imx708.json")

    def test_a_name_that_is_not_a_sensor_name_is_ignored(self) -> None:
        config = config_with(model="hq", infrared=True)
        for name in (None, "", "../etc/passwd", "USB Camera: HD"):
            with self.subTest(name=name):
                self.assertEqual(resolve_camera_tuning_file(config, detected_sensor=name), "imx477_noir.json")

    def test_the_sensor_is_read_for_the_camera_the_config_opens(self) -> None:
        class TwoCameras:
            @staticmethod
            def global_camera_info():
                return [
                    {"Num": 0, "Model": "imx477", "Id": "/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a"},
                    {"Num": 1, "Model": "ov64a40", "Id": "/base/axi/pcie@120000/rp1/i2c@80000/ov64a40@36"},
                ]

        first = load_defaults()
        self.assertEqual(selected_camera_model(first, TwoCameras), "imx477")
        second = config_with(camera_id="/base/axi/pcie@120000/rp1/i2c@80000/ov64a40@36")
        self.assertEqual(selected_camera_model(second, TwoCameras), "ov64a40")
        unplugged = config_with(camera_id="/base/axi/pcie@120000/rp1/i2c@70000/imx477@1a")
        self.assertIsNone(selected_camera_model(unplugged, TwoCameras))

        class NoCameraLibrary:
            @staticmethod
            def global_camera_info():
                raise RuntimeError("no libcamera here")

        self.assertIsNone(selected_camera_model(first, NoCameraLibrary))


class TuningFolderTests(unittest.TestCase):
    def test_a_pi_4_never_gets_a_pi_5_tuning_file(self) -> None:
        # Both folders are installed on a Raspberry Pi; a file from the other
        # pipeline's folder is refused by the camera.
        files = {"pisp": ["imx477_noir.json", "imx477.json"], "vc4": ["imx477_noir.json", "imx477.json"]}
        folders = TuningFolders(self, pipeline="vc4", files=files)
        config = config_with(model="hq", infrared=True)
        self.assertEqual(resolve_camera_tuning_file(config), folders.path("vc4", "imx477_noir.json"))

        folders = TuningFolders(self, pipeline="pisp", files=files)
        self.assertEqual(resolve_camera_tuning_file(config), folders.path("pisp", "imx477_noir.json"))

    def test_the_pi_model_names_that_use_the_pi_5_pipeline(self) -> None:
        cases = {
            "Raspberry Pi 5 Model B Rev 1.0": "pisp",
            "Raspberry Pi 500 Rev 1.0": "pisp",
            "Raspberry Pi Compute Module 5 Rev 1.0": "pisp",
            "Raspberry Pi 4 Model B Rev 1.4": "vc4",
            "Raspberry Pi Compute Module 4 Rev 1.1": "vc4",
            "Raspberry Pi Zero 2 W Rev 1.0": "vc4",
            "Some Other Board": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model"
            with mock.patch.object(tuning, "PI_MODEL_PATHS", (model_file,)):
                self.assertIsNone(tuning.pi_pipeline())  # no such file: not a Pi
                for text, expected in cases.items():
                    with self.subTest(text=text):
                        model_file.write_bytes(text.encode() + b"\x00")
                        self.assertEqual(tuning.pi_pipeline(), expected)

    def test_a_camera_with_no_noir_file_keeps_its_standard_tuning(self) -> None:
        TuningFolders(self, pipeline="pisp", files={"pisp": ["imx477.json", "imx477_noir.json", "imx296.json"]})
        config = config_with(model="auto", infrared=True)
        info = inspect_camera_tuning_resolution(config, detected_sensor="imx296")
        self.assertIsNone(info["resolved"])  # the camera opens on its own tuning, it is not refused
        self.assertEqual(info["missing"], "imx296_noir.json")
        self.assertIn("no NoIR file (imx296_noir.json)", describe_tuning(config, detected_sensor="imx296"))

    def test_the_health_check_says_when_the_noir_file_is_missing(self) -> None:
        from bumblebox_v2 import doctor

        TuningFolders(self, pipeline="pisp", files={"pisp": ["imx477.json"]})
        config = config_with(model="hq", infrared=True)
        with mock.patch.object(doctor, "connected_camera_model", lambda _config: None):
            result = doctor._camera_tuning_check(config)
        self.assertEqual(result.status, "WARN")
        self.assertIn("imx477_noir.json", result.message)
        self.assertIn("standard tuning is used", result.message)


class DescribeTuningTests(unittest.TestCase):
    def test_in_words(self) -> None:
        self.assertEqual(describe_tuning(config_with(model="hq", infrared=True)), "NoIR tuning (imx477_noir.json)")
        self.assertEqual(describe_tuning(config_with(model="hq", infrared=False)), "standard tuning (imx477.json)")
        self.assertEqual(
            describe_tuning(config_with(model="auto", infrared=True), detected_sensor="imx477"),
            "NoIR tuning (imx477_noir.json)",
        )
        self.assertEqual(
            describe_tuning(config_with(tuning_file="/somewhere/mine.json")), "tuning file mine.json (named in the config)"
        )

    def test_a_guess_from_the_recording_size_is_not_stated_as_fact(self) -> None:
        # With the model on auto and no camera asked, the file is only known
        # once the camera is opened.
        text = describe_tuning(config_with(model="auto", infrared=True))
        self.assertNotIn("imx477", text)
        self.assertIn("NoIR tuning", text)
        self.assertEqual(describe_tuning(config_with(model="auto", infrared=False)), "standard tuning")


class FakeLibcameraManager:
    """Counts how often libcamera's camera manager is started, and with which tuning."""

    started_with: list = []

    def __init__(self) -> None:
        # libcamera reads the tuning file here, when the manager starts.
        FakeLibcameraManager.started_with.append(os.environ.get(TUNING_ENV))
        self.tuning = os.environ.get(TUNING_ENV)


def fake_picamera2():
    """A Picamera2 stand-in that keeps the camera manager the way the real one does."""

    class Manager:
        def __init__(self) -> None:
            self.running = False
            self.cameras: dict = {}
            self._cms = None

        @property
        def cms(self):
            if self._cms is None:
                self._cms = FakeLibcameraManager()
            return self._cms

    class Picamera2:
        _cm = Manager()
        opened: list = []

        @staticmethod
        def global_camera_info():
            Picamera2._cm.cms  # listing the cameras starts the manager
            return [{"Num": 0, "Model": "imx477", "Id": "/base/soc/i2c0mux/i2c@1/imx477@1a"}]

        @staticmethod
        def load_tuning_file(name):
            return {"file": name}

        def __init__(self, tuning=None, **kwargs) -> None:
            if tuning is not None:
                os.environ[TUNING_ENV] = f"/tmp/deleted-after-open-{tuning['file']}"
            else:
                os.environ.pop(TUNING_ENV, None)
            self.tuning_in_use = Picamera2._cm.cms.tuning  # what libcamera loaded
            self.kwargs = kwargs
            Picamera2._cm.cameras[len(Picamera2.opened)] = self
            Picamera2._cm.running = True
            Picamera2.opened.append(self)

        def close(self) -> None:
            Picamera2._cm.cameras = {}
            Picamera2._cm.running = False
            Picamera2._cm._cms = None

    return Picamera2


class OpenCameraTests(unittest.TestCase):
    """A tuning file only counts if libcamera's camera manager starts after it is named."""

    def setUp(self) -> None:
        FakeLibcameraManager.started_with = []
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(TUNING_ENV, None)

    def test_listing_the_cameras_first_no_longer_loses_the_tuning_file(self) -> None:
        Picamera2 = fake_picamera2()
        Picamera2.global_camera_info()  # what BumbleBox does before every open

        # The old way: the manager from the listing is reused and the file is ignored.
        ignored = Picamera2(tuning=Picamera2.load_tuning_file("imx477_noir.json"))
        self.assertIsNone(ignored.tuning_in_use)
        ignored.close()
        os.environ.pop(TUNING_ENV, None)

        Picamera2.global_camera_info()
        camera = open_picamera2(Picamera2, "imx477_noir.json", {"camera_num": 1})
        self.assertEqual(camera.tuning_in_use, "/tmp/deleted-after-open-imx477_noir.json")
        self.assertEqual(camera.kwargs, {"camera_num": 1})

    def test_the_name_of_the_deleted_temporary_file_is_not_left_behind(self) -> None:
        Picamera2 = fake_picamera2()
        camera = open_picamera2(Picamera2, "imx477_noir.json")
        self.assertNotIn(TUNING_ENV, os.environ)
        camera.close()
        # The next listing starts libcamera again; it must not be handed a file that is gone.
        Picamera2.global_camera_info()
        self.assertIsNone(FakeLibcameraManager.started_with[-1])

    def test_switching_noir_off_and_on_again_takes_effect_each_time(self) -> None:
        Picamera2 = fake_picamera2()
        seen = []
        for name in ("imx477_noir.json", None, "imx477_noir.json"):
            Picamera2.global_camera_info()
            camera = open_picamera2(Picamera2, name)
            seen.append(camera.tuning_in_use)
            camera.close()
        self.assertEqual(seen, ["/tmp/deleted-after-open-imx477_noir.json", None, "/tmp/deleted-after-open-imx477_noir.json"])

    def test_the_manager_is_left_alone_while_a_camera_is_open(self) -> None:
        Picamera2 = fake_picamera2()
        first = open_picamera2(Picamera2, None)
        manager = Picamera2._cm._cms
        self.assertFalse(release_camera_manager(Picamera2))
        self.assertIs(Picamera2._cm._cms, manager)
        first.close()

    def test_a_camera_library_without_that_manager_still_opens(self) -> None:
        class OlderPicamera2:
            @staticmethod
            def load_tuning_file(name):
                return {"file": name}

            def __init__(self, tuning=None, **kwargs) -> None:
                self.tuning = tuning

        self.assertFalse(release_camera_manager(OlderPicamera2))
        self.assertEqual(open_picamera2(OlderPicamera2, "imx477.json").tuning, {"file": "imx477.json"})
        self.assertIsNone(open_picamera2(OlderPicamera2, None).tuning)

    def test_a_failed_open_also_clears_the_variable(self) -> None:
        attempts = []

        class Failing:
            @staticmethod
            def load_tuning_file(name):
                return {"file": name}

            def __init__(self, tuning=None, **kwargs) -> None:
                attempts.append(tuning)
                os.environ[TUNING_ENV] = "/tmp/gone.json"
                raise RuntimeError("camera busy" if tuning else "still busy")

        with self.assertRaises(RuntimeError) as raised:
            open_picamera2(Failing, "imx477_noir.json")
        self.assertEqual(str(raised.exception), "camera busy")  # the first error, not the second
        self.assertEqual(attempts, [{"file": "imx477_noir.json"}, None])
        self.assertNotIn(TUNING_ENV, os.environ)

    def test_a_tuning_file_the_camera_refuses_does_not_cost_the_recording(self) -> None:
        class RefusesTuning:
            @staticmethod
            def load_tuning_file(name):
                return {"file": name}

            def __init__(self, tuning=None, **kwargs) -> None:
                if tuning is not None:
                    raise RuntimeError("Failed to load a suitable IPA library")
                self.tuning = tuning

        notes: list = []
        with mock.patch("builtins.print") as printed:
            camera = open_picamera2(RefusesTuning, "imx477_noir.json", notes=notes)
        self.assertIsNone(camera.tuning)
        self.assertEqual(len(notes), 1)
        self.assertIn("could not be opened with tuning file 'imx477_noir.json'", notes[0])
        self.assertIn("standard tuning instead", notes[0])
        self.assertIn("[capture]", printed.call_args.args[0])

    def test_a_missing_file_named_by_hand_falls_back_the_same_way(self) -> None:
        class NoSuchFile:
            @staticmethod
            def load_tuning_file(name):
                raise RuntimeError("Tuning file not found")

            def __init__(self, tuning=None, **kwargs) -> None:
                self.tuning = tuning

        notes: list = []
        with mock.patch("builtins.print"):
            camera = open_picamera2(NoSuchFile, "/home/pi/mine.json", notes=notes)
        self.assertIsNone(camera.tuning)
        self.assertIn("Tuning file not found", notes[0])



class CameraSessionTests(unittest.TestCase):
    """The recorder's own camera session, with a stand-in for the camera library."""

    def session(self, config, **kwargs):
        from bumblebox_v2.run_engine import _PicameraCaptureSession

        Picamera2 = mock.MagicMock(name="Picamera2")
        Picamera2.global_camera_info.return_value = [
            {"Num": 0, "Model": "imx477", "Id": "/base/soc/i2c0mux/i2c@1/imx477@1a"}
        ]
        Picamera2.load_tuning_file.side_effect = lambda name: {"file": name}
        modules = {
            "picamera2": SimpleNamespace(Picamera2=Picamera2),
            "libcamera": SimpleNamespace(controls=mock.MagicMock(name="controls")),
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ):
            session = _PicameraCaptureSession(config, **kwargs)
        return session, Picamera2

    def test_a_recording_at_any_size_with_the_model_on_auto_gets_the_noir_file(self) -> None:
        config = config_with(model="auto", infrared=True, width=2028, height=1520)
        session, Picamera2 = self.session(config)
        self.assertEqual(session.detected_sensor, "imx477")
        self.assertEqual(session.resolved_tuning_file, "imx477_noir.json")
        Picamera2.load_tuning_file.assert_called_once_with("imx477_noir.json")
        self.assertEqual(Picamera2.call_args.kwargs["tuning"], {"file": "imx477_noir.json"})

    def test_infrared_off_asks_for_the_standard_file(self) -> None:
        session, Picamera2 = self.session(config_with(model="hq_noir", infrared=False))
        self.assertEqual(session.resolved_tuning_file, "imx477.json")
        Picamera2.load_tuning_file.assert_called_once_with("imx477.json")

    def test_a_session_that_fell_back_reports_standard_tuning(self) -> None:
        from bumblebox_v2.run_engine import _PicameraCaptureSession

        fake = mock.MagicMock(name="Picamera2")
        fake.global_camera_info.return_value = [{"Num": 0, "Model": "imx477", "Id": "x"}]
        fake.load_tuning_file.side_effect = RuntimeError("Tuning file not found")
        modules = {
            "picamera2": SimpleNamespace(Picamera2=fake),
            "libcamera": SimpleNamespace(controls=mock.MagicMock(name="controls")),
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ), mock.patch("builtins.print"):
            session = _PicameraCaptureSession(config_with(model="hq", infrared=True))
        self.assertIsNone(session.resolved_tuning_file)
        self.assertEqual(len(session.tuning_notes), 1)
        self.assertNotIn("tuning", fake.call_args.kwargs)

    def test_the_monitors_choice_is_separate_from_the_recordings(self) -> None:
        recording = config_with(model="auto", infrared=True)
        preview = deepcopy(recording)
        preview["camera"].update(width=1920, height=1440)
        monitor_choice = deepcopy(recording)
        monitor_choice["camera"]["infrared"] = False
        session, _ = self.session(preview, tuning_config=monitor_choice)
        self.assertEqual(session.resolved_tuning_file, "imx477.json")
        self.assertEqual((session.width, session.height), (1920, 1440))
        self.assertIs(recording["camera"]["infrared"], True)


if __name__ == "__main__":
    unittest.main()
