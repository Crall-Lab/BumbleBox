"""Picture settings applied inside the camera: sensor gain, white balance, contrast, noise reduction."""

from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from bumblebox_v2 import camera_setup, image_controls
from bumblebox_v2.camera_profiles import CAMERA_PROFILES, apply_camera_profile
from bumblebox_v2.config import ConfigError, load_defaults, validate_config
from bumblebox_v2.live_monitor import capture_frame_and_metadata


class FakeCamera:
    """Accepts controls the way Picamera2 does: one it does not offer raises at once."""

    def __init__(self, offered=None) -> None:
        self.offered = offered
        self.sent: dict = {}

    def set_controls(self, controls: dict) -> None:
        for name, value in controls.items():
            if self.offered is not None and name not in self.offered:
                raise RuntimeError(f"Control {name} is not advertised by libcamera")
            self.sent[name] = value


LIBCAMERA = SimpleNamespace(
    draft=SimpleNamespace(
        NoiseReductionModeEnum=SimpleNamespace(Off="nr-off", Minimal="nr-minimal", Fast="nr-fast", HighQuality="nr-hq")
    )
)


def _config(**camera):
    config = load_defaults()
    config["camera"].update(camera)
    return config


class DefaultsTests(unittest.TestCase):
    def test_a_new_setup_asks_only_for_high_quality_noise_reduction(self) -> None:
        config = load_defaults()
        validate_config(config)
        self.assertEqual(config["camera"]["analogue_gain"], "auto")
        self.assertEqual(config["camera"]["colour_gains"], "auto")
        self.assertEqual(config["camera"]["contrast"], 1.0)
        self.assertEqual(config["camera"]["noise_reduction"], image_controls.DEFAULT_NOISE_REDUCTION)
        self.assertEqual(image_controls.DEFAULT_NOISE_REDUCTION, "HighQuality")
        self.assertEqual(
            image_controls.requested_controls(config),
            [("NoiseReductionMode", "HighQuality", "camera.noise_reduction")],
        )
        camera = FakeCamera()
        self.assertEqual(image_controls.apply_image_controls(config, camera, LIBCAMERA), ["NoiseReductionMode"])
        self.assertEqual(camera.sent, {"NoiseReductionMode": "nr-hq"})
        self.assertEqual(
            image_controls.describe_settings(config),
            "sensor gain automatic; white balance automatic; noise reduction high quality",
        )

    def test_a_camera_without_noise_reduction_still_records_on_a_new_setup(self) -> None:
        camera, notes = FakeCamera(offered={"ExposureTime"}), []
        self.assertEqual(image_controls.apply_image_controls(load_defaults(), camera, LIBCAMERA, notes), [])
        self.assertEqual(camera.sent, {})
        self.assertEqual(len(notes), 1)
        self.assertIn("did not accept NoiseReductionMode (camera.noise_reduction)", notes[0])

    def test_a_box_set_up_before_the_default_changed_is_left_alone(self) -> None:
        # Saved configs name every setting, so an existing box still says "Auto".
        config = _config(noise_reduction="Auto")
        validate_config(config)
        self.assertEqual(image_controls.requested_controls(config), [])
        camera = FakeCamera()
        self.assertEqual(image_controls.apply_image_controls(config, camera, LIBCAMERA), [])
        self.assertEqual(camera.sent, {})
        self.assertEqual(
            image_controls.describe_settings(config),
            "sensor gain automatic; white balance automatic; noise reduction left to the camera",
        )

    def test_a_config_from_before_these_settings_existed_still_works(self) -> None:
        config = _config(noise_reduction="Auto")
        for key in ("analogue_gain", "colour_gains", "contrast"):
            del config["camera"][key]
        validate_config(config)
        self.assertEqual(image_controls.requested_controls(config), [])

    def test_a_dictionary_without_the_setting_sends_nothing(self) -> None:
        self.assertEqual(image_controls.noise_reduction({"camera": {}}), "Auto")
        self.assertEqual(image_controls.requested_controls({"camera": {}}), [])


class CameraProfileTests(unittest.TestCase):
    def test_no_profile_overwrites_a_picture_setting(self) -> None:
        picture_keys = {"noise_reduction", "analogue_gain", "colour_gains", "contrast"}
        for name, profile in CAMERA_PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(picture_keys & set(profile.camera_values), set())

    def test_the_noise_reduction_choice_is_kept_under_a_named_profile(self) -> None:
        for name in CAMERA_PROFILES:
            for mode in ("Auto", "Fast", "HighQuality"):
                with self.subTest(profile=name, mode=mode):
                    config = _config(profile=name, noise_reduction=mode)
                    self.assertEqual(apply_camera_profile(config)["camera"]["noise_reduction"], mode)


class FixedSettingsTests(unittest.TestCase):
    def test_fixed_values_become_camera_controls(self) -> None:
        config = _config(analogue_gain=4, colour_gains=[1.9, 1.45], contrast=1.15, noise_reduction="HighQuality")
        validate_config(config)
        camera, notes = FakeCamera(), []
        applied = image_controls.apply_image_controls(config, camera, LIBCAMERA, notes)
        self.assertEqual(applied, ["AnalogueGain", "AwbEnable", "ColourGains", "Contrast", "NoiseReductionMode"])
        self.assertEqual(
            camera.sent,
            {
                "AnalogueGain": 4.0,
                "AwbEnable": False,  # fixed gains mean automatic white balance is off
                "ColourGains": (1.9, 1.45),
                "Contrast": 1.15,
                "NoiseReductionMode": "nr-hq",
            },
        )
        self.assertEqual(notes, [])
        self.assertEqual(
            image_controls.describe_settings(config),
            "sensor gain fixed at 4; white balance fixed (red 1.9, blue 1.45); contrast 1.15; "
            "noise reduction high quality",
        )

    def test_each_setting_is_independent(self) -> None:
        only_gain = image_controls.requested_controls(_config(analogue_gain=2.5, noise_reduction="Auto"))
        self.assertEqual([name for name, _value, _key in only_gain], ["AnalogueGain"])
        only_balance = image_controls.requested_controls(_config(colour_gains=(2.0, 1.0), noise_reduction="Auto"))
        self.assertEqual([name for name, _value, _key in only_balance], ["AwbEnable", "ColourGains"])
        only_noise = image_controls.requested_controls(_config(noise_reduction="Fast"))
        self.assertEqual(only_noise, [("NoiseReductionMode", "Fast", "camera.noise_reduction")])
        self.assertEqual(image_controls.requested_controls(_config(contrast=1.0, noise_reduction="auto")), [])

    def test_noise_reduction_is_read_generously(self) -> None:
        for written, mode in (
            ("HighQuality", "HighQuality"),
            ("high_quality", "HighQuality"),
            ("High Quality", "HighQuality"),
            ("fast", "Fast"),
            ("OFF", "Off"),
            ("minimal", "Minimal"),
            ("Auto", "Auto"),
            (None, "Auto"),
        ):
            self.assertEqual(image_controls.noise_reduction(_config(noise_reduction=written)), mode, written)

    def test_a_control_the_camera_lacks_is_noted_and_the_rest_still_apply(self) -> None:
        config = _config(analogue_gain=4, colour_gains=[1.9, 1.45], contrast=1.2, noise_reduction="Auto")
        camera, notes = FakeCamera(offered={"AnalogueGain", "AwbEnable", "ColourGains"}), []
        applied = image_controls.apply_image_controls(config, camera, LIBCAMERA, notes)
        self.assertEqual(applied, ["AnalogueGain", "AwbEnable", "ColourGains"])
        self.assertEqual(len(notes), 1)
        self.assertIn("did not accept Contrast (camera.contrast)", notes[0])

    def test_preview_and_tracking_test_use_the_same_settings(self) -> None:
        config = _config(analogue_gain=3.0, colour_gains=[2.0, 1.2], shutter_us=4000)
        camera, notes = FakeCamera(), []
        # The preview looks the noise reduction mode up in the camera library itself.
        with mock.patch.dict(sys.modules, {"libcamera": SimpleNamespace(controls=LIBCAMERA)}):
            camera_setup._apply_camera_controls(config, camera, notes)
        self.assertEqual(camera.sent["ExposureTime"], 4000)
        self.assertEqual(camera.sent["AnalogueGain"], 3.0)
        self.assertEqual(camera.sent["ColourGains"], (2.0, 1.2))
        self.assertEqual(camera.sent["NoiseReductionMode"], "nr-hq")
        self.assertEqual(notes, [])


class ValidationTests(unittest.TestCase):
    def test_unusable_values_are_refused_in_words(self) -> None:
        cases = (
            ({"analogue_gain": 0.5}, "camera.analogue_gain must be between 1 and 64"),
            ({"analogue_gain": "high"}, "camera.analogue_gain must be a number"),
            ({"analogue_gain": True}, "camera.analogue_gain must be a number"),
            ({"colour_gains": [1.0]}, 'camera.colour_gains must be "auto" or two numbers [red, blue]'),
            ({"colour_gains": [1.0, 99]}, "camera.colour_gains blue must be between 0.1 and 32"),
            ({"contrast": -1}, "camera.contrast must be between 0 and 32"),
            ({"noise_reduction": "Extreme"}, "camera.noise_reduction must be one of Auto, Off, Minimal, Fast, HighQuality"),
        )
        for change, message in cases:
            with self.subTest(change=change):
                with self.assertRaises(ConfigError) as raised:
                    validate_config(_config(**change))
                self.assertIn(message, str(raised.exception))

    def test_auto_can_be_written_several_ways(self) -> None:
        for value in ("auto", "Auto", " AUTO ", None, ""):
            config = _config(analogue_gain=value, colour_gains=value)
            validate_config(config)
            self.assertIsNone(image_controls.analogue_gain(config))
            self.assertIsNone(image_controls.colour_gains(config))


class ReadoutTests(unittest.TestCase):
    def test_readout_gives_the_numbers_to_copy(self) -> None:
        text = image_controls.camera_readout({"AnalogueGain": 3.8123, "ColourGains": (1.92, 1.447), "Lux": 80})
        self.assertEqual(text, "sensor gain 3.81  |  white balance red 1.92, blue 1.45")
        self.assertEqual(image_controls.camera_readout({"AnalogueGain": 2}), "sensor gain 2.00")
        self.assertEqual(image_controls.camera_readout({}), "")
        self.assertEqual(image_controls.camera_readout(None), "")

    def test_a_frame_comes_with_what_the_camera_reports(self) -> None:
        released = []

        class Request:
            def make_array(self, name):
                return f"frame from {name}"

            def get_metadata(self):
                return {"AnalogueGain": 2.0}

            def release(self):
                released.append(True)

        modern = SimpleNamespace(capture_request=lambda: Request())
        self.assertEqual(capture_frame_and_metadata(modern), ("frame from main", {"AnalogueGain": 2.0}))
        self.assertEqual(released, [True])  # the camera's buffer is always handed back
        plain = SimpleNamespace(capture_array=lambda: "plain frame")
        self.assertEqual(capture_frame_and_metadata(plain), ("plain frame", None))


if __name__ == "__main__":
    unittest.main()
