from __future__ import annotations

import unittest

from bumblebox_v2.config import ConfigError, load_defaults, validate_config
from bumblebox_v2.hardware_profiles import apply_hardware_profile


class HardwareProfileTests(unittest.TestCase):
    def test_defaults_are_valid_and_custom(self) -> None:
        config = load_defaults()

        validate_config(config)

        self.assertEqual(config["setup"]["hardware_profile"], "custom")
        self.assertFalse(config["realsense"]["enabled"])

    def test_multimodal_profile_enables_optional_cameras(self) -> None:
        config = apply_hardware_profile(load_defaults(), "multimodal")

        self.assertTrue(config["thermal"]["enabled"])
        self.assertTrue(config["realsense"]["enabled"])

    def test_rgb_only_profile_disables_optional_cameras(self) -> None:
        config = load_defaults()
        config["thermal"]["enabled"] = True
        config["realsense"]["enabled"] = True

        updated = apply_hardware_profile(config, "rgb_only")

        self.assertFalse(updated["thermal"]["enabled"])
        self.assertFalse(updated["realsense"]["enabled"])

    def test_custom_profile_preserves_explicit_device_choices(self) -> None:
        config = load_defaults()
        config["thermal"]["enabled"] = True
        config["realsense"]["enabled"] = False

        updated = apply_hardware_profile(config, "custom")

        self.assertTrue(updated["thermal"]["enabled"])
        self.assertFalse(updated["realsense"]["enabled"])

    def test_invalid_realsense_alignment_is_rejected(self) -> None:
        config = load_defaults()
        config["realsense"]["align_to"] = "infrared"

        with self.assertRaises(ConfigError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
