from __future__ import annotations

from pathlib import Path
import sys


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

from camera_profiles import profile_config_values  # noqa: E402


def test_natural_indoor_persistent_bundle_is_complete():
    assert profile_config_values("natural-indoor") == {
        "CAMERA_PROFILE": "natural-indoor",
        "CAMERA_INPUT_FORMAT": "mjpeg",
        "CAMERA_WIDTH": "1280",
        "CAMERA_HEIGHT": "720",
        "CAMERA_FPS": "30",
        "CAMERA_BRIGHTNESS": "-12",
        "CAMERA_CONTRAST": "12",
        "CAMERA_SATURATION": "40",
        "CAMERA_HUE": "0",
        "CAMERA_GAMMA": "95",
        "CAMERA_GAIN": "0",
        "CAMERA_SHARPNESS": "2",
        "CAMERA_BACKLIGHT": "0",
        "CAMERA_POWER_LINE": "1",
        "CAMERA_AUTO_EXPOSURE": "1",
        "CAMERA_EXPOSURE": "157",
        "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE": "0",
        "CAMERA_AUTO_WHITE_BALANCE": "1",
        "CAMERA_WHITE_BALANCE": "4600",
    }


def test_packaged_config_and_installer_contain_the_measured_profile():
    root = Path(__file__).resolve().parents[1]
    config = (root / "arch-linux/config/deep-live-cam-arch.conf").read_text()
    installer = (root / "arch-linux/install.sh").read_text()

    for key, value in profile_config_values("natural-indoor").items():
        assignment = f"{key}={value}"
        assert assignment in config
        if key == "CAMERA_PROFILE":
            assert '"$camera_profile" == "natural-indoor"' in installer
        else:
            assert assignment in installer
