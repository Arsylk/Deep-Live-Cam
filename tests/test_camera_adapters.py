from __future__ import annotations

from pathlib import Path
import sys

import pytest


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

from camera_adapters import (  # noqa: E402
    CameraAdapterError,
    apply_arch_controls,
    camera_schema,
    normalize_controls,
)


def test_each_stack_advertises_only_its_own_controls():
    arch = camera_schema("arch-v4l2")
    android = camera_schema("android-camera2")
    generic = camera_schema("generic-srt")

    assert arch["manager_opens_camera"] is False
    assert {item["key"] for item in arch["controls"]} >= {
        "profile",
        "capture_size",
        "brightness",
        "gamma",
        "exposure_dynamic_framerate",
    }
    profile = arch["profiles"][0]
    assert profile["id"] == "natural-indoor"
    assert profile["measured"] is True
    assert profile["measurement_revision"] == "v4"
    assert profile["capture"] == {
        "input_format": "mjpeg",
        "width": 1280,
        "height": 720,
        "fps": 30,
    }
    assert profile["controls"]["brightness"] == -12
    assert profile["controls"]["exposure_dynamic_framerate"] is False
    assert {item["key"] for item in android["controls"]} == {
        "lens_facing",
        "rotation",
        "zoom_percent",
        "exposure_compensation",
        "ae_lock",
        "awb_lock",
        "stabilization",
    }
    assert generic["controls"] == []


def test_controls_are_allowlisted_and_bounded():
    assert normalize_controls(
        "android-camera2", {"lens_facing": "front", "ae_lock": True}
    ) == {"lens_facing": "front", "ae_lock": True}
    with pytest.raises(CameraAdapterError, match="does not support brightness"):
        normalize_controls("android-camera2", {"brightness": 3})
    with pytest.raises(CameraAdapterError, match="between -12 and 12"):
        normalize_controls("android-camera2", {"exposure_compensation": 99})
    assert normalize_controls(
        "android-camera2", {"awb_lock": True, "stabilization": "video"}
    ) == {"awb_lock": True, "stabilization": "video"}
    assert normalize_controls(
        "android-camera2", {"rotation": "auto"}
    ) == {"rotation": "auto"}
    assert normalize_controls(
        "android-camera2", {"zoom_percent": 125}
    ) == {"zoom_percent": 125}
    with pytest.raises(CameraAdapterError, match="between 100 and 300"):
        normalize_controls("android-camera2", {"zoom_percent": 99})
    with pytest.raises(CameraAdapterError, match="rotation must be one of"):
        normalize_controls("android-camera2", {"rotation": "45"})
    assert normalize_controls(
        "arch-v4l2", {"capture_size": "1920x1080"}
    ) == {"capture_size": "1920x1080"}
    assert normalize_controls(
        "arch-v4l2", {"exposure_dynamic_framerate": False}
    ) == {"exposure_dynamic_framerate": False}
    with pytest.raises(CameraAdapterError, match="must be true or false"):
        normalize_controls(
            "arch-v4l2", {"exposure_dynamic_framerate": 0}
        )


def test_arch_live_adapter_omits_manual_values_while_auto_is_enabled(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    class FakeSocket:
        responded = False

        def settimeout(self, _timeout):
            pass

        def connect(self, path):
            captured["path"] = path

        def sendall(self, payload):
            import json

            captured.update(json.loads(payload)["controls"])

        def shutdown(self, _direction):
            pass

        def recv(self, _size):
            if not self.responded:
                self.responded = True
                return b'{"ok":true}'
            return b""

        def close(self):
            pass

    monkeypatch.setattr("camera_adapters.socket.socket", lambda *_args: FakeSocket())

    result = apply_arch_controls(
        {
            "auto_exposure": True,
            "exposure_time_absolute": 157,
            "auto_white_balance": True,
            "white_balance_temperature": 4600,
        },
        tmp_path / "owner.sock",
    )

    assert result["ok"] is True
    assert "exposure_time_absolute" not in captured
    assert "white_balance_temperature" not in captured


def test_arch_named_profile_expands_to_measured_live_controls(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeSocket:
        responded = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendall(self, payload):
            import json

            captured.update(json.loads(payload)["controls"])

        def shutdown(self, _direction):
            pass

        def recv(self, _size):
            if not self.responded:
                self.responded = True
                return b'{"ok":true}'
            return b""

        def close(self):
            pass

    monkeypatch.setattr("camera_adapters.socket.socket", lambda *_args: FakeSocket())

    result = apply_arch_controls(
        {"profile": "natural-indoor"}, tmp_path / "owner.sock"
    )

    assert result["profile"] == "natural-indoor"
    assert captured == {
        "brightness": -12,
        "contrast": 12,
        "saturation": 40,
        "hue": 0,
        "gamma": 95,
        "gain": 0,
        "sharpness": 2,
        "backlight_compensation": 0,
        "power_line_frequency": 1,
        "auto_exposure": True,
        "exposure_dynamic_framerate": False,
        "auto_white_balance": True,
    }
