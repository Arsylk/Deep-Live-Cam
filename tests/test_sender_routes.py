from __future__ import annotations

from pathlib import Path
import sys

import pytest


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import sender  # noqa: E402


def build_sender(monkeypatch, *, manager_port: str = "11001"):
    monkeypatch.setenv("WINDOWS_HOST", "192.168.1.35")
    monkeypatch.setenv("PHYSICAL_CAMERA", "/dev/camera-test")
    monkeypatch.setenv("DEVICE_ID", "arch-webcam")
    monkeypatch.setenv("DEVICE_SLOT", "1")
    monkeypatch.setenv("WINDOWS_INPUT_PORT", "10002")
    monkeypatch.setenv("LOCAL_PREVIEW_PORT", "11000")
    monkeypatch.setenv("MANAGER_PREVIEW_PORT", manager_port)
    monkeypatch.setenv("NETWORK_SOURCE_PORT", "11002")
    monkeypatch.setenv("NATIVE_PROCESSOR_SOURCE_PORT", "11005")
    monkeypatch.setattr(sender.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(sender.Sender, "select_encoder", lambda _self, _name: "libx264")
    return sender.Sender()


def test_capture_is_lan_independent_and_has_four_private_outputs(monkeypatch):
    worker = build_sender(monkeypatch)

    command = " ".join(worker.command())
    network = " ".join(worker.network_command())

    assert "127.0.0.1:11000" in command
    assert "127.0.0.1:11001" in command
    assert "127.0.0.1:11002" in command
    assert "127.0.0.1:11005" in command
    assert "192.168.1.35" not in command
    assert "127.0.0.1:11002" in network
    assert "srt://192.168.1.35:10002?" in network
    assert "-c:v copy" in network
    assert "/dev/camera-test" not in network


def test_private_sender_ports_must_be_distinct(monkeypatch):
    with pytest.raises(ValueError, match="must be distinct"):
        build_sender(monkeypatch, manager_port="11000")


def test_auto_modes_remove_inactive_manual_camera_controls(monkeypatch):
    worker = build_sender(monkeypatch)
    worker.camera_control_overrides = {
        "auto_exposure": True,
        "exposure_time_absolute": 157,
        "auto_white_balance": True,
        "white_balance_temperature": 4600,
    }

    controls = worker._camera_controls()

    assert controls["auto_exposure"] == "3"
    assert controls["white_balance_automatic"] == "1"
    assert "exposure_time_absolute" not in controls
    assert "white_balance_temperature" not in controls


def test_natural_indoor_defaults_keep_capture_cadence_stable(monkeypatch):
    for key in (
        "CAMERA_PROFILE",
        "CAMERA_BRIGHTNESS",
        "CAMERA_CONTRAST",
        "CAMERA_SATURATION",
        "CAMERA_GAMMA",
        "CAMERA_SHARPNESS",
        "CAMERA_BACKLIGHT",
        "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE",
    ):
        monkeypatch.delenv(key, raising=False)
    worker = build_sender(monkeypatch)

    controls = worker._camera_controls()

    assert worker.camera_profile == "natural-indoor"
    assert controls == {
        "brightness": "-12",
        "contrast": "12",
        "saturation": "40",
        "hue": "0",
        "gamma": "95",
        "gain": "0",
        "sharpness": "2",
        "backlight_compensation": "0",
        "power_line_frequency": "1",
        "exposure_dynamic_framerate": "0",
        "auto_exposure": "3",
        "white_balance_automatic": "1",
    }
    state = worker.state("test")
    assert state["camera_profile"] == "natural-indoor"
    assert state["camera_controls"] == controls


def test_dynamic_framerate_is_a_boolean_owner_control(monkeypatch):
    worker = build_sender(monkeypatch)

    assert worker._validated_control_request(
        {"controls": {"exposure_dynamic_framerate": False}}
    ) == {"exposure_dynamic_framerate": False}
    with pytest.raises(ValueError, match="must be true or false"):
        worker._validated_control_request(
            {"controls": {"exposure_dynamic_framerate": 0}}
        )

    worker.camera_control_overrides = {"exposure_dynamic_framerate": True}
    assert worker._camera_controls()["exposure_dynamic_framerate"] == "1"


def test_named_profile_rejects_a_mislabeled_environment(monkeypatch):
    monkeypatch.setenv("CAMERA_PROFILE", "natural-indoor")
    monkeypatch.setenv("CAMERA_CONTRAST", "32")

    with pytest.raises(
        ValueError,
        match=r"CAMERA_PROFILE=natural-indoor does not match: CAMERA_CONTRAST=32",
    ):
        build_sender(monkeypatch)
