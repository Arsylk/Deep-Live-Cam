from __future__ import annotations

from pathlib import Path
import json
import sys


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import configure_camera  # noqa: E402
from camera_adapters import CameraAdapterError  # noqa: E402
from camera_profiles import profile_config_values  # noqa: E402


def test_apply_controls_includes_dynamic_framerate_without_manual_auto_values(
    monkeypatch, tmp_path
):
    calls: list[tuple[dict[str, object], Path]] = []
    monkeypatch.setattr(configure_camera, "DEFAULT_STATE_DIR", tmp_path)
    monkeypatch.setattr(
        configure_camera,
        "apply_arch_controls",
        lambda controls, socket_path: (
            calls.append((controls, socket_path))
            or {"ok": True, "controls": controls}
        ),
    )

    configure_camera.apply_controls(profile_config_values("natural-indoor"))

    controls, socket_path = calls[0]
    assert socket_path == tmp_path / "sender-control.sock"
    assert controls["brightness"] == -12
    assert controls["contrast"] == 12
    assert controls["sharpness"] == 2
    assert controls["exposure_dynamic_framerate"] is False
    assert controls["auto_exposure"] is True
    assert controls["auto_white_balance"] is True


def test_named_profile_is_persisted_without_restarting_owner(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "deep-live-cam-arch.conf"
    config_path.write_text(
        "CAMERA_PROFILE=custom\n"
        "CAMERA_INPUT_FORMAT=yuyv422\n"
        "CAMERA_WIDTH=640\n"
        "CAMERA_HEIGHT=480\n"
        "CAMERA_FPS=15\n",
        encoding="utf-8",
    )
    applied: list[dict[str, str]] = []
    monkeypatch.setattr(configure_camera, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(configure_camera.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        configure_camera,
        "apply_controls",
        lambda config: (
            applied.append(config) or {"ok": True, "controls": {"brightness": -12}}
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["configure_camera.py", "--profile", "natural-indoor"]
    )

    assert configure_camera.main() == 0

    saved = configure_camera.load_env_file(config_path)
    assert {
        key: saved[key]
        for key in profile_config_values("natural-indoor")
    } == profile_config_values("natural-indoor")
    assert applied[-1]["CAMERA_EXPOSURE_DYNAMIC_FRAMERATE"] == "0"
    result = json.loads(capsys.readouterr().out)
    assert result["capture_format"] == "staged-next-owner-start"
    assert result["owner_restarted"] is False
    assert result["live_controls_applied"] is True
    source = Path(configure_camera.__file__).read_text(encoding="utf-8")
    assert "systemctl" not in source
    assert "subprocess" not in source


def test_individual_dynamic_framerate_change_marks_profile_custom(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "deep-live-cam-arch.conf"
    config_path.write_text(
        "CAMERA_PROFILE=natural-indoor\n"
        "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE=0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(configure_camera, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(configure_camera.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        configure_camera,
        "apply_controls",
        lambda _config: {"ok": True, "controls": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "configure_camera.py",
            "--exposure-dynamic-framerate",
            "1",
        ],
    )

    assert configure_camera.main() == 0

    saved = configure_camera.load_env_file(config_path)
    assert saved["CAMERA_PROFILE"] == "custom"
    assert saved["CAMERA_EXPOSURE_DYNAMIC_FRAMERATE"] == "1"


def test_save_stays_successfully_staged_when_capture_owner_is_offline(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "deep-live-cam-arch.conf"
    config_path.write_text(
        "CAMERA_PROFILE=custom\nCAMERA_WIDTH=1280\nCAMERA_HEIGHT=720\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(configure_camera, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(configure_camera.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        configure_camera,
        "apply_controls",
        lambda _config: (_ for _ in ()).throw(CameraAdapterError("owner offline")),
    )
    monkeypatch.setattr(
        sys, "argv", ["configure_camera.py", "--capture-size", "1920x1080"]
    )

    assert configure_camera.main() == 0

    result = json.loads(capsys.readouterr().out)
    saved = configure_camera.load_env_file(config_path)
    assert saved["CAMERA_WIDTH"] == "1920"
    assert saved["CAMERA_HEIGHT"] == "1080"
    assert result["persisted"] is True
    assert result["live_controls_applied"] is False
    assert result["capture_format"] == "staged-next-owner-start"
    assert result["owner_restarted"] is False
    assert result["live_detail"] == "owner offline"
