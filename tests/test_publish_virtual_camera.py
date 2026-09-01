from __future__ import annotations

import importlib.util
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "arch-linux/bin/publish_virtual_camera.py"
SPEC = importlib.util.spec_from_file_location("publish_virtual_camera", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
publish_virtual_camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_virtual_camera)


def test_wait_for_capture_ignores_output_only_transition(monkeypatch) -> None:
    samples = iter(
        [
            ("uvcvideo", "Xiaomi Cam", "usb-0000:02:00.0-4", 0x05200002),
            ("uvcvideo", "Xiaomi Cam", "usb-0000:02:00.0-4", 0x05200001),
        ]
    )
    monkeypatch.setattr(
        publish_virtual_camera, "query_capabilities", lambda _device: next(samples)
    )
    times = iter([0.0, 0.0, 0.1])

    result = publish_virtual_camera.wait_for_capture(
        Path("/dev/video42"),
        timeout=1.0,
        interval=0.01,
        clock=lambda: next(times),
        sleeper=lambda _delay: None,
    )

    assert result[3] & publish_virtual_camera.V4L2_CAP_VIDEO_CAPTURE


def test_wait_for_capture_rejects_identity_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        publish_virtual_camera,
        "query_capabilities",
        lambda _device: ("uvcvideo", "USB Camera", "usb-0000:02:00.0-4", 0x1),
    )

    with pytest.raises(publish_virtual_camera.PublishError, match="mismatched camera"):
        publish_virtual_camera.wait_for_capture(Path("/dev/video42"), timeout=1.0)


def test_scoped_change_requires_xiaomi_name(tmp_path: Path) -> None:
    device = tmp_path / "video42-device"
    device.touch()
    sysfs = tmp_path / "sysfs"
    node = sysfs / device.name
    node.mkdir(parents=True)
    (node / "name").write_text("Wrong camera\n", encoding="utf-8")
    (node / "uevent").write_text("", encoding="ascii")

    with pytest.raises(publish_virtual_camera.PublishError, match="sysfs name"):
        publish_virtual_camera.trigger_scoped_change(device, sysfs_root=sysfs)


def test_scoped_change_writes_single_change_event(tmp_path: Path) -> None:
    device = tmp_path / "video42"
    device.touch()
    sysfs = tmp_path / "sysfs"
    node = sysfs / "video42"
    node.mkdir(parents=True)
    (node / "name").write_text("Xiaomi Cam\n", encoding="utf-8")
    (node / "uevent").write_text("", encoding="ascii")

    path = publish_virtual_camera.trigger_scoped_change(device, sysfs_root=sysfs)

    assert path.read_text(encoding="ascii") == "change\n"


def test_resolver_refuses_non_video42_character_device(monkeypatch, tmp_path: Path) -> None:
    device = tmp_path / "video2"
    device.touch()
    monkeypatch.setattr(Path, "stat", lambda _self: type("S", (), {"st_mode": stat.S_IFCHR})())

    with pytest.raises(publish_virtual_camera.PublishError, match="expected /dev/video42"):
        publish_virtual_camera.resolve_expected_device(device)
