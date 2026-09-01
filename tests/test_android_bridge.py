from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import android_bridge  # noqa: E402


def completed(command, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_parse_adb_devices_ignores_daemon_noise():
    output = """* daemon started successfully *
List of devices attached
b83607a5 device usb:3-2 product:apollon model:M2007J3SY device:apollo
emulator-5554 offline transport_id:8
"""

    devices = android_bridge.parse_adb_devices(output)

    assert devices[0]["serial"] == "b83607a5"
    assert devices[0]["model"] == "M2007J3SY"
    assert devices[1]["state"] == "offline"


def test_collect_status_reports_merged_app_module_camera_and_latest_telemetry(monkeypatch):
    monkeypatch.setattr(android_bridge, "_adb_path", lambda: "/usr/bin/adb")
    monkeypatch.setattr(
        android_bridge,
        "_select_device",
        lambda _adb, _preferred, _host="": (
            {"serial": "b83607a5", "state": "device", "model": "M2007J3SY"},
            [],
        ),
    )

    detail = """__MODEL__
M2007J3SY
__IP__
wlan0 UP 192.168.1.12/24
__APP__
package:/data/app/dev.vcam.app/base.apk
__VERSION__
1.4-phone-source-virtual-microphone
__SERVICE__
ServiceRecord{dev.vcam.app/.CameraBridgeService}
__METRICS__
D/VCamBridge(29561): telemetry
D/VCamBridge(29561): up=624s lens=back rot=auto(90) stab=video exp=-1 ae=auto awb=auto
D/VCamBridge(29561): cam : id=0 state=streaming(0) cap=17842(29.9fps) int=33.2ms jit=0.14ms
D/VCamBridge(29561): enc : c2.qti.avc.encoder started 1280x720@30 frames=17841(29.9fps) tx=743.8MB
D/VCamBridge(29561): tcp : :10020 listen=yes binds=1 conns=5 client=yes up=258s tx=321.8MB 1226KB/s wrErr=0
D/VCamBridge(29561): telemetry
D/VCamBridge(29561): up=625s lens=front rot=auto(270) stab=video zoom=1.25x/10.00x exp=+0 ae=auto awb=lock
D/VCamBridge(29561): cam : id=1 state=streaming(1) cap=17873(30.9fps) int=33.1ms jit=0.12ms
D/VCamBridge(29561): enc : c2.qti.avc.encoder started 1280x720@30 frames=17872(30.0fps) tx=745.1MB
D/VCamBridge(29561): gl  : egl=ok frames=17872 swapErr=0 err=-
D/VCamBridge(29561): tcp : :10020 listen=yes binds=1 conns=6 client=yes up=259s tx=323.1MB 1229KB/s wrErr=1
"""
    root = """module_installed=1
module_enabled=1
module_version=v0.4.6
sender=1
capture=1
return=1
output=1
provider=1
front_redirect_package=1
camera_node=1
output_config_present=1
output_enabled=1
output_mirror=1
output_rotation=90
output_revision=7
effective_enabled=1
effective_mirror=1
effective_rotation=90
effective_revision=7
effective_source=processed
effective_worker_alive=1
"""

    detail_commands: list[str] = []

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        if args and args[0] == "shell" and len(args) > 1 and args[1].startswith("su -c"):
            return completed(args, root)
        if args and args[0] == "shell" and len(args) > 1:
            detail_commands.append(args[1])
        return completed(args, detail)

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    status = android_bridge.collect_status("b83607a5", "192.168.1.12", "120")

    assert status["available"] is True
    assert status["host"] == "192.168.1.12"
    assert status["app_version"] == "1.4-phone-source-virtual-microphone"
    assert status["bridge_running"] is True
    assert status["capture_metrics"]["lens_facing"] == "front"
    assert status["capture_metrics"]["camera_id"] == "1"
    assert status["capture_metrics"]["camera_state"] == "streaming(1)"
    assert status["capture_metrics"]["captured_fps"] == 30.9
    assert status["capture_metrics"]["encoded_fps"] == 30.0
    assert status["capture_metrics"]["encoder"] == "c2.qti.avc.encoder"
    assert status["capture_metrics"]["tcp_client_connected"] is True
    assert status["capture_metrics"]["tcp_connections"] == 6
    assert status["capture_metrics"]["tcp_write_errors"] == 1
    assert status["capture_metrics"]["rotation"] == "auto"
    assert status["capture_metrics"]["zoom_percent"] == 125
    assert status["capture_metrics"]["maximum_zoom_ratio"] == 10.0
    assert status["capture_metrics"]["effective_rotation_degrees"] == 270
    assert status["capture_metrics"]["rendered_frames"] == 17872
    assert status["network_sender_running"] is True
    assert status["capture_fanout_running"] is True
    assert status["output_selector_running"] is True
    assert status["camera_published"] is True
    assert status["front_redirect"] == {
        "package_installed": True,
        "active": None,
        "processed_camera_id": "120",
    }
    assert status["output_control"] == {
        "supported": True,
        "enabled": True,
        "mirror": True,
        "rotation": 90,
        "revision": 7,
        "persisted": True,
        "effective_source": "processed",
        "effective_worker_alive": True,
        "effective_revision": 7,
        "applied": True,
    }
    assert len(detail_commands) == 1
    assert "pm path dev.vcam.app" in detail_commands[0]
    assert "dumpsys activity services dev.vcam.app" in detail_commands[0]
    assert "VCamBridge:D" in detail_commands[0]


def test_capture_metrics_parser_preserves_legacy_bridge_format():
    output = (
        "08-11 17:39:24.256 I VCamBridge: encoded=21000 fps=30.1 "
        "bitrate=10.01Mbps captureInterval=33.19ms jitter=0.14ms dropped~0 "
        "exposure=8.333ms ISO=160 exposureJitter=0.0042EV "
        "awbJitter=0.00031 ae=2 awb=2 rotation=auto "
        "effectiveRotation=90 rendered=21002 textureRotation=90 "
        "shaderRotation=0"
    )

    metrics = android_bridge.parse_capture_metrics(output)

    assert metrics["encoded_frames"] == 21000
    assert metrics["encoded_fps"] == 30.1
    assert metrics["exposure_jitter_ev"] == 0.0042
    assert metrics["awb_gain_jitter"] == 0.00031
    assert metrics["effective_rotation_degrees"] == 90
    assert metrics["texture_rotation_degrees"] == 90
    assert metrics["shader_rotation_degrees"] == 0


def test_restart_stops_then_starts_camera_service_and_opens_activity(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(android_bridge, "_require_device", lambda _serial, _host="": ("adb", "phone"))
    monkeypatch.setattr(android_bridge.time, "sleep", lambda _seconds: None)

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        calls.append(tuple(args))
        return completed(args, "Status: ok")

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    android_bridge.control("restart", "phone")

    assert calls[0] == ("shell", "am", "force-stop", android_bridge.PACKAGE)
    assert calls[1][0] == "shell"
    assert calls[1][1].startswith("su -c ")
    assert "am start-foreground-service --user 0" in calls[1][1]
    assert android_bridge.SERVICE in calls[1][1]
    assert "dev.vcam.app.START" in calls[1][1]
    assert calls[2] == (
        "shell", "am", "start", "-W", "-n", android_bridge.ACTIVITY
    )


def test_start_fails_if_authoritative_camera_service_cannot_start(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        calls.append(tuple(args))
        return completed(args, stderr="Error: service denied", returncode=1)

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    try:
        android_bridge.control("start", "phone")
    except android_bridge.AndroidBridgeError as error:
        assert "service denied" in str(error)
    else:
        raise AssertionError("failed camera foreground service was accepted")

    assert len(calls) == 1
    assert android_bridge.SERVICE in calls[0][1]


def test_select_device_falls_back_from_usb_serial_to_configured_wireless_host(monkeypatch):
    monkeypatch.setattr(
        android_bridge,
        "_run",
        lambda _command, timeout=4.0: completed(
            _command,
            """List of devices attached
emulator-5554 device model:sdk_phone64_x86_64
192.168.1.12:41505 device model:M2007J3SY
""",
        ),
    )

    selected, _devices = android_bridge._select_device(
        "/usr/bin/adb", "b83607a5", "192.168.1.12"
    )

    assert selected is not None
    assert selected["serial"] == "192.168.1.12:41505"


def test_camera_configuration_is_allowlisted_and_sent_to_existing_owner(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        calls.append(tuple(args))
        if args[:4] == ("shell", "dumpsys", "activity", "services"):
            return completed(args, "ServiceRecord{dev.vcam.app/.CameraBridgeService}")
        return completed(args, "Starting service: Intent")

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    android_bridge.configure_camera(
        {
            "lens_facing": "front",
            "exposure_compensation": -2,
            "ae_lock": True,
            "awb_lock": True,
            "stabilization": "video",
            "rotation": "auto",
            "zoom_percent": 125,
        },
        "phone",
    )

    assert len(calls) == 2
    assert calls[1][0] == "shell"
    command = calls[1][1]
    assert "dev.vcam.app.CONFIGURE" in command
    assert "--ez persist false" in command
    assert "--es lens_facing front" in command
    assert "--ei exposure_compensation -2" in command
    assert "--ez ae_lock true" in command
    assert "--ez awb_lock true" in command
    assert "--es stabilization video" in command
    assert "--es rotation auto" in command
    assert "--ei zoom_percent 125" in command


def test_camera_configuration_rejects_invalid_rotation():
    try:
        android_bridge.configure_camera({"rotation": "45"}, "phone")
    except android_bridge.AndroidBridgeError as error:
        assert str(error) == "rotation must be auto, 0, 90, 180, or 270"
    else:
        raise AssertionError("invalid rotation was accepted")


def test_camera_configuration_persists_only_when_requested(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        calls.append(tuple(args))
        if args[:4] == ("shell", "dumpsys", "activity", "services"):
            return completed(args, "ServiceRecord{dev.vcam.app/.CameraBridgeService}")
        return completed(args, "Starting service: Intent")

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    android_bridge.configure_camera(
        {"awb_lock": True}, "phone", persist=True
    )

    assert "--ez persist true" in calls[1][1]


def test_camera_configuration_never_starts_a_missing_owner(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        calls.append(tuple(args))
        return completed(args, "(nothing)")

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    with pytest.raises(android_bridge.AndroidBridgeError, match="not running"):
        android_bridge.configure_camera({"lens_facing": "front"}, "phone")

    assert len(calls) == 1
    assert "CONFIGURE" not in " ".join(calls[0])
