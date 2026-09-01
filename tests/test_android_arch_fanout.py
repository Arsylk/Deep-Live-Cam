from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "android" / "vcam-module-overlay"
ARCH_CONFIG = ROOT / "arch-linux" / "config" / "deep-live-cam-arch.conf"
LOCAL_UNIT = (
    ROOT
    / "arch-linux"
    / "systemd"
    / "deep-live-cam-phone-processed.service"
)


def _assignments(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def test_one_encoder_read_fans_out_complete_private_transport_copies():
    config = _assignments(OVERLAY / "bridge.conf")
    capture = (OVERLAY / "bridge-capture.sh").read_text(encoding="utf-8")

    private_ports = {
        int(config["ANDROID_RAW_CAMERA_PORT"]),
        int(config["ANDROID_SENDER_FEED_PORT"]),
        int(config["ANDROID_ARCH_SENDER_FEED_PORT"]),
    }
    assert len(private_ports) == 3
    assert capture.count("tcp://127.0.0.1:$ANDROID_ENCODER_PORT") == 1
    assert capture.count("-f tee") == 1
    for variable in (
        "ANDROID_RAW_CAMERA_PORT",
        "ANDROID_SENDER_FEED_PORT",
        "ANDROID_ARCH_SENDER_FEED_PORT",
    ):
        destination = f"udp://127.0.0.1:${variable}?pkt_size=1316"
        assert capture.count(destination) == 1
    assert capture.count("onfail=ignore") == 3


def test_capture_caches_h264_parameter_sets_before_keyframe_reinjection():
    capture = (OVERLAY / "bridge-capture.sh").read_text(encoding="utf-8")
    chain = (
        'extract_extradata=remove=0,setts=pts=N/($VIDEO_FPS*TB):'
        'dts=N/($VIDEO_FPS*TB):duration=1/($VIDEO_FPS*TB),'
        'dump_extra=freq=keyframe'
    )

    assert f'-bsf:v "{chain}"' in capture
    assert chain.index("extract_extradata") < chain.index("setts=")
    assert chain.index("setts=") < chain.index("dump_extra=")


def test_capture_only_hot_update_never_restarts_camera_or_network_workers():
    deployment = (OVERLAY / "deploy-capture-hot.sh").read_text(encoding="utf-8")

    for filter_name in ("extract_extradata", "setts", "dump_extra"):
        assert filter_name in deployment
    assert "android-vcam-capture.pid" in deployment
    assert '"$MODDIR/bridge-capture.sh"' in deployment
    for forbidden in (
        "android-vcam-provider.pid",
        "android-vcam-producer.pid",
        "android-vcam-sender.pid",
        "android-vcam-arch-sender.pid",
        "am force-stop",
        "am stopservice",
        "rmmod",
        "insmod",
    ):
        assert forbidden not in deployment


def test_windows_and_arch_senders_never_share_a_udp_reader_or_retry_state():
    config = _assignments(OVERLAY / "bridge.conf")
    windows = (OVERLAY / "bridge-sender.sh").read_text(encoding="utf-8")
    arch = (OVERLAY / "bridge-arch-sender.sh").read_text(encoding="utf-8")

    assert config["WINDOWS_HOST"] == "192.168.1.35"
    assert config["WINDOWS_INPUT_PORT"] == "10000"
    assert config["ARCH_HOST"] == "192.168.1.11"
    assert config["ARCH_PROCESSOR_INPUT_PORT"] == "10001"
    assert config["ANDROID_SENDER_FEED_PORT"] != config["ANDROID_ARCH_SENDER_FEED_PORT"]

    assert "udp://127.0.0.1:$ANDROID_SENDER_FEED_PORT?" in windows
    assert "ANDROID_ARCH_SENDER_FEED_PORT" not in windows
    assert "srt://$WINDOWS_HOST:$WINDOWS_INPUT_PORT?mode=caller" in windows

    assert "udp://127.0.0.1:$ANDROID_ARCH_SENDER_FEED_PORT?" in arch
    assert "ANDROID_SENDER_FEED_PORT?" not in arch
    assert "srt://$ARCH_HOST:$ARCH_PROCESSOR_INPUT_PORT?mode=caller" in arch

    # SO_REUSE on one shared UDP port would allow kernel packet load-balancing
    # between readers.  Each route must instead be the only reader of its own
    # capture-tee destination.
    assert "reuse=1" not in windows
    assert "reuse=1" not in arch
    assert "android-vcam-windows-sender.state" in windows
    assert "android-vcam-arch-sender.state" in arch
    assert "android-vcam-windows-sender.progress" in windows
    assert "android-vcam-arch-sender.progress" in arch
    for script in (windows, arch):
        assert "while true" in script
        assert "retry_delay=$((retry_delay * 2))" in script
        assert "write_state running" in script
        assert "write_state backoff" in script


def test_module_supervises_both_senders_without_touching_camera_owners():
    service = (OVERLAY / "service.sh").read_text(encoding="utf-8")
    deployment = (OVERLAY / "deploy-live.sh").read_text(encoding="utf-8")

    assert '"$MODDIR/bridge-sender.sh"' in service
    assert '"$MODDIR/bridge-arch-sender.sh"' in service
    assert "android-vcam-sender.pid" in service
    assert "android-vcam-arch-sender.pid" in service
    assert "duplicate private Android transport port" in service
    assert service.index('"$MODDIR/bridge-capture.sh"') < service.index(
        '"$MODDIR/bridge-sender.sh"'
    )
    assert service.index('"$MODDIR/bridge-capture.sh"') < service.index(
        '"$MODDIR/bridge-arch-sender.sh"'
    )

    # A future transport-only hot update may replace its workers, but it must
    # leave the published Camera2 node, producer, and providers registered.
    assert "stop_worker /data/local/tmp/android-vcam-provider.pid" not in deployment
    assert "stop_worker /data/local/tmp/android-vcam-producer.pid" not in deployment
    assert "rmmod" not in deployment
    assert "insmod" not in deployment


def test_module_retries_companion_services_across_user_unlock_race():
    service = (OVERLAY / "service.sh").read_text(encoding="utf-8")
    deployment = (OVERLAY / "deploy-live.sh").read_text(encoding="utf-8")
    helper = (OVERLAY / "bridge-app-service-common.sh").read_text(
        encoding="utf-8"
    )

    assert "APP_SERVICE_START_ATTEMPTS=30" in helper
    assert "APP_SERVICE_START_RETRY_SECONDS=2" in helper
    assert "start_app_foreground_service()" in helper
    assert 'am start-foreground-service --user 0 "$@"' in helper
    assert 'sleep "$APP_SERVICE_START_RETRY_SECONDS"' in helper
    for script in (service, deployment):
        assert '. "$MODDIR/bridge-app-service-common.sh"' in script
        assert (
            'start_app_foreground_service "physical front-camera bridge service"'
            in script
        )
        assert 'start_app_foreground_service "return-audio app service"' in script
    assert service.index("Termux FFmpeg available") < service.index(
        'start_app_foreground_service "physical front-camera bridge service"'
    )


def test_local_processor_unit_has_explicit_route_control_and_offline_manifest_config():
    config = _assignments(ARCH_CONFIG)
    unit = LOCAL_UNIT.read_text(encoding="utf-8")

    assert config["ANDROID_NATIVE_INPUT_HOST"] == "0.0.0.0"
    assert config["ANDROID_NATIVE_INPUT_PORT"] == "10001"
    assert config["LOCAL_PROCESSOR_CONTROL_SOCKET"] == (
        "/run/deep-live-cam/processor-control.sock"
    )
    assert config["ANDROID_NATIVE_HEALTH_FILE"] == (
        "/run/deep-live-cam/android-phone-processed-health.json"
    )
    assert "# DLC_NATIVE256_MANIFEST=" in ARCH_CONFIG.read_text(encoding="utf-8")
    assert "EnvironmentFile=-/etc/deep-live-cam-arch.conf" in unit
    assert "set_config PROCESSOR_XDG_DATA_HOME" in (
        ROOT / "arch-linux" / "install.sh"
    ).read_text(encoding="utf-8")
    assert "set_config PROCESSOR_HOME" in (
        ROOT / "arch-linux" / "install.sh"
    ).read_text(encoding="utf-8")
    assert "/usr/bin/env HOME=${PROCESSOR_HOME}" in unit
    assert "UMask=0022" in unit
    assert "--input-mode android-front" in unit
    assert "--input-port ${ANDROID_NATIVE_INPUT_PORT}" in unit
    assert "--control-socket /run/deep-live-cam/processor-control.sock" in unit
    assert "After=network-online.target deep-live-cam-receiver.service" in unit
    assert "Wants=network-online.target deep-live-cam-receiver.service" in unit
    assert "RuntimeDirectory=deep-live-cam" not in unit
    assert "--swapper-model inswapper-128 --swapper-backend ncnn" in unit
    assert "http://" not in unit and "https://" not in unit

    installer = (ROOT / "arch-linux" / "install.sh").read_text(encoding="utf-8")
    assert (
        'install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-phone-processed.service" '
        "/etc/systemd/system/deep-live-cam-phone-processed.service"
    ) in installer
    # Installation makes the selectable processor unit available; it remains
    # off until the manager explicitly selects/starts the Arch processor.
    assert "systemctl enable deep-live-cam-phone-processed.service" not in installer
    assert "systemctl restart deep-live-cam-phone-processed.service" not in installer


def test_installer_migrates_the_retired_split_bridge_package():
    installer = (ROOT / "arch-linux" / "install.sh").read_text(encoding="utf-8")

    assert "ANDROID_BRIDGE_PACKAGE=dev.vcam.app" in installer


def test_android_overlay_scripts_are_valid_posix_shell():
    for script in sorted(OVERLAY.glob("*.sh")):
        completed = subprocess.run(
            ["sh", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{script.name}: {completed.stderr}"


def test_packager_contains_arch_sender_and_bumps_module_version():
    packager = (ROOT / "android" / "build-vcam-module-v0.4.2.sh").read_text(
        encoding="utf-8"
    )
    properties = _assignments(OVERLAY / "module.prop")

    assert "bridge-arch-sender.sh" in packager
    assert "android-vcam-module-v0.4.9.zip" in packager
    assert properties["version"] == "v0.4.9"
    assert properties["versionCode"] == "24"
    assert not re.search(r"OUTPUT_ZIP=.*v0\.4\.[6-8]", packager)
