from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "arch-linux"


def test_packaged_topology_is_one_safe_xiaomi_identity_clone() -> None:
    config = (ARCH / "config/deep-live-cam-arch.conf").read_text()
    module = (ARCH / "config/deep-live-cam-v4l2loopback.conf").read_text()

    assert 'VIRTUAL_CAMERAS="/dev/deep-live-cam"' in config
    assert "VIRTUAL_CAMERA=/dev/deep-live-cam" in config
    assert "LOOPBACK_NODE=/dev/video42" in config
    assert "SHADOW_ORIGINAL=0" in config
    assert "LEGACY_SHADOW=0" in config
    assert "DROIDCAM_COMPAT=0" in config
    assert 'devices=1 video_nr=42 card_label="Xiaomi Cam"' in module
    assert 'driver_name="uvcvideo"' in module
    assert 'bus_info="usb-0000:02:00.0-4"' in module
    assert "exclusive_caps=1" in module
    assert "video_nr=0" not in module
    assert "video50" not in module
    assert "parent_device=" not in module


def test_udev_creates_only_the_stable_alias_without_shadow_service() -> None:
    rules = (ARCH / "config/70-deep-live-cam.rules").read_text()

    assert 'KERNEL=="video42"' in rules
    assert 'ATTR{name}=="Xiaomi Cam"' in rules
    assert 'SYMLINK+="deep-live-cam"' in rules
    assert "SYSTEMD_WANTS" not in rules
    assert "droidcam" not in rules.lower()


def test_desktop_camera_policy_exposes_only_video42() -> None:
    policy = (
        ARCH / "config/51-deep-live-cam-camera-policy.conf"
    ).read_text()

    assert 'api.v4l2.path = "!~^/dev/video42$"' in policy
    assert 'api.v4l2.path = "/dev/video42"' in policy
    assert 'device.description = "Xiaomi Cam"' in policy
    assert "device.disabled = true" in policy
    assert "node.disabled = true" in policy
    assert 'device.name = "~libcamera_device.*"' in policy


def test_direct_v4l2_browser_access_is_reserved_without_renaming_sonix() -> None:
    access = (
        ARCH / "config/72-deep-live-cam-camera-access.rules"
    ).read_text()

    assert 'KERNEL=="video[0-9]*"' in access
    assert 'KERNEL!="video42"' in access
    assert 'GROUP:="deep-live-cam"' in access
    assert 'TAG-="uaccess"' in access
    assert "/usr/bin/setfacl --remove-all" in access


def test_receiver_reannounces_exclusive_caps_transition() -> None:
    unit = (ARCH / "systemd/deep-live-cam-receiver.service").read_text()
    helper = (ARCH / "bin/publish_virtual_camera.py").read_text()

    assert "ExecStartPost=-+/usr/bin/python3" in unit
    assert "publish_virtual_camera.py --device ${LOOPBACK_NODE}" in unit
    assert "VIDIOC_QUERYCAP" in helper
    assert 'EXPECTED_DEVICE_NAME = "video42"' in helper
    assert 'EXPECTED_DRIVER = "uvcvideo"' in helper
    assert 'EXPECTED_CARD = "Xiaomi Cam"' in helper
    assert 'EXPECTED_BUS = "usb-0000:02:00.0-4"' in helper
    assert 'uevent.write_text("change\\n"' in helper


def test_installer_packages_browser_camera_policy_without_restarting_user_session() -> None:
    installer = (ARCH / "install.sh").read_text()

    assert "51-deep-live-cam-camera-policy.conf" in installer
    assert "72-deep-live-cam-camera-access.rules" in installer
    assert "publish_virtual_camera.py" in installer
    assert not any(
        line.lstrip().startswith("systemctl --user")
        for line in installer.splitlines()
    )


def test_installer_stages_a_loaded_module_change_instead_of_unloading_it() -> None:
    installer = (ARCH / "install.sh").read_text()

    assert "pacman -U --noconfirm \"$CUSTOM_LOOPBACK_PACKAGE\"" in installer
    assert "pacman -Rdd --noconfirm v4l2loopback-dkms" in installer
    assert "v4l2loopback-custom-${CUSTOM_LOOPBACK_VERSION}" in installer
    assert "deep-live-cam-arch.conf.nextboot" in installer
    assert "MIGRATION_WAS_STAGED" in installer
    assert "|| ((MIGRATION_WAS_STAGED))" in installer
    assert "deep-live-cam-topology-migration.service" in installer
    assert "modprobe -r v4l2loopback" not in installer
    assert "systemctl stop deep-live-cam-sender" not in installer
    assert "systemctl stop deep-live-cam-receiver" not in installer
    assert "LEGACY_SHADOW=0" in installer
    assert "--legacy-shadow" in installer
    assert "rm -f /usr/local/lib/deep-live-cam-arch/shadow.py" in installer
    assert "rm -f /etc/systemd/system/deep-live-cam-shadow.service" in installer


def test_installer_blocks_the_obsolete_droidcam_loopback_fork() -> None:
    installer = (ARCH / "install.sh").read_text()
    blocklist = (
        ARCH / "config/deep-live-cam-droidcam-modprobe.conf"
    ).read_text()

    assert "install v4l2loopback_dc /usr/bin/true" in blocklist
    assert '"$SCRIPT_DIR/config/deep-live-cam-droidcam-modprobe.conf"' in installer
    assert "/etc/modprobe.d/deep-live-cam-disable-v4l2loopback-dc.conf" in installer
    assert installer.index("rm -f /etc/modprobe.d/deep-live-cam-droidcam-compat.conf") < installer.index(
        "/etc/modprobe.d/deep-live-cam-disable-v4l2loopback-dc.conf"
    )


def test_early_boot_commit_precedes_module_and_camera_services() -> None:
    unit = (
        ARCH / "systemd/deep-live-cam-topology-migration.service"
    ).read_text()

    assert "DefaultDependencies=no" in unit
    assert "Before=systemd-modules-load.service" in unit
    assert "deep-live-cam-receiver.service" in unit
    assert "deep-live-cam-sender.service" in unit
    assert "ConditionPathExists=/etc/deep-live-cam-arch.conf.nextboot" in unit
