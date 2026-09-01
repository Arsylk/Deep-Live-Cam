"""The installer and the narrow deployment helper ship every manager module."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
PACKAGE = BIN / "dlc_manager"
INSTALL_SCRIPT = ROOT / "arch-linux" / "install.sh"
ARCH_CONFIG = ROOT / "arch-linux" / "config" / "deep-live-cam-arch.conf"
DESKTOP_ENTRY = ROOT / "arch-linux" / "config" / "deep-live-cam-tester.desktop"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

import deploy_live_webcam_stack as deploy  # noqa: E402


def _checkout_modules() -> set[str]:
    return {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def test_the_helper_knows_about_every_module_in_the_checkout():
    modules = {
        path.relative_to(PACKAGE).as_posix()
        for path in deploy.package_modules("dlc_manager")
    }

    assert modules == _checkout_modules()
    assert "shell.py" in modules
    assert "pages/live.py" in modules


def test_deploying_the_package_copies_every_module_and_prunes_stale_ones(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(deploy, "INSTALL_DIRECTORY", tmp_path)
    target = tmp_path / "dlc_manager"
    (target / "pages").mkdir(parents=True)
    (target / "retired.py").write_text("# removed upstream\n", encoding="utf-8")
    (target / "pages" / "retired.py").write_text("# removed\n", encoding="utf-8")
    cache = target / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"\x00")

    installed = deploy.deploy_package("dlc_manager")

    copied = {path.relative_to(target).as_posix() for path in installed}
    assert copied == _checkout_modules()
    on_disk = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*.py")
    }
    assert on_disk == _checkout_modules()
    assert not (target / "retired.py").exists()
    assert not (target / "pages" / "retired.py").exists()
    assert not cache.exists()
    assert (target / "shell.py").read_text(encoding="utf-8") == (
        PACKAGE / "shell.py"
    ).read_text(encoding="utf-8")


def test_a_missing_package_is_reported_instead_of_silently_skipped(monkeypatch):
    assert deploy.package_modules("not-a-package") == []


def test_the_entry_point_and_its_package_are_deployed_together():
    assert "tester.py" in deploy.DEPLOY_FILES
    assert "dlc_manager" in deploy.DEPLOY_PACKAGES


def test_narrow_deploy_busy_check_uses_only_configured_clean_topology_nodes(
    monkeypatch,
):
    observed: list[list[Path]] = []
    monkeypatch.setattr(
        deploy,
        "load_env_file",
        lambda _path: {
            "STATE_DIR": "/run/deep-live-cam",
            "PHYSICAL_CAMERA": "/dev/v4l/by-id/usb-camera",
            "VIRTUAL_CAMERA": "/dev/deep-live-cam",
            "VIRTUAL_CAMERAS": "/dev/deep-live-cam",
            "LOOPBACK_NODE": "/dev/video42",
            "LEGACY_SHADOW": "0",
            "SHADOW_ORIGINAL": "0",
        },
    )
    monkeypatch.setattr(deploy, "service_pids", lambda _unit: set())
    monkeypatch.setattr(
        deploy,
        "device_users",
        lambda devices: observed.append(list(devices)) or set(),
    )

    deploy.ensure_not_busy()

    assert observed == [
        [Path("/dev/deep-live-cam")],
        [Path("/dev/v4l/by-id/usb-camera")],
    ]
    assert all(
        path not in {Path("/dev/video0"), Path("/dev/video1")}
        for group in observed
        for path in group
    )


def test_narrow_deploy_refuses_the_retired_shadow_topology():
    with pytest.raises(RuntimeError, match="clean /dev/video42"):
        deploy.require_clean_topology(
            {
                "VIRTUAL_CAMERA": "/dev/video0",
                "VIRTUAL_CAMERAS": "/dev/video0 /dev/video1",
                "LOOPBACK_NODE": "/dev/video0",
                "LEGACY_SHADOW": "1",
                "SHADOW_ORIGINAL": "1",
            }
        )


def test_packaged_desktop_entry_launches_the_native_manager_without_a_web_ui():
    desktop = DESKTOP_ENTRY.read_text(encoding="utf-8")

    assert "Exec=/usr/local/bin/deep-live-cam-tester" in desktop
    assert "TryExec=/usr/local/bin/deep-live-cam-tester" in desktop
    assert "Terminal=false" in desktop
    assert "http://" not in desktop
    assert "https://" not in desktop


def test_packaged_windows_processor_address_is_the_reserved_host():
    config = ARCH_CONFIG.read_text(encoding="utf-8")

    assert "WINDOWS_HOST=192.168.1.35" in config
    assert "--windows-host 192.168.1.35" in INSTALL_SCRIPT.read_text(
        encoding="utf-8"
    )


def test_the_installer_installs_the_whole_manager_package():
    script = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "/usr/local/lib/deep-live-cam-arch/dlc_manager" in script
    # Recursive discovery, so a new page module needs no installer change.
    assert "find \"$SCRIPT_DIR/bin/dlc_manager\" -name '*.py'" in script
    assert "install -D -m 0644" in script
    assert "ln -sfn /usr/local/lib/deep-live-cam-arch/tester.py" in script
    assert subprocess.run(
        ["bash", "-n", str(INSTALL_SCRIPT)], check=False
    ).returncode == 0


def test_the_self_check_reports_the_installed_module_set():
    result = subprocess.run(
        [sys.executable, str(BIN / "tester.py"), "--self-check"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(report["manager_modules_present"]) == _checkout_modules()
    assert report["opens_camera_device"] is False
