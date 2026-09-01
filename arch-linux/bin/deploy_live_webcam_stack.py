#!/usr/bin/env python3
"""Narrowly deploy the measured webcam profile and stable receiver route.

Unlike the full installer, this helper does not touch udev, v4l2loopback,
Android, or firewall state.  It refuses to restart a camera owner while an
unrelated process has one of the configured clean-topology nodes open.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from camera_profiles import DEFAULT_CAMERA_PROFILE, profile_config_values
from common import load_env_file, resolve_capture_device, resolve_virtual_devices


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = Path("/etc/deep-live-cam-arch.conf")
INSTALL_DIRECTORY = Path("/usr/local/lib/deep-live-cam-arch")
DEPLOY_FILES = {
    "common.py": 0o644,
    "camera_profiles.py": 0o644,
    "camera_adapters.py": 0o755,
    "configure_camera.py": 0o755,
    "sender.py": 0o755,
    "receiver.py": 0o755,
    "select_receiver_source.py": 0o755,
    "configure_receiver_output.py": 0o755,
    "local_processor_control.py": 0o755,
    "tester.py": 0o755,
}
# The native manager is a package next to tester.py. Deploying the entry point
# without its modules would leave a launcher that cannot import its own UI, so
# the whole package ships as one unit.
DEPLOY_PACKAGES = ("dlc_manager",)
PACKAGE_FILE_MODE = 0o644


def package_modules(package: str) -> list[Path]:
    """Return every module of one package, in a stable order."""
    root = SCRIPT_DIRECTORY / package
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def deploy_package(package: str) -> list[Path]:
    """Copy a package atomically and remove modules that no longer exist."""
    source_root = SCRIPT_DIRECTORY / package
    target_root = INSTALL_DIRECTORY / package
    modules = package_modules(package)
    if not modules:
        raise RuntimeError(f"missing deployment package: {package}")
    installed: list[Path] = []
    expected: set[Path] = set()
    for module in modules:
        relative = module.relative_to(source_root)
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(module, destination, PACKAGE_FILE_MODE)
        installed.append(destination)
        expected.add(destination)
    if target_root.is_dir():
        for stale in target_root.rglob("*.py"):
            if "__pycache__" in stale.parts or stale in expected:
                continue
            stale.unlink()
        for cache in target_root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
    return installed
CONFIG_CHANGES = {
    **profile_config_values(DEFAULT_CAMERA_PROFILE),
    "NATIVE_PROCESSOR_SOURCE_PORT": "11005",
    "LOCAL_PROCESSED_PORT": "11006",
    "LOCAL_PROCESSED_PREVIEW_PORT": "11007",
    "RECEIVER_SOURCE": "windows",
    "RECEIVER_SOURCE_STATE_FILE": (
        "/var/lib/deep-live-cam/receiver-source.json"
    ),
}


def service_pids(unit: str) -> set[int]:
    result = subprocess.run(
        ["systemctl", "show", unit, "--property=ControlGroup", "--value"],
        capture_output=True,
        text=True,
        check=False,
    )
    group = result.stdout.strip()
    if result.returncode != 0 or not group:
        return set()
    path = Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.procs"
    try:
        return {
            int(line)
            for line in path.read_text(encoding="ascii").splitlines()
            if line.strip().isdigit()
        }
    except OSError:
        return set()


def device_users(devices: list[Path]) -> set[int]:
    targets: set[tuple[int, int]] = set()
    for device in devices:
        try:
            status = device.stat()
        except OSError:
            continue
        targets.add((status.st_dev, status.st_rdev))
    users: set[int] = set()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = (process / "fd").iterdir()
            for descriptor in descriptors:
                try:
                    status = descriptor.stat()
                except OSError:
                    continue
                if (status.st_dev, status.st_rdev) in targets:
                    users.add(int(process.name))
                    break
        except OSError:
            continue
    return users


def require_clean_topology(config: dict[str, str]) -> None:
    """Refuse to deploy userspace onto the retired shadow camera layout."""
    if (
        resolve_virtual_devices(config) != [Path("/dev/deep-live-cam")]
        or config.get("LOOPBACK_NODE", "/dev/video42") != "/dev/video42"
        or config.get("LEGACY_SHADOW", "0") != "0"
        or config.get("SHADOW_ORIGINAL", "0") != "0"
    ):
        raise RuntimeError(
            "the narrow deploy helper requires the clean /dev/video42 -> "
            "/dev/deep-live-cam topology; run arch-linux/install.sh and "
            "complete its controlled-reboot migration first"
        )


def ensure_not_busy(config: dict[str, str] | None = None) -> None:
    config = dict(config or load_env_file(CONFIG_PATH))
    require_clean_topology(config)
    state_directory = Path(config.get("STATE_DIR", "/run/deep-live-cam"))
    receiver_owned = service_pids("deep-live-cam-receiver.service")
    sender_owned = service_pids("deep-live-cam-sender.service")
    virtual_users = device_users(resolve_virtual_devices(config))
    physical_users = device_users(
        [resolve_capture_device(config, state_directory)]
    )
    unexpected = (virtual_users - receiver_owned) | (physical_users - sender_owned)
    if unexpected:
        raise RuntimeError(
            "camera deployment deferred because unrelated process IDs have a "
            "camera open: " + ", ".join(str(pid) for pid in sorted(unexpected))
        )


def atomic_copy(source: Path, destination: Path, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def update_config(changes: dict[str, str]) -> Path:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.backup-{stamp}")
    shutil.copy2(CONFIG_PATH, backup)

    remaining = dict(changes)
    output: list[str] = []
    for line in original.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)
    output.extend(f"{key}={value}" for key, value in remaining.items())

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CONFIG_PATH.name}.", dir=CONFIG_PATH.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(output) + "\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, CONFIG_PATH)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return backup


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="install and persist only; leave currently loaded services untouched",
    )
    args = parser.parse_args()
    if os.geteuid() != 0:
        print(json.dumps({"ok": False, "error": "run through pkexec or sudo"}))
        return 1
    try:
        missing = [name for name in DEPLOY_FILES if not (SCRIPT_DIRECTORY / name).is_file()]
        missing.extend(
            package for package in DEPLOY_PACKAGES if not package_modules(package)
        )
        if missing:
            raise RuntimeError("missing deployment files: " + ", ".join(missing))
        runtime_config = load_env_file(CONFIG_PATH)
        require_clean_topology(runtime_config)
        if not args.no_restart:
            ensure_not_busy(runtime_config)
        INSTALL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        backup = update_config(CONFIG_CHANGES)
        installed_modules: list[Path] = []
        for name, mode in DEPLOY_FILES.items():
            atomic_copy(
                SCRIPT_DIRECTORY / name,
                INSTALL_DIRECTORY / name,
                mode,
            )
        for package in DEPLOY_PACKAGES:
            installed_modules.extend(deploy_package(package))
        if not args.no_restart:
            run_checked(
                [
                    "systemctl",
                    "restart",
                    "deep-live-cam-receiver.service",
                    "deep-live-cam-sender.service",
                ]
            )
            run_checked(
                [
                    "systemctl",
                    "is-active",
                    "--quiet",
                    "deep-live-cam-receiver.service",
                    "deep-live-cam-sender.service",
                ]
            )
        result = {
            "ok": True,
            "profile": DEFAULT_CAMERA_PROFILE,
            "receiver_source": "windows",
            "local_model_input_port": 11005,
            "local_processed_port": 11006,
            "local_processed_preview_port": 11007,
            "services_restarted": not args.no_restart,
            "config_backup": str(backup),
            "installed": [
                *(str(INSTALL_DIRECTORY / name) for name in DEPLOY_FILES),
                *(str(path) for path in installed_modules),
            ],
        }
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
