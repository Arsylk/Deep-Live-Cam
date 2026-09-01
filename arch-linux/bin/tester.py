#!/usr/bin/env python3
"""Deep-Live-Cam native manager: entry point, configuration self-check, chrome.

The application itself lives in the :mod:`dlc_manager` package beside this
file.  This module stays small on purpose: it resolves configuration, keeps
``--self-check`` a non-GUI diagnostic, installs the application-wide wheel
guard, and re-exports the handful of names that the installed launcher and the
ownership tests address directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from common import (
    DEFAULT_CONFIG,
    DEFAULT_STATE_DIR,
    load_env_file,
    resolve_capture_device,
    resolve_preview_device,
    resolve_virtual_devices,
)

from dlc_manager.contracts import (
    ANDROID_NATIVE_PREVIEW_PORT,
    SELECTED_STREAM_PORT,
    local_mpegts_preview_command,
)
from dlc_manager.decoders import LogRateLimiter, RawVideoDecoder
from dlc_manager.health import (
    android_native_phone_route_fresh,
    android_native_preview_fresh,
    android_native_route_title,
    android_native_webcam_route_fresh,
    default_android_native_health_file,
    read_json,
)
from dlc_manager.phone_preview import PhoneReturnPreviewWindow
from dlc_manager.shell import ManagerWindow, TesterWindow
from dlc_manager.single_instance import InstanceGuardError, ManagerInstanceGuard
from dlc_manager.viewmodel import ViewInputs, build_view
from dlc_manager.wheel import WheelValueGuard
from dlc_manager.widgets import VideoPane


__all__ = [
    "ANDROID_NATIVE_PREVIEW_PORT",
    "LogRateLimiter",
    "ManagerWindow",
    "ManagerInstanceGuard",
    "PhoneReturnPreviewWindow",
    "RawVideoDecoder",
    "TesterWindow",
    "VideoPane",
    "ViewInputs",
    "WheelValueGuard",
    "android_native_phone_route_fresh",
    "android_native_preview_fresh",
    "android_native_route_title",
    "android_native_webcam_route_fresh",
    "build_view",
    "default_android_native_health_file",
    "load_env_file",
    "local_mpegts_preview_command",
    "main",
    "read_json",
    "self_check",
]


def self_check(config: dict[str, str]) -> int:
    """Validate configuration and packaging without opening a window.

    Nothing here opens a camera: device presence is a filesystem check, and the
    manager modules are only located, not imported for their side effects.
    """
    state_directory = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    capture_camera = resolve_capture_device(config, state_directory)
    virtual_devices = resolve_virtual_devices(config)
    preview_camera = resolve_preview_device(config)
    package = Path(__file__).resolve().parent / "dlc_manager"
    report = {
        "ffmpeg": shutil.which("ffmpeg"),
        "windows_host": config.get("WINDOWS_HOST"),
        "physical_camera": config.get("PHYSICAL_CAMERA"),
        "physical_camera_present": Path(
            config.get("PHYSICAL_CAMERA", "missing")
        ).exists(),
        "capture_camera": str(capture_camera),
        "capture_camera_present": capture_camera.exists(),
        "virtual_cameras": [str(device) for device in virtual_devices],
        "virtual_cameras_present": [device.exists() for device in virtual_devices],
        "preview_camera": str(preview_camera),
        "preview_camera_present": preview_camera.exists(),
        "device_id": config.get("DEVICE_ID", "arch-webcam"),
        "device_slot": int(config.get("DEVICE_SLOT", "1")),
        "windows_input_port": int(config.get("WINDOWS_INPUT_PORT", "10002")),
        "windows_return_port": int(config.get("WINDOWS_RETURN_PORT", "10003")),
        "system_preview_port": int(config.get("LOCAL_PREVIEW_PORT", "11000")),
        "manager_preview_port": int(config.get("MANAGER_PREVIEW_PORT", "11001")),
        "selected_stream_host": config.get("WINDOWS_HOST", "192.168.1.35"),
        "selected_stream_port": int(
            config.get(
                "WINDOWS_SELECTED_STREAM_PORT",
                config.get("WINDOWS_BROADCAST_PORT", str(SELECTED_STREAM_PORT)),
            )
        ),
        "manager_output_preview_port": int(
            config.get("MANAGER_OUTPUT_PREVIEW_PORT", "11003")
        ),
        "local_processed_preview_port": int(
            config.get("LOCAL_PROCESSED_PREVIEW_PORT", "11007")
        ),
        "android_native_preview_port": int(
            config.get(
                "ANDROID_NATIVE_PREVIEW_PORT", str(ANDROID_NATIVE_PREVIEW_PORT)
            )
        ),
        "manager_package": str(package),
        "manager_modules_present": sorted(
            path.relative_to(package).as_posix()
            for path in package.rglob("*.py")
            if "__pycache__" not in path.parts
        ),
        "opens_camera_device": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    complete = bool(report["manager_modules_present"])
    return (
        0
        if report["ffmpeg"]
        and report["capture_camera_present"]
        and report["preview_camera_present"]
        and complete
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate configuration without opening a window",
    )
    args = parser.parse_args()
    try:
        config = load_env_file(DEFAULT_CONFIG)
    except ValueError as exc:
        print(f"deep-live-cam-tester: {exc}", file=sys.stderr)
        return 2
    if args.self_check:
        return self_check(config)

    application = QApplication(sys.argv[:1])
    # PySide6 aborts the process on an unhandled exception raised inside a Qt
    # slot/event handler.  Log the full traceback to a file (and stderr) so a
    # crash during e.g. a preview drag is diagnosable instead of a silent exit.
    import traceback as _traceback
    _crash_log = Path(
        os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    ) / "deep-live-cam-manager-crash.log"

    def _log_uncaught(exc_type, exc_value, exc_tb):
        text = "".join(_traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(_crash_log, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass
        sys.stderr.write(text)
        sys.stderr.flush()

    sys.excepthook = _log_uncaught
    # Installed on the application so dynamically generated camera-adapter
    # controls are covered without every page having to remember to opt in.
    wheel_value_guard = WheelValueGuard(application)
    application.installEventFilter(wheel_value_guard)
    application.setApplicationName("Deep-Live-Cam Manager")
    application.setApplicationDisplayName("Deep-Live-Cam Manager")
    application.setDesktopFileName("deep-live-cam-tester")
    instance_guard = ManagerInstanceGuard()
    try:
        primary = instance_guard.try_acquire()
    except InstanceGuardError as exc:
        print(f"deep-live-cam-tester: {exc}", file=sys.stderr)
        return 2
    if not primary:
        activated = instance_guard.request_activation()
        owner = (
            f" (pid {instance_guard.owner_pid})"
            if instance_guard.owner_pid is not None
            else ""
        )
        detail = "activation requested" if activated else "already running"
        print(f"deep-live-cam-tester: {detail}{owner}", file=sys.stderr)
        instance_guard.close()
        return 0
    if not instance_guard.listen():
        # The kernel lock still prevents duplicate controllers.  Window
        # activation is a convenience, so keep the primary usable and report
        # that only that convenience is unavailable.
        print(
            "deep-live-cam-tester: existing-window activation is unavailable",
            file=sys.stderr,
        )
    icon = QIcon.fromTheme("deep-live-cam")
    if icon.isNull():
        installed_icon = Path(
            "/usr/local/share/icons/hicolor/512x512/apps/deep-live-cam.png"
        )
        if installed_icon.exists():
            icon = QIcon(str(installed_icon))
    if not icon.isNull():
        application.setWindowIcon(icon)
    try:
        window = ManagerWindow(config)
    except Exception:
        instance_guard.close()
        raise

    def activate_window() -> None:
        if window.isMinimized():
            window.showNormal()
        elif not window.isVisible():
            window.show()
        window.raise_()
        window.activateWindow()

    instance_guard.activationRequested.connect(activate_window)
    application.aboutToQuit.connect(instance_guard.close)
    window.show()
    try:
        return application.exec()
    finally:
        instance_guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
