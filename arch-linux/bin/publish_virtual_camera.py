#!/usr/bin/env python3
"""Re-announce the Xiaomi Cam after v4l2loopback becomes capture-capable.

With ``exclusive_caps=1`` the loopback device intentionally starts as an
output-only node.  Once the receiver opens its producer side, it becomes a
capture device, but a WirePlumber instance that enumerated it earlier may keep
only a device object and never create the browser-visible Video/Source node.
This root-only helper waits for that transition and emits one scoped uevent.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import stat
import struct
import sys
import time


VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_DEVICE_CAPS = 0x80000000
CAPABILITY_STRUCT_SIZE = 104
EXPECTED_DEVICE_NAME = "video42"
EXPECTED_DRIVER = "uvcvideo"
EXPECTED_CARD = "Xiaomi Cam"
EXPECTED_BUS = "usb-0000:02:00.0-4"
DEFAULT_SYSFS_ROOT = Path("/sys/class/video4linux")


class PublishError(RuntimeError):
    """The configured node is absent, unsafe, or failed to become capture."""


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def query_capabilities(device: Path) -> tuple[str, str, str, int]:
    """Return driver, card, bus identity, and effective capabilities."""

    descriptor = os.open(device, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        payload = bytearray(CAPABILITY_STRUCT_SIZE)
        fcntl.ioctl(descriptor, VIDIOC_QUERYCAP, payload, True)
    finally:
        os.close(descriptor)

    driver = _cstring(payload[0:16])
    card = _cstring(payload[16:48])
    bus = _cstring(payload[48:80])
    _version, capabilities, device_caps = struct.unpack_from("III", payload, 80)
    effective = device_caps if capabilities & V4L2_CAP_DEVICE_CAPS else capabilities
    return driver, card, bus, effective


def resolve_expected_device(device: Path) -> Path:
    try:
        resolved = device.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PublishError(f"virtual camera does not exist: {device}") from exc
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise PublishError(f"cannot inspect virtual camera {resolved}: {exc}") from exc
    if not stat.S_ISCHR(mode) or resolved.name != EXPECTED_DEVICE_NAME:
        raise PublishError(
            f"refusing to publish unexpected device {resolved}; expected /dev/video42"
        )
    return resolved


def wait_for_capture(
    device: Path,
    *,
    timeout: float,
    interval: float = 0.1,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> tuple[str, str, str, int]:
    """Wait until the verified loopback advertises its capture side."""

    deadline = clock() + timeout
    last_error: OSError | None = None
    while True:
        try:
            driver, card, bus, capabilities = query_capabilities(device)
            last_error = None
            if (
                driver != EXPECTED_DRIVER
                or card != EXPECTED_CARD
                or bus != EXPECTED_BUS
            ):
                raise PublishError(
                    "refusing to publish mismatched camera: "
                    f"driver={driver!r}, card={card!r}, bus={bus!r}"
                )
            if capabilities & V4L2_CAP_VIDEO_CAPTURE:
                return driver, card, bus, capabilities
        except OSError as exc:
            # An exclusive output-only endpoint can reject a read-side open.
            # Retry only until the bounded deadline; identity mismatches above
            # remain hard failures.
            last_error = exc
        if clock() >= deadline:
            detail = f": {last_error}" if last_error is not None else ""
            raise PublishError(
                f"{device} did not become capture-capable within {timeout:g}s{detail}"
            )
        sleeper(interval)


def trigger_scoped_change(
    device: Path, *, sysfs_root: Path = DEFAULT_SYSFS_ROOT
) -> Path:
    """Write one ``change`` event to the already-validated video42 node."""

    sysfs_device = sysfs_root / device.name
    name_path = sysfs_device / "name"
    try:
        sysfs_name = name_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PublishError(f"cannot verify {name_path}: {exc}") from exc
    if sysfs_name != EXPECTED_CARD:
        raise PublishError(
            f"refusing to trigger {sysfs_device}: sysfs name is {sysfs_name!r}"
        )
    uevent = sysfs_device / "uevent"
    try:
        uevent.write_text("change\n", encoding="ascii")
    except OSError as exc:
        raise PublishError(f"cannot trigger {uevent}: {exc}") from exc
    return uevent


def publish(device: Path, *, timeout: float, sysfs_root: Path = DEFAULT_SYSFS_ROOT) -> None:
    resolved = resolve_expected_device(device)
    wait_for_capture(resolved, timeout=timeout)
    trigger_scoped_change(resolved, sysfs_root=sysfs_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path, default=Path("/dev/video42"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if args.timeout <= 0 or args.timeout > 60:
        parser.error("--timeout must be greater than 0 and at most 60 seconds")
    if os.geteuid() != 0:
        print("virtual camera publish helper must run as root", file=sys.stderr)
        return 1
    try:
        publish(args.device, timeout=args.timeout)
    except PublishError as exc:
        print(f"deep-live-cam virtual camera publish: {exc}", file=sys.stderr)
        return 1
    print("deep-live-cam virtual camera publish: Xiaomi Cam re-announced as capture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
