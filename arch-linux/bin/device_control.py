#!/usr/bin/env python3
"""Privileged, allow-listed controls for Deep-Live-Cam local devices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from common import (
    DEFAULT_CONFIG,
    DEFAULT_STATE_DIR,
    load_env_file,
    resolve_capture_device,
    resolve_virtual_devices,
)


SYSTEMCTL = "/usr/bin/systemctl"
SENDER_UNIT = "deep-live-cam-sender.service"
RECEIVER_UNIT = "deep-live-cam-receiver.service"
SHADOW_HELPER = Path(__file__).with_name("shadow.py")
RECEIVER_CONTROL_SOCKET = Path("/run/user") / str(os.getuid()) / "deep-live-cam-receiver-source.sock"
COMPONENTS = ("input", "output", "all", "mapping", "source")
ACTIONS = ("start", "stop", "restart")
SOURCE_MODES = ("phone", "arch", "windows", "local", "prerecorded")


class ControlError(RuntimeError):
    pass


def run_checked(command: Sequence[str]) -> None:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if output:
        print(output)
    if result.returncode != 0:
        raise ControlError(
            f"{' '.join(command)} failed with exit code {result.returncode}"
        )


def unit_is_active(unit: str) -> bool:
    result = subprocess.run(
        [SYSTEMCTL, "is-active", "--quiet", unit],
        check=False,
        timeout=5,
    )
    return result.returncode == 0


def systemctl(action: str, *units: str) -> None:
    if action not in ACTIONS or not units:
        raise ControlError("invalid systemd action")
    run_checked([SYSTEMCTL, action, *units])


def required_device_paths(config: dict[str, str]) -> list[Path]:
    state_dir = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    return [
        resolve_capture_device(config, state_dir),
        *resolve_virtual_devices(config),
    ]


def mapping_is_ready(config: dict[str, str]) -> bool:
    return all(path.exists() for path in required_device_paths(config))


def repair_mapping(config: dict[str, str], force: bool = False) -> None:
    if not force and mapping_is_ready(config):
        return

    sender_was_active = unit_is_active(SENDER_UNIT)
    receiver_was_active = unit_is_active(RECEIVER_UNIT)
    if sender_was_active:
        systemctl("stop", SENDER_UNIT)
    if receiver_was_active:
        systemctl("stop", RECEIVER_UNIT)

    error: Exception | None = None
    try:
        run_checked([sys.executable, str(SHADOW_HELPER), "apply"])
    except Exception as exc:
        error = exc
    finally:
        if receiver_was_active:
            try:
                systemctl("start", RECEIVER_UNIT)
            except ControlError as exc:
                error = error or exc
        if sender_was_active:
            try:
                systemctl("start", SENDER_UNIT)
            except ControlError as exc:
                error = error or exc
    if error is not None:
        raise error
    if not mapping_is_ready(config):
        missing = ", ".join(
            str(path) for path in required_device_paths(config) if not path.exists()
        )
        raise ControlError(f"device mapping repair finished but these paths are missing: {missing}")


def set_source_mode(mode: str) -> None:
    """Send source policy change to the receiver via its control socket."""
    import socket
    import json
    import time
    
    if mode not in SOURCE_MODES:
        raise ControlError(f"invalid source mode: {mode}")
    
    if not RECEIVER_CONTROL_SOCKET.exists():
        raise ControlError(
            f"receiver control socket not found at {RECEIVER_CONTROL_SOCKET}; "
            "is the receiver service running?"
        )
    
    payload = json.dumps({"mode": mode}).encode()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(3.0)
        sock.connect(str(RECEIVER_CONTROL_SOCKET))
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        response_raw = sock.recv(4096)
        response = json.loads(response_raw.decode())
        if not response.get("ok"):
            raise ControlError(f"receiver rejected mode change: {response.get('reason')}")
    finally:
        sock.close()


def control(component: str, action: str, config: dict[str, str]) -> None:
    if component not in COMPONENTS or action not in ACTIONS:
        raise ControlError("invalid device control request")

    if component == "mapping":
        if action == "stop":
            raise ControlError("the identity mapping can be repaired, not stopped")
        repair_mapping(config, force=True)
        return
    
    if component == "source":
        # action is the source mode name when component == "source"
        set_source_mode(action)
        return

    if action in ("start", "restart"):
        repair_mapping(config)

    if component == "input":
        systemctl(action, SENDER_UNIT)
    elif component == "output":
        systemctl(action, RECEIVER_UNIT)
    elif action == "stop":
        systemctl("stop", SENDER_UNIT, RECEIVER_UNIT)
    else:
        # Bring up the virtual output before sending frames toward Windows.
        systemctl(action, RECEIVER_UNIT)
        systemctl(action, SENDER_UNIT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=COMPONENTS)
    parser.add_argument(
        "action",
        help="start|stop|restart for input/output/all/mapping, or source mode name for 'source'"
    )
    args = parser.parse_args()
    
    # Source mode changes don't need root.
    if args.component == "source":
        try:
            control(args.component, args.action, {})
        except (ControlError, OSError, subprocess.TimeoutExpired) as exc:
            print(f"deep-live-cam device control: {exc}", file=sys.stderr)
            return 1
        print(f"deep-live-cam device control: source mode set to {args.action}")
        return 0
    
    if os.geteuid() != 0:
        print("deep-live-cam device control: must run as root", file=sys.stderr)
        return 1
    
    if args.action not in ACTIONS:
        print(f"deep-live-cam device control: invalid action '{args.action}' for component '{args.component}'", file=sys.stderr)
        return 1
    
    try:
        control(args.component, args.action, load_env_file(DEFAULT_CONFIG))
    except (ControlError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"deep-live-cam device control: {exc}", file=sys.stderr)
        return 1
    print(f"deep-live-cam device control: {args.component} {args.action} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
