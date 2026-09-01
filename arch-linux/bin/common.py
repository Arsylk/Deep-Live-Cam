#!/usr/bin/env python3
"""Shared helpers for the Arch Linux Deep-Live-Cam services."""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("/etc/deep-live-cam-arch.conf")
DEFAULT_STATE_DIR = Path("/run/deep-live-cam")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def load_env_file(path: Path = DEFAULT_CONFIG) -> dict[str, str]:
    """Read the simple KEY=VALUE syntax used by systemd EnvironmentFile."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        parsed = shlex.split(raw_value, comments=True, posix=True)
        values[key] = " ".join(parsed) if parsed else ""
    return values


def state_dir() -> Path:
    return Path(os.environ.get("STATE_DIR", str(DEFAULT_STATE_DIR)))


def resolve_capture_device(values: Any, state_directory: Path | None = None) -> Path:
    """Return the device node that reaches the real camera hardware.

    The clean topology always returns PHYSICAL_CAMERA.  The private preserved
    node is recognized only when both legacy takeover gates are explicit; one
    stale setting from an older install cannot enable shadow semantics.
    """
    physical = Path(str(values.get("PHYSICAL_CAMERA", "")))
    legacy_shadow = (
        str(values.get("LEGACY_SHADOW", "0")) == "1"
        and str(values.get("SHADOW_ORIGINAL", "0")) == "1"
    )
    if not legacy_shadow:
        return physical
    if not physical.exists():
        return physical
    node = Path(os.path.realpath(physical))
    directory = state_directory or state_dir()
    return directory / "source" / node.name


def resolve_virtual_devices(values: Any) -> list[Path]:
    """Return configured processed-output nodes in receiver priority order."""
    raw = str(values.get("VIRTUAL_CAMERAS", "")).strip()
    candidates = [Path(part) for part in raw.split() if part]
    fallback = str(values.get("VIRTUAL_CAMERA", "/dev/deep-live-cam")).strip()
    if not candidates and fallback:
        candidates.append(Path(fallback))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique or [Path("/dev/deep-live-cam")]


def resolve_preview_device(values: Any) -> Path:
    """Select the first live processed-output node for desktop consumers."""
    devices = resolve_virtual_devices(values)
    return next((device for device in devices if device.exists()), devices[0])


def write_state(name: str, state: dict[str, Any]) -> None:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = time.time()
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, directory / f"{name}.json")
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sd_notify(message: str) -> None:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
            notifier.connect(address)
            notifier.sendall(message.encode("utf-8"))
    except OSError:
        # Losing a notification must not take down the video path.
        pass


def install_signal_handlers(stop_callback: Any) -> None:
    def handle_signal(_signum: int, _frame: Any) -> None:
        stop_callback()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def stop_process(process: subprocess.Popen[Any] | None, grace: float = 3.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace)


def srt_query(options: dict[str, str | int]) -> str:
    return "&".join(f"{key}={value}" for key, value in options.items())
