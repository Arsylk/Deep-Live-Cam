#!/usr/bin/env python3
"""Reading and interpreting on-disk health documents.

These helpers are pure apart from reading files that a service already wrote.
They never contact a camera, a socket, or a remote host, which keeps the rules
about "is this stream really fresh?" testable in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .contracts import ANDROID_NATIVE_PREVIEW_PORT


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def file_age(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - Path(path).stat().st_mtime)
    except OSError:
        return None


def default_android_native_health_file() -> Path:
    # The native processor is a system service and its unit writes this
    # world-readable health snapshot. A former user-session prototype used
    # $XDG_RUNTIME_DIR; reading that stale file made the manager repeatedly
    # reapply standby state while the real service was synchronized.
    return Path("/run/deep-live-cam/android-phone-processed-health.json")


def android_native_preview_fresh(
    health: dict[str, Any] | None,
    *,
    health_age: float | None = None,
    max_age: float = 2.0,
    expected_port: int = ANDROID_NATIVE_PREVIEW_PORT,
) -> bool:
    """Return whether health describes a fresh exact return-encoder tee."""
    if not isinstance(health, dict) or health.get("state") != "running":
        return False
    if health_age is not None and health_age > max(10.0, max_age * 3):
        return False
    returned = health.get("return")
    if not isinstance(returned, dict):
        return False
    preview_url = returned.get("preview_url")
    if not isinstance(preview_url, str) or not preview_url.startswith(
        f"udp://127.0.0.1:{int(expected_port)}?"
    ):
        return False
    try:
        frame_age = float(returned.get("last_frame_age"))
    except (TypeError, ValueError):
        return False
    return returned.get("worker_alive") is True and frame_age <= max_age


def android_native_route_title(health: dict[str, Any] | None) -> str:
    """Return an explicit title for the independently owned phone return."""
    values = health if isinstance(health, dict) else {}
    processing = values.get("processing")
    processing = processing if isinstance(processing, dict) else {}
    model = str(processing.get("model", "unknown"))
    model_title = {
        "inswapper-128": "INSWAPPER 128",
        "dlc_swap256m": "NATIVE 256",
    }.get(model, model.upper())
    route = values.get("route")
    if route == "android-camera-processed-to-android":
        return f"PHONE FRONT → ARCH {model_title} → CAMERA2 120"
    if route == "arch-webcam-processed-to-android":
        return f"ARCH WEBCAM → {model_title} → PHONE"
    return "PROCESSED RESULT SENT TO PHONE"


def phone_return_relay_preview_fresh(
    health: dict[str, Any] | None,
    *,
    health_age: float | None = None,
    max_age: float = 2.0,
    expected_port: int = ANDROID_NATIVE_PREVIEW_PORT,
) -> bool:
    """Return whether the exclusive relay is emitting the exact phone stream."""
    if not isinstance(health, dict) or health.get("state") != "running":
        return False
    if health_age is not None and health_age > max(10.0, max_age * 3):
        return False
    output = health.get("output")
    if not isinstance(output, dict) or output.get("preview_port") != expected_port:
        return False
    source = health.get("source")
    try:
        frame_age = float(health.get("last_frame_age"))
    except (TypeError, ValueError):
        return False
    return bool(
        source in {"local", "windows"}
        and health.get("effective_source") == source
        and health.get("worker_alive") is True
        and health.get("transport_open") is True
        and health.get("streaming") is True
        and frame_age <= max_age
    )


def phone_return_relay_title(health: dict[str, Any] | None) -> str:
    values = health if isinstance(health, dict) else {}
    return {
        "local": "ARCH LOCAL MODEL → PHONE CAMERA2 120",
        "windows": "WINDOWS PROCESSED RESULT → PHONE CAMERA2 120",
        "off": "PHONE PROCESSED RETURN DISABLED",
    }.get(str(values.get("source")), "PROCESSED RESULT SENT TO PHONE")


def android_native_phone_route_fresh(
    health: dict[str, Any] | None,
    *,
    health_age: float | None = None,
    expected_port: int = ANDROID_NATIVE_PREVIEW_PORT,
) -> bool:
    return bool(
        isinstance(health, dict)
        and health.get("route") == "android-camera-processed-to-android"
        and android_native_preview_fresh(
            health,
            health_age=health_age,
            expected_port=expected_port,
        )
    )


def android_native_webcam_route_fresh(
    health: dict[str, Any] | None,
    *,
    health_age: float | None = None,
    expected_port: int = ANDROID_NATIVE_PREVIEW_PORT,
) -> bool:
    return bool(
        isinstance(health, dict)
        and health.get("route") == "arch-webcam-processed-to-android"
        and android_native_preview_fresh(
            health,
            health_age=health_age,
            expected_port=expected_port,
        )
    )
