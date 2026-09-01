#!/usr/bin/env python3
"""Pure helpers for describing the Arch/Windows/Android camera topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit


ROUTE_ANDROID = "android"
ROUTE_ARCH = "arch"
ROUTE_UNKNOWN = "unknown"


def endpoint_host(url: Any) -> str | None:
    """Return a normalized host from an SRT-style endpoint URL."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        return urlsplit(url.strip()).hostname
    except ValueError:
        return None


def stream_is_fresh(stream: Any, maximum_age: float = 2.5) -> bool:
    """Use an explicit streaming flag and frame age without guessing from FPS."""
    if not isinstance(stream, dict):
        return False
    if stream.get("streaming") is False:
        return False
    try:
        age = float(stream.get("last_frame_age"))
    except (TypeError, ValueError):
        return False
    return age <= maximum_age


@dataclass(frozen=True)
class PipelineTopology:
    selected: str
    return_host: str | None
    arch_sender_active: bool
    android_sender_active: bool | None
    conflict: bool
    mismatch: bool
    summary: str
    warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_topology(
    windows_health: Any,
    *,
    arch_host: str,
    android_host: str,
    arch_sender_active: bool,
    android_sender_active: bool | None,
) -> PipelineTopology:
    """Infer the selected round trip from the explicit Windows slot state.

    New routers expose ``selected_device_id`` and isolated port pairs, so
    multiple clients may keep their callers alive without contention. The
    return-host heuristic remains only for health responses from the legacy
    single-port service during migration.
    """
    health_available = isinstance(windows_health, dict)
    health = windows_health if health_available else {}
    output = health.get("output") if isinstance(health.get("output"), dict) else {}
    return_host = endpoint_host(output.get("url"))
    selected_device_id = health.get("selected_device_id")
    isolated_slots = isinstance(selected_device_id, str)

    if selected_device_id == "android-phone":
        selected = ROUTE_ANDROID
        summary = "Android phone slot 0 → Windows → Android Camera2 output"
    elif selected_device_id == "arch-webcam":
        selected = ROUTE_ARCH
        summary = "Arch USB webcam slot 1 → Windows → Arch virtual camera"
    elif return_host and android_host and return_host == android_host:
        selected = ROUTE_ANDROID
        summary = "Android phone camera → Windows → Android Camera2 output"
    elif return_host and arch_host and return_host == arch_host:
        selected = ROUTE_ARCH
        summary = "Arch USB webcam → Windows → Arch virtual camera"
    elif health_available and android_sender_active is True and not arch_sender_active:
        selected = ROUTE_ANDROID
        summary = "Android sender detected; Windows return target is unconfirmed"
    elif health_available and arch_sender_active and android_sender_active is not True:
        selected = ROUTE_ARCH
        summary = "Arch sender detected; Windows return target is unconfirmed"
    else:
        selected = ROUTE_UNKNOWN
        summary = "Camera route could not be confirmed"

    conflict = bool(
        not isolated_slots and arch_sender_active and android_sender_active is True
    )
    mismatch = bool(
        not isolated_slots
        and (
            (selected == ROUTE_ANDROID and arch_sender_active)
            or (selected == ROUTE_ARCH and android_sender_active is True)
        )
    )
    warning: str | None = None
    if conflict:
        warning = (
            "Both the Android and Arch input senders are running. Stop the "
            "sender that is not selected; they compete for Windows UDP 10000."
        )
    elif not isolated_slots and selected == ROUTE_ANDROID and arch_sender_active:
        warning = "Windows returns to Android, but the Arch webcam sender is still running."
    elif not isolated_slots and selected == ROUTE_ARCH and android_sender_active is True:
        warning = "Windows returns to Arch, but the Android phone sender is still running."
    elif selected == ROUTE_ANDROID and android_sender_active is False:
        warning = "The Android route is selected, but the phone camera bridge is stopped."
    elif selected == ROUTE_UNKNOWN:
        warning = "Check the Windows output target before starting either input sender."

    return PipelineTopology(
        selected=selected,
        return_host=return_host,
        arch_sender_active=arch_sender_active,
        android_sender_active=android_sender_active,
        conflict=conflict,
        mismatch=mismatch,
        summary=summary,
        warning=warning,
    )
