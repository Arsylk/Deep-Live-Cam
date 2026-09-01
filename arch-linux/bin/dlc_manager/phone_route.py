#!/usr/bin/env python3
"""Pure ownership rules for the Android processed-return transport."""

from __future__ import annotations

from typing import Any, Mapping

from .desired_state import (
    INPUT_ANDROID_BACK,
    INPUT_ANDROID_FRONT,
    INPUT_ARCH_WEBCAM,
    INPUT_PRERECORDED,
    OUTPUT_ANDROID_PHONE,
    PROCESSOR_ARCH,
)


RELAY_OFF = "off"
RELAY_LOCAL = "local"
RELAY_WINDOWS = "windows"


def desired_relay_source(desired: Mapping[str, Any]) -> str:
    """Return the only valid Arch-originated phone-return owner.

    Windows with a phone input uses the paired slot-0 return directly, so the
    Arch relay must be off.  Every other Arch-originated return is serialized
    through the one persistent relay daemon.
    """
    outputs = desired.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    if outputs.get(OUTPUT_ANDROID_PHONE) is not True:
        return RELAY_OFF
    current_input = desired.get("input")
    # Prerecorded video is produced entirely on Arch (the receiver's file_relay
    # writes the local phone-relay port), so it returns to the phone through the
    # local relay regardless of which processor is nominally selected.
    if current_input == INPUT_PRERECORDED:
        return RELAY_LOCAL
    if desired.get("processor") == PROCESSOR_ARCH:
        return RELAY_LOCAL
    if current_input == INPUT_ARCH_WEBCAM:
        return RELAY_WINDOWS
    return RELAY_OFF


def route_signature(desired: Mapping[str, Any], windows_device: str) -> tuple[Any, ...]:
    outputs = desired.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    return (
        desired.get("processor"),
        desired.get("input"),
        outputs.get(OUTPUT_ANDROID_PHONE) is True,
        str(windows_device),
    )


def windows_runtime_ready(
    document: Mapping[str, Any] | None,
    device_id: str,
    *,
    require_selected_stream: bool = False,
) -> bool:
    """Verify the live router, not only the registry's persisted selection."""
    if not isinstance(document, Mapping):
        return False
    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping):
        return False
    if document.get("selected_device_id") != device_id:
        return False
    if runtime.get("selected_device_id") != device_id:
        return False
    if runtime.get("switching") is True or runtime.get("last_switch_error"):
        return False
    if not require_selected_stream:
        return True
    selected = runtime.get("selected_stream", runtime.get("broadcast"))
    return bool(
        isinstance(selected, Mapping)
        and selected.get("worker_alive") is True
        and selected.get("streaming") is True
    )


def relay_is_closed(document: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(document, Mapping)
        and document.get("source") == RELAY_OFF
        and document.get("effective_source") == RELAY_OFF
        and document.get("transport_open") is False
    )


def relay_desires(document: Mapping[str, Any] | None, source: str) -> bool:
    return bool(isinstance(document, Mapping) and document.get("source") == source)


__all__ = [
    "RELAY_LOCAL",
    "RELAY_OFF",
    "RELAY_WINDOWS",
    "desired_relay_source",
    "relay_desires",
    "relay_is_closed",
    "route_signature",
    "windows_runtime_ready",
]
