#!/usr/bin/env python3
"""Transport, port, and vocabulary contracts used by the native manager.

This module is deliberately free of Qt and of any I/O so the wire contract the
user interface claims to describe can be asserted directly in tests.  Every
endpoint here is either a loopback relay produced by a service that already
owns a device, or a LAN endpoint on the private five-slot router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360
PREVIEW_FPS = 30

SLOT_COUNT = 5
FIRST_SLOT_PORT = 10_000
SELECTED_STREAM_PORT = 10_010
ANDROID_NATIVE_PREVIEW_PORT = 11_004
MANAGER_RAW_PREVIEW_PORT = 11_001
LOCAL_PREVIEW_PORT = 11_000

WIRE_WIDTH = 1280
WIRE_HEIGHT = 720
WIRE_FPS = 30


def slot_input_port(slot: int) -> int:
    """Return the Windows listener port for one client slot."""
    if not 0 <= int(slot) < SLOT_COUNT:
        raise ValueError(f"slot must be between 0 and {SLOT_COUNT - 1}")
    return FIRST_SLOT_PORT + int(slot) * 2


def slot_return_port(slot: int) -> int:
    """The processed return is always the input port plus one."""
    return slot_input_port(slot) + 1


@dataclass(frozen=True)
class LocalRelay:
    """One loopback-only MPEG-TS endpoint on this machine."""

    port: int
    key: str
    title: str
    owner: str
    detail: str


# Ordered exactly as the deployed services allocate them.  The manager only
# ever reads 11001, 11003, 11004, and 11007; the rest are described so the
# System page can show the real contract instead of a summary of it.
ARCH_LOCAL_RELAYS: tuple[LocalRelay, ...] = (
    LocalRelay(
        11_000,
        "system_raw",
        "Sender raw fallback",
        "deep-live-cam-sender.service",
        "Raw capture copy consumed by the stable receiver when nothing "
        "processed is fresh.",
    ),
    LocalRelay(
        11_001,
        "manager_raw",
        "Manager raw preview",
        "deep-live-cam-sender.service",
        "Independent raw copy for this manager, so opening it cannot split "
        "the system mux input.",
    ),
    LocalRelay(
        11_002,
        "network_worker",
        "Windows transport input",
        "deep-live-cam-sender.service",
        "Private sender copy for the independently reconnecting Windows "
        "transport worker.",
    ),
    LocalRelay(
        11_003,
        "windows_preview",
        "Windows result preview",
        "deep-live-cam-receiver.service",
        "Receiver-owned relay of the selected Windows output for this manager.",
    ),
    LocalRelay(
        11_004,
        "phone_return",
        "Phone return preview",
        "deep-live-cam-phone-return-relay.service",
        "Exact encoded copy of the stream returned to the phone.",
    ),
    LocalRelay(
        11_005,
        "local_model_input",
        "Local processor input",
        "deep-live-cam-sender.service",
        "Dedicated camera copy delivered to the local Native-256 processor.",
    ),
    LocalRelay(
        11_006,
        "local_model_output",
        "Local processor output",
        "deep-live-cam-phone-processed",
        "Local Native-256 result consumed by the stable receiver.",
    ),
    LocalRelay(
        11_007,
        "local_preview",
        "Local result preview",
        "deep-live-cam-receiver.service",
        "Receiver-owned relay of the local processed output for this manager.",
    ),
    LocalRelay(
        11_008,
        "windows_phone_source",
        "Windows phone-return source",
        "deep-live-cam-receiver.service",
        "Dedicated encoded copy of the selected Windows stream consumed only "
        "by the exclusive phone-return relay.",
    ),
    LocalRelay(
        11_009,
        "local_phone_source",
        "Local phone-return source",
        "deep-live-cam-phone-processed.service",
        "Dedicated encoded local-model output consumed only by the exclusive "
        "phone-return relay.",
    ),
    LocalRelay(
        11_010,
        "prerecorded",
        "Prerecorded video input",
        "record_and_render.py stream",
        "Offline-rendered face-swap result streamed as an input source.",
    ),
)

RELAY_BY_PORT: dict[int, LocalRelay] = {
    relay.port: relay for relay in ARCH_LOCAL_RELAYS
}


@dataclass(frozen=True)
class SystemCameraPolicy:
    """One selectable policy for the stable /dev/deep-live-cam device."""

    key: str
    label: str
    summary: str
    order: tuple[str, ...]


# The order tuples mirror receiver.py SOURCE_PRIORITIES exactly.  Describing a
# fallback the receiver does not implement would be worse than showing none.
SYSTEM_CAMERA_POLICIES: tuple[SystemCameraPolicy, ...] = (
    SystemCameraPolicy(
        "local",
        "Local Native-256 first",
        "Local processed result, then the raw webcam if it goes stale.",
        ("local_processed", "local_prerecorded", "local_raw"),
    ),
    SystemCameraPolicy(
        "windows",
        "Windows result first",
        "Windows return, then the pulled selected stream, then the raw webcam.",
        ("processed_return", "selected_stream", "local_raw"),
    ),
    SystemCameraPolicy(
        "auto",
        "Best fresh processed source",
        "Local processed, Windows return, selected stream, then the raw webcam.",
        ("local_processed", "processed_return", "selected_stream", "local_raw"),
    ),
    SystemCameraPolicy(
        "prerecorded",
        "Prerecorded video",
        "Offline-rendered face-swap result, looping.",
        ("local_prerecorded",),
    ),
    SystemCameraPolicy(
        "raw",
        "Raw webcam only",
        "No processed source is used, even while one is live.",
        ("local_raw",),
    ),
)

POLICY_BY_KEY: dict[str, SystemCameraPolicy] = {
    policy.key: policy for policy in SYSTEM_CAMERA_POLICIES
}

ACTIVE_INPUT_LABELS: dict[str, str] = {
    "local_processed": "Local Native-256 result",
    "local_prerecorded": "Prerecorded video",
    "processed_return": "Windows return (direct)",
    "selected_stream": "Windows selected stream (pulled)",
    "local_raw": "Raw webcam",
}


def policy_label(key: Any) -> str:
    policy = POLICY_BY_KEY.get(str(key))
    return policy.label if policy is not None else f"unknown policy ({key})"


def active_input_label(key: Any) -> str:
    if key in (None, "", "waiting"):
        return "waiting for a source"
    return ACTIVE_INPUT_LABELS.get(str(key), str(key))


# Which engine a control actually reaches.  Every settings row in the UI is
# tagged with one of these, so "opacity" is never mistaken for a global value.
SCOPE_WINDOWS = "windows"
SCOPE_ARCH = "arch"
SCOPE_BOTH = "both"
SCOPE_COMPARISON = "comparison"
SCOPE_CAPTURE_OWNER = "capture-owner"

SCOPE_LABELS: dict[str, str] = {
    SCOPE_WINDOWS: "WINDOWS",
    SCOPE_ARCH: "ARCH",
    SCOPE_BOTH: "BOTH",
    SCOPE_COMPARISON: "VIEW ONLY",
    SCOPE_CAPTURE_OWNER: "CAMERA OWNER",
}

SCOPE_TOOLTIPS: dict[str, str] = {
    SCOPE_WINDOWS: (
        "Applies to the Windows processor. The same desired value is retained "
        "for the Arch processor and reconciled when either node is available."
    ),
    SCOPE_ARCH: (
        "Applies to the local native-256 NCNN/Vulkan processor on this machine."
    ),
    SCOPE_BOTH: (
        "One desired value is reconciled to both processor implementations."
    ),
    SCOPE_COMPARISON: (
        "Changes only this manager's comparison view. No camera, transport, or "
        "processor state is modified."
    ),
    SCOPE_CAPTURE_OWNER: (
        "Sent to the process that already owns this camera. The manager never "
        "opens the device, and the operating-system camera identity is kept."
    ),
}

TARGET_WINDOWS = "windows"
TARGET_ARCH = "arch"

PROCESSOR_TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        TARGET_WINDOWS,
        "Windows 11 remote",
        "INSwapper-128 ONNX on CUDA. Offline edits remain queued and reconcile "
        "when Windows returns.",
    ),
    (
        TARGET_ARCH,
        "This Arch workstation",
        "Native-256 semantic model on NCNN/Vulkan. Its current checkpoint is "
        "development/unqualified.",
    ),
)


@dataclass(frozen=True)
class CameraControlGroup:
    """A semantic bucket for capability-driven camera-owner controls."""

    key: str
    title: str
    detail: str
    keys: tuple[str, ...]


CAMERA_CONTROL_GROUPS: tuple[CameraControlGroup, ...] = (
    CameraControlGroup(
        "profile",
        "Profile and capture",
        "A named profile loads every value below at once. Resolution applies "
        "on the capture owner's next natural start after settings are saved; "
        "the manager never reopens an active input.",
        ("profile", "capture_size"),
    ),
    CameraControlGroup(
        "tone",
        "Tone and color",
        "Image values the capture owner applies in place, without reopening "
        "the device.",
        (
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "gamma",
            "gain",
            "sharpness",
            "backlight_compensation",
            "power_line_frequency",
        ),
    ),
    CameraControlGroup(
        "exposure",
        "Exposure and white balance",
        "Manual values stay as fallbacks while their automatic modes are on.",
        (
            "auto_exposure",
            "exposure_time_absolute",
            "exposure_dynamic_framerate",
            "exposure_compensation",
            "ae_lock",
            "auto_white_balance",
            "white_balance_temperature",
            "awb_lock",
        ),
    ),
    CameraControlGroup(
        "optics",
        "Orientation, lens and stabilization",
        "Advertised only by camera stacks that own a steerable sensor.",
        ("lens_facing", "rotation", "zoom_percent", "stabilization"),
    ),
)

OTHER_CAMERA_CONTROL_GROUP = CameraControlGroup(
    "other",
    "Other capabilities",
    "Advertised by this adapter but not part of a group this manager knows.",
    (),
)


def group_camera_controls(
    controls: Iterable[dict[str, Any]],
) -> list[tuple[CameraControlGroup, list[dict[str, Any]]]]:
    """Bucket adapter-advertised controls without hard-coding one device.

    Keys this manager has never seen still render, in a trailing group, so a
    new capture stack stays usable before this file learns about it.
    """
    remaining = list(controls)
    claimed: set[int] = set()
    grouped: list[tuple[CameraControlGroup, list[dict[str, Any]]]] = []
    for group in CAMERA_CONTROL_GROUPS:
        members: list[dict[str, Any]] = []
        for index, control in enumerate(remaining):
            if index in claimed or str(control.get("key")) not in group.keys:
                continue
            claimed.add(index)
            members.append(control)
        if members:
            grouped.append((group, members))
    leftovers = [
        control for index, control in enumerate(remaining) if index not in claimed
    ]
    if leftovers:
        grouped.append((OTHER_CAMERA_CONTROL_GROUP, leftovers))
    return grouped


def local_mpegts_preview_command(port: int) -> list[str]:
    """Decode one loopback-only MPEG-TS preview without opening a camera."""
    if not 1 <= int(port) <= 65_535:
        raise ValueError("preview port must be between 1 and 65535")
    source = (
        f"udp://127.0.0.1:{int(port)}?reuse=1&fifo_size=1000000&overrun_nonfatal=1"
    )
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0",
        "-probesize", "32768", "-f", "mpegts", "-i", source,
        "-map", "0:v:0", "-an",
        # Match the receiver's phone-accurate centre-crop fit so the preview
        # frames the source exactly as the stable camera publishes it, rather
        # than stretching a portrait or off-aspect source into the box.
        "-vf", f"scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:"
        f"force_original_aspect_ratio=increase:flags=fast_bilinear,"
        f"crop={PREVIEW_WIDTH}:{PREVIEW_HEIGHT},"
        f"fps={PREVIEW_FPS},format=rgb24",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]


def readable_age(value: Any) -> str:
    if value is None:
        return "never"
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "?"
    if age < 1:
        return f"{age * 1000:.0f} ms"
    return f"{age:.1f} s"


def integer(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def number(value: Any, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "-"
