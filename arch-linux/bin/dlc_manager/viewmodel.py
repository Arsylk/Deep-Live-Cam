#!/usr/bin/env python3
"""One normalized, immutable snapshot of everything the manager displays.

Widgets never interrogate raw service JSON.  A refresh assembles a
:class:`ManagerView` from sender, receiver, shadow, Android, native-processor,
and Windows documents, and each page renders from that value.  Keeping the
assembly pure means the page/ownership contract can be asserted off-screen
without a running Qt application, a camera, or a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

from pipeline_topology import (
    ROUTE_ANDROID,
    ROUTE_ARCH,
    PipelineTopology,
    infer_topology,
    stream_is_fresh,
)

from .contracts import (
    ACTIVE_INPUT_LABELS,
    POLICY_BY_KEY,
    SELECTED_STREAM_PORT,
    SLOT_COUNT,
    active_input_label,
    integer,
    policy_label,
    readable_age,
    slot_input_port,
    slot_return_port,
)
from .health import (
    android_native_phone_route_fresh,
    android_native_preview_fresh,
    android_native_route_title,
    android_native_webcam_route_fresh,
    phone_return_relay_preview_fresh,
    phone_return_relay_title,
)
from .desired_state import PROCESSOR_ARCH, PROCESSOR_WINDOWS


_STOPPED_STATES = ("inactive", "failed", "deactivating")

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"
_SEVERITY_RANK = {CRITICAL: 0, WARNING: 1, INFO: 2}

ROUTE_LOCAL_PHONE = "local-phone"
ROUTE_LOCAL_ARCH = "local-arch"
ROUTE_WINDOWS_ANDROID = "windows-android"
ROUTE_WINDOWS_ARCH = "windows-arch"
ROUTE_UNRESOLVED = "unresolved"

STREAM_RESULT = "result"
STREAM_RAW = "raw"


@dataclass(frozen=True)
class DecoderStats:
    """What one passive local decoder achieved since the last refresh."""

    running: bool = False
    fps: float = 0.0
    frames: int = 0
    dropped: int = 0
    age: float | None = None
    port: int = 0
    restarts: int = 0


@dataclass(frozen=True)
class Alert:
    """A problem with a named owner and a next step the user can take."""

    component: str
    message: str
    next_action: str
    severity: str = WARNING

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK.get(self.severity, 3)


@dataclass(frozen=True)
class StreamView:
    key: str
    title: str
    source: str
    model: str
    endpoint: str
    state: str
    state_text: str
    fps: float = 0.0
    frames: int = 0
    dropped: int = 0
    age: float | None = None
    delayed_ms: int = 0
    note: str = ""
    # False only when the current route genuinely has no such stream, which is
    # different from a reader that simply is not running yet.
    available: bool = True

    @property
    def delayed(self) -> bool:
        return self.delayed_ms > 0

    def metrics(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Decoded FPS", f"{self.fps:.1f}"),
            ("Frames", integer(self.frames)),
            ("Drops", integer(self.dropped)),
            ("Last frame", readable_age(self.age)),
        )


@dataclass(frozen=True)
class SlotView:
    slot: int
    device_id: str | None
    label: str
    stack: str
    input_port: int
    return_port: int
    return_host: str
    configured: bool
    enabled: bool
    selected: bool
    selectable: bool
    state: str
    state_text: str
    endpoint: str
    capability: str
    error: str | None = None

    @property
    def identity(self) -> str:
        return self.device_id or f"slot-{self.slot}"


@dataclass(frozen=True)
class NodeView:
    key: str
    title: str
    detail: str
    state: str
    state_text: str


@dataclass(frozen=True)
class RouteView:
    key: str
    badge: str
    state: str
    summary: str
    detail: str
    warning: str | None
    windows_bypassed: bool
    nodes: tuple[NodeView, ...] = ()

    @property
    def uses_windows(self) -> bool:
        return self.key in (ROUTE_WINDOWS_ANDROID, ROUTE_WINDOWS_ARCH)


@dataclass(frozen=True)
class SystemCameraView:
    configured_policy: str
    configured_label: str
    active_input: str
    active_label: str
    fallback: tuple[str, ...]
    devices: tuple[str, ...]
    state: str
    state_text: str
    detail: str
    identity_note: str


@dataclass(frozen=True)
class ProcessorView:
    windows_reachable: bool
    windows_state: str
    windows_state_text: str
    windows_mode: str
    windows_requested_model: str
    windows_active_model: str
    windows_detail: str
    local_running: bool
    local_model: str
    local_backend: str
    local_fps: float
    local_checkpoint: str
    local_detail: str
    identity_status: str
    identity_detail: str
    identity_verified: bool
    visual_effect_confirmed: bool
    checkpoint_qualified: bool
    selected_processor: str
    selected_face_detected: bool | None
    selected_face_swapped: bool | None
    selected_model: str
    selected_backend: str
    selected_render_resolution: int | None
    selected_processing_fps: float | None
    selected_runtime_reason: str
    selected_runtime_state: str


@dataclass(frozen=True)
class IdentityView:
    filename: str | None
    cache_path: str | None
    history_count: int
    windows_state: str
    windows_detail: str
    local_state: str
    local_detail: str


@dataclass(frozen=True)
class AndroidView:
    management_available: bool
    bridge_running: bool | None
    model: str
    host: str
    serial: str
    camera_id: str
    camera_published: bool
    summary: str
    state: str
    state_text: str


@dataclass(frozen=True)
class ServiceView:
    key: str
    unit: str
    title: str
    active_state: str
    unit_file_state: str
    state: str
    state_text: str


@dataclass(frozen=True)
class ManagerView:
    """The complete render input for one refresh."""

    generated_at: float
    route: RouteView
    system_camera: SystemCameraView
    slots: tuple[SlotView, ...]
    streams: Mapping[str, StreamView]
    processor: ProcessorView
    identity: IdentityView
    android: AndroidView
    services: tuple[ServiceView, ...]
    alerts: tuple[Alert, ...]
    registry_live: bool
    switching: bool
    selected_device_id: str | None
    windows_host: str
    selected_stream_port: int
    hosts: Mapping[str, str]
    capture_device: str
    virtual_devices: tuple[str, ...]
    shadow_ready: bool | None
    phone_return_live: bool = False
    phone_return_title: str = "PROCESSED RESULT SENT TO PHONE"
    host_metrics: Mapping[str, Any] = field(default_factory=dict)

    def stream(self, key: str) -> StreamView:
        return self.streams[key]

    def worst_alert(self) -> Alert | None:
        return self.alerts[0] if self.alerts else None

    def critical_alerts(self) -> tuple[Alert, ...]:
        return tuple(alert for alert in self.alerts if alert.severity == CRITICAL)


@dataclass(frozen=True)
class ViewInputs:
    """Everything one refresh reads, captured before any rendering happens."""

    config: Mapping[str, str] = field(default_factory=dict)
    sender: Mapping[str, Any] = field(default_factory=dict)
    receiver: Mapping[str, Any] = field(default_factory=dict)
    shadow: Mapping[str, Any] = field(default_factory=dict)
    native_health: Mapping[str, Any] = field(default_factory=dict)
    native_health_age: float | None = None
    native_preview_port: int = 11_004
    phone_relay_health: Mapping[str, Any] = field(default_factory=dict)
    phone_relay_health_age: float | None = None
    android_status: Mapping[str, Any] = field(default_factory=dict)
    android_error: str | None = None
    windows_health: Mapping[str, Any] | None = None
    windows_error: str | None = None
    windows_config: Mapping[str, Any] = field(default_factory=dict)
    windows_devices: Mapping[str, Any] = field(default_factory=dict)
    registry_live: bool = False
    selection_in_flight: bool = False
    services: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    result_stream: DecoderStats = DecoderStats()
    raw_stream: DecoderStats = DecoderStats()
    comparison_delay_ms: int = 0
    identity_filename: str | None = None
    identity_cache_path: str | None = None
    # Content hash of the exact source bytes selected in local history.  This
    # is intentionally separate from ``source_configured``: the latter proves
    # only that Windows has *a* picture, not that it has this picture.
    identity_identifier: str | None = None
    identity_used_at: float | None = None
    identity_history_count: int = 0
    capture_device: str = ""
    virtual_devices: Sequence[str] = ()
    device_nodes_present: bool | None = None
    host_metrics: Mapping[str, Any] = field(default_factory=dict)
    now: float | None = None

    def host(self, key: str, default: str) -> str:
        return str(self.config.get(key, default) or default)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _service_view(
    key: str, unit: str, title: str, values: Mapping[str, str]
) -> ServiceView:
    active = str(values.get("ActiveState", "unknown"))
    boot = str(values.get("UnitFileState", "unknown"))
    presentation = {
        "active": ("running", "RUNNING"),
        "activating": ("working", "STARTING"),
        "deactivating": ("working", "STOPPING"),
        "failed": ("failed", "FAILED"),
        "inactive": ("stopped", "STOPPED"),
    }
    state, text = presentation.get(active, ("unknown", "UNKNOWN"))
    boot_text = "boot enabled" if boot == "enabled" else f"boot {boot}"
    return ServiceView(
        key=key,
        unit=unit,
        title=title,
        active_state=active,
        unit_file_state=boot,
        state=state,
        state_text=f"{text} · {boot_text}",
    )


def _android_view(inputs: ViewInputs, topology: PipelineTopology) -> AndroidView:
    values = _mapping(inputs.android_status)
    available = values.get("available") is True
    installed = values.get("app_installed") is True
    bridge = values.get("bridge_running") is True
    camera_published = values.get("camera_published") is True
    host = str(values.get("host") or inputs.host("ANDROID_HOST", "192.168.1.12"))
    camera_id = inputs.host("ANDROID_CAMERA_ID", "120")
    if not available:
        state, text = "unknown", "MANAGEMENT LINK OFFLINE"
        reason = inputs.android_error or "phone not connected"
        summary = (
            f"ADB management is unavailable ({reason}). LAN video does not use "
            "the management link and can continue without it."
        )
    elif bridge and camera_published:
        state, text = "running", "BRIDGE RUNNING"
        summary = (
            f"Camera2 output {camera_id} is published and the return receiver "
            "is running."
        )
    elif bridge:
        state, text = "working", "BRIDGE STARTING"
        summary = f"Bridge is running; Camera2 output {camera_id} is not confirmed yet."
    elif installed:
        state, text = "stopped", "BRIDGE STOPPED"
        summary = "The companion app is installed but its camera bridge is stopped."
    else:
        state, text = "failed", "COMPANION APP MISSING"
        summary = "ADB is connected but the companion app is not installed."
    if topology.selected == ROUTE_ANDROID and not bridge and available:
        state = "failed"
    return AndroidView(
        management_available=available,
        bridge_running=None if not available else bridge,
        model=str(values.get("model") or inputs.host("ANDROID_MODEL", "Android phone")),
        host=host,
        serial=str(values.get("serial") or inputs.host("ANDROID_ADB_SERIAL", "")),
        camera_id=camera_id,
        camera_published=camera_published,
        summary=summary,
        state=state,
        state_text=text,
    )


def _route_view(
    inputs: ViewInputs,
    topology: PipelineTopology,
    *,
    phone_route: bool,
    webcam_route: bool,
    android: AndroidView,
) -> RouteView:
    windows = _mapping(inputs.windows_health)
    windows_input = _mapping(windows.get("input"))
    windows_output = _mapping(windows.get("output"))
    native = _mapping(inputs.native_health)
    native_processing = _mapping(native.get("processing"))
    native_input = _mapping(native.get("input"))
    native_return = _mapping(native.get("return"))
    input_fresh = stream_is_fresh(windows_input)
    output_fresh = stream_is_fresh(windows_output, 4.0)
    windows_healthy = windows.get("healthy") is True
    arch_host = inputs.host("ARCH_HOST", "192.168.1.11")
    windows_host = inputs.host("WINDOWS_HOST", "192.168.1.35")
    model = str(native_processing.get("model", "unknown"))
    backend = str(native_processing.get("backend", "unknown"))

    if phone_route or webcam_route:
        key = ROUTE_LOCAL_PHONE if phone_route else ROUTE_LOCAL_ARCH
        badge = (
            "PHONE FRONT → ARCH → CAMERA 120"
            if phone_route
            else "ARCH WEBCAM → LOCAL MODEL → PHONE"
        )
        summary = (
            f"{'Phone front sensor' if phone_route else 'Arch capture owner'} → "
            f"{model}/{backend} → phone Camera2 {android.camera_id}"
        )
        detail = (
            f"Input {native_input.get('url', 'unavailable')} · return "
            f"{native_return.get('url', 'unavailable')}"
        )
        warning = (
            "This local route bypasses Windows. Both camera identities stay "
            "registered while it runs."
        )
        return RouteView(
            key=key,
            badge=badge,
            state="running",
            summary=summary,
            detail=detail,
            warning=warning,
            windows_bypassed=True,
            nodes=(
                NodeView(
                    "android",
                    f"ANDROID · {android.host}",
                    f"stable Camera2 {android.camera_id} raw/processed mux",
                    "running" if phone_route else "running",
                    "FRONT SENSOR + CAMERA 120" if phone_route else "CAMERA 120 RETURN",
                ),
                NodeView(
                    "windows",
                    f"WINDOWS · {windows_host}",
                    f"one selected processor · shared pull :{SELECTED_STREAM_PORT}",
                    "stopped",
                    "BYPASSED FOR LOCAL ROUTE",
                ),
                NodeView(
                    "arch",
                    f"ARCH · {arch_host}",
                    "slot 1 · 10002 → 10003 · stable /dev/deep-live-cam (video42)",
                    "running",
                    f"PROCESSING · {float(native_processing.get('fps') or 0):.1f} FPS",
                ),
            ),
        )

    if topology.conflict or topology.mismatch:
        key, badge, state = ROUTE_UNRESOLVED, "INPUT ROUTE CONFLICT", "failed"
    elif topology.selected == ROUTE_ANDROID:
        key, badge = ROUTE_WINDOWS_ANDROID, "ANDROID SLOT 0 → WINDOWS"
        state = "running" if input_fresh and output_fresh else "working"
    elif topology.selected == ROUTE_ARCH:
        key, badge = ROUTE_WINDOWS_ARCH, "ARCH SLOT 1 → WINDOWS"
        state = "running" if input_fresh and output_fresh else "working"
    else:
        key, badge, state = ROUTE_UNRESOLVED, "ROUTE UNCONFIRMED", "working"

    return_target = topology.return_host or "unavailable"
    return_port = windows_output.get("port") or "?"
    detail = (
        f"Paired return {return_target}:{return_port} · selected stream "
        f"{windows_host}:{inputs.config.get('WINDOWS_SELECTED_STREAM_PORT', SELECTED_STREAM_PORT)}"
    )

    if android.bridge_running is True and topology.selected == ROUTE_ANDROID and input_fresh:
        android_state, android_text = "running", "ACTIVE · STREAMING"
    elif topology.selected == ROUTE_ANDROID and android.bridge_running is True:
        android_state, android_text = "working", "ACTIVE · WAITING"
    elif topology.selected == ROUTE_ANDROID:
        android_state = "failed" if android.management_available else "working"
        android_text = (
            "SELECTED · BRIDGE STOPPED"
            if android.management_available
            else "SELECTED · MANAGEMENT OFFLINE"
        )
    elif android.bridge_running is True:
        android_state, android_text = "stopped", "READY · UNSELECTED"
    else:
        android_state, android_text = "stopped", "STANDBY"

    if windows_healthy and input_fresh:
        windows_state, windows_text = "running", "PROCESSING"
    elif windows_healthy:
        windows_state, windows_text = "working", "ONLINE · WAITING"
    else:
        windows_state, windows_text = "failed", "UNREACHABLE"

    if topology.conflict or topology.mismatch:
        arch_state, arch_text = "failed", "COMPETING SENDER"
    elif topology.selected == ROUTE_ARCH and input_fresh:
        arch_state, arch_text = "running", "ACTIVE · STREAMING"
    elif topology.selected == ROUTE_ARCH:
        arch_state, arch_text = "working", "SELECTED · WAITING"
    else:
        arch_state, arch_text = "stopped", "CONTROL · LOCAL FALLBACK"

    return RouteView(
        key=key,
        badge=badge,
        state=state,
        summary=topology.summary,
        detail=detail,
        warning=topology.warning,
        windows_bypassed=False,
        nodes=(
            NodeView(
                "android",
                f"ANDROID · {android.host}",
                f"slot 0 · 10000 → 10001 · stable Camera2 {android.camera_id}",
                android_state,
                android_text,
            ),
            NodeView(
                "windows",
                f"WINDOWS · {windows_host}",
                f"one selected processor · shared pull :{SELECTED_STREAM_PORT}",
                windows_state,
                windows_text,
            ),
            NodeView(
                "arch",
                f"ARCH · {arch_host}",
                "slot 1 · 10002 → 10003 · stable /dev/deep-live-cam (video42)",
                arch_state,
                arch_text,
            ),
        ),
    )


def _system_camera_view(inputs: ViewInputs, local_route: bool) -> SystemCameraView:
    receiver = _mapping(inputs.receiver)
    configured = str(
        receiver.get("source_mode") or inputs.config.get("RECEIVER_SOURCE", "auto")
    )
    active = receiver.get("active_input")
    policy = POLICY_BY_KEY.get(configured)
    fallback = tuple(
        ACTIVE_INPUT_LABELS.get(name, name) for name in (policy.order if policy else ())
    )
    devices = tuple(str(device) for device in inputs.virtual_devices)
    status = str(receiver.get("status", "unknown"))
    if status == "streaming" and active:
        state, text = "running", "STREAMING"
    elif status in ("starting", "waiting"):
        state, text = "working", status.upper()
    elif status == "unknown":
        state, text = "unknown", "NOT REPORTING YET"
    else:
        state, text = "working", status.upper()
    detail = (
        f"Policy {policy_label(configured)}; the receiver is currently reading "
        f"{active_input_label(active)}."
    )
    if local_route and active == "local_processed":
        detail += " The local processed result is also being returned to the phone."
    return SystemCameraView(
        configured_policy=configured,
        configured_label=policy_label(configured),
        active_input=str(active) if active else "waiting",
        active_label=active_input_label(active),
        fallback=fallback,
        devices=devices,
        state=state,
        state_text=text,
        detail=detail,
        identity_note=(
            "Selecting a policy swaps already-owned frame queues. The sink, "
            f"{' and '.join(devices) if devices else 'the camera nodes'}, the "
            "capture owner, and the 1280×720 30 FPS contract are not restarted."
        ),
    )


def _slot_views(inputs: ViewInputs) -> tuple[tuple[SlotView, ...], str | None, bool]:
    document = _mapping(inputs.windows_devices)
    runtime = _mapping(document.get("runtime"))
    switching = bool(runtime.get("switching")) or inputs.selection_in_flight
    selected_id = document.get("selected_device_id")
    selected_id = str(selected_id) if selected_id else None
    raw_slots = document.get("slots")
    by_slot: dict[int, dict[str, Any]] = {}
    if isinstance(raw_slots, list):
        for raw in raw_slots:
            if isinstance(raw, Mapping):
                try:
                    by_slot[int(raw.get("slot"))] = dict(raw)
                except (TypeError, ValueError):
                    continue

    views: list[SlotView] = []
    for index in range(SLOT_COUNT):
        raw = by_slot.get(index, {})
        device_id = raw.get("device_id")
        device_id = str(device_id) if device_id else None
        configured = bool(device_id and raw.get("configured", True))
        enabled = bool(raw.get("enabled")) and configured
        selected = bool(configured and device_id == selected_id)
        slot_runtime = _mapping(raw.get("runtime"))
        slot_input = _mapping(slot_runtime.get("input"))
        slot_return = _mapping(slot_runtime.get("return"))
        error = raw.get("error") or (runtime.get("last_switch_error") if selected else None)

        if not configured:
            state, text = "unavailable", "NOT ASSIGNED"
        elif not enabled:
            state, text = "unavailable", "DISABLED"
        elif selected and switching:
            state, text = "switching", "SWITCHING…"
        elif selected and slot_input.get("streaming") and slot_return.get("streaming"):
            state, text = "active", "ACTIVE · LIVE"
        elif selected and slot_input.get("streaming"):
            state, text = "working", "ACTIVE · PROCESSING"
        elif selected and inputs.registry_live:
            state, text = "selected", "ACTIVE · WAITING FOR FRAMES"
        elif selected:
            state, text = "selected", "ACTIVE · LAST KNOWN"
        elif inputs.registry_live:
            state, text = "ready", "READY · LOCAL FALLBACK"
        else:
            state, text = "ready", "OFFLINE REGISTRY COPY"
        if selected and inputs.registry_live and not switching:
            age = slot_input.get("last_frame_age")
            if age is not None and not stream_is_fresh(slot_input):
                state, text = "stale", f"ACTIVE · STALE {readable_age(age)}"
        if error:
            state, text = "error", "SELECTION ERROR"

        input_port = int(raw.get("input_port") or slot_input_port(index))
        return_port = int(raw.get("return_port") or slot_return_port(index))
        host = str(raw.get("return_host") or "unassigned")
        views.append(
            SlotView(
                slot=index,
                device_id=device_id,
                label=str(raw.get("label") or device_id or "Unassigned"),
                stack=str(raw.get("stack") or "generic-srt"),
                input_port=input_port,
                return_port=return_port,
                return_host=host,
                configured=configured,
                enabled=enabled,
                selected=selected,
                selectable=bool(enabled and inputs.registry_live and not switching),
                state=state,
                state_text=text,
                endpoint=f"SRT {input_port} → Windows → {host}:{return_port}",
                capability={
                    "arch-v4l2": "V4L2 capture owner · full camera controls",
                    "android-camera2": "Camera2 owner · lens, exposure, stabilization",
                    "generic-srt": "SRT client · no camera-control adapter",
                }.get(str(raw.get("stack") or "generic-srt"), "SRT client"),
                error=str(error) if error else None,
            )
        )
    return tuple(views), selected_id, switching


def _stream_views(
    inputs: ViewInputs, route: RouteView, system_camera: SystemCameraView
) -> dict[str, StreamView]:
    native_processing = _mapping(_mapping(inputs.native_health).get("processing"))
    windows_processing = _mapping(_mapping(inputs.windows_health).get("processing"))
    local_route = route.windows_bypassed
    windows_host = inputs.host("WINDOWS_HOST", "192.168.1.35")
    selected_port = inputs.config.get(
        "WINDOWS_SELECTED_STREAM_PORT", str(SELECTED_STREAM_PORT)
    )

    result = inputs.result_stream
    if local_route:
        title = "LOCAL NATIVE-256 OUTPUT"
        source = "Local processed result (receiver relay)"
        model = (
            f"{native_processing.get('model', 'unknown')}/"
            f"{native_processing.get('backend', 'unknown')}"
        )
        note = (
            "Exact encoded copy of the local model output. Windows is bypassed "
            "on this route."
        )
    else:
        title = "SELECTED WINDOWS STREAM"
        source = "Windows selected stream (receiver relay)"
        model = (
            f"{windows_processing.get('active_swapper_model', 'not-loaded')}/"
            f"{windows_processing.get('active_swapper_backend', 'not-loaded')}"
        )
        note = (
            f"Pulled from {windows_host}:{selected_port} by the receiver; this "
            "manager reads only the local relay."
        )
    result_view = StreamView(
        key=STREAM_RESULT,
        title=title,
        source=source,
        model=model,
        endpoint=f"local MPEG-TS UDP 127.0.0.1:{result.port}",
        state=_stream_state(result),
        state_text=_stream_state_text(result),
        fps=result.fps,
        frames=result.frames,
        dropped=result.dropped,
        age=result.age,
        note=note,
    )

    raw = inputs.raw_stream
    if route.key == ROUTE_WINDOWS_ANDROID:
        raw_view = StreamView(
            key=STREAM_RAW,
            title="ARCH RAW COMPARISON",
            source="Not available on the Android slot",
            model="not processed",
            endpoint=f"local MPEG-TS UDP 127.0.0.1:{raw.port}",
            state="off",
            state_text="UNAVAILABLE",
            note=(
                "Phone frames travel directly to Windows. This manager never "
                "opens a phone or local camera to synthesize a comparison."
            ),
            available=False,
        )
    else:
        raw_view = StreamView(
            key=STREAM_RAW,
            title="ARCH RAW COMPARISON",
            source="Arch capture owner diagnostic copy",
            model="not processed",
            endpoint=f"local MPEG-TS UDP 127.0.0.1:{raw.port}",
            state=_stream_state(raw),
            state_text=_stream_state_text(raw),
            fps=raw.fps,
            frames=raw.frames,
            dropped=raw.dropped,
            age=raw.age,
            delayed_ms=max(0, int(inputs.comparison_delay_ms)),
            note=(
                "Owner-produced copy on a dedicated relay. Delay affects this "
                "view only."
            ),
        )
    _ = system_camera
    return {STREAM_RESULT: result_view, STREAM_RAW: raw_view}


def _stream_state(stats: DecoderStats) -> str:
    if not stats.running:
        return "off"
    if stats.age is None:
        return "waiting"
    if stats.age <= 2.0:
        return "live"
    return "stalled"


def _stream_state_text(stats: DecoderStats) -> str:
    return {
        "off": "READER STOPPED",
        "waiting": "WAITING FOR FRAMES",
        "live": "LIVE",
        "stalled": "STALLED",
    }[_stream_state(stats)]


def _processor_view(inputs: ViewInputs, local_route: bool) -> ProcessorView:
    windows = _mapping(inputs.windows_health)
    windows_processing = _mapping(windows.get("processing"))
    native = _mapping(inputs.native_health)
    native_processing = _mapping(native.get("processing"))
    identity = _mapping(native_processing.get("identity_swap"))
    windows_tracking = _mapping(windows_processing.get("tracking"))
    windows_quality = _mapping(windows_processing.get("quality"))
    reachable = bool(windows) and windows.get("healthy") is True
    if reachable:
        state, text = "running", str(windows.get("state", "running")).upper()
    elif windows:
        state, text = "failed", "UNHEALTHY"
    else:
        state, text = "unknown", "UNREACHABLE"
    requested = str(
        windows_processing.get("swapper_model")
        or inputs.windows_config.get("swapper_model", "auto")
    )
    active_model = str(windows_processing.get("active_swapper_model", "not-loaded"))
    active_backend = str(windows_processing.get("active_swapper_backend", "not-loaded"))
    try:
        windows_render_resolution = int(
            windows_processing.get("active_swapper_resolution") or 0
        )
    except (TypeError, ValueError):
        windows_render_resolution = 0
    if windows_render_resolution <= 0:
        windows_render_resolution = 0
    checkpoint = str(native_processing.get("quality_status", "unknown"))

    windows_face_detected: bool | None = None
    if "active" in windows_tracking:
        windows_face_detected = bool(windows_tracking.get("active"))
    elif any(
        key in windows_tracking for key in ("valid_detections", "raw_detections")
    ):
        windows_face_detected = bool(
            int(windows_tracking.get("valid_detections", 0) or 0)
            or int(windows_tracking.get("raw_detections", 0) or 0)
        )
    windows_face_swapped: bool | None = (
        bool(windows_quality.get("swap_applied"))
        if "swap_applied" in windows_quality
        else False
        if str(windows_processing.get("mode", "")) == "passthrough"
        else None
    )
    arch_face_detected = (
        bool(identity.get("face_measurable"))
        if "face_measurable" in identity
        else None
    )
    arch_face_invoked = (
        bool(identity.get("attempted")) if "attempted" in identity else None
    )
    # Invocation is not proof of a swap, especially for the development
    # native-256 checkpoint. Only the processor's conservative visual-effect
    # evidence may drive the user-facing "Face swapped" answer.
    arch_face_swapped = (
        bool(identity.get("visual_effect_confirmed"))
        if "visual_effect_confirmed" in identity
        else None
    )

    def optional_fps(values: Mapping[str, Any]) -> float | None:
        if "fps" not in values:
            return None
        try:
            return max(0.0, float(values["fps"]))
        except (TypeError, ValueError):
            return None

    if local_route:
        selected_processor = PROCESSOR_ARCH
        selected_face_detected = arch_face_detected
        selected_face_swapped = arch_face_swapped
        selected_model = str(native_processing.get("model", "not reported"))
        selected_backend = str(native_processing.get("backend", "not reported"))
        try:
            selected_render_resolution = int(
                native_processing.get("active_swapper_resolution")
                or native_processing.get("native_resolution")
                or 0
            )
        except (TypeError, ValueError):
            selected_render_resolution = 0
        if selected_render_resolution <= 0:
            selected_render_resolution = None
        selected_fps = optional_fps(native_processing)
        native_error = native_processing.get("last_error")
        if native_error:
            selected_reason = str(native_error)
            selected_runtime_state = "failed"
        elif not native or native.get("state") != "running":
            selected_reason = "Arch processor health is unavailable."
            selected_runtime_state = "unknown"
        else:
            detail = str(
                identity.get("detail")
                or "Face-swap evidence is not reported by Arch health yet."
            )
            selected_reason = (
                f"Swap invoked but not verified: {detail}"
                if arch_face_invoked is True and arch_face_swapped is not True
                else detail
            )
            selected_runtime_state = (
                "running"
                if arch_face_swapped is True
                else "waiting"
                if arch_face_detected is False or arch_face_swapped is False
                else "working"
            )
    else:
        selected_processor = PROCESSOR_WINDOWS
        selected_face_detected = windows_face_detected
        selected_face_swapped = windows_face_swapped
        selected_model = (
            active_model if active_model != "not-loaded" else "not loaded"
        )
        selected_backend = (
            active_backend if active_backend != "not-loaded" else "not loaded"
        )
        selected_render_resolution = windows_render_resolution or None
        selected_fps = optional_fps(windows_processing)
        windows_runtime_error = windows_processing.get("last_error")
        if windows_runtime_error:
            selected_reason = str(windows_runtime_error)
            selected_runtime_state = "failed"
        elif not reachable:
            selected_reason = str(
                inputs.windows_error or "Windows processor health is unavailable."
            )
            selected_runtime_state = "unknown"
        elif str(windows_processing.get("mode", "")) == "passthrough":
            selected_reason = "Passthrough is active; face swapping is disabled."
            selected_runtime_state = "stopped"
        elif windows_face_swapped is True:
            selected_reason = "Health reports a swap applied to the current frame."
            selected_runtime_state = "running"
        elif windows_face_detected is False:
            selected_reason = "Health reports no currently tracked face."
            selected_runtime_state = "waiting"
        elif windows_face_detected is True and windows_face_swapped is False:
            selected_reason = (
                "A face is tracked; health reports no swap applied to the current frame."
            )
            selected_runtime_state = "waiting"
        else:
            selected_reason = "Face detection or swap evidence is not reported yet."
            selected_runtime_state = "working"
    return ProcessorView(
        windows_reachable=reachable,
        windows_state=state,
        windows_state_text=text,
        windows_mode=str(
            windows_processing.get("mode")
            or inputs.windows_config.get("processing_mode", "unknown")
        ),
        windows_requested_model=requested,
        windows_active_model=(
            "not loaded"
            if active_model == "not-loaded"
            else f"{active_model} via {active_backend}"
        ),
        windows_detail=(
            inputs.windows_error or "Configuration is synchronized."
            if not reachable
            else "Configuration is synchronized."
        ),
        local_running=bool(local_route or native.get("state") == "running"),
        local_model=str(native_processing.get("model", "unknown")),
        local_backend=str(native_processing.get("backend", "unknown")),
        local_fps=float(native_processing.get("fps") or 0.0),
        local_checkpoint=checkpoint,
        local_detail=(
            "Development checkpoint: transport is live but identity replacement "
            "is not qualified."
            if checkpoint == "development"
            else "Production checkpoint."
            if checkpoint in ("production", "qualified")
            else f"Checkpoint status {checkpoint}."
        ),
        identity_status=str(identity.get("status", "unknown")),
        identity_detail=str(identity.get("detail", "not reported")),
        identity_verified=bool(identity.get("identity_change_verified")),
        visual_effect_confirmed=bool(identity.get("visual_effect_confirmed")),
        checkpoint_qualified=checkpoint in ("production", "qualified"),
        selected_processor=selected_processor,
        selected_face_detected=selected_face_detected,
        selected_face_swapped=selected_face_swapped,
        selected_model=selected_model,
        selected_backend=selected_backend,
        selected_render_resolution=selected_render_resolution,
        selected_processing_fps=selected_fps,
        selected_runtime_reason=selected_reason,
        selected_runtime_state=selected_runtime_state,
    )


def _identity_view(inputs: ViewInputs, processor: ProcessorView) -> IdentityView:
    configured = bool(inputs.windows_config.get("source_configured"))
    filename = inputs.identity_filename
    cache_path = inputs.identity_cache_path
    desired_identifier = inputs.identity_identifier
    remote_reports_identifier = "source_identifier" in inputs.windows_config
    remote_identifier = inputs.windows_config.get("source_identifier")
    if not processor.windows_reachable:
        windows_state = "unknown"
        windows_detail = (
            "The Windows processor is unreachable, so the applied picture "
            "cannot be checked from here."
        )
    elif (
        configured
        and filename
        and desired_identifier
        and remote_identifier == desired_identifier
    ):
        windows_state = "applied"
        windows_detail = (
            "Windows verified the exact content hash of this selected picture."
        )
    elif filename and desired_identifier and remote_reports_identifier:
        windows_state = "pending"
        windows_detail = (
            "Windows does not yet report this selected picture. The manager "
            "will keep the content-addressed upload queued until it matches."
        )
    elif configured:
        windows_state = "unverified"
        windows_detail = (
            "Windows has a source picture, but this service build did not prove "
            "that it is the selected local picture by content hash."
        )
    else:
        windows_state = "missing"
        windows_detail = "No source picture is active on the Windows processor."

    native = _mapping(inputs.native_health)
    if native.get("state") != "running":
        local_state = "unavailable"
        local_detail = "The Arch Native-256 service is not running."
    else:
        started_at = None
        try:
            started_at = float(inputs.now or time.time()) - float(
                native.get("uptime_seconds")
            )
        except (TypeError, ValueError):
            started_at = None
        if (
            started_at is not None
            and inputs.identity_used_at is not None
            and inputs.identity_used_at > started_at
        ):
            local_state = "applying"
            local_detail = (
                "This picture was selected after the local service started. "
                "The hot-control channel is applying the new content-addressed source."
            )
        else:
            local_state = "unconfirmed"
            local_detail = (
                "The local processor has not yet reported enough evidence to "
                "confirm which identity is visible in output frames."
            )
    return IdentityView(
        filename=filename,
        cache_path=cache_path,
        history_count=int(inputs.identity_history_count),
        windows_state=windows_state,
        windows_detail=windows_detail,
        local_state=local_state,
        local_detail=local_detail,
    )


def _alerts(
    inputs: ViewInputs,
    route: RouteView,
    processor: ProcessorView,
    system_camera: SystemCameraView,
    android: AndroidView,
    services: Sequence[ServiceView],
    shadow_ready: bool | None,
) -> tuple[Alert, ...]:
    alerts: list[Alert] = []
    service_by_key = {service.key: service for service in services}

    if not processor.windows_reachable and not route.windows_bypassed:
        host = inputs.host("WINDOWS_HOST", "192.168.1.35")
        alerts.append(
            Alert(
                f"Windows processor {host}:8090",
                inputs.windows_error or "not reachable",
                "Check the Windows service and the LAN link. The local "
                "camera route keeps running meanwhile.",
                CRITICAL,
            )
        )

    sender = service_by_key.get("sender")
    if sender is not None and sender.active_state in _STOPPED_STATES:
        alerts.append(
            Alert(
                "Physical capture owner (deep-live-cam-sender.service)",
                f"unit is {sender.active_state}",
                "Start it from a terminal with systemctl. This manager never "
                "starts or stops a capture service.",
                CRITICAL,
            )
        )
    receiver = service_by_key.get("receiver")
    if receiver is not None and receiver.active_state in _STOPPED_STATES:
        alerts.append(
            Alert(
                "Stable system camera (deep-live-cam-receiver.service)",
                f"unit is {receiver.active_state}",
                "Start it from a terminal with systemctl; /dev/deep-live-cam "
                "has no writer until it runs.",
                CRITICAL,
            )
        )

    if shadow_ready is False:
        alerts.append(
            Alert(
                "Camera device mapping",
                "the preserved source or a public camera node is missing",
                "Install the clean one-node v4l2loopback layout, then perform "
                "one controlled reboot. The manager will not disrupt an open "
                "camera session to repair legacy nodes in place.",
                CRITICAL,
            )
        )

    if route.warning:
        alerts.append(
            Alert(
                "Camera route",
                route.warning,
                "Confirm which slot Windows should process on the Cameras and "
                "routing page.",
                WARNING if not route.state == "failed" else CRITICAL,
            )
        )

    if system_camera.state not in ("running", "working"):
        alerts.append(
            Alert(
                "Stable system camera",
                "the receiver has not published its state yet"
                if system_camera.state == "unknown"
                else f"the receiver reports {system_camera.state_text.lower()}",
                "Open System and logs for the receiver snapshot; camera "
                "identities stay registered either way.",
                WARNING,
            )
        )

    if processor.local_running and processor.local_checkpoint == "development":
        alerts.append(
            Alert(
                "Local processor checkpoint",
                "the loaded checkpoint is development-only",
                "Treat the visible result as transport evidence, not as a "
                "qualified identity replacement.",
                WARNING,
            )
        )

    if route.key == ROUTE_WINDOWS_ANDROID and android.bridge_running is False:
        alerts.append(
            Alert(
                "Android camera bridge",
                "the phone bridge is stopped while its slot is selected",
                "Start the companion bridge on the phone, or select another "
                "slot on the Cameras and routing page.",
                WARNING,
            )
        )
    if inputs.android_error and not android.management_available:
        alerts.append(
            Alert(
                "Android management link",
                str(inputs.android_error),
                "Only status reporting is affected; LAN video does not use the "
                "ADB link.",
                INFO,
            )
        )

    alerts.sort(key=lambda alert: alert.rank)
    return tuple(alerts)


def build_view(inputs: ViewInputs) -> ManagerView:
    """Assemble one immutable render input from already-collected documents."""
    now = float(inputs.now if inputs.now is not None else time.time())
    services = (
        _service_view(
            "sender",
            "deep-live-cam-sender.service",
            "Physical capture owner",
            _mapping(inputs.services.get("sender")),
        ),
        _service_view(
            "receiver",
            "deep-live-cam-receiver.service",
            "Stable system-camera writer",
            _mapping(inputs.services.get("receiver")),
        ),
        _service_view(
            "local_processor",
            "deep-live-cam-phone-processed.service",
            "Arch Native-256 standby processor",
            _mapping(inputs.services.get("local_processor")),
        ),
        _service_view(
            "phone_relay",
            "deep-live-cam-phone-return-relay.service",
            "Exclusive Android return relay",
            _mapping(inputs.services.get("phone_relay")),
        ),
    )
    topology = infer_topology(
        dict(inputs.windows_health) if inputs.windows_health else None,
        arch_host=inputs.host("ARCH_HOST", "192.168.1.11"),
        android_host=inputs.host("ANDROID_HOST", "192.168.1.12"),
        arch_sender_active=services[0].active_state == "active",
        android_sender_active=_android_sender_active(inputs),
    )
    relay_live = phone_return_relay_preview_fresh(
        dict(inputs.phone_relay_health),
        health_age=inputs.phone_relay_health_age,
        expected_port=inputs.native_preview_port,
    )
    relay_source = inputs.phone_relay_health.get("source")
    native_route = inputs.native_health.get("route")
    legacy_phone_return = bool(
        not inputs.phone_relay_health
        and android_native_preview_fresh(
            dict(inputs.native_health),
            health_age=inputs.native_health_age,
            expected_port=inputs.native_preview_port,
        )
    )
    phone_route = bool(
        relay_live
        and relay_source == "local"
        and native_route == "android-camera-processed-to-android"
    ) or bool(
        legacy_phone_return
        and android_native_phone_route_fresh(
            dict(inputs.native_health),
            health_age=inputs.native_health_age,
            expected_port=inputs.native_preview_port,
        )
    )
    webcam_route = bool(
        relay_live
        and relay_source == "local"
        and native_route == "arch-webcam-processed-to-android"
    ) or bool(
        legacy_phone_return
        and android_native_webcam_route_fresh(
            dict(inputs.native_health),
            health_age=inputs.native_health_age,
            expected_port=inputs.native_preview_port,
        )
    )
    android = _android_view(inputs, topology)
    route = _route_view(
        inputs,
        topology,
        phone_route=phone_route,
        webcam_route=webcam_route,
        android=android,
    )
    system_camera = _system_camera_view(inputs, route.windows_bypassed)
    slots, selected_device_id, switching = _slot_views(inputs)
    streams = _stream_views(inputs, route, system_camera)
    processor = _processor_view(inputs, route.windows_bypassed)
    identity = _identity_view(inputs, processor)
    shadow = _mapping(inputs.shadow)
    if not shadow and inputs.device_nodes_present is None:
        # Nothing has reported yet; an unknown mapping is not a fault.
        shadow_ready: bool | None = None
    else:
        shadow_ready = bool(
            (not shadow or shadow.get("status") == "shadowed")
            and inputs.device_nodes_present is not False
        )
    alerts = _alerts(
        inputs, route, processor, system_camera, android, services, shadow_ready
    )
    return ManagerView(
        generated_at=now,
        route=route,
        system_camera=system_camera,
        slots=slots,
        streams=streams,
        processor=processor,
        identity=identity,
        android=android,
        services=services,
        alerts=alerts,
        registry_live=bool(inputs.registry_live),
        switching=switching,
        selected_device_id=selected_device_id,
        windows_host=inputs.host("WINDOWS_HOST", "192.168.1.35"),
        selected_stream_port=int(
            inputs.config.get("WINDOWS_SELECTED_STREAM_PORT", SELECTED_STREAM_PORT)
        ),
        hosts={
            "android": inputs.host("ANDROID_HOST", "192.168.1.12"),
            "arch": inputs.host("ARCH_HOST", "192.168.1.11"),
            "windows": inputs.host("WINDOWS_HOST", "192.168.1.35"),
        },
        capture_device=str(inputs.capture_device),
        virtual_devices=tuple(str(device) for device in inputs.virtual_devices),
        shadow_ready=shadow_ready,
        phone_return_live=relay_live or legacy_phone_return,
        phone_return_title=(
            phone_return_relay_title(dict(inputs.phone_relay_health))
            if inputs.phone_relay_health
            else android_native_route_title(dict(inputs.native_health))
        ),
        host_metrics=dict(inputs.host_metrics),
    )


def _android_sender_active(inputs: ViewInputs) -> bool | None:
    values = _mapping(inputs.android_status)
    if not values.get("available"):
        return None
    if not values.get("bridge_running"):
        return False
    if values.get("root_status_error"):
        return True
    return bool(values.get("network_sender_running"))
