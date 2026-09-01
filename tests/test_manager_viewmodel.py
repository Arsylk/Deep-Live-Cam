"""The normalized manager view model, asserted without Qt or a camera."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from dlc_manager import contracts  # noqa: E402
from dlc_manager.viewmodel import (  # noqa: E402
    CRITICAL,
    INFO,
    ROUTE_LOCAL_ARCH,
    ROUTE_LOCAL_PHONE,
    ROUTE_UNRESOLVED,
    ROUTE_WINDOWS_ANDROID,
    ROUTE_WINDOWS_ARCH,
    STREAM_RAW,
    STREAM_RESULT,
    DecoderStats,
    ViewInputs,
    build_view,
)


CONFIG = {
    "WINDOWS_HOST": "192.168.1.35",
    "ARCH_HOST": "192.168.1.11",
    "ANDROID_HOST": "192.168.1.12",
    "ANDROID_CAMERA_ID": "120",
    "RECEIVER_SOURCE": "local",
}

RUNNING_SERVICES = {
    "sender": {"ActiveState": "active", "UnitFileState": "enabled"},
    "receiver": {"ActiveState": "active", "UnitFileState": "enabled"},
}


def _windows_health(device_id: str = "arch-webcam", *, fresh: bool = True) -> dict:
    age = 0.05 if fresh else 30.0
    return {
        "healthy": True,
        "state": "streaming-face-swap",
        "selected_device_id": device_id,
        "source_configured": True,
        "input": {"streaming": fresh, "last_frame_age": age, "frames": 900},
        "output": {
            "streaming": fresh,
            "last_frame_age": age,
            "frames": 890,
            "port": 10003,
            "url": "srt://192.168.1.11:10003?mode=caller",
        },
        "processing": {
            "mode": "face_swap",
            "swapper_model": "inswapper-128",
            "active_swapper_model": "inswapper-128",
            "active_swapper_backend": "cuda",
            "frames": 890,
            "fps": 29.5,
        },
    }


def _devices(selected: str = "arch-webcam", *, streaming: bool = True) -> dict:
    runtime = {
        "input": {"streaming": streaming, "last_frame_age": 0.05},
        "return": {"streaming": streaming, "last_frame_age": 0.05},
    }
    idle = {
        "input": {"streaming": False, "worker_alive": False},
        "return": {"streaming": False, "worker_alive": False},
    }
    slots = [
        {
            "slot": 0,
            "device_id": "android-phone",
            "label": "Android phone",
            "stack": "android-camera2",
            "configured": True,
            "enabled": True,
            "return_host": "192.168.1.12",
            "input_port": 10000,
            "return_port": 10001,
        },
        {
            "slot": 1,
            "device_id": "arch-webcam",
            "label": "Arch USB webcam",
            "stack": "arch-v4l2",
            "configured": True,
            "enabled": True,
            "return_host": "192.168.1.11",
            "input_port": 10002,
            "return_port": 10003,
        },
    ]
    for slot in range(2, 5):
        slots.append(
            {
                "slot": slot,
                "device_id": None,
                "label": "Unassigned",
                "stack": "generic-srt",
                "configured": False,
                "enabled": False,
                "input_port": 10_000 + slot * 2,
                "return_port": 10_001 + slot * 2,
            }
        )
    for slot in slots:
        slot["runtime"] = runtime if slot["device_id"] == selected else idle
    return {
        "selected_device_id": selected,
        "runtime": {"switching": False},
        "slots": slots,
    }


def _native_health(route: str, *, model: str = "inswapper-128") -> dict:
    return {
        "state": "running",
        "route": route,
        "uptime_seconds": 600.0,
        "input": {"url": "udp://127.0.0.1:11005", "source": "arch-webcam"},
        "processing": {
            "model": model,
            "backend": "ncnn",
            "fps": 27.4,
            "quality_status": "production",
            "identity_swap": {
                "status": "visual-effect-confirmed",
                "detail": "face-core pixels changed",
                "visual_effect_confirmed": True,
                "identity_change_verified": False,
            },
        },
        "return": {
            "url": "srt://192.168.1.12:10001?mode=caller",
            "preview_url": "udp://127.0.0.1:11004?pkt_size=1316",
            "worker_alive": True,
            "last_frame_age": 0.03,
        },
    }


def _inputs(**overrides) -> ViewInputs:
    values = {
        "config": CONFIG,
        "services": RUNNING_SERVICES,
        "receiver": {
            "status": "streaming",
            "source_mode": "local",
            "active_input": "local_processed",
            "output_frames": 5000,
        },
        "shadow": {"status": "shadowed", "preserved": []},
        "device_nodes_present": True,
        "virtual_devices": ["/dev/deep-live-cam"],
        "capture_device": "/dev/v4l/by-id/usb-Sonix-camera-video-index0",
        "registry_live": True,
        "windows_devices": _devices(),
        "windows_health": _windows_health(),
        "result_stream": DecoderStats(running=True, fps=30.0, frames=100, age=0.03, port=11003),
        "raw_stream": DecoderStats(running=True, fps=30.0, frames=100, age=0.03, port=11001),
        "now": 10_000.0,
    }
    values.update(overrides)
    return ViewInputs(**values)


# --------------------------------------------------------------- port contract


def test_five_slots_use_the_documented_fixed_port_pairs():
    pairs = [
        (contracts.slot_input_port(slot), contracts.slot_return_port(slot))
        for slot in range(contracts.SLOT_COUNT)
    ]

    assert pairs == [
        (10000, 10001),
        (10002, 10003),
        (10004, 10005),
        (10006, 10007),
        (10008, 10009),
    ]
    assert contracts.SELECTED_STREAM_PORT == 10010
    assert contracts.SELECTED_STREAM_PORT not in {
        port for pair in pairs for port in pair
    }


def test_local_relay_contract_covers_every_documented_arch_port():
    assert [relay.port for relay in contracts.ARCH_LOCAL_RELAYS] == list(
        range(11000, 11011)  # 11000-11010 inclusive (added 11010 for pre-recorded replay)
    )
    for relay in contracts.ARCH_LOCAL_RELAYS:
        assert relay.owner and relay.detail


def test_system_camera_policies_match_the_receiver_priorities():
    priorities = {
        policy.key: policy.order for policy in contracts.SYSTEM_CAMERA_POLICIES
    }

    assert priorities == {
        "local": ("local_processed", "local_prerecorded", "local_raw"),
        "windows": ("processed_return", "selected_stream", "local_raw"),
        "auto": (
            "local_processed",
            "processed_return",
            "selected_stream",
            "local_raw",
        ),
        "raw": ("local_raw",),
        "prerecorded": ("local_prerecorded",),
    }


# ------------------------------------------------------------------ route model


def test_windows_route_is_reported_per_selected_slot():
    arch = build_view(_inputs())
    android = build_view(
        _inputs(
            windows_devices=_devices("android-phone"),
            windows_health=_windows_health("android-phone"),
            android_status={
                "available": True,
                "app_installed": True,
                "bridge_running": True,
                "network_sender_running": True,
                "camera_published": True,
            },
        )
    )

    assert arch.route.key == ROUTE_WINDOWS_ARCH
    assert arch.route.windows_bypassed is False
    assert android.route.key == ROUTE_WINDOWS_ANDROID
    assert {node.key for node in arch.route.nodes} == {"android", "windows", "arch"}


def test_local_routes_report_windows_as_bypassed_not_broken():
    phone = build_view(
        _inputs(
            native_health=_native_health("android-camera-processed-to-android"),
            native_health_age=0.2,
            windows_health=None,
            windows_error="Connection refused",
        )
    )
    webcam = build_view(
        _inputs(
            native_health=_native_health("arch-webcam-processed-to-android"),
            native_health_age=0.2,
        )
    )

    assert phone.route.key == ROUTE_LOCAL_PHONE
    assert webcam.route.key == ROUTE_LOCAL_ARCH
    for view in (phone, webcam):
        assert view.route.windows_bypassed is True
        windows_node = next(
            node for node in view.route.nodes if node.key == "windows"
        )
        assert "BYPASSED" in windows_node.state_text
    # An unreachable Windows must not be raised as critical while a local route
    # is carrying the camera.
    windows_alerts = [
        alert for alert in phone.alerts if alert.component.startswith("Windows")
    ]
    assert windows_alerts == []


def test_unknown_route_is_stated_rather_than_guessed():
    view = build_view(
        _inputs(windows_health=None, windows_error="timeout", windows_devices={})
    )

    assert view.route.key == ROUTE_UNRESOLVED
    assert view.route.badge == "ROUTE UNCONFIRMED"


# ------------------------------------------------------------------- slot model


def test_every_slot_reports_identity_capability_endpoints_and_readiness():
    view = build_view(_inputs())

    assert len(view.slots) == contracts.SLOT_COUNT
    selected = next(slot for slot in view.slots if slot.selected)
    assert selected.device_id == "arch-webcam"
    assert selected.state == "active"
    assert selected.endpoint == "SRT 10002 → Windows → 192.168.1.11:10003"
    assert "V4L2" in selected.capability
    unassigned = view.slots[4]
    assert unassigned.state == "unavailable"
    assert unassigned.state_text == "NOT ASSIGNED"
    assert unassigned.selectable is False
    assert (unassigned.input_port, unassigned.return_port) == (10008, 10009)


def test_a_stale_selected_slot_is_marked_stale_not_live():
    view = build_view(
        _inputs(
            windows_devices=_devices(streaming=False),
            windows_health=_windows_health(fresh=False),
        )
    )

    selected = next(slot for slot in view.slots if slot.selected)
    assert selected.state in ("stale", "selected")
    assert "STALE" in selected.state_text or "WAITING" in selected.state_text


def test_offline_registry_slots_are_never_presented_as_selectable():
    view = build_view(_inputs(registry_live=False))

    assert all(not slot.selectable for slot in view.slots)


# ----------------------------------------------------------- system camera model


def test_system_camera_reports_configured_policy_and_actual_input_separately():
    view = build_view(
        _inputs(
            receiver={
                "status": "streaming",
                "source_mode": "auto",
                "active_input": "local_raw",
            }
        )
    )

    assert view.system_camera.configured_policy == "auto"
    assert view.system_camera.configured_label == "Best fresh processed source"
    assert view.system_camera.active_label == "Raw webcam"
    assert view.system_camera.fallback[0] == "Local Native-256 result"
    assert "not restarted" in view.system_camera.identity_note


# ---------------------------------------------------------------- stream model


def test_streams_describe_source_model_endpoint_and_freshness():
    view = build_view(_inputs(comparison_delay_ms=350))

    result = view.stream(STREAM_RESULT)
    raw = view.stream(STREAM_RAW)
    assert result.title == "SELECTED WINDOWS STREAM"
    assert "127.0.0.1:11003" in result.endpoint
    assert result.state == "live"
    assert raw.delayed is True
    assert raw.delayed_ms == 350
    assert "127.0.0.1:11001" in raw.endpoint
    assert dict(result.metrics())["Decoded FPS"] == "30.0"


def test_the_android_slot_has_no_synthesised_raw_comparison():
    view = build_view(
        _inputs(
            windows_devices=_devices("android-phone"),
            windows_health=_windows_health("android-phone"),
        )
    )

    raw = view.stream(STREAM_RAW)
    assert raw.state == "off"
    assert "never opens" in raw.note


def test_the_local_route_switches_the_result_stream_to_the_local_relay():
    view = build_view(
        _inputs(
            native_health=_native_health("arch-webcam-processed-to-android"),
            native_health_age=0.2,
            result_stream=DecoderStats(
                running=True, fps=27.0, frames=10, age=0.04, port=11007
            ),
        )
    )

    result = view.stream(STREAM_RESULT)
    assert result.title == "LOCAL NATIVE-256 OUTPUT"
    assert "127.0.0.1:11007" in result.endpoint


# --------------------------------------------------------------- identity model


def test_a_local_source_is_never_claimed_active_without_evidence():
    # The local service started at now - uptime = 9400.0, so a picture chosen
    # before that could plausibly be loaded, and one chosen after cannot be.
    started_before = build_view(
        _inputs(
            native_health=_native_health("arch-webcam-processed-to-android"),
            native_health_age=0.2,
            identity_filename="face.png",
            identity_identifier="a" * 64,
            identity_used_at=9_000.0,
            windows_config={
                "source_configured": True,
                "source_identifier": "a" * 64,
            },
        )
    )
    chosen_after_start = build_view(
        _inputs(
            native_health=_native_health("arch-webcam-processed-to-android"),
            native_health_age=0.2,
            identity_filename="face.png",
            identity_identifier="a" * 64,
            identity_used_at=9_999.9,
            windows_config={
                "source_configured": True,
                "source_identifier": "a" * 64,
            },
        )
    )

    assert started_before.identity.local_state == "unconfirmed"
    assert "not yet reported enough evidence" in started_before.identity.local_detail
    assert chosen_after_start.identity.local_state == "applying"
    assert started_before.identity.windows_state == "applied"


def test_windows_picture_is_applied_only_when_exact_content_identity_matches():
    selected = "a" * 64
    exact = build_view(
        _inputs(
            identity_filename="face.png",
            identity_identifier=selected,
            windows_config={
                "source_configured": True,
                "source_identifier": selected,
            },
        )
    )
    different = build_view(
        _inputs(
            identity_filename="face.png",
            identity_identifier=selected,
            windows_config={
                "source_configured": True,
                "source_identifier": "b" * 64,
            },
        )
    )
    legacy = build_view(
        _inputs(
            identity_filename="face.png",
            identity_identifier=selected,
            windows_config={"source_configured": True},
        )
    )

    assert exact.identity.windows_state == "applied"
    assert "exact content hash" in exact.identity.windows_detail
    assert different.identity.windows_state == "pending"
    assert legacy.identity.windows_state == "unverified"


def test_identity_verification_is_kept_separate_from_visual_effect():
    view = build_view(
        _inputs(
            native_health=_native_health("arch-webcam-processed-to-android"),
            native_health_age=0.2,
        )
    )

    assert view.processor.visual_effect_confirmed is True
    assert view.processor.identity_verified is False
    assert view.processor.checkpoint_qualified is True


def test_selected_windows_runtime_evidence_comes_only_from_health():
    health = _windows_health()
    health["processing"].update(
        {
            "tracking": {"active": True, "valid_detections": 1},
            "quality": {"swap_applied": True},
            "active_swapper_resolution": 512,
            "last_error": None,
        }
    )

    processor = build_view(_inputs(windows_health=health)).processor

    assert processor.selected_processor == "windows"
    assert processor.selected_face_detected is True
    assert processor.selected_face_swapped is True
    assert processor.selected_model == "inswapper-128"
    assert processor.selected_backend == "cuda"
    assert processor.selected_render_resolution == 512
    assert processor.selected_processing_fps == 29.5
    assert "swap applied" in processor.selected_runtime_reason


def test_selected_windows_missing_evidence_stays_unknown_not_false():
    processor = build_view(_inputs()).processor

    assert processor.selected_processor == "windows"
    assert processor.selected_face_detected is None
    assert processor.selected_face_swapped is None
    assert "not reported" in processor.selected_runtime_reason


def test_selected_arch_runtime_evidence_preserves_unqualified_warning():
    native = _native_health("arch-webcam-processed-to-android", model="native-256")
    native["processing"]["quality_status"] = "development"
    native["processing"]["identity_swap"].update(
        {
            "face_measurable": True,
            "attempted": True,
            "visual_effect_confirmed": False,
            "detail": "development checkpoint; identity replacement is unqualified",
        }
    )

    processor = build_view(
        _inputs(native_health=native, native_health_age=0.1)
    ).processor

    assert processor.selected_processor == "arch"
    assert processor.selected_face_detected is True
    assert processor.selected_face_swapped is False
    assert processor.selected_model == "native-256"
    assert processor.selected_backend == "ncnn"
    assert processor.selected_processing_fps == 27.4
    assert "invoked but not verified" in processor.selected_runtime_reason
    assert "unqualified" in processor.selected_runtime_reason


def test_a_development_checkpoint_is_not_reported_as_a_qualified_swap():
    view = build_view(
        _inputs(
            native_health={
                **_native_health(
                    "android-camera-processed-to-android", model="dlc_swap256m"
                ),
                "processing": {
                    "model": "dlc_swap256m",
                    "backend": "ncnn",
                    "fps": 12.0,
                    "quality_status": "development",
                    "identity_swap": {
                        "status": "unqualified-checkpoint",
                        "detail": "development bundle",
                        "visual_effect_confirmed": False,
                        "identity_change_verified": False,
                    },
                },
            },
            native_health_age=0.2,
        )
    )

    assert view.processor.checkpoint_qualified is False
    assert any(
        "checkpoint" in alert.component.lower() for alert in view.alerts
    )


# ---------------------------------------------------------------------- alerts


def test_alerts_name_the_failing_component_and_a_next_action():
    view = build_view(
        _inputs(
            services={
                "sender": {"ActiveState": "inactive", "UnitFileState": "enabled"},
                "receiver": {"ActiveState": "failed", "UnitFileState": "enabled"},
            },
            device_nodes_present=False,
            shadow={"status": "unmapped"},
            windows_health=None,
            windows_error="Connection refused",
        )
    )

    assert view.alerts
    assert all(alert.component and alert.next_action for alert in view.alerts)
    assert view.alerts[0].severity == CRITICAL
    components = " ".join(alert.component for alert in view.alerts)
    assert "deep-live-cam-sender.service" in components
    assert "deep-live-cam-receiver.service" in components
    assert "Camera device mapping" in components
    # The manager must not offer to start services itself.
    actions = " ".join(alert.next_action for alert in view.alerts)
    assert "never starts or stops a capture service" in actions


def test_an_unreported_service_state_is_not_raised_as_a_failure():
    view = build_view(_inputs(services={}, device_nodes_present=None, shadow={}))

    assert view.shadow_ready is None
    assert not [
        alert for alert in view.alerts if "capture owner" in alert.component
    ]
