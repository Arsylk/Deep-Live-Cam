from __future__ import annotations

from pathlib import Path
import sys


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

from pipeline_topology import (  # noqa: E402
    ROUTE_ANDROID,
    ROUTE_ARCH,
    ROUTE_UNKNOWN,
    endpoint_host,
    infer_topology,
    stream_is_fresh,
)


def test_missing_windows_health_does_not_guess_route_from_ready_sender():
    topology = infer_topology(
        None,
        arch_host="192.168.1.11",
        android_host="192.168.1.12",
        arch_sender_active=True,
        android_sender_active=None,
    )

    assert topology.selected == ROUTE_UNKNOWN
    assert topology.summary == "Camera route could not be confirmed"


def test_endpoint_host_handles_srt_query_and_invalid_values():
    assert endpoint_host("srt://192.168.1.12:10001?mode=caller") == "192.168.1.12"
    assert endpoint_host(None) is None
    assert endpoint_host(42) is None


def test_android_return_target_selects_phone_route():
    topology = infer_topology(
        {"output": {"url": "srt://192.168.1.12:10001?mode=caller"}},
        arch_host="192.168.1.11",
        android_host="192.168.1.12",
        arch_sender_active=False,
        android_sender_active=True,
    )

    assert topology.selected == ROUTE_ANDROID
    assert topology.conflict is False
    assert topology.warning is None


def test_competing_senders_are_reported_even_with_known_route():
    topology = infer_topology(
        {"output": {"url": "srt://192.168.1.12:10001"}},
        arch_host="192.168.1.11",
        android_host="192.168.1.12",
        arch_sender_active=True,
        android_sender_active=True,
    )

    assert topology.selected == ROUTE_ANDROID
    assert topology.conflict is True
    assert "compete" in str(topology.warning)


def test_isolated_slot_clients_do_not_compete():
    topology = infer_topology(
        {
            "selected_device_id": "android-phone",
            "output": {"url": "srt://192.168.1.12:10001"},
        },
        arch_host="192.168.1.11",
        android_host="192.168.1.12",
        arch_sender_active=True,
        android_sender_active=True,
    )

    assert topology.selected == ROUTE_ANDROID
    assert topology.conflict is False
    assert topology.mismatch is False
    assert topology.warning is None


def test_arch_route_and_frame_freshness():
    topology = infer_topology(
        {"output": {"url": "srt://192.168.1.11:10001"}},
        arch_host="192.168.1.11",
        android_host="192.168.1.12",
        arch_sender_active=True,
        android_sender_active=False,
    )

    assert topology.selected == ROUTE_ARCH
    assert stream_is_fresh({"streaming": True, "last_frame_age": 0.2}) is True
    assert stream_is_fresh({"streaming": False, "last_frame_age": 0.2}) is False
    assert stream_is_fresh({"last_frame_age": 8.0}) is False
