from __future__ import annotations

from pathlib import Path
import sys

import pytest


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import device_control  # noqa: E402


def test_stopping_input_only_stops_sender(monkeypatch):
    actions: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        device_control,
        "systemctl",
        lambda action, *units: actions.append((action, units)),
    )
    monkeypatch.setattr(
        device_control,
        "repair_mapping",
        lambda *_args, **_kwargs: pytest.fail("stop must not repair mapping"),
    )

    device_control.control("input", "stop", {})

    assert actions == [("stop", (device_control.SENDER_UNIT,))]


def test_starting_all_repairs_then_starts_output_before_input(monkeypatch):
    actions: list[tuple[str, tuple[str, ...]]] = []
    repairs: list[bool] = []
    monkeypatch.setattr(
        device_control,
        "repair_mapping",
        lambda _config, force=False: repairs.append(force),
    )
    monkeypatch.setattr(
        device_control,
        "systemctl",
        lambda action, *units: actions.append((action, units)),
    )

    device_control.control("all", "start", {})

    assert repairs == [False]
    assert actions == [
        ("start", (device_control.RECEIVER_UNIT,)),
        ("start", (device_control.SENDER_UNIT,)),
    ]


def test_mapping_repair_stops_and_restores_running_streams(monkeypatch):
    actions: list[tuple[str, tuple[str, ...]]] = []
    helper_commands: list[list[str]] = []
    monkeypatch.setattr(device_control, "mapping_is_ready", lambda _config: True)
    monkeypatch.setattr(device_control, "unit_is_active", lambda _unit: True)
    monkeypatch.setattr(
        device_control,
        "systemctl",
        lambda action, *units: actions.append((action, units)),
    )
    monkeypatch.setattr(
        device_control,
        "run_checked",
        lambda command: helper_commands.append(list(command)),
    )

    device_control.repair_mapping({}, force=True)

    assert actions == [
        ("stop", (device_control.SENDER_UNIT,)),
        ("stop", (device_control.RECEIVER_UNIT,)),
        ("start", (device_control.RECEIVER_UNIT,)),
        ("start", (device_control.SENDER_UNIT,)),
    ]
    assert helper_commands == [
        [sys.executable, str(device_control.SHADOW_HELPER), "apply"]
    ]


def test_mapping_cannot_be_stopped():
    with pytest.raises(device_control.ControlError, match="repaired, not stopped"):
        device_control.control("mapping", "stop", {})
