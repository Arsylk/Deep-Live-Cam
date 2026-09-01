from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windows.modules.device_slots import (
    BROADCAST_HOST,
    BROADCAST_LISTEN_URL,
    BROADCAST_LOCAL_ADDRESS,
    BROADCAST_PORT,
    BROADCAST_URL,
    DeviceRegistry,
    DeviceRegistryError,
    DeviceSlot,
    input_port,
    return_port,
)


def test_five_fixed_port_pairs_and_broadcast_are_disjoint(tmp_path):
    registry = DeviceRegistry(tmp_path)

    assert [(input_port(slot), return_port(slot)) for slot in range(5)] == [
        (10000, 10001),
        (10002, 10003),
        (10004, 10005),
        (10006, 10007),
        (10008, 10009),
    ]
    assert BROADCAST_PORT not in {
        port for slot in range(5) for port in (input_port(slot), return_port(slot))
    }
    assert BROADCAST_HOST == "192.168.1.35"
    assert BROADCAST_LOCAL_ADDRESS == BROADCAST_HOST
    assert "srt://192.168.1.35:10010?" in BROADCAST_URL
    assert "mode=caller" in BROADCAST_URL
    assert "srt://0.0.0.0:10010?" in BROADCAST_LISTEN_URL
    assert "mode=listener" in BROADCAST_LISTEN_URL
    snapshot = registry.snapshot()
    assert snapshot["selected_stream"]["protocol"] == "mpegts-srt"
    assert snapshot["broadcast"] == snapshot["selected_stream"]
    assert len(registry.slots()) == 5


def test_defaults_preserve_android_and_isolate_arch(tmp_path):
    registry = DeviceRegistry(tmp_path)
    android = registry.resolve("android-phone")
    arch = registry.resolve("arch-webcam")

    assert registry.selected_device_id == "android-phone"
    assert (android.slot, android.input_port, android.return_port) == (0, 10000, 10001)
    assert (arch.slot, arch.input_port, arch.return_port) == (1, 10002, 10003)
    assert android.return_host == "192.168.1.12"
    assert arch.return_host == "192.168.1.11"


def test_selection_is_persisted_with_generation(tmp_path):
    registry = DeviceRegistry(tmp_path)

    selected, changed = registry.select("arch-webcam")

    assert changed is True
    assert selected.slot == 1
    assert registry.generation == 1
    reloaded = DeviceRegistry(tmp_path)
    assert reloaded.selected_device_id == "arch-webcam"
    assert reloaded.generation == 1
    document = json.loads((tmp_path / "devices.json").read_text())
    assert document["selected_device_id"] == "arch-webcam"


def test_disabled_or_unconfigured_slot_cannot_be_selected(tmp_path):
    registry = DeviceRegistry(tmp_path)

    with pytest.raises(DeviceRegistryError, match="unknown device_id"):
        registry.select("missing")


@pytest.mark.parametrize(
    "slot",
    [
        DeviceSlot(2),
        pytest.param(
            {"slot": 1, "device_id": "bad id", "return_host": "192.168.1.2"},
            id="invalid-id",
        ),
        pytest.param(
            {"slot": 1, "device_id": "valid", "return_host": "bad host!"},
            id="invalid-host",
        ),
    ],
)
def test_slot_validation(slot):
    if isinstance(slot, DeviceSlot):
        assert slot.configured is False
    else:
        with pytest.raises(DeviceRegistryError):
            DeviceSlot(**slot)
