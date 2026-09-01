from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


ROOT = Path(__file__).resolve().parents[1]
ARCH_BIN = ROOT / "arch-linux" / "bin"

if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import windows_phone_relay as relay_module  # noqa: E402
from dlc_manager.desired_state import (  # noqa: E402
    INPUT_ANDROID_BACK,
    INPUT_ANDROID_FRONT,
    INPUT_ARCH_WEBCAM,
    INPUT_PRERECORDED,
    OUTPUT_ANDROID_PHONE,
    PROCESSOR_ARCH,
    PROCESSOR_WINDOWS,
)
from dlc_manager.phone_route import (  # noqa: E402
    RELAY_LOCAL,
    RELAY_OFF,
    RELAY_WINDOWS,
    desired_relay_source,
    relay_desires,
    relay_is_closed,
    windows_runtime_ready,
)
from dlc_manager.shell import ManagerWindow  # noqa: E402


@pytest.fixture
def relay_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    intent_path = tmp_path / "phone-return-relay.json"
    monkeypatch.setenv("WINDOWS_PHONE_RELAY_SOURCE", RELAY_OFF)
    monkeypatch.setenv("WINDOWS_PHONE_RELAY_SOURCE_PORT", "12008")
    monkeypatch.setenv("LOCAL_PHONE_RELAY_SOURCE_PORT", "12009")
    monkeypatch.setenv("ANDROID_NATIVE_PREVIEW_PORT", "12004")
    monkeypatch.setenv("ANDROID_NATIVE_RETURN_PORT", "12001")
    monkeypatch.setenv("ANDROID_HOST", "10.23.0.12")
    monkeypatch.setenv("PHONE_RETURN_RELAY_STATE_FILE", str(intent_path))
    monkeypatch.setenv(
        "PHONE_RETURN_RELAY_CONTROL_SOCKET",
        str(tmp_path / "phone-return-relay-control.sock"),
    )
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(relay_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    return intent_path


@pytest.mark.parametrize(
    ("processor", "input_key", "phone_enabled", "expected"),
    [
        (processor, input_key, phone_enabled, expected)
        for processor in (PROCESSOR_WINDOWS, PROCESSOR_ARCH)
        for input_key in (
            INPUT_ARCH_WEBCAM,
            INPUT_ANDROID_FRONT,
            INPUT_ANDROID_BACK,
        )
        for phone_enabled, expected in (
            (False, RELAY_OFF),
            (
                True,
                (
                    RELAY_LOCAL
                    if processor == PROCESSOR_ARCH
                    else RELAY_WINDOWS
                    if input_key == INPUT_ARCH_WEBCAM
                    else RELAY_OFF
                ),
            ),
        )
    ],
)
def test_phone_return_owner_truth_table(
    processor: str,
    input_key: str,
    phone_enabled: bool,
    expected: str,
) -> None:
    desired = {
        "processor": processor,
        "input": input_key,
        "outputs": {OUTPUT_ANDROID_PHONE: phone_enabled},
    }

    assert desired_relay_source(desired) == expected


@pytest.mark.parametrize("processor", (PROCESSOR_WINDOWS, PROCESSOR_ARCH))
def test_prerecorded_returns_to_phone_through_the_local_relay(processor: str) -> None:
    # Prerecorded video is produced entirely on Arch (the receiver's file_relay
    # writes the local phone-relay port), so with the phone output enabled it
    # must route through the local relay regardless of the nominal processor.
    assert desired_relay_source(
        {
            "processor": processor,
            "input": INPUT_PRERECORDED,
            "outputs": {OUTPUT_ANDROID_PHONE: True},
        }
    ) == RELAY_LOCAL
    # With the phone output off it stays off.
    assert desired_relay_source(
        {
            "processor": processor,
            "input": INPUT_PRERECORDED,
            "outputs": {OUTPUT_ANDROID_PHONE: False},
        }
    ) == RELAY_OFF


def _healthy_windows_runtime() -> dict[str, Any]:
    return {
        "selected_device_id": "arch-webcam",
        "runtime": {
            "selected_device_id": "arch-webcam",
            "switching": False,
            "last_switch_error": None,
            "selected_stream": {
                "worker_alive": True,
                "streaming": True,
            },
        },
    }


def test_windows_runtime_confirmation_requires_live_selected_stream() -> None:
    document = _healthy_windows_runtime()

    assert windows_runtime_ready(
        document,
        "arch-webcam",
        require_selected_stream=True,
    )

    document["runtime"]["selected_stream"]["streaming"] = False
    # A healthy compatibility alias cannot mask an explicitly unhealthy
    # selected_stream record.
    document["runtime"]["broadcast"] = {
        "worker_alive": True,
        "streaming": True,
    }
    assert not windows_runtime_ready(
        document,
        "arch-webcam",
        require_selected_stream=True,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.update(selected_device_id="android-phone"),
        lambda document: document["runtime"].update(
            selected_device_id="android-phone"
        ),
        lambda document: document["runtime"].update(switching=True),
        lambda document: document["runtime"].update(last_switch_error="failed"),
        lambda document: document["runtime"].pop("selected_stream"),
        lambda document: document["runtime"]["selected_stream"].update(
            worker_alive=False
        ),
        lambda document: document["runtime"]["selected_stream"].update(
            streaming=False
        ),
    ],
    ids=(
        "persisted-selection-mismatch",
        "runtime-selection-mismatch",
        "switch-in-progress",
        "switch-error",
        "missing-selected-stream",
        "selected-stream-worker-dead",
        "selected-stream-stale",
    ),
)
def test_windows_runtime_confirmation_rejects_unconfirmed_state(mutate) -> None:
    document = _healthy_windows_runtime()
    mutate(document)

    assert not windows_runtime_ready(
        document,
        "arch-webcam",
        require_selected_stream=True,
    )


def test_windows_runtime_legacy_broadcast_alias_is_only_a_fallback() -> None:
    document = _healthy_windows_runtime()
    selected_stream = document["runtime"].pop("selected_stream")
    document["runtime"]["broadcast"] = selected_stream

    assert windows_runtime_ready(
        document,
        "arch-webcam",
        require_selected_stream=True,
    )
    assert windows_runtime_ready(
        document,
        "arch-webcam",
        require_selected_stream=False,
    )


def test_relay_closed_confirmation_is_fail_closed() -> None:
    closed = {
        "source": RELAY_OFF,
        "effective_source": RELAY_OFF,
        "transport_open": False,
    }

    assert relay_is_closed(closed)
    assert relay_desires(closed, RELAY_OFF)

    for field, value in (
        ("source", RELAY_WINDOWS),
        ("effective_source", RELAY_WINDOWS),
        ("transport_open", True),
    ):
        not_closed = dict(closed)
        not_closed[field] = value
        assert not relay_is_closed(not_closed)


@pytest.mark.parametrize(
    ("source", "source_port"),
    ((RELAY_WINDOWS, 12008), (RELAY_LOCAL, 12009)),
)
def test_relay_command_has_one_input_one_phone_caller_and_exact_preview(
    relay_environment: Path,
    source: str,
    source_port: int,
) -> None:
    relay = relay_module.WindowsPhoneRelay()
    command = relay.command(source)
    rendered = " ".join(command)
    targets = command[-1].split("|")

    assert command.count("-i") == 1
    assert f"udp://127.0.0.1:{source_port}?reuse=1" in rendered
    assert "-c copy" in rendered
    assert "-f tee" in rendered
    assert len(targets) == 2
    assert sum("srt://" in target for target in targets) == 1
    preview_targets = [
        target
        for target in targets
        if "udp://127.0.0.1:12004?pkt_size=1316" in target
    ]
    assert len(preview_targets) == 1
    assert rendered.count("srt://") == 1
    assert "srt://10.23.0.12:12001?" in rendered
    assert "mode=caller" in rendered
    assert "messageapi=1" in rendered
    assert ":12008" not in command[-1]
    assert ":12009" not in command[-1]


def test_set_source_is_durable_and_restored_after_restart(
    relay_environment: Path,
) -> None:
    relay = relay_module.WindowsPhoneRelay()

    response = relay.set_source(RELAY_WINDOWS, revision=7)

    assert response["source"] == RELAY_WINDOWS
    assert response["revision"] == 7
    assert json.loads(relay_environment.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "source": RELAY_WINDOWS,
        "revision": 7,
    }

    relay.stop()
    restored = relay_module.WindowsPhoneRelay()
    assert restored.source == RELAY_WINDOWS
    assert restored.revision == 7
    assert restored.last_error is None


@pytest.mark.parametrize(
    "payload",
    (
        "{not-json",
        "[]",
        '{"source":"unknown","revision":4}',
        '{"source":"windows","revision":true}',
        '{"source":"windows","revision":-1}',
    ),
)
def test_corrupt_durable_state_fails_closed(
    relay_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    # A bad durable document must not fall back to a potentially active
    # environment default.
    monkeypatch.setenv("WINDOWS_PHONE_RELAY_SOURCE", RELAY_WINDOWS)
    relay_environment.write_text(payload, encoding="utf-8")

    relay = relay_module.WindowsPhoneRelay()
    snapshot = relay.snapshot()

    assert snapshot["source"] == RELAY_OFF
    assert snapshot["effective_source"] == RELAY_OFF
    assert snapshot["transport_open"] is False
    assert snapshot["revision"] == 0
    assert "fail-closed" in str(snapshot["last_error"])


def test_stale_or_conflicting_revisions_cannot_change_durable_intent(
    relay_environment: Path,
) -> None:
    relay = relay_module.WindowsPhoneRelay()
    relay.set_source(RELAY_WINDOWS, revision=5)
    original = relay_environment.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="stale revision 4"):
        relay.set_source(RELAY_LOCAL, revision=4)
    with pytest.raises(ValueError, match="already applied"):
        relay.set_source(RELAY_LOCAL, revision=5)
    with pytest.raises(ValueError, match="non-negative integer"):
        relay.set_source(RELAY_OFF, revision=True)

    assert relay.set_source(RELAY_WINDOWS, revision=5)["revision"] == 5
    assert relay.source == RELAY_WINDOWS
    assert relay.revision == 5
    assert relay_environment.read_text(encoding="utf-8") == original


def test_source_off_synchronously_closes_worker_before_persist_and_ack(
    relay_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = relay_module.WindowsPhoneRelay()
    relay.source = RELAY_WINDOWS
    relay.revision = 10

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll() -> None:
            return None

    process = FakeProcess()
    relay.process = process  # type: ignore[assignment]
    relay.process_source = RELAY_WINDOWS
    events: list[str] = []

    def fake_stop(candidate, *, grace: float) -> None:
        assert candidate is process
        assert grace == 0.4
        assert relay.process is None
        assert relay.process_source == RELAY_OFF
        events.append("stop")

    real_persist = relay._persist_intent_locked

    def recording_persist(source: str, revision: int) -> None:
        assert events == ["stop"]
        events.append("persist")
        real_persist(source, revision)

    monkeypatch.setattr(relay_module, "stop_process", fake_stop)
    monkeypatch.setattr(relay, "_persist_intent_locked", recording_persist)

    response = relay.set_source(RELAY_OFF, revision=11)

    assert events == ["stop", "persist"]
    assert response["source"] == RELAY_OFF
    assert response["effective_source"] == RELAY_OFF
    assert response["transport_open"] is False
    assert response["worker_alive"] is False
    assert response["revision"] == 11


def test_control_socket_is_explicitly_world_writable_for_desktop_manager(
    relay_environment: Path,
) -> None:
    relay = relay_module.WindowsPhoneRelay()
    socket_path = relay.control_socket_path
    thread = threading.Thread(target=relay.control_loop, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert socket_path.exists()
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o666
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            client.connect(str(socket_path))
            client.sendall(b'{"op":"get"}\n')
            response = json.loads(client.recv(16 * 1024).decode("utf-8"))
        assert response["ok"] is True
        assert response["source"] == RELAY_OFF
        assert response["transport_open"] is False
    finally:
        relay.stop()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)

    assert not thread.is_alive()


def test_active_worker_publishes_health_and_watchdog_without_exiting(
    relay_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = relay_module.WindowsPhoneRelay()
    relay.source = RELAY_WINDOWS
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time\n"
                "for frame in range(1, 200):\n"
                " print(f'frame={frame}', flush=True)\n"
                " time.sleep(0.005)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    relay.process = process
    relay.process_source = RELAY_WINDOWS
    publications: list[bool] = []

    def publish() -> dict[str, Any]:
        publications.append(process.poll() is None)
        if len(publications) >= 2:
            relay.stop_event.set()
        return relay.snapshot()

    monkeypatch.setattr(relay_module, "STATE_PUBLISH_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(relay, "_publish_state", publish)
    try:
        relay._observe_progress(process, RELAY_WINDOWS)
    finally:
        process.terminate()
        process.wait(timeout=2.0)

    assert publications == [True, True]


class _IdleHelper:
    @staticmethod
    def busy() -> bool:
        return False


class _IdleWindowsClient:
    @staticmethod
    def selection_in_flight() -> bool:
        return False


class _ManagerHarness:
    _desired_windows_device = ManagerWindow._desired_windows_device
    _current_phone_route_signature = ManagerWindow._current_phone_route_signature
    _phone_route_runtime_ready = ManagerWindow._phone_route_runtime_ready
    _reconcile_phone_route = ManagerWindow._reconcile_phone_route
    _reconcile_windows_input = ManagerWindow._reconcile_windows_input

    def __init__(self, desired: dict[str, Any], relay_health: dict[str, Any]):
        self.desired_state = desired
        self.phone_relay_helper = _IdleHelper()
        self.windows = _IdleWindowsClient()
        self.phone_relay_health = relay_health
        self.phone_route_signature = None
        self.phone_route_quiesced_signature = None
        self.phone_route_applied_signature = None
        self.registry_live = True
        self.windows_devices = _healthy_windows_runtime()
        self._latest_native_health = {
            "return": {"active": True, "transport_open": True}
        }
        self.actions: list[str] = []

    def _run_phone_relay_control(self, source: str, _signature: tuple) -> bool:
        self.actions.append(f"relay:{source}")
        return True

    def select_windows_slot(self, device_id: str) -> None:
        self.actions.append(f"windows:{device_id}")


def _desired(processor: str, input_key: str, phone: bool = True) -> dict[str, Any]:
    return {
        "processor": processor,
        "input": input_key,
        "outputs": {OUTPUT_ANDROID_PHONE: phone},
    }


def test_manager_breaks_relay_before_selecting_direct_windows_phone_route():
    manager = _ManagerHarness(
        _desired(PROCESSOR_WINDOWS, INPUT_ANDROID_FRONT),
        {
            "source": RELAY_LOCAL,
            "effective_source": RELAY_LOCAL,
            "transport_open": True,
        },
    )

    manager._reconcile_phone_route()

    assert manager.actions == ["relay:off"]

    manager.actions.clear()
    manager.phone_relay_health = {
        "source": RELAY_OFF,
        "effective_source": RELAY_OFF,
        "transport_open": False,
    }
    manager._reconcile_phone_route()
    assert manager.actions == ["windows:android-phone"]
    assert "relay:local" not in manager.actions
    assert "relay:windows" not in manager.actions


def test_manager_waits_for_fresh_selected_stream_before_windows_relay():
    manager = _ManagerHarness(
        _desired(PROCESSOR_WINDOWS, INPUT_ARCH_WEBCAM),
        {
            "source": RELAY_OFF,
            "effective_source": RELAY_OFF,
            "transport_open": False,
        },
    )
    manager.windows_devices["runtime"]["selected_stream"]["streaming"] = False

    manager._reconcile_phone_route()
    assert manager.actions == []

    manager.windows_devices["runtime"]["selected_stream"]["streaming"] = True
    manager._reconcile_phone_route()
    assert manager.actions == ["relay:windows"]


def test_manager_never_opens_local_relay_without_verified_windows_park():
    manager = _ManagerHarness(
        _desired(PROCESSOR_ARCH, INPUT_ANDROID_BACK),
        {
            "source": RELAY_OFF,
            "effective_source": RELAY_OFF,
            "transport_open": False,
        },
    )
    manager.registry_live = False
    manager.windows_devices = {}

    manager._reconcile_phone_route()

    assert "relay:local" not in manager.actions
