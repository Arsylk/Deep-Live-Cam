"""Off-screen contract for the native manager window.

Nothing in this module opens a camera device, starts a service, or reaches the
network: the decoders, helper processes, systemd probe, and Windows client are
all stubbed, and every assertion is about the interface contract.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDockWidget,
    QTabWidget,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
MODULE_PATH = BIN / "tester.py"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location("arch_tester_workspace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tester = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tester)

from dlc_manager import services as manager_services  # noqa: E402
from dlc_manager.pages.system import LOG_LINE_LIMIT  # noqa: E402
from dlc_manager import shell as manager_shell  # noqa: E402

CONFIG_PATH = ROOT / "arch-linux" / "config" / "deep-live-cam-arch.conf"
EXPECTED_WORKSPACES = [
    "Processor",
    "Input",
    "Output",
    "Render",
]


def _application() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


class _DecoderCalls:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0


@pytest.fixture()
def manager(monkeypatch, tmp_path):
    """A fully wired window whose side effects are all stubbed out.

    State and health documents are read from an empty temporary directory, so
    the assertions describe the interface rather than whatever this machine's
    services happen to be doing while the suite runs.
    """
    app = _application()
    calls = _DecoderCalls()

    def fake_start(decoder):
        calls.starts += 1
        decoder.running = True

    def fake_stop(decoder):
        calls.stops += 1
        decoder.running = False

    monkeypatch.setattr(tester.RawVideoDecoder, "start", fake_start)
    monkeypatch.setattr(tester.RawVideoDecoder, "stop", fake_stop)
    monkeypatch.setattr(
        tester.RawVideoDecoder, "restart", lambda decoder: fake_start(decoder)
    )
    monkeypatch.setattr(
        manager_services.SystemdProbe,
        "state",
        lambda self, unit: {
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
        },
    )
    monkeypatch.setattr(
        manager_services.JsonProcess, "run", lambda self, script, arguments: False
    )
    monkeypatch.setattr(
        manager_services.HelperProcess, "run", lambda self, program, arguments: False
    )
    for name in ("request_health", "request_config", "request_devices"):
        monkeypatch.setattr(
            manager_services.WindowsControlClient, name, lambda self: None
        )

    config = tester.load_env_file(CONFIG_PATH)
    config["STATE_DIR"] = str(tmp_path / "run")
    config["MANAGER_STATE_FILE"] = str(tmp_path / "manager-state.json")
    config["SOURCE_HISTORY_DIR"] = str(tmp_path / "source-history")
    config["ANDROID_NATIVE_HEALTH_FILE"] = str(tmp_path / "no-local-processor.json")
    window = tester.TesterWindow(config)
    window.calls = calls
    try:
        yield window
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def _page_index(window, widget: QWidget) -> int:
    for index in range(window.workspace_stack.count()):
        page = window.workspace_stack.widget(index)
        if widget is page or page.isAncestorOf(widget):
            return index
    return -1


# ------------------------------------------------------------ page contract


def test_manager_uses_exactly_three_primary_decision_tabs(manager):
    assert manager.workspace_stack.count() == 4
    assert [
        manager.workspace_tabs.tabText(index) for index in range(4)
    ] == EXPECTED_WORKSPACES
    assert not manager.findChildren(QDockWidget)
    assert manager.findChildren(QTabWidget) == [manager.workspace_tabs]


def test_every_functional_area_lives_on_exactly_one_page(manager):
    assert _page_index(manager, manager.processing_page.source_preview) == 0
    assert _page_index(manager, manager.processing_page.win_opacity) == 0
    assert _page_index(manager, manager.processing_page.processor_target) == 0
    assert _page_index(manager, manager.input_page.input_buttons["arch-webcam"]) == 1
    assert _page_index(manager, manager.input_page.camera_apply) == 1
    assert _page_index(manager, manager.output_pane) == 2
    assert _page_index(manager, manager.output_page.mirror) == 2
    assert _page_index(manager, manager.output_page.rotation) == 2
    assert _page_index(manager, manager.stats_box) == -1
    assert _page_index(manager, manager.log_box) == -1


def test_each_setting_has_exactly_one_widget(manager):
    """A duplicated control would show two values for one setting."""
    controls = manager.processing_page.windows_controls()

    assert len(controls) == len({id(control) for control in controls})
    for control in controls:
        assert _page_index(manager, control) == 0


def test_output_host_switch_is_one_atomic_click_and_queues_inactive_windows_bypass(
    manager, monkeypatch
):
    route_calls: list[str] = []
    local_calls: list[bool] = []
    monkeypatch.setattr(manager.windows, "apply_config", lambda payload: False)
    monkeypatch.setattr(
        manager,
        "select_system_camera_policy",
        lambda policy: route_calls.append(policy),
    )
    monkeypatch.setattr(manager, "_apply_output_configuration", lambda: None)
    monkeypatch.setattr(manager, "_set_output_preview_source", lambda **kwargs: None)
    monkeypatch.setattr(
        manager, "_apply_local_processor_state", lambda: local_calls.append(True)
    )
    monkeypatch.setattr(manager, "_reconcile_phone_route", lambda: None)
    before_revision = manager.desired_state["revision"]

    manager.output_page.processor_processing["arch"].click()

    assert manager.desired_state["revision"] == before_revision + 1
    assert manager.desired_state["processor"] == "arch"
    assert manager.desired_state["processing"]["processing_mode"] == "face_swap"
    assert manager.pending_windows_changes["processing_mode"] == "passthrough"
    assert manager.processing_page.processor_target.currentData() == "arch"
    assert manager.output_page.processor_processing["arch"].isChecked() is True
    assert manager.output_page.processor_processing["windows"].isChecked() is False
    assert route_calls == ["local"]
    assert local_calls == [True]


def test_stale_arch_acknowledgement_immediately_reapplies_latest_safe_state(
    manager, monkeypatch
):
    scheduled: list[int] = []
    monkeypatch.setattr(
        manager_shell.QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append(delay),
    )
    monkeypatch.setattr(manager, "_reconcile_phone_route", lambda: None)
    manager.desired_state = manager.desired_store.set_processing("opacity", 0.8)
    manager.local_processor_inflight_revision = manager.desired_state["revision"] - 1
    stale = json.dumps(
        {
            "ok": True,
            "in_sync": True,
            "revision": manager.local_processor_inflight_revision,
        }
    )

    manager._local_processor_finished(True, stale)

    assert manager.pending_local_processor_state is True
    assert scheduled == [0]


def test_offline_inactive_arch_is_safe_off_without_retry_loop(manager, monkeypatch):
    scheduled: list[int] = []
    monkeypatch.setattr(
        manager_shell.QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append(delay),
    )
    monkeypatch.setattr(manager, "_reconcile_phone_route", lambda: None)
    assert manager.desired_state["processor"] == "windows"
    manager.local_processor_inflight_revision = manager.desired_state["revision"]

    manager._local_processor_finished(False, "control socket unavailable")

    assert manager.pending_local_processor_state is False
    assert scheduled == []


def test_windows_sync_requires_the_exact_selected_source_identifier(manager):
    entry = manager.source_history.remember(b"selected-source-bytes", "face.png")
    manager.desired_state = manager.desired_store.update(
        source_identifier=entry.identifier
    )
    manager.pending_source = None
    manager.source_upload_inflight = None

    manager.windows_config = {
        "source_configured": True,
        "source_identifier": "0" * 64,
    }
    assert manager._windows_source_synchronized() is False

    manager.windows_config["source_identifier"] = entry.identifier
    assert manager._windows_source_synchronized() is True

    manager.pending_source = (b"selected-source-bytes", "face.png")
    assert manager._windows_source_synchronized() is False


def test_source_aware_windows_reconnect_queues_only_a_mismatched_picture(manager):
    data = b"durable-selected-source"
    entry = manager.source_history.remember(data, "face.png")
    manager.desired_state = manager.desired_store.update(
        source_identifier=entry.identifier
    )
    manager.pending_source = None
    manager.source_upload_inflight = None

    manager._reconcile_windows_source_identity(
        {"source_configured": True, "source_identifier": "f" * 64}
    )
    assert manager.pending_source == (data, "face.png")

    manager._reconcile_windows_source_identity(
        {"source_configured": True, "source_identifier": entry.identifier}
    )
    assert manager.pending_source is None


def test_transient_source_upload_failure_retries_while_windows_remains_online(
    manager,
):
    manager.pending_source = (b"selected-source", "face.png")
    manager.source_upload_inflight = manager.pending_source
    manager.windows_was_reachable = True

    manager._source_upload_failed("temporary transport failure")

    assert manager.source_upload_inflight is None
    assert manager.pending_source == (b"selected-source", "face.png")
    assert manager.source_retry_timer.isActive() is True


def test_legacy_windows_source_status_does_not_claim_or_loop_exact_sync(manager):
    data = b"legacy-compatible-source"
    entry = manager.source_history.remember(data, "face.png")
    manager.desired_state = manager.desired_store.update(
        source_identifier=entry.identifier
    )
    manager.pending_source = None

    manager._reconcile_windows_source_identity({"source_configured": True})

    assert manager.pending_source is None
    assert manager._windows_source_synchronized() is False


def test_effective_input_status_never_calls_offline_or_mismatched_input_selected(
    manager,
):
    manager.desired_state = manager.desired_store.set_processor("arch")
    manager.desired_state = manager.desired_store.set_input("android-front")
    manager.pending_local_processor_state = False
    manager.camera_last_result_ok = True
    manager.android_status = {"controls": {"lens_facing": "back"}}
    manager._latest_native_health = {
        "state": "running",
        "control": {"effective": {"input": "android-back"}},
    }

    assert manager._effective_input_status("android-front") == "mismatch"

    manager._latest_native_health = {}
    assert manager._effective_input_status("android-front") == "offline"

    manager.camera_last_result_ok = False
    assert manager._effective_input_status("android-front") == "failed"


def test_header_distinguishes_live_output_from_configured_but_waiting(manager):
    manager.desired_state = manager.desired_store.set_output("android-phone", False)
    receiver = {
        "virtual_camera": "/dev/deep-live-cam",
        "virtual_cameras": ["/dev/deep-live-cam"],
        "source_mode": "windows",
        "output_enabled": True,
        "output_transform": {"mirror": False, "rotation": 0},
        "status": "streaming",
        "sink_pid": 4242,
    }
    active = {"receiver": {"ActiveState": "active"}}
    stopped = {"receiver": {"ActiveState": "inactive"}}
    assert manager.view is not None

    # A matching in-memory document is not proof that its writer is still
    # alive.  Without a fresh health-file timestamp the header must remain in
    # the configured-but-waiting state.
    manager._render_header(
        manager.view, receiver, active, input_status="desired"
    )
    assert manager.ribbon_cells["output"][1].text().endswith(
        "CONFIG SYNCED · WAITING"
    )

    health_file = manager.state_dir / "receiver.json"
    health_file.parent.mkdir(parents=True, exist_ok=True)
    health_file.write_text("{}", encoding="utf-8")
    manager._render_header(
        manager.view, receiver, active, input_status="desired"
    )
    assert manager.ribbon_cells["output"][1].text().endswith("STREAMS LIVE")

    manager._render_header(
        manager.view, receiver, stopped, input_status="failed"
    )
    assert manager.ribbon_cells["output"][1].text().endswith(
        "CONFIG SYNCED · WAITING"
    )
    assert manager.ribbon_cells["input"][1].text().endswith("SETTINGS FAILED")


def test_the_legacy_dock_and_tab_builders_were_removed_not_just_unused():
    source = MODULE_PATH.read_text(encoding="utf-8")
    package = (BIN / "dlc_manager").rglob("*.py")
    everything = source + "".join(
        path.read_text(encoding="utf-8")
        for path in package
        if "__pycache__" not in path.parts
    )

    for retired in (
        "_build_legacy_camera_dock",
        "_build_legacy_devices_panel",
        "_build_face_panel",
        "_build_devices_panel",
        "_build_quality_panel",
        "_build_system_panel",
        "QDockWidget",
        "DeviceControlRow",
    ):
        assert retired not in everything, retired


def test_every_semantic_input_and_output_is_represented_once(manager):
    assert sorted(manager.input_page.input_buttons) == [
        "android-back",
        "android-front",
        "arch-webcam",
        "assembler",
        "prerecorded",
    ]
    assert sorted(manager.output_page.outputs) == ["android-phone", "arch-camera"]


def _android_output_status(
    *,
    applied=True,
    persisted=True,
    available=True,
    version="v0.4.7",
    enabled=True,
    module_enabled=None,
    selector_running=None,
):
    module_enabled = available if module_enabled is None else module_enabled
    selector_running = available if selector_running is None else selector_running
    supported = version not in ("v0.4.5", "broken", "")
    return {
        "available": available,
        "module_installed": available,
        "module_enabled": module_enabled,
        "module_version": version,
        "output_selector_running": selector_running,
        "provider_running": available,
        "camera_node_ready": available,
        "camera_published": available,
        "output_control": {
            "supported": supported,
            "enabled": enabled,
            "mirror": False,
            "rotation": 0,
            "persisted": persisted,
            "effective_source": "processed" if enabled else "placeholder",
            "effective_worker_alive": bool(enabled),
            "applied": applied,
        },
        "front_redirect": {
            "package_installed": True,
            "active": None,
            "processed_camera_id": "120",
        },
    }


def test_android_output_status_is_schema_safe_and_distinguishes_offline(manager):
    desired = manager.desired_state

    manager.output_page.set_delivery_status(
        desired,
        {},
        {"available": False, "output_control": "legacy"},
    )
    assert manager.output_page.output_pills["android-phone"].text().endswith("OFFLINE")

    manager.output_page.set_delivery_status(
        desired, {}, _android_output_status()
    )
    assert manager.output_page.output_pills["android-phone"].text().endswith(
        "PROCESSED STREAM LIVE"
    )
    assert "scoped Xposed" in manager.output_page.phone_redirect_status.text()


def test_android_output_helper_uses_the_exact_latest_desired_signature(
    manager, monkeypatch
):
    calls: list[tuple[str, list[str]]] = []
    manager.android_status = _android_output_status(applied=False, persisted=False)
    monkeypatch.setattr(
        manager.android_output_helper,
        "run_python",
        lambda script, arguments: calls.append((script, list(arguments))) or True,
    )

    manager._apply_android_output_configuration()

    assert len(calls) == 1
    _script, arguments = calls[0]
    assert arguments == [
        "configure-output",
        "--host",
        "192.168.1.12",
        "--output-enabled",
        "true",
        "--output-mirror",
        "false",
        "--output-rotation",
        "0",
        "--serial",
        "192.168.1.12:46600",
    ]


def _arch_output_status(*, source="windows", enabled=True, mirror=False, rotation=0):
    return {
        "ok": True,
        "virtual_camera": "/dev/deep-live-cam",
        "virtual_cameras": ["/dev/deep-live-cam"],
        "source": source,
        "output_enabled": enabled,
        "output_transform": {
            "mirror": mirror,
            "rotation": rotation,
            "revision": 7,
        },
        "sink_pid": 4242,
    }


def test_arch_output_waits_for_exact_effective_confirmation(manager):
    signature = manager._arch_output_signature()
    manager.output_configuration_inflight_state = signature
    manager.output_configuration_inflight = True

    manager._output_configuration_finished(
        True, json.dumps(_arch_output_status(rotation=90))
    )

    assert manager.pending_output_configuration is True

    manager.output_configuration_inflight_state = signature
    manager.output_configuration_inflight = True
    manager._output_configuration_finished(True, json.dumps(_arch_output_status()))

    assert manager.pending_output_configuration is False


def test_rapid_arch_output_changes_reapply_only_the_newest_signature(
    manager, monkeypatch
):
    old_signature = manager._arch_output_signature()
    manager.output_configuration_inflight_state = old_signature
    manager.output_configuration_inflight = True
    manager.desired_state = manager.desired_store.set_transform(
        mirror=True, rotation=270
    )

    manager._output_configuration_finished(True, json.dumps(_arch_output_status()))

    assert manager.pending_output_configuration is True
    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager.output_helper,
        "run_python",
        lambda _script, arguments: calls.append(list(arguments)) or True,
    )
    manager._apply_output_configuration()

    assert calls[-1][calls[-1].index("--mirror") + 1] == "true"
    assert calls[-1][calls[-1].index("--rotation") + 1] == "270"


def test_prerecorded_input_selects_the_prerecorded_receiver_source(
    manager, monkeypatch
):
    """Prerecorded input must drive the receiver to its 'prerecorded' source.

    Otherwise the periodic output reconciliation keeps forcing 'local' and the
    still-running phone processor immediately preempts the chosen video.
    """
    manager.desired_state = manager.desired_store.set_prerecorded_path(
        "/tmp/example.mp4"
    )
    manager.desired_state = manager.desired_store.set_input("prerecorded")

    assert manager._desired_receiver_source() == "prerecorded"
    assert manager._arch_output_signature()[0] == "prerecorded"

    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager.output_helper,
        "run_python",
        lambda _script, arguments: calls.append(list(arguments)) or True,
    )
    manager._apply_output_configuration()

    assert calls, "output configuration helper was not invoked"
    latest = calls[-1]
    assert latest[latest.index("--source") + 1] == "prerecorded"


def test_prerecorded_relay_authors_source_after_killing_orphans(
    manager, monkeypatch, tmp_path
):
    """The manager must author the source path itself after starting the relay.

    A killed relay's finally-block clears prerecorded-source.txt, which can race
    the new relay's write and blank the source (receiver then delivers no
    prerecorded frames -- the switch appears broken).  The manager writes the
    path authoritatively via a delayed callback so a late clear cannot win.
    """
    manager.state_dir = tmp_path
    src = tmp_path / "prerecorded-source.txt"
    # Simulate a dying relay having just blanked the source file.
    src.write_text("")
    video = "/var/lib/deep-live-cam/renders/example.mp4"
    manager.prerecorded_relay_path = video

    # The delayed authoring callback is what the QTimer fires.
    manager._author_prerecorded_source(video)
    assert src.read_text() == video

    # If the user has since switched away (relay_path changed), a late callback
    # must NOT stamp the old video back over the new state.
    manager.prerecorded_relay_path = "/some/other.mp4"
    src.write_text("")
    manager._author_prerecorded_source(video)
    assert src.read_text() == ""


def test_live_input_keeps_the_local_receiver_source(manager, monkeypatch):
    """A live camera input on the Arch processor still selects 'local'."""
    manager.desired_state = manager.desired_store.set_processor("arch")
    manager.desired_state = manager.desired_store.set_input("android-front")

    assert manager._desired_receiver_source() == "local"
    assert manager._arch_output_signature()[0] == "local"

    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager.output_helper,
        "run_python",
        lambda _script, arguments: calls.append(list(arguments)) or True,
    )
    manager._apply_output_configuration()

    assert calls, "output configuration helper was not invoked"
    latest = calls[-1]
    assert latest[latest.index("--source") + 1] == "local"


def test_arch_processor_parks_windows_away_from_the_phone_return(manager):
    manager.desired_state = manager.desired_store.set_input("android-front")
    assert manager._desired_windows_device() == "android-phone"

    manager.desired_state = manager.desired_store.set_processor("arch")

    assert manager._desired_windows_device() == "arch-webcam"


def test_local_processor_control_receives_explicit_return_ownership(
    manager, monkeypatch
):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager.local_processor_helper,
        "run_python",
        lambda _script, arguments: calls.append(list(arguments)) or True,
    )
    manager.desired_state = manager.desired_store.set_processor("windows")
    manager._apply_local_processor_state()
    assert ["--active", "false"] == calls[-1][2:4]
    assert "--activate-native256" in calls[-1]

    manager.local_processor_helper._process = None
    manager.desired_state = manager.desired_store.set_processor("arch")
    manager._apply_local_processor_state()
    assert ["--active", "true"] == calls[-1][2:4]
    assert "--activate-native256" in calls[-1]


def test_android_output_stays_pending_until_effective_state_matches(manager):
    manager.android_output_retry_timer.stop()
    manager.android_output_inflight_state = manager._android_output_signature()
    mismatch = _android_output_status(applied=False)

    manager._android_output_configuration_finished(True, json.dumps(mismatch))

    assert manager.pending_android_output_configuration is True
    assert manager.android_output_retry_timer.isActive()

    manager.android_output_retry_timer.stop()
    manager.android_output_inflight_state = manager._android_output_signature()
    effective = _android_output_status(applied=True)
    manager._android_output_configuration_finished(True, json.dumps(effective))

    assert manager.pending_android_output_configuration is False
    assert not manager.android_output_retry_timer.isActive()


def test_rapid_android_output_changes_reapply_only_the_newest_signature(
    manager, monkeypatch
):
    old_signature = manager._android_output_signature()
    manager.android_output_inflight_state = old_signature
    manager.desired_state = manager.desired_store.set_transform(
        mirror=True, rotation=270
    )
    old_result = _android_output_status(applied=True)

    manager._android_output_configuration_finished(True, json.dumps(old_result))

    assert manager.pending_android_output_configuration is True
    manager.android_output_retry_timer.stop()
    manager.android_status = _android_output_status(applied=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager.android_output_helper,
        "run_python",
        lambda _script, arguments: calls.append(list(arguments)) or True,
    )
    manager._apply_android_output_configuration()

    assert calls[-1][calls[-1].index("--output-mirror") + 1] == "true"
    assert calls[-1][calls[-1].index("--output-rotation") + 1] == "270"


def test_old_android_module_is_retained_as_pending_not_claimed_applied(
    manager, monkeypatch
):
    calls = []
    manager.android_status = _android_output_status(version="v0.4.5")
    monkeypatch.setattr(
        manager.android_output_helper,
        "run_python",
        lambda script, arguments: calls.append((script, arguments)) or True,
    )

    manager._apply_android_output_configuration()

    assert not calls
    assert manager.pending_android_output_configuration is True


def test_disabled_phone_output_can_be_fully_synchronized(manager):
    manager.desired_state = manager.desired_store.set_output("android-phone", False)
    status = _android_output_status(enabled=False)
    status["output_control"].update(
        {"effective_source": "placeholder", "effective_worker_alive": False}
    )

    assert manager._android_output_applied(
        status, manager._android_output_signature()
    )
    assert manager._android_output_delivery_matches(
        status, manager._android_output_signature()
    )


def test_phone_reconnect_waits_without_rewriting_matching_persisted_config(
    manager, monkeypatch
):
    # Drain the constructor's one-shot whole-topology reconciliation before
    # observing the reconnect-specific request.
    QApplication.instance().processEvents()
    calls: list[bool] = []
    monkeypatch.setattr(
        manager,
        "_apply_android_output_configuration",
        lambda: calls.append(True),
    )

    manager._android_status_parsed(_android_output_status(applied=False), "")
    QApplication.instance().processEvents()

    assert calls == []
    assert manager.pending_android_output_configuration is True


def test_phone_reconnect_reconciles_stale_persisted_config_once(
    manager, monkeypatch
):
    QApplication.instance().processEvents()
    calls: list[bool] = []
    monkeypatch.setattr(
        manager,
        "_apply_android_output_configuration",
        lambda: calls.append(True),
    )
    stale = _android_output_status(applied=False, persisted=True)
    stale["output_control"]["rotation"] = 90

    manager._android_status_parsed(stale, "")
    QApplication.instance().processEvents()

    assert calls == [True]


def test_phone_delivery_status_distinguishes_fallback_off_and_old_module(manager):
    desired = manager.desired_state
    raw = _android_output_status()
    raw["output_control"]["effective_source"] = "raw"
    manager.output_page.set_delivery_status(desired, {}, raw)
    assert manager.output_page.output_pills["android-phone"].text().endswith(
        "RAW FALLBACK"
    )
    assert not manager._android_output_delivery_matches(
        raw, manager._android_output_signature()
    )

    manager.desired_state = manager.desired_store.set_output("android-phone", False)
    disabled = _android_output_status(enabled=False)
    manager.output_page.set_delivery_status(manager.desired_state, {}, disabled)
    assert manager.output_page.output_pills["android-phone"].text().endswith(
        "DELIVERY OFF · SYNCED"
    )

    old = _android_output_status(version="v0.4.5")
    manager.output_page.set_delivery_status(desired, {}, old)
    assert manager.output_page.output_pills["android-phone"].text().endswith(
        "MODULE UPDATE REQUIRED"
    )


def test_matching_config_with_dead_selector_is_not_rewritten(manager, monkeypatch):
    calls = []
    manager.android_status = _android_output_status(
        applied=False, selector_running=False
    )
    monkeypatch.setattr(
        manager.android_output_helper,
        "run_python",
        lambda script, arguments: calls.append((script, arguments)) or True,
    )

    manager._apply_android_output_configuration()

    assert calls == []
    assert manager.pending_android_output_configuration is True


def test_disabled_module_with_stale_config_is_not_written(manager, monkeypatch):
    calls = []
    manager.android_status = _android_output_status(
        applied=False, module_enabled=False, selector_running=False
    )
    manager.android_status["output_control"]["rotation"] = 90
    monkeypatch.setattr(
        manager.android_output_helper,
        "run_python",
        lambda script, arguments: calls.append((script, arguments)) or True,
    )

    manager._apply_android_output_configuration()

    assert calls == []
    assert manager.pending_android_output_configuration is True


def test_failed_android_output_command_is_not_logged_as_accepted(manager):
    manager.android_output_inflight_state = manager._android_output_signature()

    manager._android_output_configuration_finished(False, "permission denied")

    assert "failed" in manager.log_box.toPlainText().lower()
    assert "permission denied" in manager.log_box.toPlainText().lower()
    assert "accepted" not in manager.log_box.toPlainText().lower()


# ----------------------------------------------------------------- navigation


def test_page_changes_never_start_or_stop_a_decoder(manager):
    manager.calls.starts = 0
    manager.calls.stops = 0

    for index in range(manager.workspace_stack.count()):
        manager.show_workspace(index)
        assert manager.workspace_stack.currentIndex() == index

    assert manager.calls.starts == 0
    assert manager.calls.stops == 0


def test_output_clicks_never_restart_preview_or_camera_owners(manager, monkeypatch):
    manager.calls.starts = 0
    manager.calls.stops = 0
    camera_calls: list[object] = []
    monkeypatch.setattr(manager, "_apply_output_configuration", lambda: None)
    monkeypatch.setattr(manager, "_apply_android_output_configuration", lambda: None)
    monkeypatch.setattr(
        manager.camera_helper,
        "run_python",
        lambda *arguments: camera_calls.append(arguments) or True,
    )

    manager.set_output_transform(True, 90)
    manager.set_output_enabled("android-phone", False)
    manager.set_output_enabled("arch-camera", False)

    assert manager.calls.starts == 0
    assert manager.calls.stops == 0
    assert camera_calls == []


def test_live_previews_stay_alive_across_navigation_and_refreshes(manager):
    manager.input_decoder.running = True
    manager.output_decoder.running = True

    for index in range(manager.workspace_stack.count()):
        manager.show_workspace(index)
        manager.refresh_stats()
        assert manager.output_decoder.running is True

    assert manager.output_decoder.running is True


def test_a_refresh_preserves_the_selected_page_and_splitter_sizes(manager):
    manager.show_workspace(2)
    live_sizes = manager.output_page.splitter.sizes()

    for _ in range(3):
        manager.refresh_stats()

    assert manager.workspace_stack.currentIndex() == 2
    assert manager.output_page.splitter.sizes() == live_sizes


# ------------------------------------------------------------------- geometry


@pytest.mark.parametrize("size", [(980, 680), (1500, 900), (1920, 1080)])
def test_the_window_lays_out_at_the_supported_sizes(manager, size):
    width, height = size
    manager.show()
    QApplication.instance().processEvents()
    manager.resize(width, height)
    QApplication.instance().processEvents()
    for index in range(manager.workspace_stack.count()):
        manager.show_workspace(index)
        manager.refresh_stats()
        QApplication.instance().processEvents()

    assert manager.minimumSize().width() == 980
    assert manager.minimumSize().height() == 680
    # The layout itself has to fit the documented minimum, otherwise Qt would
    # silently refuse to let the window be that small.
    hint = manager.minimumSizeHint()
    assert hint.width() <= 980, hint
    assert hint.height() <= 680, hint
    page = manager.workspace_stack.currentWidget()
    assert page.width() > 0 and page.height() > 0


def test_input_cards_reflow_into_a_grid_on_wide_windows(manager):
    manager.show_workspace(1)
    grid = manager.input_page.input_grid
    manager.resize(1900, 1000)
    manager.show()
    QApplication.instance().processEvents()
    wide = grid.columns()

    manager.resize(980, 680)
    QApplication.instance().processEvents()
    narrow = grid.columns()

    assert wide >= 2
    assert narrow >= 1
    assert wide >= narrow


# ------------------------------------------------------------ ownership safety


def test_no_preview_reader_addresses_a_camera_device(manager):
    for decoder in (manager.input_decoder, manager.output_decoder):
        rendered = " ".join(decoder.command)
        assert "/dev/video" not in rendered
        assert "v4l2" not in rendered
        assert rendered.count(" -i ") == 1
        assert "udp://127.0.0.1:" in rendered


def test_preview_readers_use_dedicated_relays_not_the_system_mux_port(manager):
    raw = " ".join(manager.input_decoder.command)
    result = " ".join(manager.output_decoder.command)

    assert f"udp://127.0.0.1:{manager.preview_port}?" in raw
    assert ":11000" not in raw
    assert manager.preview_port == 11001
    assert manager.active_output_preview_port in (11003, 11007)
    assert f"udp://127.0.0.1:{manager.active_output_preview_port}?" in result


def test_switching_the_result_relay_never_restarts_a_camera_owner(manager):
    manager.calls.stops = 0
    manager._set_output_preview_source(local_processed=True)
    assert manager.active_output_preview_port == manager.local_processed_preview_port

    manager._set_output_preview_source(local_processed=False)
    assert manager.active_output_preview_port == manager.output_preview_port
    # Only this manager's own reader is recycled by the switch.
    assert manager.calls.stops <= 2


def test_the_ui_offers_no_general_service_start_stop_or_restart(manager):
    from PySide6.QtWidgets import QAbstractButton

    # These patterns detect service lifecycle buttons, not recording controls.
    forbidden = ("start service", "stop camera", "restart service", "restart camera", "start camera")
    # Allow-listed labels that contain "start" but are not service controls.
    allowed = ("start recording",)
    for button in manager.findChildren(QAbstractButton):
        label = button.text().lower()
        if any(word in label for word in allowed):
            continue
        assert not any(word in label for word in forbidden), label


def test_manager_has_no_privileged_legacy_mapping_action(manager):
    assert not hasattr(manager, "repair_device_mapping")
    assert not hasattr(manager.system_page, "repair_button")
    source = Path(manager_shell.__file__).read_text(encoding="utf-8")
    assert "device_control.py" not in source
    assert "shadow device" not in source


def test_selecting_a_system_camera_policy_uses_the_receiver_socket_helper(
    monkeypatch, manager
):
    ran: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        manager_services.HelperProcess,
        "run_python",
        lambda self, script, arguments: ran.append((script, list(arguments))) or True,
    )

    manager.select_system_camera_policy("raw")

    assert len(ran) == 1
    script, arguments = ran[0]
    assert script.endswith("select_receiver_source.py")
    assert arguments[0] == "raw"
    assert arguments[1] == "--socket"
    assert arguments[2].endswith("receiver-control.sock")


def test_a_successful_policy_change_states_that_nothing_was_restarted(manager):
    manager.pending_policy = "local"
    manager._policy_finished(True, "")

    message = manager.routing_page.policy_result.text()
    assert "not restarted" in message
    assert "/dev/deep-live-cam" in message


# ------------------------------------------------------------------- overview


def test_the_header_summarises_route_processor_and_system_camera(manager):
    manager.refresh_stats()

    assert set(manager.ribbon_cells) == {"input", "processor", "output"}
    for value, pill in manager.ribbon_cells.values():
        assert value.text()
        assert pill.text()
    # Colour is always paired with a word.
    assert len(manager.alert_pill.text().split(" ", 1)) == 2


def test_header_does_not_claim_healthy_output_from_arch_when_only_phone_is_enabled(
    manager,
):
    manager.desired_state = manager.desired_store.set_output("arch-camera", False)
    manager.android_status = {"available": False}

    manager.refresh_stats()

    assert manager.ribbon_cells["output"][1].text().endswith("SYNCING / OFFLINE")


def test_an_idle_windows_client_is_not_reported_as_switching(manager):
    manager.refresh_stats()

    selected = next(slot for slot in manager.view.slots if slot.selected)
    assert selected.state != "switching"


def test_the_diagnostic_snapshot_records_that_no_camera_is_opened(manager):
    manager.refresh_stats()

    assert manager.snapshot["opens_camera_device"] is False
    assert "route" in manager.snapshot
    assert len(manager.snapshot["slots"]) == 5
    assert manager.stats_box.toPlainText().startswith("ACTIVE ROUTE")


def test_the_event_log_is_bounded(manager):
    for index in range(LOG_LINE_LIMIT * 3):
        manager.append_log(f"line {index}")

    assert LOG_LINE_LIMIT <= 1000
    assert manager.log_box.document().blockCount() <= LOG_LINE_LIMIT


def test_keyboard_navigation_and_focus_are_preserved(manager):
    from PySide6.QtCore import Qt

    manager.show()
    QApplication.instance().processEvents()
    assert manager.workspace_tabs.tabBar().focusPolicy() != Qt.FocusPolicy.NoFocus
    for button in manager.input_page.input_buttons.values():
        assert button.focusPolicy() == Qt.FocusPolicy.StrongFocus

    style = manager.styleSheet()
    for rule in (
        "QPushButton:focus",
        "QComboBox:focus",
        "QTabBar::tab:focus",
    ):
        assert rule in style, rule


def test_a_refresh_never_steals_focus_or_resets_an_edit(manager):
    from PySide6.QtCore import Qt

    manager.show()
    manager.show_workspace(0)
    QApplication.instance().processEvents()
    # An always-enabled control, so the assertion is about the refresh rather
    # than about a control that a disconnected processor legitimately disables.
    delay = manager.processing_page.win_detection_interval.slider
    delay.setFocus(Qt.FocusReason.OtherFocusReason)
    delay.setValue(4)
    manager.processing_page.win_opacity.setValue(37)
    QApplication.instance().processEvents()

    for _ in range(4):
        manager.refresh_stats()
        QApplication.instance().processEvents()

    # Edits survive, on the visible page and on a background one.
    assert manager.processing_page.win_detection_interval.value() == 4
    assert manager.processing_page.win_opacity.value() == 37
    # Window-scoped focus is deterministic off-screen, unlike the
    # application-wide focus widget, which depends on window activation.
    assert manager.focusWidget() is delay


def test_a_refresh_does_not_rebuild_the_input_cards_or_camera_form(manager):
    manager.refresh_stats()
    cards = list(manager.input_page.input_buttons.values())
    controls = dict(manager.input_page.camera_controls)

    for _ in range(3):
        manager.refresh_stats()

    assert [id(card) for card in manager.input_page.input_buttons.values()] == [
        id(card) for card in cards
    ]
    assert {key: id(widget) for key, widget in manager.input_page.camera_controls.items()} == {
        key: id(widget) for key, widget in controls.items()
    }


def _android_slot_registry() -> dict:
    return {
        "selected_device_id": "android-phone",
        "slots": [
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
        ],
    }


def test_the_raw_reader_runs_whenever_the_route_actually_has_a_raw_copy(manager):
    manager.input_decoder.running = False
    manager.refresh_stats()

    assert manager.view.stream("raw").available is True
    assert manager.input_decoder.running is True


def test_the_raw_reader_is_stopped_only_when_there_is_nothing_to_read(manager):
    manager.registry_live = True
    manager.windows_devices = _android_slot_registry()
    manager.windows_health = {
        "healthy": True,
        "state": "streaming-face-swap",
        "selected_device_id": "android-phone",
        "input": {"streaming": True, "last_frame_age": 0.05},
        "output": {"streaming": True, "last_frame_age": 0.05, "port": 10001},
        "processing": {},
    }

    manager.refresh_stats()

    assert manager.view.route.key == "windows-android"
    assert manager.view.stream("raw").available is False
    assert manager.input_decoder.running is False
    # The pane says why rather than showing a substitute picture.
    assert "never opens" in manager.view.stream("raw").note


def test_no_page_is_wider_than_the_minimum_window(manager):
    """A page that cannot narrow would be clipped: these areas never scroll
    sideways, so their content has to re-flow instead."""
    from PySide6.QtWidgets import QScrollArea

    manager.show()
    QApplication.instance().processEvents()
    manager.resize(980, 680)
    QApplication.instance().processEvents()

    available = manager.workspace_stack.width()
    assert available > 0
    for index in range(manager.workspace_stack.count()):
        manager.show_workspace(index)
        manager.refresh_stats()
        QApplication.instance().processEvents()
        page = manager.workspace_stack.widget(index)
        scroll = page.findChild(QScrollArea, "workspaceScroll")
        if scroll is None:
            widest = page.minimumSizeHint().width()
        else:
            widest = scroll.widget().minimumSizeHint().width()
        assert widest <= available, (index, widest, available)


def test_card_grids_stack_at_the_minimum_width_and_spread_when_wide(manager):
    manager.show()
    QApplication.instance().processEvents()
    grids = {
        "inputs": manager.input_page.input_grid,
        "outputs": manager.output_page.output_grid,
        "processing top": manager.processing_page.top_grid,
    }
    manager.resize(980, 680)
    for index in range(manager.workspace_stack.count()):
        manager.show_workspace(index)
        QApplication.instance().processEvents()
    narrow = {name: grid.columns() for name, grid in grids.items()}

    manager.resize(1900, 1000)
    for index in range(manager.workspace_stack.count()):
        manager.show_workspace(index)
        QApplication.instance().processEvents()
    wide = {name: grid.columns() for name, grid in grids.items()}

    for name in grids:
        assert wide[name] >= narrow[name], (name, narrow[name], wide[name])
    assert wide["inputs"] >= 2
