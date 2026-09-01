"""What each control actually does, asserted off-screen without a camera."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from camera_adapters import camera_schema  # noqa: E402
from camera_profiles import (  # noqa: E402
    CUSTOM_CAMERA_PROFILE,
    DEFAULT_CAMERA_PROFILE,
    profile_live_values,
)
from dlc_manager.pages.processing import ProcessingPage  # noqa: E402
from dlc_manager.pages.input import InputPage  # noqa: E402
from dlc_manager.pages.output import OutputPage  # noqa: E402
from dlc_manager.pages.routing import RoutingPage  # noqa: E402
from dlc_manager.desired_state import (  # noqa: E402
    PROCESSOR_ARCH,
    PROCESSOR_WINDOWS,
    DesiredStateStore,
)
from dlc_manager.services import (  # noqa: E402
    adapter_arguments,
    arch_persist_arguments,
)
from dlc_manager.shell import PRESETS  # noqa: E402
from dlc_manager.viewmodel import ViewInputs, build_view  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _application():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app


@pytest.fixture()
def processing():
    page = ProcessingPage()
    page.set_windows_available(True)
    yield page
    page.deleteLater()


@pytest.fixture()
def routing():
    page = RoutingPage()
    page.rebuild_camera_controls(
        device_id="arch-webcam",
        label="Arch USB webcam",
        stack="arch-v4l2",
        schema=camera_schema("arch-v4l2"),
        defaults={},
    )
    yield page
    page.deleteLater()


@pytest.fixture()
def output():
    page = OutputPage()
    yield page
    page.deleteLater()


# ------------------------------------------------------------ processing page


def test_strict_preset_uses_measured_flow_bridge_and_boundary_profile():
    strict = PRESETS[3]

    assert strict["detection_interval"] == 2
    assert strict["repair_boundary_strength"] == pytest.approx(0.5)


def test_every_everyday_control_reports_the_field_the_processor_expects(processing):
    changes: list[tuple[str, object]] = []
    processing.settingChanged.connect(lambda field, value: changes.append((field, value)))

    processing.win_opacity.setValue(55)
    processing.win_color_match.setValue(20)
    processing.win_mouth_mask.setValue(12)
    processing.win_sharpness.setValue(25)
    processing.win_many_faces.setChecked(True)

    assert changes == [
        ("opacity", 0.55),
        ("color_match_strength", 0.2),
        ("mouth_mask_size", 12.0),
        ("sharpness", 2.5),
        ("many_faces", True),
    ]


def test_every_advanced_control_reports_its_own_field(processing):
    changes: dict[str, object] = {}
    processing.settingChanged.connect(lambda field, value: changes.__setitem__(field, value))

    processing.win_tracking_enabled.setChecked(False)
    processing.win_detection_interval.setValue(3)
    processing.win_tracking_smoothing.setValue(40)
    processing.win_tracking_grace.setValue(7)
    processing.win_detection_score.setValue(60)
    processing.win_minimum_face_size.setValue(96)
    processing.win_enhancer.setCurrentIndex(1)
    processing.win_enable_interpolation.setChecked(True)
    processing.win_interpolation_weight.setValue(80)
    processing.win_show_fps.setChecked(True)
    processing.win_quality_mode.setCurrentIndex(2)
    processing.win_quality_auto_correct.setChecked(False)
    processing.win_repair_hf.setValue(25)
    processing.win_repair_checkerboard.setValue(35)
    processing.win_repair_wavelet.setValue(45)
    processing.win_repair_camera_detail.setValue(350)
    processing.win_repair_boundary.setChecked(False)
    processing.win_repair_boundary_strength.setValue(50)

    assert changes == {
        "tracking_enabled": False,
        "detection_interval": 3,
        "tracking_smoothing": 0.4,
        "tracking_grace_frames": 7,
        "minimum_detection_score": 0.6,
        "minimum_face_size": 96,
        "enhancer": "gfpgan",
        "enable_interpolation": True,
        "interpolation_weight": 0.8,
        "show_fps": True,
        "quality_mode": "strict",
        "quality_auto_correct": False,
        "repair_hf_strength": 0.25,
        "repair_checkerboard": 0.35,
        "repair_wavelet": 0.45,
        "repair_camera_detail": 3.5,
        "repair_boundary_mask": False,
        "repair_boundary_strength": 0.5,
    }


def test_loading_processor_values_does_not_echo_them_back(processing):
    changes: list[tuple[str, object]] = []
    processing.settingChanged.connect(lambda field, value: changes.append((field, value)))

    processing.apply_windows_config(
        {
            "opacity": 0.4,
            "quality_mode": "strict",
            "active_swapper_model": "inswapper-128",
            "active_swapper_backend": "cuda",
            "repair_hf_strength": 0.25,
            "repair_boundary_mask": False,
        }
    )

    assert changes == []
    assert processing.win_opacity.value() == 40
    assert processing.win_repair_hf.value() == 25
    assert processing.win_repair_boundary.isChecked() is False


def test_processor_tab_has_no_second_face_swap_mode_control(processing):
    assert "processing_mode" not in processing.setting_widgets()
    assert not hasattr(processing, "win_processing_mode")


def test_output_processor_switches_load_initial_and_passthrough_without_echo(output):
    toggles: list[tuple[str, bool]] = []
    output.processorProcessingToggled.connect(
        lambda processor, enabled: toggles.append((processor, enabled))
    )
    store = DesiredStateStore(Path("/definitely/not/read"))

    output.set_output_state(store.snapshot())

    assert output.processor_processing[PROCESSOR_WINDOWS].isChecked() is True
    assert output.processor_processing[PROCESSOR_ARCH].isChecked() is False
    assert "Active target" in output.processor_processing_status[PROCESSOR_WINDOWS].text()
    assert "Click once" in output.processor_processing_status[PROCESSOR_ARCH].text()
    assert toggles == []

    document = store.snapshot()
    document["processing"]["processing_mode"] = "passthrough"
    output.set_output_state(document)

    assert not any(toggle.isChecked() for toggle in output.processor_processing.values())
    assert "Selected target" in output.processor_processing_status[PROCESSOR_WINDOWS].text()
    assert toggles == []


def test_output_windows_switch_enables_and_disables_in_one_click(output):
    changes: list[tuple[str, bool]] = []
    output.processorProcessingToggled.connect(
        lambda processor, enabled: changes.append((processor, enabled))
    )
    document = DesiredStateStore(Path("/definitely/not/read" )).snapshot()
    document["processing"]["processing_mode"] = "passthrough"
    output.set_output_state(document)

    output.processor_processing[PROCESSOR_WINDOWS].click()
    output.processor_processing[PROCESSOR_WINDOWS].click()

    assert changes == [
        (PROCESSOR_WINDOWS, True),
        (PROCESSOR_WINDOWS, False),
    ]
    assert not any(toggle.isChecked() for toggle in output.processor_processing.values())


def test_output_enabling_arch_is_exclusive_and_does_not_emit_peer_signal(output):
    changes: list[tuple[str, bool]] = []
    output.processorProcessingToggled.connect(
        lambda processor, enabled: changes.append((processor, enabled))
    )
    output.set_output_state(DesiredStateStore(Path("/definitely/not/read")).snapshot())

    output.processor_processing[PROCESSOR_ARCH].click()

    assert output.processor_processing[PROCESSOR_ARCH].isChecked() is True
    assert output.processor_processing[PROCESSOR_WINDOWS].isChecked() is False
    assert changes == [(PROCESSOR_ARCH, True)]


def test_output_processor_switch_persists_host_and_mode_atomically(output, tmp_path):
    store = DesiredStateStore(tmp_path / "manager-state.json")
    revisions: list[int] = []

    def persist(processor: str, enabled: bool) -> None:
        revisions.append(store.set_processor_processing(processor, enabled)["revision"])

    output.processorProcessingToggled.connect(persist)
    output.set_output_state(store.snapshot())

    output.processor_processing[PROCESSOR_ARCH].click()

    saved = DesiredStateStore(tmp_path / "manager-state.json").snapshot()
    assert revisions == [1]
    assert saved["processor"] == PROCESSOR_ARCH
    assert saved["processing"]["processing_mode"] == "face_swap"


def test_output_pipeline_strip_renders_health_evidence_not_stream_inference(output):
    health = {
        "healthy": True,
        "state": "streaming-face-swap",
        "processing": {
            "mode": "face_swap",
            "active_swapper_model": "inswapper-128",
            "active_swapper_backend": "cuda",
            "fps": 17.25,
            "tracking": {"active": True, "valid_detections": 1},
            "quality": {"swap_applied": False},
            "last_error": None,
        },
    }
    view = build_view(ViewInputs(windows_health=health))

    output.render(view, phone_return_live=False)

    rows = output.pipeline_metrics.rows
    assert rows["Face detected"].value.full_text() == "YES"
    assert rows["Face swapped"].value.full_text() == "NO"
    assert rows["Model / backend"].value.full_text() == "inswapper-128 / cuda"
    assert rows["Processing FPS"].value.full_text() == "17.2 FPS"
    assert "no swap applied" in rows["Error / waiting reason"].value.full_text()

    # A live stream with no processor evidence must remain UNKNOWN; output
    # frame presence is deliberately not used to manufacture a YES or NO.
    output.render(
        build_view(
            ViewInputs(
                windows_health={
                    "healthy": True,
                    "state": "streaming-face-swap",
                    "processing": {
                        "mode": "face_swap",
                        "active_swapper_model": "inswapper-128",
                        "active_swapper_backend": "cuda",
                    },
                }
            )
        ),
        phone_return_live=False,
    )
    assert rows["Face detected"].value.full_text() == "UNKNOWN"
    assert rows["Face swapped"].value.full_text() == "UNKNOWN"


def test_output_never_calls_stale_or_missing_arch_health_a_live_stream(output):
    desired = {
        "processor": "windows",
        "outputs": {"arch-camera": True, "android-phone": False},
        "output_transform": {"mirror": False, "rotation": 0},
    }
    receiver = {
        "virtual_camera": "/dev/deep-live-cam",
        "virtual_cameras": ["/dev/deep-live-cam"],
        "source_mode": "windows",
        "output_enabled": True,
        "output_transform": {"mirror": False, "rotation": 0},
        "status": "streaming",
        "sink_pid": 4242,
    }

    output.set_delivery_status(
        desired,
        receiver,
        {},
        receiver_service_active=True,
        receiver_health_age=None,
    )
    assert output.output_pills["arch-camera"].text().endswith(
        "CONFIG SYNCED · WAITING"
    )

    output.set_delivery_status(
        desired,
        receiver,
        {},
        receiver_service_active=True,
        receiver_health_age=0.1,
    )
    assert output.output_pills["arch-camera"].text().endswith("STREAM LIVE")


def test_changing_a_value_moves_the_preset_back_to_custom(processing):
    processing.set_preset_index(2)
    processing.win_opacity.setValue(70)

    assert processing.processing_preset.currentIndex() == 0


def test_targeting_the_local_engine_keeps_the_shared_controls_editable(processing):
    selected: list[str] = []
    processing.processorChanged.connect(selected.append)
    processing.processor_target.setCurrentIndex(1)

    assert all(control.isEnabled() for control in processing.windows_controls())
    assert selected == ["arch"]
    note = processing.target_note.text()
    assert "development" in note.lower()
    assert "native-256" in processing.processor_model.text().lower()

    processing.processor_target.setCurrentIndex(0)
    assert all(control.isEnabled() for control in processing.windows_controls())


def test_an_unreachable_processor_keeps_edits_enabled_and_explains_reconciliation(processing):
    processing.set_windows_available(False, "Connection refused")

    assert all(control.isEnabled() for control in processing.windows_controls())
    assert "Connection refused" in processing.target_note.text()
    assert "automatically" in processing.target_note.text()


def test_the_history_strip_reports_the_identifier_it_was_given(processing, tmp_path):
    from source_history import SourceHistoryEntry

    entries = [
        SourceHistoryEntry(
            identifier=f"id-{index}",
            filename=f"face-{index}.png",
            cache_path=tmp_path / f"face-{index}.png",
            used_at=float(index),
        )
        for index in range(8)
    ]
    chosen: list[str] = []
    processing.historyPictureRequested.connect(chosen.append)

    processing.rebuild_history(entries, "id-3")

    assert len(processing.history_buttons) == 8
    assert processing.history_buttons[3].isChecked() is True
    processing.history_buttons[5].click()
    assert chosen == ["id-5"]


# --------------------------------------------------------------- routing page


def test_selecting_a_named_profile_loads_the_whole_bundle_atomically(routing):
    profile = routing.camera_controls["profile"]
    profile.setCurrentIndex(profile.findData(CUSTOM_CAMERA_PROFILE))
    routing.set_camera_values({"brightness": 20, "contrast": 50})
    assert routing.camera_values()["brightness"] == 20

    profile.setCurrentIndex(profile.findData(DEFAULT_CAMERA_PROFILE))

    values = routing.camera_values()
    for key, expected in profile_live_values(DEFAULT_CAMERA_PROFILE).items():
        assert values[key] == expected, key
    assert values["profile"] == DEFAULT_CAMERA_PROFILE


def test_editing_one_component_makes_the_visible_profile_custom(routing):
    profile = routing.camera_controls["profile"]
    profile.setCurrentIndex(profile.findData(DEFAULT_CAMERA_PROFILE))

    routing.camera_controls["saturation"].setValue(90)

    assert routing.camera_values()["profile"] == CUSTOM_CAMERA_PROFILE


def test_camera_edits_are_announced_as_session_only_until_saved(routing):
    routing.camera_controls["brightness"].setValue(-30)

    assert "NOT SAVED" in routing.camera_state_pill.text()

    routing.set_camera_status("Saved.", state="saved")
    assert "SAVED" in routing.camera_state_pill.text()


def test_a_generic_client_says_it_has_no_camera_adapter(routing):
    routing.rebuild_camera_controls(
        device_id="other",
        label="Some SRT client",
        stack="generic-srt",
        schema=camera_schema("generic-srt"),
        defaults={},
    )

    assert routing.camera_controls == {}
    assert routing.camera_apply.isEnabled() is False


def test_an_android_client_gets_its_own_capabilities(routing):
    routing.rebuild_camera_controls(
        device_id="android-phone",
        label="Android phone",
        stack="android-camera2",
        schema=camera_schema("android-camera2"),
        defaults={},
    )

    assert set(routing.camera_controls) == {
        "lens_facing",
        "rotation",
        "zoom_percent",
        "exposure_compensation",
        "ae_lock",
        "awb_lock",
        "stabilization",
    }
    assert "brightness" not in routing.camera_controls


def test_input_cards_are_the_only_phone_lens_selector():
    page = InputPage()
    try:
        page.rebuild_camera_controls(
            device_id="android-phone",
            label="Android phone front camera",
            stack="android-camera2",
            schema=camera_schema("android-camera2"),
            defaults={"lens_facing": "front"},
        )

        assert "lens_facing" not in page.camera_controls
        assert {"rotation", "zoom_percent", "exposure_compensation", "stabilization"} <= set(
            page.camera_controls
        )
        assert len(page.input_buttons) == 5
    finally:
        page.deleteLater()


def test_only_assigned_slots_can_be_selected(routing):
    view = build_view(ViewInputs(registry_live=True))
    chosen: list[str] = []
    routing.slotSelected.connect(chosen.append)

    routing.render(view)
    for card in routing.slot_cards:
        card.click()

    assert chosen == []


def test_the_policy_selector_reports_the_receiver_policy_key(routing):
    chosen: list[str] = []
    routing.policySelected.connect(chosen.append)

    routing.policy_buttons["windows"].setChecked(True)

    assert chosen == ["windows"]


def test_showing_the_receivers_own_policy_does_not_request_a_change(routing):
    chosen: list[str] = []
    routing.policySelected.connect(chosen.append)

    view = build_view(
        ViewInputs(receiver={"status": "streaming", "source_mode": "raw"})
    )
    routing.render(view)

    assert routing.policy_buttons["raw"].isChecked() is True
    assert chosen == []


# ---------------------------------------------------------- helper invocation


def test_an_unmodified_profile_is_persisted_as_a_profile():
    values = dict(profile_live_values(DEFAULT_CAMERA_PROFILE))
    values["profile"] = DEFAULT_CAMERA_PROFILE

    arguments = arch_persist_arguments(values)

    assert arguments[0].endswith("configure_camera.py")
    assert arguments[1:] == ["--profile", DEFAULT_CAMERA_PROFILE]


def test_an_edited_profile_is_persisted_as_explicit_values():
    values = dict(profile_live_values(DEFAULT_CAMERA_PROFILE))
    values["profile"] = DEFAULT_CAMERA_PROFILE
    values["saturation"] = 90

    arguments = arch_persist_arguments(values)

    assert "--profile" not in arguments
    assert arguments[arguments.index("--saturation") + 1] == "90"
    assert arguments[arguments.index("--auto-exposure") + 1] == "1"


def test_an_android_preview_never_persists_unless_asked():
    preview = adapter_arguments("android-camera2", {"ae_lock": True}, serial="abc")
    persisted = adapter_arguments(
        "android-camera2", {"ae_lock": True}, serial="abc", persist=True
    )

    assert "--persist" not in preview
    assert "--persist" in persisted
    assert "/dev/video" not in " ".join(persisted)
