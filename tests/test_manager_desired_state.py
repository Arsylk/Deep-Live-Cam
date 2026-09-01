"""Durable one-way desired-state contract for both processors and outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from dlc_manager.desired_state import (  # noqa: E402
    INPUT_ANDROID_BACK,
    OUTPUT_ARCH_CAMERA,
    PROCESSOR_ARCH,
    PROCESSOR_SPECS,
    PROCESSOR_WINDOWS,
    DesiredStateStore,
    local_processor_payload,
    local_processor_matches,
    normalize_processing,
    processor_payload,
    reconciliation_payload,
)


def test_processors_have_fixed_distinct_models_and_truthful_quality_labels():
    windows = PROCESSOR_SPECS[PROCESSOR_WINDOWS]
    arch = PROCESSOR_SPECS[PROCESSOR_ARCH]

    assert windows.model != arch.model
    assert windows.backend == "CUDA"
    assert "256" in windows.model
    assert "native 256" in windows.detail
    assert "256" in arch.model
    assert "NCNN" in arch.backend
    assert "development" in arch.detail.lower()


def test_every_click_persists_one_shared_document_atomically(tmp_path):
    path = tmp_path / "manager-state.json"
    store = DesiredStateStore(path)

    first = store.set_processing("opacity", 0.42)
    second = store.set_processor(PROCESSOR_ARCH)
    third = store.set_input(INPUT_ANDROID_BACK)
    fourth = store.set_output(OUTPUT_ARCH_CAMERA, False)
    final = store.set_transform(mirror=True, rotation=270)

    reloaded = DesiredStateStore(path).snapshot()
    assert [first["revision"], second["revision"], third["revision"], fourth["revision"], final["revision"]] == [1, 2, 3, 4, 5]
    assert reloaded == final
    assert reloaded["processing"]["opacity"] == pytest.approx(0.42)
    assert reloaded["processor"] == PROCESSOR_ARCH
    assert reloaded["input"] == INPUT_ANDROID_BACK
    assert reloaded["outputs"][OUTPUT_ARCH_CAMERA] is False
    assert reloaded["output_transform"] == {"mirror": True, "rotation": 270}
    assert not path.with_suffix(".tmp").exists()


def test_reconnect_payload_pushes_desired_values_without_adopting_stale_remote(tmp_path):
    store = DesiredStateStore(tmp_path / "state.json")
    desired = store.set_processing_values(
        {"opacity": 0.65, "quality_mode": "strict", "show_fps": True}
    )
    stale_remote = processor_payload(desired)
    stale_remote.update({"opacity": 0.2, "quality_mode": "monitor"})

    patch = reconciliation_payload(desired, stale_remote)

    assert patch == {"opacity": 0.65, "quality_mode": "strict"}
    assert desired["processing"]["show_fps"] is True


def test_shared_repair_profile_and_hidden_orientation_invariant_reconcile(tmp_path):
    store = DesiredStateStore(tmp_path / "state.json")
    desired = store.snapshot()
    remote = processor_payload(desired)
    remote.update(
        {
            "repair_hf_strength": 0.0,
            "repair_boundary_mask": False,
            "repair_boundary_strength": 0.0,
            "live_mirror": True,
            "processing_off_output": "black",
        }
    )

    patch = reconciliation_payload(desired, remote)

    assert patch == {
        "repair_hf_strength": 0.3,
        "repair_boundary_mask": True,
        "repair_boundary_strength": 0.35,
        "live_mirror": False,
        "processing_off_output": "passthrough",
    }


def test_legacy_strict_profile_migrates_measured_final_resolution_repairs():
    migrated = normalize_processing({"quality_mode": "strict"})

    assert migrated["repair_camera_detail"] == pytest.approx(3.5)
    assert migrated["repair_boundary_strength"] == pytest.approx(0.5)


def test_restarted_local_processor_is_not_mistaken_for_synchronized(tmp_path):
    store = DesiredStateStore(tmp_path / "state.json")
    desired = store.snapshot()
    source = tmp_path / "source.jpg"
    effective_processing = processor_payload(desired, PROCESSOR_ARCH)
    effective_processing["source_path"] = str(source)
    synchronized = {
        "state": "running",
        "control": {
            "in_sync": True,
            "effective": {
                "active": False,
                "input": "android-front",
                "model": {
                    "swapper_model": "native-256",
                    "swapper_backend": "ncnn",
                    "configured": True,
                    "ready": True,
                },
                "processing": effective_processing,
            },
        },
    }

    assert local_processor_matches(desired, synchronized, source_path=source)

    restarted = {
        **synchronized,
        "control": {
            **synchronized["control"],
            "in_sync": False,
            "effective": {
                **synchronized["control"]["effective"],
                "model": {
                    "swapper_model": "inswapper-128",
                    "swapper_backend": "ncnn",
                    "configured": False,
                    "ready": False,
                },
            },
        },
    }

    assert not local_processor_matches(desired, restarted, source_path=source)


def test_inactive_arch_standby_does_not_require_a_resident_model(tmp_path):
    desired = DesiredStateStore(tmp_path / "state.json").snapshot()
    processing = processor_payload(desired, PROCESSOR_ARCH)
    health = {
        "state": "running",
        "control": {
            "in_sync": True,
            "effective": {
                "active": False,
                "input": desired["input"],
                "model": {
                    "swapper_model": "native-256",
                    "swapper_backend": "ncnn",
                    "configured": True,
                    "ready": False,
                },
                "processing": processing,
            },
        },
    }

    assert local_processor_matches(desired, health)

    active = DesiredStateStore(tmp_path / "active.json").set_processor_processing(
        PROCESSOR_ARCH, True
    )
    health["control"]["effective"]["active"] = True
    health["control"]["effective"]["processing"] = processor_payload(
        active, PROCESSOR_ARCH
    )
    assert not local_processor_matches(active, health)


def test_inactive_legacy_arch_worker_receives_only_advertised_fields(tmp_path):
    desired = DesiredStateStore(tmp_path / "state.json").snapshot()
    complete = processor_payload(desired, PROCESSOR_ARCH)
    legacy_fields = set(complete) - {
        "repair_boundary_strength",
        "repair_camera_detail",
    }
    legacy_processing = {
        key: value for key, value in complete.items() if key in legacy_fields
    }
    health = {
        "state": "running",
        "control": {
            "in_sync": True,
            "capabilities": {
                "enhancers": ["none"],
                "processing_fields": sorted(legacy_fields),
            },
            "effective": {
                "active": False,
                "input": desired["input"],
                "model": {
                    "swapper_model": "native-256",
                    "swapper_backend": "ncnn",
                    "configured": True,
                    "ready": False,
                },
                "processing": legacy_processing,
            },
        },
    }

    payload = local_processor_payload(desired, health)

    assert "repair_boundary_strength" not in payload
    assert "repair_camera_detail" not in payload
    assert local_processor_matches(desired, health)

    active = DesiredStateStore(tmp_path / "active.json").set_processor(
        PROCESSOR_ARCH
    )
    assert "repair_boundary_strength" in local_processor_payload(active, health)
    assert not local_processor_matches(active, health)


def test_inactive_arch_standby_accepts_advertised_enhancer_clamp(tmp_path):
    store = DesiredStateStore(tmp_path / "state.json")
    desired = store.set_processing("enhancer", "gfpgan")
    processing = processor_payload(desired, PROCESSOR_ARCH)
    processing["enhancer"] = "none"
    health = {
        "state": "running",
        "control": {
            "in_sync": False,
            "capabilities": {"enhancers": ["none"]},
            "effective": {
                "active": False,
                "input": desired["input"],
                "model": {
                    "swapper_model": "native-256",
                    "swapper_backend": "ncnn",
                    "configured": True,
                    "ready": False,
                },
                "processing": processing,
            },
        },
    }

    assert local_processor_matches(desired, health)

    active = store.set_processor_processing(PROCESSOR_ARCH, True)
    health["control"]["effective"]["active"] = True
    health["control"]["effective"]["processing"] = {
        **processor_payload(active, PROCESSOR_ARCH),
        "enhancer": "none",
    }
    assert not local_processor_matches(active, health)


def test_processor_enablement_is_atomic_exclusive_and_target_aware(tmp_path):
    path = tmp_path / "state.json"
    store = DesiredStateStore(path)

    # A stale OFF event from the already-off peer is a no-op.
    ignored = store.set_processor_processing(PROCESSOR_ARCH, False)
    assert ignored["revision"] == 0
    assert ignored["processor"] == PROCESSOR_WINDOWS

    passthrough = store.set_processor_processing(PROCESSOR_WINDOWS, False)
    assert passthrough["revision"] == 1
    assert passthrough["processor"] == PROCESSOR_WINDOWS
    assert passthrough["processing"]["processing_mode"] == "passthrough"

    arch = store.set_processor_processing(PROCESSOR_ARCH, True)
    assert arch["revision"] == 2
    assert arch["processor"] == PROCESSOR_ARCH
    assert arch["processing"]["processing_mode"] == "face_swap"
    assert processor_payload(arch, PROCESSOR_ARCH)["processing_mode"] == "face_swap"
    assert (
        processor_payload(arch, PROCESSOR_WINDOWS)["processing_mode"]
        == "passthrough"
    )
    assert DesiredStateStore(path).snapshot() == arch


def test_windows_reconciliation_always_disables_the_non_selected_host(tmp_path):
    store = DesiredStateStore(tmp_path / "state.json")
    desired = store.set_processor_processing(PROCESSOR_ARCH, True)

    patch = reconciliation_payload(
        desired,
        {**processor_payload(desired, PROCESSOR_WINDOWS), "processing_mode": "face_swap"},
    )

    assert patch["processing_mode"] == "passthrough"


@pytest.mark.parametrize("rotation", [-90, 45, 360])
def test_invalid_output_rotations_never_replace_last_valid_state(tmp_path, rotation):
    store = DesiredStateStore(tmp_path / "state.json")
    before = store.set_transform(mirror=False, rotation=90)

    with pytest.raises(ValueError):
        store.set_transform(mirror=True, rotation=rotation)

    assert store.snapshot() == before


def test_prerecorded_adjust_defaults_and_persists(tmp_path):
    path = tmp_path / "state.json"
    store = DesiredStateStore(path)

    # Fresh document carries neutral framing.
    fresh = store.snapshot()
    assert fresh["prerecorded_adjust"] == {
        "offset_x": 0,
        "offset_y": 0,
        "zoom": 1.0,
    }

    updated = store.set_prerecorded_adjust(offset_x=120, offset_y=-40, zoom=1.5)
    assert updated["prerecorded_adjust"] == {
        "offset_x": 120,
        "offset_y": -40,
        "zoom": 1.5,
    }
    # Partial update keeps the other axes.
    partial = store.set_prerecorded_adjust(zoom=2.0)
    assert partial["prerecorded_adjust"] == {
        "offset_x": 120,
        "offset_y": -40,
        "zoom": 2.0,
    }
    assert DesiredStateStore(path).snapshot()["prerecorded_adjust"]["zoom"] == 2.0


def test_prerecorded_adjust_clamps_zoom_to_supported_range(tmp_path):
    store = DesiredStateStore(tmp_path / "state.json")

    assert store.set_prerecorded_adjust(zoom=99.0)["prerecorded_adjust"]["zoom"] == 4.0
    assert store.set_prerecorded_adjust(zoom=0.01)["prerecorded_adjust"]["zoom"] == 0.25
