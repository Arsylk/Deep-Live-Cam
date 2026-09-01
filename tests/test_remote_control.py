from __future__ import annotations

import hashlib
import json

import cv2
import modules.globals
import numpy as np
import pytest
from windows.modules.remote_control import ControlState


def _state(tmp_path):
    return ControlState(
        str(tmp_path),
        health=lambda: {},
        devices=lambda: {},
        select_device=lambda _device: {},
    )


def _jpeg_bytes(value: int = 80) -> bytes:
    image = np.full((96, 128, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_tracking_configuration_is_clamped_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(modules.globals, "tracking_enabled", True)
    monkeypatch.setattr(modules.globals, "detection_interval", 1)
    monkeypatch.setattr(modules.globals, "tracking_smoothing", 0.65)
    monkeypatch.setattr(modules.globals, "tracking_grace_frames", 5)
    monkeypatch.setattr(modules.globals, "minimum_detection_score", 0.45)
    monkeypatch.setattr(modules.globals, "minimum_face_size", 64)
    monkeypatch.setattr(modules.globals, "color_match_strength", 0.35)
    monkeypatch.setattr(modules.globals, "repair_boundary_strength", 0.35)
    state = ControlState(
        str(tmp_path),
        health=lambda: {},
        devices=lambda: {},
        select_device=lambda _device: {},
    )

    state.apply(
        {
            "tracking_enabled": False,
            "detection_interval": 99,
            "tracking_smoothing": 2.0,
            "tracking_grace_frames": -5,
            "minimum_detection_score": 0.01,
            "minimum_face_size": 2,
            "color_match_strength": 0.5,
            "repair_boundary_strength": 2.0,
        }
    )

    config = state.config()
    assert config["tracking_enabled"] is False
    assert config["detection_interval"] == 5
    assert config["tracking_smoothing"] == 0.95
    assert config["tracking_grace_frames"] == 0
    assert config["minimum_detection_score"] == 0.1
    assert config["minimum_face_size"] == 32
    assert config["color_match_strength"] == 0.5
    assert config["repair_boundary_strength"] == 1.0
    assert (tmp_path / "settings.json").exists()


def test_swapper_model_selection_is_explicit_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(modules.globals, "swapper_model", "auto")
    state = ControlState(
        str(tmp_path),
        health=lambda: {},
        devices=lambda: {},
        select_device=lambda _device: {},
    )

    state.apply({"swapper_model": "native-256"})

    assert modules.globals.swapper_model == "native-256"
    assert state.config()["swapper_model"] == "native-256"

    try:
        state.apply({"swapper_model": "unknown"})
    except ValueError as error:
        assert "swapper_model must be" in str(error)
    else:
        raise AssertionError("invalid swapper model was accepted")

    state.apply({"swapper_model": "simswap-512"})
    assert modules.globals.swapper_model == "simswap-512"
    assert state.config()["swapper_model"] == "simswap-512"

    state.apply({"swapper_model": "instyle-256"})
    assert modules.globals.swapper_model == "instyle-256"
    assert state.config()["swapper_model"] == "instyle-256"


def test_shared_processing_mode_maps_to_windows_inference_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(modules.globals, "processing_enabled", True)
    monkeypatch.setattr(modules.globals, "processing_off_output", "passthrough")
    state = _state(tmp_path)

    state.apply({"processing_mode": "passthrough"})
    assert modules.globals.processing_enabled is False
    assert modules.globals.processing_off_output == "passthrough"
    assert state.config()["processing_mode"] == "passthrough"
    assert state.config()["processing_enabled"] is False

    state.apply({"processing_mode": "face_swap"})
    assert modules.globals.processing_enabled is True
    assert state.config()["processing_mode"] == "face_swap"
    assert state.config()["processing_enabled"] is True


def test_valid_source_upload_persists_exact_upload_identity(tmp_path):
    state = _state(tmp_path)
    body = _jpeg_bytes()
    identifier = hashlib.sha256(body).hexdigest()

    assert state.upload(body, identifier) == identifier

    metadata = json.loads((tmp_path / "source.metadata.json").read_text("utf-8"))
    stored = (tmp_path / "source.jpg").read_bytes()
    assert metadata == {
        "source_identifier": identifier,
        "stored_jpeg_sha256": hashlib.sha256(stored).hexdigest(),
    }
    assert state.config()["source_identifier"] == identifier


def test_forged_source_identifier_is_rejected_without_changing_source(tmp_path):
    state = _state(tmp_path)
    original = _jpeg_bytes(50)
    original_identifier = hashlib.sha256(original).hexdigest()
    state.upload(original, original_identifier)
    stored_before = (tmp_path / "source.jpg").read_bytes()
    metadata_before = (tmp_path / "source.metadata.json").read_bytes()

    with pytest.raises(ValueError, match="does not match"):
        state.upload(_jpeg_bytes(170), "0" * 64)

    assert (tmp_path / "source.jpg").read_bytes() == stored_before
    assert (tmp_path / "source.metadata.json").read_bytes() == metadata_before
    assert state.config()["source_identifier"] == original_identifier


def test_source_identifier_is_restored_after_restart(tmp_path):
    body = _jpeg_bytes()
    identifier = hashlib.sha256(body).hexdigest()
    _state(tmp_path).upload(body, identifier)

    restarted = _state(tmp_path)

    assert restarted.source_identifier == identifier
    assert restarted.config()["source_identifier"] == identifier


@pytest.mark.parametrize("tamper_target", ["source", "metadata"])
def test_source_or_metadata_mismatch_becomes_unverified(tmp_path, tamper_target):
    body = _jpeg_bytes()
    identifier = hashlib.sha256(body).hexdigest()
    state = _state(tmp_path)
    state.upload(body, identifier)

    if tamper_target == "source":
        (tmp_path / "source.jpg").write_bytes(_jpeg_bytes(190))
    else:
        metadata_path = tmp_path / "source.metadata.json"
        metadata = json.loads(metadata_path.read_text("utf-8"))
        metadata["stored_jpeg_sha256"] = "f" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert state.config()["source_identifier"] is None
    assert _state(tmp_path).config()["source_identifier"] is None


def test_legacy_source_without_metadata_is_unverified(tmp_path):
    (tmp_path / "source.jpg").write_bytes(_jpeg_bytes())

    state = _state(tmp_path)

    assert state.config()["source_identifier"] is None
