from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.prepare_native256_teacher import (
    TEACHER_TYPE,
    TeacherPreparationError,
    prepare_teacher,
)


class _TeacherSession:
    def __init__(self):
        self.feeds = None

    def get_inputs(self):
        return [
            SimpleNamespace(
                name="target", shape=[1, 3, 128, 128], type="tensor(float)"
            ),
            SimpleNamespace(name="source", shape=[1, 512], type="tensor(float)"),
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="output", shape=[1, 3, 128, 128], type="tensor(float)"
            )
        ]

    def run(self, outputs, feeds):
        assert outputs == ["output"]
        self.feeds = feeds
        return [np.ascontiguousarray(feeds["target"] * np.float32(0.5))]


def _inputs(tmp_path):
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"local teacher fixture")
    target = tmp_path / "target.png"
    mask = tmp_path / "mask.png"
    latent = tmp_path / "latent.npy"
    pixels = np.arange(256, dtype=np.uint8)[None, :, None]
    rgb = np.broadcast_to(pixels, (256, 256, 3)).copy()
    Image.fromarray(rgb).save(target)
    Image.fromarray(np.full((256, 256), 192, dtype=np.uint8)).save(mask)
    value = np.arange(1, 513, dtype=np.float32)
    value /= np.linalg.norm(value)
    np.save(latent, value, allow_pickle=False)
    return model, target, latent, mask


def test_prepares_float_teacher_pair_with_pinned_local_model(tmp_path):
    model, target, latent, mask = _inputs(tmp_path)
    session = _TeacherSession()
    observed = {}

    def factory(path, providers):
        observed.update(path=path, providers=providers)
        return session

    metadata = prepare_teacher(
        model_path=model,
        expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        target_path=target,
        source_latent_path=latent,
        face_mask_path=mask,
        output_dir=tmp_path / "teacher",
        sample_id="sample-1",
        session_factory=factory,
    )

    candidate = np.load(tmp_path / "teacher/sample-1-candidate.npy")
    alpha = np.load(tmp_path / "teacher/sample-1-alpha.npy")
    assert candidate.dtype == np.float32
    assert candidate.shape == (3, 128, 128)
    assert alpha.dtype == np.float32
    assert alpha.shape == (1, 128, 128)
    assert float(alpha.mean()) == pytest.approx(192 / 255)
    assert session.feeds["target"].shape == (1, 3, 128, 128)
    assert session.feeds["source"].shape == (1, 512)
    assert observed["providers"] == ["CPUExecutionProvider"]
    assert metadata["teacher_type"] == TEACHER_TYPE
    assert len(metadata["outputs"]["teacher_candidate_sha256"]) == 64


def test_rejects_wrong_model_hash_and_existing_outputs(tmp_path):
    model, target, latent, mask = _inputs(tmp_path)
    arguments = dict(
        model_path=model,
        expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        target_path=target,
        source_latent_path=latent,
        face_mask_path=mask,
        output_dir=tmp_path / "teacher",
        sample_id="sample-1",
        session_factory=lambda path, providers: _TeacherSession(),
    )
    with pytest.raises(TeacherPreparationError, match="SHA-256 mismatch"):
        prepare_teacher(**{**arguments, "expected_model_sha256": "0" * 64})

    prepare_teacher(**arguments)
    with pytest.raises(TeacherPreparationError, match="refusing to overwrite"):
        prepare_teacher(**arguments)
