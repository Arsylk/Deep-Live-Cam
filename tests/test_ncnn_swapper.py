from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import modules.ncnn_swapper as ncnn_swapper


def test_availability_requires_every_offline_asset(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    library = tmp_path / "bridge.so"
    monkeypatch.setenv("DLC_NCNN_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DLC_NCNN_LIBRARY", str(library))

    assert not ncnn_swapper.ncnn_swapper_available()
    library.touch()
    (model_dir / "inswapper_128.ncnn.param").touch()
    (model_dir / "inswapper_128.ncnn.bin").touch()
    assert not ncnn_swapper.ncnn_swapper_available()
    np.save(
        model_dir / "inswapper_128_emap.npy",
        np.eye(512, dtype=np.float32),
        allow_pickle=False,
    )
    assert ncnn_swapper.ncnn_swapper_available()


def test_get_preserves_insightface_tensor_contract(monkeypatch):
    swapper = object.__new__(ncnn_swapper.NcnnSwapper)
    swapper.emap = np.eye(512, dtype=np.float32)
    aligned = np.full((128, 128, 3), (10, 20, 30), dtype=np.uint8)
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    monkeypatch.setattr(
        ncnn_swapper.face_align,
        "norm_crop2",
        lambda image, keypoints, size: (aligned, matrix),
    )

    captured = {}

    def fake_forward(target, source):
        captured["target"] = target
        captured["source"] = source
        output = np.empty((1, 3, 128, 128), dtype=np.float32)
        output[:, 0] = 1.0
        output[:, 1] = 0.5
        output[:, 2] = 0.0
        return output

    swapper.forward = fake_forward
    source_embedding = np.arange(1, 513, dtype=np.float32)
    source = SimpleNamespace(normed_embedding=source_embedding)
    target = SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32))

    result, returned_matrix = swapper.get(
        np.zeros((256, 256, 3), dtype=np.uint8), target, source, paste_back=False
    )

    assert captured["target"].shape == (1, 3, 128, 128)
    assert captured["target"].dtype == np.float32
    assert np.isclose(np.linalg.norm(captured["source"]), 1.0)
    assert captured["source"].dtype == np.float32
    assert np.array_equal(returned_matrix, matrix)
    # Network output is RGB; the app contract is BGR uint8.
    assert result.dtype == np.uint8
    assert np.array_equal(result[0, 0], np.array([0, 127, 255], dtype=np.uint8))


def test_forward_rejects_wrong_tensor_shapes():
    swapper = object.__new__(ncnn_swapper.NcnnSwapper)
    with pytest.raises(ValueError, match="target must have shape"):
        swapper.forward(
            np.zeros((3, 128, 128), dtype=np.float32),
            np.zeros((1, 512), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="source must have shape"):
        swapper.forward(
            np.zeros((1, 3, 128, 128), dtype=np.float32),
            np.zeros(512, dtype=np.float32),
        )
