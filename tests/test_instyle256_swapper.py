from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import modules.instyle256_swapper as instyle


@dataclass
class _Node:
    name: str
    shape: list[int]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(self, output_size: int = 256) -> None:
        self.output_size = output_size
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self):
        return [
            _Node("target", [1, 3, 256, 256]),
            _Node("source", [1, 512]),
        ]

    def get_outputs(self):
        return [_Node("output", [1, 3, self.output_size, self.output_size])]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, output_names, feeds):
        self.calls.append((list(output_names), dict(feeds)))
        output = np.empty((1, 3, 256, 256), dtype=np.float32)
        output[:, 0] = 1.0
        output[:, 1] = 0.5
        output[:, 2] = 0.0
        return [output]


def _factory(session):
    return lambda _path, *, sess_options=None, providers=None: session


def _faces():
    keypoints = instyle._TARGET_TEMPLATE * np.array([256, 256], dtype=np.float32)
    source = SimpleNamespace(
        normed_embedding=np.arange(1, 513, dtype=np.float32)
    )
    target = SimpleNamespace(kps=keypoints)
    return target, source


def test_native_256_candidate_and_identity_mapping_are_cached():
    session = _FakeSession()
    swapper = instyle.InStyle256Swapper(
        providers=["CUDAExecutionProvider"],
        session_factory=_factory(session),
        verify_assets=False,
        identity_map=np.eye(512, dtype=np.float32),
    )
    target, source = _faces()
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    first = swapper.get(frame, target, source, paste_back=False)
    second = swapper.get(frame, target, source, paste_back=False)

    assert first.face_bgr.shape == (256, 256, 3)
    assert first.face_bgr.dtype == np.uint8
    assert tuple(first.face_bgr[128, 128]) == (0, 127, 255)
    assert first.alpha is None
    assert first.model_id == "instyle-swapper-256-b"
    assert first.backend == "ort"
    assert np.allclose(first.affine, np.eye(2, 3), atol=1e-4)
    assert np.array_equal(first.face_bgr, second.face_bgr)
    assert len(session.calls) == 2
    assert session.calls[0][1]["target"].shape == (1, 3, 256, 256)
    latent = session.calls[0][1]["source"]
    assert latent.shape == (1, 512)
    assert np.isclose(np.linalg.norm(latent), 1.0)
    assert session.calls[0][1]["source"] is session.calls[1][1]["source"]


def test_session_contract_rejects_a_generator_that_is_not_native_256():
    with pytest.raises(instyle.InStyle256LoadError, match="output must be"):
        session = _FakeSession(output_size=128)
        instyle.InStyle256Swapper(
            session_factory=_factory(session),
            verify_assets=False,
            identity_map=np.eye(512, dtype=np.float32),
        )


def test_paste_back_is_owned_by_the_common_application_path():
    swapper = instyle.InStyle256Swapper(
        session_factory=_factory(_FakeSession()),
        verify_assets=False,
        identity_map=np.eye(512, dtype=np.float32),
    )
    target, source = _faces()
    with pytest.raises(ValueError, match="common paste path"):
        swapper.get(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            target,
            source,
            paste_back=True,
        )
