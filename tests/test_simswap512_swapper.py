from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import modules.simswap512_swapper as simswap


@dataclass
class _Node:
    name: str
    shape: list[int]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(self, kind: str, *, output_size: int = 512) -> None:
        self.kind = kind
        self.output_size = output_size
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self):
        if self.kind == "arcface":
            return [_Node("input", [1, 3, 112, 112])]
        return [
            _Node("input", [1, 3, 512, 512]),
            _Node("onnx::Gemm_1", [1, 512]),
        ]

    def get_outputs(self):
        if self.kind == "arcface":
            return [_Node("output", [1, 512])]
        return [_Node("output", [1, 3, self.output_size, self.output_size])]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, output_names, feeds):
        self.calls.append((list(output_names), dict(feeds)))
        if self.kind == "arcface":
            return [np.arange(1, 513, dtype=np.float32).reshape(1, 512)]
        candidate = np.empty((1, 3, 512, 512), dtype=np.float32)
        candidate[:, 0] = 1.0
        candidate[:, 1] = 0.5
        candidate[:, 2] = 0.0
        return [candidate]


class _Factory:
    def __init__(self, *, output_size: int = 512) -> None:
        self.arcface = _FakeSession("arcface")
        self.swapper = _FakeSession("swapper", output_size=output_size)

    def __call__(self, path, *, sess_options=None, providers=None):
        if "arcface" in str(path):
            return self.arcface
        return self.swapper


def _faces():
    target_keypoints = simswap._TARGET_TEMPLATE * np.array([512, 512], dtype=np.float32)
    source = SimpleNamespace(
        embedding=np.arange(1, 513, dtype=np.float32),
        kps=simswap._ARCFACE_TEMPLATE.copy(),
        _dlc_source_frame=np.full((112, 112, 3), 100, dtype=np.uint8),
    )
    target = SimpleNamespace(kps=target_keypoints)
    return target, source


def test_native_512_candidate_and_paired_recognition_are_real_model_sized():
    factory = _Factory()
    swapper = simswap.SimSwap512Swapper(
        providers=["CUDAExecutionProvider"],
        session_factory=factory,
        verify_assets=False,
    )
    target, source = _faces()
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    first = swapper.get(frame, target, source, paste_back=False)
    second = swapper.get(frame, target, source, paste_back=False)

    assert first.face_bgr.shape == (512, 512, 3)
    assert first.face_bgr.dtype == np.uint8
    assert tuple(first.face_bgr[256, 256]) == (0, 127, 255)
    assert first.alpha is None
    assert first.model_id == "simswap-512-visomaster"
    assert first.backend == "ort"
    assert np.allclose(first.affine, np.eye(2, 3), atol=1e-4)
    assert np.array_equal(first.face_bgr, second.face_bgr)
    assert len(factory.arcface.calls) == 1
    assert len(factory.swapper.calls) == 2
    recognizer_input = factory.arcface.calls[0][1]["input"]
    assert recognizer_input.shape == (1, 3, 112, 112)
    source_input = factory.swapper.calls[0][1]["onnx::Gemm_1"]
    assert source_input.shape == (1, 512)
    assert np.isclose(np.linalg.norm(source_input), 1.0)
    assert factory.swapper.calls[0][1]["input"].shape == (1, 3, 512, 512)


def test_session_contract_rejects_a_generator_that_is_not_native_512():
    with pytest.raises(simswap.SimSwap512LoadError, match="output must be"):
        simswap.SimSwap512Swapper(
            session_factory=_Factory(output_size=256),
            verify_assets=False,
        )


def test_paste_back_is_owned_by_the_common_application_path():
    swapper = simswap.SimSwap512Swapper(
        session_factory=_Factory(), verify_assets=False
    )
    target, source = _faces()
    with pytest.raises(ValueError, match="common paste path"):
        swapper.get(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            target,
            source,
            paste_back=True,
        )
