from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import modules.globals
import modules.native256_swapper as native256
from modules.processors.frame import face_swapper
from modules.swapper_contract import SwapResult


def _reset_model_state() -> None:
    current = face_swapper.FACE_SWAPPER
    close = getattr(current, "close", None)
    if callable(close):
        close()
    face_swapper.FACE_SWAPPER = None
    face_swapper.FACE_SWAPPER_KEY = None
    modules.globals.active_swapper_model = "not-loaded"
    modules.globals.active_swapper_backend = "not-loaded"
    modules.globals.active_swapper_resolution = 0


def test_explicit_simswap_uses_the_native_512_ort_backend(monkeypatch):
    _reset_model_state()
    import modules.simswap512_swapper as simswap512

    observed = {}

    class FakeSimSwap:
        input_size = (512, 512)
        native_resolution = 512
        backend = "ort"
        model_id = "simswap-512-visomaster"
        device_name = "CUDAExecutionProvider"

        def __init__(self, *, providers):
            observed["providers"] = providers

    monkeypatch.setattr(simswap512, "simswap512_available", lambda: True)
    monkeypatch.setattr(simswap512, "SimSwap512Swapper", FakeSimSwap)
    monkeypatch.setattr(modules.globals, "swapper_model", "simswap-512")
    monkeypatch.setattr(modules.globals, "swapper_backend", "ort")
    monkeypatch.setattr(
        modules.globals, "execution_providers", ["CUDAExecutionProvider"]
    )

    result = face_swapper.get_face_swapper()

    assert isinstance(result, FakeSimSwap)
    assert observed["providers"] == ["CUDAExecutionProvider"]
    assert modules.globals.active_swapper_model == "simswap-512-visomaster"
    assert modules.globals.active_swapper_backend == "ort"
    assert modules.globals.active_swapper_resolution == 512
    _reset_model_state()


def test_explicit_instyle_uses_the_native_256_ort_backend(monkeypatch):
    _reset_model_state()
    import modules.instyle256_swapper as instyle256

    observed = {}

    class FakeInStyle:
        input_size = (256, 256)
        native_resolution = 256
        backend = "ort"
        model_id = "instyle-swapper-256-b"
        device_name = "CUDAExecutionProvider"

        def __init__(self, *, providers):
            observed["providers"] = providers

    monkeypatch.setattr(instyle256, "instyle256_available", lambda: True)
    monkeypatch.setattr(instyle256, "InStyle256Swapper", FakeInStyle)
    monkeypatch.setattr(modules.globals, "swapper_model", "instyle-256")
    monkeypatch.setattr(modules.globals, "swapper_backend", "ort")
    monkeypatch.setattr(
        modules.globals, "execution_providers", ["CUDAExecutionProvider"]
    )

    result = face_swapper.get_face_swapper()

    assert isinstance(result, FakeInStyle)
    assert observed["providers"] == ["CUDAExecutionProvider"]
    assert modules.globals.active_swapper_model == "instyle-swapper-256-b"
    assert modules.globals.active_swapper_backend == "ort"
    assert modules.globals.active_swapper_resolution == 256
    _reset_model_state()


def test_auto_only_admits_a_qualified_native_bundle(monkeypatch):
    _reset_model_state()
    observed: dict[str, object] = {}

    class FakeNative:
        input_size = (256, 256)
        backend = "ort"
        device_name = "test-provider"
        spec = SimpleNamespace(model_id="dlc-native256-test")
        manifest = SimpleNamespace(quality_status="qualified")

        def __init__(self, *, providers, require_qualified=False):
            observed["providers"] = providers
            observed["constructor_require_qualified"] = require_qualified

    def available(*, require_qualified=False):
        observed["require_qualified"] = require_qualified
        return True

    monkeypatch.setattr(native256, "native256_swapper_available", available)
    monkeypatch.setattr(native256, "Native256Swapper", FakeNative)
    monkeypatch.setattr(modules.globals, "swapper_model", "auto")
    monkeypatch.setattr(modules.globals, "swapper_backend", "ort")
    monkeypatch.setattr(
        modules.globals, "execution_providers", ["CPUExecutionProvider"]
    )

    model = face_swapper.get_face_swapper()

    assert isinstance(model, FakeNative)
    assert observed["require_qualified"] is True
    assert observed["providers"] == ["CPUExecutionProvider"]
    assert observed["constructor_require_qualified"] is True
    assert modules.globals.active_swapper_model == "dlc-native256-test"
    assert modules.globals.active_swapper_backend == "ort"
    _reset_model_state()


def test_explicit_native_ncnn_does_not_fall_back_to_ort(monkeypatch):
    _reset_model_state()
    import modules.native256_ncnn_swapper as native_ncnn

    ort_constructed = False

    class UnexpectedOrt:
        def __init__(self, **_kwargs):
            nonlocal ort_constructed
            ort_constructed = True

    monkeypatch.setattr(native_ncnn, "native256_ncnn_available", lambda **_k: False)
    monkeypatch.setattr(native256, "Native256Swapper", UnexpectedOrt)
    monkeypatch.setattr(modules.globals, "swapper_model", "native-256")
    monkeypatch.setattr(modules.globals, "swapper_backend", "ncnn")
    monkeypatch.setattr(modules.globals, "execution_providers", [])
    monkeypatch.setattr(face_swapper.platform, "system", lambda: "Linux")

    assert face_swapper.get_face_swapper() is None
    assert not ort_constructed
    assert modules.globals.active_swapper_model == "not-loaded"
    _reset_model_state()


def test_explicit_native_ncnn_uses_native_bundle(monkeypatch):
    _reset_model_state()


def test_auto_ncnn_admits_only_qualified_native_bundle(monkeypatch):
    _reset_model_state()
    import modules.native256_ncnn_swapper as native_ncnn

    observed = {}

    class FakeNcnn:
        backend = "ncnn"
        device_name = "RX 570"
        spec = SimpleNamespace(model_id="dlc-native256-qualified")
        manifest = SimpleNamespace(quality_status="qualified")

        def __init__(self, *, require_qualified=False):
            observed["constructor_require_qualified"] = require_qualified

    def available(*, require_qualified=False):
        observed.setdefault("availability", []).append(require_qualified)
        return require_qualified

    monkeypatch.setattr(native_ncnn, "native256_ncnn_available", available)
    monkeypatch.setattr(native_ncnn, "Native256NcnnSwapper", FakeNcnn)
    monkeypatch.setattr(modules.globals, "swapper_model", "auto")
    monkeypatch.setattr(modules.globals, "swapper_backend", "ncnn")
    monkeypatch.setattr(modules.globals, "execution_providers", [])
    monkeypatch.setattr(face_swapper.platform, "system", lambda: "Linux")

    result = face_swapper.get_face_swapper()

    assert isinstance(result, FakeNcnn)
    assert observed["availability"] == [True, True]
    assert observed["constructor_require_qualified"] is True
    assert modules.globals.active_swapper_backend == "ncnn"
    _reset_model_state()
    import modules.native256_ncnn_swapper as native_ncnn

    observed = {}

    class FakeNcnn:
        backend = "ncnn"
        device_name = "RX 570"
        spec = SimpleNamespace(model_id="dlc-native256-test")
        manifest = SimpleNamespace(quality_status="development")

        def __init__(self, *, require_qualified=False):
            observed["require_qualified"] = require_qualified

    monkeypatch.setattr(native_ncnn, "native256_ncnn_available", lambda **_k: True)
    monkeypatch.setattr(native_ncnn, "Native256NcnnSwapper", FakeNcnn)
    monkeypatch.setattr(modules.globals, "swapper_model", "native-256")
    monkeypatch.setattr(modules.globals, "swapper_backend", "ncnn")
    monkeypatch.setattr(modules.globals, "execution_providers", [])
    monkeypatch.setattr(face_swapper.platform, "system", lambda: "Linux")

    result = face_swapper.get_face_swapper()

    assert isinstance(result, FakeNcnn)
    assert observed["require_qualified"] is False
    assert modules.globals.active_swapper_backend == "ncnn"
    _reset_model_state()


def test_auto_backend_falls_back_from_native_ncnn_to_native_ort(monkeypatch):
    _reset_model_state()
    import modules.native256_ncnn_swapper as native_ncnn

    class BrokenNcnn:
        def __init__(self, **_kwargs):
            raise RuntimeError("test Vulkan failure")

    class FakeOrt:
        backend = "ort"
        device_name = "CPUExecutionProvider"
        spec = SimpleNamespace(model_id="dlc-native256-test")
        manifest = SimpleNamespace(quality_status="qualified")

        def __init__(self, *, providers, require_qualified=False):
            assert require_qualified

    monkeypatch.setattr(native256, "native256_swapper_available", lambda **_k: True)
    monkeypatch.setattr(native256, "Native256Swapper", FakeOrt)
    monkeypatch.setattr(native_ncnn, "native256_ncnn_available", lambda **_k: True)
    monkeypatch.setattr(native_ncnn, "Native256NcnnSwapper", BrokenNcnn)
    monkeypatch.setattr(modules.globals, "swapper_model", "auto")
    monkeypatch.setattr(modules.globals, "swapper_backend", "auto")
    monkeypatch.setattr(modules.globals, "execution_providers", ["CPUExecutionProvider"])
    monkeypatch.setattr(face_swapper.platform, "system", lambda: "Linux")

    result = face_swapper.get_face_swapper()

    assert isinstance(result, FakeOrt)
    assert modules.globals.active_swapper_backend == "ort"
    _reset_model_state()


def test_native_semantic_alpha_is_pasted_once_and_legacy_repairs_are_skipped(
    monkeypatch,
):
    alpha = np.full((256, 256), 137, dtype=np.uint8)
    result = SwapResult(
        face_bgr=np.full((256, 256, 3), 180, dtype=np.uint8),
        affine=np.eye(2, 3, dtype=np.float32),
        alpha=alpha,
        model_id="dlc-native256-test",
        backend="ort",
    )

    class FakeSwapper:
        input_size = (256, 256)

        def get(self, *_args, **_kwargs):
            return result

    paste_calls: list[np.ndarray | None] = []

    def fake_paste(target, _candidate, _aligned, _affine, override_alpha=None):
        paste_calls.append(override_alpha)
        return target

    monkeypatch.setattr(face_swapper, "get_face_swapper", lambda: FakeSwapper())
    monkeypatch.setattr(face_swapper, "_fast_paste_back", fake_paste)
    monkeypatch.setattr(
        face_swapper,
        "apply_frequency_repair",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("legacy repair ran")),
    )
    monkeypatch.setattr(
        face_swapper,
        "content_aware_boundary_mask",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("legacy mask ran")),
    )
    monkeypatch.setattr(modules.globals, "opacity", 1.0)
    monkeypatch.setattr(modules.globals, "mouth_mask", False)
    monkeypatch.setattr(modules.globals, "poisson_blend", False)
    monkeypatch.setattr(modules.globals, "color_match_strength", 0.0)
    monkeypatch.setattr(modules.globals, "repair_hf_strength", 0.5)
    monkeypatch.setattr(modules.globals, "repair_checkerboard", 1.0)
    monkeypatch.setattr(modules.globals, "repair_wavelet", 1.0)
    monkeypatch.setattr(modules.globals, "repair_boundary_mask", True)
    monkeypatch.setattr(modules.globals, "execution_providers", [])
    source = SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))
    target = SimpleNamespace(track_id=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    returned = face_swapper.swap_face(source, target, frame)

    assert np.array_equal(returned, frame)
    assert len(paste_calls) == 1
    assert paste_calls[0] is alpha


def test_legacy_boundary_mask_replaces_default_mask_before_single_paste(monkeypatch):
    generated = np.full((128, 128, 3), 180, dtype=np.uint8)
    affine = np.eye(2, 3, dtype=np.float32)
    learned_boundary = np.full((128, 128), 99, dtype=np.uint8)

    class FakeLegacy:
        input_size = (128, 128)

        def get(self, *_args, **_kwargs):
            return generated, affine

    paste_calls: list[np.ndarray | None] = []
    monkeypatch.setattr(face_swapper, "get_face_swapper", lambda: FakeLegacy())
    monkeypatch.setattr(
        face_swapper,
        "content_aware_boundary_mask",
        lambda *_args, **_kwargs: learned_boundary,
    )
    monkeypatch.setattr(
        face_swapper,
        "_fast_paste_back",
        lambda target, _candidate, _aligned, _affine, override_alpha=None: (
            paste_calls.append(override_alpha) or target
        ),
    )
    monkeypatch.setattr(modules.globals, "opacity", 1.0)
    monkeypatch.setattr(modules.globals, "mouth_mask", False)
    monkeypatch.setattr(modules.globals, "poisson_blend", False)
    monkeypatch.setattr(modules.globals, "color_match_strength", 0.0)
    monkeypatch.setattr(modules.globals, "repair_hf_strength", 0.0)
    monkeypatch.setattr(modules.globals, "repair_checkerboard", 0.0)
    monkeypatch.setattr(modules.globals, "repair_wavelet", 0.0)
    monkeypatch.setattr(modules.globals, "repair_boundary_mask", True)
    monkeypatch.setattr(modules.globals, "execution_providers", [])
    source = SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))
    target = SimpleNamespace(track_id=1)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    face_swapper.swap_face(source, target, frame)

    assert paste_calls == [learned_boundary]
