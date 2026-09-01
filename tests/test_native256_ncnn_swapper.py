from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import onnx
import pytest

import modules.native256_ncnn_swapper as native_ncnn


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _embedded_onnx() -> bytes:
    input_value = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1]
    )
    output_value = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "fixture",
        [input_value],
        [output_value],
    )
    return onnx.helper.make_model(graph).SerializeToString()


def _write_bundle(tmp_path, *, quality="development", include_ncnn=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    embedded_onnx = _embedded_onnx()
    files = {
        "identity.onnx": embedded_onnx,
        "generator.onnx": embedded_onnx,
        "conditioner.ncnn.param": b"7767517\n3 3\nInput in 0 1 in0\n",
        "conditioner.ncnn.bin": b"conditioner weights",
        "generator.ncnn.param": b"7767517\n4 4\nInput in 0 1 in0\n",
        "generator.ncnn.bin": b"generator weights",
    }
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    identity_map = np.eye(512, dtype=np.float32)
    np.save(tmp_path / "identity_map.npy", identity_map, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "format": "onnx",
        "model_id": "test_native_256",
        "quality_status": quality,
        "auto_select_eligible": quality == "qualified",
        "input_size": [256, 256],
        "embedding_size": 512,
        "layout": "NCHW",
        "color_order": "RGB",
        "identity_type": "inswapper_mapped_512",
        "preprocess": {
            "scale": 1.0 / 255.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
        "candidate_range": [0.0, 1.0],
        "alpha_range": [0.0, 1.0],
        "candidate_precomposited": False,
        "identity_map": {
            "file": "identity_map.npy",
            "sha256": _hash((tmp_path / "identity_map.npy").read_bytes()),
        },
        "identity_conditioner": {
            "file": "identity.onnx",
            "sha256": _hash(files["identity.onnx"]),
            "input": "mapped_identity",
            "output": "style",
        },
        "swapper": {
            "file": "generator.onnx",
            "sha256": _hash(files["generator.onnx"]),
            "target_input": "target",
            "style_input": "style",
            "candidate_output": "candidate",
            "alpha_output": "alpha",
        },
    }
    if include_ncnn:
        manifest["ncnn"] = {
            "backend": "ncnn-vulkan",
            "fp16_storage": True,
            "identity_conditioner": {
                "param_file": "conditioner.ncnn.param",
                "param_sha256": _hash(files["conditioner.ncnn.param"]),
                "bin_file": "conditioner.ncnn.bin",
                "bin_sha256": _hash(files["conditioner.ncnn.bin"]),
                "input": "in0",
                "output": "out0",
            },
            "swapper": {
                "param_file": "generator.ncnn.param",
                "param_sha256": _hash(files["generator.ncnn.param"]),
                "bin_file": "generator.ncnn.bin",
                "bin_sha256": _hash(files["generator.ncnn.bin"]),
                "target_input": "in0",
                "style_input": "in1",
                "candidate_output": "out0",
                "alpha_output": "out1",
            },
        }
    if manifest["auto_select_eligible"]:
        artifacts = {
            "identity_map_sha256": manifest["identity_map"]["sha256"],
            "identity_conditioner_sha256": manifest[
                "identity_conditioner"
            ]["sha256"],
            "swapper_sha256": manifest["swapper"]["sha256"],
        }
        if include_ncnn:
            artifacts.update(
                {
                    "ncnn_identity_conditioner_param_sha256": manifest["ncnn"][
                        "identity_conditioner"
                    ]["param_sha256"],
                    "ncnn_identity_conditioner_bin_sha256": manifest["ncnn"][
                        "identity_conditioner"
                    ]["bin_sha256"],
                    "ncnn_swapper_param_sha256": manifest["ncnn"]["swapper"][
                        "param_sha256"
                    ],
                    "ncnn_swapper_bin_sha256": manifest["ncnn"]["swapper"][
                        "bin_sha256"
                    ],
                }
            )
        report = {
            "schema_version": 1,
            "model_id": manifest["model_id"],
            "verdict": "qualified",
            "artifacts": artifacts,
            "gates": {
                "identity_similarity": True,
                "pose_landmarks": True,
                "background_temporal": True,
                "ncnn_parity": True,
                "android_qnn_admission": True,
                "rx570_latency": True,
                "target_phone_latency": True,
                "blind_review": True,
            },
        }
        report_data = json.dumps(report).encode("utf-8")
        (tmp_path / "qualification.json").write_bytes(report_data)
        manifest["qualification_report"] = {
            "file": "qualification.json",
            "sha256": _hash(report_data),
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    library = tmp_path / "libnative256.so"
    library.write_bytes(b"test library")
    return path, library


class _Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeLibrary:
    def __init__(self):
        self.created = None
        self.destroyed = 0
        self.cleared = 0
        self.run_calls = 0
        self.condition_count = 0
        self.last_identity = None
        self.last_target = None
        self.dlc_ncnn_abi_version = _Function(lambda: 2)
        self.dlc_ncnn_last_error = _Function(lambda: b"test native error")
        self.dlc_ncnn_native256_create = _Function(self._create)
        self.dlc_ncnn_native256_destroy = _Function(self._destroy)
        self.dlc_ncnn_native256_device_name = _Function(lambda _handle: b"RX 570")
        self.dlc_ncnn_native256_clear_style = _Function(self._clear)
        self.dlc_ncnn_native256_run = _Function(self._run)

    def _create(self, *args):
        self.created = args
        return 123

    def _destroy(self, _handle):
        self.destroyed += 1

    def _clear(self, _handle):
        self.cleared += 1
        self.last_identity = None

    def _run(self, _handle, target, identity, candidate, alpha):
        self.run_calls += 1
        current_identity = np.ctypeslib.as_array(identity, shape=(512,)).copy()
        if self.last_identity is None or not np.array_equal(
            current_identity, self.last_identity
        ):
            self.condition_count += 1
            self.last_identity = current_identity
        self.last_target = np.ctypeslib.as_array(
            target, shape=(3 * 256 * 256,)
        ).copy()
        output = np.ctypeslib.as_array(candidate, shape=(3 * 256 * 256,))
        output[: 256 * 256] = 1.0
        output[256 * 256 : 2 * 256 * 256] = 0.5
        output[2 * 256 * 256 :] = 0.0
        np.ctypeslib.as_array(alpha, shape=(256 * 256,)).fill(0.25)
        return 0


def _faces():
    target = SimpleNamespace(kps=np.arange(10, dtype=np.float32).reshape(5, 2))
    source = SimpleNamespace(normed_embedding=np.arange(1, 513, dtype=np.float32))
    return target, source


def test_adapter_runs_split_contract_and_returns_semantic_alpha(tmp_path, monkeypatch):
    manifest, library_path = _write_bundle(tmp_path)
    library = _FakeLibrary()
    crop = np.full((256, 256, 3), (10, 20, 30), dtype=np.uint8)
    affine = np.array([[1, 0, 2], [0, 1, 3]], dtype=np.float32)
    monkeypatch.setattr(
        native_ncnn.face_align, "norm_crop2", lambda *_args: (crop, affine)
    )
    swapper = native_ncnn.Native256NcnnSwapper(
        manifest,
        library_path=library_path,
        library_loader=lambda _path: library,
    )
    target, source = _faces()

    result = swapper.get(
        np.zeros((480, 640, 3), dtype=np.uint8), target, source
    )
    equivalent = SimpleNamespace(normed_embedding=source.normed_embedding * 2.0)
    swapper.get(np.zeros((480, 640, 3), dtype=np.uint8), target, equivalent)

    assert result.backend == "ncnn"
    assert result.model_id == "test_native_256"
    assert result.face_bgr.dtype == np.uint8
    assert np.array_equal(result.face_bgr[0, 0], [0, 127, 255])
    assert result.alpha[0, 0] == 64
    assert np.array_equal(result.affine, affine)
    assert swapper.device_name == "RX 570"
    assert library.created[-1] == 1
    assert library.run_calls == 2
    assert library.condition_count == 1
    assert swapper.style_cache_size == 1
    assert np.isclose(np.linalg.norm(library.last_identity), 1.0)
    # BGR aligned crop became RGB NCHW with declared 1/255 scale.
    assert library.last_target[0] == pytest.approx(30 / 255)
    assert library.last_target[256 * 256] == pytest.approx(20 / 255)
    assert library.last_target[2 * 256 * 256] == pytest.approx(10 / 255)

    swapper.clear_style_cache()
    assert swapper.style_cache_size == 0
    assert library.cleared == 1
    swapper.close()
    assert library.destroyed == 1
    with pytest.raises(native_ncnn.Native256NcnnInferenceError, match="closed"):
        swapper.clear_style_cache()


def test_availability_requires_ncnn_assets_library_and_qualification(tmp_path):
    development, library = _write_bundle(tmp_path / "development")
    assert native_ncnn.native256_ncnn_available(
        development, library_path=library
    )
    assert not native_ncnn.native256_ncnn_available(
        development, library_path=library, require_qualified=True
    )

    qualified, qualified_library = _write_bundle(
        tmp_path / "qualified", quality="qualified"
    )
    assert native_ncnn.native256_ncnn_available(
        qualified, library_path=qualified_library, require_qualified=True
    )
    assert not native_ncnn.native256_ncnn_available(
        qualified, library_path=tmp_path / "missing.so"
    )

    onnx_only, onnx_library = _write_bundle(
        tmp_path / "onnx-only", include_ncnn=False
    )
    assert not native_ncnn.native256_ncnn_available(
        onnx_only, library_path=onnx_library
    )


def test_corrupt_ncnn_artifact_is_rejected_before_loading_library(tmp_path):
    manifest, library = _write_bundle(tmp_path)
    (tmp_path / "generator.ncnn.bin").write_bytes(b"tampered")
    loaded = False

    def loader(_path):
        nonlocal loaded
        loaded = True
        return _FakeLibrary()

    with pytest.raises(native_ncnn.Native256NcnnLoadError, match="SHA-256 mismatch"):
        native_ncnn.Native256NcnnSwapper(
            manifest, library_path=library, library_loader=loader
        )
    assert not loaded


def test_manifest_rejects_blob_names_outside_fixed_native_abi(tmp_path):
    manifest, _library = _write_bundle(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["ncnn"]["swapper"]["alpha_output"] = "candidate_again"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    assert not native_ncnn.native256_ncnn_available(
        manifest, library_path=tmp_path / "libnative256.so"
    )
