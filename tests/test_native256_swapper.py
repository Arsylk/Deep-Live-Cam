from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import onnx
import pytest

import modules.native256_swapper as native256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity_map():
    return np.diag(np.linspace(0.5, 2.0, 512, dtype=np.float32))


def _embedded_onnx():
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


def _write_bundle(
    tmp_path,
    *,
    quality_status="development",
    auto_select_eligible=None,
    identity_map=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    conditioner_data = _embedded_onnx()
    swapper_data = _embedded_onnx()
    if identity_map is None:
        identity_map = _identity_map()
    (tmp_path / "identity.onnx").write_bytes(conditioner_data)
    (tmp_path / "swapper.onnx").write_bytes(swapper_data)
    np.save(tmp_path / "identity_map.npy", identity_map, allow_pickle=False)
    identity_map_data = (tmp_path / "identity_map.npy").read_bytes()
    eligible = (
        quality_status == "qualified"
        if auto_select_eligible is None
        else auto_select_eligible
    )
    manifest = {
        "schema_version": 1,
        "format": "onnx",
        "model_id": "test_native_256",
        "quality_status": quality_status,
        "auto_select_eligible": eligible,
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
            "sha256": _sha256(identity_map_data),
        },
        "identity_conditioner": {
            "file": "identity.onnx",
            "sha256": _sha256(conditioner_data),
            "input": "identity_embedding",
            "output": "conditioned_style",
        },
        "swapper": {
            "file": "swapper.onnx",
            "sha256": _sha256(swapper_data),
            "target_input": "target_rgb",
            "style_input": "conditioned_style",
            "candidate_output": "candidate_rgb",
            "alpha_output": "face_alpha",
        },
    }
    if eligible:
        report = {
            "schema_version": 1,
            "model_id": manifest["model_id"],
            "verdict": "qualified",
            "artifacts": {
                "identity_map_sha256": manifest["identity_map"]["sha256"],
                "identity_conditioner_sha256": manifest[
                    "identity_conditioner"
                ]["sha256"],
                "swapper_sha256": manifest["swapper"]["sha256"],
            },
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
            "sha256": _sha256(report_data),
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@dataclass
class _Node:
    name: str
    shape: list[int]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(
        self,
        kind,
        *,
        alpha_shape=(1, 1, 256, 256),
        conditioner_nchw=False,
    ):
        self.kind = kind
        self.alpha_shape = alpha_shape
        self.conditioner_nchw = conditioner_nchw
        self.calls = []
        self.candidate_value = None
        self.alpha_value = None

    def get_inputs(self):
        if self.kind == "identity":
            shape = [1, 512, 1, 1] if self.conditioner_nchw else [1, 512]
            return [_Node("identity_embedding", shape)]
        style_shape = [1, 8, 1, 1] if self.conditioner_nchw else [1, 8]
        return [
            _Node("target_rgb", [1, 3, 256, 256]),
            _Node("conditioned_style", style_shape),
        ]

    def get_outputs(self):
        if self.kind == "identity":
            shape = [1, 8, 1, 1] if self.conditioner_nchw else [1, 8]
            return [_Node("conditioned_style", shape)]
        return [
            _Node("candidate_rgb", [1, 3, 256, 256]),
            _Node("face_alpha", list(self.alpha_shape)),
        ]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feeds):
        self.calls.append((list(output_names), dict(feeds)))
        if self.kind == "identity":
            assert output_names == ["conditioned_style"]
            value = feeds["identity_embedding"].reshape(1, 512)[:, :8]
            if self.conditioner_nchw:
                value = value.reshape(1, 8, 1, 1)
            return [np.ascontiguousarray(value)]
        assert output_names == ["candidate_rgb", "face_alpha"]
        candidate = np.empty((1, 3, 256, 256), dtype=np.float32)
        candidate[:, 0] = 1.0
        candidate[:, 1] = 0.5
        candidate[:, 2] = 0.0
        alpha = np.full((1, 1, 256, 256), 0.25, dtype=np.float32)
        if self.candidate_value is not None:
            candidate.fill(self.candidate_value)
        if self.alpha_value is not None:
            alpha.fill(self.alpha_value)
        return [candidate, alpha]


class _SessionFactory:
    def __init__(
        self,
        *,
        alpha_shape=(1, 1, 256, 256),
        conditioner_nchw=False,
    ):
        self.identity = _FakeSession(
            "identity", conditioner_nchw=conditioner_nchw
        )
        self.swapper = _FakeSession(
            "swapper",
            alpha_shape=alpha_shape,
            conditioner_nchw=conditioner_nchw,
        )
        self.created = []

    def __call__(self, path, *, sess_options=None, providers=None):
        self.created.append((path, providers, sess_options))
        return self.identity if path.endswith("identity.onnx") else self.swapper


def _faces():
    target = SimpleNamespace(kps=np.arange(10, dtype=np.float32).reshape(5, 2))
    source = SimpleNamespace(normed_embedding=np.arange(1, 513, dtype=np.float32))
    return target, source


def _aligned(monkeypatch):
    crop = np.full((256, 256, 3), (10, 20, 30), dtype=np.uint8)
    affine = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    observed = {}

    def fake_norm_crop(image, keypoints, size):
        observed["image"] = image
        observed["keypoints"] = keypoints
        observed["size"] = size
        return crop, affine

    monkeypatch.setattr(native256.face_align, "norm_crop2", fake_norm_crop)
    return crop, affine, observed


def test_native256_runs_named_two_stage_contract_and_returns_semantic_mask(
    tmp_path, monkeypatch
):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory()
    crop, affine, observed = _aligned(monkeypatch)
    swapper = native256.Native256Swapper(
        manifest,
        providers=["CPUExecutionProvider"],
        session_factory=factory,
    )
    target, source = _faces()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = swapper.infer(frame, target, source)

    assert observed["size"] == 256
    assert np.array_equal(observed["keypoints"], target.kps)
    assert np.array_equal(result.affine, affine)
    assert result.face_bgr.dtype == np.uint8
    # Model candidate is RGB (1.0, 0.5, 0.0); application output is BGR.
    assert np.array_equal(result.face_bgr[0, 0], [0, 127, 255])
    assert result.alpha.dtype == np.uint8
    assert int(result.alpha[0, 0]) == 64
    assert result.model_id == "test_native_256"
    assert result.backend == "ort"
    assert result.has_semantic_alpha
    assert swapper.input_size == (256, 256)
    assert swapper.device_name == "CPUExecutionProvider"

    assert len(factory.identity.calls) == 1
    assert len(factory.swapper.calls) == 1
    output_names, feeds = factory.swapper.calls[0]
    assert output_names == ["candidate_rgb", "face_alpha"]
    assert feeds["target_rgb"].shape == (1, 3, 256, 256)
    # BGR crop (10,20,30) becomes normalized RGB planes (30,20,10).
    assert feeds["target_rgb"][0, 0, 0, 0] == pytest.approx(30 / 255)
    assert feeds["target_rgb"][0, 1, 0, 0] == pytest.approx(20 / 255)
    assert feeds["target_rgb"][0, 2, 0, 0] == pytest.approx(10 / 255)
    assert feeds["conditioned_style"].shape == (1, 8)
    conditioner_feed = factory.identity.calls[0][1]["identity_embedding"]
    expected_latent = source.normed_embedding.reshape(1, 512) @ _identity_map()
    expected_latent /= np.linalg.norm(expected_latent)
    mapped_twice = expected_latent @ _identity_map()
    mapped_twice /= np.linalg.norm(mapped_twice)
    assert np.allclose(conditioner_feed, expected_latent, rtol=0.0, atol=1e-7)
    assert not np.allclose(
        conditioner_feed, mapped_twice, rtol=0.0, atol=1e-5
    )
    assert all(item[1] == ["CPUExecutionProvider"] for item in factory.created)
    assert np.array_equal(crop[0, 0], [10, 20, 30])


def test_style_is_cached_by_normalized_source_embedding(tmp_path, monkeypatch):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory()
    _aligned(monkeypatch)
    swapper = native256.Native256Swapper(
        manifest, session_factory=factory, style_cache_entries=2
    )
    target, source = _faces()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    swapper.get(frame, target, source)
    # Scaling the same vector produces the same normalized embedding/cache key.
    equivalent = SimpleNamespace(normed_embedding=source.normed_embedding * 2.0)
    swapper.get(frame, target, equivalent)

    assert len(factory.identity.calls) == 1
    assert len(factory.swapper.calls) == 2
    assert swapper.style_cache_size == 1

    different = SimpleNamespace(
        normed_embedding=np.arange(512, 0, -1, dtype=np.float32)
    )
    swapper.get(frame, target, different)
    assert len(factory.identity.calls) == 2
    assert swapper.style_cache_size == 2


def test_nchw_conditioner_input_preserves_identity_semantics(tmp_path, monkeypatch):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory(conditioner_nchw=True)
    _aligned(monkeypatch)
    swapper = native256.Native256Swapper(manifest, session_factory=factory)
    target, source = _faces()

    swapper.infer(np.zeros((300, 300, 3), dtype=np.uint8), target, source)

    conditioner_feed = factory.identity.calls[0][1]["identity_embedding"]
    style_feed = factory.swapper.calls[0][1]["conditioned_style"]
    assert conditioner_feed.shape == (1, 512, 1, 1)
    assert style_feed.shape == (1, 8, 1, 1)
    expected = source.normed_embedding.reshape(1, 512) @ _identity_map()
    expected /= np.linalg.norm(expected)
    assert np.allclose(
        conditioner_feed.reshape(1, 512), expected, rtol=0.0, atol=1e-7
    )


def test_style_cache_is_bounded_and_can_be_cleared(tmp_path, monkeypatch):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory()
    _aligned(monkeypatch)
    swapper = native256.Native256Swapper(
        manifest, session_factory=factory, style_cache_entries=1
    )
    target, first = _faces()
    second = SimpleNamespace(normed_embedding=np.linspace(2, 5, 512, dtype=np.float32))
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    swapper.infer(frame, target, first)
    swapper.infer(frame, target, second)
    assert swapper.style_cache_size == 1
    swapper.clear_style_cache()
    assert swapper.style_cache_size == 0


def test_session_tensor_metadata_is_validated_before_inference(tmp_path):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory(alpha_shape=(1, 3, 256, 256))

    with pytest.raises(native256.Native256LoadError, match="alpha output must have shape"):
        native256.Native256Swapper(manifest, session_factory=factory)


def test_identity_map_hash_shape_and_finite_values_are_strict(tmp_path):
    corrupt_dir = tmp_path / "corrupt"
    corrupt_manifest = _write_bundle(corrupt_dir)
    document = json.loads(corrupt_manifest.read_text(encoding="utf-8"))
    document["identity_map"]["sha256"] = "0" * 64
    corrupt_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(native256.Native256LoadError, match="SHA-256 mismatch"):
        native256.Native256Swapper(
            corrupt_manifest, session_factory=_SessionFactory()
        )

    wrong_shape_dir = tmp_path / "wrong-shape"
    wrong_shape_manifest = _write_bundle(wrong_shape_dir)
    np.save(
        wrong_shape_dir / "identity_map.npy",
        np.eye(256, dtype=np.float32),
        allow_pickle=False,
    )
    document = json.loads(wrong_shape_manifest.read_text(encoding="utf-8"))
    document["identity_map"]["sha256"] = _sha256(
        (wrong_shape_dir / "identity_map.npy").read_bytes()
    )
    wrong_shape_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(native256.Native256LoadError, match=r"shape \(512, 512\)"):
        native256.Native256Swapper(
            wrong_shape_manifest, session_factory=_SessionFactory()
        )

    nonfinite_dir = tmp_path / "nonfinite"
    nonfinite_map = _identity_map()
    nonfinite_map[2, 3] = np.nan
    nonfinite_manifest = _write_bundle(
        nonfinite_dir, identity_map=nonfinite_map
    )
    with pytest.raises(native256.Native256LoadError, match="non-finite"):
        native256.Native256Swapper(
            nonfinite_manifest, session_factory=_SessionFactory()
        )


def test_quality_status_controls_only_automatic_selection(tmp_path):
    development = _write_bundle(
        tmp_path / "development", quality_status="development"
    )

    assert native256.native256_swapper_available(development)
    assert not native256.native256_swapper_available(
        development, require_qualified=True
    )
    assert native256.native256_swapper_quality_status(development) == "development"
    assert native256.load_native256_manifest(development).quality_status == "development"
    explicit = native256.Native256Swapper(
        development, session_factory=_SessionFactory()
    )
    assert explicit.quality_status == "development"
    with pytest.raises(native256.Native256LoadError, match="not eligible"):
        native256.Native256Swapper(
            development,
            session_factory=_SessionFactory(),
            require_qualified=True,
        )

    qualified = _write_bundle(
        tmp_path / "qualified", quality_status="qualified"
    )
    assert native256.native256_swapper_available(
        qualified, require_qualified=True
    )
    automatic = native256.Native256Swapper(
        qualified,
        session_factory=_SessionFactory(),
        require_qualified=True,
    )
    assert automatic.manifest.auto_select_eligible

    qualified_but_blocked = _write_bundle(
        tmp_path / "qualified-blocked",
        quality_status="qualified",
        auto_select_eligible=False,
    )
    assert not native256.native256_swapper_available(
        qualified_but_blocked, require_qualified=True
    )
    assert automatic.quality_status == "qualified"


@pytest.mark.parametrize(
    ("candidate", "alpha", "message"),
    [
        (np.nan, None, "candidate contains a non-finite"),
        (1.1, None, "candidate is outside"),
        (None, np.nan, "alpha contains a non-finite"),
        (None, -0.1, "alpha is outside"),
    ],
)
def test_invalid_runtime_outputs_fail_without_model_fallback(
    tmp_path, monkeypatch, candidate, alpha, message
):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory()
    factory.swapper.candidate_value = candidate
    factory.swapper.alpha_value = alpha
    _aligned(monkeypatch)
    swapper = native256.Native256Swapper(manifest, session_factory=factory)
    target, source = _faces()

    with pytest.raises(native256.Native256InferenceError, match=message):
        swapper.infer(np.zeros((256, 256, 3), dtype=np.uint8), target, source)

    # The explicitly constructed adapter remains the same model; no legacy
    # session is created as a side effect of the failed frame.
    assert len(factory.created) == 2
    assert swapper.spec.model_id == "test_native_256"


def test_invalid_source_target_and_paste_requests_are_rejected(tmp_path, monkeypatch):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory()
    _aligned(monkeypatch)
    swapper = native256.Native256Swapper(manifest, session_factory=factory)
    target, source = _faces()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="aligned output"):
        swapper.get(frame, target, source, paste_back=True)
    with pytest.raises(native256.Native256InferenceError, match="identity embedding"):
        swapper.infer(frame, target, SimpleNamespace(normed_embedding=None))
    with pytest.raises(native256.Native256InferenceError, match="five keypoints"):
        swapper.infer(frame, SimpleNamespace(kps=None), source)
    with pytest.raises(native256.Native256InferenceError, match="uint8 HxWx3"):
        swapper.infer(frame.astype(np.float32), target, source)


def test_missing_bundle_is_an_explicit_load_error_and_availability_is_read_only(tmp_path):
    missing = tmp_path / "missing.json"

    assert not native256.native256_swapper_available(missing)
    with pytest.raises(native256.Native256LoadError, match="does not exist"):
        native256.Native256Swapper(missing, session_factory=_SessionFactory())


def test_external_onnx_weights_are_not_runtime_available(tmp_path):
    manifest = _write_bundle(tmp_path)
    tensor = onnx.TensorProto()
    tensor.name = "external_weight"
    tensor.data_type = onnx.TensorProto.FLOAT
    tensor.dims.extend([1])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = "mutable-weights.bin"
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "external", [], [], [tensor])
    )
    swapper_path = tmp_path / "swapper.onnx"
    swapper_path.write_bytes(model.SerializeToString())
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["swapper"]["sha256"] = _sha256(swapper_path.read_bytes())
    manifest.write_text(json.dumps(document), encoding="utf-8")

    assert not native256.native256_swapper_available(manifest)


def test_closed_swapper_does_not_reinitialize_or_fallback(tmp_path, monkeypatch):
    manifest = _write_bundle(tmp_path)
    factory = _SessionFactory()
    _aligned(monkeypatch)
    swapper = native256.Native256Swapper(manifest, session_factory=factory)
    target, source = _faces()

    swapper.prepare_source(source)
    swapper.close()

    assert swapper.style_cache_size == 0
    with pytest.raises(native256.Native256InferenceError, match="closed"):
        swapper.infer(np.zeros((256, 256, 3), dtype=np.uint8), target, source)
    assert len(factory.created) == 2
