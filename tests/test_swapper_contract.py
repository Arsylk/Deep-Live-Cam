from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from modules.swapper_contract import (
    Native256Manifest,
    SwapResult,
    SwapperContractError,
    mapped_inswapper_identity,
    normalized_embedding,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(tmp_path, **updates):
    conditioner_data = b"conditioner"
    swapper_data = b"swapper"
    identity_map = np.eye(512, dtype=np.float32)
    (tmp_path / "identity.onnx").write_bytes(conditioner_data)
    (tmp_path / "swapper.onnx").write_bytes(swapper_data)
    np.save(tmp_path / "identity_map.npy", identity_map, allow_pickle=False)
    identity_map_data = (tmp_path / "identity_map.npy").read_bytes()
    document = {
        "schema_version": 1,
        "format": "onnx",
        "model_id": "distilled_native_256",
        "quality_status": "development",
        "auto_select_eligible": False,
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
    document.update(updates)
    if (
        document["quality_status"] == "qualified"
        and document["auto_select_eligible"] is True
        and "qualification_report" not in updates
    ):
        report = {
            "schema_version": 1,
            "model_id": document["model_id"],
            "verdict": "qualified",
            "artifacts": {
                "identity_map_sha256": document["identity_map"]["sha256"],
                "identity_conditioner_sha256": document[
                    "identity_conditioner"
                ]["sha256"],
                "swapper_sha256": document["swapper"]["sha256"],
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
        document["qualification_report"] = {
            "file": "qualification.json",
            "sha256": _sha256(report_data),
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_manifest_is_local_self_describing_and_hash_pinned(tmp_path):
    path, _ = _write_manifest(tmp_path)

    manifest = Native256Manifest.load(path)

    assert manifest.spec.model_id == "distilled_native_256"
    assert manifest.spec.input_size == (256, 256)
    assert manifest.spec.embedding_size == 512
    assert manifest.spec.identity_type == "inswapper_mapped_512"
    assert manifest.quality_status == "development"
    assert not manifest.auto_select_eligible
    assert manifest.spec.has_semantic_alpha
    assert not manifest.spec.face_precomposited
    assert manifest.conditioner.asset.path == (tmp_path / "identity.onnx").resolve()
    assert manifest.swapper.asset.path == (tmp_path / "swapper.onnx").resolve()
    assert manifest.swapper.candidate_output_name == "candidate_rgb"
    assert manifest.swapper.alpha_output_name == "face_alpha"


@pytest.mark.parametrize(
    "quality_status", ["development", "experimental", "qualified"]
)
def test_manifest_accepts_each_defined_quality_status(tmp_path, quality_status):
    path, _ = _write_manifest(
        tmp_path,
        quality_status=quality_status,
        auto_select_eligible=quality_status == "qualified",
    )

    assert Native256Manifest.load(path).quality_status == quality_status


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_size", [128, 128], "input_size must be"),
        ("embedding_size", 256, "embedding_size must be 512"),
        ("layout", "NHWC", "layout must be 'NCHW'"),
        ("color_order", "BGR", "color_order must be 'RGB'"),
        (
            "identity_type",
            "raw_arcface",
            "identity_type must be 'inswapper_mapped_512'",
        ),
        ("quality_status", "ready", "quality_status must be"),
        ("auto_select_eligible", "yes", "must be a JSON boolean"),
        (
            "candidate_precomposited",
            True,
            "candidate RGB must be uncomposited",
        ),
        ("alpha_range", [-1.0, 1.0], "alpha_range must be"),
    ],
)
def test_manifest_rejects_incompatible_native256_contract(
    tmp_path, field, value, message
):
    path, _ = _write_manifest(tmp_path, **{field: value})

    with pytest.raises(SwapperContractError, match=message):
        Native256Manifest.load(path)


def test_auto_select_eligibility_requires_qualified_status(tmp_path):
    path, _ = _write_manifest(
        tmp_path,
        quality_status="development",
        auto_select_eligible=True,
    )

    with pytest.raises(SwapperContractError, match="only for a qualified"):
        Native256Manifest.load(path)


def test_auto_select_eligibility_requires_hash_pinned_qualification(tmp_path):
    path, document = _write_manifest(
        tmp_path,
        quality_status="qualified",
        auto_select_eligible=True,
    )
    document.pop("qualification_report")
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SwapperContractError, match="qualification_report"):
        Native256Manifest.load(path)


def test_qualification_report_binds_artifacts_and_every_release_gate(tmp_path):
    path, document = _write_manifest(
        tmp_path,
        quality_status="qualified",
        auto_select_eligible=True,
    )
    report_path = tmp_path / "qualification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"]["swapper_sha256"] = "0" * 64
    report_data = json.dumps(report).encode("utf-8")
    report_path.write_bytes(report_data)
    document["qualification_report"]["sha256"] = _sha256(report_data)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SwapperContractError, match="does not match"):
        Native256Manifest.load(path)

    report["artifacts"]["swapper_sha256"] = document["swapper"]["sha256"]
    report["gates"]["blind_review"] = False
    report_data = json.dumps(report).encode("utf-8")
    report_path.write_bytes(report_data)
    document["qualification_report"]["sha256"] = _sha256(report_data)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SwapperContractError, match="unpassed release gates"):
        Native256Manifest.load(path)


def test_manifest_rejects_remote_or_escaping_model_paths(tmp_path):
    path, document = _write_manifest(tmp_path)
    document["identity_conditioner"]["file"] = "https://example.test/model.onnx"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SwapperContractError, match="relative local path"):
        Native256Manifest.load(path)

    document["identity_conditioner"]["file"] = "../identity.onnx"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SwapperContractError, match="manifest directory"):
        Native256Manifest.load(path)


def test_manifest_rejects_model_hash_mismatch(tmp_path):
    path, document = _write_manifest(tmp_path)
    document["swapper"]["sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SwapperContractError, match="SHA-256 mismatch"):
        Native256Manifest.load(path)


def test_swap_result_requires_matching_uint8_face_and_alpha():
    face = np.zeros((256, 256, 3), dtype=np.uint8)
    alpha = np.full((256, 256), 127, dtype=np.uint8)
    affine = np.array([[1, 0, 3], [0, 1, 4]], dtype=np.float32)

    result = SwapResult(face, affine, alpha, "native_256", "ort")

    assert result.has_semantic_alpha
    with pytest.raises(SwapperContractError, match="alpha shape"):
        SwapResult(face, affine, alpha[:128], "native_256", "ort")
    with pytest.raises(SwapperContractError, match="face_bgr must have dtype"):
        SwapResult(face.astype(np.float32), affine, alpha, "native_256", "ort")
    with pytest.raises(SwapperContractError, match="affine contains"):
        bad_affine = affine.copy()
        bad_affine[0, 0] = np.nan
        SwapResult(face, bad_affine, alpha, "native_256", "ort")


def test_legacy_result_may_explicitly_have_no_semantic_alpha():
    result = SwapResult(
        np.zeros((128, 128, 3), dtype=np.uint8),
        np.eye(2, 3, dtype=np.float32),
        None,
        "inswapper_128",
        "ort",
    )

    assert not result.has_semantic_alpha


def test_source_embedding_is_finite_normalized_and_batch_shaped():
    embedding = normalized_embedding(np.arange(1, 513, dtype=np.float64))

    assert embedding.shape == (1, 512)
    assert embedding.dtype == np.float32
    assert embedding.flags.c_contiguous
    assert np.linalg.norm(embedding) == pytest.approx(1.0)

    with pytest.raises(SwapperContractError, match="zero norm"):
        normalized_embedding(np.zeros(512, dtype=np.float32))
    with pytest.raises(SwapperContractError, match="not finite"):
        invalid = np.ones(512, dtype=np.float32)
        invalid[3] = np.nan
        normalized_embedding(invalid)


def test_inswapper_identity_is_projected_once_then_normalized():
    raw = np.arange(1, 513, dtype=np.float32)
    identity_map = np.diag(np.linspace(0.5, 2.0, 512, dtype=np.float32))

    mapped = mapped_inswapper_identity(raw, identity_map)
    expected = raw.reshape(1, 512) @ identity_map
    expected /= np.linalg.norm(expected)
    mapped_twice = expected @ identity_map
    mapped_twice /= np.linalg.norm(mapped_twice)

    assert np.allclose(mapped, expected, rtol=0.0, atol=1e-7)
    assert not np.allclose(mapped, mapped_twice, rtol=0.0, atol=1e-5)
