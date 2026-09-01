from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image


torch = pytest.importorskip(
    "torch", reason="native-256 training architecture has an optional PyTorch dependency"
)

from native256.config import load_config  # noqa: E402
from native256.dataset import Native256Dataset, read_manifest, resolve_under  # noqa: E402
from native256.identity import IDENTITY_CONTRACT, validate_emap  # noqa: E402
from native256.export_onnx import (  # noqa: E402
    NchwConditionerExport,
    build_runtime_manifest,
    validate_generator_operators,
    validate_runtime_manifest,
)
from native256.losses import mask_foreground_loss, temporal_losses  # noqa: E402
from native256.model import (  # noqa: E402
    CONDITION_CHANNELS,
    DlcSwap256M,
    GENERATOR_MACS,
    IdentityConditioner,
    STYLE_SIZE,
    parameter_count,
    split_style,
)


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_medium_architecture_has_fixed_budget_and_identity_contract():
    model = DlcSwap256M()

    assert STYLE_SIZE == 2016
    assert sum(CONDITION_CHANNELS) == 1008
    assert GENERATOR_MACS == 970_907_648
    assert 560_000 <= parameter_count(model) <= 590_000
    assert IDENTITY_CONTRACT == "inswapper-buff2fs-mapped-l2-v1"


def test_target_identity_and_cached_style_paths_are_equivalent():
    torch.manual_seed(7)
    model = DlcSwap256M().eval()
    target = torch.rand(1, 3, 256, 256)
    mapped_identity = torch.nn.functional.normalize(torch.rand(1, 512), dim=1)

    with torch.inference_mode():
        style = model.condition(mapped_identity)
        direct_candidate, direct_alpha = model(target, mapped_identity)
        cached_candidate, cached_alpha = model.forward_with_style(target, style)

    assert style.shape == (1, 2016, 1, 1)
    # The final conditioner projection deliberately starts at zero for stable
    # reconstruction warm-up.
    assert torch.count_nonzero(style) == 0
    assert direct_candidate.shape == (1, 3, 256, 256)
    assert direct_alpha.shape == (1, 1, 256, 256)
    assert torch.equal(direct_candidate, cached_candidate)
    assert torch.equal(direct_alpha, cached_alpha)
    assert bool(torch.all((direct_candidate >= 0.0) & (direct_candidate <= 1.0)))
    assert bool(torch.all((direct_alpha >= 0.0) & (direct_alpha <= 1.0)))


def test_semantic_alpha_cannot_modify_pixels_outside_safety_prior():
    model = DlcSwap256M().eval()
    target = torch.rand(1, 3, 256, 256)
    identity = torch.nn.functional.normalize(torch.rand(1, 512), dim=1)
    with torch.inference_mode():
        candidate, alpha = model(target, identity)
        output = model.composite(target, candidate, alpha)

    prior = model.generator.mask_head.safety_prior
    outside = prior == 0.0
    assert int(outside.sum()) > 10_000
    assert torch.count_nonzero(alpha[outside]) == 0
    assert torch.equal(
        output.expand_as(target)[outside.expand_as(target)],
        target[outside.expand_as(target)],
    )


@pytest.mark.parametrize(
    ("target_shape", "identity_shape", "message"),
    [
        ((1, 3, 255, 256), (1, 512), "target must have shape"),
        ((1, 3, 256, 256), (1, 511), "mapped_identity must have shape"),
    ],
)
def test_eager_graph_rejects_noncontract_shapes(
    target_shape, identity_shape, message
):
    model = DlcSwap256M().eval()
    with pytest.raises(ValueError, match=message):
        model(torch.zeros(target_shape), torch.zeros(identity_shape))


def test_deployment_graph_avoids_known_problematic_layer_classes():
    model = DlcSwap256M().eval()
    forbidden = (
        torch.nn.ConvTranspose2d,
        torch.nn.InstanceNorm2d,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
    )
    assert not any(isinstance(module, forbidden) for module in model.modules())
    assert isinstance(model.conditioner, IdentityConditioner)


def test_film_conditions_are_explicit_nchw_broadcast_tensors():
    model = DlcSwap256M().eval()
    style = model.condition(torch.zeros(2, 512))

    assert style.shape == (2, STYLE_SIZE, 1, 1)
    conditions = split_style(style)
    assert tuple(gamma.shape[1:] for gamma, _ in conditions) == tuple(
        (channels, 1, 1) for channels in CONDITION_CHANNELS
    )
    assert tuple(beta.shape for _, beta in conditions) == tuple(
        gamma.shape for gamma, _ in conditions
    )


def test_nchw_export_conditioner_is_the_trained_linear_map():
    torch.manual_seed(19)
    conditioner = IdentityConditioner()
    torch.nn.init.normal_(conditioner.to_style.weight, std=0.01)
    torch.nn.init.normal_(conditioner.to_style.bias, std=0.01)
    exported = NchwConditionerExport(conditioner)
    identity = torch.randn(2, 512)

    with torch.inference_mode():
        expected = conditioner(identity)
        actual = exported(identity[:, :, None, None])

    assert actual.shape == (2, STYLE_SIZE, 1, 1)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_generator_export_rejects_shape_ops_in_per_frame_graph():
    validate_generator_operators({"Conv", "Slice", "Mul"})
    with pytest.raises(RuntimeError, match="explicit NCHW"):
        validate_generator_operators({"Conv", "Reshape"})


def test_semantic_mask_fuses_every_target_encoder_scale_with_gradients():
    head = DlcSwap256M().generator.mask_head
    features = (
        torch.randn(1, 160, 16, 16, requires_grad=True),
        torch.randn(1, 112, 32, 32, requires_grad=True),
        torch.randn(1, 72, 64, 64, requires_grad=True),
        torch.randn(1, 48, 128, 128, requires_grad=True),
        torch.randn(1, 32, 256, 256, requires_grad=True),
    )

    head(*features).mean().backward()

    assert all(feature.grad is not None for feature in features)
    assert all(torch.count_nonzero(feature.grad) > 0 for feature in features)


def test_foreground_mask_supervision_penalizes_false_negatives():
    face_mask = torch.ones(1, 1, 8, 8)

    missing = mask_foreground_loss(torch.zeros_like(face_mask), face_mask)
    covered = mask_foreground_loss(torch.ones_like(face_mask), face_mask)

    assert float(missing) == pytest.approx(1.0)
    assert float(covered) == pytest.approx(0.0)


def test_default_config_is_strict_and_matches_architecture():
    config = load_config("native256/config.json")

    assert config.model.input_size == 256
    assert config.model.identity_size == 512
    assert config.model.channels == (32, 48, 72, 112, 160)
    assert config.losses.identity == 8.0
    assert config.losses.mask_foreground == 2.0


def _record(sample_id: str, split: str, identity: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "target": f"crops/{sample_id}.png",
        "target_identity": identity,
        "source_identity": identity,
        "source_latent": f"latents/{identity}.npy",
        "identity_type": "inswapper_mapped_512",
        "identity_map_sha256": "0" * 64,
        "source_embedding": f"embeddings/{identity}.npy",
        "teacher_candidate": None,
        "teacher_alpha": None,
        "teacher_type": None,
        "face_mask": f"masks/{sample_id}.png",
        "teacher_confidence": 1.0,
        "provenance": "authorized test fixture",
        "license": "test-only",
        "usage_authorized": True,
    }


def test_manifest_rejects_identity_leakage_and_path_escape(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                _record("a", "train", "person-1"),
                _record("b", "validation", "person-1"),
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity leakage"):
        read_manifest(path)
    with pytest.raises(ValueError, match="escapes data root"):
        resolve_under(tmp_path, "../outside.png")


def test_manifest_requires_explicit_authorization(tmp_path):
    item = _record("a", "train", "person-1")
    item["usage_authorized"] = False
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(item), encoding="utf-8")

    with pytest.raises(ValueError, match="usage_authorized"):
        read_manifest(path)


def test_manifest_requires_unambiguous_composited_teacher_contract(tmp_path):
    item = _record("a", "train", "person-1")
    item["source_identity"] = "person-2"
    item["teacher_candidate"] = "teacher/a.png"
    item["teacher_alpha"] = "teacher/a-alpha.png"
    item["teacher_type"] = "raw-candidate"
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(item), encoding="utf-8")

    with pytest.raises(ValueError, match="teacher_type must be"):
        read_manifest(path)


def test_identity_map_is_shape_dtype_and_hash_pinned(tmp_path):
    path = tmp_path / "emap.npy"
    np.save(path, np.eye(512, dtype=np.float32), allow_pickle=False)

    metadata = validate_emap(path)

    assert metadata["contract"] == IDENTITY_CONTRACT
    assert len(str(metadata["emap_sha256"])) == 64
    assert metadata["emap_bytes"] == path.stat().st_size


def test_dataset_verifies_mapped_identity_and_loads_temporal_tuple(tmp_path):
    for name in ("crops", "latents", "embeddings", "masks", "flow"):
        (tmp_path / name).mkdir()
    rgb = np.full((256, 256, 3), 96, dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "crops/a.png")
    Image.fromarray(rgb + 1).save(tmp_path / "crops/a-next.png")
    Image.fromarray(np.full((256, 256), 255, dtype=np.uint8)).save(
        tmp_path / "masks/a.png"
    )
    identity_map = np.diag(np.linspace(0.5, 1.5, 512, dtype=np.float32))
    source_embedding = np.arange(1, 513, dtype=np.float32)
    source_embedding /= np.linalg.norm(source_embedding)
    source_latent = source_embedding.reshape(1, 512) @ identity_map
    source_latent = (source_latent / np.linalg.norm(source_latent)).reshape(512)
    np.save(tmp_path / "embeddings/person-1.npy", source_embedding)
    np.save(tmp_path / "latents/person-1.npy", source_latent.astype(np.float32))
    np.save(tmp_path / "flow/a.npy", np.zeros((2, 256, 256), dtype=np.float32))
    np.save(tmp_path / "flow/a-valid.npy", np.ones((1, 256, 256), dtype=np.float32))
    item = _record("a", "train", "person-1")
    item.update(
        {
            "next_target": "crops/a-next.png",
            "flow": "flow/a.npy",
            "flow_valid": "flow/a-valid.npy",
        }
    )
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(item), encoding="utf-8")

    dataset = Native256Dataset(manifest, tmp_path, identity_map=identity_map)
    sample = dataset[0]

    assert bool(sample["has_temporal"])
    assert sample["next_target"].shape == (3, 256, 256)
    assert sample["flow"].shape == (2, 256, 256)
    assert sample["flow_valid"].shape == (1, 256, 256)

    np.save(
        tmp_path / "latents/person-1.npy",
        np.roll(source_latent, 1).astype(np.float32),
    )
    with pytest.raises(ValueError, match="does not equal"):
        dataset[0]


def test_export_manifest_is_runtime_schema_v1_and_never_auto_qualified(tmp_path):
    identity_map = tmp_path / "identity_map.npy"
    np.save(identity_map, np.eye(512, dtype=np.float32), allow_pickle=False)
    conditioner = tmp_path / "conditioner.onnx"
    generator = tmp_path / "generator.onnx"
    conditioner.write_bytes(b"conditioner fixture")
    generator.write_bytes(b"generator fixture")
    document = build_runtime_manifest(
        model_id="dlc_swap256m_test",
        identity_map_path=identity_map,
        conditioner_path=conditioner,
        generator_path=generator,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    validate_runtime_manifest(manifest)

    assert document["schema_version"] == 1
    assert document["format"] == "onnx"
    assert document["quality_status"] == "development"
    assert document["auto_select_eligible"] is False
    assert document["identity_type"] == "inswapper_mapped_512"
    assert document["candidate_precomposited"] is False


def test_residual_temporal_loss_is_zero_for_consistent_motion():
    target = torch.rand(1, 3, 32, 32)
    output = target + 0.05
    alpha = torch.full((1, 1, 32, 32), 0.5)
    flow = torch.zeros(1, 2, 32, 32)
    valid = torch.ones(1, 1, 32, 32)

    residual, mask = temporal_losses(
        output,
        target,
        alpha,
        output,
        target,
        alpha,
        flow,
        valid,
    )

    assert float(residual) == pytest.approx(0.0, abs=1e-7)
    assert float(mask) == pytest.approx(0.0, abs=1e-7)
