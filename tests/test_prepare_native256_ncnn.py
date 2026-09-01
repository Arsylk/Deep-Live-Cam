from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys

import onnx
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import prepare_native256_ncnn as prepare  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _write_source_bundle(
    root: Path,
    *,
    quality_status: str = "development",
    auto_select_eligible: bool = False,
) -> Path:
    embedded_onnx = _embedded_onnx()
    assets = {
        "assets/identity_map.npy": b"fixture identity map",
        "onnx/conditioner.onnx": embedded_onnx,
        "onnx/generator.onnx": embedded_onnx,
    }
    for relative, content in assets.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    document = {
        "schema_version": 1,
        "format": "onnx",
        "model_id": "dlc-swap256-test",
        "quality_status": quality_status,
        "auto_select_eligible": auto_select_eligible,
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
            "file": "assets/identity_map.npy",
            "sha256": _sha256(assets["assets/identity_map.npy"]),
        },
        "identity_conditioner": {
            "file": "onnx/conditioner.onnx",
            "sha256": _sha256(assets["onnx/conditioner.onnx"]),
            "input": "mapped_identity",
            "output": "style",
        },
        "swapper": {
            "file": "onnx/generator.onnx",
            "sha256": _sha256(assets["onnx/generator.onnx"]),
            "target_input": "target",
            "style_input": "style",
            "candidate_output": "candidate",
            "alpha_output": "alpha",
        },
    }
    if auto_select_eligible:
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
        (root / "qualification.json").write_bytes(report_data)
        document["qualification_report"] = {
            "file": "qualification.json",
            "sha256": _sha256(report_data),
        }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return manifest


def _write_fake_pnnx(
    path: Path,
    log_path: Path,
    *,
    missing_generator_weights: bool = False,
    custom_generator_layer: bool = False,
) -> Path:
    script = f"""#!/usr/bin/env python3
import json
from pathlib import Path
import sys

if sys.argv[1:] == ["--version"]:
    print("fake-pnnx 1.2.3")
    raise SystemExit(0)

with Path({str(log_path)!r}).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")

options = {{}}
for argument in sys.argv[2:]:
    if "=" in argument:
        key, value = argument.split("=", 1)
        options[key] = value
param = Path(options["ncnnparam"])
weights = Path(options["ncnnbin"])
if param.name.startswith("conditioner"):
    param.write_text(
        "7767517\\n"
        "2 2\\n"
        "Input in_layer 0 1 in0\\n"
        "ReLU output_layer 1 1 in0 out0\\n",
        encoding="utf-8",
    )
    weights.write_bytes(b"conditioner weights")
else:
    layer_type = "DlcCustomLayer" if {custom_generator_layer!r} else "BinaryOp"
    param.write_text(
        "7767517\\n"
        "4 5\\n"
        "Input target_input 0 1 in0\\n"
        "Input style_input 0 1 in1\\n"
        + layer_type + " merge 2 1 in0 in1 merged\\n"
        "Split outputs 1 2 merged out0 out1\\n",
        encoding="utf-8",
    )
    if not {missing_generator_weights!r}:
        weights.write_bytes(b"generator weights")
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_prepare_bundle_is_offline_hash_pinned_atomic_and_contract_valid(tmp_path):
    source_manifest = _write_source_bundle(tmp_path / "source")
    invocation_log = tmp_path / "pnnx invocations.jsonl"
    pnnx = _write_fake_pnnx(
        tmp_path / "explicit pnnx;still-one-argument", invocation_log
    )
    output = tmp_path / "published bundle"

    result = prepare.prepare_bundle(
        manifest_path=source_manifest,
        pnnx_path=pnnx,
        output_dir=output,
        fp16_storage=True,
    )

    assert result == output / "manifest.json"
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["quality_status"] == "development"
    assert document["auto_select_eligible"] is False
    assert document["ncnn"]["backend"] == "ncnn-vulkan"
    assert document["ncnn"]["fp16_storage"] is True
    assert document["ncnn"]["identity_conditioner"]["input"] == "in0"
    assert document["ncnn"]["identity_conditioner"]["output"] == "out0"
    assert document["ncnn"]["swapper"]["target_input"] == "in0"
    assert document["ncnn"]["swapper"]["style_input"] == "in1"
    assert document["ncnn"]["swapper"]["candidate_output"] == "out0"
    assert document["ncnn"]["swapper"]["alpha_output"] == "out1"

    for section_name in ("identity_conditioner", "swapper"):
        section = document["ncnn"][section_name]
        for kind in ("param", "bin"):
            artifact = output / section[f"{kind}_file"]
            assert artifact.is_file()
            assert prepare.sha256_file(artifact) == section[f"{kind}_sha256"]

    # Required ONNX/runtime assets remain in their manifest-relative paths.
    assert (output / "assets/identity_map.npy").read_bytes() == b"fixture identity map"
    assert (output / "onnx/conditioner.onnx").read_bytes() == _embedded_onnx()
    assert (output / "onnx/generator.onnx").read_bytes() == _embedded_onnx()
    loaded = prepare.load_manifest(result)
    assert loaded.ncnn is not None
    assert loaded.ncnn.backend == "ncnn-vulkan"

    audit = json.loads((output / "ncnn-audit.json").read_text(encoding="utf-8"))
    assert audit["pnnx"]["sha256"] == prepare.sha256_file(pnnx)
    assert audit["pnnx"]["version"] == "fake-pnnx 1.2.3"
    commands = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    assert len(commands) == 2
    assert commands[0][0] == "model.onnx"
    assert "inputshape=[1,512,1,1]f32" in commands[0]
    assert "inputshape=[1,3,256,256]f32,[1,2016,1,1]f32" in commands[1]
    assert all("fp16=1" in command for command in commands)


def test_validate_ncnn_param_rejects_custom_layer(tmp_path):
    path = tmp_path / "custom.ncnn.param"
    path.write_text(
        "7767517\n"
        "2 2\n"
        "Input input 0 1 in0\n"
        "pnnx.Expression custom 1 1 in0 out0 expr=sin(@0)\n",
        encoding="utf-8",
    )

    with pytest.raises(prepare.PreparationError, match="unsupported or custom"):
        prepare.validate_ncnn_param(
            path, expected_inputs={"in0"}, expected_outputs={"out0"}
        )


def test_validate_ncnn_param_requires_exact_public_blob_names(tmp_path):
    path = tmp_path / "wrong-abi.ncnn.param"
    path.write_text(
        "7767517\n"
        "2 2\n"
        "Input input 0 1 target\n"
        "ReLU output 1 1 target candidate\n",
        encoding="utf-8",
    )

    with pytest.raises(prepare.PreparationError, match="unexpected ncnn inputs"):
        prepare.validate_ncnn_param(
            path, expected_inputs={"in0"}, expected_outputs={"out0"}
        )


def test_source_hash_mismatch_stops_before_converter_and_publishes_nothing(tmp_path):
    source_manifest = _write_source_bundle(tmp_path / "source")
    (source_manifest.parent / "onnx/generator.onnx").write_bytes(b"tampered")
    log = tmp_path / "invocations"
    pnnx = _write_fake_pnnx(tmp_path / "pnnx", log)
    output = tmp_path / "bundle"

    with pytest.raises(prepare.PreparationError, match="SHA-256 mismatch"):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=pnnx,
            output_dir=output,
            fp16_storage=False,
        )

    assert not log.exists()
    assert not output.exists()


def test_external_onnx_tensor_data_is_rejected_before_converter(tmp_path):
    source_manifest = _write_source_bundle(tmp_path / "source")
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
    generator = source_manifest.parent / "onnx/generator.onnx"
    generator.write_bytes(model.SerializeToString())
    document = json.loads(source_manifest.read_text(encoding="utf-8"))
    document["swapper"]["sha256"] = prepare.sha256_file(generator)
    source_manifest.write_text(json.dumps(document), encoding="utf-8")
    log = tmp_path / "invocations"
    pnnx = _write_fake_pnnx(tmp_path / "pnnx", log)

    with pytest.raises(prepare.PreparationError, match="external ONNX tensor data"):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=pnnx,
            output_dir=tmp_path / "bundle",
            fp16_storage=False,
        )

    assert not log.exists()


@pytest.mark.parametrize(
    ("missing_weights", "custom_layer", "message"),
    [
        (True, False, "weights was not produced"),
        (False, True, "unsupported or custom"),
    ],
)
def test_failed_conversion_cleans_staging_and_does_not_publish(
    tmp_path, missing_weights, custom_layer, message
):
    source_manifest = _write_source_bundle(tmp_path / "source")
    pnnx = _write_fake_pnnx(
        tmp_path / "pnnx",
        tmp_path / "log",
        missing_generator_weights=missing_weights,
        custom_generator_layer=custom_layer,
    )
    output = tmp_path / "bundle"

    with pytest.raises(prepare.PreparationError, match=message):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=pnnx,
            output_dir=output,
            fp16_storage=False,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".bundle.staging-*")) == []


def test_existing_output_and_non_executable_converter_are_rejected(tmp_path):
    source_manifest = _write_source_bundle(tmp_path / "source")
    converter = tmp_path / "pnnx"
    converter.write_bytes(b"not executable")
    output = tmp_path / "bundle"

    with pytest.raises(prepare.PreparationError, match="not executable"):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=converter,
            output_dir=output,
            fp16_storage=False,
        )

    converter = _write_fake_pnnx(converter, tmp_path / "log")
    output.mkdir()
    with pytest.raises(prepare.PreparationError, match="already exists"):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=converter,
            output_dir=output,
            fp16_storage=False,
        )


def test_auto_qualified_release_must_be_converted_then_requalified(tmp_path):
    source_manifest = _write_source_bundle(
        tmp_path / "source",
        quality_status="qualified",
        auto_select_eligible=True,
    )
    log = tmp_path / "log"
    pnnx = _write_fake_pnnx(tmp_path / "pnnx", log)

    with pytest.raises(prepare.PreparationError, match="convert before qualification"):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=pnnx,
            output_dir=tmp_path / "bundle",
            fp16_storage=True,
        )

    assert not log.exists()
    assert not (tmp_path / "bundle").exists()

def test_manifest_with_existing_ncnn_section_is_not_silently_replaced(tmp_path):
    source_manifest = _write_source_bundle(tmp_path / "source")
    document = json.loads(source_manifest.read_text(encoding="utf-8"))
    document["ncnn"] = {"backend": "ncnn-vulkan"}
    source_manifest.write_text(json.dumps(document), encoding="utf-8")
    pnnx = _write_fake_pnnx(tmp_path / "pnnx", tmp_path / "log")

    # Contract validation happens first, so an incomplete existing section is
    # rejected without a converter invocation or a best-effort overwrite.
    with pytest.raises(prepare.PreparationError, match="invalid native-256"):
        prepare.prepare_bundle(
            manifest_path=source_manifest,
            pnnx_path=pnnx,
            output_dir=tmp_path / "bundle",
            fp16_storage=False,
        )
    assert not (tmp_path / "log").exists()
