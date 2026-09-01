"""Export, validate and fingerprint the split DLC-Swap256-M ONNX graphs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from torch import nn

from .config import Native256Config, load_config
from .identity import file_sha256, validate_emap
from .model import (
    DlcSwap256M,
    GENERATOR_MACS,
    IDENTITY_SIZE,
    IdentityConditioner,
    INPUT_SIZE,
    STYLE_SIZE,
    parameter_count,
)


ALLOWED_ONNX_OPS = frozenset(
    {
        "Add",
        "BatchNormalization",
        "Concat",
        "Constant",
        "Conv",
        "Gemm",
        "Identity",
        "LeakyRelu",
        "Mul",
        "Reshape",
        "Resize",
        "Sigmoid",
        "Slice",
        "Tanh",
        "Unsqueeze",
    }
)
GENERATOR_SHAPE_OPS = frozenset({"Expand", "Flatten", "Reshape", "Squeeze", "Unsqueeze"})


class CombinedExport(nn.Module):
    def __init__(self, model: DlcSwap256M) -> None:
        super().__init__()
        self.model = model

    def forward(self, target: torch.Tensor, identity: torch.Tensor):
        return self.model(target, identity)


class NchwConditionerExport(nn.Module):
    """Deployment-only 1x1-convolution form of the trained MLP.

    The weights are exactly the two trained Linear layers with singleton
    kernels. Both input and output are explicit NCHW broadcast tensors, so
    pnnx/ncnn never has to infer whether the identity/style width is a spatial,
    depth, or channel dimension.
    """

    def __init__(self, conditioner: IdentityConditioner) -> None:
        super().__init__()
        hidden = conditioner.project.out_features
        self.project = nn.Conv2d(IDENTITY_SIZE, hidden, 1)
        self.activation = nn.LeakyReLU(0.1, inplace=False)
        self.to_style = nn.Conv2d(hidden, STYLE_SIZE, 1)
        with torch.no_grad():
            self.project.weight.copy_(conditioner.project.weight[:, :, None, None])
            self.project.bias.copy_(conditioner.project.bias)
            self.to_style.weight.copy_(conditioner.to_style.weight[:, :, None, None])
            self.to_style.bias.copy_(conditioner.to_style.bias)

    def forward(self, mapped_identity: torch.Tensor) -> torch.Tensor:
        return self.to_style(self.activation(self.project(mapped_identity)))


def lint_onnx(path: str | Path) -> set[str]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX validation requires the native256 training dependencies"
        ) from error
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    external = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or initializer.external_data
    ]
    if external:
        raise RuntimeError(
            f"{path} uses external ONNX tensor data; deployment bundles must "
            "embed every weight in the hash-pinned model file"
        )
    operators = {node.op_type for node in model.graph.node}
    forbidden = sorted(operators - ALLOWED_ONNX_OPS)
    if forbidden:
        raise RuntimeError(
            f"{path} contains deployment-forbidden ONNX ops: {', '.join(forbidden)}"
        )
    for value in list(model.graph.input) + list(model.graph.output):
        dimensions = [item.dim_value for item in value.type.tensor_type.shape.dim]
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise RuntimeError(f"{path}: {value.name} has a dynamic or unknown shape")
    return operators


def validate_generator_operators(operators: set[str]) -> None:
    """Keep per-frame FiLM broadcasting free of ambiguous shape operators."""
    found = sorted(operators & GENERATOR_SHAPE_OPS)
    if found:
        raise RuntimeError(
            "native-256 generator must use explicit NCHW style broadcasting; "
            f"found shape operators: {', '.join(found)}"
        )


def validate_runtime_manifest(path: Path) -> None:
    """Load the application contract without importing its dependency-heavy package."""
    contract_path = Path(__file__).resolve().parents[1] / "modules" / "swapper_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_dlc_native256_swapper_contract", contract_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load runtime contract from {contract_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        module.Native256Manifest.load(path, verify_hashes=True)
    finally:
        sys.modules.pop(spec.name, None)


def build_runtime_manifest(
    *,
    model_id: str,
    identity_map_path: Path,
    conditioner_path: Path,
    generator_path: Path,
) -> dict[str, object]:
    """Return the exact schema consumed by ``modules.swapper_contract``."""
    return {
        "schema_version": 1,
        "format": "onnx",
        "model_id": model_id,
        "quality_status": "development",
        "auto_select_eligible": False,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "embedding_size": IDENTITY_SIZE,
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
            "file": identity_map_path.name,
            "sha256": file_sha256(identity_map_path),
        },
        "identity_conditioner": {
            "file": conditioner_path.name,
            "sha256": file_sha256(conditioner_path),
            "input": "mapped_identity",
            "output": "style",
        },
        "swapper": {
            "file": generator_path.name,
            "sha256": file_sha256(generator_path),
            "target_input": "target",
            "style_input": "style",
            "candidate_output": "candidate",
            "alpha_output": "alpha",
        },
    }


def export_graph(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> None:
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamo=False,
        )


def ort_parity(
    conditioner_path: Path,
    generator_path: Path,
    model: DlcSwap256M,
    target: torch.Tensor,
    mapped_identity: torch.Tensor,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"checked": 0.0}
    conditioner_session = ort.InferenceSession(
        str(conditioner_path), providers=["CPUExecutionProvider"]
    )
    generator_session = ort.InferenceSession(
        str(generator_path), providers=["CPUExecutionProvider"]
    )
    target_np = target.detach().cpu().numpy()
    identity_np = mapped_identity.detach().cpu().numpy()
    conditioner_input = identity_np.reshape(1, IDENTITY_SIZE, 1, 1)
    style_np = conditioner_session.run(
        None, {"mapped_identity": conditioner_input}
    )[0]
    candidate_np, alpha_np = generator_session.run(
        None, {"target": target_np, "style": style_np}
    )
    with torch.inference_mode():
        style = model.condition(mapped_identity).cpu().numpy()
        candidate, alpha = model.forward_with_style(target, torch.from_numpy(style))
    return {
        "checked": 1.0,
        "style_max_abs": float(np.max(np.abs(style_np - style))),
        "candidate_max_abs": float(
            np.max(np.abs(candidate_np - candidate.cpu().numpy()))
        ),
        "alpha_max_abs": float(np.max(np.abs(alpha_np - alpha.cpu().numpy()))),
    }


def load_model(
    checkpoint_path: Path,
    config: Native256Config,
    identity_metadata: dict[str, str | int],
) -> DlcSwap256M:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "dlc-swap256m-v1":
        raise RuntimeError("checkpoint has an incompatible format")
    if checkpoint.get("identity") != identity_metadata:
        raise RuntimeError(
            "checkpoint identity contract or emap hash does not match "
            "--identity-map"
        )
    state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint does not contain model weights")
    model = DlcSwap256M(config.model).eval()
    model.load_state_dict(state)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--identity-map",
        dest="identity_map",
        type=Path,
        required=True,
        help="exact float32 [512,512] buff2fs map bound to the checkpoint",
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="dlc_swap256m")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="also export a convenience target+identity graph",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.opset < 13:
        raise SystemExit("opset 13 or newer is required")
    config = load_config(args.config)
    identity_metadata = validate_emap(args.identity_map)
    model = load_model(args.checkpoint, config, identity_metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    identity_map_path = args.output / "identity_map.npy"
    if args.identity_map.resolve() != identity_map_path.resolve():
        shutil.copyfile(args.identity_map, identity_map_path)
    copied_identity_metadata = validate_emap(identity_map_path)
    if copied_identity_metadata != identity_metadata:
        raise RuntimeError("copied identity map does not match the checkpoint map")
    torch.manual_seed(config.training.seed)
    target = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    mapped_identity = torch.nn.functional.normalize(
        torch.rand(1, IDENTITY_SIZE, dtype=torch.float32), dim=1
    )
    with torch.inference_mode():
        style = model.condition(mapped_identity)
    conditioner_export = NchwConditionerExport(model.conditioner).eval()
    mapped_identity_nchw = mapped_identity.unsqueeze(2).unsqueeze(3)

    conditioner_path = args.output / "dlc_swap256m_conditioner.onnx"
    generator_path = args.output / "dlc_swap256m_generator.onnx"
    export_graph(
        conditioner_export,
        (mapped_identity_nchw,),
        conditioner_path,
        ["mapped_identity"],
        ["style"],
        args.opset,
    )
    export_graph(
        model.generator,
        (target, style),
        generator_path,
        ["target", "style"],
        ["candidate", "alpha"],
        args.opset,
    )
    paths = [conditioner_path, generator_path]
    if args.combined:
        combined_path = args.output / "dlc_swap256m_combined.onnx"
        export_graph(
            CombinedExport(model),
            (target, mapped_identity),
            combined_path,
            ["target", "mapped_identity"],
            ["candidate", "alpha"],
            args.opset,
        )
        paths.append(combined_path)

    operators: set[str] = set()
    for path in paths:
        path_operators = lint_onnx(path)
        if path == generator_path:
            validate_generator_operators(path_operators)
        operators.update(path_operators)
    parity = ort_parity(
        conditioner_path, generator_path, model, target, mapped_identity
    )
    if parity.get("checked") and max(
        parity["style_max_abs"],
        parity["candidate_max_abs"],
        parity["alpha_max_abs"],
    ) > 1e-4:
        raise RuntimeError(f"ONNX Runtime parity exceeded tolerance: {parity}")
    runtime_manifest = build_runtime_manifest(
        model_id=args.model_id,
        identity_map_path=identity_map_path,
        conditioner_path=conditioner_path,
        generator_path=generator_path,
    )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(runtime_manifest, indent=2) + "\n", encoding="utf-8"
    )
    # Loading the runtime schema here prevents the exporter and application
    # contracts from drifting silently without importing the full app package.
    validate_runtime_manifest(manifest_path)

    audit = {
        "format": "dlc-swap256m-export-audit-v1",
        "opset": args.opset,
        "model_id": args.model_id,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "identity_size": IDENTITY_SIZE,
        "style_size": STYLE_SIZE,
        "conditioner_input_shape": [1, IDENTITY_SIZE, 1, 1],
        "style_shape": [1, STYLE_SIZE, 1, 1],
        "identity": identity_metadata,
        "quality_status": "development",
        "auto_select_eligible": False,
        "parameters": parameter_count(model),
        "generator_macs": GENERATOR_MACS,
        "operators": sorted(operators),
        "ort_parity": parity,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths + [identity_map_path, manifest_path]
        },
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(runtime_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
