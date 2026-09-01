#!/usr/bin/env python3
"""Prepare a deterministic, offline model pack for the mobile prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.tools.make_dynamic_shape_fixed import make_input_shape_fixed
from onnxruntime.transformers.float16 import convert_float_to_float16


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from modules.onnx_optimize import _decompose_reflect_pad, _fold_shape_gather  # noqa: E402


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, path)


def topological_sort(model: onnx.ModelProto) -> None:
    """Restore node order after ORT's FP16 converter appends graph-I/O casts."""
    available = {item.name for item in model.graph.input}
    available.update(item.name for item in model.graph.initializer)
    pending = list(model.graph.node)
    ordered = []
    while pending:
        ready = [
            node
            for node in pending
            if all(not name or name in available for name in node.input)
        ]
        if not ready:
            names = ", ".join(node.name or node.op_type for node in pending[:5])
            raise RuntimeError(f"could not topologically sort converted model near {names}")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
            pending.remove(node)
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def fixed_detector(source: Path, target: Path) -> None:
    model = onnx.load(source)
    make_input_shape_fixed(model.graph, "input.1", [1, 3, 640, 640])
    _fold_shape_gather(model, (1, 3, 640, 640))
    # SCRFD's three residual downsample pools receive 160, 80, and 40 pixel
    # feature maps. For those even dimensions ceil_mode=1 and ceil_mode=0 are
    # exactly equivalent, while Qualcomm's GPU backend only accepts mode 0.
    for node in model.graph.node:
        if node.op_type == "AveragePool":
            for attribute in node.attribute:
                if attribute.name == "ceil_mode":
                    attribute.i = 0
    save_model(model, target)


def fixed_recognizer(source: Path, target: Path) -> None:
    model = onnx.load(source)
    make_input_shape_fixed(model.graph, "input.1", [1, 3, 112, 112])
    save_model(model, target)


def mobile_swapper(
    source: Path, target: Path, fp16_target: Path, emap_target: Path
) -> None:
    model = onnx.load(source)
    initializer = next(
        (item for item in model.graph.initializer if item.name == "buff2fs"),
        model.graph.initializer[-1],
    )
    emap = np.asarray(numpy_helper.to_array(initializer), dtype="<f4")
    if emap.shape != (512, 512):
        raise RuntimeError(f"unexpected INSwapper embedding map shape {emap.shape}")
    emap.tofile(emap_target)
    model.graph.initializer.remove(initializer)
    if not _decompose_reflect_pad(model):
        raise RuntimeError("INSwapper had no reflection pads to rewrite")
    # `pytorch_half_pixel` differs from standard `half_pixel` only when the
    # resized output has length one. These two fixed 64x64 and 128x128 outputs
    # are therefore bit-for-bit coordinate equivalent, and QNN accepts the
    # standard spelling without leaving its NHWC graph invalid.
    for node in model.graph.node:
        if node.op_type == "Resize":
            for attribute in node.attribute:
                if (
                    attribute.name == "coordinate_transformation_mode"
                    and attribute.s == b"pytorch_half_pixel"
                ):
                    attribute.s = b"half_pixel"
    save_model(model, target)
    fp16 = convert_float_to_float16(model, keep_io_types=True)
    topological_sort(fp16)
    save_model(fp16, fp16_target)


def qnn_probe(target: Path) -> None:
    """Create a tiny graph used to prove that QNN, not CPU, owns execution."""
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"], name="qnn_backend_probe")],
        "deep-live-mobile-qnn-probe",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="deep-live-mobile",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    save_model(model, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapper", type=Path, default=ROOT / "models/inswapper_128.onnx")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path.home() / ".insightface/models/buffalo_l/det_10g.onnx",
    )
    parser.add_argument(
        "--recognizer",
        type=Path,
        default=Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "android/vcam-mobile-app/model-pack",
    )
    args = parser.parse_args()
    for path in (args.swapper, args.detector, args.recognizer):
        if not path.is_file():
            parser.error(f"missing model: {path}")

    output = args.output.resolve()
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    detector = staging / "det_10g_640.onnx"
    recognizer = staging / "w600k_r50_112.onnx"
    swapper = staging / "inswapper_128_mobile.onnx"
    swapper_fp16 = staging / "inswapper_128_mobile_fp16.onnx"
    emap = staging / "inswapper_emap.f32"
    probe = staging / "qnn_probe.onnx"
    fixed_detector(args.detector, detector)
    fixed_recognizer(args.recognizer, recognizer)
    mobile_swapper(args.swapper, swapper, swapper_fp16, emap)
    qnn_probe(probe)

    files = {}
    for path in (detector, recognizer, swapper, swapper_fp16, emap, probe):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = {
        "version": 5,
        "format": "Deep Live Mobile offline model pack",
        "models": files,
        "preprocessing": {
            "detector": "SCRFD 640x640 top-left letterbox; equivalent ceil-mode rewrite",
            "recognizer": "ArcFace 112x112",
            "swapper": "INSwapper 128x128 FP32 fallback plus FP16 QNN graph; equivalent Pad and half-pixel rewrites",
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    print(output)
    for name, metadata in files.items():
        print(f"{name}: {metadata['bytes']} bytes sha256={metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
