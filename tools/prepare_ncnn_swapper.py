#!/usr/bin/env python3
"""Convert INSwapper to the ncnn model used by the Linux Vulkan backend.

The converter's large intermediate files live in a temporary directory. Only
the runtime param/bin files, the identity projection, and a manifest are kept.
No network access is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "models" / "inswapper_128.onnx"
DEFAULT_OUTPUT = ROOT / "models" / "ncnn"
RESHAPE_4D = "0=1 1=1 11=2048 12=0 13=0 2=1"
RESHAPE_3D = "0=1 1=1 2=2048"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pnnx",
        type=Path,
        default=shutil.which("pnnx"),
        help="path to the pnnx executable",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_source(model: onnx.ModelProto) -> np.ndarray:
    inputs = {value.name: value for value in model.graph.input}
    if set(inputs) != {"target", "source"}:
        raise RuntimeError(f"unexpected INSwapper inputs: {sorted(inputs)}")
    outputs = {value.name for value in model.graph.output}
    if outputs != {"output"}:
        raise RuntimeError(f"unexpected INSwapper outputs: {sorted(outputs)}")
    initializers = {item.name: item for item in model.graph.initializer}
    if "buff2fs" not in initializers:
        raise RuntimeError("INSwapper is missing the buff2fs identity projection")
    emap = np.ascontiguousarray(
        numpy_helper.to_array(initializers["buff2fs"]), dtype=np.float32
    )
    if emap.shape != (512, 512):
        raise RuntimeError(f"unexpected buff2fs shape: {emap.shape}")
    return emap


def main() -> int:
    args = parse_args()
    if args.pnnx is None or not args.pnnx.is_file():
        raise SystemExit("pnnx was not found; pass its local path with --pnnx")
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"source model does not exist: {source}")

    model = onnx.load(source, load_external_data=False)
    emap = validate_source(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="deep-live-cam-ncnn-") as temporary:
        work = Path(temporary)
        (work / "inswapper_128.onnx").symlink_to(source)
        command = [
            str(args.pnnx.resolve()),
            "inswapper_128.onnx",
            "inputshape=[1,3,128,128]f32,[1,512]f32",
            "fp16=0",
            "optlevel=2",
            "pnnxparam=swapper.pnnx.param",
            "pnnxbin=swapper.pnnx.bin",
            "pnnxpy=swapper_pnnx.py",
            "pnnxonnx=swapper.pnnx.onnx",
            "ncnnparam=swapper.ncnn.param",
            "ncnnbin=swapper.ncnn.bin",
            "ncnnpy=swapper_ncnn.py",
        ]
        subprocess.run(command, cwd=work, check=True)

        generated_param = work / "swapper.ncnn.param"
        generated_bin = work / "swapper.ncnn.bin"
        if not generated_param.is_file() or not generated_bin.is_file():
            raise RuntimeError("pnnx did not produce the expected ncnn model")

        param_text = generated_param.read_text(encoding="utf-8")
        reshape_count = param_text.count(RESHAPE_4D)
        if reshape_count != 12:
            raise RuntimeError(
                "expected 12 pnnx style-vector reshapes, found "
                f"{reshape_count}; refusing an unverified graph"
            )
        param_text = param_text.replace(RESHAPE_4D, RESHAPE_3D)

        output_param = args.output_dir / "inswapper_128.ncnn.param"
        output_bin = args.output_dir / "inswapper_128.ncnn.bin"
        output_param.write_text(param_text, encoding="utf-8")
        shutil.copyfile(generated_bin, output_bin)

    emap_path = args.output_dir / "inswapper_128_emap.npy"
    np.save(emap_path, emap, allow_pickle=False)
    manifest = {
        "backend": "ncnn-vulkan-fp32",
        "source": str(source),
        "source_sha256": sha256(source),
        "param_sha256": sha256(output_param),
        "model_sha256": sha256(output_bin),
        "reshape_fix": "12 style vectors: [1,2048,1,1] -> [2048,1,1]",
        "fp16_storage": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared offline ncnn swapper in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
