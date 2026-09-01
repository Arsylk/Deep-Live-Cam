#!/usr/bin/env python3
"""Convert a local native-256 ONNX bundle into an offline ncnn bundle.

The command never discovers, downloads, or installs a converter.  It requires
an explicit executable path and a hash-valid ``Native256Manifest`` as input.
All output is assembled in a sibling staging directory and published only
after the ONNX and ncnn assets and the final manifest have been validated.

Example::

    python tools/prepare_native256_ncnn.py \
      --manifest /data/swap256/onnx/manifest.json \
      --pnnx /opt/pnnx/pnnx \
      --output-dir /data/swap256/ncnn-bundle \
      --fp16-storage

No existing output directory is replaced.  Exporting or converting a model
does not alter its quality status or make it eligible for automatic selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
NCNN_MAGIC = "7767517"
NCNN_BACKEND = "ncnn-vulkan"
STYLE_CHANNELS = 2016

# Native ncnn layers admitted by the exported DLC-Swap256-M graph.  A small
# number of equivalent built-ins are included because pnnx may choose a fused
# or unfused representation across releases. Reshape is deliberately absent:
# the deployment graphs use explicit NCHW tensors because ambiguous pnnx
# vector reshapes previously corrupted style channels at runtime. Unknown and
# pnnx custom layers are rejected rather than delegated to runtime registration.
ALLOWED_NCNN_LAYERS = frozenset(
    {
        "BinaryOp",
        "Concat",
        "Convolution",
        "ConvolutionDepthWise",
        "InnerProduct",
        "Input",
        "Interp",
        "MemoryData",
        "Padding",
        "Permute",
        "Pooling",
        "ReLU",
        "Sigmoid",
        "Slice",
        "Split",
        "TanH",
        "UnaryOp",
    }
)


class PreparationError(RuntimeError):
    """A local input, conversion, or generated artifact failed validation."""


@dataclass(frozen=True, slots=True)
class NcnnGraph:
    """Structural facts read from one ncnn text parameter file."""

    layer_count: int
    blob_count: int
    layer_types: frozenset[str]
    inputs: frozenset[str]
    outputs: frozenset[str]


@dataclass(frozen=True, slots=True)
class ConvertedGraph:
    param: Path
    weights: Path
    structure: NcnnGraph


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreparationError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _contract_module():
    """Load the runtime contract without importing the OpenCV-heavy package."""

    contract_path = ROOT / "modules" / "swapper_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_dlc_native256_ncnn_contract", contract_path
    )
    if spec is None or spec.loader is None:
        raise PreparationError(f"could not load runtime contract: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise PreparationError(f"could not load runtime contract: {error}") from error
    return module


def load_manifest(path: Path):
    """Load a local manifest and verify every source artifact hash."""

    module = _contract_module()
    try:
        manifest = module.Native256Manifest.load(path, verify_hashes=True)
        module.verify_embedded_onnx(
            manifest.conditioner.asset, "identity conditioner"
        )
        module.verify_embedded_onnx(manifest.swapper.asset, "native-256 swapper")
        return manifest
    except Exception as error:
        raise PreparationError(f"invalid native-256 source bundle: {error}") from error
    finally:
        sys.modules.pop(module.__name__, None)


def _regular_nonempty_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} was not produced: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise PreparationError(f"{label} must be a regular file: {path}")
    if info.st_size <= 0:
        raise PreparationError(f"{label} is empty: {path}")


def validate_ncnn_param(
    path: Path,
    *,
    expected_inputs: set[str] | frozenset[str],
    expected_outputs: set[str] | frozenset[str],
) -> NcnnGraph:
    """Validate ncnn syntax, built-in layer types, and public blob names."""

    _regular_nonempty_file(path, "ncnn parameter file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PreparationError(
            f"could not read ncnn parameter file {path}: {error}"
        ) from error
    if len(lines) < 3 or lines[0].strip() != NCNN_MAGIC:
        raise PreparationError(f"invalid ncnn parameter header: {path}")
    declaration = lines[1].split()
    if len(declaration) != 2:
        raise PreparationError(f"invalid ncnn layer/blob declaration: {path}")
    try:
        declared_layers, declared_blobs = (int(value) for value in declaration)
    except ValueError as error:
        raise PreparationError(f"invalid ncnn layer/blob counts: {path}") from error
    if declared_layers <= 0 or declared_blobs <= 0:
        raise PreparationError(f"ncnn layer/blob counts must be positive: {path}")

    layer_lines = [line for line in lines[2:] if line.strip()]
    if len(layer_lines) != declared_layers:
        raise PreparationError(
            f"ncnn layer count mismatch in {path}: declared {declared_layers}, "
            f"found {len(layer_lines)}"
        )

    layer_names: set[str] = set()
    layer_types: set[str] = set()
    produced: set[str] = set()
    consumed: set[str] = set()
    graph_inputs: set[str] = set()
    for index, line in enumerate(layer_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise PreparationError(f"malformed ncnn layer on line {index}: {path}")
        layer_type, layer_name = fields[0], fields[1]
        if layer_type not in ALLOWED_NCNN_LAYERS:
            raise PreparationError(
                f"unsupported or custom ncnn layer {layer_type!r} on line "
                f"{index}: {path}"
            )
        if layer_name in layer_names:
            raise PreparationError(f"duplicate ncnn layer name {layer_name!r}: {path}")
        layer_names.add(layer_name)
        layer_types.add(layer_type)
        try:
            bottom_count = int(fields[2])
            top_count = int(fields[3])
        except ValueError as error:
            raise PreparationError(
                f"invalid ncnn blob counts on line {index}: {path}"
            ) from error
        if bottom_count < 0 or top_count <= 0:
            raise PreparationError(
                f"invalid ncnn layer arity on line {index}: {path}"
            )
        required_fields = 4 + bottom_count + top_count
        if len(fields) < required_fields:
            raise PreparationError(f"truncated ncnn layer on line {index}: {path}")
        bottoms = fields[4 : 4 + bottom_count]
        tops = fields[4 + bottom_count : required_fields]

        if layer_type == "Input":
            if bottom_count != 0 or top_count != 1:
                raise PreparationError(
                    f"ncnn Input must have zero bottoms and one top: {path}"
                )
            graph_inputs.add(tops[0])
        elif bottom_count == 0 and layer_type != "MemoryData":
            raise PreparationError(
                f"only Input or MemoryData may have no inputs: {path}"
            )
        missing = [blob for blob in bottoms if blob not in produced]
        if missing:
            raise PreparationError(
                f"ncnn layer {layer_name!r} consumes undefined blobs {missing}: {path}"
            )
        duplicates = [blob for blob in tops if blob in produced]
        if duplicates:
            raise PreparationError(
                f"ncnn layer {layer_name!r} redefines blobs {duplicates}: {path}"
            )
        consumed.update(bottoms)
        produced.update(tops)

    if len(produced) != declared_blobs:
        raise PreparationError(
            f"ncnn blob count mismatch in {path}: declared {declared_blobs}, "
            f"found {len(produced)}"
        )
    graph_outputs = produced - consumed
    wanted_inputs = set(expected_inputs)
    wanted_outputs = set(expected_outputs)
    if graph_inputs != wanted_inputs:
        raise PreparationError(
            f"unexpected ncnn inputs in {path}: expected {sorted(wanted_inputs)}, "
            f"got {sorted(graph_inputs)}"
        )
    if graph_outputs != wanted_outputs:
        raise PreparationError(
            f"unexpected ncnn outputs in {path}: expected {sorted(wanted_outputs)}, "
            f"got {sorted(graph_outputs)}"
        )
    return NcnnGraph(
        layer_count=declared_layers,
        blob_count=declared_blobs,
        layer_types=frozenset(layer_types),
        inputs=frozenset(graph_inputs),
        outputs=frozenset(graph_outputs),
    )


def validate_pnnx(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    _regular_nonempty_file(resolved, "pnnx executable")
    if not os.access(resolved, os.X_OK):
        raise PreparationError(f"pnnx is not executable: {resolved}")
    return resolved


def _pnnx_version(pnnx: Path, working_directory: Path) -> str | None:
    """Best-effort informational version; conversion never trusts this value."""

    try:
        completed = subprocess.run(
            [str(pnnx), "--version"],
            check=False,
            capture_output=True,
            cwd=working_directory,
            text=True,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    return lines[0][:256] if lines else None


def convert_graph(
    *,
    pnnx: Path,
    onnx_path: Path,
    graph_name: str,
    input_shape: str,
    expected_inputs: set[str],
    expected_outputs: set[str],
    fp16_storage: bool,
    work_root: Path,
) -> ConvertedGraph:
    """Run one explicit local pnnx process and validate its ncnn outputs."""

    graph_dir = work_root / graph_name
    graph_dir.mkdir(mode=0o700)
    local_onnx = graph_dir / "model.onnx"
    shutil.copyfile(onnx_path, local_onnx)
    param_name = f"{graph_name}.ncnn.param"
    bin_name = f"{graph_name}.ncnn.bin"
    command = [
        str(pnnx),
        local_onnx.name,
        f"inputshape={input_shape}",
        f"fp16={1 if fp16_storage else 0}",
        "optlevel=2",
        f"pnnxparam={graph_name}.pnnx.param",
        f"pnnxbin={graph_name}.pnnx.bin",
        f"pnnxpy={graph_name}_pnnx.py",
        f"pnnxonnx={graph_name}.pnnx.onnx",
        f"ncnnparam={param_name}",
        f"ncnnbin={bin_name}",
        f"ncnnpy={graph_name}_ncnn.py",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=graph_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=30 * 60,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreparationError(f"pnnx failed to run for {graph_name}: {error}") from error
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()[-2000:]
        raise PreparationError(
            f"pnnx conversion failed for {graph_name} with status "
            f"{completed.returncode}: {details}"
        )

    param_path = graph_dir / param_name
    weights_path = graph_dir / bin_name
    _regular_nonempty_file(weights_path, f"{graph_name} ncnn weights")
    structure = validate_ncnn_param(
        param_path,
        expected_inputs=expected_inputs,
        expected_outputs=expected_outputs,
    )
    return ConvertedGraph(param_path, weights_path, structure)


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy into an unpublished staging tree and verify the resulting bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PreparationError(f"refusing to overwrite staged asset: {destination}")
    shutil.copyfile(source, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise PreparationError(
            f"copied asset SHA-256 mismatch for {destination.name}: "
            f"expected {expected_sha256}, got {actual}"
        )


def _manifest_asset_relative(source_root: Path, asset_path: Path) -> Path:
    try:
        relative = asset_path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise PreparationError(
            f"manifest asset escapes its bundle directory: {asset_path}"
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise PreparationError(f"unsafe manifest asset path: {relative}")
    return relative


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_bundle(
    *,
    manifest_path: Path,
    pnnx_path: Path,
    output_dir: Path,
    fp16_storage: bool,
) -> Path:
    """Validate, convert, stage, and atomically publish one ncnn bundle."""

    source_manifest_path = manifest_path.expanduser().resolve()
    source_manifest_hash = sha256_file(source_manifest_path)
    try:
        document = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"could not reread source manifest: {error}") from error
    if sha256_file(source_manifest_path) != source_manifest_hash:
        raise PreparationError("source manifest changed while it was being read")
    source = load_manifest(source_manifest_path)
    if sha256_file(source_manifest_path) != source_manifest_hash:
        raise PreparationError("source manifest changed during contract validation")
    if source.auto_select_eligible:
        raise PreparationError(
            "refusing to convert an auto-select-eligible release: ncnn adds new "
            "deployable artifacts that must be measured and included in a new "
            "qualification report; convert before qualification"
        )
    pnnx = validate_pnnx(pnnx_path)
    pnnx_hash = sha256_file(pnnx)
    if not isinstance(document, dict):
        raise PreparationError("source manifest must be a JSON object")
    if "ncnn" in document:
        raise PreparationError("source manifest already contains an ncnn section")
    original_quality = document.get("quality_status")
    original_eligibility = document.get("auto_select_eligible")

    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PreparationError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    stage_path = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    work_path = Path(tempfile.mkdtemp(prefix="native256-pnnx-"))
    published = False
    try:
        source_root = source_manifest_path.parent
        source_assets = (
            (source.identity_map.path, source.identity_map.sha256),
            (source.conditioner.asset.path, source.conditioner.asset.sha256),
            (source.swapper.asset.path, source.swapper.asset.sha256),
        )
        copied: set[Path] = set()
        staged_assets: dict[Path, Path] = {}
        for asset_path, expected_hash in source_assets:
            relative = _manifest_asset_relative(source_root, asset_path)
            staged_asset = stage_path / relative
            if relative not in copied:
                _copy_verified(asset_path, staged_asset, expected_hash)
                copied.add(relative)
            staged_assets[asset_path] = staged_asset

        conditioner = convert_graph(
            pnnx=pnnx,
            # Convert the already hash-verified staging copy, not a mutable
            # source path that could diverge from the published ONNX bytes.
            onnx_path=staged_assets[source.conditioner.asset.path],
            graph_name="conditioner",
            input_shape="[1,512,1,1]f32",
            expected_inputs={"in0"},
            expected_outputs={"out0"},
            fp16_storage=fp16_storage,
            work_root=work_path,
        )
        generator = convert_graph(
            pnnx=pnnx,
            onnx_path=staged_assets[source.swapper.asset.path],
            graph_name="generator",
            input_shape=f"[1,3,256,256]f32,[1,{STYLE_CHANNELS},1,1]f32",
            expected_inputs={"in0", "in1"},
            expected_outputs={"out0", "out1"},
            fp16_storage=fp16_storage,
            work_root=work_path,
        )

        converted_assets = {
            "conditioner.ncnn.param": conditioner.param,
            "conditioner.ncnn.bin": conditioner.weights,
            "generator.ncnn.param": generator.param,
            "generator.ncnn.bin": generator.weights,
        }
        converted_hashes: dict[str, str] = {}
        for filename, generated in converted_assets.items():
            digest = sha256_file(generated)
            _copy_verified(generated, stage_path / filename, digest)
            converted_hashes[filename] = digest

        document["ncnn"] = {
            "backend": NCNN_BACKEND,
            "fp16_storage": bool(fp16_storage),
            "identity_conditioner": {
                "param_file": "conditioner.ncnn.param",
                "param_sha256": converted_hashes["conditioner.ncnn.param"],
                "bin_file": "conditioner.ncnn.bin",
                "bin_sha256": converted_hashes["conditioner.ncnn.bin"],
                "input": "in0",
                "output": "out0",
            },
            "swapper": {
                "param_file": "generator.ncnn.param",
                "param_sha256": converted_hashes["generator.ncnn.param"],
                "bin_file": "generator.ncnn.bin",
                "bin_sha256": converted_hashes["generator.ncnn.bin"],
                "target_input": "in0",
                "style_input": "in1",
                "candidate_output": "out0",
                "alpha_output": "out1",
            },
        }
        if document.get("quality_status") != original_quality:
            raise PreparationError("conversion changed quality_status")
        if document.get("auto_select_eligible") != original_eligibility:
            raise PreparationError("conversion changed auto_select_eligible")

        final_manifest = stage_path / "manifest.json"
        _write_json(final_manifest, document)
        pnnx_version = _pnnx_version(pnnx, work_path)
        if sha256_file(pnnx) != pnnx_hash:
            raise PreparationError("pnnx executable changed during conversion")
        audit = {
            "schema_version": 1,
            "operation": "offline-native256-onnx-to-ncnn",
            "source_manifest_sha256": source_manifest_hash,
            "pnnx": {
                "sha256": pnnx_hash,
                "version": pnnx_version,
            },
            "fp16_storage": bool(fp16_storage),
            "graphs": {
                "identity_conditioner": {
                    "layers": conditioner.structure.layer_count,
                    "blobs": conditioner.structure.blob_count,
                    "layer_types": sorted(conditioner.structure.layer_types),
                },
                "swapper": {
                    "layers": generator.structure.layer_count,
                    "blobs": generator.structure.blob_count,
                    "layer_types": sorted(generator.structure.layer_types),
                },
            },
        }
        _write_json(stage_path / "ncnn-audit.json", audit)

        # This validates the original ONNX pins and, when the runtime contract
        # supports it, every generated ncnn pin as well.  The structural ncnn
        # validation above remains mandatory and independent of that support.
        final = load_manifest(final_manifest)
        if getattr(final, "ncnn", None) is None:
            raise PreparationError("final runtime contract did not admit the ncnn section")
        if final.quality_status != source.quality_status:
            raise PreparationError("final contract changed quality_status")
        if final.auto_select_eligible != source.auto_select_eligible:
            raise PreparationError("final contract changed auto-select eligibility")

        # A same-filesystem directory rename publishes all files together.
        if destination.exists() or destination.is_symlink():
            raise PreparationError(
                f"output directory appeared during conversion: {destination}"
            )
        stage_path.rename(destination)
        published = True
        return destination / "manifest.json"
    finally:
        shutil.rmtree(work_path, ignore_errors=True)
        if not published:
            shutil.rmtree(stage_path, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="local hash-pinned native-256 ONNX manifest",
    )
    parser.add_argument(
        "--pnnx",
        required=True,
        type=Path,
        help="explicit path to a local pnnx executable (never downloaded)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new bundle directory; an existing path is never overwritten",
    )
    parser.add_argument(
        "--fp16-storage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store converted ncnn weights as fp16 (default: fp32)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_bundle(
            manifest_path=args.manifest,
            pnnx_path=args.pnnx,
            output_dir=args.output_dir,
            fp16_storage=args.fp16_storage,
        )
    except PreparationError as error:
        raise SystemExit(f"native-256 ncnn preparation failed: {error}") from error
    print(f"prepared offline native-256 ncnn bundle: {result.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
