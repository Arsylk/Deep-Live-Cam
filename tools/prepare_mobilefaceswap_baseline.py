#!/usr/bin/env python3
"""Prepare an offline, source-conditioned MobileFaceSwap experiment.

This script deliberately does not download, install, register, or select a
runtime backend.  It consumes an audited local upstream checkout and the
official local checkpoint archive, then writes a source-specific cache entry.
Heavy optional dependencies are imported only after all local inputs pass
validation so the integrity helpers remain unit-testable without Paddle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import BinaryIO


OFFICIAL_ARCHIVE_SHA256 = (
    "8e45272de4cc55d64294d282bb7605343a406696a05209203065aa1b51aed58e"
)
OFFICIAL_ARCHIVE_SIZE = 569_453_568
OFFICIAL_UPSTREAM_COMMIT = "29552474c2f621c27818ff7fc32447e1a60a96fb"
PUBLIC_TRAINING_SIZE = 224
SUPPORTED_SIZES = (224, 256)

REQUIRED_ARCHIVE_MEMBERS = {
    "checkpoints/MobileFaceSwap_224.pdparams": 82_834_948,
    "checkpoints/arcface.pdparams": 209_017_341,
    "checkpoints/landmarks/scrfd_10g_bnkps.onnx": 16_923_827,
}

AUDITED_UPSTREAM_FILES = {
    "models/model.py": (
        "946d80b501c45eee3b667af135fa3175ad2e799fc22a985bc3066d0b9b7d64c6"
    ),
    "models/arcface.py": (
        "248edb4e03ea161d049d10514bf9b56f1a9f75bcc17324d40f90a50a4c1b7c39"
    ),
    "utils/align_face.py": (
        "7e620521326c1f6810f4e26d8914408ecc83180cc75741195c60c0caddde6c0e"
    ),
    "LICENSE": (
        "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
    ),
}

LICENSE_WARNING = (
    "The MobileFaceSwap repository source is Apache-2.0, but the separately "
    "hosted checkpoint archive contains no explicit model-weight license. It "
    "also bundles InsightFace-derived assets. Do not assume that commercial "
    "redistribution rights were granted for these weights."
)
QUALITY_WARNING = (
    "This is an experimental comparison baseline, not a production-quality "
    "or default backend. Validate identity, eyes, occlusions, profiles, masks, "
    "and temporal stability on authorized holdout material before use."
)


class PreparationError(RuntimeError):
    """A local input or generated artifact failed a safety check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_size_warning(size: int) -> str:
    if size == PUBLIC_TRAINING_SIZE:
        return (
            "The only public checkpoint is trained and released for 224x224; "
            "this cache entry uses that public training size."
        )
    return (
        "No public native-256 MobileFaceSwap checkpoint exists. This cache "
        "entry evaluates the public 224-trained fully convolutional graph at "
        "256x256 and may introduce artifacts; it is not a native-256 model."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-tar",
        required=True,
        type=Path,
        help="local official checkpoints.tar (no download is attempted)",
    )
    parser.add_argument(
        "--upstream-dir",
        required=True,
        type=Path,
        help="local official MobileFaceSwap checkout at the audited revision",
    )
    parser.add_argument(
        "--source-image",
        required=True,
        type=Path,
        help="local source-identity image to condition into the model",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="cache root; a source-hash-specific directory is created below it",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=SUPPORTED_SIZES,
        default=PUBLIC_TRAINING_SIZE,
        help="inference crop size; 256 still uses the public 224-trained weights",
    )
    parser.add_argument(
        "--pnnx",
        type=Path,
        help="optional explicit local pnnx executable; omit for ONNX-only output",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_cli_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("checkpoint archive", args.checkpoint_tar),
        ("source image", args.source_image),
    ):
        if not path.is_file():
            raise PreparationError(f"{label} does not exist: {path}")
    if not args.upstream_dir.is_dir():
        raise PreparationError(
            f"upstream checkout does not exist: {args.upstream_dir}"
        )
    for relative_path in AUDITED_UPSTREAM_FILES:
        candidate = args.upstream_dir / relative_path
        if not candidate.is_file():
            raise PreparationError(
                f"upstream checkout is missing {relative_path}: {candidate}"
            )
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise PreparationError(
            f"output cache path exists but is not a directory: {args.output_dir}"
        )
    if args.size not in SUPPORTED_SIZES:
        raise PreparationError(
            f"unsupported size {args.size}; expected one of {SUPPORTED_SIZES}"
        )
    if args.pnnx is not None:
        if not args.pnnx.is_file():
            raise PreparationError(f"pnnx does not exist: {args.pnnx}")
        if not os.access(args.pnnx, os.X_OK):
            raise PreparationError(f"pnnx is not executable: {args.pnnx}")


def verify_official_archive(path: Path) -> str:
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_hash != OFFICIAL_ARCHIVE_SHA256:
        raise PreparationError(
            "checkpoint archive SHA-256 mismatch: "
            f"expected {OFFICIAL_ARCHIVE_SHA256}, got {actual_hash}"
        )
    if actual_size != OFFICIAL_ARCHIVE_SIZE:
        raise PreparationError(
            "checkpoint archive size mismatch despite matching digest: "
            f"expected {OFFICIAL_ARCHIVE_SIZE}, got {actual_size}"
        )
    return actual_hash


def verify_upstream_checkout(path: Path) -> dict[str, object]:
    hashes: dict[str, str] = {}
    for relative_path, expected_hash in AUDITED_UPSTREAM_FILES.items():
        actual_hash = sha256_file(path / relative_path)
        if actual_hash != expected_hash:
            raise PreparationError(
                f"upstream file is not the audited version: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}"
            )
        hashes[relative_path] = actual_hash

    commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0:
        commit = completed.stdout.strip()
        if commit != OFFICIAL_UPSTREAM_COMMIT:
            raise PreparationError(
                "upstream checkout is not at the audited revision: "
                f"expected {OFFICIAL_UPSTREAM_COMMIT}, got {commit}"
            )
    return {"commit": commit or OFFICIAL_UPSTREAM_COMMIT, "file_sha256": hashes}


def _validate_tar_member(member: tarfile.TarInfo) -> PurePosixPath:
    if "\\" in member.name:
        raise PreparationError(f"archive member contains a backslash: {member.name}")
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise PreparationError(f"unsafe archive member path: {member.name}")
    if member.issym() or member.islnk():
        raise PreparationError(f"archive links are not accepted: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise PreparationError(
            f"unsupported archive member type: {member.name}"
        )
    return member_path


def _copy_exact(stream: BinaryIO, destination: Path, expected_size: int) -> None:
    written = 0
    with destination.open("xb") as output:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > expected_size:
                raise PreparationError(
                    f"archive member exceeds declared size: {destination.name}"
                )
            output.write(block)
    if written != expected_size:
        raise PreparationError(
            f"archive member was truncated: {destination.name}; "
            f"expected {expected_size}, got {written}"
        )


def safe_extract_members(
    archive_path: Path,
    destination: Path,
    required_members: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Validate every tar member and extract only the requested regular files."""

    required = required_members or REQUIRED_ARCHIVE_MEMBERS
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(member)

        for member in members:
            if member.name not in required:
                continue
            if member.name in extracted:
                raise PreparationError(
                    f"duplicate required archive member: {member.name}"
                )
            expected_size = required[member.name]
            if not member.isfile() or member.size != expected_size:
                raise PreparationError(
                    f"unexpected size or type for {member.name}: "
                    f"expected {expected_size} regular-file bytes, got {member.size}"
                )
            relative = PurePosixPath(member.name)
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise PreparationError(
                    f"could not read archive member: {member.name}"
                )
            with stream:
                _copy_exact(stream, output, expected_size)
            extracted[member.name] = output

    missing = sorted(set(required) - set(extracted))
    if missing:
        raise PreparationError(
            f"checkpoint archive is missing required members: {missing}"
        )
    return extracted


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"could not load audited upstream module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_runtime_dependencies():
    try:
        import cv2
        import numpy as np
        import onnx
        import paddle
        import paddle2onnx
        from insightface.model_zoo import model_zoo
    except ImportError as error:
        raise PreparationError(
            "preparation requires local installations of Paddle 2.6.x, "
            "Paddle2ONNX 1.3.x, ONNX, OpenCV, NumPy, and InsightFace; no "
            f"package installation was attempted ({error})"
        ) from error
    if not paddle.__version__.startswith("2.6."):
        raise PreparationError(
            "the audited exporter requires Paddle 2.6.x; found "
            f"{paddle.__version__}"
        )
    if not paddle2onnx.__version__.startswith("1.3."):
        raise PreparationError(
            "the audited exporter requires Paddle2ONNX 1.3.x; found "
            f"{paddle2onnx.__version__}"
        )
    return cv2, np, onnx, paddle, paddle2onnx, model_zoo


def _align_source(
    source_path: Path,
    detector_path: Path,
    align_module: ModuleType,
    cv2,
    np,
    model_zoo,
):
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise PreparationError(f"source image is not readable RGB/BGR data: {source_path}")

    try:
        detector = model_zoo.get_model(
            str(detector_path), providers=["CPUExecutionProvider"]
        )
    except TypeError:
        detector = model_zoo.get_model(str(detector_path))
    if detector is None:
        raise PreparationError("InsightFace could not load the local SCRFD detector")
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.6)
    try:
        # InsightFace 0.2.x accepted an explicit threshold here.
        bboxes, keypoints = detector.detect(
            image, threshold=0.6, max_num=0, metric="default"
        )
    except TypeError as error:
        if "threshold" not in str(error):
            raise
        # InsightFace 0.7.x stores the threshold during prepare().
        bboxes, keypoints = detector.detect(image, max_num=0, metric="default")
    if bboxes is None or len(bboxes) == 0 or keypoints is None:
        raise PreparationError("InsightFace found no alignable face in the source image")
    best_index = int(np.argmax(bboxes[:, 4]))
    aligned, _ = align_module.align_img(
        image, keypoints[best_index], PUBLIC_TRAINING_SIZE
    )
    expected_shape = (PUBLIC_TRAINING_SIZE, PUBLIC_TRAINING_SIZE, 3)
    if tuple(aligned.shape) != expected_shape:
        raise PreparationError(
            f"unexpected aligned source shape {aligned.shape}; expected {expected_shape}"
        )
    return aligned


def _candidate_alpha_wrapper(paddle, swap_model):
    class CandidateAlpha(paddle.nn.Layer):
        def __init__(self, conditioned_model):
            super().__init__()
            self.conditioned_model = conditioned_model

        def forward(self, target_rgb):
            model = self.conditioned_model
            value = (target_rgb - 0.5) / 0.5
            encoder_values = []
            for index in range(len(model.Encoder)):
                value = model.relu(model.Encoder[index](value))
                encoder_values.append(value)

            alpha = value.detach()
            for index in range(len(model.mask)):
                alpha = model.mask[index](alpha)

            candidate = encoder_values[-1]
            for index in range(len(model.Decoder)):
                candidate = model.up(candidate)
                candidate = model.relu(model.Decoder[index](candidate))
                if index != len(model.Decoder) - 1:
                    skip = encoder_values[len(model.Decoder) - 1 - index]
                    candidate = paddle.concat((candidate, skip), axis=1)
            candidate = (1.0 + model.final(candidate)) / 2.0
            return candidate, alpha

    return CandidateAlpha(swap_model)


def _condition_model(
    upstream_dir: Path,
    checkpoint_paths: dict[str, Path],
    aligned_source,
    cv2,
    np,
    paddle,
):
    model_module = _load_module(
        upstream_dir / "models/model.py", "audited_mobilefaceswap_model"
    )
    arcface_module = _load_module(
        upstream_dir / "models/arcface.py", "audited_mobilefaceswap_arcface"
    )
    paddle.set_device("cpu")

    identity_model = arcface_module.ResNet(
        block=arcface_module.IRBlock, layers=[3, 4, 23, 3]
    )
    identity_model.set_dict(
        paddle.load(str(checkpoint_paths["checkpoints/arcface.pdparams"]))
    )
    identity_model.eval()

    source_112 = cv2.resize(aligned_source, (112, 112))
    source_rgb = cv2.cvtColor(source_112, cv2.COLOR_BGR2RGB)
    source_array = np.ascontiguousarray(
        np.transpose(source_rgb, (2, 0, 1))[None], dtype=np.float32
    )
    source_tensor = paddle.to_tensor(source_array / 255.0, dtype="float32")
    mean = paddle.to_tensor([0.485, 0.456, 0.406], dtype="float32").reshape(
        (1, 3, 1, 1)
    )
    standard_deviation = paddle.to_tensor(
        [0.229, 0.224, 0.225], dtype="float32"
    ).reshape((1, 3, 1, 1))
    source_tensor = (source_tensor - mean) / standard_deviation

    with paddle.no_grad():
        identity_embedding, identity_features = identity_model(source_tensor)
        identity_embedding = model_module.l2_norm(identity_embedding)
        conditioned = model_module.FaceSwap(use_gpu=False)
        conditioned.set_model_param(
            identity_embedding,
            identity_features,
            model_weight=paddle.load(
                str(
                    checkpoint_paths[
                        "checkpoints/MobileFaceSwap_224.pdparams"
                    ]
                )
            ),
        )
        conditioned.swap_model.eval()
    wrapper = _candidate_alpha_wrapper(paddle, conditioned.swap_model)
    wrapper.eval()
    return wrapper


def _rename_onnx_value(graph, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    for collection in (graph.input, graph.output, graph.value_info):
        for value in collection:
            if value.name == old_name:
                value.name = new_name
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old_name:
                node.input[index] = new_name
        for index, value in enumerate(node.output):
            if value == old_name:
                node.output[index] = new_name


def _export_onnx(wrapper, size: int, work_dir: Path, paddle, paddle2onnx, onnx):
    prefix = work_dir / "conditioned"
    static_model = paddle.jit.to_static(
        wrapper,
        input_spec=[
            paddle.static.InputSpec(
                shape=[1, 3, size, size], dtype="float32", name="target_rgb"
            )
        ],
        full_graph=True,
    )
    paddle.jit.save(static_model, str(prefix))
    model_path = prefix.with_suffix(".pdmodel")
    params_path = prefix.with_suffix(".pdiparams")
    if not model_path.is_file() or not params_path.is_file():
        raise PreparationError("Paddle did not write the expected static model")

    onnx_path = work_dir / "mobilefaceswap.onnx"
    paddle2onnx.export(
        str(model_path),
        str(params_path),
        save_file=str(onnx_path),
        opset_version=13,
        verbose=False,
        enable_onnx_checker=True,
        enable_optimize=True,
    )
    if not onnx_path.is_file():
        raise PreparationError("Paddle2ONNX did not write an ONNX model")

    graph_model = onnx.load(str(onnx_path))
    if len(graph_model.graph.input) != 1 or len(graph_model.graph.output) != 2:
        raise PreparationError(
            "unexpected exported contract: expected one input and two outputs"
        )
    _rename_onnx_value(
        graph_model.graph, graph_model.graph.input[0].name, "target_rgb"
    )
    _rename_onnx_value(
        graph_model.graph, graph_model.graph.output[0].name, "candidate_rgb"
    )
    _rename_onnx_value(graph_model.graph, graph_model.graph.output[1].name, "alpha")
    onnx.checker.check_model(graph_model)
    onnx.save(graph_model, str(onnx_path))
    return onnx_path


def _run_pnnx(pnnx: Path, onnx_path: Path, size: int, work_dir: Path):
    command = [
        str(pnnx.resolve()),
        onnx_path.name,
        f"inputshape=[1,3,{size},{size}]f32",
        "fp16=0",
        "optlevel=2",
        "pnnxparam=mobilefaceswap.pnnx.param",
        "pnnxbin=mobilefaceswap.pnnx.bin",
        "pnnxpy=mobilefaceswap_pnnx.py",
        "pnnxonnx=mobilefaceswap.pnnx.onnx",
        "ncnnparam=mobilefaceswap.ncnn.param",
        "ncnnbin=mobilefaceswap.ncnn.bin",
        "ncnnpy=mobilefaceswap_ncnn.py",
    ]
    subprocess.run(command, cwd=work_dir, check=True)
    param = work_dir / "mobilefaceswap.ncnn.param"
    weights = work_dir / "mobilefaceswap.ncnn.bin"
    if not param.is_file() or not weights.is_file():
        raise PreparationError("pnnx did not write the expected ncnn model")
    param_text = param.read_text(encoding="utf-8")
    if " out0" not in param_text or " out1" not in param_text:
        raise PreparationError(
            "ncnn graph does not expose both candidate RGB and alpha outputs"
        )
    return param, weights


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def build_manifest(
    *,
    source_path: Path,
    archive_path: Path,
    archive_sha256: str,
    upstream_dir: Path,
    upstream_info: dict[str, object],
    size: int,
    onnx_path: Path,
    ncnn_param_path: Path | None = None,
    ncnn_model_path: Path | None = None,
    pnnx_path: Path | None = None,
) -> dict[str, object]:
    model_hashes = {"onnx": sha256_file(onnx_path)}
    artifacts: dict[str, object] = {
        "onnx": {
            "file": onnx_path.name,
            "sha256": model_hashes["onnx"],
            "bytes": onnx_path.stat().st_size,
        }
    }
    if ncnn_param_path is not None and ncnn_model_path is not None:
        model_hashes["ncnn_param"] = sha256_file(ncnn_param_path)
        model_hashes["ncnn_model"] = sha256_file(ncnn_model_path)
        artifacts["ncnn"] = {
            "param_file": ncnn_param_path.name,
            "param_sha256": model_hashes["ncnn_param"],
            "model_file": ncnn_model_path.name,
            "model_sha256": model_hashes["ncnn_model"],
            "fp16_weights": False,
            "pnnx_sha256": sha256_file(pnnx_path) if pnnx_path else None,
        }

    return {
        "schema_version": 1,
        "backend": "mobilefaceswap-source-conditioned-baseline",
        "quality_status": "experimental",
        "default_backend": False,
        "auto_select": False,
        "offline_preparation": True,
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "checkpoint_archive": {
            "file": archive_path.name,
            "sha256": archive_sha256,
            "verified_official_sha256": archive_sha256 == OFFICIAL_ARCHIVE_SHA256,
        },
        "upstream": {
            "directory_name": upstream_dir.name,
            "audited_commit": upstream_info.get("commit"),
            "file_sha256": upstream_info.get("file_sha256", {}),
        },
        "public_training_size": PUBLIC_TRAINING_SIZE,
        "inference_size": size,
        "training_size_warning": training_size_warning(size),
        "license_warning": LICENSE_WARNING,
        "quality_warning": QUALITY_WARNING,
        "contract": {
            "input": {
                "name": "target_rgb",
                "shape": [1, 3, size, size],
                "dtype": "float32",
                "color": "RGB",
                "range": [0.0, 1.0],
            },
            "outputs": [
                {
                    "name": "candidate_rgb",
                    "shape": [1, 3, size, size],
                    "dtype": "float32",
                    "range": [0.0, 1.0],
                    "uncomposited": True,
                },
                {
                    "name": "alpha",
                    "shape": [1, 1, size, size],
                    "dtype": "float32",
                    "range": [0.0, 1.0],
                },
            ],
            "caller_composite": (
                "candidate_rgb * alpha + target_rgb * (1.0 - alpha)"
            ),
        },
        "model_hashes": model_hashes,
        "artifacts": artifacts,
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> Path:
    validate_cli_paths(args)
    archive_hash = verify_official_archive(args.checkpoint_tar)
    upstream_info = verify_upstream_checkout(args.upstream_dir)
    source_hash = sha256_file(args.source_image)
    entry_dir = args.output_dir / f"{source_hash[:16]}-{args.size}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entry_dir.mkdir(parents=True, exist_ok=True)

    cv2, np, onnx, paddle, paddle2onnx, model_zoo = (
        _require_runtime_dependencies()
    )
    align_module = _load_module(
        args.upstream_dir / "utils/align_face.py",
        "audited_mobilefaceswap_alignment",
    )

    with tempfile.TemporaryDirectory(
        prefix=".prepare-mobilefaceswap-", dir=args.output_dir
    ) as temporary:
        work_dir = Path(temporary)
        checkpoint_paths = safe_extract_members(
            args.checkpoint_tar, work_dir / "official-assets"
        )
        aligned_source = _align_source(
            args.source_image,
            checkpoint_paths[
                "checkpoints/landmarks/scrfd_10g_bnkps.onnx"
            ],
            align_module,
            cv2,
            np,
            model_zoo,
        )
        wrapper = _condition_model(
            args.upstream_dir,
            checkpoint_paths,
            aligned_source,
            cv2,
            np,
            paddle,
        )
        generated_onnx = _export_onnx(
            wrapper, args.size, work_dir, paddle, paddle2onnx, onnx
        )

        base_name = f"mobilefaceswap_source_{args.size}"
        output_onnx = entry_dir / f"{base_name}.onnx"
        _atomic_copy(generated_onnx, output_onnx)

        output_param: Path | None = None
        output_model: Path | None = None
        if args.pnnx is not None:
            generated_param, generated_model = _run_pnnx(
                args.pnnx, generated_onnx, args.size, work_dir
            )
            output_param = entry_dir / f"{base_name}.ncnn.param"
            output_model = entry_dir / f"{base_name}.ncnn.bin"
            _atomic_copy(generated_param, output_param)
            _atomic_copy(generated_model, output_model)

    manifest = build_manifest(
        source_path=args.source_image,
        archive_path=args.checkpoint_tar,
        archive_sha256=archive_hash,
        upstream_dir=args.upstream_dir,
        upstream_info=upstream_info,
        size=args.size,
        onnx_path=output_onnx,
        ncnn_param_path=output_param,
        ncnn_model_path=output_model,
        pnnx_path=args.pnnx,
    )
    manifest_path = entry_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    try:
        manifest_path = prepare(parse_args(argv))
    except (PreparationError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"prepared experimental offline baseline: {manifest_path}")
    print("not installed, registered, auto-selected, or made the default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
