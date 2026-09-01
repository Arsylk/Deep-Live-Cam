#!/usr/bin/env python3
"""Create one deterministic, offline INSwapper teacher candidate/alpha pair.

The input target must already be the exact aligned 256px RGB crop referenced by
the native256 training manifest. The source latent is the mapped, L2-normalized
512-value vector consumed by INSwapper. No model or data is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
from PIL import Image


TEACHER_TYPE = "uncomposited-candidate-alpha-rgb-128-v1"
_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class TeacherPreparationError(RuntimeError):
    """A local input or teacher output violated the training contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb256(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (256, 256):
            raise TeacherPreparationError(
                f"target must be an aligned 256x256 RGB image: {path}"
            )
        value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(value)


def _mask256(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (256, 256):
            raise TeacherPreparationError(
                f"face mask must be a 256x256 grayscale image: {path}"
            )
        value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(value)


def _latent512(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.size != 512:
        raise TeacherPreparationError(
            f"source latent must be float32 with 512 values: {path}"
        )
    value = np.ascontiguousarray(value.reshape(1, 512))
    if not np.isfinite(value).all():
        raise TeacherPreparationError("source latent contains a non-finite value")
    norm = float(np.linalg.norm(value))
    if not 0.999 <= norm <= 1.001:
        raise TeacherPreparationError(
            f"source latent must already be L2-normalized, got norm {norm:.8f}"
        )
    return value


def _area_half(value: np.ndarray) -> np.ndarray:
    if value.ndim == 3:
        return value.reshape(128, 2, 128, 2, value.shape[2]).mean(axis=(1, 3))
    return value.reshape(128, 2, 128, 2).mean(axis=(1, 3))


def _node_contract(session: Any) -> None:
    inputs = {node.name: node for node in session.get_inputs()}
    outputs = {node.name: node for node in session.get_outputs()}
    if set(inputs) != {"target", "source"} or set(outputs) != {"output"}:
        raise TeacherPreparationError(
            "teacher ONNX must expose target, source -> output"
        )
    expected = {
        "target": ([1, 3, 128, 128], "tensor(float)"),
        "source": ([1, 512], "tensor(float)"),
        "output": ([1, 3, 128, 128], "tensor(float)"),
    }
    for name, node in {**inputs, **outputs}.items():
        shape, tensor_type = expected[name]
        if list(node.shape) != shape or node.type != tensor_type:
            raise TeacherPreparationError(
                f"teacher tensor {name!r} must be {tensor_type} {shape}, "
                f"got {node.type} {node.shape}"
            )


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def prepare_teacher(
    *,
    model_path: Path,
    expected_model_sha256: str,
    target_path: Path,
    source_latent_path: Path,
    face_mask_path: Path,
    output_dir: Path,
    sample_id: str,
    session_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not _SAMPLE_ID.fullmatch(sample_id):
        raise TeacherPreparationError("sample_id contains unsupported characters")
    for label, path in (
        ("teacher model", model_path),
        ("target", target_path),
        ("source latent", source_latent_path),
        ("face mask", face_mask_path),
    ):
        if not path.is_file():
            raise TeacherPreparationError(f"{label} does not exist: {path}")
    model_hash = sha256_file(model_path)
    if model_hash != expected_model_sha256.lower():
        raise TeacherPreparationError(
            "teacher model SHA-256 mismatch: "
            f"expected {expected_model_sha256.lower()}, got {model_hash}"
        )

    target = _rgb256(target_path)
    alpha = _area_half(_mask256(face_mask_path)).astype(np.float32)[None]
    latent = _latent512(source_latent_path)
    target_128 = _area_half(target).astype(np.float32)
    target_nchw = np.ascontiguousarray(target_128.transpose(2, 0, 1)[None])

    if session_factory is None:
        try:
            import onnxruntime
        except ImportError as error:
            raise TeacherPreparationError(
                "teacher preparation requires the local onnxruntime package"
            ) from error
        session_factory = onnxruntime.InferenceSession
    try:
        session = session_factory(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        _node_contract(session)
        outputs = session.run(["output"], {"target": target_nchw, "source": latent})
    except TeacherPreparationError:
        raise
    except Exception as error:
        raise TeacherPreparationError(f"teacher inference failed: {error}") from error
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise TeacherPreparationError("teacher inference returned an invalid output list")
    candidate = outputs[0]
    if not isinstance(candidate, np.ndarray) or candidate.dtype != np.float32:
        raise TeacherPreparationError("teacher output must be a float32 NumPy tensor")
    if candidate.shape != (1, 3, 128, 128) or not np.isfinite(candidate).all():
        raise TeacherPreparationError(
            f"teacher output must be finite with shape (1,3,128,128), got {candidate.shape}"
        )
    minimum = float(candidate.min())
    maximum = float(candidate.max())
    if minimum < -1e-4 or maximum > 1.0001:
        raise TeacherPreparationError(
            f"teacher output is outside [0,1]: [{minimum}, {maximum}]"
        )
    candidate = np.ascontiguousarray(np.clip(candidate[0], 0.0, 1.0))
    alpha = np.ascontiguousarray(np.clip(alpha, 0.0, 1.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / f"{sample_id}-candidate.npy"
    alpha_path = output_dir / f"{sample_id}-alpha.npy"
    metadata_path = output_dir / f"{sample_id}-teacher.json"
    for path in (candidate_path, alpha_path, metadata_path):
        if path.exists():
            raise TeacherPreparationError(f"refusing to overwrite existing output: {path}")
    _atomic_npy(candidate_path, candidate)
    _atomic_npy(alpha_path, alpha)
    metadata = {
        "schema_version": 1,
        "teacher_type": TEACHER_TYPE,
        "sample_id": sample_id,
        "model": {"path": str(model_path.resolve()), "sha256": model_hash},
        "inputs": {
            "target_sha256": sha256_file(target_path),
            "source_latent_sha256": sha256_file(source_latent_path),
            "face_mask_sha256": sha256_file(face_mask_path),
        },
        "outputs": {
            "teacher_candidate": candidate_path.name,
            "teacher_candidate_sha256": sha256_file(candidate_path),
            "teacher_alpha": alpha_path.name,
            "teacher_alpha_sha256": sha256_file(alpha_path),
        },
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-latent", type=Path, required=True)
    parser.add_argument("--face-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = prepare_teacher(
        model_path=args.model,
        expected_model_sha256=args.model_sha256,
        target_path=args.target,
        source_latent_path=args.source_latent,
        face_mask_path=args.face_mask,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
