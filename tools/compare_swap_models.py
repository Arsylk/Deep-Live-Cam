#!/usr/bin/env python3
"""Create a paired, aligned corpus from multiple InsightFace swap models.

The same target detection, source face, full-resolution frame and alignment
transform are used for every model.  This keeps model-resolution comparisons
separate from live transport, tracking and post-processing differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# The deployed Windows runtime registers its bundled CUDA/cuDNN DLL folders
# in run.py. Do that before importing ONNX Runtime when a root is supplied.
bootstrap_root = os.environ.get("DLC_BOOTSTRAP_ROOT")
if bootstrap_root:
    sys.path.insert(0, bootstrap_root)
    import run as _windows_runtime_bootstrap  # noqa: F401

import cv2
import insightface
import numpy as np
import onnxruntime as ort
from insightface.model_zoo.inswapper import INSwapper
from skimage.transform import SimilarityTransform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest(faces):
    return max(
        faces,
        key=lambda face: float(
            max(0.0, face.bbox[2] - face.bbox[0])
            * max(0.0, face.bbox[3] - face.bbox[1])
        ),
        default=None,
    )


def _deepfakebench_crop(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    size = 256
    destination = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    destination *= size / 112.0
    margin = size * 0.3 / 2.0
    destination += margin
    destination *= size / (size + 2.0 * margin)
    transform = SimilarityTransform()
    if not transform.estimate(landmarks.astype(np.float32), destination):
        raise RuntimeError("could not estimate crop transform")
    return cv2.warpAffine(image, transform.params[:2], (size, size))


def _session(model_path: Path, providers: list[str]) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=providers
    )


def _load_swapper(model_path: Path, providers: list[str]):
    session = _session(model_path, providers)
    inputs = session.get_inputs()
    target_shape = next(
        (item.shape for item in inputs if len(item.shape) == 4), None
    )
    if target_shape is None:
        raise RuntimeError(f"no image input in {model_path}")
    size = int(target_shape[-1])
    if size not in (128, 256):
        raise RuntimeError(f"unsupported INSwapper input size {size} in {model_path}")
    # Direct construction avoids a second ORT session for 128px models and
    # also supports compatible native-256 checkpoints that the historical
    # InsightFace model router misclassifies.
    return INSwapper(model_file=str(model_path), session=session)


def _embedding(face) -> np.ndarray | None:
    value = getattr(face, "normed_embedding", None)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-8 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    analyser.prepare(ctx_id=0, det_size=(640, 640))
    source_image = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if source_image is None:
        raise ValueError(f"could not read source image: {args.source}")
    source_face = _largest(analyser.get(source_image))
    source_embedding = _embedding(source_face)
    if source_face is None or source_embedding is None:
        raise RuntimeError("no usable source face was detected")

    models = {
        name: _load_swapper(Path(path), providers) for name, path in args.model
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "real").mkdir(exist_ok=True)
    for name in models:
        (args.output / name).mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    records = []
    index = 0
    misses = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames and index >= args.max_frames:
                break
            target_face = _largest(analyser.get(frame))
            if target_face is None:
                misses += 1
                index += 1
                continue
            filename = f"{index:04d}.png"
            real_crop = _deepfakebench_crop(frame, target_face.kps)
            cv2.imwrite(str(args.output / "real" / filename), real_crop)
            record = {
                "index": index,
                "filename": filename,
                "detection_score": float(target_face.det_score),
                "models": {},
            }
            for name, model in models.items():
                model_started = time.perf_counter()
                swapped = model.get(
                    frame.copy(), target_face, source_face, paste_back=True
                )
                crop = _deepfakebench_crop(swapped, target_face.kps)
                cv2.imwrite(str(args.output / name / filename), crop)
                output_face = _largest(analyser.get(swapped))
                output_embedding = _embedding(output_face)
                similarity = (
                    float(np.dot(source_embedding, output_embedding))
                    if output_embedding is not None
                    else None
                )
                record["models"][name] = {
                    "latency_ms": (time.perf_counter() - model_started) * 1000.0,
                    "source_identity_cosine": similarity,
                }
            records.append(record)
            index += 1
    finally:
        capture.release()

    report = {
        "schema_version": "1.0",
        "classification": "paired offline model ablation",
        "video": str(args.video),
        "video_sha256": _sha256(args.video),
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "models": {
            name: {"path": str(path), "sha256": _sha256(Path(path))}
            for name, path in args.model
        },
        "frames_seen": index,
        "frames_written": len(records),
        "detection_misses": misses,
        "runtime_seconds": time.perf_counter() - started,
        "records": records,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    summary = {
        "frames": len(records),
        "misses": misses,
        "models": {
            name: {
                "latency_ms_mean": float(
                    np.mean([row["models"][name]["latency_ms"] for row in records])
                ),
                "source_identity_cosine_mean": float(
                    np.mean(
                        [
                            row["models"][name]["source_identity_cosine"]
                            for row in records
                            if row["models"][name]["source_identity_cosine"] is not None
                        ]
                    )
                ),
            }
            for name in models
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
