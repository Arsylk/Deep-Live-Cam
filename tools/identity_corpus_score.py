#!/usr/bin/env python3
"""Measure source-identity similarity across one or more face-crop corpora."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

bootstrap_root = os.environ.get("DLC_BOOTSTRAP_ROOT")
if bootstrap_root:
    sys.path.insert(0, bootstrap_root)
    import run as _windows_runtime_bootstrap  # noqa: F401

import cv2
import insightface
import numpy as np


def _largest(faces):
    return max(
        faces,
        key=lambda face: float(
            max(0.0, face.bbox[2] - face.bbox[0])
            * max(0.0, face.bbox[3] - face.bbox[1])
        ),
        default=None,
    )


def _embedding(face) -> np.ndarray | None:
    value = getattr(face, "normed_embedding", None)
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        action="append",
        nargs=2,
        metavar=("NAME", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
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
    source_embedding = _embedding(_largest(analyser.get(source_image)))
    if source_embedding is None:
        raise RuntimeError("source image has no recognized face")

    result = {"source": str(args.source), "corpora": {}}
    for name, raw_directory in args.corpus:
        directory = Path(raw_directory)
        similarities: list[float] = []
        paths = sorted(directory.glob("*.png"))
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            embedding = _embedding(_largest(analyser.get(image)))
            if embedding is not None:
                similarities.append(float(np.dot(source_embedding, embedding)))
        values = np.asarray(similarities, dtype=np.float64)
        result["corpora"][name] = {
            "images": len(paths),
            "recognized": len(similarities),
            "source_identity_cosine_mean": (
                float(values.mean()) if values.size else None
            ),
            "source_identity_cosine_median": (
                float(np.median(values)) if values.size else None
            ),
            "source_identity_cosine_p05": (
                float(np.percentile(values, 5)) if values.size else None
            ),
        }
    document = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(document + "\n", encoding="utf-8")
    print(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
