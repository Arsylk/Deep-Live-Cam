#!/usr/bin/env python3
"""Create paired DeepfakeBench-style crops from a synchronized capture.

The face detector runs only on each original camera frame.  Its exact affine
transform is reused for the corresponding processed frame, preventing crop
jitter or detector differences from becoming a synthetic classification cue.
"""

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
from skimage.transform import SimilarityTransform


def _largest(faces):
    return max(
        faces,
        key=lambda face: float(
            max(0.0, face.bbox[2] - face.bbox[0])
            * max(0.0, face.bbox[3] - face.bbox[1])
        ),
        default=None,
    )


def _transform(landmarks: np.ndarray) -> np.ndarray:
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
        raise RuntimeError("could not estimate face-crop transform")
    return transform.params[:2].astype(np.float32)


def _open(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    return capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--reuse-misses",
        type=int,
        default=2,
        help="maximum consecutive detector misses allowed to reuse the last transform",
    )
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=providers,
    )
    analyser.prepare(ctx_id=0, det_size=(640, 640))

    real_dir = args.output / "real"
    fake_dir = args.output / "fake"
    real_dir.mkdir(parents=True, exist_ok=False)
    fake_dir.mkdir(parents=True, exist_ok=False)
    reference_capture = _open(args.reference)
    processed_capture = _open(args.processed)
    records: list[dict] = []
    last_transform: np.ndarray | None = None
    consecutive_misses = 0
    seen = 0
    try:
        while not args.max_frames or seen < args.max_frames:
            reference_ok, reference = reference_capture.read()
            processed_ok, processed = processed_capture.read()
            if reference_ok != processed_ok:
                raise RuntimeError("paired videos ended at different frames")
            if not reference_ok:
                break
            if reference.shape != processed.shape:
                raise RuntimeError("paired frame dimensions differ")

            face = _largest(analyser.get(reference))
            reused = False
            if face is not None:
                affine = _transform(face.kps)
                last_transform = affine
                consecutive_misses = 0
            elif last_transform is not None and consecutive_misses < args.reuse_misses:
                affine = last_transform
                consecutive_misses += 1
                reused = True
            else:
                last_transform = None
                consecutive_misses += 1
                seen += 1
                continue

            filename = f"{len(records):04d}.png"
            real_crop = cv2.warpAffine(reference, affine, (256, 256))
            fake_crop = cv2.warpAffine(processed, affine, (256, 256))
            if not cv2.imwrite(str(real_dir / filename), real_crop):
                raise OSError(f"could not write {real_dir / filename}")
            if not cv2.imwrite(str(fake_dir / filename), fake_crop):
                raise OSError(f"could not write {fake_dir / filename}")
            records.append(
                {
                    "source_frame": seen,
                    "filename": filename,
                    "detection_score": (
                        float(face.det_score) if face is not None else None
                    ),
                    "bbox": (
                        np.asarray(face.bbox, dtype=float).round(4).tolist()
                        if face is not None
                        else None
                    ),
                    "kps": (
                        np.asarray(face.kps, dtype=float).round(4).tolist()
                        if face is not None
                        else None
                    ),
                    "reused_transform": reused,
                }
            )
            seen += 1
    finally:
        reference_capture.release()
        processed_capture.release()

    metadata = {
        "schema_version": "1.0",
        "classification": "paired diagnostic corpus; not a leaderboard dataset",
        "preprocessing": (
            "DeepfakeBench 256px alignment template and 1.3 scale; "
            "one reference-frame transform applied identically to each pair"
        ),
        "reference": str(args.reference),
        "processed": str(args.processed),
        "frames_seen": seen,
        "frame_pairs": len(records),
        "records": records,
    }
    (args.output / "crop-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"frames_seen": seen, "frame_pairs": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
