#!/usr/bin/env python3
"""Compare technical quality of decoded raw and returned frame samples."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np


def frame_metrics(path: str, exclude_bottom: int) -> dict[str, float]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {path}")
    if exclude_bottom and image.shape[0] > exclude_bottom + 64:
        image = image[:-exclude_bottom]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = gray.astype(np.float32)
    vertical_all = float(np.mean(np.abs(pixels[:, 1:] - pixels[:, :-1])))
    horizontal_all = float(np.mean(np.abs(pixels[1:, :] - pixels[:-1, :])))
    vertical_blocks = float(
        np.mean(np.abs(pixels[:, 8::8] - pixels[:, 7:-1:8]))
    )
    horizontal_blocks = float(
        np.mean(np.abs(pixels[8::8, :] - pixels[7:-1:8, :]))
    )
    return {
        "luma": float(gray.mean()),
        "saturation": float(hsv[:, :, 1].mean()),
        "detail_laplacian": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "edge_energy": float(
            np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)))
            + np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)))
        ),
        "blockiness_ratio": (vertical_blocks + horizontal_blocks)
        / max(0.001, vertical_all + horizontal_all),
        "black_clip_percent": 100.0 * float(np.mean(gray <= 3)),
        "white_clip_percent": 100.0 * float(np.mean(gray >= 252)),
    }


def summarize(pattern: str, exclude_bottom: int) -> dict[str, object]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"no frames matched {pattern}")
    frames = [frame_metrics(path, exclude_bottom) for path in paths]
    return {
        "frames": len(frames),
        **{
            key: round(fmean(frame[key] for frame in frames), 4)
            for key in frames[0]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="glob for decoded raw frames")
    parser.add_argument("--returned", required=True, help="glob for decoded return frames")
    parser.add_argument("--exclude-bottom", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw = summarize(args.raw, args.exclude_bottom)
    returned = summarize(args.returned, args.exclude_bottom)
    result = {
        "raw": raw,
        "returned": returned,
        "return_over_raw_percent": {
            key: round(100.0 * float(returned[key]) / max(0.0001, float(raw[key])), 2)
            for key in ("detail_laplacian", "edge_energy")
        },
        "blockiness_delta": round(
            float(returned["blockiness_ratio"])
            - float(raw["blockiness_ratio"]),
            4,
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
