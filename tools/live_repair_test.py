#!/usr/bin/env python3
"""Headless live camera test for frequency/boundary repair.

Spawns a local camera, runs the full swap pipeline with repair features
enabled, and prints per-frame metrics so you can verify the repairs
work without a GUI.

Usage:
    .venv/bin/python tools/live_repair_test.py \
        --source path/to/face.jpg \
        --camera 0 \
        --repair-mode balanced
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import modules.globals
from modules.face_analyser import get_one_face, get_face_analyser
from modules.processors.frame.face_swapper import swap_face, reset_temporal_state
from modules.processors.frame.frequency_repair import (
    apply_frequency_repair,
    _spectral_energy_ratio,
)
from modules import imread_unicode


REPAIR_PRESETS = {
    "off": dict(hf=0.0, checker=0.0, wavelet=0.0, boundary=False, seam=0.0),
    "light": dict(hf=0.15, checker=0.2, wavelet=0.2, boundary=False, seam=0.2),
    "balanced": dict(hf=0.3, checker=0.4, wavelet=0.5, boundary=True, seam=0.35),
    "aggressive": dict(hf=0.45, checker=0.6, wavelet=0.7, boundary=True, seam=0.5),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source face image")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--repair-mode", choices=REPAIR_PRESETS.keys(), default="balanced")
    parser.add_argument("--frames", type=int, default=150, help="Frames to process")
    parser.add_argument("--show", action="store_true", help="Show cv2 window (needs display)")
    args = parser.parse_args()

    # Apply repair preset
    preset = REPAIR_PRESETS[args.repair_mode]
    modules.globals.repair_hf_strength = preset["hf"]
    modules.globals.repair_checkerboard = preset["checker"]
    modules.globals.repair_wavelet = preset["wavelet"]
    modules.globals.repair_boundary_mask = preset["boundary"]
    modules.globals.repair_boundary_strength = preset["seam"]
    modules.globals.color_match_strength = 0.35

    # Load source face
    source_img = imread_unicode(str(args.source))
    if source_img is None:
        print(f"ERROR: cannot read source image: {args.source}")
        sys.exit(1)

    print(f"Loading face analyser and source face...")
    source_face = get_one_face(source_img)
    if source_face is None:
        print("ERROR: no face detected in source image")
        sys.exit(1)
    print(f"Source face: embedding dim={source_face.normed_embedding.shape}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera {args.camera}: {w}x{h} @ {fps:.1f} FPS")
    print(f"Repair mode: {args.repair_mode} {preset}")
    print(f"Processing {args.frames} frames...")
    print()

    # Metrics accumulators
    swap_times = []
    repair_times = []
    face_stabilities = []
    spectral_ratios = []
    frame_count = 0

    reset_temporal_state()

    try:
        while frame_count < args.frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {frame_count}: camera read failed, stopping")
                break

            t_total_start = time.perf_counter()

            # Detect face in frame
            target_face = get_one_face(frame)
            if target_face is None:
                frame_count += 1
                if args.show:
                    cv2.imshow("Repair Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Run swap
            t_swap = time.perf_counter()
            result = swap_face(source_face, target_face, frame.copy())
            swap_ms = (time.perf_counter() - t_swap) * 1000
            swap_times.append(swap_ms)

            # Compute spectral stats on the face region if we have a bbox
            if hasattr(target_face, 'bbox') and target_face.bbox is not None:
                bbox = target_face.bbox.astype(int)
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(result.shape[1], x2), min(result.shape[0], y2)
                if x2 > x1 + 16 and y2 > y1 + 16:
                    face_crop = result[y1:y2, x1:x2]
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    low, high = _spectral_energy_ratio(face_gray)
                    spectral_ratios.append(high / (low + 1e-8))

            total_ms = (time.perf_counter() - t_total_start) * 1000
            frame_count += 1

            # Print every 10 frames
            if frame_count % 10 == 0:
                avg_swap = np.mean(swap_times[-10:]) if swap_times else 0
                avg_spec = np.mean(spectral_ratios[-10:]) if spectral_ratios else 0
                print(
                    f"Frame {frame_count:4d} | "
                    f"swap: {avg_swap:.1f}ms | "
                    f"total: {total_ms:.1f}ms | "
                    f"spectral_ratio: {avg_spec:.3f} | "
                    f"fps: {1000/total_ms:.1f}"
                )

            if args.show:
                cv2.imshow("Repair Test", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    # Summary
    print()
    print("=" * 60)
    print(f"REPAIR MODE: {args.repair_mode}")
    print(f"FRAMES PROCESSED: {frame_count}")
    if swap_times:
        st = np.array(swap_times)
        print(f"SWAP LATENCY:   mean={st.mean():.1f}ms  P95={np.percentile(st, 95):.1f}ms  max={st.max():.1f}ms")
    if spectral_ratios:
        sr = np.array(spectral_ratios)
        print(f"SPECTRAL RATIO: mean={sr.mean():.4f}  std={sr.std():.4f}")
        print(f"  (higher = more natural HF content; ~5-6 is camera-captured, ~0.15 is raw GAN)")
    print("=" * 60)


if __name__ == "__main__":
    main()
