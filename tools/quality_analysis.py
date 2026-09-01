#!/usr/bin/env python3
"""Full quality analysis of live face-swap output.

Runs the swap pipeline on camera frames, records every quality metric the
quality_pipeline tracks, and produces a detailed report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import modules.globals
from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import swap_face, reset_temporal_state
from modules.processors.frame.frequency_repair import _spectral_energy_ratio
from modules.quality_pipeline import QualityPipeline
from modules import imread_unicode


def analyze_frames(
    source_path: str,
    camera: int = 0,
    num_frames: int = 120,
    repair_mode: str = "balanced",
) -> dict:
    """Run swap pipeline and collect comprehensive quality metrics."""

    PRESETS = {
        "off": dict(
            hf=0.0, checker=0.0, wavelet=0.0, boundary=False, seam=0.0
        ),
        "balanced": dict(
            hf=0.3, checker=0.4, wavelet=0.5, boundary=True, seam=0.35
        ),
        "aggressive": dict(
            hf=0.45, checker=0.6, wavelet=0.7, boundary=True, seam=0.5
        ),
    }
    preset = PRESETS.get(repair_mode, PRESETS["balanced"])
    modules.globals.repair_hf_strength = preset["hf"]
    modules.globals.repair_checkerboard = preset["checker"]
    modules.globals.repair_wavelet = preset["wavelet"]
    modules.globals.repair_boundary_mask = preset["boundary"]
    modules.globals.repair_boundary_strength = preset["seam"]
    modules.globals.color_match_strength = 0.35
    modules.globals.sharpness = 0.0
    modules.globals.enable_interpolation = False

    # Load source
    source_img = imread_unicode(source_path)
    source_face = get_one_face(source_img)
    if source_face is None:
        return {"error": "no face in source"}

    # Open camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        return {"error": f"camera {camera} not available"}
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Warm up camera
    for _ in range(15):
        cap.read()

    reset_temporal_state()
    qp = QualityPipeline()

    # Per-frame records
    records = []
    swap_times = []
    detection_misses = 0
    frame_count = 0

    while frame_count < num_frames:
        ret, raw_frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        target_face = get_one_face(raw_frame)

        if target_face is None:
            detection_misses += 1
            frame_count += 1
            continue

        # Record quality gate source
        qp.capture_source(raw_frame)

        # Run swap
        t_swap = time.perf_counter()
        result = swap_face(source_face, target_face, raw_frame.copy())
        swap_ms = (time.perf_counter() - t_swap) * 1000
        swap_times.append(swap_ms)

        # Build tracking dict
        face_bbox = target_face.bbox
        tracking = {
            "detection_miss_percent": detection_misses / max(1, frame_count + 1) * 100,
            "landmark_correction_p95_percent": 0.0,
            "face_height_px": float(face_bbox[3] - face_bbox[1]) if face_bbox is not None else 0,
            "detection_score": float(getattr(target_face, "det_score", 0)),
        }

        # Feed quality pipeline
        result = qp.process(
            result,
            fallback=raw_frame,
            processing_active=True,
            face_bbox=face_bbox,
            tracking=tracking,
            swap_applied=True,
        )

        # Additional spectral analysis
        if face_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(result.shape[1], x2), min(result.shape[0], y2)
            if x2 > x1 + 16 and y2 > y1 + 16:
                face_crop = result[y1:y2, x1:x2]
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                orig_crop = raw_frame[y1:y2, x1:x2]
                orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
                _, hf_out = _spectral_energy_ratio(face_gray)
                _, hf_orig = _spectral_energy_ratio(orig_gray)
                records.append({
                    "frame": frame_count,
                    "swap_ms": swap_ms,
                    "hf_output": hf_out,
                    "hf_original": hf_orig,
                })

        qp.record_pipeline_timing(swap_ms / 1000)
        frame_count += 1

    cap.release()
    snapshot = qp.snapshot()

    # Build report
    st = np.array(swap_times) if swap_times else np.array([0])
    hf_out = np.array([r["hf_output"] for r in records]) if records else np.array([0])
    hf_orig = np.array([r["hf_original"] for r in records]) if records else np.array([0])

    return {
        "repair_mode": repair_mode,
        "preset": preset,
        "frames_processed": frame_count,
        "frames_with_face": len(swap_times),
        "detection_misses": detection_misses,
        "detection_miss_rate": round(detection_misses / max(1, frame_count) * 100, 1),
        "swap_latency_ms": {
            "mean": round(float(st.mean()), 1),
            "p50": round(float(np.median(st)), 1),
            "p95": round(float(np.percentile(st, 95)), 1),
            "max": round(float(st.max()), 1),
        },
        "quality_pipeline": {
            "whole_frame_score": snapshot.get("score", 0),
            "face_stability_score": snapshot.get("face", {}).get("stability_score", 0),
            "luma": snapshot.get("luma", 0),
            "detail_laplacian": snapshot.get("detail_laplacian", 0),
            "temporal_delta": snapshot.get("temporal_delta", 0),
            "repeat_streak": snapshot.get("repeat_streak", 0),
        },
        "face_metrics": {
            "detail_ratio": snapshot.get("face", {}).get("detail_ratio", 0),
            "luma_mismatch": snapshot.get("face", {}).get("luma_mismatch", 0),
            "chroma_mismatch_lab": snapshot.get("face", {}).get("chroma_mismatch_lab", 0),
            "excess_flicker": snapshot.get("face", {}).get("excess_flicker", 0),
            "excess_flicker_p95": snapshot.get("face", {}).get("excess_flicker_p95", 0),
            "seam_ring_delta": snapshot.get("face", {}).get("seam_ring_delta", 0),
            "seam_ring_delta_lab": snapshot.get("face", {}).get("seam_ring_delta_lab", 0),
            "seam_temporal_std": snapshot.get("face", {}).get("seam_temporal_std", 0),
            "swap_transitions": snapshot.get("face", {}).get("swap_transitions", 0),
        },
        "spectral_analysis": {
            "hf_energy_output_mean": round(float(hf_out.mean()), 4),
            "hf_energy_output_std": round(float(hf_out.std()), 4),
            "hf_energy_original_mean": round(float(hf_orig.mean()), 4),
            "hf_energy_original_std": round(float(hf_orig.std()), 4),
            "hf_ratio": round(float(hf_out.mean() / (hf_orig.mean() + 1e-8)), 4),
        },
        "pipeline_timing_ms": {
            "mean": snapshot.get("pipeline_ms", 0),
            "p95": snapshot.get("pipeline_ms_p95", 0),
            "quality_gate": snapshot.get("quality_gate_ms", 0),
        },
        "warnings": snapshot.get("warnings", []),
    }


def grade_report(report: dict) -> dict:
    """Assign letter grades to each metric category."""

    def grade(score: float, thresholds: list[tuple[float, str]]) -> str:
        for threshold, letter in thresholds:
            if score >= threshold:
                return letter
        return "F"

    face = report.get("face_metrics", {})
    spec = report.get("spectral_analysis", {})
    pipeline = report.get("quality_pipeline", {})
    timing = report.get("swap_latency_ms", {})

    grades = {}

    # 1. Face stability (composite score)
    stability = pipeline.get("face_stability_score", 0)
    grades["face_stability"] = {
        "value": stability,
        "grade": grade(stability, [(90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D")]),
    }

    # 2. Seam visibility (lower is better)
    seam_lab = face.get("seam_ring_delta_lab", 100)
    grades["seam_visibility"] = {
        "value": seam_lab,
        "grade": grade(100 - seam_lab, [(95, "A+"), (90, "A"), (80, "B"), (60, "C"), (40, "D")]),
    }

    # 3. Temporal flicker (lower is better)
    flicker = face.get("excess_flicker_p95", 100)
    grades["temporal_flicker"] = {
        "value": flicker,
        "grade": grade(100 - flicker, [(98, "A+"), (95, "A"), (90, "B"), (80, "C"), (70, "D")]),
    }

    # 4. Color coherence (lower mismatch is better)
    luma_mm = face.get("luma_mismatch", 100)
    chroma_mm = face.get("chroma_mismatch_lab", 100)
    color_err = max(luma_mm, chroma_mm)
    grades["color_coherence"] = {
        "value": {"luma": luma_mm, "chroma": chroma_mm},
        "grade": grade(100 - color_err, [(97, "A+"), (93, "A"), (85, "B"), (70, "C"), (50, "D")]),
    }

    # 5. Texture preservation (detail ratio, ideal=1.0)
    detail_r = face.get("detail_ratio", 0)
    detail_err = abs(np.log(max(0.01, detail_r)))
    grades["texture_preservation"] = {
        "value": detail_r,
        "grade": grade(100 - detail_err * 100, [(95, "A+"), (85, "A"), (70, "B"), (50, "C"), (30, "D")]),
    }

    # 6. Spectral naturalness (HF ratio, ideal ≈ 1.0 = matched to original)
    hf_ratio = spec.get("hf_ratio", 0)
    # Ideal is close to 1.0 (HF content matches original)
    spectral_err = abs(hf_ratio - 1.0)
    grades["spectral_naturalness"] = {
        "value": hf_ratio,
        "grade": grade(100 - spectral_err * 100, [(90, "A+"), (80, "A"), (60, "B"), (40, "C"), (20, "D")]),
    }

    # 7. Detection robustness (detection miss rate, lower is better)
    miss_rate = report.get("detection_miss_rate", 100)
    grades["detection_reliability"] = {
        "value": miss_rate,
        "grade": grade(100 - miss_rate, [(95, "A+"), (90, "A"), (80, "B"), (70, "C"), (50, "D")]),
    }

    # 8. Latency (for live viability)
    p95 = timing.get("p95", 9999)
    grades["live_latency"] = {
        "value": p95,
        "grade": grade(1000 - p95, [(900, "A+"), (700, "A"), (400, "B"), (0, "C"), (-500, "D")]),
    }

    return grades


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repair-mode", default="balanced")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Running quality analysis: {args.frames} frames, repair={args.repair_mode}")
    report = analyze_frames(args.source, args.camera, args.frames, args.repair_mode)

    if "error" in report:
        print(f"ERROR: {report['error']}")
        sys.exit(1)

    grades = grade_report(report)
    report["grades"] = grades

    # Print summary
    print()
    print("=" * 65)
    print(f"  QUALITY REPORT — repair_mode={report['repair_mode']}")
    print("=" * 65)
    print(f"  Frames: {report['frames_with_face']}/{report['frames_processed']} with face")
    print(f"  Detection miss rate: {report['detection_miss_rate']}%")
    print()

    print("  GRADE  METRIC                      VALUE")
    print("  -----  -------------------------   ----")
    for name, g in grades.items():
        label = name.replace("_", " ").title()
        if isinstance(g["value"], dict):
            parts = [f"{k}={v}" for k, v in g["value"].items()]
            val = ", ".join(parts)
        else:
            val = g["value"]
        print(f"  {g['grade']:>4}   {label:27s}   {val}")

    print()
    print("  SWAP LATENCY")
    lat = report["swap_latency_ms"]
    print(f"    mean: {lat['mean']}ms  P50: {lat['p50']}ms  P95: {lat['p95']}ms  max: {lat['max']}ms")

    print()
    print("  FACE METRICS")
    fm = report["face_metrics"]
    print(f"    detail_ratio:        {fm['detail_ratio']:.3f}  (ideal=1.0)")
    print(f"    luma_mismatch:       {fm['luma_mismatch']:.2f}  (target<8.0)")
    print(f"    chroma_mismatch:     {fm['chroma_mismatch_lab']:.2f}  (target<7.0)")
    print(f"    excess_flicker:      {fm['excess_flicker']:.3f}  (target<1.5)")
    print(f"    flicker_p95:         {fm['excess_flicker_p95']:.3f}")
    print(f"    seam_ring_delta_lab: {fm['seam_ring_delta_lab']:.3f}  (target<5.0)")
    print(f"    seam_temporal_std:   {fm['seam_temporal_std']:.3f}  (target<3.0)")

    print()
    print("  SPECTRAL ANALYSIS")
    sa = report["spectral_analysis"]
    print(f"    HF energy (output):  {sa['hf_energy_output_mean']:.4f} ± {sa['hf_energy_output_std']:.4f}")
    print(f"    HF energy (original):{sa['hf_energy_original_mean']:.4f} ± {sa['hf_energy_original_std']:.4f}")
    print(f"    HF ratio:            {sa['hf_ratio']:.4f}  (1.0=matched, <1=GAN-smooth)")

    print()
    print("  PIPELINE TIMING")
    pt = report["pipeline_timing_ms"]
    print(f"    pipeline_ms:    {pt['mean']:.1f}ms mean")
    print(f"    pipeline_ms_p95:{pt['p95']:.1f}ms")
    print(f"    quality_gate:   {pt['quality_gate']:.2f}ms")

    warnings = report.get("warnings", [])
    if warnings and warnings != ["collecting samples"]:
        print()
        print(f"  WARNINGS: {', '.join(warnings)}")

    print()
    print("=" * 65)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"  Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
