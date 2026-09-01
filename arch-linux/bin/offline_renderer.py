#!/usr/bin/env python3
"""Offline high-quality face swap renderer with progress reporting.

Takes a recorded video, renders it frame-by-frame with unlimited time per
frame, and reports progress to stdout. Used by the manager's Render page.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add repository root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from modules.core import decode_execution_providers, suggest_execution_providers
from modules.face_analyser import (
    detect_many_faces_fast,
    ensure_landmarks,
    get_many_faces,
    get_one_face,
)
from modules.processors.frame.face_swapper import (
    get_face_swapper,
    process_frame,
    reset_temporal_state,
)
from render_quality_score import score_frames, score_swap_pairs, QualityScore


# ---------------------------------------------------------------- enhancement

_ENHANCER_READY: bool | None = None


def _try_load_enhancer() -> bool:
    """Load the optional GFPGAN final-restoration model if present on disk.

    Per repository policy the app never downloads models, so face restoration
    is available only when the operator has placed the model in models/.  When
    absent this returns False and the renderer proceeds without it rather than
    failing.
    """
    global _ENHANCER_READY
    if _ENHANCER_READY is not None:
        return _ENHANCER_READY
    try:
        from modules.processors.frame import face_enhancer
        if not face_enhancer.pre_check():
            _ENHANCER_READY = False
            return False
        face_enhancer.get_face_enhancer()
        _ENHANCER_READY = True
    except Exception as exc:  # model missing / load failure -> skip gracefully
        print(f"[warn] face enhancer unavailable, skipping: {exc}", file=sys.stderr)
        _ENHANCER_READY = False
    return _ENHANCER_READY


def _enhance_frame(frame: np.ndarray, strength: float) -> np.ndarray:
    """Blend a GFPGAN-restored face back at `strength` (0..1).

    A partial blend is deliberate: full-strength GFPGAN produces pore-less,
    over-smooth skin that reads as obviously AI and can drift identity, which
    HURTS indistinguishability against real footage.  A ~0.5 blend recovers
    eye/teeth/edge sharpness while keeping real skin texture.
    """
    from modules.processors.frame import face_enhancer
    original = frame.copy()
    enhanced = face_enhancer.enhance_face(frame)
    strength = max(0.0, min(1.0, strength))
    if strength >= 0.999:
        return enhanced
    return cv2.addWeighted(enhanced, strength, original, 1.0 - strength, 0.0)


def _apply_globals(app_globals, values: dict) -> None:
    for key, value in values.items():
        setattr(app_globals, key, value)


def log_progress(frame: int, total: int, fps: float = 0.0, thumb: str | None = None) -> None:
    """Emit progress JSON line for the manager to parse."""
    progress: dict[str, Any] = {
        "frame": frame,
        "total": total,
        "percent": round(100 * frame / total, 1) if total > 0 else 0,
        "fps": round(fps, 1),
    }
    if thumb:
        progress["thumb"] = thumb
    print(json.dumps(progress), flush=True)


def render_video(
    input_path: Path,
    output_path: Path,
    face_path: Path | None,
    quality: str = "high",
    enhance_strength: float = 0.0,
    globals_override: dict | None = None,
) -> None:
    """Render video with face swap, unlimited time per frame.

    enhance_strength > 0 runs an optional GFPGAN face-restoration final pass
    (blended at that strength) when the model is available.  globals_override,
    when set, replaces the tier's pipeline globals (used by the auto tuner).
    """
    
    # Quality presets.  Unlike the live path, the offline renderer has an
    # unlimited per-frame budget, so higher tiers spend it on the seam and
    # colour reconciliation that actually drive indistinguishability, while
    # keeping the face's high-frequency detail MATCHED to the surrounding real
    # footage (detail ratio ~1.0 is the project's own quality target; pushing
    # HF above the background reads as an over-sharpened, pasted-on face).
    # So camera-detail matching and sharpness stay gentle; colour match and
    # target-preserving seam reblend escalate with the tier.
    quality_settings = {
        "fast": {
            "execution_providers": ["CPUExecutionProvider"],
            "face_detector_score": 0.5,
            "globals": {
                "color_match_strength": 0.4,
                "sharpness": 0.1,
                "mouth_mask": True,
                "mouth_mask_size": 8.0,
                "repair_boundary_mask": True,
                "repair_boundary_strength": 0.3,
                "repair_camera_detail": 0.5,
                "repair_hf_strength": 0.25,
                "repair_wavelet": 0.4,
                "repair_checkerboard": 0.35,
            },
        },
        "balanced": {
            "execution_providers": ["CPUExecutionProvider"],
            "face_detector_score": 0.6,
            "globals": {
                "color_match_strength": 0.5,
                "sharpness": 0.12,
                "mouth_mask": True,
                "mouth_mask_size": 10.0,
                "repair_boundary_mask": True,
                "repair_boundary_strength": 0.45,
                "repair_camera_detail": 0.8,
                "repair_hf_strength": 0.3,
                "repair_wavelet": 0.5,
                "repair_checkerboard": 0.4,
            },
        },
        "high": {
            "execution_providers": ["CPUExecutionProvider"],
            "face_detector_score": 0.7,
            "globals": {
                "color_match_strength": 0.6,
                "sharpness": 0.12,
                "mouth_mask": True,
                "mouth_mask_size": 10.0,
                "repair_boundary_mask": True,
                "repair_boundary_strength": 0.55,
                "repair_camera_detail": 0.6,
                "repair_hf_strength": 0.3,
                "repair_wavelet": 0.55,
                "repair_checkerboard": 0.45,
            },
        },
        "max": {
            "execution_providers": ["CPUExecutionProvider"],
            "face_detector_score": 0.8,
            "globals": {
                "color_match_strength": 0.7,
                "sharpness": 0.18,
                "mouth_mask": True,
                "mouth_mask_size": 12.0,
                "repair_boundary_mask": True,
                "repair_boundary_strength": 0.65,
                "repair_camera_detail": 1.3,
                "repair_hf_strength": 0.35,
                "repair_wavelet": 0.6,
                "repair_checkerboard": 0.5,
            },
        },
    }
    
    # 'auto' tunes the realism knobs per-clip using the high tier as its base.
    auto_tune_requested = quality == "auto"
    tier = "high" if auto_tune_requested else quality
    settings = quality_settings.get(tier, quality_settings["high"])
    
    # Initialize models
    print(f"[init] Loading models (quality={quality})...", file=sys.stderr)
    from modules import globals as app_globals
    app_globals.execution_providers = decode_execution_providers(
        settings["execution_providers"]
    )
    # Drive the FULL swap pipeline (colour match, seam reblend, detail repair)
    # rather than a bare swap, so the rendered output is at least as convincing
    # as the live stream and better where the extra per-frame budget allows.
    app_globals.many_faces = True
    app_globals.opacity = 1.0
    app_globals.enable_interpolation = False  # offline is deterministic; no EMA
    active_globals = dict(settings["globals"])
    if globals_override:
        active_globals.update(globals_override)
    _apply_globals(app_globals, active_globals)
    reset_temporal_state()
    print(
        "[init] pipeline: color_match=%.2f camera_detail=%.1f boundary=%.2f"
        % (
            active_globals["color_match_strength"],
            active_globals["repair_camera_detail"],
            active_globals["repair_boundary_strength"],
        ),
        file=sys.stderr,
    )
    enhance_on = enhance_strength > 0.0 and _try_load_enhancer()
    if enhance_strength > 0.0:
        print(
            "[init] face restoration: %s (strength=%.2f)"
            % ("on" if enhance_on else "unavailable", enhance_strength),
            file=sys.stderr,
        )
    
    # Load face swapper
    face_swapper = get_face_swapper()
    if not face_swapper:
        print("[error] Failed to load face swapper model", file=sys.stderr)
        sys.exit(1)
    
    # Load source face if provided
    source_face = None
    if face_path and face_path.exists():
        print(f"[init] Loading source face from {face_path}", file=sys.stderr)
        source_img = cv2.imread(str(face_path))
        if source_img is not None:
            faces = get_many_faces(source_img)
            if faces:
                source_face = faces[0]
                print(f"[init] Source face loaded", file=sys.stderr)
                # Per-clip auto tuning: search realism knobs, adopt the winner.
                if auto_tune_requested:
                    tuned = auto_tune(
                        input_path, source_face, app_globals,
                        active_globals, enhance_strength, enhance_on,
                    )
                    _apply_globals(app_globals, tuned)
                    reset_temporal_state()
            else:
                print("[warn] No face found in source image", file=sys.stderr)
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[error] Cannot open input video: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(
        f"[init] Input: {width}×{height} @ {fps:.1f} fps, {total_frames} frames",
        file=sys.stderr,
    )
    
    # Create output writer via ffmpeg pipe for H.264 output.
    # cv2's mp4v codec produces MPEG4 Part 2 which cannot be copy-muxed
    # into MPEG-TS by the prerecorded relay.
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        # The receiver's file_relay fully re-decodes this MP4, so it is an
        # intermediate master, not a copy-muxed source: use High profile with a
        # slow preset and a low CRF to keep the swapped detail the pipeline
        # worked to produce, instead of the old Baseline/medium/CRF18 that
        # threw quality away for a constraint that no longer applies.
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "15",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-x264-params", "aq-mode=3:psy-rd=1.0,0.15",
        "-movflags", "+faststart",
        str(output_path),
    ]
    out = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    if out.stdin is None:
        print(f"[error] Cannot open ffmpeg pipe for output", file=sys.stderr)
        sys.exit(1)
    
    # Process frames
    print("[render] Starting frame-by-frame processing...", file=sys.stderr)
    frame_idx = 0
    start_time = time.time()
    last_log_time = start_time
    last_thumb_time = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Render frame with the full face-swap pipeline (detection + swap +
        # colour match + seam reblend + detail repair).  We detect faces with
        # the detection-only fast path (skipping the recognition embedding the
        # swap never uses) and pass them in, so process_frame does no redundant
        # recognition.  This is provably pixel-identical to the full-detection
        # path; only wasted CPU is removed.  Landmarks are added on demand only
        # when mouth masking (which needs them) is active.
        if source_face is not None:
            try:
                targets = detect_many_faces_fast(frame)
                if targets:
                    if active_globals.get("mouth_mask"):
                        ensure_landmarks(frame, targets)
                    frame = process_frame(
                        source_face, frame, many_faces_list=targets
                    )
                if enhance_on:
                    frame = _enhance_frame(frame, enhance_strength)
            except Exception as e:
                print(f"[warn] Frame {frame_idx} swap failed: {e}", file=sys.stderr)
        
        out.stdin.write(frame.tobytes())
        frame_idx += 1
        
        # Log progress every 0.5s, thumbnail every 1s
        now = time.time()
        if now - last_log_time >= 0.5 or frame_idx == total_frames:
            elapsed = now - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            thumb = None
            if now - last_thumb_time >= 1.0:
                import base64
                small = cv2.resize(frame, (320, 180))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
                thumb = base64.b64encode(buf).decode("ascii")
                last_thumb_time = now
            log_progress(frame_idx, total_frames, current_fps, thumb)
            last_log_time = now
    
    # Cleanup
    cap.release()
    out.stdin.close()
    out.wait()
    
    elapsed = time.time() - start_time
    final_fps = total_frames / elapsed if elapsed > 0 else 0
    print(
        f"[done] Rendered {total_frames} frames in {elapsed:.1f}s ({final_fps:.1f} fps)",
        file=sys.stderr,
    )

    # Grade the finished product against commercial-quality thresholds so the
    # operator sees an objective realism + identity verdict, not just "done".
    # Realism is measured swap-relative (rendered output vs the input clip at
    # the face boundary), which isolates swap artifacts from scene contrast.
    if source_face is not None:
        try:
            from render_quality_score import sample_frames
            orig = sample_frames(str(input_path), count=16)
            done = sample_frames(str(output_path), count=16)
            pairs = list(zip(orig, done))
            final = score_swap_pairs(pairs, get_one_face, source_face) if pairs else None
            if final is not None:
                summary = {
                    "grade": final.grade,
                    "composite": final.composite,
                    "realism": final.realism,
                    "identity": final.identity,
                    "seam_delta_lab": final.seam_delta_lab,
                    "detail_ratio": final.detail_ratio,
                }
                # Machine-readable line for the manager to parse + display.
                print("[grade] " + json.dumps(summary), flush=True)
                print(
                    "[grade] commercial-quality grade %s (realism=%.1f identity=%s composite=%.1f)"
                    % (
                        final.grade, final.realism,
                        "n/a" if final.identity is None else f"{final.identity:.1f}",
                        final.composite,
                    ),
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[warn] final grading skipped: {exc}", file=sys.stderr)


def _render_sample(
    frames: list[np.ndarray],
    source_face,
    app_globals,
    candidate: dict,
    enhance_strength: float,
    enhance_on: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Swap frames with one candidate set; return (original, swapped) pairs."""
    _apply_globals(app_globals, candidate)
    reset_temporal_state()
    out = []
    for frame in frames:
        original = frame.copy()
        swapped = process_frame(source_face, frame.copy())
        if enhance_on:
            swapped = _enhance_frame(swapped, enhance_strength)
        out.append((original, swapped))
    return out


def auto_tune(
    input_path: Path,
    source_face,
    app_globals,
    base_globals: dict,
    enhance_strength: float,
    enhance_on: bool,
) -> dict:
    """Search a few candidate parameter sets and keep the most indistinguishable.

    Offline has no time budget, so we render a handful of sampled frames under
    each candidate, score them against the surrounding real footage (seam +
    skin-tone + detail), and return the winning globals for the full render.
    This is a local, model-free optimisation over the exact realism knobs — no
    network, no external service.
    """
    from render_quality_score import sample_frames

    frames = sample_frames(str(input_path), count=8)
    if not frames or source_face is None:
        return base_globals

    # Candidate grid over the two dominant realism drivers: colour match and
    # target-preserving seam reblend.  Detail knobs stay at the base tier's
    # gentle values so the search cannot over-sharpen.
    candidates: list[dict] = []
    for cm in (0.4, 0.55, 0.7, 0.85):
        for bs in (0.4, 0.55, 0.7):
            cand = dict(base_globals)
            cand["color_match_strength"] = cm
            cand["repair_boundary_strength"] = bs
            candidates.append(cand)

    best_globals = base_globals
    best_score = None
    print(
        f"[auto] searching {len(candidates)} candidates over {len(frames)} sample frames…",
        file=sys.stderr,
    )
    for i, cand in enumerate(candidates, 1):
        pairs = _render_sample(
            frames, source_face, app_globals, cand, enhance_strength, enhance_on
        )
        score = score_swap_pairs(pairs, get_one_face, source_face)
        if score is None:
            continue
        print(
            "[auto] %2d/%d cm=%.2f bs=%.2f -> composite=%.2f (realism=%.1f identity=%s grade=%s)"
            % (
                i, len(candidates), cand["color_match_strength"],
                cand["repair_boundary_strength"], score.composite,
                score.realism,
                "n/a" if score.identity is None else f"{score.identity:.1f}",
                score.grade,
            ),
            file=sys.stderr,
        )
        if best_score is None or score.composite > best_score:
            best_score = score.composite
            best_globals = cand
    if best_score is not None:
        print(
            "[auto] best: cm=%.2f bs=%.2f composite=%.2f"
            % (
                best_globals["color_match_strength"],
                best_globals["repair_boundary_strength"],
                best_score,
            ),
            file=sys.stderr,
        )
    return best_globals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline high-quality face swap renderer"
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, help="Output video file")
    parser.add_argument(
        "--face", type=Path, help="Source face image (optional)"
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high", "max", "auto"],
        default="high",
        help=(
            "Quality preset (default: high). 'auto' searches the realism knobs "
            "per-clip and keeps the most indistinguishable result."
        ),
    )
    parser.add_argument(
        "--enhance",
        type=float,
        default=0.0,
        metavar="STRENGTH",
        help=(
            "Optional GFPGAN face-restoration final pass, blended at STRENGTH "
            "(0..1). Requires the model in models/; ignored if absent. ~0.5 "
            "recovers eye/teeth sharpness without a plastic, over-smooth look."
        ),
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[error] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        render_video(
            args.input, args.output, args.face, args.quality,
            enhance_strength=args.enhance,
        )
    except KeyboardInterrupt:
        print("\n[cancel] Render cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[error] Render failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
