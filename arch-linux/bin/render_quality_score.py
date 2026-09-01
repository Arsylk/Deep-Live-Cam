#!/usr/bin/env python3
"""Measure how indistinguishable a rendered face swap is from real footage.

The score compares the swapped face region against the surrounding real pixels
of the SAME frame — the two biggest "this is fake" tells are:

  * a visible seam (the face border sits at a different colour/brightness than
    the skin just outside it), and
  * a skin-tone mismatch between the swapped face and the person's own neck.

A third term keeps the face's high-frequency detail close to the surrounding
footage: a face that is sharper (or softer) than everything around it reads as
pasted on.  Detail ratio ~1.0 is ideal.

These are cheap, reference-free, and monotonic enough to drive an offline
parameter search (see offline_renderer.py --quality auto).  They are a proxy for
perceived realism, not a identity-preservation metric.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityScore:
    seam_delta_lab: float       # face-border vs outer-ring LAB delta (lower better)
    neck_delta_lab: float       # face vs neck skin-tone LAB delta (lower better)
    detail_ratio: float         # face HF energy / ring HF energy (ideal ~1.0)
    realism: float              # 0-100, how seamlessly the face blends in
    identity: float | None      # 0-100, how well it matches the SOURCE identity
    composite: float            # 0-100, combined realism + identity
    grade: str                  # letter grade vs commercial-quality thresholds
    frames_scored: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hf_energy(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _boundary_band(h: int, w: int, bbox: tuple[int, int, int, int], thickness: int):
    """A mask of the ring straddling the face bbox edge (inner+outer band).

    This is where a swap either blends seamlessly into the real footage or
    leaves a visible seam.  Measuring HERE isolates swap-induced discontinuity
    from unrelated scene contrast (collar, hair, neck shadow).
    """
    x1, y1, x2, y2 = bbox
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(outer, (x1 - thickness, y1 - thickness),
                  (x2 + thickness, y2 + thickness), 255, -1)
    cv2.rectangle(inner, (x1 + thickness, y1 + thickness),
                  (x2 - thickness, y2 - thickness), 255, -1)
    band = cv2.subtract(outer, inner)
    return band > 0


def _swap_seam_delta(
    original: np.ndarray, swapped: np.ndarray, bbox: tuple[int, int, int, int]
) -> float:
    """How much the swap disturbs the boundary relative to the original.

    A seamless swap leaves the transition across the face edge looking like it
    did before (only the interior changes); a visible seam introduces a NEW
    gradient/colour step at the boundary.  We measure the change the swap made
    to the boundary band, so a naturally high-contrast scene does not penalise
    a clean swap.  Lower is better.
    """
    h, w = original.shape[:2]
    x1, y1, x2, y2 = bbox
    thickness = max(3, (x2 - x1) // 16)
    band = _boundary_band(h, w, (x1, y1, x2, y2), thickness)
    if not np.any(band):
        return 0.0
    lab_o = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Per-pixel LAB change the swap introduced, averaged over the boundary band.
    delta = np.linalg.norm(lab_s - lab_o, axis=2)
    return float(delta[band].mean())


def _swap_detail_delta(
    original: np.ndarray, swapped: np.ndarray, bbox: tuple[int, int, int, int]
) -> float:
    """Ratio of face-interior high-frequency detail swapped vs original.

    Ideal ~1.0: the swapped face should carry about as much fine detail as the
    real face it replaced.  Far below 1 = smeared/plastic; far above = an
    over-sharpened, pasted-on look.  Measured interior-only (avoids the seam).
    """
    x1, y1, x2, y2 = bbox
    ix1, iy1 = x1 + (x2 - x1) // 6, y1 + (y2 - y1) // 6
    ix2, iy2 = x2 - (x2 - x1) // 6, y2 - (y2 - y1) // 6
    if ix2 - ix1 < 8 or iy2 - iy1 < 8:
        return 1.0
    o_hf = _hf_energy(original[iy1:iy2, ix1:ix2]) + 1e-6
    s_hf = _hf_energy(swapped[iy1:iy2, ix1:ix2])
    return s_hf / o_hf


def _frame_metrics(frame: np.ndarray, face: Any) -> dict[str, float] | None:
    if face is None or getattr(face, "bbox", None) is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None

    face_roi = frame[y1:y2, x1:x2]
    pad = max(8, (x2 - x1) // 8)
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = frame[ry1:ry2, rx1:rx2]

    lab_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ring = cv2.cvtColor(ring, cv2.COLOR_BGR2LAB).astype(np.float32)
    seam = float(
        np.linalg.norm(
            lab_face.reshape(-1, 3).mean(0) - lab_ring.reshape(-1, 3).mean(0)
        )
    )

    # Neck / below-jaw band as a real skin-tone reference for the same person.
    ny1, ny2 = min(h - 1, y2), min(h, y2 + pad * 2)
    neck = float("nan")
    if ny2 > ny1:
        neck_roi = frame[ny1:ny2, x1:x2]
        if neck_roi.size:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            hw, hh = (x2 - x1) // 4, (y2 - y1) // 4
            centre = frame[cy - hh:cy + hh, cx - hw:cx + hw]
            if centre.size:
                lab_c = cv2.cvtColor(centre, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab_n = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                neck = float(
                    np.linalg.norm(
                        lab_c.reshape(-1, 3).mean(0) - lab_n.reshape(-1, 3).mean(0)
                    )
                )

    ring_hf = _hf_energy(ring) + 1e-6
    detail_ratio = _hf_energy(face_roi) / ring_hf
    return {"seam": seam, "neck": neck, "detail": detail_ratio}


def _embedding(face: Any) -> np.ndarray | None:
    """Unit-normalised ArcFace embedding of a detected face, if available."""
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    vec = np.asarray(emb, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def identity_similarity(face_a: Any, face_b: Any) -> float | None:
    """Cosine similarity (0..1) between two faces' ArcFace embeddings.

    1.0 = identical identity, ~0 = unrelated.  Commercial swappers land the
    swapped-vs-source similarity in roughly 0.55-0.75; the same-person baseline
    across frames is ~0.85+.
    """
    ea, eb = _embedding(face_a), _embedding(face_b)
    if ea is None or eb is None:
        return None
    cos = float(np.dot(ea, eb))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0)) if cos < 0 else max(0.0, min(1.0, cos))


def _grade(realism: float, identity: float | None) -> str:
    """Letter grade against commercial-quality thresholds.

    Realism (seamless blend, measured swap-relative) and identity (right
    person) are graded jointly: the result is only as good as its weaker axis,
    so the grade is the LOWER of the two bands.  A/A+ stands next to a
    commercial face swapper (near-invisible seam + strong identity transfer),
    B/B+ is a solid, clearly-usable result, C/D/F flag a visible seam or weak
    identity.  Plus/minus bands make a 'B+ consistently' target well-defined.
    """
    def band(v: float, cuts: list[tuple[float, str]]) -> str:
        for threshold, letter in cuts:
            if v >= threshold:
                return letter
        return "F"

    realism_grade = band(
        realism,
        [(92, "A+"), (86, "A"), (80, "A-"), (74, "B+"), (68, "B"),
         (60, "C"), (48, "D")],
    )
    if identity is None:
        return realism_grade
    identity_grade = band(
        identity,
        # identity is 0-100 (cosine*100).  ~68+ rivals commercial transfer,
        # 58-68 clearly the source, below leans toward a blended identity.
        [(72, "A+"), (66, "A"), (60, "A-"), (54, "B+"), (48, "B"),
         (40, "C"), (32, "D")],
    )
    order = ["F", "D", "C", "B", "B+", "A-", "A", "A+"]
    # The overall grade is the weaker of the two axes.
    return min(realism_grade, identity_grade, key=order.index)


def score_swap_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    detector: Any,
    source_face: Any = None,
) -> QualityScore | None:
    """Score (original, swapped) frame pairs for swap-INDUCED realism.

    This is the accurate realism measure: it compares the swapped frame against
    the original at the face boundary and interior, so it captures the seam and
    texture the SWAP introduced rather than the scene's inherent contrast (a
    striped collar or shadowed neck no longer penalises a clean swap).

      * boundary_delta: LAB change the swap made across the face edge band
        (lower = the transition still looks like the real footage).
      * detail_delta: interior HF ratio swapped/original (ideal ~1.0).

    identity (if source given) is measured on the swapped frame as before.
    """
    boundary: list[float] = []
    detail: list[float] = []
    ident: list[float] = []
    for original, swapped in pairs:
        face = detector(swapped)
        if face is None or getattr(face, "bbox", None) is None:
            continue
        h, w = swapped.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue
        boundary.append(_swap_seam_delta(original, swapped, (x1, y1, x2, y2)))
        detail.append(_swap_detail_delta(original, swapped, (x1, y1, x2, y2)))
        if source_face is not None:
            sim = identity_similarity(face, source_face)
            if sim is not None:
                ident.append(sim)
    if not boundary:
        return None

    boundary_med = float(np.median(boundary))
    detail_med = float(np.median(detail))

    # Boundary LAB change: 0 -> perfect, ~25 -> a clearly visible seam.
    boundary_score = max(0.0, 100.0 - boundary_med * 4.0)
    detail_score = max(0.0, 100.0 - abs(np.log(max(0.05, detail_med))) * 100.0)
    # The seam dominates perceived realism; interior detail is the guard rail.
    realism = round(0.7 * boundary_score + 0.3 * detail_score, 2)

    identity = round(float(np.median(ident)) * 100.0, 2) if ident else None
    composite = realism if identity is None else round(0.5 * realism + 0.5 * identity, 2)
    return QualityScore(
        seam_delta_lab=round(boundary_med, 2),
        neck_delta_lab=0.0,
        detail_ratio=round(detail_med, 3),
        realism=realism,
        identity=identity,
        composite=composite,
        grade=_grade(realism, identity),
        frames_scored=len(boundary),
    )


def score_frames(
    frames: list[np.ndarray],
    detector: Any,
    source_face: Any = None,
) -> QualityScore | None:
    """Score decoded BGR frames for realism and (if source given) identity.

    detector(frame) -> face-with-.bbox/.kps/.normed_embedding or None.
    source_face: the source identity's Face (with embedding) to compare against;
    when provided the composite blends realism and identity transfer.
    """
    seam: list[float] = []
    neck: list[float] = []
    detail: list[float] = []
    ident: list[float] = []
    for frame in frames:
        face = detector(frame)
        m = _frame_metrics(frame, face)
        if m is None:
            continue
        seam.append(m["seam"])
        if not np.isnan(m["neck"]):
            neck.append(m["neck"])
        detail.append(m["detail"])
        if source_face is not None:
            sim = identity_similarity(face, source_face)
            if sim is not None:
                ident.append(sim)
    if not seam:
        return None

    seam_med = float(np.median(seam))
    neck_med = float(np.median(neck)) if neck else seam_med
    detail_med = float(np.median(detail))

    seam_score = max(0.0, 100.0 - seam_med * 3.0)
    neck_score = max(0.0, 100.0 - neck_med * 2.5)
    detail_score = max(0.0, 100.0 - abs(np.log(max(0.05, detail_med))) * 100.0)
    realism = round(0.45 * seam_score + 0.35 * neck_score + 0.20 * detail_score, 2)

    identity = round(float(np.median(ident)) * 100.0, 2) if ident else None
    if identity is None:
        composite = realism
    else:
        # Both axes matter equally: a seamless swap of the wrong face, or the
        # right face with a visible seam, are both distinguishable.
        composite = round(0.5 * realism + 0.5 * identity, 2)

    return QualityScore(
        seam_delta_lab=round(seam_med, 2),
        neck_delta_lab=round(neck_med, 2),
        detail_ratio=round(detail_med, 3),
        realism=realism,
        identity=identity,
        composite=composite,
        grade=_grade(realism, identity),
        frames_scored=len(seam),
    )


def sample_frames(video_path: str, count: int = 12) -> list[np.ndarray]:
    """Evenly sample up to `count` frames across a video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max(1, count)) if total else 1
    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < count:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def score_video(
    video_path: str,
    detector: Any,
    count: int = 12,
    source_face: Any = None,
) -> QualityScore | None:
    return score_frames(sample_frames(video_path, count), detector, source_face)


def main() -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from modules.face_analyser import get_one_face, get_many_faces

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument(
        "--source",
        help="source face image; enables identity-transfer scoring + grade",
    )
    args = parser.parse_args()

    source_face = None
    if args.source:
        import cv2 as _cv2
        src_img = _cv2.imread(args.source)
        if src_img is not None:
            faces = get_many_faces(src_img)
            if faces:
                source_face = faces[0]

    result = score_video(args.video, get_one_face, args.frames, source_face)
    print(json.dumps(result.as_dict() if result else {}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
