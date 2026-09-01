"""Quality invariants for the offline face-swap renderer.

The offline renderer has an unlimited per-frame budget, so it must run the FULL
swap pipeline (colour match + seam reblend + detail repair), not a bare swap,
and its quality tiers must escalate the indistinguishability-driving knobs while
keeping the swapped face's detail natural (not over-sharpened past the source).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "arch-linux" / "bin" / "offline_renderer.py"


def _quality_settings() -> dict:
    """Extract the quality_settings literal without importing heavy deps."""
    tree = ast.parse(RENDERER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "quality_settings":
                    return ast.literal_eval(node.value)
    raise AssertionError("quality_settings not found in offline_renderer.py")


TIERS = ["fast", "balanced", "high", "max"]


def test_renderer_drives_the_full_pipeline_not_a_bare_swap():
    source = RENDERER.read_text(encoding="utf-8")
    # The renderer must call the full pipeline entry point, and must NOT fall
    # back to the bare swap that skips colour match / seam reblend / repair.
    assert "process_frame(" in source
    assert "import" in source and "swap_face" not in source.split("def render_video")[0].split("import")[-1]


def test_every_tier_enables_the_seam_and_colour_pipeline():
    settings = _quality_settings()
    for tier in TIERS:
        g = settings[tier]["globals"]
        # Colour match and the boundary reblend are the biggest seam / skin-tone
        # indistinguishability drivers and must be active on every tier.
        assert g["color_match_strength"] > 0.0, tier
        assert g["repair_boundary_mask"] is True, tier
        assert g["repair_boundary_strength"] > 0.0, tier
        # Mouth mask preserves real mouth motion (critical for talking heads).
        assert g["mouth_mask"] is True, tier


def test_quality_tiers_escalate_the_reconciliation():
    settings = _quality_settings()
    colour = [settings[t]["globals"]["color_match_strength"] for t in TIERS]
    boundary = [settings[t]["globals"]["repair_boundary_strength"] for t in TIERS]
    detector = [settings[t]["face_detector_score"] for t in TIERS]
    # Non-decreasing across fast -> balanced -> high -> max.
    assert colour == sorted(colour), colour
    assert boundary == sorted(boundary), boundary
    assert detector == sorted(detector), detector
    # Max must be strictly stronger than fast on the key seam/colour knobs.
    assert colour[-1] > colour[0]
    assert boundary[-1] > boundary[0]


def test_detail_repair_stays_in_the_natural_range():
    settings = _quality_settings()
    for tier in TIERS:
        g = settings[tier]["globals"]
        # camera-detail matching and sharpness must stay gentle: pushing the
        # face's high-frequency detail above the surrounding footage reads as an
        # over-sharpened, pasted-on face (detail ratio should stay near 1.0).
        assert 0.0 <= g["repair_camera_detail"] <= 2.0, tier
        assert 0.0 <= g["sharpness"] <= 0.25, tier


def test_renderer_exposes_auto_tune_and_optional_enhance():
    source = RENDERER.read_text(encoding="utf-8")
    # 'auto' per-clip tuning is a selectable quality, and the search is a local,
    # model-free loop scored with the swap-relative realism + identity metric.
    assert '"auto"' in source
    assert "def auto_tune(" in source
    assert "score_swap_pairs(" in source
    # The optional GFPGAN pass must degrade gracefully when the model is absent
    # (policy: the app never downloads models) and blend at a strength.
    assert "def _try_load_enhancer(" in source
    assert "enhance_strength" in source
    assert "cv2.addWeighted" in source  # partial blend, not full replace


def test_renderer_grades_the_final_product_with_identity():
    source = RENDERER.read_text(encoding="utf-8")
    # The finished render is scored SWAP-RELATIVE (rendered vs input at the face
    # boundary) against the source identity, and emits a machine-readable
    # commercial-quality grade line for the manager.
    assert "score_swap_pairs(pairs, get_one_face, source_face)" in source
    assert "[grade] " in source
    # The auto tuner must feed the source face into scoring so it balances
    # realism and identity transfer, not realism alone.
    assert "score_swap_pairs(\n            pairs, get_one_face, source_face" in source or \
        "score_swap_pairs(pairs, get_one_face, source_face)" in source


def test_quality_score_is_reference_free_and_monotonic():
    """The scorer must reward a clean composite over a seam/tone-mismatched one."""
    import sys

    bin_dir = ROOT / "arch-linux" / "bin"
    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))
    import numpy as np
    from render_quality_score import score_frames

    class _Face:
        def __init__(self, bbox):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.kps = None

    h, w = 240, 320
    bbox = (110, 70, 210, 190)

    def detector(_frame):
        return _Face(bbox)

    # Clean: face region blends with the surround (same mid-grey skin tone).
    clean = np.full((h, w, 3), 150, dtype=np.uint8)
    # Mismatched: face region is a jarring off-tone block (visible seam + tone).
    mism = clean.copy()
    mism[70:190, 110:210] = (40, 200, 60)  # green-ish, far from surround

    clean_score = score_frames([clean], detector)
    mism_score = score_frames([mism], detector)
    assert clean_score is not None and mism_score is not None
    # A seamless region must score strictly higher than a mismatched one.
    assert clean_score.composite > mism_score.composite
    assert clean_score.seam_delta_lab < mism_score.seam_delta_lab


def test_identity_similarity_and_joint_grade():
    """Identity term rewards a matching embedding and gates the joint grade."""
    import sys

    bin_dir = ROOT / "arch-linux" / "bin"
    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))
    import numpy as np
    from render_quality_score import score_frames, identity_similarity, _grade

    rng = np.random.default_rng(0)
    vec = rng.standard_normal(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    other = rng.standard_normal(512).astype(np.float32)
    other /= np.linalg.norm(other)

    class _Face:
        def __init__(self, bbox, emb):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.kps = None
            self.normed_embedding = emb

    # Same embedding -> ~1.0; unrelated -> well below.
    same = identity_similarity(_Face((0, 0, 1, 1), vec), _Face((0, 0, 1, 1), vec))
    diff = identity_similarity(_Face((0, 0, 1, 1), vec), _Face((0, 0, 1, 1), other))
    assert same is not None and same > 0.99
    assert diff is not None and diff < same

    # Joint grade is driven by the WEAKER axis: strong identity cannot rescue
    # poor realism, and a clean seam cannot rescue a wrong identity.
    assert _grade(95.0, 20.0) in ("D", "F")   # perfect blend, wrong person
    assert _grade(40.0, 95.0) in ("D", "F")   # right person, visible seam
    assert _grade(90.0, 72.0) in ("A", "A+")  # both strong -> commercial grade

    # A swapped region carrying the source embedding scores identity; a frame
    # with no source given has identity=None and grades on realism alone.
    h, w = 240, 320
    frame = np.full((h, w, 3), 150, dtype=np.uint8)
    src = _Face((0, 0, 1, 1), vec)

    def detector(_f):
        return _Face((110, 70, 210, 190), vec)

    with_id = score_frames([frame], detector, source_face=src)
    without = score_frames([frame], detector)
    assert with_id is not None and with_id.identity is not None
    assert with_id.identity > 95.0
    assert without is not None and without.identity is None


def test_swap_relative_realism_ignores_scene_contrast():
    """Realism must measure swap-induced seam, not the scene's own contrast."""
    import sys

    bin_dir = ROOT / "arch-linux" / "bin"
    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))
    import numpy as np
    from render_quality_score import score_swap_pairs

    class _Face:
        def __init__(self, bbox):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.kps = None

    h, w = 240, 320
    bbox = (110, 70, 210, 190)
    rng = np.random.default_rng(1)
    original = (np.full((h, w, 3), 150, dtype=np.int16)
                + rng.integers(-20, 20, (h, w, 3))).clip(0, 255).astype(np.uint8)

    def detector(_f):
        return _Face(bbox)

    # A clean swap changes only the face interior, leaving the boundary intact.
    clean = original.copy()
    clean[95:165, 135:185] = (120, 130, 140)
    # A seamed swap slaps a hard block over the whole bbox -> new edge step.
    seamed = original.copy()
    seamed[70:190, 110:210] = (60, 180, 80)

    clean_s = score_swap_pairs([(original, clean)], detector)
    seamed_s = score_swap_pairs([(original, seamed)], detector)
    assert clean_s is not None and seamed_s is not None
    # The clean swap must score far higher realism than the seamed one, and the
    # clean boundary delta must be small (near-invisible seam).
    assert clean_s.realism > seamed_s.realism + 30
    assert clean_s.seam_delta_lab < 5.0
    assert seamed_s.seam_delta_lab > 20.0


def test_grade_bands_define_a_reachable_b_plus():
    import sys

    bin_dir = ROOT / "arch-linux" / "bin"
    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))
    from render_quality_score import _grade

    # A clean swap (realism ~77) with strong identity (~80) must reach B+ or
    # better -- the target we tune the offline pipeline to hit consistently.
    assert _grade(77.0, 80.0) in ("B+", "A-", "A", "A+")
    assert _grade(81.0, 81.0) in ("A-", "A", "A+")
    # The weaker axis still gates: a visible seam cannot pass as B+.
    assert _grade(62.0, 90.0) in ("C", "D", "F")
