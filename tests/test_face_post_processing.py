from __future__ import annotations

import cv2
import numpy as np

import modules.globals
from modules.processors.frame import face_swapper
from modules.processors.frame import frequency_repair
from modules.processors.frame.frequency_repair import (
    FacePasteRegion,
    match_camera_detail,
)


def test_color_match_is_bounded_and_moves_toward_target():
    face_swapper.reset_temporal_state()
    generated = np.full((128, 128, 3), 60, dtype=np.uint8)
    target = np.full((128, 128, 3), 100, dtype=np.uint8)

    matched = face_swapper._match_aligned_face_color(
        generated, target, 1.0, track_id=0
    )

    assert float(matched.mean()) > float(generated.mean())
    assert float(matched.mean()) < float(target.mean())


def test_temporal_smoothing_is_motion_compensated_and_face_local(monkeypatch):
    face_swapper.reset_temporal_state()
    monkeypatch.setattr(modules.globals, "sharpness", 0.0)
    monkeypatch.setattr(modules.globals, "enable_interpolation", True)
    monkeypatch.setattr(modules.globals, "interpolation_weight", 0.75)
    previous = np.zeros((120, 160, 3), dtype=np.uint8)
    previous[35:85, 45:95] = 180
    current = np.zeros_like(previous)
    current[36:86, 47:97] = 100
    bbox = [np.array([47, 36, 97, 86], dtype=np.int32)]

    face_swapper.apply_post_processing(previous, bbox, motion_matrix=None)
    result = face_swapper.apply_post_processing(
        current,
        bbox,
        motion_matrix=np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float32),
    )

    assert np.array_equal(result[0, 0], current[0, 0])
    assert int(result[60, 70, 0]) > int(current[60, 70, 0])


def test_camera_detail_match_uses_generated_residual_without_copying_reference():
    yy, xx = np.indices((96, 96))
    camera = np.empty((96, 96, 3), dtype=np.uint8)
    for channel, offset in enumerate((0, 9, 17)):
        camera[:, :, channel] = np.clip(
            100 + offset + ((xx + yy) % 5) * 8,
            0,
            255,
        )
    generated = cv2.GaussianBlur(camera, (0, 0), 1.4)
    # Make the synthetic face materially different so the change mask owns it.
    generated[18:78, 18:78] = np.clip(
        generated[18:78, 18:78].astype(np.int16) + 12,
        0,
        255,
    ).astype(np.uint8)

    repaired = match_camera_detail(
        generated,
        camera,
        [np.array([16, 16, 80, 80])],
        strength=3.5,
    )

    generated_detail = generated.astype(np.float32) - cv2.GaussianBlur(
        generated.astype(np.float32), (0, 0), 1.0
    )
    repaired_detail = repaired.astype(np.float32) - cv2.GaussianBlur(
        repaired.astype(np.float32), (0, 0), 1.0
    )
    assert float(repaired_detail[24:72, 24:72].std()) > float(
        generated_detail[24:72, 24:72].std()
    )
    assert np.array_equal(repaired[0, 0], generated[0, 0])
    assert not np.array_equal(repaired, camera)


def test_camera_detail_uses_exact_paste_alpha_and_leaves_seam_untouched(
    monkeypatch,
):
    frequency_repair.reset_camera_detail_state()
    rng = np.random.default_rng(19)
    camera_luma = rng.integers(70, 150, (96, 96), dtype=np.uint8)
    camera = np.repeat(
        camera_luma[:, :, None],
        3,
        axis=2,
    )
    generated = camera.copy()
    generated[16:80, 16:80] = cv2.GaussianBlur(
        generated[16:80, 16:80], (0, 0), 1.4
    )
    generated[16:80, 16:80] = np.clip(
        generated[16:80, 16:80].astype(np.int16) + 10,
        0,
        255,
    ).astype(np.uint8)
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:56, 8:56] = 100  # paste transition: deliberately protected
    alpha[14:50, 14:50] = 255
    region = FacePasteRegion((16, 16, 80, 80), alpha, track_id=9)

    monkeypatch.setattr(
        frequency_repair,
        "_changed_region_mask",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("pixel-difference fallback was used")
        ),
    )
    repaired = match_camera_detail(
        generated,
        camera,
        [],
        strength=3.5,
        paste_regions=[region],
    )

    assert np.array_equal(repaired[:16], generated[:16])
    assert np.array_equal(repaired[16:24, 24:72], generated[16:24, 24:72])
    generated_detail = generated[34:62, 34:62].astype(np.float32)
    repaired_detail = repaired[34:62, 34:62].astype(np.float32)
    assert float(repaired_detail.std()) > float(generated_detail.std())


def test_camera_detail_gain_is_smoothed_and_reset_per_track():
    frequency_repair.reset_camera_detail_state()
    assert frequency_repair._smooth_detail_scale(4, 1.0) == 1.0
    assert frequency_repair._smooth_detail_scale(4, 2.0) == 1.18
    assert frequency_repair._smooth_detail_scale(5, 2.0) == 2.0
    frequency_repair.reset_camera_detail_state()
    assert frequency_repair._smooth_detail_scale(4, 2.0) == 2.0


def test_boundary_preservation_changes_only_translucent_paste_edge():
    camera = np.zeros((16, 16, 3), dtype=np.uint8)
    generated = np.full_like(camera, 100)
    alpha = np.zeros((12, 12), dtype=np.uint8)
    alpha[2:10, 2:10] = 128
    alpha[4:8, 4:8] = 255
    region = FacePasteRegion((2, 2, 14, 14), alpha, track_id=3)

    repaired = match_camera_detail(
        generated,
        camera,
        [],
        strength=0.0,
        paste_regions=[region],
        boundary_strength=0.5,
    )

    # Outside the authoritative paste, and at alpha=0, nothing is touched.
    assert np.array_equal(repaired[0, 0], generated[0, 0])
    assert np.array_equal(repaired[2, 2], generated[2, 2])
    # The solid identity-bearing core is also exactly preserved.
    assert np.array_equal(repaired[7, 7], generated[7, 7])
    # Only the half-transparent transition moves toward the camera frame.
    assert 45 <= int(repaired[5, 5, 0]) <= 55


def test_boundary_preservation_requires_authoritative_paste_alpha():
    camera = np.zeros((24, 24, 3), dtype=np.uint8)
    generated = np.full_like(camera, 100)

    repaired = match_camera_detail(
        generated,
        camera,
        [np.array([4, 4, 20, 20])],
        strength=0.0,
        boundary_strength=1.0,
    )

    # Bounding boxes are insufficient to identify a safe seam. The legacy
    # changed-pixel fallback remains detail-only and cannot copy camera RGB.
    assert np.array_equal(repaired, generated)


def test_fast_paste_back_exports_exact_output_region():
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    generated = np.full((32, 32, 3), 180, dtype=np.uint8)
    aligned = np.empty_like(generated)
    affine = np.array([[1.0, 0.0, -24.0], [0.0, 1.0, -20.0]], dtype=np.float32)
    regions: list[FacePasteRegion] = []

    face_swapper._fast_paste_back(
        frame,
        generated,
        aligned,
        affine,
        paste_regions=regions,
        track_id=7,
        opacity=0.75,
    )

    assert len(regions) == 1
    region = regions[0]
    x1, y1, x2, y2 = region.bounds
    assert region.alpha.shape == (y2 - y1, x2 - x1)
    assert region.alpha.dtype == np.uint8
    assert region.track_id == 7
    assert region.opacity == 0.75
    assert int(frame.max()) > 0
