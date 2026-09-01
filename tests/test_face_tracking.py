from __future__ import annotations

import numpy as np
import cv2
from insightface.app.common import Face

from modules.face_tracking import TemporalFaceTracker, bbox_iou


def make_face(x: float, y: float, size: float = 120.0, score: float = 0.95) -> Face:
    kps = np.array(
        [
            [x + size * 0.32, y + size * 0.38],
            [x + size * 0.68, y + size * 0.38],
            [x + size * 0.50, y + size * 0.55],
            [x + size * 0.38, y + size * 0.72],
            [x + size * 0.62, y + size * 0.72],
        ],
        dtype=np.float32,
    )
    return Face(
        bbox=np.array([x, y, x + size, y + size], dtype=np.float32),
        kps=kps,
        det_score=score,
    )


def textured_frame(face: Face) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for index, point in enumerate(face.kps.astype(int)):
        cv2.circle(frame, tuple(point), 7, (60 + index * 30,) * 3, -1)
        cv2.line(
            frame,
            (point[0] - 8, point[1]),
            (point[0] + 8, point[1]),
            (255, 255, 255),
            1,
        )
    return frame


def test_bbox_iou_handles_overlap_and_disjoint_boxes():
    first = np.array([0, 0, 100, 100], dtype=np.float32)
    assert bbox_iou(first, first) == 1.0
    assert bbox_iou(first, np.array([200, 200, 250, 250])) == 0.0
    assert 0.1 < bbox_iou(first, np.array([50, 50, 150, 150])) < 0.2


def test_detection_interval_one_rechecks_every_frame():
    tracker = TemporalFaceTracker(30)
    face = make_face(200, 90)
    frame = textured_frame(face)
    tracker.update(frame, [face], detection_ran=True)

    assert tracker.should_detect(1)
    assert not tracker.should_detect(2)


def test_tracker_keeps_prior_identity_instead_of_leftmost_face():
    tracker = TemporalFaceTracker(30)
    target = make_face(260, 100)
    frame = textured_frame(target)
    tracker.update(frame, [target], detection_ran=True)

    left_decoy = make_face(25, 100, score=0.99)
    moved_target = make_face(263, 100, score=0.90)
    result = tracker.update(
        frame,
        [left_decoy, moved_target],
        detection_ran=True,
        smoothing=0.6,
    )

    assert result.primary is not None
    assert float(result.primary.bbox[0]) > 200
    assert tracker.snapshot()["reacquisitions"] == 0


def test_smoothing_rejects_stationary_detector_jitter():
    tracker = TemporalFaceTracker(30)
    first = make_face(200, 90)
    frame = textured_frame(first)
    tracker.update(frame, [first], detection_ran=True)

    jittered = make_face(208, 94)
    result = tracker.update(
        frame,
        [jittered],
        detection_ran=True,
        smoothing=0.75,
    )

    assert result.primary is not None
    assert 200.0 < float(result.primary.bbox[0]) < 208.0
    metrics = tracker.snapshot()
    assert metrics["landmark_correction_iod_percent"] > 0
    assert metrics["bbox_prediction_iou"] < 1


def test_detector_miss_uses_motion_prediction_without_hard_toggle():
    tracker = TemporalFaceTracker(30)
    first = make_face(200, 90)
    frame = textured_frame(first)
    tracker.update(frame, [first], detection_ran=True)
    moved = cv2.warpAffine(
        frame,
        np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float32),
        (frame.shape[1], frame.shape[0]),
    )

    result = tracker.update(
        moved,
        [],
        detection_ran=True,
        smoothing=0.65,
        grace_frames=5,
    )

    assert result.primary is not None
    assert result.mode in ("flow", "hold")
    assert 0.0 < result.swap_alpha < 1.0
    metrics = tracker.snapshot()
    assert metrics["detection_misses"] == 1
    assert metrics["swap_transitions"] == 0


def test_detector_miss_grace_expires_instead_of_swapping_forever():
    tracker = TemporalFaceTracker(30)
    face = make_face(200, 90)
    original = textured_frame(face)
    tracker.update(original, [face], detection_ran=True)

    result = None
    for step in range(1, 4):
        moved = cv2.warpAffine(
            original,
            np.array([[1, 0, step * 2], [0, 1, step]], dtype=np.float32),
            (original.shape[1], original.shape[0]),
        )
        result = tracker.update(
            moved,
            [],
            detection_ran=True,
            grace_frames=2,
        )

    assert result is not None
    assert result.primary is None
    assert result.mode == "lost"
    assert result.swap_alpha == 0.0


def test_single_face_reacquires_after_one_large_association_rejection():
    tracker = TemporalFaceTracker(30)
    first = make_face(80, 90)
    frame = textured_frame(first)
    tracker.update(frame, [first], detection_ran=True)

    moved = make_face(400, 90)
    rejected = tracker.update(frame, [moved], detection_ran=True)
    reacquired = tracker.update(frame, [moved], detection_ran=True)

    assert rejected.primary is not None  # grace/flow, not an abrupt toggle
    assert reacquired.primary is not None
    assert float(reacquired.primary.bbox[0]) > 300.0
    metrics = tracker.snapshot()
    assert metrics["raw_detections"] == 1
    assert metrics["valid_detections"] == 1
    assert metrics["association_rejections"] == 1
    assert metrics["reacquisitions"] == 1
