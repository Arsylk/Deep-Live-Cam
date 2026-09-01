"""Motion-aware temporal face tracking for the network live pipeline.

The detector remains the source of truth, but detector frames are reconciled
with an optical-flow prediction from the preceding frame.  This removes the
two most visible live-mode failure modes: stale landmarks between detector
runs and a one-frame hard reversion to the unprocessed image when detection
briefly misses.

The module intentionally contains no model inference.  That keeps it cheap,
testable with synthetic geometry, and usable with InsightFace ``Face`` objects
without coupling it to a particular execution provider.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np


def _geometry(face: Any, name: str) -> np.ndarray | None:
    value = getattr(face, name, None)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    return array.copy() if array.size else None


def _clone_face(face: Any) -> Any:
    """Clone an InsightFace-style mapping without sharing geometry arrays."""
    try:
        clone = face.__class__(**dict(face))
    except (TypeError, ValueError):
        from insightface.app.common import Face

        clone = Face(**dict(face))
    for name in ("bbox", "kps", "landmark_2d_106"):
        value = _geometry(face, name)
        if value is not None:
            setattr(clone, name, value)
    return clone


def bbox_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None or len(first) != 4 or len(second) != 4:
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    return intersection / max(1e-6, first_area + second_area - intersection)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)


def _transform_bbox(bbox: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    moved = _transform_points(corners, matrix)
    return np.array(
        [
            moved[:, 0].min(),
            moved[:, 1].min(),
            moved[:, 0].max(),
            moved[:, 1].max(),
        ],
        dtype=np.float32,
    )


def _face_size(face: Any) -> tuple[float, float]:
    bbox = _geometry(face, "bbox")
    if bbox is None or len(bbox) != 4:
        return 0.0, 0.0
    return max(0.0, float(bbox[2] - bbox[0])), max(
        0.0, float(bbox[3] - bbox[1])
    )


def _face_center(face: Any) -> np.ndarray:
    bbox = _geometry(face, "bbox")
    if bbox is None:
        return np.zeros(2, dtype=np.float32)
    return np.array(
        [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5],
        dtype=np.float32,
    )


def _interocular_distance(kps: np.ndarray | None, bbox: np.ndarray | None) -> float:
    if kps is not None and kps.shape[0] >= 2:
        distance = float(np.linalg.norm(kps[0] - kps[1]))
        if distance > 1.0:
            return distance
    if bbox is not None:
        return max(1.0, float(bbox[2] - bbox[0]) * 0.35)
    return 1.0


def _percentile(values: deque[float], percentile: float = 95.0) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


@dataclass
class TrackingResult:
    faces: list[Any]
    mode: str
    confidence: float
    swap_alpha: float
    motion_matrix: np.ndarray | None

    @property
    def primary(self) -> Any | None:
        return self.faces[0] if self.faces else None


class TemporalFaceTracker:
    """Track one primary face, with a conservative multi-face fallback."""

    def __init__(self, input_fps: int = 30) -> None:
        self.input_fps = max(1, int(input_fps))
        self._jitter_history: deque[float] = deque(maxlen=self.input_fps * 20)
        self._scale_history: deque[float] = deque(maxlen=self.input_fps * 20)
        self._rotation_history: deque[float] = deque(maxlen=self.input_fps * 20)
        self._flow_history: deque[float] = deque(maxlen=self.input_fps * 20)
        self.reset(reset_counters=True)

    def reset(self, *, reset_counters: bool = False) -> None:
        self.previous_gray: np.ndarray | None = None
        self.active_face: Any | None = None
        self.many_faces: list[Any] = []
        self.mode = "lost"
        self.confidence = 0.0
        self.swap_alpha = 0.0
        self.motion_matrix: np.ndarray | None = None
        self.frames_since_detection = 1_000_000
        self.consecutive_misses = 0
        self._force_detection = True
        self._many_mode = False
        if reset_counters:
            self.frames = 0
            self.detection_runs = 0
            self.detection_hits = 0
            self.detection_misses = 0
            self.max_consecutive_misses = 0
            self.flow_frames = 0
            self.flow_failures = 0
            self.hold_frames = 0
            self.reacquisitions = 0
            self.swap_transitions = 0
            self._last_swap_present = False
            self.last_detection_score = 0.0
            self.last_face_width = 0.0
            self.last_face_height = 0.0
            self.last_face_area_percent = 0.0
            self.last_landmark_correction_percent = 0.0
            self.last_bbox_iou = 0.0
            self.last_scale_correction_percent = 0.0
            self.last_rotation_correction_degrees = 0.0
            self.last_raw_detection_count = 0
            self.last_valid_detection_count = 0
            self.association_rejections = 0
            self._jitter_history.clear()
            self._scale_history.clear()
            self._rotation_history.clear()
            self._flow_history.clear()

    def reset_metrics_window(self) -> None:
        """Clear counters for an A/B run while retaining the tracked face."""
        self.frames = 0
        self.detection_runs = 0
        self.detection_hits = 0
        self.detection_misses = 0
        self.max_consecutive_misses = self.consecutive_misses
        self.flow_frames = 0
        self.flow_failures = 0
        self.hold_frames = 0
        self.reacquisitions = 0
        self.swap_transitions = 0
        self.association_rejections = 0
        self._jitter_history.clear()
        self._scale_history.clear()
        self._rotation_history.clear()
        self._flow_history.clear()

    def should_detect(self, interval: int, *, many_faces: bool = False) -> bool:
        interval = max(1, int(interval))
        return bool(
            many_faces
            or self._force_detection
            or self.active_face is None
            # ``frames_since_detection`` is zero immediately after a detector
            # frame.  Interval one therefore means detect on the very next
            # frame, while interval two permits exactly one flow-only frame.
            or self.frames_since_detection >= interval - 1
            or self.confidence < 0.55
        )

    @staticmethod
    def _valid_candidates(
        detections: Iterable[Any] | None,
        minimum_score: float,
        minimum_size: float,
    ) -> list[Any]:
        candidates = []
        for face in detections or ():
            width, height = _face_size(face)
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            if (
                width >= minimum_size
                and height >= minimum_size
                and score >= minimum_score
            ):
                candidates.append(face)
        return candidates

    def update(
        self,
        frame: np.ndarray,
        detections: Iterable[Any] | None,
        *,
        detection_ran: bool,
        enabled: bool = True,
        smoothing: float = 0.65,
        grace_frames: int = 5,
        minimum_score: float = 0.45,
        minimum_size: float = 64.0,
        many_faces: bool = False,
    ) -> TrackingResult:
        self.frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smoothing = float(np.clip(smoothing, 0.0, 0.95)) if enabled else 0.0
        grace_frames = max(0, int(grace_frames)) if enabled else 0
        if not enabled and not many_faces:
            self.active_face = None
            self.confidence = 0.0
        raw_detections = list(detections or ())
        candidates = self._valid_candidates(
            raw_detections, minimum_score, minimum_size
        )
        if detection_ran:
            self.last_raw_detection_count = len(raw_detections)
            self.last_valid_detection_count = len(candidates)

        if many_faces != self._many_mode:
            self.reset(reset_counters=False)
            self._many_mode = many_faces

        if many_faces:
            result = self._update_many(gray, candidates, detection_ran, smoothing)
        else:
            result = self._update_one(
                gray,
                candidates,
                detection_ran=detection_ran,
                smoothing=smoothing,
                grace_frames=grace_frames,
            )

        self.previous_gray = gray
        self.mode = result.mode
        self.confidence = result.confidence
        self.swap_alpha = result.swap_alpha
        self.motion_matrix = result.motion_matrix
        present = bool(result.faces and result.swap_alpha > 0.01)
        if self.frames > 1 and present != self._last_swap_present:
            self.swap_transitions += 1
        self._last_swap_present = present
        self._update_face_size(result.primary, frame.shape)
        return result

    def _update_one(
        self,
        gray: np.ndarray,
        candidates: list[Any],
        *,
        detection_ran: bool,
        smoothing: float,
        grace_frames: int,
    ) -> TrackingResult:
        predicted, matrix, flow_quality = self._predict(gray, self.active_face)
        self.motion_matrix = matrix
        if matrix is not None:
            self._flow_history.append(flow_quality)

        if detection_ran:
            self.detection_runs += 1
            self.frames_since_detection = 0
            selected = self._select_candidate(candidates, predicted)
            if selected is not None:
                self.detection_hits += 1
                self.consecutive_misses = 0
                self._force_detection = False
                tracked = self._reconcile(predicted, selected, smoothing)
                self.active_face = tracked
                score = float(getattr(selected, "det_score", 0.0) or 0.0)
                self.last_detection_score = score
                self.confidence = float(np.clip(0.65 + score * 0.35, 0.0, 1.0))
                return TrackingResult(
                    [tracked], "detected", self.confidence, 1.0, matrix
                )

            self.detection_misses += 1
            self.consecutive_misses += 1
            self.max_consecutive_misses = max(
                self.max_consecutive_misses, self.consecutive_misses
            )
        else:
            self.frames_since_detection += 1

        if (
            predicted is not None
            and flow_quality >= 0.35
            and (
                not detection_ran
                or self.consecutive_misses <= max(0, grace_frames)
            )
        ):
            self.active_face = predicted
            self.flow_frames += 1
            self._force_detection = detection_ran
            decay = 0.08 * self.consecutive_misses if detection_ran else 0.015
            self.confidence = float(np.clip(self.confidence - decay, 0.0, 1.0))
            alpha = max(
                0.0,
                1.0
                - self.consecutive_misses / max(1.0, grace_frames + 1.0),
            )
            return TrackingResult(
                [predicted], "flow", self.confidence, alpha, matrix
            )

        if self.active_face is not None:
            self.flow_failures += 1
        if self.active_face is not None and self.consecutive_misses <= grace_frames:
            self.hold_frames += 1
            self._force_detection = True
            alpha = max(
                0.0,
                1.0 - (self.consecutive_misses + 1) / max(1.0, grace_frames + 1.0),
            )
            self.confidence *= 0.7
            return TrackingResult(
                [self.active_face], "hold", self.confidence, alpha, None
            )

        self.active_face = None
        self.confidence = 0.0
        self._force_detection = True
        return TrackingResult([], "lost", 0.0, 0.0, None)

    def _update_many(
        self,
        gray: np.ndarray,
        candidates: list[Any],
        detection_ran: bool,
        smoothing: float,
    ) -> TrackingResult:
        # Multi-face mode deliberately detects every frame.  Greedy IoU matching
        # stabilizes each geometry without risking a stale face being pasted on
        # a person who has left the image.
        del gray
        if not detection_ran:
            return TrackingResult(self.many_faces, "hold", 0.5, 1.0, None)
        self.detection_runs += 1
        self.frames_since_detection = 0
        if not candidates:
            self.detection_misses += 1
            self.consecutive_misses += 1
            self.many_faces = []
            return TrackingResult([], "lost", 0.0, 0.0, None)

        unmatched = list(self.many_faces)
        stabilized: list[Any] = []
        for candidate in sorted(
            candidates,
            key=lambda face: _face_size(face)[0] * _face_size(face)[1],
            reverse=True,
        ):
            match = max(
                unmatched,
                key=lambda face: bbox_iou(
                    _geometry(face, "bbox"), _geometry(candidate, "bbox")
                ),
                default=None,
            )
            if match is not None and bbox_iou(
                _geometry(match, "bbox"), _geometry(candidate, "bbox")
            ) >= 0.15:
                stabilized.append(self._reconcile(match, candidate, smoothing))
                unmatched.remove(match)
            else:
                stabilized.append(_clone_face(candidate))
        self.many_faces = stabilized
        self.detection_hits += 1
        self.consecutive_misses = 0
        score = min(
            float(getattr(face, "det_score", 0.0) or 0.0)
            for face in candidates
        )
        self.last_detection_score = score
        return TrackingResult(stabilized, "detected-many", score, 1.0, None)

    def _predict(
        self, gray: np.ndarray, face: Any | None
    ) -> tuple[Any | None, np.ndarray | None, float]:
        if self.previous_gray is None or face is None:
            return None, None, 0.0
        points = _geometry(face, "kps")
        if points is None or points.ndim != 2 or points.shape[0] < 3:
            return None, None, 0.0
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            points.reshape(-1, 1, 2),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.01,
            ),
        )
        if next_points is None or status is None:
            return None, None, 0.0
        status = status.reshape(-1).astype(bool)
        error_values = (
            errors.reshape(-1) if errors is not None else np.zeros(len(status))
        )
        good = status & np.isfinite(error_values) & (error_values < 35.0)
        if int(good.sum()) < 3:
            return None, None, float(good.mean())
        old_good = points[good]
        new_good = next_points.reshape(-1, 2)[good]
        matrix, inliers = cv2.estimateAffinePartial2D(
            old_good,
            new_good,
            method=cv2.LMEDS,
        )
        if matrix is None or not np.isfinite(matrix).all():
            return None, None, float(good.mean())
        scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
        rotation = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        width, height = _face_size(face)
        translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2]))
        if (
            not 0.72 <= scale <= 1.38
            or rotation > 30.0
            or translation > max(80.0, max(width, height) * 0.8)
        ):
            return None, None, 0.0
        predicted = _clone_face(face)
        bbox = _geometry(face, "bbox")
        if bbox is not None:
            predicted.bbox = _transform_bbox(bbox, matrix)
        predicted.kps = _transform_points(points, matrix)
        landmarks = _geometry(face, "landmark_2d_106")
        if landmarks is not None:
            predicted.landmark_2d_106 = _transform_points(landmarks, matrix)
        median_error = float(np.median(error_values[good]))
        inlier_fraction = (
            float(np.mean(inliers)) if inliers is not None else 1.0
        )
        quality = float(
            np.clip(good.mean() * inlier_fraction * math.exp(-median_error / 30.0), 0, 1)
        )
        return predicted, matrix.astype(np.float32), quality

    def _select_candidate(
        self, candidates: list[Any], predicted: Any | None
    ) -> Any | None:
        if not candidates:
            return None
        if predicted is None:
            # Prefer a confident, information-rich face for initial acquisition.
            return max(
                candidates,
                key=lambda face: (
                    float(getattr(face, "det_score", 0.0) or 0.0)
                    * math.sqrt(max(1.0, _face_size(face)[0] * _face_size(face)[1]))
                ),
            )
        predicted_bbox = _geometry(predicted, "bbox")
        predicted_center = _face_center(predicted)
        predicted_width, predicted_height = _face_size(predicted)
        normalizer = max(1.0, predicted_width, predicted_height)

        def match_score(face: Any) -> float:
            overlap = bbox_iou(predicted_bbox, _geometry(face, "bbox"))
            distance = float(np.linalg.norm(_face_center(face) - predicted_center))
            proximity = math.exp(-distance / normalizer)
            confidence = float(getattr(face, "det_score", 0.0) or 0.0)
            return 0.65 * overlap + 0.25 * proximity + 0.10 * confidence

        selected = max(candidates, key=match_score)
        overlap = bbox_iou(predicted_bbox, _geometry(selected, "bbox"))
        distance = float(np.linalg.norm(_face_center(selected) - predicted_center))
        if overlap < 0.03 and distance > normalizer * 0.75:
            # Do not jump to a different person on one ambiguous detector
            # frame. In single-face mode, however, accept the only remaining
            # candidate on the next detector frame rather than rejecting that
            # same face forever after a large camera movement.
            if len(candidates) == 1 and self.consecutive_misses >= 1:
                self.reacquisitions += 1
                return selected
            self.association_rejections += 1
            return None
        if overlap < 0.15:
            self.reacquisitions += 1
        return selected

    def _reconcile(self, predicted: Any | None, detected: Any, smoothing: float) -> Any:
        if predicted is None or smoothing <= 0.0:
            self.last_bbox_iou = 1.0
            self.last_landmark_correction_percent = 0.0
            self.last_scale_correction_percent = 0.0
            self.last_rotation_correction_degrees = 0.0
            return _clone_face(detected)

        predicted_bbox = _geometry(predicted, "bbox")
        detected_bbox = _geometry(detected, "bbox")
        predicted_kps = _geometry(predicted, "kps")
        detected_kps = _geometry(detected, "kps")
        if predicted_bbox is None or detected_bbox is None:
            return _clone_face(detected)

        self.last_bbox_iou = bbox_iou(predicted_bbox, detected_bbox)
        iod = _interocular_distance(detected_kps, detected_bbox)
        correction = 0.0
        if (
            predicted_kps is not None
            and detected_kps is not None
            and predicted_kps.shape == detected_kps.shape
        ):
            correction = float(
                np.median(np.linalg.norm(detected_kps - predicted_kps, axis=1))
                / iod
                * 100.0
            )
        self.last_landmark_correction_percent = correction
        self._jitter_history.append(correction)

        predicted_width, predicted_height = _face_size(predicted)
        detected_width, detected_height = _face_size(detected)
        predicted_scale = math.sqrt(max(1.0, predicted_width * predicted_height))
        detected_scale = math.sqrt(max(1.0, detected_width * detected_height))
        scale_correction = abs(detected_scale / predicted_scale - 1.0) * 100.0
        self.last_scale_correction_percent = scale_correction
        self._scale_history.append(scale_correction)

        rotation_correction = 0.0
        if predicted_kps is not None and detected_kps is not None:
            old_angle = math.degrees(
                math.atan2(
                    float(predicted_kps[1, 1] - predicted_kps[0, 1]),
                    float(predicted_kps[1, 0] - predicted_kps[0, 0]),
                )
            )
            new_angle = math.degrees(
                math.atan2(
                    float(detected_kps[1, 1] - detected_kps[0, 1]),
                    float(detected_kps[1, 0] - detected_kps[0, 0]),
                )
            )
            rotation_correction = abs((new_angle - old_angle + 180.0) % 360.0 - 180.0)
        self.last_rotation_correction_degrees = rotation_correction
        self._rotation_history.append(rotation_correction)

        # Optical flow already follows real motion.  Use more of the detector
        # only when it disagrees substantially, which keeps fast movements
        # responsive without exposing stationary detector noise.
        detector_weight = float(
            np.clip((1.0 - smoothing) + max(0.0, correction - 1.0) * 0.035, 0.12, 0.9)
        )
        tracked = _clone_face(detected)
        tracked.bbox = (
            predicted_bbox * (1.0 - detector_weight)
            + detected_bbox * detector_weight
        ).astype(np.float32)
        if predicted_kps is not None and detected_kps is not None:
            tracked.kps = (
                predicted_kps * (1.0 - detector_weight)
                + detected_kps * detector_weight
            ).astype(np.float32)
        predicted_landmarks = _geometry(predicted, "landmark_2d_106")
        detected_landmarks = _geometry(detected, "landmark_2d_106")
        if (
            predicted_landmarks is not None
            and detected_landmarks is not None
            and predicted_landmarks.shape == detected_landmarks.shape
        ):
            tracked.landmark_2d_106 = (
                predicted_landmarks * (1.0 - detector_weight)
                + detected_landmarks * detector_weight
            ).astype(np.float32)
        elif detected_landmarks is None and predicted_landmarks is not None:
            tracked.landmark_2d_106 = predicted_landmarks
        tracked.track_id = 0
        return tracked

    def _update_face_size(self, face: Any | None, shape: tuple[int, ...]) -> None:
        if face is None:
            self.last_face_width = self.last_face_height = 0.0
            self.last_face_area_percent = 0.0
            return
        width, height = _face_size(face)
        self.last_face_width = width
        self.last_face_height = height
        frame_area = max(1.0, float(shape[0] * shape[1]))
        self.last_face_area_percent = width * height / frame_area * 100.0

    def snapshot(self) -> dict[str, Any]:
        miss_rate = 100.0 * self.detection_misses / max(1, self.detection_runs)
        return {
            "active": self.active_face is not None or bool(self.many_faces),
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "swap_alpha": round(self.swap_alpha, 4),
            "frames": self.frames,
            "detection_runs": self.detection_runs,
            "detection_hits": self.detection_hits,
            "detection_misses": self.detection_misses,
            "detection_miss_percent": round(miss_rate, 3),
            "raw_detections": self.last_raw_detection_count,
            "valid_detections": self.last_valid_detection_count,
            "association_rejections": self.association_rejections,
            "consecutive_misses": self.consecutive_misses,
            "max_consecutive_misses": self.max_consecutive_misses,
            "flow_frames": self.flow_frames,
            "flow_failures": self.flow_failures,
            "hold_frames": self.hold_frames,
            "reacquisitions": self.reacquisitions,
            "swap_transitions": self.swap_transitions,
            "detection_score": round(self.last_detection_score, 4),
            "face_width_px": round(self.last_face_width, 1),
            "face_height_px": round(self.last_face_height, 1),
            "face_area_percent": round(self.last_face_area_percent, 3),
            "landmark_correction_iod_percent": round(
                self.last_landmark_correction_percent, 3
            ),
            "landmark_correction_p95_percent": round(
                _percentile(self._jitter_history), 3
            ),
            "bbox_prediction_iou": round(self.last_bbox_iou, 4),
            "scale_correction_percent": round(
                self.last_scale_correction_percent, 3
            ),
            "scale_correction_p95_percent": round(
                _percentile(self._scale_history), 3
            ),
            "rotation_correction_degrees": round(
                self.last_rotation_correction_degrees, 3
            ),
            "rotation_correction_p95_degrees": round(
                _percentile(self._rotation_history), 3
            ),
            "flow_quality_p5": round(
                _percentile(self._flow_history, 5.0), 4
            ),
        }
