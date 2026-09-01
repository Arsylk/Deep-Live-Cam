"""Labeled technical-quality gate for the network live pipeline.

This module deliberately limits automation to whole-frame signal integrity:
format validation, conservative luminance matching, freeze/clipping/detail
measurements.  It does not
perform face-local restoration or conceal manipulation-specific artifacts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import modules.globals


class QualityPipeline:
    """Analyze every frame and apply bounded, globally scoped corrections."""

    SAMPLE_SIZE = (160, 90)
    VALID_MODES = {"monitor", "balanced", "strict"}

    def __init__(self, state_dir: str | None = None) -> None:
        self._lock = threading.Lock()
        self._frame_number = 0
        self._samples = 0
        self._corrections = 0
        self._fallbacks = 0
        self._repeat_streak = 0
        self._previous_output: np.ndarray | None = None
        self._previous_face_output: np.ndarray | None = None
        self._previous_face_source: np.ndarray | None = None
        self._face_flicker_history: list[float] = []
        self._seam_history: list[float] = []
        self._pipeline_timing_history: list[float] = []
        self._last_swap_applied = False
        self._swap_transitions = 0
        self._source_gray: np.ndarray | None = None
        self._source_luma = 0.0
        self._source_delta = 0.0
        self._last_correction_delta = 0
        self._last_report_at = 0.0
        self._report_path = (
            Path(state_dir) / "quality.json" if state_dir else None
        )
        self._history_path = (
            Path(state_dir) / "quality-history.jsonl" if state_dir else None
        )
        self._metrics: dict[str, Any] = {
            "mode": self.mode,
            "automatic_correction": self.automatic_correction,
            "score": 0.0,
            "samples": 0,
            "corrections": 0,
            "fallbacks": 0,
            "pipeline_ms": 0.0,
            "pipeline_ms_p95": 0.0,
            "quality_gate_ms": 0.0,
            "warnings": ["collecting samples"],
        }

    @property
    def mode(self) -> str:
        requested = str(
            getattr(modules.globals, "quality_mode", "balanced")
        ).lower()
        return requested if requested in self.VALID_MODES else "balanced"

    @property
    def automatic_correction(self) -> bool:
        return bool(
            getattr(modules.globals, "quality_auto_correct", True)
        )

    def capture_source(self, frame: np.ndarray) -> None:
        """Capture a tiny pre-processing reference without copying a frame."""
        # ``process`` increments the frame number later in the same worker
        # iteration. Sample both sides on the same every-third-frame cadence.
        if (self._frame_number + 1) % 3 != 0:
            return
        if not self._valid_frame(frame):
            self._source_gray = None
            return
        sample = cv2.resize(frame, self.SAMPLE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        if self._source_gray is not None:
            self._source_delta = float(
                np.mean(
                    np.abs(
                        gray.astype(np.int16)
                        - self._source_gray.astype(np.int16)
                    )
                )
            )
        self._source_gray = gray
        self._source_luma = float(gray.mean())

    def process(
        self,
        frame: np.ndarray,
        fallback: np.ndarray | None = None,
        processing_active: bool = True,
        face_bbox: np.ndarray | None = None,
        tracking: dict[str, Any] | None = None,
        swap_applied: bool = False,
    ) -> np.ndarray:
        """Return a valid frame and update rolling diagnostics."""
        started = time.perf_counter()
        self._frame_number += 1
        used_fallback = False

        if not self._valid_frame(frame):
            if not self._valid_frame(fallback):
                raise ValueError("quality gate received no valid frame")
            frame = fallback
            used_fallback = True
            self._fallbacks += 1

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            self._corrections += 1
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
            self._corrections += 1

        # Match only global luminance, with a deliberately small bound. This
        # repairs accidental whole-frame level shifts without touching local
        # facial texture or geometry.
        applied_delta = 0
        sample: np.ndarray | None = None
        sample_due = self._frame_number % 3 == 0
        if sample_due:
            sample = cv2.resize(
                frame, self.SAMPLE_SIZE, interpolation=cv2.INTER_AREA
            )
        if processing_active and self.automatic_correction and self.mode != "monitor":
            if sample is not None:
                output_luma = float(
                    cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).mean()
                )
                limit = 4 if self.mode == "balanced" else 8
                self._last_correction_delta = int(
                    round(
                        np.clip(
                            self._source_luma - output_luma, -limit, limit
                        )
                    )
                )
            delta = self._last_correction_delta
            if abs(delta) >= 2:
                lookup = np.clip(
                    np.arange(256, dtype=np.int16) + delta, 0, 255
                ).astype(np.uint8)
                frame = cv2.LUT(frame, lookup)
                if sample is not None:
                    sample = cv2.LUT(sample, lookup)
                applied_delta = delta
                self._corrections += 1
        else:
            self._last_correction_delta = 0

        if sample_due:
            self._measure(
                frame,
                applied_delta,
                used_fallback,
                sample,
                source_frame=fallback,
                face_bbox=face_bbox,
                tracking=tracking,
                swap_applied=swap_applied,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            previous = float(self._metrics.get("quality_gate_ms", 0.0))
            self._metrics["quality_gate_ms"] = (
                elapsed_ms if previous == 0.0
                else previous * 0.9 + elapsed_ms * 0.1
            )
            self._metrics["mode"] = self.mode
            self._metrics["automatic_correction"] = self.automatic_correction
            self._metrics["processing_active"] = processing_active
            self._metrics["corrections"] = self._corrections
            self._metrics["fallbacks"] = self._fallbacks
            self._metrics["tracking"] = dict(tracking or {})
            self._metrics["swap_applied"] = bool(swap_applied)
        self._write_report_if_due()
        return frame

    def record_pipeline_timing(self, elapsed_seconds: float) -> None:
        """Record detector-to-output latency separately from gate overhead."""
        elapsed_ms = max(0.0, float(elapsed_seconds) * 1000.0)
        self._pipeline_timing_history.append(elapsed_ms)
        self._pipeline_timing_history = self._pipeline_timing_history[-600:]
        with self._lock:
            previous = float(self._metrics.get("pipeline_ms", 0.0))
            self._metrics["pipeline_ms"] = (
                elapsed_ms
                if previous == 0.0
                else previous * 0.9 + elapsed_ms * 0.1
            )
            self._metrics["pipeline_ms_p95"] = float(
                np.percentile(self._pipeline_timing_history, 95)
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metrics)

    def reset_metrics_window(self) -> None:
        """Start a clean diagnostic window without changing video output."""
        with self._lock:
            self._samples = 0
            self._corrections = 0
            self._fallbacks = 0
            self._repeat_streak = 0
            self._previous_output = None
            self._previous_face_output = None
            self._previous_face_source = None
            self._face_flicker_history.clear()
            self._seam_history.clear()
            self._pipeline_timing_history.clear()
            self._last_swap_applied = False
            self._swap_transitions = 0
            # Preserve the source luminance and last bounded correction: they
            # affect rendering. Only diagnostic aggregates are reset.
            self._metrics = {
                "mode": self.mode,
                "automatic_correction": self.automatic_correction,
                "score": 0.0,
                "samples": 0,
                "corrections": 0,
                "fallbacks": 0,
                "pipeline_ms": 0.0,
                "pipeline_ms_p95": 0.0,
                "quality_gate_ms": 0.0,
                "warnings": ["collecting benchmark samples"],
            }

    @staticmethod
    def _valid_frame(frame: Any) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[0] >= 64
            and frame.shape[1] >= 64
            and frame.shape[2] == 3
            and frame.size > 0
        )

    def _measure(
        self, frame: np.ndarray, applied_delta: int, used_fallback: bool,
        sample: np.ndarray | None = None,
        *,
        source_frame: np.ndarray | None = None,
        face_bbox: np.ndarray | None = None,
        tracking: dict[str, Any] | None = None,
        swap_applied: bool = False,
    ) -> None:
        if sample is None:
            sample = cv2.resize(
                frame, self.SAMPLE_SIZE, interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        luma = float(gray.mean())
        saturation = float(hsv[:, :, 1].mean())
        black_clip = 100.0 * float(np.mean(gray <= 3))
        white_clip = 100.0 * float(np.mean(gray >= 252))
        detail = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        temporal_delta = 0.0
        if self._previous_output is not None:
            temporal_delta = float(
                np.mean(
                    np.abs(
                        gray.astype(np.int16)
                        - self._previous_output.astype(np.int16)
                    )
                )
            )
            if temporal_delta < 0.08 and self._source_delta > 0.8:
                self._repeat_streak += 1
            else:
                self._repeat_streak = 0
        self._previous_output = gray

        warnings: list[str] = []
        score = 100.0
        if luma < 28:
            warnings.append("underexposed")
            score -= min(25.0, 28.0 - luma)
        elif luma > 225:
            warnings.append("overexposed")
            score -= min(25.0, luma - 225.0)
        if black_clip > 8.0:
            warnings.append("shadow clipping")
            score -= min(20.0, black_clip - 8.0)
        if white_clip > 5.0:
            warnings.append("highlight clipping")
            score -= min(20.0, white_clip - 5.0)
        if detail < 18.0:
            warnings.append("low detail")
            score -= min(20.0, (18.0 - detail) * 0.8)
        if self._repeat_streak >= 3:
            warnings.append("output freeze suspected")
            score -= min(30.0, self._repeat_streak * 3.0)
        if used_fallback:
            warnings.append("invalid frame replaced")
            score -= 15.0

        face_quality = self._measure_face(
            frame,
            source_frame,
            face_bbox,
            tracking or {},
            swap_applied,
        )
        face_warnings = face_quality.get("warnings", [])
        warnings.extend(
            warning for warning in face_warnings if warning not in warnings
        )

        self._samples += 1
        values = {
            "score": round(max(0.0, score), 1),
            "samples": self._samples,
            "luma": round(luma, 2),
            "saturation": round(saturation, 2),
            "black_clip_percent": round(black_clip, 3),
            "white_clip_percent": round(white_clip, 3),
            "detail_laplacian": round(detail, 2),
            "temporal_delta": round(temporal_delta, 3),
            "source_delta": round(self._source_delta, 3),
            "repeat_streak": self._repeat_streak,
            "applied_luma_delta": applied_delta,
            "warnings": warnings,
            "face": face_quality,
            "updated_at": time.time(),
        }
        with self._lock:
            self._metrics.update(values)

    @staticmethod
    def _face_crop(
        frame: np.ndarray | None, bbox: np.ndarray | None
    ) -> np.ndarray | None:
        if not QualityPipeline._valid_frame(frame) or bbox is None:
            return None
        values = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if values.size != 4 or not np.isfinite(values).all():
            return None
        x1, y1, x2, y2 = values
        width, height = x2 - x1, y2 - y1
        if width < 16 or height < 16:
            return None
        margin_x, margin_y = width * 0.12, height * 0.12
        frame_height, frame_width = frame.shape[:2]
        left = max(0, int(np.floor(x1 - margin_x)))
        top = max(0, int(np.floor(y1 - margin_y)))
        right = min(frame_width, int(np.ceil(x2 + margin_x)))
        bottom = min(frame_height, int(np.ceil(y2 + margin_y)))
        if right - left < 16 or bottom - top < 16:
            return None
        return cv2.resize(
            frame[top:bottom, left:right],
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )

    def _measure_face(
        self,
        output: np.ndarray,
        source: np.ndarray | None,
        bbox: np.ndarray | None,
        tracking: dict[str, Any],
        swap_applied: bool,
    ) -> dict[str, Any]:
        output_crop = self._face_crop(output, bbox)
        source_crop = self._face_crop(source, bbox)
        if output_crop is None or source_crop is None:
            self._previous_face_output = None
            self._previous_face_source = None
            if self._last_swap_applied != bool(swap_applied):
                self._swap_transitions += 1
            self._last_swap_applied = bool(swap_applied)
            return {
                "available": False,
                "stability_score": 0.0,
                "swap_transitions": self._swap_transitions,
                "warnings": ["no measurable face"],
            }

        output_gray = cv2.cvtColor(output_crop, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
        output_delta = 0.0
        source_delta = 0.0
        if self._previous_face_output is not None:
            output_delta = float(
                np.mean(
                    np.abs(
                        output_gray.astype(np.int16)
                        - self._previous_face_output.astype(np.int16)
                    )
                )
            )
        if self._previous_face_source is not None:
            source_delta = float(
                np.mean(
                    np.abs(
                        source_gray.astype(np.int16)
                        - self._previous_face_source.astype(np.int16)
                    )
                )
            )
        self._previous_face_output = output_gray
        self._previous_face_source = source_gray
        excess_flicker = max(0.0, output_delta - source_delta)
        flicker_ratio = output_delta / max(0.5, source_delta)
        if output_delta or source_delta:
            self._face_flicker_history.append(excess_flicker)
            self._face_flicker_history = self._face_flicker_history[-200:]

        output_detail = float(cv2.Laplacian(output_gray, cv2.CV_32F).var())
        source_detail = float(cv2.Laplacian(source_gray, cv2.CV_32F).var())
        detail_ratio = output_detail / max(0.1, source_detail)
        luma_mismatch = abs(float(output_gray.mean()) - float(source_gray.mean()))
        output_lab = cv2.cvtColor(output_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        source_lab = cv2.cvtColor(source_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma_mismatch = float(
            np.linalg.norm(
                output_lab[:, :, 1:3].mean(axis=(0, 1))
                - source_lab[:, :, 1:3].mean(axis=(0, 1))
            )
        )

        # The intended identity change dominates the crop core.  A thin
        # elliptical ring near the paste boundary is a more useful seam proxy.
        difference = np.mean(
            np.abs(
                output_crop.astype(np.int16) - source_crop.astype(np.int16)
            ),
            axis=2,
        )
        yy, xx = np.ogrid[-1.0:1.0:128j, -1.0:1.0:128j]
        radius = np.sqrt((xx / 0.92) ** 2 + (yy / 0.98) ** 2)
        # ``swap_applied`` only means that the processor was invoked for a
        # detected face.  Measure the actual face-local pixel effect as well,
        # otherwise an identity-like/untrained checkpoint can be reported as a
        # successful swap while merely drawing its fusion mask.  Keep the core
        # separate from the boundary so a bright paste ring cannot satisfy the
        # effect check by itself.
        core = radius <= 0.62
        face_mean_delta = float(difference.mean())
        core_mean_delta = float(difference[core].mean())
        core_changed_percent = 100.0 * float(np.mean(difference[core] >= 2.0))
        ring = (radius >= 0.72) & (radius <= 0.90)
        seam_delta = float(difference[ring].mean())
        seam_lab_delta = float(
            np.linalg.norm(output_lab[ring] - source_lab[ring], axis=1).mean()
        )
        self._seam_history.append(seam_delta)
        self._seam_history = self._seam_history[-200:]
        seam_std = float(np.std(self._seam_history))

        if self._last_swap_applied != bool(swap_applied):
            self._swap_transitions += 1
        self._last_swap_applied = bool(swap_applied)

        miss_rate = float(tracking.get("detection_miss_percent", 0.0) or 0.0)
        jitter_p95 = float(
            tracking.get("landmark_correction_p95_percent", 0.0) or 0.0
        )
        face_height = float(tracking.get("face_height_px", 0.0) or 0.0)
        detection_score = float(tracking.get("detection_score", 0.0) or 0.0)
        score = 100.0
        warnings: list[str] = []
        if face_height and face_height < 180:
            warnings.append("low facial pixel density")
            score -= min(20.0, (180.0 - face_height) / 6.0)
        if miss_rate > 0.5:
            warnings.append("face detection misses")
            score -= min(25.0, (miss_rate - 0.5) * 3.0)
        if detection_score and detection_score < 0.65:
            warnings.append("low detection confidence")
            score -= min(15.0, (0.65 - detection_score) * 50.0)
        if jitter_p95 > 1.0:
            warnings.append("landmark correction jitter")
            score -= min(20.0, (jitter_p95 - 1.0) * 4.0)
        if excess_flicker > 1.5:
            warnings.append("excess face-region flicker")
            score -= min(20.0, (excess_flicker - 1.5) * 3.0)
        if seam_std > 3.0:
            warnings.append("unstable face boundary")
            score -= min(15.0, (seam_std - 3.0) * 2.0)
        if not 0.65 <= detail_ratio <= 1.45:
            warnings.append("face detail mismatch")
            score -= min(15.0, abs(np.log(max(0.01, detail_ratio))) * 12.0)
        if luma_mismatch > 8.0:
            warnings.append("face illumination mismatch")
            score -= min(12.0, (luma_mismatch - 8.0) * 0.6)
        if chroma_mismatch > 7.0:
            warnings.append("face chroma mismatch")
            score -= min(12.0, (chroma_mismatch - 7.0) * 0.8)

        return {
            "available": True,
            "stability_score": round(max(0.0, score), 1),
            "source_luma": round(float(source_gray.mean()), 2),
            "output_luma": round(float(output_gray.mean()), 2),
            "source_detail_laplacian": round(source_detail, 2),
            "output_detail_laplacian": round(output_detail, 2),
            "detail_ratio": round(detail_ratio, 4),
            "luma_mismatch": round(luma_mismatch, 3),
            "chroma_mismatch_lab": round(chroma_mismatch, 3),
            "source_temporal_delta": round(source_delta, 3),
            "output_temporal_delta": round(output_delta, 3),
            "excess_flicker": round(excess_flicker, 3),
            "flicker_ratio": round(flicker_ratio, 3),
            "mean_absolute_delta": round(face_mean_delta, 3),
            "core_mean_absolute_delta": round(core_mean_delta, 3),
            "core_changed_pixels_percent": round(core_changed_percent, 3),
            "excess_flicker_p95": round(
                float(np.percentile(self._face_flicker_history, 95))
                if self._face_flicker_history
                else 0.0,
                3,
            ),
            "seam_ring_delta": round(seam_delta, 3),
            "seam_ring_delta_lab": round(seam_lab_delta, 3),
            "seam_temporal_std": round(seam_std, 3),
            "swap_applied": bool(swap_applied),
            "swap_transitions": self._swap_transitions,
            "warnings": warnings,
        }

    def _write_report_if_due(self) -> None:
        if self._report_path is None:
            return
        now = time.monotonic()
        if now - self._last_report_at < 5.0:
            return
        self._last_report_at = now
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._report_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.snapshot(), indent=2), encoding="utf-8"
            )
            os.replace(temporary, self._report_path)
            if self._history_path is not None:
                # Keep a compact rolling history for controlled A/B runs.  The
                # five-second cadence is enough for trends without turning
                # diagnostics into a material processing or disk-I/O cost.
                if (
                    self._history_path.exists()
                    and self._history_path.stat().st_size > 2 * 1024 * 1024
                ):
                    rotated = self._history_path.with_suffix(".previous.jsonl")
                    os.replace(self._history_path, rotated)
                record = self.snapshot()
                record["reported_at"] = time.time()
                with self._history_path.open("a", encoding="utf-8") as history:
                    history.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            with self._lock:
                self._metrics["report_error"] = str(exc)
