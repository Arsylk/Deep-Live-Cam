from __future__ import annotations

import numpy as np

import modules.globals
from modules.quality_pipeline import QualityPipeline


def test_pipeline_reports_quality_metrics(tmp_path):
    modules.globals.quality_mode = "monitor"
    modules.globals.quality_auto_correct = False
    pipeline = QualityPipeline(str(tmp_path))
    source = np.full((720, 1280, 3), 110, dtype=np.uint8)

    for value in (110, 111, 112):
        frame = np.full_like(source, value)
        pipeline.capture_source(source)
        pipeline.process(frame, fallback=source, processing_active=True)

    metrics = pipeline.snapshot()
    assert metrics["samples"] == 1
    assert (tmp_path / "quality.json").exists()


def test_balanced_mode_bounds_whole_frame_luminance_correction():
    modules.globals.quality_mode = "balanced"
    modules.globals.quality_auto_correct = True
    pipeline = QualityPipeline()
    source = np.full((120, 160, 3), 120, dtype=np.uint8)
    output = np.full_like(source, 80)

    for _ in range(3):
        pipeline.capture_source(source)
        result = pipeline.process(
            output.copy(), fallback=source, processing_active=True
        )

    # Balanced mode is deliberately capped at +4 levels.
    assert int(result[10, 10, 0]) == 84
    assert pipeline.snapshot()["corrections"] == 1


def test_invalid_output_uses_labeled_fallback():
    modules.globals.quality_mode = "monitor"
    modules.globals.quality_auto_correct = False
    pipeline = QualityPipeline()
    source = np.full((120, 160, 3), 90, dtype=np.uint8)
    pipeline.capture_source(source)

    result = pipeline.process(None, fallback=source, processing_active=False)

    assert result.shape == source.shape
    assert pipeline.snapshot()["fallbacks"] == 1


def test_pipeline_reports_face_local_stability_metrics():
    modules.globals.quality_mode = "monitor"
    modules.globals.quality_auto_correct = False
    pipeline = QualityPipeline()
    source = np.zeros((240, 320, 3), dtype=np.uint8)
    source[60:200, 90:230] = 100
    output = source.copy()
    output[80:180, 110:210] = 125
    tracking = {
        "detection_miss_percent": 0.0,
        "landmark_correction_p95_percent": 0.5,
        "face_height_px": 140,
        "detection_score": 0.9,
    }

    for offset in (0, 1, 2):
        current = np.clip(output.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        pipeline.capture_source(source)
        pipeline.process(
            current,
            fallback=source,
            processing_active=True,
            face_bbox=np.array([90, 60, 230, 200], dtype=np.float32),
            tracking=tracking,
            swap_applied=True,
        )

    face = pipeline.snapshot()["face"]
    assert face["available"] is True
    assert face["detail_ratio"] >= 0
    assert face["luma_mismatch"] > 0
    assert face["chroma_mismatch_lab"] >= 0
    assert face["seam_ring_delta_lab"] >= 0
    assert face["mean_absolute_delta"] > 0
    assert face["core_mean_absolute_delta"] > 0
    assert face["core_changed_pixels_percent"] > 0
    assert face["swap_applied"] is True
    assert "low facial pixel density" in face["warnings"]


def test_face_effect_metric_distinguishes_invocation_from_unchanged_output():
    modules.globals.quality_mode = "monitor"
    modules.globals.quality_auto_correct = False
    pipeline = QualityPipeline()
    source = np.full((240, 320, 3), 100, dtype=np.uint8)

    for _ in range(3):
        pipeline.capture_source(source)
        pipeline.process(
            source.copy(),
            fallback=source,
            processing_active=True,
            face_bbox=np.array([90, 60, 230, 200], dtype=np.float32),
            swap_applied=True,
        )

    face = pipeline.snapshot()["face"]
    assert face["swap_applied"] is True
    assert face["core_mean_absolute_delta"] == 0.0
    assert face["core_changed_pixels_percent"] == 0.0


def test_pipeline_timing_is_separate_from_quality_gate_cost():
    pipeline = QualityPipeline()

    pipeline.record_pipeline_timing(0.020)
    pipeline.record_pipeline_timing(0.040)

    metrics = pipeline.snapshot()
    assert 20.0 < metrics["pipeline_ms"] < 40.0
    assert metrics["pipeline_ms_p95"] > 35.0
    assert metrics["quality_gate_ms"] == 0.0
