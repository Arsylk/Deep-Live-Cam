from __future__ import annotations

from tools.stability_report import compare, summarize


def record(score: float, flicker: float, misses: float, jitter: float) -> dict:
    return {
        "score": score,
        "pipeline_ms": 3.0,
        "face": {
            "stability_score": score,
            "excess_flicker": flicker,
            "excess_flicker_p95": flicker * 1.2,
            "seam_temporal_std": flicker / 2,
            "detail_ratio": 1.0,
        },
        "tracking": {
            "detection_miss_percent": misses,
            "landmark_correction_p95_percent": jitter,
            "scale_correction_p95_percent": jitter,
            "rotation_correction_p95_degrees": jitter,
        },
    }


def test_summary_treats_detail_ratio_as_distance_from_one():
    result = summarize([record(90, 2, 1, 1) | {"face": {"detail_ratio": 1.2}}])
    assert result["detail_ratio_error"]["mean"] == 0.2


def test_comparison_accepts_uniform_quality_gain():
    baseline = [record(70, 4, 3, 2), record(72, 3, 2, 2)]
    candidate = [record(88, 1, 0.2, 0.5), record(90, 1, 0.1, 0.4)]

    result = compare(baseline, candidate)

    assert result["changes"]["face_stability_score"]["verdict"] == "improved"
    assert result["changes"]["face_excess_flicker"]["verdict"] == "improved"
    assert result["verdict"]["accepted"] is True
