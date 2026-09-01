from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.forensic_metrics import (
    average_precision,
    brier_score,
    bootstrap_intervals,
    cdr_at_far,
    equal_error_rate,
    evaluate_rows,
    expected_calibration_error,
    roc_auc,
)


def test_perfect_detector_has_expected_forensic_metrics():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert roc_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    assert equal_error_rate(labels, scores)[0] == 0.0
    assert cdr_at_far(labels, scores, 0.05)[0] == 1.0
    assert brier_score(labels, scores) == pytest.approx(0.025)


def test_tied_uninformative_scores_have_chance_auc():
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]

    assert roc_auc(labels, scores) == 0.5
    assert average_precision(labels, scores) == 0.5
    assert expected_calibration_error(labels, scores, 10) == 0.0


def test_bootstrap_is_deterministic_and_stratified():
    first = bootstrap_intervals(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], repeats=20, seed=7
    )
    second = bootstrap_intervals(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], repeats=20, seed=7
    )

    assert first == second
    assert first["roc_auc"] == {"low": 1.0, "high": 1.0}


def test_detector_report_keeps_metrics_out_of_product_grade():
    rows = [
        {
            "detector": "fixture",
            "sample_id": "a0",
            "label": 0,
            "score": 0.1,
            "subgroups": {"device": "phone"},
        },
        {
            "detector": "fixture",
            "sample_id": "a1",
            "label": 1,
            "score": 0.9,
            "subgroups": {"device": "phone"},
        },
        {
            "detector": "fixture",
            "sample_id": "b0",
            "label": 0,
            "score": 0.2,
            "subgroups": {"device": "webcam"},
        },
        {
            "detector": "fixture",
            "sample_id": "b1",
            "label": 1,
            "score": 0.8,
            "subgroups": {"device": "webcam"},
        },
    ]

    report = evaluate_rows(rows, bootstrap=10)

    assert report["purpose"] == "detector evaluation and forensic exposure reporting"
    assert "Do not tune" in report["optimization_warning"]
    assert report["detectors"]["fixture"]["overall"]["roc_auc"] == 1.0
    assert report["detectors"]["fixture"]["subgroups"]["device=phone"][
        "available"
    ] is True


def test_forensic_metrics_reject_single_class_cohort():
    with pytest.raises(ValueError, match="both authentic and manipulated"):
        roc_auc([1, 1], [0.8, 0.9])
