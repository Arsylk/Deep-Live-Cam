from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.subjective_metrics import evaluate_subjective_rows


def test_subjective_report_excludes_failed_checks_and_computes_paired_dmos():
    rows = [
        {
            "sample_id": "clip",
            "rater_id": "r1",
            "condition": "reference",
            "ratings": {"overall": 5, "naturalness": 5},
        },
        {
            "sample_id": "clip",
            "rater_id": "r1",
            "condition": "processed",
            "ratings": {"overall": 4, "naturalness": 3},
        },
        {
            "sample_id": "clip",
            "rater_id": "failed",
            "condition": "processed",
            "attention_check_passed": False,
            "ratings": {"overall": 1, "naturalness": 1},
        },
    ]

    report = evaluate_subjective_rows(rows)

    assert report["rows_included"] == 2
    assert report["rows_excluded_attention_check"] == 1
    assert report["samples"]["clip"]["processed"]["overall"][
        "mean_opinion_score"
    ] == 4.0
    assert report["paired_degradation"]["overall"]["mean_degradation"] == 1.0
    assert report["paired_degradation"]["naturalness"]["mean_degradation"] == 2.0

