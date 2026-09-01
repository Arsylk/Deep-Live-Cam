#!/usr/bin/env python3
"""Compare two Windows quality-history JSONL captures.

Run one controlled clip with the baseline settings, save/rename
``runtime/network-live/quality-history.jsonl``, repeat with the candidate
settings, then pass both files here.  Lower-is-better and higher-is-better
metrics are handled explicitly so the report cannot accidentally celebrate a
larger miss or flicker value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np


MetricPath = tuple[str, ...]


METRICS: dict[str, tuple[MetricPath, str]] = {
    "whole_frame_score": (("score",), "higher"),
    "face_stability_score": (("face", "stability_score"), "higher"),
    "face_excess_flicker": (("face", "excess_flicker"), "lower"),
    "face_excess_flicker_p95": (("face", "excess_flicker_p95"), "lower"),
    "seam_temporal_std": (("face", "seam_temporal_std"), "lower"),
    "seam_ring_delta_lab": (("face", "seam_ring_delta_lab"), "lower"),
    "face_luma_mismatch": (("face", "luma_mismatch"), "lower"),
    "face_chroma_mismatch_lab": (
        ("face", "chroma_mismatch_lab"),
        "lower",
    ),
    "detail_ratio_error": (("face", "detail_ratio"), "one"),
    "detection_miss_percent": (("tracking", "detection_miss_percent"), "lower"),
    "landmark_correction_p95_percent": (
        ("tracking", "landmark_correction_p95_percent"),
        "lower",
    ),
    "scale_correction_p95_percent": (
        ("tracking", "scale_correction_p95_percent"),
        "lower",
    ),
    "rotation_correction_p95_degrees": (
        ("tracking", "rotation_correction_p95_degrees"),
        "lower",
    ),
    "pipeline_ms": (("pipeline_ms",), "lower"),
    "pipeline_ms_p95": (("pipeline_ms_p95",), "lower"),
}


def nested_number(record: dict[str, Any], path: MetricPath) -> float | None:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def read_history(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no metric records")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"records": len(records)}
    for name, (path, direction) in METRICS.items():
        values = [
            value
            for record in records
            if (value := nested_number(record, path)) is not None
        ]
        if direction == "one":
            values = [abs(value - 1.0) for value in values]
        if not values:
            continue
        summary[name] = {
            "samples": len(values),
            "mean": round(fmean(values), 5),
            "p50": round(float(np.percentile(values, 50)), 5),
            "p95": round(float(np.percentile(values, 95)), 5),
            "minimum": round(min(values), 5),
            "maximum": round(max(values), 5),
            "preferred": direction,
        }
    return summary


def compare_metric(
    baseline: dict[str, Any], candidate: dict[str, Any], direction: str
) -> dict[str, Any]:
    baseline_value = float(baseline["mean"])
    candidate_value = float(candidate["mean"])
    if direction == "higher":
        signed_gain = candidate_value - baseline_value
        denominator = max(1e-6, abs(baseline_value))
    else:
        signed_gain = baseline_value - candidate_value
        denominator = max(1e-6, abs(baseline_value))
    gain_percent = 100.0 * signed_gain / denominator
    tolerance = max(0.001, abs(baseline_value) * 0.01)
    verdict = "improved" if signed_gain > tolerance else (
        "regressed" if signed_gain < -tolerance else "unchanged"
    )
    return {
        "baseline_mean": baseline_value,
        "candidate_mean": candidate_value,
        "gain_percent": round(gain_percent, 3),
        "verdict": verdict,
    }


def compare(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = summarize(baseline_records)
    candidate = summarize(candidate_records)
    changes: dict[str, Any] = {}
    for name, (_path, direction) in METRICS.items():
        if name in baseline and name in candidate:
            changes[name] = compare_metric(
                baseline[name], candidate[name], direction
            )
    improved = sum(value["verdict"] == "improved" for value in changes.values())
    regressed = sum(value["verdict"] == "regressed" for value in changes.values())
    return {
        "baseline": baseline,
        "candidate": candidate,
        "changes": changes,
        "verdict": {
            "improved_metrics": improved,
            "regressed_metrics": regressed,
            "accepted": improved > regressed and regressed == 0,
            "rule": "candidate must improve more metrics than it regresses and regress none",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(read_history(args.baseline), read_history(args.candidate))
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0 if result["verdict"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
