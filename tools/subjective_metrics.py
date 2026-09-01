#!/usr/bin/env python3
"""Offline aggregation for consented, blinded video-quality panel ratings."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


REQUIRED = {"sample_id", "rater_id", "condition", "ratings"}


def load_subjective_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text("utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("rows")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a non-empty JSON list or JSONL")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be an object")
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"{path}: row {index} missing {', '.join(sorted(missing))}")
        ratings = row["ratings"]
        if not isinstance(ratings, dict) or not ratings:
            raise ValueError(f"{path}: row {index} ratings must be a non-empty object")
        normalized_ratings: dict[str, float] = {}
        for dimension, score in ratings.items():
            number = float(score)
            if not 1.0 <= number <= 5.0:
                raise ValueError(
                    f"{path}: row {index} rating {dimension} must be in [1, 5]"
                )
            normalized_ratings[str(dimension)] = number
        rows.append(
            {
                **row,
                "sample_id": str(row["sample_id"]),
                "rater_id": str(row["rater_id"]),
                "condition": str(row["condition"]),
                "ratings": normalized_ratings,
                "attention_check_passed": row.get("attention_check_passed") is not False,
            }
        )
    return rows


def _summary(values: list[float]) -> dict[str, Any]:
    mean = fmean(values)
    standard_deviation = stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return {
        "ratings": len(values),
        "mean_opinion_score": round(mean, 4),
        "standard_deviation": round(standard_deviation, 4),
        "confidence_interval_95": [
            round(max(1.0, mean - half_width), 4),
            round(min(5.0, mean + half_width), 4),
        ],
    }


def evaluate_subjective_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ACR-style ratings and paired degradation scores.

    This is analysis support, not by itself a claim of ITU compliance. A study
    still needs the viewing environment, randomization, training, and exclusion
    protocol described in its preregistration.
    """
    supplied = list(rows)
    included = [row for row in supplied if row.get("attention_check_passed") is not False]
    if not included:
        raise ValueError("no subjective ratings remain after attention checks")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    dimensions: set[str] = set()
    for row in included:
        for dimension, value in row["ratings"].items():
            dimensions.add(dimension)
            grouped[(row["sample_id"], row["condition"], dimension)].append(
                float(value)
            )
    samples: dict[str, Any] = defaultdict(dict)
    for (sample_id, condition, dimension), values in sorted(grouped.items()):
        samples[sample_id].setdefault(condition, {})[dimension] = _summary(values)

    paired_differences: dict[str, list[float]] = defaultdict(list)
    by_rater: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in included:
        by_rater[(row["sample_id"], row["rater_id"])][row["condition"]] = row[
            "ratings"
        ]
    for conditions in by_rater.values():
        reference = conditions.get("reference")
        processed = conditions.get("processed")
        if not reference or not processed:
            continue
        for dimension in reference.keys() & processed.keys():
            # Positive DMOS here means the processed clip was rated worse.
            paired_differences[dimension].append(
                float(reference[dimension]) - float(processed[dimension])
            )
    dmos = {
        dimension: {
            **_summary([value + 3.0 for value in values]),
            "mean_degradation": round(fmean(values), 4),
            "interpretation": "positive means processed was rated lower",
        }
        for dimension, values in sorted(paired_differences.items())
        if values
    }
    # Remove the shifted MOS fields: the helper is used only to obtain the
    # standard error/interval width for degradation values.
    for dimension, values in paired_differences.items():
        if dimension not in dmos or not values:
            continue
        deviation = stdev(values) if len(values) > 1 else 0.0
        half_width = 1.96 * deviation / math.sqrt(len(values))
        dmos[dimension] = {
            "paired_ratings": len(values),
            "mean_degradation": round(fmean(values), 4),
            "confidence_interval_95": [
                round(fmean(values) - half_width, 4),
                round(fmean(values) + half_width, 4),
            ],
            "interpretation": "positive means processed was rated lower",
        }
    return {
        "schema_version": "1.0",
        "scale": "1 (bad) to 5 (excellent)",
        "method": "offline ACR-style aggregation",
        "compliance_note": (
            "Aggregation alone does not establish ITU-T P.910/P.918 compliance; "
            "follow the preregistered viewing and randomization protocol."
        ),
        "rows_supplied": len(supplied),
        "rows_included": len(included),
        "rows_excluded_attention_check": len(supplied) - len(included),
        "unique_raters": len({row["rater_id"] for row in included}),
        "dimensions": sorted(dimensions),
        "samples": dict(samples),
        "paired_degradation": dmos,
    }

