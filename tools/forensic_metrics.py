#!/usr/bin/env python3
"""Offline, detector-facing metrics for a labeled media cohort.

Scores are always interpreted as probabilities of the *manipulated* class.
This module evaluates detectors; its outputs are deliberately excluded from
the product quality/release score so the application cannot optimize itself
to hide a synthetic edit.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REQUIRED_FIELDS = {"detector", "sample_id", "label", "score"}


def _arrays(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and scores must be equally sized 1-D sequences")
    if len(y_true) < 2:
        raise ValueError("at least two labeled samples are required")
    if not np.all(np.isin(y_true, (0, 1))):
        raise ValueError("labels must contain only 0 (authentic) or 1 (manipulated)")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("scores must be finite")
    if np.any((y_score < 0.0) | (y_score > 1.0)):
        raise ValueError("scores must be manipulated-class probabilities in [0, 1]")
    if np.unique(y_true).size != 2:
        raise ValueError("both authentic and manipulated samples are required")
    return y_true, y_score


def roc_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Return tie-aware ROC AUC using the Mann-Whitney rank statistic."""
    y_true, y_score = _arrays(labels, scores)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; ties receive their average rank.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = y_true == 1
    positive_count = int(np.sum(positives))
    negative_count = len(y_true) - positive_count
    statistic = float(np.sum(ranks[positives])) - (
        positive_count * (positive_count + 1) / 2.0
    )
    return statistic / (positive_count * negative_count)


def average_precision(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> float:
    """Return non-interpolated average precision for the manipulated class."""
    y_true, y_score = _arrays(labels, scores)
    order = np.argsort(-y_score, kind="mergesort")
    sorted_labels = y_true[order]
    sorted_scores = y_score[order]
    # Evaluate only after a complete tie group so AP cannot change when equal
    # scores happen to arrive in a different row order.
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    true_positives = np.cumsum(sorted_labels)[group_ends]
    predictions = group_ends + 1
    precision = true_positives / predictions
    recall = true_positives / np.sum(sorted_labels)
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_gain * precision))


def roc_points(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> list[dict[str, float]]:
    """Return ROC operating points at every distinct threshold."""
    y_true, y_score = _arrays(labels, scores)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    points = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    for threshold in np.unique(y_score)[::-1]:
        predicted = y_score >= threshold
        true_positive = int(np.sum(predicted & (y_true == 1)))
        false_positive = int(np.sum(predicted & (y_true == 0)))
        points.append(
            {
                "threshold": float(threshold),
                "fpr": false_positive / negatives,
                "tpr": true_positive / positives,
            }
        )
    return points


def equal_error_rate(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> tuple[float, float]:
    """Return nearest discrete EER and its decision threshold."""
    points = roc_points(labels, scores)
    selected = min(points, key=lambda point: abs(point["fpr"] - (1.0 - point["tpr"])))
    false_negative_rate = 1.0 - selected["tpr"]
    return (selected["fpr"] + false_negative_rate) / 2.0, selected["threshold"]


def cdr_at_far(
    labels: Sequence[int | bool], scores: Sequence[float], far: float = 0.05
) -> tuple[float, float]:
    """Return NIST-style correct detection rate at or below the FAR bound."""
    if not 0.0 <= far <= 1.0:
        raise ValueError("far must be in [0, 1]")
    eligible = [point for point in roc_points(labels, scores) if point["fpr"] <= far]
    selected = max(eligible, key=lambda point: (point["tpr"], -point["fpr"]))
    return selected["tpr"], selected["threshold"]


def brier_score(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    y_true, y_score = _arrays(labels, scores)
    return float(np.mean(np.square(y_score - y_true)))


def expected_calibration_error(
    labels: Sequence[int | bool], scores: Sequence[float], bins: int = 10
) -> float:
    """Return equal-width expected calibration error."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    y_true, y_score = _arrays(labels, scores)
    # Score 1.0 belongs to the final bin rather than falling outside the range.
    assignments = np.minimum((y_score * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(np.sum(selected))
        if not count:
            continue
        confidence = float(np.mean(y_score[selected]))
        accuracy = float(np.mean(y_true[selected]))
        error += count / len(y_true) * abs(confidence - accuracy)
    return error


def threshold_confusion(
    labels: Sequence[int | bool], scores: Sequence[float], threshold: float = 0.5
) -> dict[str, float | int]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y_true, y_score = _arrays(labels, scores)
    predicted = y_score >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(y_true),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "precision": tp / max(1, tp + fp),
        "recall": tpr,
        "specificity": tnr,
    }


def metric_set(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    far: float = 0.05,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute the common DeepfakeBench/OpenMFC detector scorecard."""
    y_true, y_score = _arrays(labels, scores)
    eer, eer_threshold = equal_error_rate(y_true, y_score)
    cdr, cdr_threshold = cdr_at_far(y_true, y_score, far)
    confusion = threshold_confusion(y_true, y_score, threshold)
    return {
        "samples": len(y_true),
        "authentic_samples": int(np.sum(y_true == 0)),
        "manipulated_samples": int(np.sum(y_true == 1)),
        "roc_auc": roc_auc(y_true, y_score),
        "average_precision": average_precision(y_true, y_score),
        "equal_error_rate": eer,
        "equal_error_threshold": eer_threshold,
        "cdr_at_far": cdr,
        "far_limit": far,
        "cdr_threshold": cdr_threshold,
        "brier_score": brier_score(y_true, y_score),
        "expected_calibration_error": expected_calibration_error(
            y_true, y_score, calibration_bins
        ),
        "calibration_bins": calibration_bins,
        "declared_threshold": confusion,
    }


def bootstrap_intervals(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    repeats: int = 1000,
    seed: int = 2026,
    far: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Return deterministic, stratified 95% bootstrap intervals."""
    if repeats <= 0:
        return {}
    y_true, y_score = _arrays(labels, scores)
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        indices = np.concatenate(
            (
                rng.choice(positive_indices, len(positive_indices), replace=True),
                rng.choice(negative_indices, len(negative_indices), replace=True),
            )
        )
        sampled_labels = y_true[indices]
        sampled_scores = y_score[indices]
        cdr, _ = cdr_at_far(sampled_labels, sampled_scores, far)
        values["roc_auc"].append(roc_auc(sampled_labels, sampled_scores))
        values["average_precision"].append(
            average_precision(sampled_labels, sampled_scores)
        )
        values["cdr_at_far"].append(cdr)
        values["brier_score"].append(
            brier_score(sampled_labels, sampled_scores)
        )
    return {
        name: {
            "low": float(np.percentile(samples, 2.5)),
            "high": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def _round(value: Any) -> Any:
    if isinstance(value, float):
        if np.isposinf(value):
            return "infinity"
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load detector rows from JSON, a JSON ``rows`` object, or JSONL."""
    raw = path.read_text("utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("rows")
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{path}: expected a non-empty JSON row list or JSONL")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(parsed, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be an object")
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{path}: row {index} missing {', '.join(sorted(missing))}")
        normalized = dict(row)
        normalized["detector"] = str(row["detector"])
        normalized["sample_id"] = str(row["sample_id"])
        normalized["label"] = int(row["label"])
        normalized["score"] = float(row["score"])
        if normalized["label"] not in (0, 1):
            raise ValueError(f"{path}: row {index} label must be 0 or 1")
        if not 0.0 <= normalized["score"] <= 1.0:
            raise ValueError(f"{path}: row {index} score must be in [0, 1]")
        subgroups = row.get("subgroups", {})
        if not isinstance(subgroups, dict):
            raise ValueError(f"{path}: row {index} subgroups must be an object")
        normalized["subgroups"] = {
            str(key): str(value) for key, value in subgroups.items()
        }
        rows.append(normalized)
    return rows


def _cohort_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    far: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    result = metric_set(labels, scores, threshold=threshold, far=far)
    result["confidence_intervals_95"] = bootstrap_intervals(
        labels, scores, repeats=bootstrap, seed=seed, far=far
    )
    return _round(result)


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.5,
    far: float = 0.05,
    bootstrap: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate each detector and any viable declared subgroup cohorts."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["detector"])].append(row)
    if not grouped:
        raise ValueError("no detector rows supplied")
    detector_reports: dict[str, Any] = {}
    for detector, detector_rows in sorted(grouped.items()):
        report: dict[str, Any] = {
            "version": str(detector_rows[0].get("detector_version", "unreported")),
            "source": str(detector_rows[0].get("detector_source", "unreported")),
            "overall": _cohort_metrics(
                detector_rows,
                threshold=threshold,
                far=far,
                bootstrap=bootstrap,
                seed=seed,
            ),
            "subgroups": {},
        }
        subgroup_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in detector_rows:
            for key, value in row.get("subgroups", {}).items():
                subgroup_rows[f"{key}={value}"].append(row)
        for subgroup, selected in sorted(subgroup_rows.items()):
            if {row["label"] for row in selected} != {0, 1}:
                report["subgroups"][subgroup] = {
                    "samples": len(selected),
                    "available": False,
                    "reason": "both classes are required",
                }
                continue
            report["subgroups"][subgroup] = {
                "available": True,
                **_cohort_metrics(
                    selected,
                    threshold=threshold,
                    far=far,
                    bootstrap=bootstrap,
                    seed=seed,
                ),
            }
        detector_reports[detector] = report
    return {
        "schema_version": "1.0",
        "score_semantics": "probability that the sample is manipulated",
        "purpose": "detector evaluation and forensic exposure reporting",
        "optimization_warning": (
            "These metrics are excluded from the product quality grade. "
            "Do not tune media generation to minimize detector scores."
        ),
        "detectors": detector_reports,
    }
