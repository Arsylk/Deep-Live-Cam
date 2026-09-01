#!/usr/bin/env python3
"""Record and compare immutable live face-swap pipeline baselines.

All capture and analysis is local. The recorder is a passive in-process tap;
this command never opens a camera, SRT listener, or preview UDP port.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modules.pipeline_benchmark import validate_request  # noqa: E402
from tools.media_integrity_benchmark import evaluate_pair  # noqa: E402
from tools.stability_report import read_history, summarize  # noqa: E402


SCHEMA_VERSION = "1.0"
DEFAULT_STATE_DIR = (
    REPOSITORY_ROOT / "arch-linux" / "runtime" / "android-phone-processed"
)

METRICS: dict[str, tuple[str, str]] = {
    "face_stability_score": ("higher", "face quality"),
    "face_excess_flicker_p95": ("lower", "face quality"),
    "seam_temporal_std": ("lower", "face quality"),
    "seam_ring_delta_lab": ("lower", "face quality"),
    "face_luma_mismatch": ("lower", "face quality"),
    "face_chroma_mismatch_lab": ("lower", "face quality"),
    "detail_ratio_error": ("lower", "face quality"),
    "detection_miss_percent": ("lower", "tracking"),
    "landmark_correction_p95_percent": ("lower", "tracking"),
    "pipeline_ms": ("lower", "performance"),
    "pipeline_ms_p95": ("lower", "performance"),
    "processing_fps": ("higher", "performance"),
    "vmaf": ("higher", "signal preservation"),
    "ssim": ("higher", "signal preservation"),
    "psnr": ("higher", "signal preservation"),
    "processed_exact_repeat_percent": ("lower", "cadence"),
    "processed_blockiness_ratio": ("lower", "signal preservation"),
}

CRITICAL_QUALITY_METRICS = {
    "face_stability_score",
    "face_excess_flicker_p95",
    "seam_temporal_std",
    "seam_ring_delta_lab",
    "detail_ratio_error",
    "detection_miss_percent",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = completed.stdout.strip()
    return text if completed.returncode == 0 and text else None


def repository_state(root: Path) -> dict[str, Any]:
    commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = command_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"]
    )
    status = status or ""
    return {
        "commit": commit,
        "working_tree_dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status_entries": len(status.splitlines()),
    }


def fingerprint_artifact(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser()
    result: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    else:
        result["missing"] = True
    return result


def artifact_inventory(context: dict[str, Any]) -> dict[str, Any]:
    source = context.get("source_artifact")
    model_paths = context.get("model_artifacts", [])
    code_paths = context.get("code_artifacts", [])
    root_value = context.get("repository_root") or REPOSITORY_ROOT
    return {
        "source": fingerprint_artifact(source) if source else None,
        "models": [fingerprint_artifact(path) for path in model_paths],
        "code": [fingerprint_artifact(path) for path in code_paths],
        "repository": repository_state(Path(str(root_value))),
    }


def _numbers(values: Sequence[Any]) -> list[float]:
    return [number for value in values if (number := finite(value)) is not None]


def _distribution(values: Sequence[Any]) -> dict[str, Any] | None:
    numbers = _numbers(values)
    if not numbers:
        return None
    ordered = sorted(numbers)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "samples": len(numbers),
        "mean": round(statistics.fmean(numbers), 6),
        "p50": round(statistics.median(numbers), 6),
        "p95": round(ordered[index], 6),
        "minimum": round(min(numbers), 6),
        "maximum": round(max(numbers), 6),
    }


def runtime_summary(path: Path) -> dict[str, Any]:
    records = read_history(path)
    stability = summarize(records)
    benchmark_rows = [
        record.get("benchmark", {})
        for record in records
        if isinstance(record.get("benchmark"), dict)
    ]
    face_rows = [
        record.get("face", {})
        for record in records
        if isinstance(record.get("face"), dict)
    ]
    swap_values = [bool(record.get("swap_applied", False)) for record in records]
    measurable = [bool(face.get("available", False)) for face in face_rows]
    return {
        "stability": stability,
        "processing_fps": _distribution(
            [row.get("processing_fps") for row in benchmark_rows]
        ),
        "processor_total_ms": _distribution(
            [
                row.get("timings_ms", {}).get("total")
                for row in benchmark_rows
                if isinstance(row.get("timings_ms"), dict)
            ]
        ),
        "swap_applied_percent": round(
            100.0 * sum(swap_values) / max(1, len(swap_values)), 3
        ),
        "face_measurable_percent": round(
            100.0 * sum(measurable) / max(1, len(measurable)), 3
        ),
        "core_mean_absolute_delta": _distribution(
            [face.get("core_mean_absolute_delta") for face in face_rows]
        ),
        "core_changed_pixels_percent": _distribution(
            [face.get("core_changed_pixels_percent") for face in face_rows]
        ),
    }


def metric_vector(report: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    full = report.get("full_reference_metrics", {})
    for output_name, report_name in (("vmaf", "vmaf"), ("ssim", "ssim"), ("psnr", "psnr")):
        metric = full.get(report_name, {}) if isinstance(full, dict) else {}
        number = finite(metric.get("mean") if isinstance(metric, dict) else None)
        if number is not None:
            values[output_name] = number
    temporal = report.get("temporal", {})
    processed = temporal.get("processed", {}) if isinstance(temporal, dict) else {}
    for output_name, report_name in (
        ("processed_exact_repeat_percent", "exact_repeat_percent"),
        ("processed_blockiness_ratio", "blockiness_ratio_mean"),
    ):
        number = finite(processed.get(report_name) if isinstance(processed, dict) else None)
        if number is not None:
            values[output_name] = number

    if runtime is None:
        embedded = report.get("runtime_quality_history")
        if isinstance(embedded, dict):
            summary = embedded.get("summary")
            runtime = {"stability": summary} if isinstance(summary, dict) else None
    if runtime:
        stability = runtime.get("stability", {})
        if isinstance(stability, dict):
            for name in METRICS:
                item = stability.get(name)
                number = finite(item.get("mean")) if isinstance(item, dict) else None
                if number is not None:
                    values[name] = number
        fps = runtime.get("processing_fps")
        number = finite(fps.get("mean")) if isinstance(fps, dict) else None
        if number is not None:
            values["processing_fps"] = number
    return values


def validate_capture(capture_dir: Path) -> dict[str, Any]:
    manifest_path = capture_dir / "capture.json"
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{manifest_path}: missing files map")
    for name in ("reference", "processed", "quality_history", "settings"):
        entry = files.get(name)
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(f"{manifest_path}: missing {name} artifact")
        path = capture_dir / str(entry["path"])
        if not path.is_file():
            raise ValueError(f"{path}: missing capture artifact")
        expected = str(entry.get("sha256", ""))
        if expected and sha256_file(path) != expected:
            raise ValueError(f"{path}: SHA-256 does not match capture manifest")
    return manifest


def _retime_lossless(
    source: Path, destination: Path, fps: float, frames: int
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to retime a capture")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-r",
            f"{fps:.9f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-pix_fmt",
            "bgr0",
            "-r",
            f"{fps:.9f}",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"retime failed: {completed.stderr.strip()[:1000]}")


def _decoded_frame_hashes(path: Path) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to validate a retimed capture")
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"frame-hash validation failed: {completed.stderr.strip()[:1000]}"
        )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in completed.stdout.splitlines()
        if line and not line.startswith("#")
    ]


def _decoded_bgr_sha256(path: Path) -> str:
    """Hash ordered decoded BGR pixels, independent of container metadata."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to fingerprint decoded frames")
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("could not read ffmpeg decoded-frame output")
    digest = hashlib.sha256()
    while chunk := process.stdout.read(1024 * 1024):
        digest.update(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait(timeout=300)
    if return_code != 0:
        raise RuntimeError(f"decoded-frame fingerprint failed: {stderr.strip()[:1000]}")
    return digest.hexdigest()


def derive_retimed_capture(source_dir: Path, run_id: str) -> Path:
    """Correct only clip timestamps while proving decoded pixels unchanged."""
    source_dir = source_dir.expanduser().resolve()
    capture = validate_capture(source_dir)
    fps = finite(capture.get("observed_sample_fps"))
    if fps is None or fps <= 0:
        raise ValueError("source capture has no usable observed_sample_fps")
    destination = source_dir.parent / run_id
    if destination.exists():
        raise FileExistsError(f"{destination}: derived captures are immutable")
    destination.mkdir(parents=True)
    try:
        for name in ("quality-history.jsonl", "settings.json"):
            shutil.copy2(source_dir / name, destination / name)
        request = read_json(source_dir / "request.json")
        request.update(
            {
                "id": run_id,
                "token": f"{run_id}-{time.time_ns()}",
                "notes": (
                    f"Timestamp-corrected derivative of {capture['id']}; decoded pixels unchanged"
                ),
            }
        )
        atomic_json(destination / "request.json", request)
        for name in ("reference.mkv", "processed.mkv"):
            _retime_lossless(
                source_dir / name,
                destination / name,
                fps,
                int(capture["frames"]),
            )
            if _decoded_frame_hashes(source_dir / name) != _decoded_frame_hashes(
                destination / name
            ):
                raise RuntimeError(f"{name}: retime changed decoded frame pixels")
        context = dict(capture.get("context", {}))
        context["derived_from_capture"] = {
            "id": capture["id"],
            "capture_sha256": sha256_file(source_dir / "capture.json"),
            "reason": "correct nominal cadence to observed sampler cadence",
        }
        capture.update(
            {
                "id": run_id,
                "token": request["token"],
                "role": "baseline",
                "sample_fps": fps,
                "nominal_duration_seconds": round(capture["frames"] / fps, 6),
                "derived_at": time.time(),
                "context": context,
            }
        )
        capture["files"] = {
            "reference": {
                "path": "reference.mkv",
                "bytes": (destination / "reference.mkv").stat().st_size,
                "sha256": sha256_file(destination / "reference.mkv"),
                "frame_content_sha256": _decoded_bgr_sha256(
                    destination / "reference.mkv"
                ),
                "frame_content_format": "bgr24-frame-sequence-v1",
            },
            "processed": {
                "path": "processed.mkv",
                "bytes": (destination / "processed.mkv").stat().st_size,
                "sha256": sha256_file(destination / "processed.mkv"),
                "frame_content_sha256": _decoded_bgr_sha256(
                    destination / "processed.mkv"
                ),
                "frame_content_format": "bgr24-frame-sequence-v1",
            },
            "quality_history": {
                "path": "quality-history.jsonl",
                "bytes": (destination / "quality-history.jsonl").stat().st_size,
                "sha256": sha256_file(destination / "quality-history.jsonl"),
            },
            "settings": {
                "path": "settings.json",
                "sha256": sha256_file(destination / "settings.json"),
            },
        }
        atomic_json(destination / "capture.json", capture)
        return destination
    except Exception:
        # Keep a failed derivation out of the immutable capture namespace.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def derive_enriched_capture(source_dir: Path, run_id: str) -> Path:
    """Add decoded-corpus fingerprints without changing either media file."""
    source_dir = source_dir.expanduser().resolve()
    capture = validate_capture(source_dir)
    destination = source_dir.parent / run_id
    if destination.exists():
        raise FileExistsError(f"{destination}: derived captures are immutable")
    destination.mkdir(parents=True)
    try:
        for name in (
            "reference.mkv",
            "processed.mkv",
            "quality-history.jsonl",
            "settings.json",
        ):
            shutil.copy2(source_dir / name, destination / name)
        request = read_json(source_dir / "request.json")
        request.update(
            {
                "id": run_id,
                "token": f"{run_id}-{time.time_ns()}",
                "notes": (
                    f"Content-fingerprint derivative of {capture['id']}; media bytes unchanged"
                ),
            }
        )
        atomic_json(destination / "request.json", request)
        context = dict(capture.get("context", {}))
        context["derived_from_capture"] = {
            "id": capture["id"],
            "capture_sha256": sha256_file(source_dir / "capture.json"),
            "reason": "add container-independent decoded-corpus fingerprints",
        }
        capture.update(
            {
                "id": run_id,
                "token": request["token"],
                "role": "baseline",
                "derived_at": time.time(),
                "context": context,
            }
        )
        files = capture["files"]
        for key, name in (("reference", "reference.mkv"), ("processed", "processed.mkv")):
            files[key].update(
                {
                    "path": name,
                    "bytes": (destination / name).stat().st_size,
                    "sha256": sha256_file(destination / name),
                    "frame_content_sha256": _decoded_bgr_sha256(destination / name),
                    "frame_content_format": "bgr24-frame-sequence-v1",
                }
            )
        files["quality_history"].update(
            {
                "path": "quality-history.jsonl",
                "bytes": (destination / "quality-history.jsonl").stat().st_size,
                "sha256": sha256_file(destination / "quality-history.jsonl"),
            }
        )
        files["settings"].update(
            {
                "path": "settings.json",
                "sha256": sha256_file(destination / "settings.json"),
            }
        )
        atomic_json(destination / "capture.json", capture)
        return destination
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def analyze_capture(
    capture_dir: Path,
    *,
    capture_subject_consent: bool,
    source_identity_authorized: bool,
    activate: bool,
) -> dict[str, Any]:
    capture_dir = capture_dir.expanduser().resolve()
    run_path = capture_dir / "run.json"
    if run_path.exists():
        raise FileExistsError(
            f"{run_path} already exists; captured baselines are immutable"
        )
    capture = validate_capture(capture_dir)
    reference = capture_dir / str(capture["files"]["reference"]["path"])
    processed = capture_dir / str(capture["files"]["processed"]["path"])
    settings = capture_dir / str(capture["files"]["settings"]["path"])
    quality_history = capture_dir / str(
        capture["files"]["quality_history"]["path"]
    )
    duration = float(capture["nominal_duration_seconds"])
    sample = {
        "id": capture["id"],
        "reference": str(reference),
        "processed": str(processed),
        "duration_seconds": duration,
        "reference_offset_seconds": 0.0,
        "processed_offset_seconds": 0.0,
        "intended_use": "internal",
        "split": "diagnostic",
        "capture_subject_consent": capture_subject_consent,
        "source_identity_authorized": source_identity_authorized,
        "device": capture.get("context", {}).get("input_mode", "pipeline"),
        "settings": str(settings),
        "quality_history": str(quality_history),
    }
    analysis_dir = capture_dir / "analysis"
    report = evaluate_pair(sample, base=Path("/"), output_dir=analysis_dir)
    report_path = analysis_dir / "samples" / capture["id"] / "report.json"
    runtime = runtime_summary(quality_history)
    artifacts = artifact_inventory(capture.get("context", {}))
    reference_content_sha256 = capture["files"]["reference"].get(
        "frame_content_sha256"
    ) or _decoded_bgr_sha256(reference)
    processed_content_sha256 = capture["files"]["processed"].get(
        "frame_content_sha256"
    ) or _decoded_bgr_sha256(processed)
    run = {
        "schema_version": SCHEMA_VERSION,
        "id": capture["id"],
        "role": capture["role"],
        "created_at": time.time(),
        "capture_manifest": {
            "path": "capture.json",
            "sha256": sha256_file(capture_dir / "capture.json"),
        },
        "artifacts": artifacts,
        "analysis": {
            "report": str(report_path.relative_to(capture_dir)),
            "report_sha256": sha256_file(report_path),
            "runtime_summary": runtime,
            "metric_vector": metric_vector(report, runtime),
        },
        "comparison_contract": {
            "reference_sha256": capture["files"]["reference"]["sha256"],
            "reference_content_sha256": reference_content_sha256,
            "processed_sha256": capture["files"]["processed"]["sha256"],
            "processed_content_sha256": processed_content_sha256,
            "frames": capture["frames"],
            "sample_fps": capture["sample_fps"],
            "source_identity_sha256": (
                artifacts.get("source") or {}
            ).get("sha256"),
            "resolution": capture.get("context", {}).get("resolution"),
            "pairing": "same in-process invocation",
        },
        "interpretation": {
            "full_reference_metrics": (
                "signal/background preservation only; the intended identity edit lowers them"
            ),
            "identity": (
                "face-core change confirms a visual effect, not identity similarity"
            ),
            "definitive_cross_pipeline_comparison": (
                "requires replay of this exact decoded reference corpus through each candidate"
            ),
        },
    }
    atomic_json(run_path, run)
    atomic_text(capture_dir / "run.md", run_markdown(run))
    if activate:
        root = capture_dir.parents[1]
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "id": run["id"],
            "run": str(run_path),
            "run_sha256": sha256_file(run_path),
            "activated_at": time.time(),
        }
        atomic_json(root / "active-baseline.json", pointer)
    return run


def import_report(
    report_path: Path,
    *,
    run_id: str,
    output: Path,
    health_path: Path | None = None,
) -> dict[str, Any]:
    """Wrap a legacy media report as a diagnostic-only candidate run."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output}: imported runs are immutable")
    report_path = report_path.expanduser().resolve()
    report = read_json(report_path)
    metrics = metric_vector(report)
    health: dict[str, Any] | None = None
    if health_path is not None:
        health = read_json(health_path.expanduser().resolve())
        processing = health.get("processing", {})
        quality = processing.get("quality", {}) if isinstance(processing, dict) else {}
        face = quality.get("face", {}) if isinstance(quality, dict) else {}
        tracking = quality.get("tracking", {}) if isinstance(quality, dict) else {}
        mappings = {
            "face_stability_score": face.get("stability_score"),
            "face_excess_flicker_p95": face.get("excess_flicker_p95"),
            "seam_temporal_std": face.get("seam_temporal_std"),
            "seam_ring_delta_lab": face.get("seam_ring_delta_lab"),
            "face_luma_mismatch": face.get("luma_mismatch"),
            "face_chroma_mismatch_lab": face.get("chroma_mismatch_lab"),
            "detection_miss_percent": tracking.get("detection_miss_percent"),
            "landmark_correction_p95_percent": tracking.get(
                "landmark_correction_p95_percent"
            ),
            "pipeline_ms": quality.get("pipeline_ms"),
            "pipeline_ms_p95": quality.get("pipeline_ms_p95"),
            "processing_fps": processing.get("fps"),
        }
        detail = finite(face.get("detail_ratio"))
        if detail is not None:
            mappings["detail_ratio_error"] = abs(detail - 1.0)
        for name, value in mappings.items():
            number = finite(value)
            if number is not None:
                metrics[name] = number
    reference = report.get("reference", {})
    processed = report.get("processed", {})
    reference_video = reference.get("video", {}) if isinstance(reference, dict) else {}
    full = report.get("full_reference_metrics", {})
    normalization = full.get("normalization", {}) if isinstance(full, dict) else {}
    reference_path_value = reference.get("path") if isinstance(reference, dict) else None
    reference_content_sha256 = None
    if reference_path_value and Path(str(reference_path_value)).is_file():
        reference_content_sha256 = _decoded_bgr_sha256(
            Path(str(reference_path_value))
        )
    run = {
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "role": "candidate",
        "created_at": time.time(),
        "imported_legacy_report": True,
        "artifacts": {
            "reference": {
                "path": reference.get("path"),
                "sha256": reference.get("sha256"),
            },
            "processed": {
                "path": processed.get("path") if isinstance(processed, dict) else None,
                "sha256": processed.get("sha256") if isinstance(processed, dict) else None,
            },
            "health": (
                {
                    "path": str(health_path.expanduser().resolve()),
                    "sha256": sha256_file(health_path.expanduser().resolve()),
                }
                if health_path is not None
                else None
            ),
        },
        "analysis": {
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "runtime_summary": {"point_in_time_health": health},
            "metric_vector": metrics,
        },
        "comparison_contract": {
            "reference_sha256": reference.get("sha256"),
            "reference_content_sha256": reference_content_sha256,
            "processed_sha256": (
                processed.get("sha256") if isinstance(processed, dict) else None
            ),
            "frames": (
                full.get("vmaf", {}).get("frames")
                if isinstance(full.get("vmaf"), dict)
                else None
            ),
            "sample_fps": reference_video.get("fps"),
            "source_identity_sha256": None,
            "resolution": [
                reference_video.get("width"),
                reference_video.get("height"),
            ],
            "pairing": "legacy aligned live recordings",
            "alignment": normalization,
        },
        "interpretation": {
            "comparison_status": (
                "diagnostic only unless reference SHA-256 and capture contract match"
            )
        },
    }
    atomic_json(output, run)
    return run


def run_markdown(run: dict[str, Any]) -> str:
    metrics = run["analysis"]["metric_vector"]
    lines = [
        f"# Pipeline benchmark: {run['id']}",
        "",
        f"Role: **{run['role']}**",
        "",
        "The clips are synchronized lossless frame pairs captured inside one processor invocation. No camera or transport endpoint was opened by the recorder.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in METRICS:
        if name in metrics:
            lines.append(f"| {name} | {metrics[name]:.5f} |")
    lines.extend(
        [
            "",
            "VMAF, SSIM, and PSNR measure signal/background preservation and also penalize the intended identity change. They are not facial-realism scores.",
            "",
            "A definitive comparison must replay the exact frozen decoded reference corpus. Independent live captures are diagnostic only.",
        ]
    )
    return "\n".join(lines)


def resolve_run(value: Path, state_dir: Path | None = None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = value.expanduser()
    if str(value) == "active":
        if state_dir is None:
            raise ValueError("active baseline requires --state-dir")
        pointer = read_json(state_dir / "benchmarks" / "active-baseline.json")
        path = Path(str(pointer["run"]))
    if path.is_dir():
        path = path / "run.json"
    run = read_json(path.resolve())
    report_value = run.get("analysis", {}).get("report")
    if not report_value:
        raise ValueError(f"{path}: not a pipeline run manifest")
    report_path = path.parent / str(report_value)
    expected = run.get("analysis", {}).get("report_sha256")
    if expected and sha256_file(report_path) != expected:
        raise ValueError(f"{report_path}: report hash does not match run manifest")
    return run, path.resolve(), read_json(report_path)


def _metric_change(name: str, baseline: float, candidate: float) -> dict[str, Any]:
    direction, category = METRICS[name]
    gain = candidate - baseline if direction == "higher" else baseline - candidate
    scale = abs(baseline)
    tolerance = max(0.001, scale * 0.01)
    verdict = "improved" if gain > tolerance else "regressed" if gain < -tolerance else "unchanged"
    return {
        "category": category,
        "preferred": direction,
        "baseline": round(baseline, 6),
        "candidate": round(candidate, 6),
        "absolute_gain": round(gain, 6),
        "gain_percent": (
            round(100.0 * gain / scale, 3) if scale >= 0.001 else None
        ),
        "verdict": verdict,
    }


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    base_contract = baseline.get("comparison_contract", {})
    candidate_contract = candidate.get("comparison_contract", {})
    reasons: list[str] = []
    for field in ("frames", "sample_fps", "resolution"):
        if base_contract.get(field) != candidate_contract.get(field):
            reasons.append(f"{field} differs")
    base_corpus = base_contract.get("reference_content_sha256")
    candidate_corpus = candidate_contract.get("reference_content_sha256")
    if not base_corpus or not candidate_corpus:
        reasons.append("decoded reference fingerprint unavailable")
    elif base_corpus != candidate_corpus:
        reasons.append("decoded reference corpus differs")
    base_identity = base_contract.get("source_identity_sha256")
    candidate_identity = candidate_contract.get("source_identity_sha256")
    if not base_identity or not candidate_identity:
        reasons.append("source identity fingerprint unavailable")
    elif base_identity != candidate_identity:
        reasons.append("source identity differs")
    base_metrics = baseline.get("analysis", {}).get("metric_vector", {})
    candidate_metrics = candidate.get("analysis", {}).get("metric_vector", {})
    changes = {
        name: _metric_change(name, float(base_metrics[name]), float(candidate_metrics[name]))
        for name in METRICS
        if name in base_metrics and name in candidate_metrics
    }
    critical_regressions = [
        name
        for name, result in changes.items()
        if name in CRITICAL_QUALITY_METRICS and result["verdict"] == "regressed"
    ]
    improved = [name for name, result in changes.items() if result["verdict"] == "improved"]
    regressed = [name for name, result in changes.items() if result["verdict"] == "regressed"]
    comparable = not reasons
    if not comparable:
        verdict = "diagnostic-only"
        detail = "raw corpus/contract differs; no winner is declared"
    elif critical_regressions:
        verdict = "rejected"
        detail = "candidate has a critical quality regression"
    elif improved and not regressed:
        verdict = "pareto-improved"
        detail = "at least one measured dimension improved and none regressed"
    elif not regressed:
        verdict = "non-inferior"
        detail = "no measured dimension regressed beyond the 1% tolerance"
    else:
        verdict = "tradeoff"
        detail = "candidate improves and regresses different dimensions"
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": baseline.get("id"),
        "candidate_id": candidate.get("id"),
        "comparable": comparable,
        "comparability_reasons": reasons,
        "changes": changes,
        "verdict": {
            "status": verdict,
            "detail": detail,
            "improved": improved,
            "regressed": regressed,
            "critical_regressions": critical_regressions,
            "rule": (
                "same frozen raw corpus required; critical quality metrics are non-inferiority gates"
            ),
        },
    }


def comparison_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['candidate_id']} versus {result['baseline_id']}",
        "",
        f"Verdict: **{result['verdict']['status']}** — {result['verdict']['detail']}.",
        "",
    ]
    if result["comparability_reasons"]:
        lines.extend(
            [
                "Comparability limits: " + ", ".join(result["comparability_reasons"]),
                "",
            ]
        )
    lines.extend(["| Metric | Baseline | Candidate | Delta verdict |", "|---|---:|---:|---|"])
    for name, value in result["changes"].items():
        delta = (
            f"{value['gain_percent']:+.2f}%"
            if value["gain_percent"] is not None
            else f"absolute {value['absolute_gain']:+.5f}"
        )
        lines.append(
            f"| {name} | {value['baseline']:.5f} | {value['candidate']:.5f} | "
            f"{value['verdict']} ({delta}) |"
        )
    return "\n".join(lines)


def request_recording(args: argparse.Namespace) -> Path:
    state_dir = args.state_dir.expanduser().resolve()
    root = state_dir / "benchmarks"
    request_path = root / "request.json"
    status_path = root / "status.json"
    request = validate_request(
        {
            "id": args.id,
            "token": f"{args.id}-{time.time_ns()}",
            "role": args.role,
            "frame_count": args.frames,
            "sample_fps": args.sample_fps,
            "requested_at": time.time(),
            "notes": args.notes,
        }
    )
    if request_path.exists():
        raise FileExistsError(f"{request_path}: another request is pending")
    atomic_json(request_path, request)
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        if status_path.is_file():
            status = read_json(status_path)
            if status.get("token") == request["token"]:
                if status.get("state") == "complete":
                    return Path(str(status["capture_dir"]))
                if status.get("state") == "failed":
                    raise RuntimeError(str(status.get("error", "capture failed")))
                print(
                    f"capture {status.get('state')}: "
                    f"{status.get('accepted_frames', 0)}/"
                    f"{status.get('target_frames', request['frame_count'])}",
                    end="\r",
                    flush=True,
                    file=sys.stderr,
                )
        time.sleep(0.25)
    raise TimeoutError(
        f"service did not complete benchmark request within {args.wait_seconds}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="request and analyze a live capture")
    record.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    record.add_argument("--id", required=True)
    record.add_argument("--role", choices=("baseline", "candidate"), default="candidate")
    record.add_argument("--frames", type=int, default=50)
    record.add_argument("--sample-fps", type=float, default=5.0)
    record.add_argument("--wait-seconds", type=float, default=180.0)
    record.add_argument("--notes", default="")
    record.add_argument("--capture-subject-consent", action="store_true")
    record.add_argument("--source-identity-authorized", action="store_true")
    record.add_argument("--activate", action="store_true")

    analyze = subparsers.add_parser("analyze", help="analyze a completed capture")
    analyze.add_argument("capture_dir", type=Path)
    analyze.add_argument("--capture-subject-consent", action="store_true")
    analyze.add_argument("--source-identity-authorized", action="store_true")
    analyze.add_argument("--activate", action="store_true")

    retime = subparsers.add_parser(
        "retime", help="derive a timestamp-corrected lossless baseline"
    )
    retime.add_argument("source_capture", type=Path)
    retime.add_argument("--id", required=True)
    retime.add_argument("--capture-subject-consent", action="store_true")
    retime.add_argument("--source-identity-authorized", action="store_true")
    retime.add_argument("--activate", action="store_true")

    enrich = subparsers.add_parser(
        "enrich", help="derive an unchanged-media capture with corpus fingerprints"
    )
    enrich.add_argument("source_capture", type=Path)
    enrich.add_argument("--id", required=True)
    enrich.add_argument("--capture-subject-consent", action="store_true")
    enrich.add_argument("--source-identity-authorized", action="store_true")
    enrich.add_argument("--activate", action="store_true")

    compare = subparsers.add_parser("compare", help="compare two analyzed runs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--strict", action="store_true")

    imported = subparsers.add_parser(
        "import-report", help="wrap a legacy media report for diagnostic comparison"
    )
    imported.add_argument("--report", type=Path, required=True)
    imported.add_argument("--id", required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--health", type=Path)

    status = subparsers.add_parser("status", help="show recorder status")
    status.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        path = args.state_dir.expanduser().resolve() / "benchmarks" / "status.json"
        print(json.dumps(read_json(path), indent=2, sort_keys=True))
        return 0
    if args.command == "record":
        capture_dir = request_recording(args)
        print(file=sys.stderr)
        run = analyze_capture(
            capture_dir,
            capture_subject_consent=args.capture_subject_consent,
            source_identity_authorized=args.source_identity_authorized,
            activate=args.activate,
        )
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze":
        run = analyze_capture(
            args.capture_dir,
            capture_subject_consent=args.capture_subject_consent,
            source_identity_authorized=args.source_identity_authorized,
            activate=args.activate,
        )
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0
    if args.command == "retime":
        capture_dir = derive_retimed_capture(args.source_capture, args.id)
        run = analyze_capture(
            capture_dir,
            capture_subject_consent=args.capture_subject_consent,
            source_identity_authorized=args.source_identity_authorized,
            activate=args.activate,
        )
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0
    if args.command == "enrich":
        capture_dir = derive_enriched_capture(args.source_capture, args.id)
        run = analyze_capture(
            capture_dir,
            capture_subject_consent=args.capture_subject_consent,
            source_identity_authorized=args.source_identity_authorized,
            activate=args.activate,
        )
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        baseline, _baseline_path, _baseline_report = resolve_run(
            args.baseline, args.state_dir.expanduser().resolve()
        )
        candidate, _candidate_path, _candidate_report = resolve_run(
            args.candidate, args.state_dir.expanduser().resolve()
        )
        result = compare_runs(baseline, candidate)
        if args.output:
            atomic_json(args.output, result)
            atomic_text(args.output.with_suffix(".md"), comparison_markdown(result))
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.strict and result["verdict"]["status"] in {"rejected", "tradeoff", "diagnostic-only"}:
            return 2
        return 0
    if args.command == "import-report":
        run = import_report(
            args.report,
            run_id=args.id,
            output=args.output,
            health_path=args.health,
        )
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
