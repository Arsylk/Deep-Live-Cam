#!/usr/bin/env python3
"""Offline quality, integrity, detector, and provenance benchmark.

The tool accepts paired camera/reference and processed clips or a frozen study
manifest. It never uploads media, downloads detector weights, or contacts a
service. Detector scores are ingested as labeled results and are intentionally
kept outside the responsible-release grade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from .c2pa_provenance import (
        build_manifest,
        inspect_asset,
        OFFLINE_SETTINGS,
        signing_command,
        write_manifest,
    )
    from .forensic_metrics import evaluate_rows, load_rows
    from .subjective_metrics import evaluate_subjective_rows, load_subjective_rows
    from .stability_report import read_history, summarize as summarize_quality_history
except ImportError:  # Direct ``python tools/media_integrity_benchmark.py`` use.
    from c2pa_provenance import (  # type: ignore[no-redef]
        build_manifest,
        inspect_asset,
        OFFLINE_SETTINGS,
        signing_command,
        write_manifest,
    )
    from forensic_metrics import evaluate_rows, load_rows  # type: ignore[no-redef]
    from subjective_metrics import (  # type: ignore[no-redef]
        evaluate_subjective_rows,
        load_subjective_rows,
    )
    from stability_report import (  # type: ignore[no-redef]
        read_history,
        summarize as summarize_quality_history,
    )


SCHEMA_VERSION = "1.0"
DEFAULT_DURATION = 30.0
DEFAULT_MAX_FRAMES = 1800
FFMPEG_SUMMARY = {
    "vmaf": re.compile(r"VMAF score:\s*([0-9.]+)", re.IGNORECASE),
    "ssim": re.compile(r"SSIM .*?All:([0-9.]+)", re.IGNORECASE),
    "psnr": re.compile(r"PSNR .*?average:([0-9.]+|inf)", re.IGNORECASE),
}


@dataclass
class TemporalTrace:
    report: dict[str, Any]
    frame_luma: np.ndarray
    frame_delta: np.ndarray


def _json_number(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isinf(number):
            return "infinity"
        if math.isnan(number):
            return None
        return round(number, digits)
    if isinstance(value, dict):
        return {key: _json_number(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_number(item, digits) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_number(value), indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_version(executable: str) -> str | None:
    path = shutil.which(executable)
    if not path:
        return None
    completed = subprocess.run(
        [path, "-version"], check=False, capture_output=True, text=True, timeout=15
    )
    line = (completed.stdout or completed.stderr).splitlines()
    return line[0].strip() if line else None


def runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ffmpeg": _tool_version("ffmpeg"),
        "ffprobe": _tool_version("ffprobe"),
        "c2patool_available": bool(shutil.which("c2patool")),
    }


def _fraction(value: Any) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            if float(denominator) == 0:
                return None
            return float(numerator) / float(denominator)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_media(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"media file does not exist: {path}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required but is not available on PATH")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        (
            "format=format_name,duration,size,bit_rate:"
            "stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,"
            "avg_frame_rate,bit_rate,duration,nb_frames,color_range,color_space,"
            "color_transfer,color_primaries"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise ValueError(
            f"ffprobe could not read {path}: {completed.stderr.strip()[:1000]}"
        )
    report = json.loads(completed.stdout)
    streams = report.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    if not isinstance(video, dict):
        raise ValueError(f"{path}: no video stream")
    format_info = report.get("format", {})
    duration = _fraction(video.get("duration")) or _fraction(format_info.get("duration"))
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(
        video.get("r_frame_rate")
    )
    bit_rate = _fraction(video.get("bit_rate")) or _fraction(
        format_info.get("bit_rate")
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format": format_info.get("format_name"),
        "duration_seconds": duration,
        "video": {
            "codec": video.get("codec_name"),
            "width": int(video.get("width", 0)),
            "height": int(video.get("height", 0)),
            "pixel_format": video.get("pix_fmt"),
            "fps": fps,
            "bit_rate": bit_rate,
            "reported_frames": int(video["nb_frames"])
            if str(video.get("nb_frames", "")).isdigit()
            else None,
            "color_range": video.get("color_range"),
            "color_space": video.get("color_space"),
            "color_transfer": video.get("color_transfer"),
            "color_primaries": video.get("color_primaries"),
        },
    }


def ffmpeg_filter_names() -> set[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return set()
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    names: set[str] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0][0:1] in {".", "T", "S", "C", "A", "V", "N", "|"}:
            names.add(fields[1])
    return names


def _ffmpeg_pair_command(
    reference: Path,
    processed: Path,
    *,
    reference_offset: float,
    processed_offset: float,
    duration: float,
    width: int,
    height: int,
    fps: float,
    metric_filter: str,
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but is not available on PATH")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info"]
    if reference_offset > 0:
        command.extend(("-ss", f"{reference_offset:.6f}"))
    command.extend(("-i", str(reference)))
    if processed_offset > 0:
        command.extend(("-ss", f"{processed_offset:.6f}"))
    command.extend(("-i", str(processed)))
    normalized_fps = max(1.0, fps)
    complex_filter = (
        f"[0:v]settb=AVTB,setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:flags=bicubic,fps={normalized_fps:.6f},"
        "format=yuv420p[reference];"
        f"[1:v]settb=AVTB,setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:flags=bicubic,fps={normalized_fps:.6f},"
        "format=yuv420p[processed];"
        f"[processed][reference]{metric_filter}"
    )
    command.extend(("-filter_complex", complex_filter, "-an", "-sn", "-dn"))
    if duration > 0:
        command.extend(("-t", f"{duration:.6f}"))
    command.extend(("-f", "null", "-"))
    return command


def _run_ffmpeg_metric(
    name: str,
    metric_filter: str,
    reference: Path,
    processed: Path,
    *,
    reference_offset: float,
    processed_offset: float,
    duration: float,
    width: int,
    height: int,
    fps: float,
    timeout: float,
) -> dict[str, Any]:
    command = _ffmpeg_pair_command(
        reference,
        processed,
        reference_offset=reference_offset,
        processed_offset=processed_offset,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        metric_filter=metric_filter,
    )
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    match = FFMPEG_SUMMARY[name].search(completed.stderr)
    if completed.returncode or not match:
        detail = completed.stderr.strip().splitlines()
        return {
            "available": False,
            "error": detail[-1][:1000] if detail else "FFmpeg returned no metric",
            "exit_code": completed.returncode,
        }
    raw = match.group(1).lower()
    value = float("inf") if raw == "inf" else float(raw)
    return {"available": True, "mean": value}


def full_reference_metrics(
    reference: Path,
    processed: Path,
    reference_info: dict[str, Any],
    *,
    reference_offset: float = 0.0,
    processed_offset: float = 0.0,
    duration: float = DEFAULT_DURATION,
) -> dict[str, Any]:
    """Measure aligned full-frame fidelity using FFmpeg's reference metrics."""
    filters = ffmpeg_filter_names()
    video = reference_info["video"]
    width, height = int(video["width"]), int(video["height"])
    fps = float(video.get("fps") or 30.0)
    timeout = max(120.0, (duration if duration > 0 else 60.0) * 8.0)
    result: dict[str, Any] = {
        "normalization": {
            "width": width,
            "height": height,
            "fps": fps,
            "reference_offset_seconds": reference_offset,
            "processed_offset_seconds": processed_offset,
            "duration_seconds": duration if duration > 0 else "full",
            "note": (
                "Both streams are aligned to zero, scaled to reference dimensions, "
                "and sampled at the reference cadence. Intentional face changes also "
                "affect these full-frame scores."
            ),
        },
        "vmaf": {"available": False, "error": "libvmaf filter is unavailable"},
        "ssim": {"available": False, "error": "ssim filter is unavailable"},
        "psnr": {"available": False, "error": "psnr filter is unavailable"},
    }
    if "libvmaf" in filters:
        with tempfile.TemporaryDirectory(prefix="dlc-vmaf-") as directory:
            log_path = Path(directory) / "vmaf.json"
            result["vmaf"] = _run_ffmpeg_metric(
                "vmaf",
                f"libvmaf=log_fmt=json:log_path={log_path}",
                reference,
                processed,
                reference_offset=reference_offset,
                processed_offset=processed_offset,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                timeout=timeout,
            )
            if log_path.exists():
                try:
                    data = json.loads(log_path.read_text("utf-8"))
                    pooled = data.get("pooled_metrics", {}).get("vmaf", {})
                    if pooled:
                        result["vmaf"].update(
                            {
                                "mean": pooled.get("mean"),
                                "minimum": pooled.get("min"),
                                "maximum": pooled.get("max"),
                                "harmonic_mean": pooled.get("harmonic_mean"),
                                "frames": len(data.get("frames", [])),
                            }
                        )
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
    if "ssim" in filters:
        result["ssim"] = _run_ffmpeg_metric(
            "ssim",
            "ssim",
            reference,
            processed,
            reference_offset=reference_offset,
            processed_offset=processed_offset,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            timeout=timeout,
        )
    if "psnr" in filters:
        result["psnr"] = _run_ffmpeg_metric(
            "psnr",
            "psnr",
            reference,
            processed,
            reference_offset=reference_offset,
            processed_offset=processed_offset,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            timeout=timeout,
        )
    # Preserve infinity internally (identical clips have infinite PSNR). The
    # JSON serializer converts it to the explicit string ``infinity`` later.
    return result


def _blockiness(gray: np.ndarray) -> float:
    pixels = gray.astype(np.float32)
    if min(pixels.shape) < 16:
        return 1.0
    vertical_all = float(np.mean(np.abs(pixels[:, 1:] - pixels[:, :-1])))
    horizontal_all = float(np.mean(np.abs(pixels[1:, :] - pixels[:-1, :])))
    vertical_blocks = float(np.mean(np.abs(pixels[:, 8::8] - pixels[:, 7:-1:8])))
    horizontal_blocks = float(np.mean(np.abs(pixels[8::8, :] - pixels[7:-1:8, :])))
    return (vertical_blocks + horizontal_blocks) / max(
        0.001, vertical_all + horizontal_all
    )


def _sample_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 320, 180
    scale = min(320.0 / width, 180.0 / height, 1.0)
    return max(16, round(width * scale)), max(16, round(height * scale))


def temporal_trace(
    path: Path,
    *,
    offset: float = 0.0,
    duration: float = DEFAULT_DURATION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> TemporalTrace:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not decode {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if offset > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    size = _sample_size(width, height)
    lumas: list[float] = []
    deltas: list[float] = []
    details: list[float] = []
    blocks: list[float] = []
    black_clip: list[float] = []
    white_clip: list[float] = []
    previous: np.ndarray | None = None
    repeat_streak = longest_repeat_streak = 0
    started_msec = offset * 1000.0
    while len(lumas) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if duration > 0:
            position = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if position and position - started_msec > duration * 1000.0:
                break
        sample = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        lumas.append(float(np.mean(gray)))
        details.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        blocks.append(_blockiness(gray))
        black_clip.append(100.0 * float(np.mean(gray <= 3)))
        white_clip.append(100.0 * float(np.mean(gray >= 252)))
        if previous is not None:
            delta = float(
                np.mean(np.abs(gray.astype(np.int16) - previous.astype(np.int16)))
            )
            deltas.append(delta)
            if delta <= 0.08:
                repeat_streak += 1
                longest_repeat_streak = max(longest_repeat_streak, repeat_streak)
            else:
                repeat_streak = 0
        previous = gray
    capture.release()
    if len(lumas) < 2:
        raise ValueError(f"{path}: fewer than two frames decoded")
    luma_array = np.asarray(lumas, dtype=np.float64)
    delta_array = np.asarray(deltas, dtype=np.float64)
    luma_changes = np.abs(np.diff(luma_array))
    report = {
        "frames": len(lumas),
        "decoder_fps": fps or None,
        "sample_dimensions": list(size),
        "luma_mean": float(np.mean(luma_array)),
        "luma_change_p95": float(np.percentile(luma_changes, 95)),
        "detail_laplacian_mean": float(np.mean(details)),
        "detail_laplacian_p10": float(np.percentile(details, 10)),
        "blockiness_ratio_mean": float(np.mean(blocks)),
        "black_clip_percent_mean": float(np.mean(black_clip)),
        "white_clip_percent_mean": float(np.mean(white_clip)),
        "frame_delta_mean": float(np.mean(delta_array)),
        "frame_delta_p95": float(np.percentile(delta_array, 95)),
        "exact_repeat_percent": 100.0 * float(np.mean(delta_array <= 0.08)),
        "near_repeat_percent": 100.0 * float(np.mean(delta_array <= 0.5)),
        "longest_exact_repeat_run": longest_repeat_streak,
        "truncated_at_max_frames": len(lumas) >= max_frames,
    }
    return TemporalTrace(_json_number(report), luma_array, delta_array)


def probe_timestamps(
    path: Path,
    *,
    offset: float = 0.0,
    duration: float = DEFAULT_DURATION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"available": False, "error": "ffprobe is unavailable"}
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if offset > 0 or duration > 0:
        interval = f"{offset:.6f}%"
        if duration > 0:
            interval += f"+{duration:.6f}"
        command.extend(("-read_intervals", interval))
    command.extend(
        (
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        )
    )
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    if completed.returncode:
        return {
            "available": False,
            "error": completed.stderr.strip()[-1000:],
        }
    frames = json.loads(completed.stdout).get("frames", [])[:max_frames]
    timestamps = [
        float(frame["best_effort_timestamp_time"])
        for frame in frames
        if frame.get("best_effort_timestamp_time") not in (None, "N/A")
    ]
    if len(timestamps) < 3:
        return {"available": False, "error": "fewer than three frame timestamps"}
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive = intervals[intervals > 0]
    if not len(positive):
        return {"available": False, "error": "no increasing frame timestamps"}
    median = float(np.median(positive))
    missing = int(
        sum(max(0, round(float(interval) / median) - 1) for interval in positive)
    )
    return _json_number(
        {
            "available": True,
            "timestamps": len(timestamps),
            "median_interval_ms": median * 1000.0,
            "effective_fps": 1.0 / float(np.mean(positive)),
            "jitter_std_ms": float(np.std(positive - median) * 1000.0),
            "interval_p95_ms": float(np.percentile(positive, 95) * 1000.0),
            "duplicate_or_nonmonotonic_timestamps": int(np.sum(intervals <= 0)),
            "estimated_missing_frames": missing,
            "estimated_missing_percent": 100.0 * missing / max(1, len(positive) + missing),
            "truncated_at_max_frames": len(frames) >= max_frames,
        }
    )


def compare_temporal(reference: TemporalTrace, processed: TemporalTrace) -> dict[str, Any]:
    count = min(len(reference.frame_delta), len(processed.frame_delta))
    if count < 1:
        return {"available": False, "error": "no aligned frame deltas"}
    raw_delta = reference.frame_delta[:count]
    returned_delta = processed.frame_delta[:count]
    moving = raw_delta > 0.8
    artificial = moving & (returned_delta <= 0.08)
    luma_count = min(len(reference.frame_luma), len(processed.frame_luma))
    raw_luma_change = np.diff(reference.frame_luma[:luma_count])
    returned_luma_change = np.diff(processed.frame_luma[:luma_count])
    result: dict[str, Any] = {
        "available": True,
        "aligned_transitions": count,
        "artificial_repeat_frames": int(np.sum(artificial)),
        "artificial_repeat_percent": 100.0 * float(np.mean(artificial)),
        "luma_change_error_mean": float(
            np.mean(np.abs(returned_luma_change - raw_luma_change))
        ),
    }
    if np.any(moving):
        ratios = returned_delta[moving] / np.maximum(raw_delta[moving], 1e-6)
        result["motion_retention_ratio_median"] = float(np.median(ratios))
    else:
        result["motion_retention_ratio_median"] = None
        result["motion_note"] = "reference segment contained no measured motion"
    return _json_number(result)


def _criterion(
    name: str,
    category: str,
    value: Any,
    target: str,
    weight: int,
    passed: bool,
    *,
    available: bool = True,
    note: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "value": value,
        "target": target,
        "weight": weight,
        "available": available,
        "passed": bool(passed) if available else False,
        "points": weight if available and passed else 0,
        "note": note,
    }


def responsible_release_assessment(
    reference_info: dict[str, Any],
    processed_info: dict[str, Any],
    full_reference: dict[str, Any],
    processed_temporal: dict[str, Any],
    temporal_comparison: dict[str, Any],
    timestamp_report: dict[str, Any],
    sample: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Apply declared engineering gates without using detector scores."""
    ref_video, out_video = reference_info["video"], processed_info["video"]
    ref_pixels = max(1, int(ref_video["width"]) * int(ref_video["height"]))
    out_pixels = int(out_video["width"]) * int(out_video["height"])
    resolution_ratio = out_pixels / ref_pixels
    ref_fps = float(ref_video.get("fps") or 0.0)
    out_fps = float(out_video.get("fps") or 0.0)
    fps_ratio = out_fps / ref_fps if ref_fps > 0 else 0.0

    intended_use = sample.get("intended_use", "recorded")
    if intended_use == "live":
        provenance_criterion = _criterion(
            "live transport recorded",
            "transparency",
            True,
            "live path recorded",
            10,
            True,
            note="Live camera transports do not carry file-level C2PA manifests.",
        )
    else:
        provenance_criterion = _criterion(
            "valid C2PA manifest",
            "transparency",
            provenance.get("valid"),
            "valid signed manifest",
            10,
            provenance.get("valid") is True,
            available=provenance.get("valid") is not None,
            note="A generated JSON template is not a signed Content Credential.",
        )

    criteria: list[dict[str, Any]] = []
    for metric, target, weight, predicate in (
        ("vmaf", ">= 90", 15, lambda value: value >= 90.0),
        ("ssim", ">= 0.95", 10, lambda value: value >= 0.95),
        ("psnr", ">= 30 dB", 5, lambda value: value >= 30.0),
    ):
        record = full_reference.get(metric, {})
        available = bool(record.get("available")) and isinstance(
            record.get("mean"), (float, int)
        )
        value = record.get("mean")
        criteria.append(
            _criterion(
                metric.upper(),
                "objective_quality",
                value,
                target,
                weight,
                predicate(float(value)) if available else False,
                available=available,
                note=(
                    "Full-frame fidelity; the intentionally changed face region affects it."
                ),
            )
        )
    criteria.extend(
        (
            _criterion(
                "resolution retention",
                "objective_quality",
                resolution_ratio,
                ">= 1.0x source pixels",
                5,
                resolution_ratio >= 1.0,
            ),
            _criterion(
                "nominal cadence retention",
                "objective_quality",
                fps_ratio,
                ">= 0.95x source FPS",
                5,
                fps_ratio >= 0.95,
            ),
            _criterion(
                "output exact repeats",
                "temporal_stability",
                processed_temporal.get("exact_repeat_percent"),
                "<= 0.5%",
                5,
                float(processed_temporal.get("exact_repeat_percent", 100.0)) <= 0.5,
            ),
            _criterion(
                "motion-time artificial repeats",
                "temporal_stability",
                temporal_comparison.get("artificial_repeat_percent"),
                "<= 0.2%",
                5,
                float(temporal_comparison.get("artificial_repeat_percent", 100.0))
                <= 0.2,
                available=bool(temporal_comparison.get("available")),
            ),
            _criterion(
                "timestamp continuity",
                "temporal_stability",
                timestamp_report.get("estimated_missing_percent"),
                "<= 0.5% estimated missing",
                5,
                float(timestamp_report.get("estimated_missing_percent", 100.0)) <= 0.5,
                available=bool(timestamp_report.get("available")),
            ),
            _criterion(
                "capture subject consent",
                "rights_and_protocol",
                bool(sample.get("capture_subject_consent", False)),
                "explicitly true",
                10,
                sample.get("capture_subject_consent") is True,
            ),
            _criterion(
                "source identity authorization",
                "rights_and_protocol",
                bool(sample.get("source_identity_authorized", False)),
                "explicitly true",
                10,
                sample.get("source_identity_authorized") is True,
            ),
            _criterion(
                "frozen holdout split",
                "rights_and_protocol",
                sample.get("split", "unreported"),
                "holdout",
                5,
                sample.get("split") == "holdout",
            ),
            _criterion(
                "configuration fingerprint",
                "rights_and_protocol",
                sample.get("configuration_fingerprint"),
                "recorded",
                5,
                bool(sample.get("configuration_fingerprint")),
            ),
            provenance_criterion,
        )
    )
    total_weight = sum(int(item["weight"]) for item in criteria)
    points = sum(int(item["points"]) for item in criteria)
    score = 100.0 * points / total_weight
    hard_failures = [
        item["name"]
        for item in criteria
        if item["name"]
        in {
            "capture subject consent",
            "source identity authorization",
        }
        and not item["passed"]
    ]
    if intended_use == "recorded" and provenance.get("valid") is not True:
        hard_failures.append("valid C2PA manifest for recorded release")
    if hard_failures:
        status = "blocked"
        grade = "not release-ready"
    else:
        status = "pass" if score >= 80.0 else "needs-work"
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
    return _json_number(
        {
            "score": score,
            "grade": grade,
            "status": status,
            "hard_failures": hard_failures,
            "criteria": criteria,
            "detector_metrics_included": False,
            "scoring_note": (
                "Engineering readiness score, not an academic certification. "
                "Detector probabilities never contribute positive points."
            ),
        }
    )


def _load_settings(sample: dict[str, Any], base: Path) -> tuple[Any, str | None]:
    settings = sample.get("settings")
    if isinstance(settings, str):
        path = (base / settings).resolve()
        settings = json.loads(path.read_text("utf-8"))
    if settings is None:
        return None, None
    return settings, canonical_fingerprint(settings)


def _safe_sample_id(value: Any) -> str:
    sample_id = str(value or "sample")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", sample_id).strip(".-")
    if not safe:
        raise ValueError("sample id must contain at least one letter or number")
    return safe[:100]


def evaluate_pair(
    sample: dict[str, Any],
    *,
    base: Path,
    output_dir: Path,
    default_duration: float = DEFAULT_DURATION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    sample_id = _safe_sample_id(sample.get("id"))
    reference = (base / str(sample["reference"])).resolve()
    processed = (base / str(sample["processed"])).resolve()
    duration = float(sample.get("duration_seconds", default_duration))
    reference_offset = float(sample.get("reference_offset_seconds", 0.0))
    processed_offset = float(sample.get("processed_offset_seconds", 0.0))
    if min(duration, reference_offset, processed_offset) < 0:
        raise ValueError(f"{sample_id}: duration and offsets cannot be negative")
    settings, settings_hash = _load_settings(sample, base)
    normalized_sample = dict(sample)
    if settings_hash:
        normalized_sample["configuration_fingerprint"] = settings_hash
    sample_dir = output_dir / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    reference_info = probe_media(reference)
    processed_info = probe_media(processed)
    full = full_reference_metrics(
        reference,
        processed,
        reference_info,
        reference_offset=reference_offset,
        processed_offset=processed_offset,
        duration=duration,
    )
    raw_trace = temporal_trace(
        reference,
        offset=reference_offset,
        duration=duration,
        max_frames=max_frames,
    )
    processed_trace = temporal_trace(
        processed,
        offset=processed_offset,
        duration=duration,
        max_frames=max_frames,
    )
    temporal_pair = compare_temporal(raw_trace, processed_trace)
    timestamps = {
        "reference": probe_timestamps(
            reference,
            offset=reference_offset,
            duration=duration,
            max_frames=max_frames,
        ),
        "processed": probe_timestamps(
            processed,
            offset=processed_offset,
            duration=duration,
            max_frames=max_frames,
        ),
    }
    provenance = inspect_asset(processed)
    manifest_path = sample_dir / "c2pa-manifest.json"
    write_manifest(manifest_path, build_manifest(processed.name))
    assessment = responsible_release_assessment(
        reference_info,
        processed_info,
        full,
        processed_trace.report,
        temporal_pair,
        timestamps["processed"],
        normalized_sample,
        provenance,
    )
    runtime_quality: dict[str, Any] | None = None
    if normalized_sample.get("quality_history"):
        history_path = (base / str(normalized_sample["quality_history"])).resolve()
        runtime_quality = {
            "path": str(history_path),
            "sha256": sha256_file(history_path),
            "summary": summarize_quality_history(read_history(history_path)),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "environment": runtime_environment(),
        "sample_id": sample_id,
        "sample": {
            key: value
            for key, value in normalized_sample.items()
            if key not in {"reference", "processed", "settings"}
        },
        "configuration": settings,
        "configuration_fingerprint": settings_hash,
        "reference": reference_info,
        "processed": processed_info,
        "full_reference_metrics": full,
        "temporal": {
            "reference": raw_trace.report,
            "processed": processed_trace.report,
            "comparison": temporal_pair,
            "timestamps": timestamps,
        },
        "runtime_quality_history": runtime_quality,
        "provenance": {
            **provenance,
            "generated_template": str(manifest_path.resolve()),
            "template_is_signed": False,
        },
        "responsible_release": assessment,
        "limitations": [
            "Full-reference metrics include the intentionally replaced face region.",
            "Objective metrics do not replace a blinded ITU-style subjective panel.",
            "A C2PA JSON template is not trusted until signed and independently verified.",
            "Detector outputs assess forensic exposure and are never a quality target.",
        ],
    }
    atomic_json(sample_dir / "report.json", result)
    atomic_text(sample_dir / "report.md", pair_markdown(result))
    return _json_number(result)


def _display(value: Any) -> str:
    if value is None:
        return "not evaluated"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def pair_markdown(result: dict[str, Any]) -> str:
    assessment = result["responsible_release"]
    full = result["full_reference_metrics"]
    processed = result["temporal"]["processed"]
    lines = [
        f"# Media integrity report: {result['sample_id']}",
        "",
        f"Responsible-release status: **{assessment['status']}** — "
        f"{assessment['score']}/100 ({assessment['grade']}).",
        "",
        "This is an engineering readiness score, not a detector-evasion score or "
        "academic certification. Detector probabilities contribute no points.",
        "",
        "## Objective and temporal measurements",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| VMAF | {_display(full['vmaf'].get('mean'))} |",
        f"| SSIM | {_display(full['ssim'].get('mean'))} |",
        f"| PSNR | {_display(full['psnr'].get('mean'))} |",
        f"| Exact repeated frames | {_display(processed.get('exact_repeat_percent'))}% |",
        f"| Longest exact repeat run | {_display(processed.get('longest_exact_repeat_run'))} |",
        f"| Mean blockiness ratio | {_display(processed.get('blockiness_ratio_mean'))} |",
        "",
        "## Release criteria",
        "",
        "| Criterion | Target | Result | Points |",
        "|---|---|---:|---:|",
    ]
    for item in assessment["criteria"]:
        verdict = "pass" if item["passed"] else (
            "not evaluated" if not item["available"] else "fail"
        )
        lines.append(
            f"| {item['name']} | {item['target']} | {verdict} | "
            f"{item['points']}/{item['weight']} |"
        )
    if assessment["hard_failures"]:
        lines.extend(
            (
                "",
                "## Blocking release issues",
                "",
                *[f"- {failure}" for failure in assessment["hard_failures"]],
            )
        )
    lines.extend(
        (
            "",
            "## Provenance",
            "",
            f"C2PA tool available: {_display(result['provenance']['tool_available'])}  ",
            f"Signed manifest valid: {_display(result['provenance']['valid'])}  ",
            f"Unsigned template: `{result['provenance']['generated_template']}`",
            "",
            "## Interpretation limits",
            "",
            *[f"- {item}" for item in result["limitations"]],
        )
    )
    return "\n".join(lines)


def _coverage_report(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def distinct(key: str) -> set[str]:
        return {str(sample[key]) for sample in samples if sample.get(key) not in (None, "")}

    requirements = {
        "samples": (len(samples), 60),
        "pseudonymous identities": (len(distinct("identity_id")), 30),
        "capture devices": (len(distinct("device")), 2),
        "lighting conditions": (len(distinct("lighting")), 3),
        "motion conditions": (len(distinct("motion")), 3),
        "face-size bins": (len(distinct("face_size")), 3),
        "compression conditions": (len(distinct("compression")), 2),
    }
    checks = [
        {
            "name": name,
            "observed": observed,
            "minimum": minimum,
            "passed": observed >= minimum,
        }
        for name, (observed, minimum) in requirements.items()
    ]
    checks.extend(
        (
            {
                "name": "all samples are frozen holdout",
                "observed": sum(sample.get("split") == "holdout" for sample in samples),
                "minimum": len(samples),
                "passed": all(sample.get("split") == "holdout" for sample in samples),
            },
            {
                "name": "all capture subjects consented",
                "observed": sum(
                    sample.get("capture_subject_consent") is True for sample in samples
                ),
                "minimum": len(samples),
                "passed": all(
                    sample.get("capture_subject_consent") is True for sample in samples
                ),
            },
            {
                "name": "all source identities authorized",
                "observed": sum(
                    sample.get("source_identity_authorized") is True for sample in samples
                ),
                "minimum": len(samples),
                "passed": all(
                    sample.get("source_identity_authorized") is True for sample in samples
                ),
            },
        )
    )
    passed = sum(check["passed"] for check in checks)
    return {
        "checks": checks,
        "score_percent": round(100.0 * passed / len(checks), 2),
        "complete": passed == len(checks),
        "note": (
            "Minimums are a defensible internal protocol floor, not claims made by "
            "Deepfake-Eval-2024 or NIST. Report uncertainty and subgroup results."
        ),
    }


def study_markdown(report: dict[str, Any]) -> str:
    release = report["responsible_release_summary"]
    lines = [
        f"# Study report: {report['study']['name']}",
        "",
        f"Samples completed: **{release['samples']}**  ",
        f"Mean responsible-release score: **{_display(release['mean_score'])}**  ",
        f"Release-blocked samples: **{release['blocked']}**  ",
        f"Protocol coverage: **{report['protocol_coverage']['score_percent']}%**",
        "",
        "Detector results, when present, evaluate the detectors and forensic exposure. "
        "They do not contribute to the release score.",
        "",
        "## Samples",
        "",
        "| Sample | Device | Status | Score |",
        "|---|---|---|---:|",
    ]
    for sample in report["samples"]:
        summary = sample["responsible_release"]
        lines.append(
            f"| {sample['sample_id']} | {sample['sample'].get('device', 'unreported')} "
            f"| {summary['status']} | {summary['score']} |"
        )
    lines.extend(("", "## Protocol coverage", "", "| Check | Observed | Minimum |", "|---|---:|---:|"))
    for check in report["protocol_coverage"]["checks"]:
        lines.append(
            f"| {check['name']} | {check['observed']} | {check['minimum']} |"
        )
    if report.get("detector_evaluation"):
        lines.extend(
            (
                "",
                "## Detector evaluation",
                "",
                "These are manipulated-class detector metrics, reported for auditability.",
                "",
                "| Detector | AUC | AP | EER | CDR@FAR 5% |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for name, detector in report["detector_evaluation"]["detectors"].items():
            metrics = detector["overall"]
            lines.append(
                f"| {name} | {metrics['roc_auc']} | {metrics['average_precision']} "
                f"| {metrics['equal_error_rate']} | {metrics['cdr_at_far']} |"
            )
    if report.get("subjective_evaluation"):
        subjective = report["subjective_evaluation"]
        lines.extend(
            (
                "",
                "## Blinded subjective evaluation",
                "",
                f"Valid rating rows: **{subjective['rows_included']}** from "
                f"**{subjective['unique_raters']}** pseudonymous raters.  ",
                f"Attention-check exclusions: **{subjective['rows_excluded_attention_check']}**.",
                "",
                "Paired degradation is positive when processed media was rated lower.",
                "",
                "| Dimension | Paired ratings | Mean degradation | 95% CI |",
                "|---|---:|---:|---:|",
            )
        )
        for dimension, values in subjective["paired_degradation"].items():
            interval = values["confidence_interval_95"]
            lines.append(
                f"| {dimension} | {values['paired_ratings']} | "
                f"{values['mean_degradation']} | {interval[0]} to {interval[1]} |"
            )
    return "\n".join(lines)


def evaluate_study(
    manifest_path: Path,
    output_dir: Path,
    *,
    bootstrap: int = 1000,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("study manifest must be a JSON object")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("study manifest requires a non-empty samples list")
    study = manifest.get("study", {})
    if not isinstance(study, dict):
        raise ValueError("study must be an object")
    study.setdefault("name", manifest_path.stem)
    if study.get("protocol") != "frozen-holdout":
        raise ValueError("study.protocol must be 'frozen-holdout'")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = manifest_path.resolve().parent
    results = [
        evaluate_pair(
            sample,
            base=base,
            output_dir=output_dir,
            default_duration=float(manifest.get("duration_seconds", DEFAULT_DURATION)),
            max_frames=max_frames,
        )
        for sample in samples
    ]
    scores = [float(result["responsible_release"]["score"]) for result in results]
    by_device: dict[str, list[float]] = defaultdict(list)
    for result, score in zip(results, scores):
        by_device[str(result["sample"].get("device", "unreported"))].append(score)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": study,
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_coverage": _coverage_report(samples),
        "responsible_release_summary": {
            "samples": len(results),
            "mean_score": fmean(scores),
            "minimum_score": min(scores),
            "maximum_score": max(scores),
            "blocked": sum(
                result["responsible_release"]["status"] == "blocked"
                for result in results
            ),
            "by_device_mean": {
                device: fmean(values) for device, values in sorted(by_device.items())
            },
            "detector_metrics_included": False,
        },
        "samples": results,
        "detector_evaluation": None,
        "subjective_evaluation": None,
    }
    detector_path = manifest.get("detector_results")
    if detector_path:
        rows = load_rows((base / str(detector_path)).resolve())
        report["detector_evaluation"] = evaluate_rows(
            rows, bootstrap=bootstrap
        )
    subjective_path = manifest.get("subjective_results")
    if subjective_path:
        report["subjective_evaluation"] = evaluate_subjective_rows(
            load_subjective_rows((base / str(subjective_path)).resolve())
        )
    report = _json_number(report)
    atomic_json(output_dir / "study-report.json", report)
    atomic_text(output_dir / "study-report.md", study_markdown(report))
    return report


def _pair_sample_from_args(args: argparse.Namespace) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "id": args.sample_id,
        "reference": str(args.reference),
        "processed": str(args.processed),
        "duration_seconds": args.duration,
        "reference_offset_seconds": args.reference_offset,
        "processed_offset_seconds": args.processed_offset,
        "intended_use": args.intended_use,
        "split": args.split,
        "capture_subject_consent": args.capture_subject_consent,
        "source_identity_authorized": args.source_identity_authorized,
        "device": args.device,
    }
    if args.settings:
        sample["settings"] = str(args.settings.resolve())
    if args.quality_history:
        sample["quality_history"] = str(args.quality_history.resolve())
    return sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair = subparsers.add_parser("pair", help="evaluate one aligned clip pair")
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--processed", type=Path, required=True)
    pair.add_argument("--output-dir", type=Path, required=True)
    pair.add_argument("--sample-id", default="pair")
    pair.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    pair.add_argument("--reference-offset", type=float, default=0.0)
    pair.add_argument("--processed-offset", type=float, default=0.0)
    pair.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    pair.add_argument("--settings", type=Path)
    pair.add_argument(
        "--quality-history",
        type=Path,
        help="optional Windows quality-history JSONL captured for this clip",
    )
    pair.add_argument("--device", default="unreported")
    pair.add_argument(
        "--intended-use", choices=("live", "recorded", "internal"), default="internal"
    )
    pair.add_argument("--split", choices=("development", "holdout", "diagnostic"), default="diagnostic")
    pair.add_argument("--capture-subject-consent", action="store_true")
    pair.add_argument("--source-identity-authorized", action="store_true")

    study = subparsers.add_parser("study", help="run a frozen study manifest")
    study.add_argument("--manifest", type=Path, required=True)
    study.add_argument("--output-dir", type=Path, required=True)
    study.add_argument("--bootstrap", type=int, default=1000)
    study.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)

    detectors = subparsers.add_parser(
        "detectors", help="evaluate labeled detector probability exports"
    )
    detectors.add_argument("--input", type=Path, required=True)
    detectors.add_argument("--output", type=Path, required=True)
    detectors.add_argument("--threshold", type=float, default=0.5)
    detectors.add_argument("--far", type=float, default=0.05)
    detectors.add_argument("--bootstrap", type=int, default=1000)
    detectors.add_argument("--seed", type=int, default=2026)

    template = subparsers.add_parser(
        "c2pa-template", help="write an unsigned C2PA manifest template"
    )
    template.add_argument("--title", required=True)
    template.add_argument("--output", type=Path, required=True)

    sign = subparsers.add_parser(
        "sign", help="sign a recorded asset offline with explicit credentials"
    )
    sign.add_argument("--asset", type=Path, required=True)
    sign.add_argument("--parent", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)
    sign.add_argument("--certificate", type=Path, required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--manifest", type=Path)
    sign.add_argument("--algorithm", default="es256")
    sign.add_argument("--c2patool", default="c2patool")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pair":
            if args.duration < 0 or args.max_frames < 2:
                parser.error("duration must be non-negative and max-frames at least 2")
            report = evaluate_pair(
                _pair_sample_from_args(args),
                base=Path.cwd(),
                output_dir=args.output_dir.resolve(),
                max_frames=args.max_frames,
            )
            print(json.dumps(report["responsible_release"], indent=2))
            return 0 if report["responsible_release"]["status"] == "pass" else 2
        if args.command == "study":
            report = evaluate_study(
                args.manifest.resolve(),
                args.output_dir.resolve(),
                bootstrap=args.bootstrap,
                max_frames=args.max_frames,
            )
            print(json.dumps(report["responsible_release_summary"], indent=2))
            return 0 if report["responsible_release_summary"]["blocked"] == 0 else 2
        if args.command == "detectors":
            result = evaluate_rows(
                load_rows(args.input),
                threshold=args.threshold,
                far=args.far,
                bootstrap=args.bootstrap,
                seed=args.seed,
            )
            atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "c2pa-template":
            write_manifest(args.output, build_manifest(args.title))
            print(args.output.resolve())
            return 0
        if args.command == "sign":
            executable = shutil.which(args.c2patool)
            if not executable:
                raise ValueError(f"c2patool is not available: {args.c2patool}")
            manifest_path = args.manifest or args.output.with_suffix(
                args.output.suffix + ".manifest.json"
            )
            command, manifest = signing_command(
                args.asset,
                args.output,
                manifest_path,
                parent=args.parent,
                certificate=args.certificate,
                private_key=args.private_key,
                algorithm=args.algorithm,
                c2patool=executable,
            )
            write_manifest(manifest_path, manifest)
            with tempfile.TemporaryDirectory(
                prefix="dlc-c2pa-sign-offline-"
            ) as directory:
                offline_settings = Path(directory) / "c2pa.toml"
                offline_settings.write_text(OFFLINE_SETTINGS, encoding="utf-8")
                command.extend(("--settings", str(offline_settings)))
                completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
            verification = inspect_asset(args.output, executable)
            print(json.dumps(_json_number(verification), indent=2))
            return 0 if verification.get("valid") else 3
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
