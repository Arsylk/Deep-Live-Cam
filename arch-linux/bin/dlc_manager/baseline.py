#!/usr/bin/env python3
"""Read the active reproducible quality baseline.

The pointer file names one capture run; the run and its settings hold the
metrics and the camera profile they were measured with.  Reading is tolerant:
a missing or partial baseline degrades to an explicit "not registered" state
rather than an exception or a fabricated number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .health import read_json


DEFAULT_POINTER = (
    Path("arch-linux")
    / "runtime"
    / "android-phone-processed"
    / "benchmarks"
    / "active-baseline.json"
)

# Read verbatim from the run when present; these fixed sentences are the
# fallback so the limitation is never silently dropped.
DEFAULT_INTERPRETATION = {
    "identity": (
        "a measured face-core change confirms a visual effect, not identity "
        "similarity"
    ),
    "full_reference_metrics": (
        "VMAF, SSIM, and PSNR measure signal and background preservation only; "
        "the intended identity edit lowers them by design"
    ),
    "definitive_cross_pipeline_comparison": (
        "requires replaying this exact frozen decoded reference corpus through "
        "each candidate"
    ),
}


@dataclass(frozen=True)
class BaselineView:
    available: bool
    identifier: str = "none"
    run_path: str = ""
    camera_profile: str = "not recorded"
    camera_mode: str = "not recorded"
    model: str = "not recorded"
    backend: str = "not recorded"
    processing_fps: float | None = None
    stability_score: float | None = None
    detection_miss_percent: float | None = None
    seam_ring_delta_lab: float | None = None
    frame_count: int | None = None
    pairing: str = "not recorded"
    interpretation: Mapping[str, str] = field(default_factory=dict)
    error: str = ""

    def metrics(self) -> tuple[tuple[str, str, str], ...]:
        """Label, value, and plain-language meaning for each headline metric."""
        return (
            (
                "Processing rate",
                _decimal(self.processing_fps, 2, " FPS"),
                "Unique processed frames per second during the capture.",
            ),
            (
                "Face stability",
                _decimal(self.stability_score, 1, " / 100"),
                "Higher means less frame-to-frame flicker on the swapped face.",
            ),
            (
                "Detector misses",
                _decimal(self.detection_miss_percent, 2, " %"),
                "Share of frames where the detector found no face.",
            ),
            (
                "Seam colour delta",
                _decimal(self.seam_ring_delta_lab, 2, " ΔE"),
                "Colour difference across the blend ring; lower blends better.",
            ),
            (
                "Compared frames",
                "not recorded" if self.frame_count is None else f"{self.frame_count:,}",
                "Size of the frozen reference corpus used for this baseline.",
            ),
        )


def _decimal(value: float | None, places: int, suffix: str = "") -> str:
    if value is None:
        return "not recorded"
    return f"{value:.{places}f}{suffix}"


def _float(values: Mapping[str, Any], key: str) -> float | None:
    try:
        return float(values[key])
    except (KeyError, TypeError, ValueError):
        return None


def default_pointer_path(repository_root: Path | None = None) -> Path:
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return root / DEFAULT_POINTER


def load_active_baseline(pointer_path: Path | None = None) -> BaselineView:
    """Load the registered baseline, or explain why there is none."""
    path = Path(pointer_path) if pointer_path is not None else default_pointer_path()
    pointer = read_json(path)
    if not pointer:
        return BaselineView(
            available=False,
            error=f"No active baseline pointer at {path}.",
        )
    run_value = pointer.get("run")
    if not run_value:
        return BaselineView(
            available=False,
            identifier=str(pointer.get("id", "unknown")),
            error="The baseline pointer names no capture run.",
        )
    run_path = Path(str(run_value))
    run = read_json(run_path)
    if not run:
        return BaselineView(
            available=False,
            identifier=str(pointer.get("id", "unknown")),
            run_path=str(run_path),
            error=f"The registered run document is unreadable: {run_path}.",
        )
    analysis = run.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    metrics = analysis.get("metric_vector")
    metrics = metrics if isinstance(metrics, dict) else {}
    contract = run.get("comparison_contract")
    contract = contract if isinstance(contract, dict) else {}
    interpretation = run.get("interpretation")
    interpretation = interpretation if isinstance(interpretation, dict) else {}

    settings = read_json(run_path.parent / "settings.json") or {}
    camera = settings.get("camera_profile")
    camera = camera if isinstance(camera, dict) else {}

    frames: int | None
    try:
        frames = int(contract["frames"])
    except (KeyError, TypeError, ValueError):
        frames = None

    return BaselineView(
        available=True,
        identifier=str(pointer.get("id", run.get("id", "unknown"))),
        run_path=str(run_path),
        camera_profile=str(camera.get("profile", "not recorded")),
        camera_mode=str(camera.get("camera_mode", "not recorded")),
        model=str(settings.get("model", "not recorded")),
        backend=str(settings.get("backend", "not recorded")),
        processing_fps=_float(metrics, "processing_fps"),
        stability_score=_float(metrics, "face_stability_score"),
        detection_miss_percent=_float(metrics, "detection_miss_percent"),
        seam_ring_delta_lab=_float(metrics, "seam_ring_delta_lab"),
        frame_count=frames,
        pairing=str(contract.get("pairing", "not recorded")),
        interpretation={**DEFAULT_INTERPRETATION, **{
            key: str(value) for key, value in interpretation.items()
        }},
    )
