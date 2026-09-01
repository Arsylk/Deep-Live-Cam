from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.c2pa_provenance import (
    AI_EDIT_SOURCE_TYPE,
    OFFLINE_SETTINGS,
    build_manifest,
    signing_command,
    validate_manifest,
)
from tools.media_integrity_benchmark import (
    compare_temporal,
    full_reference_metrics,
    probe_media,
    responsible_release_assessment,
    temporal_trace,
)


def test_c2pa_template_discloses_ai_edit():
    manifest = build_manifest("processed.mp4")

    assert validate_manifest(manifest) == []
    action = manifest["assertions"][0]["data"]["actions"][0]
    assert action["action"] == "c2pa.edited"
    assert action["digitalSourceType"] == AI_EDIT_SOURCE_TYPE
    assert "Face identity" in action["description"]
    assert "private_key" not in manifest
    assert "remote_manifest_fetch = false" in OFFLINE_SETTINGS


def test_offline_signing_command_requires_explicit_credentials(tmp_path):
    paths = {
        name: tmp_path / name
        for name in ("processed.mp4", "raw.mp4", "chain.pem", "key.pem")
    }
    for path in paths.values():
        path.write_bytes(b"fixture")
    output = tmp_path / "signed.mp4"
    manifest_path = tmp_path / "manifest.json"

    command, manifest = signing_command(
        paths["processed.mp4"],
        output,
        manifest_path,
        parent=paths["raw.mp4"],
        certificate=paths["chain.pem"],
        private_key=paths["key.pem"],
    )

    assert "--parent" in command
    assert "--output" in command
    assert "ta_url" not in manifest
    assert manifest["sign_cert"].endswith("chain.pem")
    assert manifest["private_key"].endswith("key.pem")


def test_live_release_grade_does_not_require_file_metadata():
    media = {"video": {"width": 1280, "height": 720, "fps": 30.0}}
    report = responsible_release_assessment(
        media,
        media,
        {
            "vmaf": {"available": True, "mean": 95.0},
            "ssim": {"available": True, "mean": 0.98},
            "psnr": {"available": True, "mean": 35.0},
        },
        {"exact_repeat_percent": 0.0},
        {"available": True, "artificial_repeat_percent": 0.0},
        {"available": True, "estimated_missing_percent": 0.0},
        {
            "intended_use": "live",
            "capture_subject_consent": True,
            "source_identity_authorized": True,
            "split": "holdout",
            "configuration_fingerprint": "abc",
        },
        {"valid": None},
    )

    assert report["score"] == 100.0
    assert report["status"] == "pass"
    assert report["detector_metrics_included"] is False


def test_recorded_release_is_blocked_without_signed_c2pa():
    media = {"video": {"width": 1280, "height": 720, "fps": 30.0}}
    report = responsible_release_assessment(
        media,
        media,
        {
            "vmaf": {"available": True, "mean": 95.0},
            "ssim": {"available": True, "mean": 0.98},
            "psnr": {"available": True, "mean": 35.0},
        },
        {"exact_repeat_percent": 0.0},
        {"available": True, "artificial_repeat_percent": 0.0},
        {"available": True, "estimated_missing_percent": 0.0},
        {
            "intended_use": "recorded",
            "capture_subject_consent": True,
            "source_identity_authorized": True,
            "split": "holdout",
            "configuration_fingerprint": "abc",
        },
        {"valid": None},
    )

    assert report["status"] == "blocked"
    assert "valid C2PA manifest for recorded release" in report["hard_failures"]


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg integration tools unavailable",
)
def test_identical_synthetic_clips_pass_reference_metrics(tmp_path):
    reference = tmp_path / "reference.mp4"
    processed = tmp_path / "processed.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=10",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(reference),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        pytest.skip(f"test encoder unavailable: {completed.stderr}")
    shutil.copyfile(reference, processed)

    info = probe_media(reference)
    metrics = full_reference_metrics(
        reference, processed, info, duration=1.0
    )
    raw = temporal_trace(reference, duration=1.0, max_frames=30)
    returned = temporal_trace(processed, duration=1.0, max_frames=30)
    temporal = compare_temporal(raw, returned)

    assert metrics["ssim"]["mean"] == pytest.approx(1.0)
    assert metrics["psnr"]["mean"] == float("inf")
    if metrics["vmaf"]["available"]:
        assert metrics["vmaf"]["mean"] > 99.0
    assert temporal["artificial_repeat_frames"] == 0
    assert temporal["motion_retention_ratio_median"] == pytest.approx(1.0)
