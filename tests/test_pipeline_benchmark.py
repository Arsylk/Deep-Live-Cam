from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Barrier, Thread
import time

import cv2
import numpy as np
import pytest

from modules.pipeline_benchmark import (
    PairedBenchmarkRecorder,
    _atomic_json,
    validate_request,
)
from tools.pipeline_baseline import compare_runs


def wait_for(recorder: PairedBenchmarkRecorder, state: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = recorder.status()
        if status.get("state") == state:
            return status
        if status.get("state") == "failed":
            raise AssertionError(status.get("error"))
        time.sleep(0.02)
    raise AssertionError(f"recorder did not reach {state}: {recorder.status()}")


def test_recorder_copies_synchronized_pairs_and_never_opens_camera(tmp_path, monkeypatch):
    original_video_capture = cv2.VideoCapture
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recorder opened a camera or transport")
        ),
    )
    resets = []
    recorder = PairedBenchmarkRecorder(
        tmp_path,
        context_supplier=lambda: {"pipeline": "fixture"},
        start_callback=lambda: resets.append(True),
    )
    recorder.request(
        {
            "id": "fixture-baseline",
            "role": "baseline",
            "frame_count": 10,
            "sample_fps": 30,
        }
    )

    for index in range(10):
        raw = np.full((64, 96, 3), index, dtype=np.uint8)
        processed = np.full((64, 96, 3), 100 + index, dtype=np.uint8)
        recorder.observe(raw, processed, {"score": index, "benchmark": {}})
        raw[:] = 250
        processed[:] = 251
        time.sleep(0.04)

    status = wait_for(recorder, "complete")
    capture = Path(status["capture_dir"])
    manifest = json.loads((capture / "capture.json").read_text("utf-8"))
    assert resets == [True]
    assert manifest["frames"] == 10
    assert manifest["lossless"] is True
    assert manifest["queue_drops"] == 0

    raw_reader = original_video_capture(str(capture / "reference.mkv"))
    processed_reader = original_video_capture(str(capture / "processed.mkv"))
    ok_raw, first_raw = raw_reader.read()
    ok_processed, first_processed = processed_reader.read()
    raw_reader.release()
    processed_reader.release()
    assert ok_raw and ok_processed
    assert np.max(first_raw) == 0
    assert np.min(first_processed) == 100
    assert len((capture / "quality-history.jsonl").read_text("utf-8").splitlines()) == 10

    with pytest.raises(FileExistsError, match="immutable"):
        recorder.request(
            {
                "id": "fixture-baseline",
                "role": "baseline",
                "frame_count": 10,
                "sample_fps": 30,
            }
        )
    recorder.close()


def test_request_contract_rejects_unbounded_or_unsafe_values():
    with pytest.raises(ValueError, match="safe filename"):
        validate_request({"id": "../escape", "frame_count": 10})
    with pytest.raises(ValueError, match="frame_count"):
        validate_request({"id": "okay", "frame_count": 2})
    with pytest.raises(ValueError, match="sample_fps"):
        validate_request({"id": "okay", "frame_count": 10, "sample_fps": 100})


def test_atomic_json_uses_a_distinct_temporary_for_concurrent_writers(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "status.json"
    payloads = [{"writer": "first"}, {"writer": "second"}]
    barrier = Barrier(len(payloads))
    temporary_paths: list[Path] = []
    errors: list[BaseException] = []
    original_replace = os.replace

    def synchronized_replace(source, target):
        temporary_paths.append(Path(source))
        barrier.wait(timeout=2.0)
        original_replace(source, target)

    monkeypatch.setattr(
        "modules.pipeline_benchmark.os.replace",
        synchronized_replace,
    )

    def write(payload):
        try:
            _atomic_json(destination, payload)
        except BaseException as error:  # surfaced below from the worker thread
            errors.append(error)

    writers = [Thread(target=write, args=(payload,)) for payload in payloads]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=3.0)

    assert all(not writer.is_alive() for writer in writers)
    assert errors == []
    assert len(set(temporary_paths)) == len(payloads)
    assert json.loads(destination.read_text("utf-8")) in payloads
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_sampler_does_not_alias_matching_processor_cadence_to_half_rate(
    tmp_path,
):
    recorder = PairedBenchmarkRecorder(tmp_path)
    recorder.request(
        {
            "id": "cadence",
            "role": "candidate",
            "frame_count": 10,
            "sample_fps": 5.0,
        }
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(10):
        recorder.observe(frame, frame, {})
        time.sleep(0.198)

    assert recorder.status()["accepted_frames"] == 10
    wait_for(recorder, "complete")
    recorder.close()


def _run(run_id: str, reference: str, stability: float, latency: float):
    return {
        "id": run_id,
        "analysis": {
            "metric_vector": {
                "face_stability_score": stability,
                "pipeline_ms_p95": latency,
            }
        },
        "comparison_contract": {
            "reference_sha256": reference,
            "reference_content_sha256": reference,
            "frames": 50,
            "sample_fps": 5.0,
            "resolution": [1280, 720],
            "source_identity_sha256": "identity",
        },
    }


def test_comparator_never_declares_winner_for_different_live_corpus():
    result = compare_runs(
        _run("baseline", "raw-a", 70.0, 200.0),
        _run("windows", "raw-b", 90.0, 80.0),
    )

    assert result["comparable"] is False
    assert result["verdict"]["status"] == "diagnostic-only"
    assert "decoded reference corpus differs" in result["comparability_reasons"]
    assert result["changes"]["face_stability_score"]["verdict"] == "improved"


def test_comparator_uses_non_inferiority_gate_on_same_frozen_corpus():
    result = compare_runs(
        _run("baseline", "same-raw", 80.0, 200.0),
        _run("candidate", "same-raw", 70.0, 100.0),
    )

    assert result["comparable"] is True
    assert result["verdict"]["status"] == "rejected"
    assert result["verdict"]["critical_regressions"] == [
        "face_stability_score"
    ]
