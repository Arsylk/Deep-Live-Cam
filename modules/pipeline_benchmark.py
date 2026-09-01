"""Lossless, synchronized benchmark capture for live camera processors.

The recorder is deliberately attached *after* a live processor has received a
frame.  It therefore never opens a camera, network socket, or preview port and
cannot compete with the workers that own those resources.  A request file is
used so a long-running service can capture a bounded sample without a restart.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCHEMA_VERSION = "1.0"
CAPTURE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_STOP = object()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("benchmark request must be a JSON object")
    capture_id = str(value.get("id", "")).strip()
    if not CAPTURE_ID.fullmatch(capture_id):
        raise ValueError(
            "benchmark id must be 1-80 safe filename characters"
        )
    frame_count = int(value.get("frame_count", 50))
    sample_fps = float(value.get("sample_fps", 5.0))
    if not 10 <= frame_count <= 600:
        raise ValueError("frame_count must be between 10 and 600")
    if not 0.5 <= sample_fps <= 30.0:
        raise ValueError("sample_fps must be between 0.5 and 30")
    role = str(value.get("role", "candidate")).strip().lower()
    if role not in {"baseline", "candidate"}:
        raise ValueError("role must be baseline or candidate")
    token = str(value.get("token") or f"{capture_id}-{time.time_ns()}")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": capture_id,
        "token": token[:160],
        "role": role,
        "frame_count": frame_count,
        "sample_fps": sample_fps,
        "requested_at": value.get("requested_at") or time.time(),
        "notes": str(value.get("notes", ""))[:1000],
    }


class PairedBenchmarkRecorder:
    """Record exact pre/post processor frame pairs without owning the source."""

    def __init__(
        self,
        state_dir: str | Path,
        context_supplier: Callable[[], dict[str, Any]] | None = None,
        start_callback: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(state_dir).expanduser().resolve() / "benchmarks"
        self.request_path = self.root / "request.json"
        self.status_path = self.root / "status.json"
        self.captures_dir = self.root / "captures"
        self.context_supplier = context_supplier
        self.start_callback = start_callback
        self._lock = threading.Lock()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=12)
        self._writer: threading.Thread | None = None
        self._request: dict[str, Any] | None = None
        self._capture_dir: Path | None = None
        self._accepted = 0
        self._accepting = False
        self._queue_drops = 0
        self._first_sample_at = 0.0
        self._last_sample_at = 0.0
        self._next_sample_at = 0.0
        self._closed = False
        self._last_status: dict[str, Any] = {
            "state": "idle",
            "request_path": str(self.request_path),
            "captures_dir": str(self.captures_dir),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.status_path, self._last_status)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._last_status)

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self._last_status = {**self._last_status, **values}
            document = copy.deepcopy(self._last_status)
        _atomic_json(self.status_path, document)

    def request(self, value: Any) -> dict[str, Any]:
        request = validate_request(value)
        with self._lock:
            if self._closed:
                raise RuntimeError("benchmark recorder is closed")
            if self._request is not None:
                raise RuntimeError("a benchmark capture is already active")
            destination = self.captures_dir / request["id"]
            if destination.exists():
                raise FileExistsError(
                    f"capture {request['id']} already exists; baselines are immutable"
                )
            destination.mkdir(parents=True, exist_ok=False)
            self._request = request
            self._capture_dir = destination
            self._accepted = 0
            self._accepting = True
            self._queue_drops = 0
            self._first_sample_at = 0.0
            self._last_sample_at = 0.0
            self._next_sample_at = 0.0
            self._queue = queue.Queue(maxsize=12)
            if self.start_callback is not None:
                self.start_callback()
            context = (
                copy.deepcopy(self.context_supplier())
                if self.context_supplier is not None
                else {}
            )
            self._writer = threading.Thread(
                target=self._write_capture,
                args=(destination, copy.deepcopy(request), context),
                name=f"benchmark-writer-{request['id']}",
                daemon=True,
            )
            self._writer.start()
        self._set_status(
            state="recording",
            id=request["id"],
            token=request["token"],
            role=request["role"],
            accepted_frames=0,
            target_frames=request["frame_count"],
            queue_drops=0,
            capture_dir=str(destination),
            error=None,
        )
        return self.status()

    def _poll_request(self) -> bool:
        if not self.request_path.is_file():
            return False
        accepted = False
        try:
            value = json.loads(self.request_path.read_text("utf-8"))
            self.request(value)
            accepted = True
        except Exception as exc:
            self._set_status(
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            try:
                self.request_path.unlink()
            except FileNotFoundError:
                pass
        return accepted

    def observe(
        self,
        reference: np.ndarray,
        processed: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Accept a synchronized frame pair; safe to call on every output."""
        if self._closed:
            return
        if self._request is None:
            # The request callback resets diagnostic aggregates. Start with
            # the *next* processor frame because this call's metadata was
            # sampled immediately before the reset.
            if self._poll_request():
                return
        request = self._request
        if request is None or not self._accepting:
            return
        now = time.monotonic()
        period = 1.0 / request["sample_fps"]
        # Accept a frame slightly before the ideal deadline. Without this
        # tolerance, a ~5 FPS processor sampled at 5 FPS can alias to 2.5 FPS
        # whenever ordinary scheduler jitter puts each next frame a few
        # milliseconds early.
        early_tolerance = min(0.05, period * 0.25)
        if self._next_sample_at and now + early_tolerance < self._next_sample_at:
            return
        if (
            not isinstance(reference, np.ndarray)
            or not isinstance(processed, np.ndarray)
            or reference.ndim != 3
            or processed.ndim != 3
            or reference.shape != processed.shape
            or reference.dtype != np.uint8
            or processed.dtype != np.uint8
        ):
            self._fail("invalid or mismatched frame pair")
            return
        if self._queue.full():
            self._queue_drops += 1
            self._next_sample_at = now + period
            self._set_status(queue_drops=self._queue_drops)
            return

        if not self._first_sample_at:
            self._first_sample_at = now
        self._last_sample_at = now
        record = {
            "index": self._accepted,
            "elapsed_seconds": round(now - self._first_sample_at, 6),
            "captured_at": time.time(),
            **copy.deepcopy(metadata or {}),
        }
        # Copy after the queue capacity check. The live processor reuses or
        # mutates frame arrays on later iterations, while the writer is async.
        self._queue.put_nowait((reference.copy(), processed.copy(), record))
        self._accepted += 1
        if not self._next_sample_at:
            self._next_sample_at = now + period
        else:
            self._next_sample_at += period
            if self._next_sample_at < now - period:
                self._next_sample_at = now + period
        self._set_status(
            accepted_frames=self._accepted,
            queue_drops=self._queue_drops,
        )
        if self._accepted >= request["frame_count"]:
            self._accepting = False
            self._set_status(state="finalizing")
            self._queue.put(_STOP)

    def _write_capture(
        self,
        destination: Path,
        request: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        reference_path = destination / "reference.mkv"
        processed_path = destination / "processed.mkv"
        reference_spool = destination / ".reference.bgr24.tmp"
        processed_spool = destination / ".processed.bgr24.tmp"
        metrics_path = destination / "quality-history.jsonl"
        written = 0
        dimensions: tuple[int, int] | None = None
        reference_content = hashlib.sha256()
        processed_content = hashlib.sha256()
        started_wall = time.time()
        try:
            _atomic_json(destination / "request.json", request)
            _atomic_json(destination / "settings.json", context)
            # Spool raw bytes while measuring. FFV1 compression happens only
            # after the requested window, so benchmark timings are not skewed
            # by two concurrent software encoders.
            with (
                metrics_path.open("w", encoding="utf-8") as metrics,
                reference_spool.open("wb") as raw_reference,
                processed_spool.open("wb") as raw_processed,
            ):
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        break
                    reference, processed, record = item
                    height, width = reference.shape[:2]
                    if dimensions is None:
                        dimensions = (width, height)
                    elif dimensions != (width, height):
                        raise RuntimeError("frame dimensions changed during capture")
                    reference_bytes = reference.tobytes(order="C")
                    processed_bytes = processed.tobytes(order="C")
                    raw_reference.write(reference_bytes)
                    raw_processed.write(processed_bytes)
                    reference_content.update(reference_bytes)
                    processed_content.update(processed_bytes)
                    metrics.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
            if written != request["frame_count"]:
                raise RuntimeError(
                    f"capture wrote {written} of {request['frame_count']} frames"
                )
            if dimensions is None:
                raise RuntimeError("capture contains no frames")
            self._encode_spool(
                reference_spool,
                reference_path,
                dimensions,
                request["sample_fps"],
                written,
            )
            self._encode_spool(
                processed_spool,
                processed_path,
                dimensions,
                request["sample_fps"],
                written,
            )
            reference_spool.unlink()
            processed_spool.unlink()
            elapsed = max(0.0, self._last_sample_at - self._first_sample_at)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "id": request["id"],
                "token": request["token"],
                "role": request["role"],
                "lossless": True,
                "codec": "ffv1",
                "sample_fps": request["sample_fps"],
                "frames": written,
                "nominal_duration_seconds": round(
                    written / request["sample_fps"], 6
                ),
                "observed_span_seconds": round(elapsed, 6),
                "observed_sample_fps": round(
                    (written - 1) / elapsed if written > 1 and elapsed else 0.0,
                    6,
                ),
                "queue_drops": self._queue_drops,
                "started_at": started_wall,
                "completed_at": time.time(),
                "files": {
                    "reference": {
                        "path": reference_path.name,
                        "bytes": reference_path.stat().st_size,
                        "sha256": _sha256(reference_path),
                        "frame_content_sha256": reference_content.hexdigest(),
                        "frame_content_format": "bgr24-frame-sequence-v1",
                    },
                    "processed": {
                        "path": processed_path.name,
                        "bytes": processed_path.stat().st_size,
                        "sha256": _sha256(processed_path),
                        "frame_content_sha256": processed_content.hexdigest(),
                        "frame_content_format": "bgr24-frame-sequence-v1",
                    },
                    "quality_history": {
                        "path": metrics_path.name,
                        "bytes": metrics_path.stat().st_size,
                        "sha256": _sha256(metrics_path),
                    },
                    "settings": {
                        "path": "settings.json",
                        "sha256": _sha256(destination / "settings.json"),
                    },
                },
                "context": context,
            }
            _atomic_json(destination / "capture.json", manifest)
            self._set_status(
                state="complete",
                written_frames=written,
                capture_manifest=str(destination / "capture.json"),
                completed_at=manifest["completed_at"],
            )
        except Exception as exc:
            self._set_status(
                state="failed", error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            with self._lock:
                self._request = None
                self._capture_dir = None
                self._accepting = False

    @staticmethod
    def _encode_spool(
        source: Path,
        destination: Path,
        dimensions: tuple[int, int],
        fps: float,
        frames: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for lossless benchmark capture")
        width, height = dimensions
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                str(source),
                "-frames:v",
                str(frames),
                "-an",
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
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg FFV1 encode failed: {completed.stderr.strip()[:1000]}"
            )

    def _fail(self, detail: str) -> None:
        self._accepting = False
        self._set_status(state="failed", error=detail)
        try:
            self._queue.put(_STOP, timeout=2.0)
        except queue.Full:
            pass

    def close(self) -> None:
        self._closed = True
        writer = self._writer
        if self._request is not None:
            self._fail("service stopped before capture completed")
        if writer is not None and writer.is_alive():
            writer.join(timeout=10.0)
