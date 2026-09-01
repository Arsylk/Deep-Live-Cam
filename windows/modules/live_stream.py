"""FFmpeg-backed SRT input/output for the remote live-camera prototype."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class RoutedFrame:
    """A decoded frame tagged with the route generation that produced it."""

    frame: np.ndarray
    route_token: int


class LatestFrame:
    """A one-item, overwrite-on-publish frame buffer."""

    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self.published = 0
        self.overwritten = 0
        self.cleared = 0

    def put(self, frame: object, route_token: int | None = None) -> None:
        # ``route_token`` is accepted for destination compatibility.  A plain
        # buffer does not route; FrameFanout performs the generation check.
        del route_token
        self.published += 1
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            self.overwritten += 1
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def get(self, timeout: float = 0.1) -> object:
        return self._queue.get(timeout=timeout)

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self.cleared += 1
            except queue.Empty:
                return


class FrameFanout:
    """Publish each current-route frame to independent overwrite buffers.

    Consumers never compete for a shared queue, and a late result from the
    previously selected device cannot leak into the new device's return.
    """

    def __init__(
        self,
        subscribers: dict[str, LatestFrame],
        current_route_token: Callable[[], int],
    ) -> None:
        self.subscribers = dict(subscribers)
        self.current_route_token = current_route_token
        self.published = 0
        self.dropped_old_route = 0

    def put(self, frame: np.ndarray, route_token: int | None = None) -> None:
        if route_token is not None and route_token != self.current_route_token():
            self.dropped_old_route += 1
            return
        for subscriber in tuple(self.subscribers.values()):
            subscriber.put(frame)
        self.published += 1

    def clear(self) -> None:
        for subscriber in self.subscribers.values():
            subscriber.clear()


def _drain_stderr(proc: subprocess.Popen, label: str) -> None:
    if proc.stderr is None:
        return
    for raw in iter(proc.stderr.readline, b""):
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            print(f"[{label}] {line}", flush=True)


def _record_cadence(worker: object, now: float, expected_fps: int) -> None:
    previous = float(getattr(worker, "_last_frame_monotonic", 0.0) or 0.0)
    setattr(worker, "_last_frame_monotonic", now)
    if not previous:
        return
    interval_ms = (now - previous) * 1000.0
    expected_ms = 1000.0 / max(1, expected_fps)
    jitter_ms = abs(interval_ms - expected_ms)
    prior_interval = float(getattr(worker, "interval_ms_ema", 0.0) or 0.0)
    prior_jitter = float(getattr(worker, "jitter_ms_ema", 0.0) or 0.0)
    setattr(
        worker,
        "interval_ms_ema",
        interval_ms if prior_interval == 0.0 else prior_interval * 0.95 + interval_ms * 0.05,
    )
    setattr(
        worker,
        "jitter_ms_ema",
        jitter_ms if prior_jitter == 0.0 else prior_jitter * 0.95 + jitter_ms * 0.05,
    )
    setattr(
        worker,
        "max_interval_ms",
        max(float(getattr(worker, "max_interval_ms", 0.0) or 0.0), interval_ms),
    )
    if interval_ms > expected_ms * 1.5:
        estimated = max(1, round(interval_ms / expected_ms) - 1)
        setattr(
            worker,
            "estimated_drops",
            int(getattr(worker, "estimated_drops", 0) or 0) + estimated,
        )


class SrtInput(threading.Thread):
    def __init__(self, url: str, width: int, height: int,
                 frames: LatestFrame, stop: threading.Event,
                 *, label: str = "input", route_token: int | None = None,
                 expected_fps: int = 30) -> None:
        super().__init__(name=f"srt-input-{label}", daemon=True)
        self.url, self.width, self.height = url, width, height
        self.frames, self.stop_event = frames, stop
        self.label, self.route_token = label, route_token
        self.expected_fps = max(1, int(expected_fps))
        self.process: Optional[subprocess.Popen] = None
        self.received = 0
        self.last_frame_at = 0.0
        self.connections = 0
        self.disconnects = 0
        self._queue_published_start = int(getattr(frames, "published", 0))
        self._queue_overwritten_start = int(getattr(frames, "overwritten", 0))
        self._last_frame_monotonic = 0.0
        self.interval_ms_ema = 0.0
        self.jitter_ms_ema = 0.0
        self.max_interval_ms = 0.0
        self.estimated_drops = 0

    def _start_ffmpeg(self) -> subprocess.Popen:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-fflags", "nobuffer", "-flags", "low_delay",
            # Tight probe + no reorder buffer so the live SRT feed starts and
            # recovers quickly instead of waiting on ffmpeg's 5MB/5s defaults.
            "-probesize", "500000", "-analyzeduration", "500000",
            "-max_delay", "0",
            "-i", self.url, "-an", "-sn", "-dn",
            "-vf", f"scale={self.width}:{self.height}",
            "-fps_mode", "passthrough",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]
        print(f"[{self.label}] listening on {self.url}", flush=True)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=_drain_stderr, args=(proc, f"{self.label}-ffmpeg"),
                         daemon=True).start()
        return proc

    def run(self) -> None:
        frame_bytes = self.width * self.height * 3
        while not self.stop_event.is_set():
            proc: subprocess.Popen | None = None
            try:
                proc = self._start_ffmpeg()
                self.process = proc
                self.connections += 1
                # A reconnect is a transport event, not thousands of missing
                # camera frames. Start a fresh cadence window at its first
                # decoded frame and report the reconnect separately.
                self._last_frame_monotonic = 0.0
                assert proc.stdout is not None
                while not self.stop_event.is_set():
                    data = bytearray()
                    while len(data) < frame_bytes:
                        # Route selection may clear ``self.process`` from a
                        # different thread. Keep this immutable local handle
                        # so teardown yields EOF/OSError instead of racing into
                        # ``None.stdout``.
                        chunk = proc.stdout.read(frame_bytes - len(data))
                        if not chunk:
                            break
                        data.extend(chunk)
                    if len(data) != frame_bytes:
                        if not self.stop_event.is_set():
                            self.disconnects += 1
                        break
                    # ``data`` is a fresh bytearray for every frame.  Let the
                    # ndarray own that buffer instead of copying another
                    # 2.7 MiB at 720p; the next read allocates a new buffer.
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(
                        self.height, self.width, 3
                    )
                    packet: object = frame
                    if self.route_token is not None:
                        packet = RoutedFrame(frame, self.route_token)
                    self.frames.put(packet)
                    self.received += 1
                    self.last_frame_at = time.time()
                    _record_cadence(self, time.monotonic(), self.expected_fps)
            except Exception as exc:
                print(f"[{self.label}] {exc}", flush=True)
            finally:
                self._terminate()
            if not self.stop_event.wait(1.0):
                print(f"[{self.label}] waiting for sender/reconnecting", flush=True)

    def _terminate(self) -> None:
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def close(self) -> None:
        self._terminate()


class SrtOutput(threading.Thread):
    def __init__(self, url: str, width: int, height: int, fps: int,
                 bitrate: str, frames: LatestFrame,
                 stop: threading.Event, *, label: str = "return",
                 stale_seconds: float = 1.0,
                 encoder: str | None = None) -> None:
        super().__init__(name=f"video-output-{label}", daemon=True)
        self.url, self.width, self.height = url, width, height
        self.fps, self.bitrate = fps, bitrate
        self.frames, self.stop_event = frames, stop
        self.label = label
        self.stale_seconds = max(0.2, float(stale_seconds))
        self.process: Optional[subprocess.Popen] = None
        self.sent = 0
        self.source_frames = 0
        self.repeated_frames = 0
        self.last_frame_at = 0.0
        self.connections = 0
        self.disconnects = 0
        self._queue_published_start = int(getattr(frames, "published", 0))
        self._queue_overwritten_start = int(getattr(frames, "overwritten", 0))
        self.encoder = encoder or self._select_encoder()
        self._last_frame_monotonic = 0.0
        self.interval_ms_ema = 0.0
        self.jitter_ms_ema = 0.0
        self.max_interval_ms = 0.0
        self.estimated_drops = 0
        # Both transformations cost quality and CPU, and cadence jitter is
        # counterproductive for a bounded-latency live stream.  Keep them as
        # explicit diagnostic opt-ins only.
        self.dither = os.environ.get("DLC_DITHER", "0") == "1"
        self.jitter = os.environ.get("DLC_JITTER", "0") == "1"
        self._rng = np.random.default_rng()
        self._noise_pool = (
            self._make_noise_pool(width, height) if self.dither else []
        )
        if self.dither or self.jitter:
            print(
                f"[output] temporal dither={'on' if self.dither else 'off'} "
                f"cadence jitter={'on' if self.jitter else 'off'}",
                flush=True,
            )

    @staticmethod
    def _make_noise_pool(width: int, height: int, count: int = 32) -> list:
        """Precomputed triangular-distribution dither frames (sigma ~1.6 LSB)."""
        rng = np.random.default_rng()
        pool = []
        for _ in range(count):
            noise = rng.integers(-2, 3, size=(height, width, 3), dtype=np.int16)
            noise += rng.integers(-2, 3, size=(height, width, 3), dtype=np.int16)
            pool.append(noise)
        return pool

    def _dither(self, frame: np.ndarray) -> np.ndarray:
        """Add opt-in diagnostic dither to the outgoing frame."""
        index = int(self._rng.integers(0, len(self._noise_pool)))
        shift = int(self._rng.integers(0, self.height))
        noise = np.roll(self._noise_pool[index], shift, axis=0)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _select_encoder(self) -> str:
        """Select the first hardware encoder that actually opens.

        The RTX driver currently exposes NVENC API 13.0 while the installed
        FFmpeg build requires 13.1.  The laptop's AMD GPU provides AMF, which
        is a safe hardware fallback and avoids burning CPU on libx264.
        """
        candidates = (
            ("h264_nvenc", ["-preset", "p1", "-tune", "ull"]),
            ("h264_amf", ["-usage", "lowlatency", "-quality", "quality"]),
        )
        for encoder, options in candidates:
            probe = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                # AMF rejects tiny probe frames even though it supports the
                # configured live format. Probe the actual stream geometry
                # to avoid a false software-encoder fallback.
                "-f", "lavfi", "-i",
                f"color=size={self.width}x{self.height}:rate={self.fps}",
                "-frames:v", "1", "-c:v", encoder, *options,
                "-f", "null", "-",
            ]
            try:
                result = subprocess.run(
                    probe, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, timeout=15,
                )
                if result.returncode == 0:
                    print(f"[{self.label}] encoder selected: {encoder}", flush=True)
                    return encoder
                reason = result.stderr.decode(
                    "utf-8", errors="replace"
                ).strip().splitlines()
                detail = reason[0] if reason else f"exit {result.returncode}"
                print(f"[{self.label}] {encoder} unavailable: {detail}", flush=True)
            except Exception as exc:
                print(f"[{self.label}] {encoder} probe failed: {exc}", flush=True)
        print(f"[{self.label}] hardware encoding unavailable; using libx264", flush=True)
        return "libx264"

    def _start_ffmpeg(self) -> subprocess.Popen:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps), "-i", "pipe:0", "-an",
        ]
        if self.encoder == "h264_nvenc":
            command.extend([
                "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull",
                "-rc", "cbr", "-b:v", self.bitrate,
                "-maxrate", self.bitrate, "-bufsize", self.bitrate,
            ])
        elif self.encoder == "h264_amf":
            command.extend([
                "-c:v", "h264_amf", "-usage", "lowlatency",
                "-quality", "quality", "-rc", "cbr",
                "-b:v", self.bitrate, "-maxrate", self.bitrate,
                "-bufsize", self.bitrate,
            ])
        else:
            command.extend([
                "-c:v", "libx264", "-preset", "ultrafast",
                "-tune", "zerolatency", "-b:v", self.bitrate,
                "-maxrate", self.bitrate, "-bufsize", self.bitrate,
                "-x264-params", "scenecut=0:force-cfr=1",
            ])
        command.extend([
            "-g", str(max(1, self.fps // 2)), "-bf", "0", "-pix_fmt", "yuv420p",
            "-f", "mpegts", self.url,
        ])
        print(f"[{self.label}] opening {self.url}", flush=True)
        proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=_drain_stderr, args=(proc, f"{self.label}-ffmpeg"),
                         daemon=True).start()
        return proc

    def run(self) -> None:
        latest = None
        latest_at = 0.0
        period = 1.0 / max(1, self.fps)
        deadline = time.perf_counter()
        while not self.stop_event.is_set():
            fresh_frame = False
            try:
                while True:
                    latest = self.frames.get(
                        timeout=0.01 if latest is None else 0.0
                    )
                    latest_at = time.monotonic()
                    self.source_frames += 1
                    fresh_frame = True
            except queue.Empty:
                pass
            if latest is None:
                if (
                    self.process is not None
                    and latest_at
                    and time.monotonic() - latest_at > self.stale_seconds
                ):
                    self._terminate()
                self.stop_event.wait(0.02)
                continue
            if time.monotonic() - latest_at > self.stale_seconds:
                # Never turn a dead input into an apparently-live frozen
                # return.  Closing the transport lets clients fail over to
                # their local raw stream while preserving camera identity.
                latest = None
                self._terminate()
                continue
            if not isinstance(latest, np.ndarray):
                print(f"[{self.label}] dropped invalid frame payload", flush=True)
                latest = None
                continue
            if self.process is None or self.process.poll() is not None:
                self._terminate()
                try:
                    self.process = self._start_ffmpeg()
                    self.connections += 1
                    self._last_frame_monotonic = 0.0
                except Exception as exc:
                    print(f"[{self.label}] {exc}", flush=True)
                    self.stop_event.wait(1.0)
                    continue
            now = time.perf_counter()
            # Re-anchor after startup/reconnect.  Otherwise an old deadline
            # permits a short burst before the nominal frame clock catches up.
            if deadline < now:
                deadline = now
            delay = deadline - now
            if delay > 0:
                self.stop_event.wait(delay)
            step = period
            if self.jitter:
                # Optional transport-stress test for timing-sensitive clients.
                step = period * (1.0 + float(self._rng.uniform(-0.02, 0.02)))
            try:
                # ``close`` can run concurrently during a route switch. Use a
                # stable local reference so the expected closed-pipe path is
                # handled without an AttributeError.
                proc = self.process
                if proc is None or proc.stdin is None:
                    raise BrokenPipeError("FFmpeg input pipe closed")
                payload = self._dither(latest) if self.dither else latest
                if not payload.flags.c_contiguous:
                    payload = np.ascontiguousarray(payload)
                # FileIO accepts the buffer protocol; memoryview avoids a
                # full-frame ``tobytes()`` allocation on every output tick.
                remaining = memoryview(payload).cast("B")
                while remaining:
                    written = proc.stdin.write(remaining)
                    if not written:
                        raise BrokenPipeError("FFmpeg input pipe closed")
                    remaining = remaining[written:]
                self.sent += 1
                if not fresh_frame:
                    self.repeated_frames += 1
                self.last_frame_at = time.time()
                _record_cadence(self, time.monotonic(), self.fps)
                # A bounded hold keeps MPEG-TS timestamps and wall-clock
                # delivery genuinely CFR through short inference variance.
                # The stale check above still closes the transport after one
                # second so clients can fail over instead of seeing a dead
                # camera disguised as a live frozen feed.
                deadline += step
            except (BrokenPipeError, OSError) as exc:
                print(f"[{self.label}] receiver unavailable: {exc}", flush=True)
                self.disconnects += 1
                self._terminate()

    def _terminate(self) -> None:
        proc, self.process = self.process, None
        if proc is not None:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def close(self) -> None:
        self._terminate()


class UdpBroadcastOutput(SrtOutput):
    """Independent selected-stream output used by the Arch pull client.

    The historical class name is retained to avoid needless deployment/API
    churn; its URL is now an SRT listener rather than unreliable Wi-Fi
    multicast.
    """

    def __init__(self, url: str, width: int, height: int, fps: int,
                 bitrate: str, frames: LatestFrame,
                 stop: threading.Event, *, stale_seconds: float = 1.0) -> None:
        super().__init__(
            url, width, height, fps, bitrate, frames, stop,
            label="broadcast", stale_seconds=stale_seconds,
        )
