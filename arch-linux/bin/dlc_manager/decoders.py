#!/usr/bin/env python3
"""Passive local decoders and the log discipline that keeps their noise usable.

A decoder is an ``ffmpeg`` child process reading one loopback MPEG-TS relay.
Joining an H.264 stream mid-GOP legitimately produces a burst of identical
"no frame!" / "non-existing PPS" warnings for the first second or two.  Those
are aggregated instead of flooding the visible log, while distinct failures
still surface immediately.
"""

from __future__ import annotations

from collections import OrderedDict
import re
import time
from typing import Any

from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtGui import QImage

from .contracts import PREVIEW_HEIGHT, PREVIEW_WIDTH


_DIGITS = re.compile(r"\d+")

# Warnings a decoder always emits while it acquires the first key frame.  They
# are expected, not actionable, and are only shown as one aggregated line.
_STARTUP_NOISE = (
    "non-existing pps",
    "no frame!",
    "decode_slice_header error",
    "sps unavailable",
    "error while decoding mb",
    "corrupt decoded frame",
    "co located pocs unavailable",
)


def _signature(line: str) -> str:
    """Collapse counters and timestamps so repeats compare equal."""
    without_prefix = line.split(": ", 1)[-1] if ": " in line else line
    return _DIGITS.sub("#", without_prefix).strip().lower()


class LogRateLimiter:
    """Show a message once, then summarize repeats on a fixed cadence."""

    def __init__(
        self,
        *,
        window: float = 12.0,
        startup_grace: float = 4.0,
        capacity: int = 64,
    ) -> None:
        self.window = float(window)
        self.startup_grace = float(startup_grace)
        self.capacity = int(capacity)
        self._seen: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._started_at: float | None = None

    def restart(self, now: float) -> None:
        """Mark a fresh decoder start so its acquisition burst is expected."""
        self._started_at = float(now)

    def observe(self, line: str, now: float) -> str | None:
        """Return the line to display, or ``None`` when it is a repeat."""
        text = line.strip()
        if not text:
            return None
        key = _signature(text)
        record = self._seen.get(key)
        startup = self._started_at is not None and (
            now - self._started_at <= self.startup_grace
        )
        noisy = any(marker in key for marker in _STARTUP_NOISE)
        if record is not None and now - float(record["last"]) <= self.window:
            record["last"] = now
            record["suppressed"] = int(record["suppressed"]) + 1
            record["text"] = text
            self._seen.move_to_end(key)
            return None
        self._seen[key] = {
            "last": now,
            "first": now,
            "suppressed": 0,
            "text": text,
            "announced": not (startup and noisy),
        }
        self._seen.move_to_end(key)
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        if startup and noisy:
            # Defer to ``drain`` so a single aggregated line is emitted with a
            # count instead of one entry per acquisition warning.
            return None
        return text

    def drain(self, now: float) -> list[str]:
        """Emit aggregated summaries for anything suppressed since last call."""
        lines: list[str] = []
        for key, record in list(self._seen.items()):
            expired = now - float(record["last"]) > self.window
            suppressed = int(record["suppressed"])
            if not record["announced"] and (expired or suppressed):
                lines.append(
                    f"{record['text']} · {suppressed + 1} decoder startup "
                    "message(s) aggregated"
                )
                record["announced"] = True
                record["suppressed"] = 0
            elif suppressed and expired:
                lines.append(f"{record['text']} · repeated {suppressed}×")
                record["suppressed"] = 0
            if expired and not suppressed and record["announced"]:
                self._seen.pop(key, None)
        return lines


class RawVideoDecoder(QObject):
    """One passive ffmpeg reader for a loopback relay; never opens a camera."""

    frame_ready = Signal(QImage)
    log_line = Signal(str)
    lifecycle = Signal(str)

    def __init__(
        self,
        name: str,
        command: list[str],
        parent: QObject | None = None,
        *,
        width: int = PREVIEW_WIDTH,
        height: int = PREVIEW_HEIGHT,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.command = list(command)
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * 3
        self.buffer = bytearray()
        self.frames = 0
        self.dropped = 0
        self.bytes_received = 0
        self.last_frame_at: float | None = None
        self.running = False
        self.restarts = 0
        self.limiter = LogRateLimiter()
        self._stderr = bytearray()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.SeparateChannels
        )
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

    def start(self) -> None:
        self.running = True
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.buffer.clear()
            self.limiter.restart(time.monotonic())
            self.process.start(self.command[0], self.command[1:])

    def stop(self) -> None:
        self.running = False
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1000):
                self.process.kill()
                self.process.waitForFinished(1000)

    def restart(self) -> None:
        was_running = self.running
        self.stop()
        if was_running:
            self.running = True
            QTimer.singleShot(100, self.start)

    def set_command(self, command: list[str]) -> None:
        """Switch inputs without leaving frames or counters from the old route."""
        self.stop()
        self.command = list(command)
        self.buffer.clear()
        self._stderr.clear()
        self.frames = 0
        self.dropped = 0
        self.bytes_received = 0
        self.last_frame_at = None

    def display_fps(self, previous_frames: int) -> float:
        return max(0, self.frames - previous_frames)

    def age(self) -> float | None:
        if self.last_frame_at is None:
            return None
        return time.monotonic() - self.last_frame_at

    def flush_log(self) -> None:
        """Emit aggregated startup/repeat summaries on the refresh cadence."""
        for line in self.limiter.drain(time.monotonic()):
            self.log_line.emit(f"{self.name}: {line}")

    def _started(self) -> None:
        self.restarts += 1
        self.limiter.restart(time.monotonic())
        self.lifecycle.emit(
            f"{self.name} decoder started (pid {self.process.processId()})"
        )

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.lifecycle.emit(f"{self.name} decoder exited ({exit_code})")
        if self.running:
            QTimer.singleShot(1000, self.start)

    def _error(self, _error: QProcess.ProcessError) -> None:
        self.log_line.emit(f"{self.name}: {self.process.errorString()}")

    def _read_stdout(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput())
        if not chunk:
            return
        self.bytes_received += len(chunk)
        self.buffer.extend(chunk)
        complete = len(self.buffer) // self.frame_size
        if complete <= 0:
            return
        newest = (complete - 1) * self.frame_size
        frame = bytes(self.buffer[newest : newest + self.frame_size])
        del self.buffer[: complete * self.frame_size]
        self.dropped += complete - 1
        self.frames += complete
        self.last_frame_at = time.monotonic()
        image = QImage(
            frame,
            self.width,
            self.height,
            self.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        self.frame_ready.emit(image)

    def _read_stderr(self) -> None:
        self._stderr.extend(bytes(self.process.readAllStandardError()))
        while b"\n" in self._stderr:
            raw, _, remainder = self._stderr.partition(b"\n")
            self._stderr = bytearray(remainder)
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            shown = self.limiter.observe(line, time.monotonic())
            if shown is not None:
                self.log_line.emit(f"{self.name}: {shown}")
