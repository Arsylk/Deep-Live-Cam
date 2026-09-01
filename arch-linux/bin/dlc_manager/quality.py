#!/usr/bin/env python3
"""Passive measurement of decoded previews and of this host.

Nothing here writes to the pipeline.  The analyzer only looks at frames the
manager has already decoded for display, so enabling or disabling it cannot
change capture, processing, or camera ownership.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import time
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

try:
    import numpy as np
except ImportError:  # Measurements degrade to "unavailable", never to a crash.
    np = None

try:
    import cv2
except ImportError:  # Optional: metrics use NumPy with a center-ROI fallback.
    cv2 = None


def face_detector() -> Any:
    """Return an OpenCV cascade when one is installed, otherwise ``None``."""
    if cv2 is None:
        return None
    
    # Check if CascadeClassifier is available (headless cv2 builds may omit it)
    if not hasattr(cv2, 'CascadeClassifier'):
        return None
    
    try:
        candidate = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return None if candidate.empty() else candidate
    except (AttributeError, Exception):
        return None


class StreamQualityAnalyzer:
    """Passive frame-quality measurements; never modifies pipeline frames."""

    def __init__(self, detector: Any = None) -> None:
        self.detector = detector
        self.last_heavy_sample = 0.0
        self.previous_signature: Any = None
        self.repeat_history: deque[bool] = deque(maxlen=300)
        self.motion_history: deque[float] = deque(maxlen=300)
        self.face_history: deque[bool] = deque(maxlen=120)
        self.latest: dict[str, Any] = {
            "available": np is not None,
            "face_detector_available": self.detector is not None,
            "samples": 0,
            "face_detected": False,
        }

    def observe(self, image: QImage) -> None:
        if np is None:
            return
        try:
            rgb = self._array(image)
            if cv2 is not None:
                tiny = cv2.resize(rgb, (48, 27), interpolation=cv2.INTER_AREA)
            else:
                tiny_image = image.scaled(
                    48,
                    27,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                tiny = self._array(tiny_image)
            signature = self._gray(tiny)
            if self.previous_signature is not None:
                delta = float(
                    np.mean(
                        np.abs(
                            signature.astype(np.int16)
                            - self.previous_signature.astype(np.int16)
                        )
                    )
                )
                self.motion_history.append(delta)
                self.repeat_history.append(
                    bool(np.array_equal(signature, self.previous_signature))
                )
            self.previous_signature = signature.copy()

            now = time.monotonic()
            if now - self.last_heavy_sample < 0.25:
                self._update_temporal()
                return
            self.last_heavy_sample = now
            gray = self._gray(rgb)
            face = self._largest_face(gray)
            face_detected = face is not None
            self.face_history.append(face_detected)
            if face is None:
                height, width = gray.shape
                x, y, w, h = width // 4, height // 6, width // 2, height * 2 // 3
            else:
                x, y, w, h = face
            roi = gray[y : y + h, x : x + w]
            if roi.size == 0:
                return

            if cv2 is not None:
                laplacian_variance = float(cv2.Laplacian(roi, cv2.CV_64F).var())
                sobel_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                edge_energy = float(
                    np.mean(np.abs(sobel_x)) + np.mean(np.abs(sobel_y))
                )
            else:
                values = roi.astype(np.float32)
                laplacian = (
                    -4 * values[1:-1, 1:-1]
                    + values[:-2, 1:-1]
                    + values[2:, 1:-1]
                    + values[1:-1, :-2]
                    + values[1:-1, 2:]
                )
                laplacian_variance = float(laplacian.var())
                edge_energy = float(
                    np.mean(np.abs(values[:, 1:] - values[:, :-1]))
                    + np.mean(np.abs(values[1:, :] - values[:-1, :]))
                )
            blockiness = self._blockiness(gray)

            self.latest.update(
                {
                    "samples": int(self.latest.get("samples", 0)) + 1,
                    "face_detected": face_detected,
                    "roi_kind": "face" if face_detected else "center",
                    "face_width": int(w) if face_detected else None,
                    "face_height": int(h) if face_detected else None,
                    "face_pixels": int(w * h) if face_detected else None,
                    "roi_width": int(w),
                    "roi_height": int(h),
                    "detail_laplacian": self._ema(
                        "detail_laplacian", laplacian_variance
                    ),
                    "edge_energy": self._ema("edge_energy", edge_energy),
                    "blockiness_ratio": self._ema("blockiness_ratio", blockiness),
                }
            )
            self._update_temporal()
        except (ValueError, TypeError, RuntimeError):
            return

    @staticmethod
    def _gray(rgb: Any) -> Any:
        return np.clip(
            rgb[:, :, 0].astype(np.float32) * 0.299
            + rgb[:, :, 1].astype(np.float32) * 0.587
            + rgb[:, :, 2].astype(np.float32) * 0.114,
            0,
            255,
        ).astype(np.uint8)

    def _array(self, image: QImage) -> Any:
        converted = (
            image
            if image.format() == QImage.Format.Format_RGB888
            else image.convertToFormat(QImage.Format.Format_RGB888)
        )
        view = np.frombuffer(converted.bits(), dtype=np.uint8)
        return view.reshape((converted.height(), converted.bytesPerLine()))[
            :, : converted.width() * 3
        ].reshape((converted.height(), converted.width(), 3))

    def _largest_face(self, gray: Any) -> tuple[int, int, int, int] | None:
        if self.detector is None:
            return None
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(48, 48),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _blockiness(gray: Any) -> float:
        pixels = gray.astype(np.float32)
        vertical_all = float(np.mean(np.abs(pixels[:, 1:] - pixels[:, :-1])))
        horizontal_all = float(np.mean(np.abs(pixels[1:, :] - pixels[:-1, :])))
        vertical_blocks = float(np.mean(np.abs(pixels[:, 8::8] - pixels[:, 7:-1:8])))
        horizontal_blocks = float(
            np.mean(np.abs(pixels[8::8, :] - pixels[7:-1:8, :]))
        )
        return (vertical_blocks + horizontal_blocks) / max(
            0.001, vertical_all + horizontal_all
        )

    def _ema(self, key: str, value: float) -> float:
        previous = self.latest.get(key)
        return value if previous is None else float(previous) * 0.82 + value * 0.18

    def _update_temporal(self) -> None:
        self.latest["exact_repeat_percent"] = (
            100.0 * sum(self.repeat_history) / len(self.repeat_history)
            if self.repeat_history
            else 0.0
        )
        self.latest["mean_frame_delta"] = (
            sum(self.motion_history) / len(self.motion_history)
            if self.motion_history
            else 0.0
        )
        self.latest["face_detection_percent"] = (
            100.0 * sum(self.face_history) / len(self.face_history)
            if self.face_history
            else 0.0
        )

    def reset(self) -> None:
        self.previous_signature = None
        self.repeat_history.clear()
        self.motion_history.clear()
        self.face_history.clear()
        available = bool(self.latest.get("available"))
        detector_available = bool(self.latest.get("face_detector_available"))
        self.latest = {
            "available": available,
            "face_detector_available": detector_available,
            "samples": 0,
            "face_detected": False,
        }


# Plain-language descriptions, so the Analyze page never shows a bare
# "detail_laplacian" number without saying what it means or what it does not.
METRIC_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "detail_laplacian",
        "Fine detail",
        "{value:.1f}",
        "Local contrast energy inside the measured region. Higher is sharper; "
        "it is not a realism score.",
    ),
    (
        "edge_energy",
        "Edge strength",
        "{value:.1f}",
        "Average gradient magnitude. Falls when a stream is softened or "
        "upscaled.",
    ),
    (
        "blockiness_ratio",
        "Compression blocking",
        "{value:.2f}",
        "Ratio of energy on 8-pixel boundaries. Rises when the encoder is "
        "starved of bitrate.",
    ),
    (
        "exact_repeat_percent",
        "Exactly repeated frames",
        "{value:.1f} %",
        "Share of decoded frames identical to the previous one. High values "
        "mean the source is holding frames.",
    ),
    (
        "mean_frame_delta",
        "Motion between frames",
        "{value:.2f}",
        "Average change between consecutive frames. Near zero means a frozen "
        "picture, not a still subject.",
    ),
)


def readiness(values: dict[str, Any]) -> tuple[str, str]:
    """Return a (state, text) pair describing how usable a sample window is."""
    if not values.get("available"):
        return "unavailable", "NUMPY UNAVAILABLE"
    samples = int(values.get("samples") or 0)
    if samples <= 0:
        return "waiting", "COLLECTING FIRST SAMPLES"
    if samples < 20:
        return "working", f"WARMING UP · {samples} SAMPLES"
    return "running", f"READY · {samples} SAMPLES"


def region_text(values: dict[str, Any]) -> str:
    if values.get("face_width") is not None:
        return (
            f"face {values.get('face_width')}×{values.get('face_height')} px"
        )
    reason = (
        "no face detected"
        if values.get("face_detector_available")
        else "face detector unavailable"
    )
    return (
        f"center region {values.get('roi_width', '?')}×"
        f"{values.get('roi_height', '?')} px ({reason})"
    )


class HostMetrics:
    """Coarse local telemetry sampled from procfs; opens no device."""

    def __init__(self) -> None:
        self.previous_cpu: tuple[int, int] | None = None
        self.previous_network: tuple[float, int, int] | None = None

    def sample(self, pids: set[int]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cpu_percent": 0.0,
            "rss_mb": 0.0,
            "tx_mbps": 0.0,
            "rx_mbps": 0.0,
        }
        try:
            values = [
                int(value)
                for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            ]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            if self.previous_cpu is not None:
                delta_total = total - self.previous_cpu[0]
                delta_idle = idle - self.previous_cpu[1]
                if delta_total > 0:
                    result["cpu_percent"] = (
                        100.0 * (delta_total - delta_idle) / delta_total
                    )
            self.previous_cpu = (total, idle)
        except (OSError, ValueError, IndexError):
            pass

        page_size = os.sysconf("SC_PAGE_SIZE")
        rss_pages = 0
        for pid in pids:
            try:
                rss_pages += int(Path(f"/proc/{pid}/statm").read_text().split()[1])
            except (OSError, ValueError, IndexError):
                continue
        result["rss_mb"] = rss_pages * page_size / (1024 * 1024)

        now = time.monotonic()
        try:
            tx = int(Path("/sys/class/net/wlan0/statistics/tx_bytes").read_text())
            rx = int(Path("/sys/class/net/wlan0/statistics/rx_bytes").read_text())
            if self.previous_network is not None:
                elapsed = now - self.previous_network[0]
                if elapsed > 0:
                    result["tx_mbps"] = max(
                        0.0,
                        (tx - self.previous_network[1]) * 8 / elapsed / 1_000_000,
                    )
                    result["rx_mbps"] = max(
                        0.0,
                        (rx - self.previous_network[2]) * 8 / elapsed / 1_000_000,
                    )
            self.previous_network = (now, tx, rx)
        except (OSError, ValueError):
            pass
        try:
            result["load"] = Path("/proc/loadavg").read_text().split()[0]
        except OSError:
            result["load"] = "?"
        return result
