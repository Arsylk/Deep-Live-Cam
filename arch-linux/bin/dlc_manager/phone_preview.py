#!/usr/bin/env python3
"""Pop-out window for frames carried by the exclusive phone-return relay.

It decodes one loopback MPEG-TS relay and applies the desired orientation for
display: no camera is opened, no encoder is started, and closing it stops only
its own decoder. When the return is not fresh, no substitute is shown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import QMainWindow, QWidget

from .contracts import integer, local_mpegts_preview_command, readable_age
from .decoders import RawVideoDecoder
from .health import (
    file_age,
    phone_return_relay_preview_fresh,
    phone_return_relay_title,
    read_json,
)
from .widgets import VideoPane
from .preview_transform import transform_preview_image


class PhoneReturnPreviewWindow(QMainWindow):
    """Passive view of the exact encoded stream tee sent toward Android."""

    def __init__(
        self,
        preview_port: int,
        health_file: Path,
        parent: QWidget | None = None,
        *,
        transform_supplier: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.preview_port = int(preview_port)
        self.health_file = Path(health_file)
        self.transform_supplier = transform_supplier or (lambda: {})
        self.previous_frames = 0
        self.stale_cleared = False
        self.setWindowTitle("Phone return preview — Deep-Live-Cam")
        self.resize(1060, 700)

        self.pane = VideoPane(
            "PROCESSED RESULT SENT TO PHONE",
            f"phone-relay frames, locally oriented for display · MPEG-TS/UDP "
            f"127.0.0.1:{self.preview_port} · opens no camera device",
        )
        self.setCentralWidget(self.pane)
        self.decoder = RawVideoDecoder(
            "phone-return-preview",
            local_mpegts_preview_command(self.preview_port),
            self,
        )
        self.decoder.frame_ready.connect(self._present_frame)
        self.decoder.log_line.connect(self._decoder_log)
        self.decoder.lifecycle.connect(self._decoder_log)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._refresh)

    def _present_frame(self, image: QImage) -> None:
        self.stale_cleared = False
        transform = self.transform_supplier()
        self.pane.set_image(
            transform_preview_image(
                image,
                mirror=bool(transform.get("mirror", False)),
                rotation=int(transform.get("rotation", 0)),
            )
        )

    def _decoder_log(self, line: str) -> None:
        if "error" in line.lower() or "invalid" in line.lower():
            self.pane.stats.setText(line)

    def _refresh(self) -> None:
        health = read_json(self.health_file) or {}
        fresh = phone_return_relay_preview_fresh(
            health,
            health_age=file_age(self.health_file),
            expected_port=self.preview_port,
        )
        returned = health.get("output") if isinstance(health, dict) else {}
        source = health.get("source") if isinstance(health, dict) else "unknown"
        returned = returned if isinstance(returned, dict) else {}

        self.pane.set_heading(
            phone_return_relay_title(health),
            f"phone-relay frames, locally oriented for display · MPEG-TS/UDP "
            f"127.0.0.1:{self.preview_port} · opens no camera device",
        )
        display_fps = self.decoder.display_fps(self.previous_frames)
        self.previous_frames = self.decoder.frames
        self.pane.metrics.update_values(
            (
                ("Decoded FPS", f"{display_fps:.1f}"),
                ("Frames", integer(self.decoder.frames)),
                ("Drops", integer(self.decoder.dropped)),
                ("Last frame", readable_age(self.decoder.age())),
            )
        )
        self.pane.status_pill.set_state(
            "LIVE" if fresh else "WAITING FOR THE RETURN ENCODER",
            "live" if fresh else "waiting",
        )
        self.pane.stats.setText(
            f"stream copy  ·  relayed {integer(health.get('frames'))}  ·  "
            f"source {source}  ·  Android {returned.get('host', 'unknown')}:"
            f"{returned.get('port', 'unknown')}  ·  relay restarts "
            f"{integer(health.get('restarts'))}"
        )
        if not fresh and not self.stale_cleared:
            self.pane.clear_image(
                "WAITING FOR THE EXCLUSIVE PHONE-RETURN RELAY\n"
                "No substitute stream is shown"
            )
            self.stale_cleared = True

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not self.decoder.running:
            self.decoder.start()
        self.timer.start()
        self._refresh()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        self.decoder.stop()
        event.accept()
