#!/usr/bin/env python3
"""Render workspace: offline high-quality face swap rendering & replay.

Record from the raw UDP source (front/back/arch) → pre-render with no time
constraints → loop-replay as a receiver input source. The rendered video
uses the same swap_face() path as live but with unlimited time per frame,
allowing higher quality settings and post-processing passes.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QProcess, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..contracts import (
    ANDROID_NATIVE_PREVIEW_PORT,
    LOCAL_PREVIEW_PORT,
    MANAGER_RAW_PREVIEW_PORT,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    local_mpegts_preview_command,
)
from ..decoders import RawVideoDecoder
from ..theme import readable_size
from ..widgets import StatusPill, scrollable

if TYPE_CHECKING:
    from ..viewmodel import ManagerView


RENDERS_DIR = Path("/var/lib/deep-live-cam/renders")

# Prefer the project virtual environment for model-backed subprocesses.
_VENV_PYTHON = Path(__file__).parent.parent.parent.parent.parent / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else "python3"

# Source definitions: user-visible label → UDP preview port.
# Using the receiver's UDP relays gives us the *actual* source pixels before
# they are transformed into /dev/deep-live-cam.
SOURCE_PORTS: dict[str, int] = {
    "Phone Front": ANDROID_NATIVE_PREVIEW_PORT,
    "Phone Back": ANDROID_NATIVE_PREVIEW_PORT,
    "Arch Webcam": MANAGER_RAW_PREVIEW_PORT,
}

# Recording ports.  For Arch Webcam we use the sender's raw fallback (11000)
# instead of the manager raw preview (11001) so the preview decoder and the
# recording ffmpeg are not competing for the same UDP datagrams.
RECORD_PORTS: dict[str, int] = {
    "Phone Front": ANDROID_NATIVE_PREVIEW_PORT,
    "Phone Back": ANDROID_NATIVE_PREVIEW_PORT,
    "Arch Webcam": LOCAL_PREVIEW_PORT,
}


class RenderPage(QWidget):
    """Offline rendering workspace: record → render → publish as input."""

    prerecordedSourceRequested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.render_process: QProcess | None = None
        self.record_process: QProcess | None = None
        self.preview_decoder: RawVideoDecoder | None = None
        self.active_face_path: str | None = None
        self._selected_video_path: str | None = None

        self._build_ui()
        self._refresh_file_list()
        self._start_preview("Arch Webcam")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- 1. Record Source ---
        record_group = QGroupBox("1. Record Source")
        record_layout = QVBoxLayout(record_group)

        preview_help = QLabel(
            "Preview and recording use the actual source UDP stream, not the transformed "
            "/dev/deep-live-cam output. For Arch Webcam the preview and recording read from "
            "different receiver relays so recording does not freeze the preview."
        )
        preview_help.setWordWrap(True)
        preview_help.setStyleSheet(
            "color: #9A8F7F; font-size: 11px; padding: 8px; background: #2A2520; border-radius: 4px;"
        )
        record_layout.addWidget(preview_help)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(460, 258)
        self.preview_label.setStyleSheet(
            "background: #1F2421; border: 1px solid #E7E1D7; border-radius: 4px;"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setText("Waiting for source stream...")
        self.preview_label.setScaledContents(False)
        record_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        form_layout = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_PORTS.keys()))
        self.source_combo.currentTextChanged.connect(self._source_changed)
        self.source_combo.wheelEvent = lambda event: event.ignore()
        form_layout.addRow("Source:", self.source_combo)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 600)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" sec")
        self.duration_spin.wheelEvent = lambda event: event.ignore()
        form_layout.addRow("Duration:", self.duration_spin)

        face_info = QLabel(
            "Uses the face selected on the Identity tab."
        )
        face_info.setStyleSheet("color: #9A8F7F; font-size: 11px;")
        form_layout.addRow("Face:", face_info)
        record_layout.addLayout(form_layout)

        record_btn_layout = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self._record_clicked)
        record_btn_layout.addWidget(self.record_btn)

        self.stop_record_btn = QPushButton("Stop")
        self.stop_record_btn.clicked.connect(self._stop_record)
        self.stop_record_btn.setEnabled(False)
        record_btn_layout.addWidget(self.stop_record_btn)
        record_layout.addLayout(record_btn_layout)

        self.record_status = StatusPill("Idle", "gray")
        record_layout.addWidget(self.record_status, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(record_group)

        # --- 2. Files ---
        files_group = QGroupBox("2. Recorded / Rendered Files")
        files_layout = QVBoxLayout(files_group)

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self._file_selected)
        files_layout.addWidget(self.file_list)

        action_layout = QVBoxLayout()
        action_layout.setSpacing(6)

        self.set_input_btn = QPushButton("Use as Input")
        self.set_input_btn.setToolTip("Publish the selected video as the active camera input")
        self.set_input_btn.clicked.connect(self._set_as_input_clicked)
        self.set_input_btn.setEnabled(False)
        action_layout.addWidget(self.set_input_btn)

        self.preview_file_btn = QPushButton("Preview")
        self.preview_file_btn.setToolTip("Open selected video in the default media player")
        self.preview_file_btn.clicked.connect(self._preview_file_clicked)
        self.preview_file_btn.setEnabled(False)
        action_layout.addWidget(self.preview_file_btn)

        self.rename_file_btn = QPushButton("Rename")
        self.rename_file_btn.setToolTip("Rename the selected file")
        self.rename_file_btn.clicked.connect(self._rename_file_clicked)
        self.rename_file_btn.setEnabled(False)
        action_layout.addWidget(self.rename_file_btn)

        self.delete_file_btn = QPushButton("Delete")
        self.delete_file_btn.setToolTip("Delete the selected file")
        self.delete_file_btn.clicked.connect(self._delete_file_clicked)
        self.delete_file_btn.setEnabled(False)
        action_layout.addWidget(self.delete_file_btn)

        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.setToolTip("Refresh the file list")
        self.refresh_btn.clicked.connect(self._refresh_file_list)
        action_layout.addWidget(self.refresh_btn)

        files_layout.addLayout(action_layout)
        layout.addWidget(files_group)

        # --- 3. Render with Face Swap ---
        render_group = QGroupBox("3. Render with Face Swap")
        render_layout = QVBoxLayout(render_group)

        quality_form = QFormLayout()
        self.quality_combo = QComboBox()
        self.quality_combo.addItems([
            "Fast (live settings)", "Balanced", "High Quality", "Maximum",
            "Auto (tuned per clip)",
        ])
        self.quality_combo.setCurrentIndex(2)
        self.quality_combo.wheelEvent = lambda event: event.ignore()
        quality_form.addRow("Quality:", self.quality_combo)
        # Optional GFPGAN face-restoration final pass.  Off by default: it can
        # sharpen eyes/teeth but over-smooths skin and can read as obviously AI,
        # so it is opt-in and only active when the model is present in models/.
        self.enhance_combo = QComboBox()
        self.enhance_combo.addItems([
            "Off", "Light (0.35)", "Medium (0.5)", "Strong (0.7)",
        ])
        self.enhance_combo.setCurrentIndex(0)
        self.enhance_combo.wheelEvent = lambda event: event.ignore()
        self.enhance_combo.setToolTip(
            "Optional AI face restoration final pass (needs the GFPGAN model in "
            "models/). A partial blend keeps real skin texture."
        )
        quality_form.addRow("Face restore:", self.enhance_combo)
        render_layout.addLayout(quality_form)

        self.render_preview_label = QLabel()
        self.render_preview_label.setFixedSize(320, 180)
        self.render_preview_label.setStyleSheet(
            "background: #1F2421; border: 1px solid #E7E1D7; border-radius: 4px;"
        )
        self.render_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_preview_label.setText("Last rendered frame")
        self.render_preview_label.setVisible(False)
        render_layout.addWidget(self.render_preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setVisible(False)
        render_layout.addWidget(self.render_progress)

        self.render_stats = QLabel()
        self.render_stats.setVisible(False)
        self.render_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        render_layout.addWidget(self.render_stats)

        # Commercial-quality verdict shown when a render finishes: realism
        # (seamless blend) + identity transfer, graded against commercial
        # face-swap thresholds.
        self.render_grade = StatusPill("", "unknown")
        self.render_grade.setVisible(False)
        render_layout.addWidget(
            self.render_grade, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.render_log = QTextEdit()
        self.render_log.setReadOnly(True)
        self.render_log.setMaximumHeight(120)
        self.render_log.setVisible(False)
        render_layout.addWidget(self.render_log)

        render_btn_layout = QHBoxLayout()
        self.render_btn = QPushButton("Render Selected")
        self.render_btn.clicked.connect(self._render_clicked)
        self.render_btn.setEnabled(False)
        render_btn_layout.addWidget(self.render_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_render)
        self.cancel_btn.setVisible(False)
        render_btn_layout.addWidget(self.cancel_btn)
        render_layout.addLayout(render_btn_layout)

        layout.addWidget(render_group)
        layout.addStretch()
        outer.addWidget(scrollable(panel))

    def render(self, view: "ManagerView") -> None:
        """Update with current viewmodel state."""
        if view.identity and view.identity.cache_path:
            self.active_face_path = view.identity.cache_path
        else:
            self.active_face_path = None

    def _start_preview(self, source: str) -> None:
        """Start a UDP preview decoder for the selected source."""
        self._stop_preview()
        port = SOURCE_PORTS.get(source, MANAGER_RAW_PREVIEW_PORT)
        self.preview_label.setText(f"Waiting for {source} stream...")
        self.preview_decoder = RawVideoDecoder(
            f"render-preview-{source.lower().replace(' ', '-')}",
            local_mpegts_preview_command(port),
            self,
        )
        self.preview_decoder.frame_ready.connect(self._on_preview_frame)
        self.preview_decoder.lifecycle.connect(self._on_preview_lifecycle)
        self.preview_decoder.log_line.connect(self._on_preview_log)
        self.preview_decoder.start()

    def _stop_preview(self) -> None:
        """Stop the current preview decoder."""
        if self.preview_decoder is not None:
            try:
                self.preview_decoder.frame_ready.disconnect(self._on_preview_frame)
                self.preview_decoder.lifecycle.disconnect(self._on_preview_lifecycle)
                self.preview_decoder.log_line.disconnect(self._on_preview_log)
            except RuntimeError:
                pass
            self.preview_decoder.stop()
            self.preview_decoder = None

    def _source_changed(self, text: str) -> None:
        """Switch preview to the newly selected source."""
        self._start_preview(text)

    def _on_preview_frame(self, image: QImage) -> None:
        """Display a preview frame."""
        if image.isNull():
            return
        scaled = image.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(QPixmap.fromImage(scaled))

    def _on_preview_lifecycle(self, message: str) -> None:
        """Show decoder lifecycle messages in the preview area."""
        if self.preview_label.pixmap() is None:
            self.preview_label.setText(message)

    def _on_preview_log(self, message: str) -> None:
        """Ignore routine decoder log lines."""
        pass

    def _refresh_file_list(self) -> None:
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        self.file_list.clear()

        recordings = sorted(RENDERS_DIR.glob("recording_*.mp4"), reverse=True)
        renders = sorted(RENDERS_DIR.glob("render_*.mp4"), reverse=True)

        for path in recordings + renders:
            size = readable_size(path.stat().st_size) if path.exists() else "?"
            prefix = "🎬" if path.name.startswith("render_") else "📹"
            self.file_list.addItem(f"{prefix} {path.name} ({size})")

    def _file_selected(self) -> None:
        has_selection = bool(self.file_list.selectedItems())
        self.render_btn.setEnabled(has_selection and not self.render_process)
        self.set_input_btn.setEnabled(has_selection)
        self.preview_file_btn.setEnabled(has_selection)
        self.rename_file_btn.setEnabled(has_selection)
        self.delete_file_btn.setEnabled(has_selection)
        if has_selection:
            path = self._selected_path()
            self._selected_video_path = str(path) if path else None
        else:
            self._selected_video_path = None

    def _record_clicked(self) -> None:
        try:
            self._start_recording()
        except Exception as exc:
            self.record_status.set_state(f"Record error: {exc}", "red")
            self.record_btn.setEnabled(True)
            self.stop_record_btn.setEnabled(False)
            self.record_process = None

    def _start_recording(self) -> None:
        source = self.source_combo.currentText()
        duration = self.duration_spin.value()
        port = RECORD_PORTS[source]

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_slug = source.lower().replace(" ", "_")
        output = RENDERS_DIR / f"recording_{source_slug}_{timestamp}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        self.record_process = QProcess(self)
        self.record_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.record_process.readyReadStandardOutput.connect(self._record_output)
        self.record_process.finished.connect(self._record_finished)
        self.record_process.errorOccurred.connect(self._record_error)

        # Re-encode the UDP MPEG-TS stream.  A straight -c:v copy fails because
        # the receiver's H.264 stream can start mid-GOP, so ffmpeg cannot derive
        # dimensions from the first packets it sees.  Error- resilience flags
        # keep recording alive across transient UDP corruption.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-analyzeduration", "2000000",
            "-probesize", "2000000",
            "-err_detect", "ignore_err",
            "-f", "mpegts",
            "-i", f"udp://127.0.0.1:{port}?reuse=1&fifo_size=1000000&overrun_nonfatal=1",
            "-t", str(duration),
            "-fps_mode", "cfr",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y",
            str(output),
        ]

        self.record_log_buffer: list[str] = []
        self.record_status.set_state(f"Recording {source} ({duration}s)", "green")
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.record_process.start(cmd[0], cmd[1:])

    def _record_output(self) -> None:
        if not self.record_process:
            return
        data = bytes(self.record_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.strip().split("\n"):
            if line:
                self.record_log_buffer.append(line)

    def _record_error(self, error: QProcess.ProcessError) -> None:
        self.record_status.set_state(f"Record error: {error.name}", "red")
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.record_process = None

    def _stop_record(self) -> None:
        """Stop recording gracefully by sending SIGINT (like Ctrl+C to ffmpeg)."""
        if self.record_process and self.record_process.state() == QProcess.ProcessState.Running:
            import signal as _sig
            pid = self.record_process.processId()
            if pid:
                import os as _os
                _os.kill(pid, _sig.SIGINT)
            else:
                self.record_process.terminate()
            self.record_status.set_state("Finishing...", "gray")
            self.stop_record_btn.setEnabled(False)

    def _record_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Handle recording completion."""
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)

        # ffmpeg exits 255 on SIGINT, treat as success if file was created.
        if exit_code == 0 or exit_code == 255:
            self.record_status.set_state("Complete", "green")
            self._refresh_file_list()
        else:
            msg = f"Failed (exit {exit_code})"
            if self.record_log_buffer:
                msg += f": {self.record_log_buffer[-1][:80]}"
            self.record_status.set_state(msg, "red")
            if self.record_log_buffer:
                self.render_log.setVisible(True)
                self.render_log.append("[record] " + "\n[record] ".join(self.record_log_buffer[-10:]))

        self.record_process = None
        self.record_log_buffer = []
        QTimer.singleShot(3000, lambda: self.record_status.set_state("Idle", "gray"))

    def _render_clicked(self) -> None:
        selected = self.file_list.selectedItems()
        if not selected:
            return

        filename = selected[0].text().split(" ", 1)[1].split(" (")[0]
        input_path = RENDERS_DIR / filename

        if not input_path.exists():
            self.render_log.setVisible(True)
            self.render_log.append(f"ERROR: file not found: {filename}")
            return

        if not self.active_face_path or not Path(self.active_face_path).exists():
            self.render_log.setVisible(True)
            self.render_log.append("ERROR: No face image selected. Go to Identity tab to choose a face.")
            return
        face_path = self.active_face_path

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = RENDERS_DIR / f"render_{timestamp}.mp4"

        quality_map = {
            "Fast (live settings)": "fast",
            "Balanced": "balanced",
            "High Quality": "high",
            "Maximum": "max",
            "Auto (tuned per clip)": "auto",
        }
        quality = quality_map[self.quality_combo.currentText()]
        enhance_map = {
            "Off": 0.0,
            "Light (0.35)": 0.35,
            "Medium (0.5)": 0.5,
            "Strong (0.7)": 0.7,
        }
        enhance_strength = enhance_map.get(self.enhance_combo.currentText(), 0.0)

        self.render_log.setVisible(True)
        self.render_log.clear()
        self.render_grade.setVisible(False)
        self.render_progress.setVisible(True)
        self.render_stats.setVisible(True)
        self.render_progress.setRange(0, 100)
        self.render_progress.setValue(0)
        self.render_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)

        self.render_process = QProcess(self)
        self.render_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.render_process.readyReadStandardOutput.connect(self._render_output)
        self.render_process.finished.connect(self._render_finished)
        self.render_process.errorOccurred.connect(self._render_error)

        cmd = [
            PYTHON,
            "/opt/github/Deep-Live-Cam/arch-linux/bin/offline_renderer.py",
            str(input_path),
            str(output),
            "--face", face_path,
            "--quality", quality,
        ]
        if enhance_strength > 0.0:
            cmd += ["--enhance", str(enhance_strength)]

        self.render_log.append(f"Starting render: {input_path.name} → {output.name}")
        self.render_log.append(
            f"Quality: {quality}"
            + (f", face-restore: {enhance_strength}" if enhance_strength else "")
            + f", Face: {Path(face_path).name}"
        )
        self.render_process.start(cmd[0], cmd[1:])

    def _render_error(self, error: QProcess.ProcessError) -> None:
        self.render_log.append(f"ERROR: render process failed to start ({error.name})")
        self.render_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.render_process = None

    def _render_output(self) -> None:
        if not self.render_process:
            return

        data = bytes(self.render_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.strip().split("\n"):
            if not line:
                continue
            if line.startswith("{"):
                try:
                    progress = json.loads(line)
                    if "frame" in progress and "total" in progress:
                        self.render_progress.setMaximum(progress["total"])
                        self.render_progress.setValue(progress["frame"])
                        fps = progress.get("fps", 0.0)
                        self.render_progress.setFormat(
                            f"{progress['percent']:.1f}% - Frame {progress['frame']}/{progress['total']} - {fps:.1f} fps"
                        )
                        elapsed = progress["frame"] / fps if fps > 0 else 0
                        remaining = (progress["total"] - progress["frame"]) / fps if fps > 0 else 0
                        self.render_stats.setText(
                            f"Elapsed: {elapsed:.1f}s | Remaining: ~{remaining:.1f}s"
                        )
                        # Display thumbnail if present
                        thumb = progress.get("thumb")
                        if thumb:
                            import base64
                            img_data = base64.b64decode(thumb)
                            img = QImage()
                            img.loadFromData(img_data)
                            if not img.isNull():
                                self.render_preview_label.setVisible(True)
                                self.render_preview_label.setPixmap(
                                    QPixmap.fromImage(img)
                                )
                    continue
                except (json.JSONDecodeError, Exception):
                    pass
            # Commercial-quality grade line emitted by the renderer.
            if line.startswith("[grade] {"):
                try:
                    self._show_render_grade(json.loads(line[len("[grade] "):]))
                    continue
                except (json.JSONDecodeError, Exception):
                    pass
            self.render_log.append(line)

    def _show_render_grade(self, summary: dict) -> None:
        """Surface the commercial-quality verdict on the render page."""
        grade = str(summary.get("grade", "?"))
        realism = summary.get("realism")
        identity = summary.get("identity")
        composite = summary.get("composite")
        ident_txt = "n/a" if identity is None else f"{identity:.0f}"
        # Colour the pill by band so an A/A+ reads as commercial-grade at a
        # glance and a C/D/F flags a visible seam or weak identity transfer.
        state = (
            "running" if grade in ("A+", "A")
            else "working" if grade in ("B", "C")
            else "failed"
        )
        self.render_grade.set_state(
            f"Quality {grade}  ·  realism {realism:.0f}  ·  identity {ident_txt}"
            f"  ·  composite {composite:.0f}/100",
            state,
        )
        self.render_grade.setVisible(True)
        self.render_log.append(
            f"Commercial-quality grade: {grade} "
            f"(realism {realism:.0f}, identity {ident_txt}, composite {composite:.0f})"
        )

    def _render_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.render_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.render_preview_label.setVisible(False)

        if exit_code == 0:
            self.render_log.append("✓ Render complete")
            self._refresh_file_list()
        elif exit_code == 130:
            self.render_log.append("✗ Render cancelled")
        else:
            self.render_log.append(f"✗ Render failed (exit {exit_code})")

        self.render_process = None

    def _cancel_render(self) -> None:
        if self.render_process and self.render_process.state() == QProcess.ProcessState.Running:
            self.render_process.terminate()
            QTimer.singleShot(2000, lambda: (
                self.render_process.kill() if self.render_process and
                self.render_process.state() == QProcess.ProcessState.Running else None
            ))
            self.render_log.append("Cancelling render...")

    def _selected_path(self) -> Path | None:
        """Return the filesystem path of the currently selected file, or None."""
        selected = self.file_list.selectedItems()
        if not selected:
            return None
        filename = selected[0].text().split(" ", 1)[1].split(" (")[0]
        path = RENDERS_DIR / filename
        return path if path.exists() else None

    def _preview_file_clicked(self) -> None:
        """Open the selected file in the default media player."""
        path = self._selected_path()
        if path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _rename_file_clicked(self) -> None:
        """Rename the selected file."""
        path = self._selected_path()
        if path is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename File", "New name:", text=path.name
        )
        if not ok or not new_name:
            return
        new_name = new_name.strip()
        if new_name == path.name:
            return
        # Strip any emoji prefix the user may have pasted.
        new_name = re.sub(r"^[🎬📹]\\s*", "", new_name)
        if not new_name.endswith(".mp4"):
            new_name += ".mp4"
        new_path = path.with_name(new_name)
        if new_path.exists():
            QMessageBox.warning(self, "Rename Failed", f"{new_path.name} already exists.")
            return
        try:
            path.rename(new_path)
            self._refresh_file_list()
        except OSError as exc:
            QMessageBox.warning(self, "Rename Failed", str(exc))

    def _delete_file_clicked(self) -> None:
        """Delete the selected file after confirmation."""
        path = self._selected_path()
        if path is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete File",
            f"Delete {path.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
            self._refresh_file_list()
        except OSError as exc:
            QMessageBox.warning(self, "Delete Failed", str(exc))

    def _set_as_input_clicked(self) -> None:
        path = self._selected_path()
        if path is None:
            return
        self.prerecordedSourceRequested.emit(str(path))

    def closeEvent(self, event) -> None:
        """Clean up decoders when the page is destroyed."""
        self._stop_preview()
        event.accept()
