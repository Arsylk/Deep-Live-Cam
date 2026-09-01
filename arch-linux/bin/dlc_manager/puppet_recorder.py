#!/usr/bin/env python3
"""Guided puppet-recording session (standalone GUI).

Walks the operator through the motion vocabulary in front of the camera with
large on-screen prompts and countdowns, records ONE continuous take from the
arch webcam raw stream, and writes:

    puppet_recording_<ts>.mp4        the raw take
    puppet_recording_<ts>.cues.json  cue sheet: {action, t_start, t_end} per
                                     performed action, in seconds relative to
                                     recording start

The cue sheet is what puppet_library_build.py consumes to cut the swapped
footage into the segment library that puppet_assemble.py plays back.

Run:  python3 puppet_recorder.py   (pkexec/root not required; needs the arch
webcam stream on MANAGER_RAW_PREVIEW_PORT, default 11001)
"""
from __future__ import annotations

import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dlc_manager.contracts import (
    LOCAL_PREVIEW_PORT,
    MANAGER_RAW_PREVIEW_PORT,
)
from dlc_manager.decoders import RawVideoDecoder
from dlc_manager.theme import ACCENT, stylesheet
from dlc_manager.widgets import Card, StatusPill, page_heading

RECORD_PORT = MANAGER_RAW_PREVIEW_PORT
RENDERS_DIR = Path("/var/lib/deep-live-cam/renders")
PHONE_PREVIEW_PORT = 11_012  # phone-processed service's encoded preview tap
REC_PREVIEW_PORT = 11_013  # live preview OF the recording (encoder tee)

# Recordable sources.  Phone reads the phone processor's dedicated preview
# tap (raw passthrough while Identity processing is off).  The capture app
# renders the camera aspect-true (v1.7 fit mode): vertical content arrives
# pillarboxed inside the 1280x720 transport, so the preview crops the
# pillarbox (405x720 centred = the camera's full vertical FOV) and outputs a
# vertical 360x640 frame.  The webcam record port is the sender's raw
# fallback (11000) so preview and recording never compete for datagrams.
RECORD_SOURCES: dict[str, dict] = {
    "Arch Webcam": {
        "preview": MANAGER_RAW_PREVIEW_PORT,
        "record": LOCAL_PREVIEW_PORT,
        "out_w": 640, "out_h": 360, "crop": None,
    },
    "Phone": {
        "preview": PHONE_PREVIEW_PORT,
        "record": PHONE_PREVIEW_PORT,
        "out_w": 360, "out_h": 640,
        "crop": None,
    },
}

# --- session script ---------------------------------------------------------
# Each entry: (kind, action, seconds).  perform steps emit cues; the lengths
# are tuned so every cue starts and ends near the neutral pose.

PERFORM_SECONDS = {
    "turn_left": 1.6,
    "turn_right": 1.6,
    "look_up": 1.4,
    "look_down": 1.4,
    "blink": 1.0,
    "open_mouth": 1.2,
}

ACTION_ORDER = [
    "turn_left", "turn_right", "look_up", "look_down", "blink", "open_mouth",
    *[f"digits_{d}" for d in range(10)],
]

PREPARE_SECONDS = 1.5
REST_SECONDS = 1.6
LEAD_IN_SECONDS = 2.5
TRAIL_SECONDS = 2.0


def preview_command(spec: dict) -> list[str]:
    """Aspect-correct preview decoder for one source.

    Phone sources arrive pillarboxed inside the 1280x720 transport; the crop
    extracts the vertical content so the preview shows the phone camera
    vertical, as it is.  Output is a FIXED w x h frame (the decoder parses
    raw frames at exactly this size) with black bars only where the source
    aspect does not fill the box.
    """
    vf = []
    if spec.get("crop"):
        vf.append(f"crop={spec['crop']}")
    vf.append(
        f"scale=w={spec['out_w']}:h={spec['out_h']}"
        ":force_original_aspect_ratio=decrease:flags=fast_bilinear"
    )
    vf.append(
        f"pad={spec['out_w']}:{spec['out_h']}:(ow-iw)/2:(oh-ih)/2:black"
    )
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-fflags", "nobuffer+discardcorrupt", "-flags", "low_delay",
        "-analyzeduration", "1000000", "-probesize", "1000000",
        "-err_detect", "ignore_err",
        "-f", "mpegts",
        "-i", (f"udp://127.0.0.1:{int(spec['preview'])}"
               "?reuse=1&fifo_size=1000000&overrun_nonfatal=1"),
        "-map", "0:v:0", "-an",
        "-vf", ",".join(vf),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]


def build_script() -> list[dict]:
    """Session timeline: lead-in neutral, then prepare/perform/rest per action."""
    script: list[dict] = [
        {"kind": "lead", "action": "neutral", "seconds": LEAD_IN_SECONDS,
         "label": "Sit still, look at the camera"},
    ]
    for action in ACTION_ORDER:
        script.append({"kind": "prepare", "action": action,
                       "seconds": PREPARE_SECONDS,
                       "label": f"Next: {action}"})
        seconds = (PERFORM_SECONDS.get(action)
                   or 1.0)  # digits perform ~1s each
        script.append({"kind": "perform", "action": action,
                       "seconds": seconds,
                       "label": action})
        script.append({"kind": "rest", "action": "neutral",
                       "seconds": REST_SECONDS, "label": "Relax"})
    script.append({"kind": "trail", "action": "neutral",
                   "seconds": TRAIL_SECONDS,
                   "label": "Hold still — finishing"})
    return script


# --- GUI --------------------------------------------------------------------

class RecorderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Puppet Recorder")
        self.resize(980, 680)
        self.script = build_script()
        self.step_index = -1
        self.step_deadline = 0.0
        self.recording = False
        self.record_process = None
        self.t0 = 0.0
        self.cues: list[dict] = []
        self.output_path: Path | None = None
        self.cues_path: Path | None = None
        self.preview_decoder: RawVideoDecoder | None = None

        self._build_ui()

        # Step timer drives the guided timeline while recording.
        self.step_timer = QTimer(self)
        self.step_timer.setInterval(50)
        self.step_timer.timeout.connect(self._tick)

        # Preview runs from launch (not only while recording) so the operator
        # can frame themselves before starting a take.
        self._start_preview()

    # UI construction --------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        layout.addWidget(page_heading(
            "Puppet recorder",
            "Follow the prompts; one continuous take becomes the segment "
            "library for prompt playback",
        ))

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        # left: preview
        preview_card = Card("Camera preview")
        preview_layout = preview_card.content

        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(6)
        source_label = QLabel("Source:")
        source_label.setObjectName("hintText")
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(RECORD_SOURCES.keys()))
        self.source_combo.setToolTip(
            "Which camera feed to preview and record. Phone reads the "
            "phone-return tap (set the lens from the manager's Input tab)."
        )
        self.source_combo.wheelEvent = lambda event: event.ignore()
        self.source_combo.currentTextChanged.connect(self._switch_source)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo, 1)
        # Picker pinned to the top of the card with wrap-content height.
        preview_layout.addWidget(source_row, 0, Qt.AlignmentFlag.AlignTop)

        self.preview_label = QLabel("Waiting for stream…")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(360, 270)
        self.preview_label.setStyleSheet(
            "background: #10121a; color: #6c7086; border-radius: 8px;")
        from PySide6.QtWidgets import QSizePolicy
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview_label, 1)
        body.addWidget(preview_card, stretch=3)

        # right: guidance
        guide_card = Card("Follow the prompt")
        guide = guide_card.content
        self.action_label = QLabel("Ready")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet(
            f"font-size: 34px; font-weight: 800; color: {ACCENT};")
        guide.addWidget(self.action_label, stretch=1)

        self.next_label = QLabel(" ")
        self.next_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_label.setStyleSheet("color: #9399b2; font-size: 13px;")
        guide.addWidget(self.next_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(14)
        guide.addWidget(self.progress)

        self.step_label = QLabel("step –/–")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_label.setStyleSheet("color: #6c7086; font-size: 12px;")
        guide.addWidget(self.step_label)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start session")
        self.start_btn.clicked.connect(self._start_session)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_session)
        self.stop_btn.setEnabled(False)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        guide.addLayout(buttons)

        self.status = StatusPill("idle", "gray")
        guide.addWidget(self.status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setStyleSheet(
            "color: #9399b2; font-size: 11px; font-family: monospace;"
            "background: #11111b; border: 1px solid #313244; border-radius: 6px;")
        guide.addWidget(self.log_view)

        body.addWidget(guide_card, stretch=2)

    # preview ----------------------------------------------------------------

    def _log_line(self, text: str) -> None:
        """Append a timestamped line to the on-screen activity log."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {text}")

    def _start_preview(self, port: int | None = None) -> None:
        if self.preview_decoder is not None:
            self.preview_decoder.stop()
            self.preview_decoder = None
        spec = RECORD_SOURCES[self.source_combo.currentText()]
        if port is not None:
            spec = dict(spec, preview=port)
        self._preview_spec = spec
        self.preview_decoder = RawVideoDecoder(
            "puppet-recorder-preview", preview_command(spec), self,
            width=spec["out_w"], height=spec["out_h"])
        self.preview_decoder.frame_ready.connect(self._on_frame)
        self.preview_decoder.start()
        # The preview label EXPANDS to fill the card; the pixmap letterboxes
        # inside it (KeepAspectRatio), so the image is centered and uses all
        # available space for both vertical and horizontal sources.
        self.preview_label.setMinimumSize(320, 180)

    def _switch_source(self, source: str) -> None:
        """Restart the preview decoder on the newly selected source."""
        if source not in RECORD_SOURCES or self.recording:
            return
        self.preview_label.setText(f"Waiting for {source} stream…")
        self.preview_label.setMinimumSize(360, 270)
        self._start_preview()

    def _on_frame(self, image: QImage) -> None:
        if image.isNull():
            return
        scaled = image.scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(QPixmap.fromImage(scaled))

    # session flow -------------------------------------------------------------

    def _start_session(self) -> None:
        if self.recording:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = RENDERS_DIR / f"puppet_recording_{timestamp}.mp4"
        self.cues_path = RENDERS_DIR / f"puppet_recording_{timestamp}.cues.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        source = self.source_combo.currentText()
        record_port = RECORD_SOURCES[source]["record"]

        # The encoder tees TWO outputs: the take file and a live preview tap
        # (REC_PREVIEW_PORT).  The preview pane switches to that tap while
        # recording, so it never competes for the source's UDP datagrams and
        # shows exactly what is being written to disk.
        file_out = f"[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]{self.output_path}"
        preview_out = ("[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]"
                       f"udp://127.0.0.1:{REC_PREVIEW_PORT}?pkt_size=1316")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-fflags", "nobuffer+discardcorrupt", "-flags", "low_delay",
            "-analyzeduration", "2000000", "-probesize", "2000000",
            "-err_detect", "ignore_err",
            "-f", "mpegts",
            "-i", (f"udp://127.0.0.1:{record_port}"
                   "?reuse=1&fifo_size=1000000&overrun_nonfatal=1"),
            "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            # NOTE: no +faststart.  It forces a full remux pass after the
            # encoder stops, which is what made finalizing appear to hang;
            # ffmpeg reads the file fine with the moov at the end.
            "-f", "tee",
            "-use_fifo", "1",
            "-fifo_options", "attempt_recovery=1:recover_any_error=1"
                             ":drop_pkts_on_overflow=1",
            "-map", "0:v:0",
            "-y", f"{file_out}|{preview_out}",
        ]
        from PySide6.QtCore import QProcess
        self.record_process = QProcess(self)
        self.record_process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self.record_process.readyReadStandardOutput.connect(
            self._record_output)
        self.record_process.finished.connect(self._record_finished)
        self.record_process.errorOccurred.connect(self._record_error)
        self.record_process.start(cmd[0], cmd[1:])

        self.cues = []
        self.step_index = -1
        self.recording = True
        self.t0 = time.monotonic()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.status.set_state("recording", "working")
        self._log_line(f"recording {source} ({record_port}) -> "
                       f"{self.output_path.name}")
        self._log_line("preview switched to recording tap")
        # Preview now shows the recording tee, not the shared source port.
        self._start_preview(REC_PREVIEW_PORT)
        self.step_timer.start()

    def _record_output(self) -> None:
        """Surface ffmpeg warnings in the activity log."""
        if self.record_process is None:
            return
        data = bytes(
            self.record_process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")
        for line in data.strip().splitlines():
            line = line.strip()
            if line:
                self._log_line(f"ffmpeg: {line[:160]}")

    def _tick(self) -> None:
        if not self.recording:
            return
        now = time.monotonic()
        if self.step_index < 0 or now >= self.step_deadline:
            self._advance_step(now)
        else:
            step = self.script[self.step_index]
            elapsed = now - self.step_started
            self.progress.setValue(int(1000 * elapsed / step["seconds"]))

    def _advance_step(self, now: float) -> None:
        # close the previous perform cue
        if (self.step_index >= 0
                and self.script[self.step_index]["kind"] == "perform"):
            self.cues[-1]["t_end"] = round(now - self.t0, 3)

        self.step_index += 1
        if self.step_index >= len(self.script):
            self._stop_session()
            return
        step = self.script[self.step_index]
        self.step_started = now
        self.step_deadline = now + step["seconds"]

        if step["kind"] == "perform":
            self.cues.append({"action": step["action"],
                              "t_start": round(now - self.t0, 3),
                              "t_end": None})
            self.action_label.setText(self._perform_text(step["action"]))
            QApplication.beep()
            self._log_line(f"perform: {step['action']} "
                           f"({step['seconds']:.1f}s)")
        elif step["kind"] == "prepare":
            self.action_label.setText(f"Get ready…")
            self.next_label.setText(f"next: {step['action']}")
        else:
            self.action_label.setText(step["label"])
            self.next_label.setText(" ")
            self._log_line(step["label"].lower())

        done = sum(1 for s in self.script[:self.step_index + 1]
                   if s["kind"] == "perform")
        total = sum(1 for s in self.script if s["kind"] == "perform")
        self.step_label.setText(f"action {done}/{total}")

    def _perform_text(self, action: str) -> str:
        if action.startswith("digits_"):
            spoken = ["zero", "one", "two", "three", "four", "five", "six",
                      "seven", "eight", "nine"][int(action.split("_")[1])]
            return f'Say "{spoken.upper()}"'
        return {
            "turn_left": "Turn head LEFT",
            "turn_right": "Turn head RIGHT",
            "look_up": "Look UP",
            "look_down": "Look DOWN",
            "blink": "Blink",
            "open_mouth": "Open mouth",
        }.get(action, action)

    def _stop_session(self) -> None:
        if not self.recording:
            return
        self.step_timer.stop()
        now = time.monotonic()
        if (self.step_index < len(self.script)
                and self.step_index >= 0
                and self.script[self.step_index]["kind"] == "perform"):
            self.cues[-1]["t_end"] = round(now - self.t0, 3)
        self.recording = False
        self.status.set_state("stopping", "working")
        self._log_line("stop requested — waiting for encoder to finish")
        if self.record_process and self.record_process.state() in (
                QProcess.ProcessState.Running,
                QProcess.ProcessState.Starting):
            pid = self.record_process.processId()
            if pid:
                import os as _os
                try:
                    _os.kill(pid, signal.SIGINT)
                    self._log_line("sent SIGINT to encoder")
                except ProcessLookupError:
                    pass
            # Escalate if the encoder does not exit on its own: SIGTERM after
            # 8s, SIGKILL after 16s.  Without faststart the encoder exits in
            # well under a second, so escalation should never trigger.
            QTimer.singleShot(8000, lambda: self._escalate_stop(pid, 1))
        else:
            self._record_finished(0, None)

    def _escalate_stop(self, pid: int, level: int) -> None:
        """Escalate the encoder shutdown if it ignored the previous signal."""
        if self.recording or self.record_process is None:
            return
        if self.record_process.state() not in (
                QProcess.ProcessState.Running,
                QProcess.ProcessState.Starting):
            return
        import os as _os
        try:
            _os.kill(pid, signal.SIGTERM if level == 1 else signal.SIGKILL)
            self._log_line(f"escalated to {'SIGTERM' if level == 1 else 'SIGKILL'}")
        except ProcessLookupError:
            return
        if level == 1:
            QTimer.singleShot(8000, lambda: self._escalate_stop(pid, 2))

    def _record_finished(self, exit_code, exit_status) -> None:
        self.step_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.source_combo.setEnabled(True)
        self._log_line("encoder stopped")
        # Back to the live source preview (the recording tee is gone).
        self._start_preview()
        self._log_line("preview back on live source")
        document = {
            "schema": 1,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "video": self.output_path.name if self.output_path else None,
            "cues": [c for c in self.cues if c["t_end"] is not None],
        }
        if self.cues_path is not None:
            self.cues_path.write_text(json.dumps(document, indent=1),
                                      encoding="utf-8")
        size_mb = (self.output_path.stat().st_size / 1e6
                   if self.output_path and self.output_path.exists() else 0.0)
        self.status.set_state("done", "running")
        self.action_label.setText("Session complete")
        self.next_label.setText(" ")
        self._log_line(f"wrote {len(document['cues'])} cues -> "
                       f"{self.cues_path.name}")
        self._log_line(f"take file: {size_mb:.1f} MB")

    def _record_error(self, error) -> None:
        self.status.set_state("record error", "failed")
        self._log_line(f"record error: {error}")
        self.recording = False
        self.step_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.source_combo.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.recording:
            self._stop_session()
        if self.preview_decoder is not None:
            self.preview_decoder.stop()
            self.preview_decoder = None
        event.accept()


def main() -> int:
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(stylesheet())
    window = RecorderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
