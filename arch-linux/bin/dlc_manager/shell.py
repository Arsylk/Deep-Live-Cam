#!/usr/bin/env python3
"""Application shell: chrome, navigation, the refresh loop, and action wiring.

The shell owns the long-lived objects (decoders, timers, clients, history) and
does exactly three things per tick: collect documents, assemble one immutable
:class:`~dlc_manager.viewmodel.ManagerView`, and hand that value to each page.
Pages never reach back into the shell for state, and nothing here opens a
physical or virtual camera device.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import re
import time
from typing import Any

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camera_adapters import camera_schema
from camera_profiles import (
    CUSTOM_CAMERA_PROFILE,
    DEFAULT_CAMERA_PROFILE,
    profile_live_values,
)
from common import (
    DEFAULT_STATE_DIR,
    resolve_capture_device,
    resolve_preview_device,
    resolve_virtual_devices,
)
from source_history import SourceHistoryStore, default_source_history_directory

from .baseline import load_active_baseline
from .contracts import (
    ANDROID_NATIVE_PREVIEW_PORT,
    PREVIEW_FPS,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    SELECTED_STREAM_PORT,
    integer,
    local_mpegts_preview_command,
    readable_age,
)

RENDERS_DIR = Path("/var/lib/deep-live-cam/renders")
from .decoders import RawVideoDecoder
from .health import default_android_native_health_file, file_age, read_json
from .pages.analysis import AnalysisPage
from .pages.render import RenderPage
from .desired_state import (
    INPUT_ANDROID_BACK,
    INPUT_ANDROID_FRONT,
    INPUT_ARCH_WEBCAM,
    INPUT_ASSEMBLER,
    INPUT_PRERECORDED,
    INPUT_SPECS,
    OUTPUT_ANDROID_PHONE,
    OUTPUT_ARCH_CAMERA,
    PROCESSOR_ARCH,
    PROCESSOR_SPECS,
    PROCESSOR_WINDOWS,
    DesiredStateStore,
    default_state_path,
    local_processor_payload,
    local_processor_matches,
    processor_payload,
    reconciliation_payload,
)
from .pages.input import InputPage
from .pages.output import OutputPage
from .pages.processing import ProcessingPage
from .pages.system import SystemPage
from .phone_preview import PhoneReturnPreviewWindow
from .phone_route import (
    RELAY_LOCAL,
    RELAY_OFF,
    RELAY_WINDOWS,
    desired_relay_source,
    relay_desires,
    relay_is_closed,
    route_signature,
    windows_runtime_ready,
)
from .preview_transform import transform_preview_image
from .quality import HostMetrics, StreamQualityAnalyzer, face_detector
from .services import (
    HelperProcess,
    JsonProcess,
    SystemdProbe,
    WindowsControlClient,
    adapter_arguments,
    arch_persist_arguments,
    local_helper,
    offline_device_registry,
)
from .theme import stylesheet
from .viewmodel import (
    STREAM_RAW,
    STREAM_RESULT,
    DecoderStats,
    ManagerView,
    ViewInputs,
    build_view,
)
from .widgets import ElidedLabel, StatusPill


WORKSPACES = (
    ("Processor", "Machine, fixed model, source picture, and shared settings"),
    ("Input", "Arch webcam or the phone front/back camera"),
    ("Output", "Arch Xiaomi Cam, phone return, orientation, and preview"),
    ("Render", "Record, offline high-quality render, and replay pre-recorded video"),
)

SERVICE_UNITS = {
    "sender": "deep-live-cam-sender.service",
    "receiver": "deep-live-cam-receiver.service",
    "local_processor": "deep-live-cam-phone-processed.service",
    "phone_relay": "deep-live-cam-phone-return-relay.service",
}

PRESETS: dict[int, dict[str, Any]] = {
    1: {
        "processing_mode": "face_swap",
        "opacity": 1.0,
        "sharpness": 0.0,
        "mouth_mask_size": 0.0,
        "color_match_strength": 0.2,
        "interpolation_weight": 0.0,
        "many_faces": False,
        "enable_interpolation": False,
        "tracking_enabled": True,
        "detection_interval": 2,
        "tracking_smoothing": 0.5,
        "tracking_grace_frames": 3,
        "minimum_detection_score": 0.45,
        "minimum_face_size": 64,
        "enhancer": "none",
        "quality_mode": "monitor",
        "quality_auto_correct": False,
        "repair_hf_strength": 0.0,
        "repair_checkerboard": 0.0,
        "repair_wavelet": 0.0,
        "repair_boundary_mask": False,
        "repair_boundary_strength": 0.0,
        "repair_camera_detail": 0.0,
    },
    2: {
        "processing_mode": "face_swap",
        "opacity": 1.0,
        "sharpness": 0.2,
        "mouth_mask_size": 8.0,
        "color_match_strength": 0.35,
        "interpolation_weight": 0.75,
        "many_faces": False,
        "enable_interpolation": False,
        "tracking_enabled": True,
        "detection_interval": 1,
        "tracking_smoothing": 0.65,
        "tracking_grace_frames": 5,
        "minimum_detection_score": 0.45,
        "minimum_face_size": 64,
        "enhancer": "none",
        "quality_mode": "balanced",
        "quality_auto_correct": True,
        "repair_hf_strength": 0.3,
        "repair_checkerboard": 0.4,
        "repair_wavelet": 0.5,
        "repair_boundary_mask": True,
        "repair_boundary_strength": 0.35,
        "repair_camera_detail": 0.0,
    },
    3: {
        "processing_mode": "face_swap",
        "opacity": 1.0,
        "sharpness": 0.25,
        "mouth_mask_size": 10.0,
        "color_match_strength": 0.5,
        "interpolation_weight": 0.75,
        "many_faces": False,
        "enable_interpolation": True,
        "tracking_enabled": True,
        # Measured on the Windows RTX path: one optical-flow bridge frame
        # improved stability/flicker and reduced mean latency versus running
        # the noisier detector on every frame, without introducing misses.
        "detection_interval": 2,
        "tracking_smoothing": 0.72,
        "tracking_grace_frames": 6,
        "minimum_detection_score": 0.45,
        "minimum_face_size": 64,
        "enhancer": "none",
        "quality_mode": "strict",
        "quality_auto_correct": True,
        "repair_hf_strength": 0.35,
        "repair_checkerboard": 0.5,
        "repair_wavelet": 0.65,
        "repair_boundary_mask": True,
        "repair_boundary_strength": 0.5,
        "repair_camera_detail": 3.5,
    },
}


class ManagerWindow(QMainWindow):
    """The Deep-Live-Cam native manager window."""

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__()
        self.config = dict(config)
        self.state_dir = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
        self.windows_host = config.get("WINDOWS_HOST", "192.168.1.35")
        self.arch_host = config.get("ARCH_HOST", "192.168.1.11")
        self.android_enabled = config.get("ANDROID_BRIDGE_ENABLED", "1") == "1"
        self.android_serial = config.get("ANDROID_ADB_SERIAL", "")
        self.android_host = config.get("ANDROID_HOST", "192.168.1.12")
        self.android_camera_id = config.get("ANDROID_CAMERA_ID", "120")
        self.android_native_preview_port = self._config_int(
            "ANDROID_NATIVE_PREVIEW_PORT", ANDROID_NATIVE_PREVIEW_PORT
        )
        configured_health = config.get("ANDROID_NATIVE_HEALTH_FILE", "").strip()
        self.android_native_health_file = (
            Path(configured_health).expanduser()
            if configured_health
            else default_android_native_health_file()
        )
        self.phone_relay_socket = Path(
            config.get(
                "PHONE_RETURN_RELAY_CONTROL_SOCKET",
                str(self.state_dir / "phone-return-relay-control.sock"),
            )
        ).expanduser()
        self.phone_relay_health_file = self.state_dir / "phone-return-relay.json"
        self.phone_relay_health: dict[str, Any] = {}
        self._latest_native_health: dict[str, Any] = {}
        self.preview_port = self._config_int("MANAGER_PREVIEW_PORT", 11_001)
        self.selected_stream_port = int(
            config.get(
                "WINDOWS_SELECTED_STREAM_PORT",
                config.get("WINDOWS_BROADCAST_PORT", str(SELECTED_STREAM_PORT)),
            )
        )
        self.output_preview_port = self._config_int(
            "MANAGER_OUTPUT_PREVIEW_PORT", 11_003
        )
        self.local_processed_preview_port = self._config_int(
            "LOCAL_PROCESSED_PREVIEW_PORT", 11_007
        )
        self.prerecorded_preview_port = self._config_int(
            "PRERECORDED_PREVIEW_PORT", 11_011
        )
        self.active_output_preview_port = self.output_preview_port
        self.virtual_devices = resolve_virtual_devices(config)
        self.virtual_camera = str(resolve_preview_device(config))
        self.physical_camera = config.get("PHYSICAL_CAMERA", "unknown")
        self.capture_camera = str(resolve_capture_device(config, self.state_dir))

        configured_desired_state = config.get("MANAGER_STATE_FILE", "").strip()
        self.desired_store = DesiredStateStore(
            Path(configured_desired_state).expanduser()
            if configured_desired_state
            else default_state_path()
        )
        self.desired_state = self.desired_store.snapshot()
        self.desired_initialized = self.desired_store.loaded_from_disk
        self.windows_confirmed_config: dict[str, Any] = {}
        self.windows_was_reachable = False
        self.windows_reconcile_requested = True

        self.windows = WindowsControlClient(self.windows_host, self)
        self.windows_health: dict[str, Any] | None = None
        self.windows_error: str | None = "not queried"
        self.windows_config: dict[str, Any] = {}
        self.windows_devices: dict[str, Any] = {}
        self.registry_live = False
        self.pending_windows_changes: dict[str, Any] = {}
        self.windows_inflight_changes: dict[str, Any] = {}
        self.windows_config_timer = QTimer(self)
        self.windows_config_timer.setSingleShot(True)
        self.windows_config_timer.setInterval(180)
        self.windows_config_timer.timeout.connect(self._post_windows_config)

        self.android_status: dict[str, Any] = {}
        self.android_error: str | None = "not queried"
        self.android_last_query = 0.0
        self.android_probe = JsonProcess("Android bridge", self)
        self.android_output_retry_timer = QTimer(self)
        self.android_output_retry_timer.setSingleShot(True)
        self.android_output_retry_timer.setInterval(3000)
        self.android_output_retry_timer.timeout.connect(
            self._apply_android_output_configuration
        )
        self.policy_helper = HelperProcess("system-camera selector", self)
        self.output_helper = HelperProcess("output transformer", self)
        self.android_output_helper = HelperProcess("Android output transformer", self)
        self.local_processor_helper = HelperProcess("local processor control", self)
        self.local_processor_inflight_revision: int | None = None
        self.phone_relay_helper = HelperProcess("phone-return relay control", self)
        self.prerecorded_relay_process: QProcess | None = None
        self.prerecorded_relay_path: str | None = None
        self.prerecorded_relay_mode: str | None = None
        self._assembler_process: QProcess | None = None
        self._assembler_output: Path | None = None
        self.camera_helper = HelperProcess("camera adapter", self)
        self.systemd = SystemdProbe()
        self.metrics = HostMetrics()

        configured_history = config.get("SOURCE_HISTORY_DIR", "").strip()
        self.source_history = SourceHistoryStore(
            Path(configured_history).expanduser()
            if configured_history
            else default_source_history_directory()
        )
        self.pending_source: tuple[bytes, str] | None = None
        self.source_upload_inflight: tuple[bytes, str] | None = None
        self.source_retry_timer = QTimer(self)
        self.source_retry_timer.setSingleShot(True)
        self.source_retry_timer.setInterval(2500)
        self.source_retry_timer.timeout.connect(self._retry_source_upload)

        self.snapshot: dict[str, Any] = {}
        self.view: ManagerView | None = None
        self.previous_result_frames = 0
        self.previous_raw_frames = 0
        self.last_pipeline_state = ""
        self.delayed_frames: deque[tuple[float, QImage]] = deque(maxlen=120)
        self.phone_return_preview_window: PhoneReturnPreviewWindow | None = None
        self.phone_return_live = False
        self.pending_policy = ""
        self.prerecorded_paused = False
        self.prerecorded_seek: float | None = None
        self.pending_output_configuration = False
        self.output_configuration_inflight = False
        self.output_configuration_inflight_state: tuple[str, bool, bool, int] | None = None
        self.pending_android_output_configuration = False
        self.android_output_inflight_state: tuple[bool, bool, int] | None = None
        self.pending_local_processor_state = False
        self.phone_route_signature: tuple[Any, ...] | None = None
        self.phone_route_quiesced_signature: tuple[Any, ...] | None = None
        self.phone_route_applied_signature: tuple[Any, ...] | None = None
        self.phone_relay_inflight_source: str | None = None
        self.phone_relay_inflight_signature: tuple[Any, ...] | None = None
        self.camera_persisting = False
        self.camera_last_result_ok: bool | None = None
        self.camera_last_result_detail = ""
        self.pending_camera_request: tuple[str, dict[str, Any], bool] | None = None
        self.camera_live_timer = QTimer(self)
        self.camera_live_timer.setSingleShot(True)
        self.camera_live_timer.setInterval(180)
        self.camera_live_timer.timeout.connect(self._preview_camera_configuration)

        detector = face_detector()
        self.raw_quality = StreamQualityAnalyzer(detector)
        self.result_quality = StreamQualityAnalyzer(detector)
        self.quality_analysis_enabled = True

        self.setWindowTitle("Deep-Live-Cam Manager")
        self.resize(1500, 900)
        self.setMinimumSize(980, 680)

        self.input_decoder = RawVideoDecoder("raw-preview", self._input_command(), self)
        self.output_decoder = RawVideoDecoder(
            "result-preview", self._output_command(), self
        )
        # A dedicated reader for the prerecorded framing preview.  It taps the
        # same encoded relay the file_relay writes its framed picture to, so the
        # Input tab shows exactly what the receiver produces (zoom + offset +
        # black fill) rather than a client-side re-render.  Started only while
        # prerecorded input is active.
        self.framing_decoder = RawVideoDecoder(
            "framing-preview",
            local_mpegts_preview_command(self.prerecorded_preview_port),
            self,
        )
        self.input_decoder.frame_ready.connect(self._raw_frame)
        self.output_decoder.frame_ready.connect(self._result_frame)
        self.framing_decoder.frame_ready.connect(self._framing_frame)
        for decoder in (self.input_decoder, self.output_decoder):
            decoder.log_line.connect(self.append_log)
            decoder.lifecycle.connect(self.append_log)

        self._build_ui()
        self._connect_actions()
        self.setStyleSheet(stylesheet())
        self.analysis_page.load_baseline(load_active_baseline())
        self._use_offline_device_registry()
        self._initialize_desired_source()
        self._refresh_source_history()
        self._sync_camera_form()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh_stats)
        self.timer.start()
        self.alignment_timer = QTimer(self)
        self.alignment_timer.setInterval(16)
        self.alignment_timer.timeout.connect(self.present_aligned_input)
        self.alignment_timer.start()
        self.output_decoder.start()
        self.refresh_stats()
        QTimer.singleShot(0, self.windows.request_config)
        QTimer.singleShot(0, self.windows.request_devices)
        self.pending_output_configuration = True
        self.pending_android_output_configuration = True
        self.pending_local_processor_state = True
        QTimer.singleShot(0, self._reconcile_desired_endpoints)
        # If the manager launches with prerecorded already selected, bring up
        # the framing preview and push the persisted framing to the receiver.
        if str(self.desired_state.get("input")) in (INPUT_PRERECORDED, INPUT_ASSEMBLER):
            self._write_prerecorded_adjust()
            QTimer.singleShot(0, self._start_framing_preview)

    # ------------------------------------------------------------ construction

    def _config_int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, str(default)))
        except (TypeError, ValueError):
            return default

    def _input_command(self) -> list[str]:
        return local_mpegts_preview_command(self.preview_port)

    def _output_command(self, port: int | None = None) -> list[str]:
        selected_port = self.output_preview_port if port is None else port
        source = (
            f"udp://127.0.0.1:{selected_port}?"
            "reuse=1&fifo_size=1000000&overrun_nonfatal=1"
        )
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0",
            "-probesize", "32768", "-f", "mpegts", "-i", source,
            "-map", "0:v:0", "-an",
            "-vf", f"scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:flags=fast_bilinear,"
            f"fps={PREVIEW_FPS},format=rgb24",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]

    def _set_output_preview_source(self, *, local_processed: bool) -> None:
        """Point the result reader at whichever receiver relay is authoritative.

        Both endpoints are receiver-owned encoded relays, so switching between
        them cannot open, close, or contend for a camera device.
        """
        selected_port = (
            self.local_processed_preview_port
            if local_processed
            else self.output_preview_port
        )
        if selected_port == self.active_output_preview_port:
            if local_processed:
                self.output_pane.set_heading(
                    "LOCAL NATIVE-256 OUTPUT",
                    f"receiver-owned encoded relay · local MPEG-TS UDP "
                    f":{selected_port}",
                )
            return
        was_running = self.output_decoder.running
        self.output_decoder.set_command(self._output_command(selected_port))
        self.active_output_preview_port = selected_port
        self.previous_result_frames = 0
        if was_running:
            self.output_decoder.start()
        if local_processed:
            self.output_pane.set_heading(
                "LOCAL NATIVE-256 OUTPUT",
                f"receiver-owned encoded relay · local MPEG-TS UDP :{selected_port}",
            )
            self.output_pane.clear_image("WAITING FOR LOCAL NATIVE-256")
        else:
            self.output_pane.set_heading(
                "SELECTED WINDOWS STREAM",
                f"Windows SRT {self.windows_host}:{self.selected_stream_port} "
                f"via receiver-local UDP :{selected_port}",
            )
            self.output_pane.clear_image("WAITING FOR SELECTED WINDOWS STREAM")

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())

        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setObjectName("primaryTabs")
        self.workspace_tabs.setDocumentMode(True)
        self.workspace_tabs.setMovable(False)
        self.workspace_tabs.setTabPosition(QTabWidget.TabPosition.North)
        # Compatibility name retained for automation which only relies on the
        # common count/currentIndex/widget API.
        self.workspace_stack = self.workspace_tabs
        layout.addWidget(self.workspace_tabs, 1)
        self.setCentralWidget(root)

        self.processing_page = ProcessingPage()
        self.input_page = InputPage()
        self.output_page = OutputPage()
        self.render_page = RenderPage()
        self.routing_page = self.input_page
        self.live_page = self.output_page
        # Passive diagnostics remain controller-owned but are no longer mixed
        # into the three primary decisions. They feed snapshots and metrics.
        self.analysis_page = AnalysisPage(self)
        self.system_page = SystemPage(self)
        self.analysis_page.setVisible(False)
        self.system_page.setVisible(False)
        self.pages = (
            self.processing_page,
            self.input_page,
            self.output_page,
            self.render_page,
        )
        for title, description in WORKSPACES:
            page = self.pages[self.workspace_tabs.count()]
            index = self.workspace_tabs.addTab(page, title)
            self.workspace_tabs.setTabToolTip(index, description)
            self.workspace_tabs.setTabWhatsThis(index, description)

        self.output_pane = self.output_page.result_pane
        self.input_pane = self.output_page.raw_pane
        self.stats_box = self.system_page.stats_box
        self.log_box = self.system_page.log_box
        self.workspace_navigation_buttons: list[QPushButton] = []
        self.workspace_tabs.setCurrentIndex(0)
        self.processing_page.set_processor(str(self.desired_state["processor"]))
        self.processing_page.apply_windows_config(
            dict(self.desired_state["processing"])
        )
        self.input_page.set_input(str(self.desired_state["input"]))
        self.input_page.set_prerecorded_path(self.desired_state.get("prerecorded_path"))
        self.input_page.set_prerecorded_mode(self.desired_state.get("prerecorded_mode", "loop"))
        if str(self.desired_state["input"]) in (INPUT_PRERECORDED, INPUT_ASSEMBLER):
            self._apply_prerecorded_input()
        self.output_page.set_output_state(self.desired_state)

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("appHeader")
        header = QHBoxLayout(frame)
        header.setContentsMargins(18, 10, 18, 10)
        header.setSpacing(14)
        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title = QLabel("DEEP-LIVE-CAM")
        title.setObjectName("appTitle")
        subtitle = QLabel("INPUT · PROCESSOR · OUTPUT · OFFLINE-READY")
        subtitle.setObjectName("appSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addWidget(self._build_ribbon(), 1)
        self.alert_pill = StatusPill("STARTING", "unknown")
        self.alert_pill.setAccessibleName("Overall warning state")
        header.addWidget(self.alert_pill)
        return frame

    def _build_ribbon(self) -> QWidget:
        """One sentence across the top: input → processor → system camera."""
        frame = QFrame()
        frame.setObjectName("routeRibbon")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(12)
        self.ribbon_cells: dict[str, tuple[ElidedLabel, StatusPill]] = {}
        for index, (key, caption) in enumerate(
            (
                ("input", "INPUT"),
                ("processor", "PROCESSOR"),
                ("output", "SYSTEM CAMERA"),
            )
        ):
            if index:
                arrow = QLabel("→")
                arrow.setObjectName("ribbonArrow")
                layout.addWidget(arrow)
            cell = QVBoxLayout()
            cell.setSpacing(1)
            caption_label = QLabel(caption)
            caption_label.setObjectName("ribbonLabel")
            value = ElidedLabel("…")
            value.setObjectName("ribbonValue")
            # Runtime text here is long and variable; it must never become the
            # floor for the window width.
            pill = StatusPill("…", elide=True)
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(value, 1)
            row.addWidget(pill)
            cell.addWidget(caption_label)
            cell.addLayout(row)
            layout.addLayout(cell, 1)
            self.ribbon_cells[key] = (value, pill)
        return frame

    def show_workspace(self, index: int) -> None:
        """Switch pages. Decoders, timers, and helper processes are untouched."""
        self.workspace_tabs.setCurrentIndex(index)

    def _connect_actions(self) -> None:
        self.output_page.phoneReturnRequested.connect(self.open_phone_return_preview)
        self.output_page.reconnectRequested.connect(self.restart_readers)
        self.output_page.outputToggled.connect(self.set_output_enabled)
        self.output_page.transformChanged.connect(self.set_output_transform)
        self.output_page.processorProcessingToggled.connect(
            self.set_processor_processing_enabled
        )

        self.processing_page.sourcePictureRequested.connect(self.select_source_image)
        self.processing_page.historyPictureRequested.connect(self.apply_history_picture)
        self.processing_page.settingChanged.connect(self.windows_setting_changed)
        self.processing_page.presetRequested.connect(self.apply_processing_preset)
        self.processing_page.processorChanged.connect(self.select_processor)

        self.input_page.inputSelected.connect(self.select_input)
        self.input_page.prerecordedVideoSelected.connect(self.select_prerecorded_source)
        self.input_page.prerecordedModeChanged.connect(self.set_prerecorded_mode)
        self.input_page.prerecordedAdjustChanged.connect(
            lambda ox, oy, zoom: self.set_prerecorded_adjust(
                offset_x=ox, offset_y=oy, zoom=zoom
            )
        )
        self.input_page.prerecordedPauseToggled.connect(
            self.toggle_prerecorded_paused
        )
        self.input_page.prerecordedSeekRequested.connect(self.seek_prerecorded)
        self.input_page.assemblerLibChanged.connect(self.set_assembler_lib)
        self.input_page.assemblerTokensChanged.connect(self.set_assembler_tokens)
        self.input_page.assemblerAssembleRequested.connect(self.assemble_and_load)
        self.input_page.cameraControlChanged.connect(self.camera_control_changed)
        self.input_page.cameraSaveRequested.connect(self.save_camera_configuration)

        self.render_page.prerecordedSourceRequested.connect(self.select_prerecorded_source)

        self.analysis_page.comparisonChanged.connect(self.delayed_frames.clear)
        self.analysis_page.measurementToggled.connect(self.set_measurement_enabled)
        self.analysis_page.resetRequested.connect(self.reset_measurements)

        self.system_page.reconnectRequested.connect(self.restart_readers)
        self.system_page.copySnapshotRequested.connect(self.copy_snapshot)
        self.system_page.reloadWindowsRequested.connect(self.windows.request_config)

        self.windows.healthReceived.connect(self._windows_health_received)
        self.windows.healthFailed.connect(self._windows_health_failed)
        self.windows.configReceived.connect(self._windows_config_received)
        self.windows.configFailed.connect(self._windows_config_failed)
        self.windows.configApplied.connect(self._windows_config_applied)
        self.windows.configRejected.connect(self._windows_config_rejected)
        self.windows.devicesReceived.connect(self._windows_devices_received)
        self.windows.devicesFailed.connect(self._windows_devices_failed)
        self.windows.selectionSucceeded.connect(self._windows_selection_succeeded)
        self.windows.selectionFailed.connect(self._windows_selection_failed)
        self.windows.sourceUploaded.connect(self._source_uploaded)
        self.windows.sourceUploadFailed.connect(self._source_upload_failed)

        self.android_probe.parsed.connect(self._android_status_parsed)
        self.policy_helper.finished.connect(self._policy_finished)
        self.policy_helper.failedToStart.connect(self._policy_unavailable)
        self.output_helper.finished.connect(self._output_configuration_finished)
        self.output_helper.failedToStart.connect(self._output_configuration_unavailable)
        self.android_output_helper.finished.connect(
            self._android_output_configuration_finished
        )
        self.android_output_helper.failedToStart.connect(
            self._android_output_configuration_unavailable
        )
        self.local_processor_helper.finished.connect(self._local_processor_finished)
        self.local_processor_helper.failedToStart.connect(
            self._local_processor_unavailable
        )
        self.phone_relay_helper.finished.connect(self._phone_relay_finished)
        self.phone_relay_helper.failedToStart.connect(
            self._phone_relay_unavailable
        )
        self.camera_helper.finished.connect(self._camera_finished)
        self.camera_helper.failedToStart.connect(self._camera_unavailable)

    # -------------------------------------------------------------- frame path

    def _raw_frame(self, image: QImage) -> None:
        if self.quality_analysis_enabled:
            self.raw_quality.observe(image)
        if self.analysis_page.comparison_delay_ms() > 0:
            self.delayed_frames.append((time.monotonic(), image))
        else:
            self.delayed_frames.clear()
            self.input_pane.set_image(image)

    def present_aligned_input(self) -> None:
        delay = self.analysis_page.comparison_delay_ms()
        if delay <= 0 or not self.delayed_frames:
            return
        target = time.monotonic() - delay / 1000.0
        selected: QImage | None = None
        while self.delayed_frames and self.delayed_frames[0][0] <= target:
            _, selected = self.delayed_frames.popleft()
        if selected is not None:
            self.input_pane.set_image(selected)

    def _result_frame(self, image: QImage) -> None:
        if self.quality_analysis_enabled:
            self.result_quality.observe(image)
        transform = self.desired_state["output_transform"]
        self.output_pane.set_image(
            transform_preview_image(
                image,
                mirror=bool(transform["mirror"]),
                rotation=int(transform["rotation"]),
            )
        )

    def _framing_frame(self, image: QImage) -> None:
        """Feed the prerecorded framing preview on the Input tab."""
        self.input_page.set_prerecorded_preview_frame(image)

    # ----------------------------------------------------------------- refresh

    def refresh_stats(self) -> None:
        sender = read_json(self.state_dir / "sender.json") or {}
        receiver = read_json(self.state_dir / "receiver.json") or {}
        self._latest_receiver = receiver
        shadow = read_json(self.state_dir / "shadow.json") or {}
        native_health = read_json(self.android_native_health_file) or {}
        self._latest_native_health = native_health
        # Feed the prerecorded transport bar its live position/duration.
        prerecorded_state = receiver.get("prerecorded")
        if isinstance(prerecorded_state, dict):
            self.input_page.set_prerecorded_playback(
                position=prerecorded_state.get("position"),
                duration=prerecorded_state.get("duration"),
                paused=bool(prerecorded_state.get("paused", False)),
            )
        relay_health = read_json(self.phone_relay_health_file) or {}
        try:
            disk_relay_revision = int(relay_health.get("revision", -1))
            known_relay_revision = int(self.phone_relay_health.get("revision", -1))
        except (TypeError, ValueError):
            disk_relay_revision = known_relay_revision = -1
        if disk_relay_revision >= known_relay_revision:
            self.phone_relay_health = relay_health
        services = self.systemd.states(SERVICE_UNITS)
        active_entry = self.source_history.active_entry()
        local_service_running = (
            services.get("local_processor", {}).get("ActiveState") == "active"
        )
        local_source_path = active_entry.cache_path if active_entry is not None else None
        if local_service_running and native_health.get("state") == "running":
            if local_processor_matches(
                self.desired_state,
                native_health,
                source_path=local_source_path,
            ):
                if not self.local_processor_helper.busy():
                    self.pending_local_processor_state = False
            elif not self.local_processor_helper.busy():
                # A process restart resets the worker to its safe inactive
                # bootstrap. Reapply the durable document without touching a
                # physical camera, virtual sink, or phone provider.
                self.pending_local_processor_state = True
                QTimer.singleShot(0, self._apply_local_processor_state)

        pids: set[int] = set()
        for document in (sender, receiver):
            for key in ("pid", "decoder_pid", "sink_pid"):
                try:
                    value = int(document.get(key))
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    pids.add(value)

        inputs = ViewInputs(
            config=self.config,
            sender=sender,
            receiver=receiver,
            shadow=shadow,
            native_health=native_health,
            native_health_age=file_age(self.android_native_health_file),
            native_preview_port=self.android_native_preview_port,
            phone_relay_health=self.phone_relay_health,
            phone_relay_health_age=file_age(self.phone_relay_health_file),
            android_status=self.android_status,
            android_error=self.android_error,
            windows_health=self.windows_health,
            windows_error=self.windows_error,
            windows_config=self.windows_config,
            windows_devices=self.windows_devices,
            registry_live=self.registry_live,
            selection_in_flight=self.windows.selection_in_flight(),
            services=services,
            result_stream=self._decoder_stats(
                self.output_decoder,
                self.previous_result_frames,
                self.active_output_preview_port,
            ),
            raw_stream=self._decoder_stats(
                self.input_decoder, self.previous_raw_frames, self.preview_port
            ),
            comparison_delay_ms=self.analysis_page.comparison_delay_ms(),
            identity_filename=active_entry.filename if active_entry else None,
            identity_cache_path=str(active_entry.cache_path) if active_entry else None,
            identity_identifier=active_entry.identifier if active_entry else None,
            identity_used_at=active_entry.used_at if active_entry else None,
            identity_history_count=len(self.source_history.entries()),
            capture_device=self.capture_camera,
            virtual_devices=[str(device) for device in self.virtual_devices],
            device_nodes_present=self._device_nodes_present(shadow),
            host_metrics=self.metrics.sample(pids),
            now=time.time(),
        )
        self.previous_result_frames = self.output_decoder.frames
        self.previous_raw_frames = self.input_decoder.frames

        view = build_view(inputs)
        self.view = view
        self.phone_return_live = view.phone_return_live
        self._apply_decoder_policy(view, services)
        self._render(view, sender, receiver, shadow, native_health, services)
        self.input_decoder.flush_log()
        self.output_decoder.flush_log()
        self._query_android_status()
        self.windows.request_health()
        self.windows.request_devices()
        if self.windows_reconcile_requested or self.pending_windows_changes:
            self.windows.request_config()
        self._reconcile_phone_route()

    @staticmethod
    def _decoder_stats(
        decoder: RawVideoDecoder, previous: int, port: int
    ) -> DecoderStats:
        return DecoderStats(
            running=decoder.running,
            fps=decoder.display_fps(previous),
            frames=decoder.frames,
            dropped=decoder.dropped,
            age=decoder.age(),
            port=port,
            restarts=decoder.restarts,
        )

    def _device_nodes_present(self, shadow: dict[str, Any]) -> bool | None:
        preserved = [Path(path) for path in shadow.get("preserved", [])]
        candidates = list(self.virtual_devices) + (
            preserved if preserved else [Path(self.capture_camera)]
        )
        if not candidates:
            return None
        return all(path.exists() for path in candidates)

    def _apply_decoder_policy(
        self, view: ManagerView, services: dict[str, dict[str, str]]
    ) -> None:
        """Keep both readers pointed at the relay that is currently meaningful.

        Driven by service health only.  Page navigation never reaches this, and
        every endpoint involved is an owner-produced loopback relay.
        """
        self._set_output_preview_source(
            local_processed=self.desired_state["processor"] == PROCESSOR_ARCH
        )
        if not self.output_decoder.running:
            self.output_decoder.start()
        capture_running = services.get("sender", {}).get("ActiveState") == "active"
        # StreamView.available answers "does this route have a raw copy at
        # all?"; its state would still read "off" while the reader is stopped,
        # which would keep it stopped forever.
        raw_available = capture_running and view.stream(STREAM_RAW).available
        if raw_available and not self.input_decoder.running:
            self.input_decoder.start()
        elif not raw_available and self.input_decoder.running:
            self.input_decoder.stop()
            self.delayed_frames.clear()

    def _render(
        self,
        view: ManagerView,
        sender: dict[str, Any],
        receiver: dict[str, Any],
        shadow: dict[str, Any],
        native_health: dict[str, Any],
        services: dict[str, dict[str, str]],
    ) -> None:
        self.live_page.render(view, phone_return_live=view.phone_return_live)
        self.processing_page.set_windows_available(
            view.processor.windows_reachable, self.windows_error or ""
        )
        self.processing_page.render(view)
        self.input_page.render(view)
        selected_input = str(self.desired_state["input"])
        input_status = self._effective_input_status(selected_input)
        self.input_page.set_input(selected_input, status=input_status)
        arch_output_matches = self._arch_output_matches(
            receiver, self._arch_output_signature()
        )
        if arch_output_matches and not self.output_helper.busy():
            self.pending_output_configuration = False
        elif receiver and not self.output_helper.busy():
            # A restarted or externally changed receiver must converge back to
            # the durable document.  This is a hot socket update and never
            # replaces the V4L2 sink or its public identity.
            self.pending_output_configuration = True
            QTimer.singleShot(0, self._apply_output_configuration)
        self.output_page.set_output_state(
            self.desired_state,
            pending=bool(
                self.pending_output_configuration
                or self.output_configuration_inflight
                or self.pending_android_output_configuration
                or self.android_output_helper.busy()
            ),
        )
        self.output_page.set_delivery_status(
            self.desired_state,
            receiver,
            self.android_status,
            receiver_service_active=(
                services.get("receiver", {}).get("ActiveState") == "active"
            ),
            receiver_health_age=file_age(self.state_dir / "receiver.json"),
        )
        self.analysis_page.render(
            view,
            raw_metrics=self.raw_quality.latest,
            result_metrics=self.result_quality.latest,
            enabled=self.quality_analysis_enabled,
        )
        self.render_page.render(view)
        self.system_page.render(
            view,
            ports={
                "raw": self.preview_port,
                "result": self.active_output_preview_port,
                "phone_return": self.android_native_preview_port,
            },
        )
        self._render_header(view, receiver, services, input_status=input_status)
        self._render_snapshot(view, sender, receiver, shadow, native_health, services)
        if self.pending_source is not None:
            self.processing_page.set_source_status(
                "The selected picture is active locally; Windows upload is "
                "queued and will retry automatically."
            )
        elif (
            self.source_history.active_entry() is not None
            and not self.windows.upload_in_flight()
        ):
            self.processing_page.set_source_status(
                "The selected picture is the durable identity for both processors. "
                + view.identity.windows_detail
            )
        elif not self.windows.upload_in_flight():
            self.processing_page.set_source_status(
                "Choose a source picture once; it will be cached locally and "
                "kept synchronized to both processors."
            )

    def _effective_input_status(self, selected_input: str) -> str:
        """Describe whether the durable semantic input is effective right now."""
        selected_specification = INPUT_SPECS[selected_input]
        camera_switching = bool(
            self.camera_helper.busy() or self.pending_camera_request is not None
        )
        camera_failed = self.camera_last_result_ok is False
        lens_mismatch = False
        if selected_specification.lens_facing:
            reported = self.android_status.get("capture_metrics")
            if not isinstance(reported, dict):
                reported = self.android_status.get("controls")
            effective_lens = (
                reported.get("lens_facing") if isinstance(reported, dict) else None
            )
            if effective_lens and effective_lens != selected_specification.lens_facing:
                lens_mismatch = True
        route_offline = False
        route_switching = False
        route_mismatch = False
        if self.desired_state["processor"] == PROCESSOR_WINDOWS:
            if self.windows.selection_in_flight():
                route_switching = True
            elif not windows_runtime_ready(
                self.windows_devices, self._desired_windows_device()
            ):
                if self.registry_live:
                    route_switching = True
                else:
                    route_offline = True
        else:
            control = self._latest_native_health.get("control")
            control = control if isinstance(control, dict) else {}
            effective = control.get("effective")
            effective = effective if isinstance(effective, dict) else {}
            if self._latest_native_health.get("state") != "running":
                route_offline = True
            elif self.local_processor_helper.busy() or self.pending_local_processor_state:
                route_switching = True
            elif effective.get("input") not in (None, selected_input):
                route_mismatch = True
        if camera_failed:
            return "failed"
        if route_offline:
            return "offline"
        if camera_switching or route_switching:
            return "switching"
        if lens_mismatch or route_mismatch:
            return "mismatch"
        return "desired"

    def _render_header(
        self,
        view: ManagerView,
        receiver: dict[str, Any],
        services: dict[str, dict[str, str]],
        *,
        input_status: str,
    ) -> None:
        route = view.route
        processor = view.processor
        input_value, input_pill = self.ribbon_cells["input"]
        processor_value, processor_pill = self.ribbon_cells["processor"]
        output_value, output_pill = self.ribbon_cells["output"]

        input_specification = INPUT_SPECS[str(self.desired_state["input"])]
        input_value.setText(input_specification.label)
        input_presentations = {
            "desired": ("SELECTED", "running"),
            "switching": ("APPLYING", "working"),
            "offline": ("SAVED · PENDING", "warning"),
            "failed": ("SETTINGS FAILED", "failed"),
            "mismatch": ("SYNC PENDING", "warning"),
        }
        input_pill.set_state(*input_presentations.get(
            input_status, ("SAVED · PENDING", "warning")
        ))

        processor_key = str(self.desired_state["processor"])
        specification = PROCESSOR_SPECS[processor_key]
        processor_value.setText(
            f"{specification.label} · {specification.model}/{specification.backend}"
        )
        if processor_key == PROCESSOR_WINDOWS:
            source_synced = self._windows_source_synchronized()
            settings_synced = not (
                self.pending_windows_changes or self.windows_inflight_changes
            )
            processor_pill.set_state(
                "SYNCED"
                if processor.windows_reachable and settings_synced and source_synced
                else ("SYNCING" if processor.windows_reachable else "OFFLINE · SAVED"),
                "running"
                if processor.windows_reachable and settings_synced and source_synced
                else ("working" if processor.windows_reachable else "warning"),
            )
        else:
            processor_pill.set_state(
                "DEVELOPMENT · RUNNING" if processor.local_running else "DEVELOPMENT · OFFLINE",
                "warning" if processor.local_running else "failed",
            )

        camera = view.system_camera
        enabled_outputs = [
            "Arch Xiaomi Cam" if key == OUTPUT_ARCH_CAMERA else "Phone Camera2 120"
            for key, enabled in self.desired_state["outputs"].items()
            if enabled
        ]
        transform = self.desired_state["output_transform"]
        output_value.setText(
            f"{', '.join(enabled_outputs) or 'no delivery'} · "
            f"{'mirror · ' if transform['mirror'] else ''}{transform['rotation']}°"
        )
        desired_outputs = self.desired_state["outputs"]
        arch_synced = self._arch_output_matches(
            receiver, self._arch_output_signature()
        )
        phone_synced = self._android_output_delivery_matches(
            self.android_status, self._android_output_signature()
        )
        arch_live = bool(
            arch_synced
            and services.get("receiver", {}).get("ActiveState") == "active"
            and (age := file_age(self.state_dir / "receiver.json")) is not None
            and age <= 3.0
            and receiver.get("status") == "streaming"
            and isinstance(receiver.get("sink_pid"), int)
            and receiver.get("sink_pid", 0) > 0
        )
        phone_config_synced = self._android_output_configuration_matches(
            self.android_status, self._android_output_signature()
        )
        configured = []
        live = []
        if desired_outputs[OUTPUT_ARCH_CAMERA]:
            configured.append(arch_synced)
            live.append(arch_live)
        if desired_outputs[OUTPUT_ANDROID_PHONE]:
            configured.append(phone_config_synced)
            live.append(phone_synced)
        if not configured:
            output_pill.set_state("DELIVERY OFF", "warning")
        elif all(live):
            output_pill.set_state("STREAMS LIVE", "running")
        elif all(configured):
            output_pill.set_state("CONFIG SYNCED · WAITING", "working")
        else:
            output_pill.set_state("SYNCING / OFFLINE", "working")

        worst = view.worst_alert()
        if worst is None:
            self.alert_pill.set_state("NO WARNINGS", "running")
            self.alert_pill.setToolTip("Every node reported a usable state.")
        else:
            count = len(view.alerts)
            self.alert_pill.set_state(
                f"{count} WARNINGS" if count > 1 else "1 WARNING", worst.severity
            )
            self.alert_pill.setToolTip(
                f"{worst.component}: {worst.message}\nNext: {worst.next_action}"
            )

        pipeline_state = (
            f"{route.key}/{camera.active_input}/{processor.windows_state}/"
            f"{view.android.state}"
        )
        if pipeline_state != self.last_pipeline_state:
            self.append_log(f"pipeline state → {pipeline_state}")
            self.last_pipeline_state = pipeline_state

    def _render_snapshot(
        self,
        view: ManagerView,
        sender: dict[str, Any],
        receiver: dict[str, Any],
        shadow: dict[str, Any],
        native_health: dict[str, Any],
        services: dict[str, dict[str, str]],
    ) -> None:
        windows = self.windows_health or {}
        win_input = windows.get("input") if isinstance(windows.get("input"), dict) else {}
        win_output = (
            windows.get("output") if isinstance(windows.get("output"), dict) else {}
        )
        processing = (
            windows.get("processing")
            if isinstance(windows.get("processing"), dict)
            else {}
        )
        result = view.stream(STREAM_RESULT)
        raw = view.stream(STREAM_RAW)
        mapping_state = (
            "ready"
            if view.shadow_ready
            else "not confirmed"
            if view.shadow_ready is None
            else "INCOMPLETE"
        )
        text = (
            "ACTIVE ROUTE\n"
            f"  route             {view.route.badge}\n"
            f"  summary           {view.route.summary}\n"
            f"  detail            {view.route.detail}\n"
            f"  warning           {view.route.warning or 'none'}\n"
            f"  windows bypassed  {view.route.windows_bypassed}\n\n"
            "STABLE SYSTEM CAMERA\n"
            f"  configured policy {view.system_camera.configured_label}\n"
            f"  active input      {view.system_camera.active_label}\n"
            f"  fallback order    "
            f"{' → '.join(view.system_camera.fallback) or 'not described'}\n"
            f"  receiver status   {receiver.get('status', 'unknown')}  output "
            f"{integer(receiver.get('output_frames'))} frames\n"
            f"  public identities {'  '.join(view.virtual_devices)}\n\n"
            "WINDOWS SLOTS\n"
            + "\n".join(
                f"  slot {slot.slot}  {slot.state_text:<30} {slot.identity:<16} "
                f"{slot.endpoint}"
                for slot in view.slots
            )
            + "\n\n"
            "WINDOWS PROCESSOR\n"
            f"  service           "
            f"{windows.get('state', self.windows_error or 'unknown')}\n"
            f"  source picture    "
            f"{'configured' if windows.get('source_configured') else 'missing'}\n"
            f"  loaded model      {view.processor.windows_active_model}\n"
            f"  input frames      {integer(win_input.get('frames'))}  age "
            f"{readable_age(win_input.get('last_frame_age'))}\n"
            f"  return frames     {integer(win_output.get('frames'))}  age "
            f"{readable_age(win_output.get('last_frame_age'))}\n"
            f"  processed         {integer(processing.get('frames'))} frames  "
            f"{float(processing.get('fps') or 0):.1f} fps\n\n"
            "ANDROID NODE\n"
            f"  management        {view.android.state_text}\n"
            f"  summary           {view.android.summary}\n"
            f"  local processor   {native_health.get('state', 'not running')}  "
            f"route {native_health.get('route', 'n/a')}\n"
            f"  identity evidence {view.processor.identity_status} — "
            f"{view.processor.identity_detail}\n\n"
            "ARCH NODE\n"
            f"  sender unit       "
            f"{services.get('sender', {}).get('ActiveState', 'unknown')}  state "
            f"{sender.get('status', 'unknown')}\n"
            f"  receiver unit     "
            f"{services.get('receiver', {}).get('ActiveState', 'unknown')}\n"
            f"  capture owner     {self.capture_camera}\n"
            f"  device mapping    {shadow.get('status', 'not reported')} "
            f"({mapping_state})\n\n"
            "THIS MANAGER (PASSIVE READERS ONLY)\n"
            f"  result relay      udp 127.0.0.1:{self.active_output_preview_port}"
            f"  {result.state_text}  {result.fps:.1f} fps  "
            f"{integer(result.frames)} frames  {integer(result.dropped)} drops\n"
            f"  raw relay         udp 127.0.0.1:{self.preview_port}"
            f"  {raw.state_text}  {raw.fps:.1f} fps  {integer(raw.frames)} frames  "
            f"{integer(raw.dropped)} drops\n"
            f"  comparison delay  {raw.delayed_ms} ms (view only)\n"
            "  opens camera      no\n\n"
            "HOST\n"
            f"  cpu               "
            f"{float(view.host_metrics.get('cpu_percent') or 0):.1f} %   load "
            f"{view.host_metrics.get('load', '?')}\n"
            f"  pipeline rss      "
            f"{float(view.host_metrics.get('rss_mb') or 0):.1f} MiB\n"
            f"  wlan0             TX "
            f"{float(view.host_metrics.get('tx_mbps') or 0):.2f} Mbps  RX "
            f"{float(view.host_metrics.get('rx_mbps') or 0):.2f} Mbps\n\n"
            "ATTENTION\n"
            + (
                "\n".join(
                    f"  [{alert.severity}] {alert.component}: {alert.message}\n"
                    f"      next: {alert.next_action}"
                    for alert in view.alerts
                )
                or "  none"
            )
        )
        self.system_page.set_snapshot_text(text)
        self.snapshot = {
            "timestamp": view.generated_at,
            "config": dict(self.config),
            "sender": sender,
            "receiver": receiver,
            "shadow": shadow,
            "local_services": services,
            "route": {
                "key": view.route.key,
                "badge": view.route.badge,
                "summary": view.route.summary,
                "warning": view.route.warning,
                "windows_bypassed": view.route.windows_bypassed,
            },
            "system_camera": {
                "configured_policy": view.system_camera.configured_policy,
                "active_input": view.system_camera.active_input,
                "fallback": list(view.system_camera.fallback),
                "devices": list(view.virtual_devices),
            },
            "slots": [
                {
                    "slot": slot.slot,
                    "device_id": slot.device_id,
                    "selected": slot.selected,
                    "state": slot.state,
                    "endpoint": slot.endpoint,
                }
                for slot in view.slots
            ],
            "android": self.android_status,
            "android_error": self.android_error,
            "android_native_processor": native_health,
            "windows": self.windows_health,
            "windows_error": self.windows_error,
            "host": dict(view.host_metrics),
            "viewer": {
                "result": {
                    "port": self.active_output_preview_port,
                    "frames": result.frames,
                    "dropped": result.dropped,
                    "age": result.age,
                },
                "raw": {
                    "port": self.preview_port,
                    "frames": raw.frames,
                    "dropped": raw.dropped,
                    "age": raw.age,
                    "comparison_delay_ms": raw.delayed_ms,
                },
            },
            "passive_quality_analysis": {
                "enabled": self.quality_analysis_enabled,
                "raw": dict(self.raw_quality.latest),
                "result": dict(self.result_quality.latest),
            },
            "alerts": [
                {
                    "component": alert.component,
                    "message": alert.message,
                    "next_action": alert.next_action,
                    "severity": alert.severity,
                }
                for alert in view.alerts
            ],
            "opens_camera_device": False,
        }

    # ---------------------------------------------------------------- identity

    def _initialize_desired_source(self) -> None:
        """Restore and queue the durable identity for every processor.

        Re-uploading on manager start is intentional: the legacy Windows API
        reports only that *some* source exists, not its content hash. Sending
        the small cached file is the only way to guarantee that a machine
        which was offline cannot retain a different identity.
        """
        identifier = self.desired_state.get("source_identifier")
        entry = self.source_history.find(str(identifier)) if identifier else None
        if entry is None:
            entry = self.source_history.active_entry()
            if entry is not None:
                self.desired_state = self.desired_store.update(
                    source_identifier=entry.identifier
                )
        elif self.source_history.active_entry() != entry:
            try:
                entry = self.source_history.activate(entry.identifier)
            except (OSError, ValueError):
                entry = None
        if entry is None:
            return
        try:
            self.pending_source = (entry.cache_path.read_bytes(), entry.filename)
        except OSError as exc:
            self.append_log(f"durable source could not be restored: {exc}")

    def _queue_active_source_for_windows_reconnect(self) -> None:
        """Reassert identity after a remote restart, not only scalar settings."""
        if self.pending_source is not None or self.source_upload_inflight is not None:
            return
        entry = self.source_history.active_entry()
        if entry is None:
            return
        try:
            self.pending_source = (entry.cache_path.read_bytes(), entry.filename)
        except OSError as exc:
            self.append_log(f"Windows identity resync could not read cache: {exc}")

    def _windows_source_synchronized(self) -> bool:
        """Return true only when Windows proves the exact desired picture.

        Scalar settings and source bytes have separate requests.  Treating the
        former as sufficient caused the old header to say SYNCED while Windows
        still held a different face after reconnecting.
        """
        desired_identifier = self.desired_state.get("source_identifier")
        if not desired_identifier:
            return self.pending_source is None and self.source_upload_inflight is None
        return bool(
            self.pending_source is None
            and self.source_upload_inflight is None
            and self.windows_config.get("source_identifier") == desired_identifier
        )

    def _reconcile_windows_source_identity(self, document: dict[str, Any]) -> None:
        """Converge a source-aware Windows service without an upload loop."""
        if "source_identifier" not in document:
            # Compatibility with an older service: startup/reconnect still
            # uploads once, but an API that cannot attest identity must not
            # trigger an endless upload/config cycle.
            return
        desired_identifier = self.desired_state.get("source_identifier")
        if not desired_identifier:
            return
        if document.get("source_identifier") != desired_identifier:
            self._queue_active_source_for_windows_reconnect()
            return
        if self.source_upload_inflight is None and self.pending_source is not None:
            pending_identifier = hashlib.sha256(self.pending_source[0]).hexdigest()
            if pending_identifier == desired_identifier:
                self.pending_source = None

    def _reconcile_desired_endpoints(self) -> None:
        """Apply the saved document without reopening any camera endpoint."""
        self._try_upload_source()
        self._apply_local_processor_state()
        self._apply_output_configuration()
        self._apply_android_output_configuration()
        self._reconcile_phone_route()

    def _refresh_source_history(self) -> None:
        entries = self.source_history.entries()
        active = self.source_history.active_entry()
        self.processing_page.rebuild_history(
            entries, active.identifier if active is not None else None
        )
        if active is not None:
            try:
                self.processing_page.show_source_image(
                    active.cache_path.read_bytes(), active.filename
                )
            except OSError:
                self.processing_page.show_source_image(b"", active.filename)

    def select_source_image(self) -> None:
        filename, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose source face",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            data = path.read_bytes()
        except OSError as exc:
            self.processing_page.set_source_status(f"Could not read that image: {exc}")
            return
        self._upload_source(data, path.name)

    def apply_history_picture(self, identifier: str) -> None:
        entry = self.source_history.find(identifier)
        if entry is None:
            self.processing_page.set_source_status(
                "That recent picture is no longer available."
            )
            self._refresh_source_history()
            return
        try:
            data = entry.cache_path.read_bytes()
        except OSError as exc:
            self.processing_page.set_source_status(
                f"Could not read that recent picture: {exc}"
            )
            self._refresh_source_history()
            return
        self._upload_source(data, entry.filename)

    def _upload_source(self, data: bytes, filename: str) -> None:
        page = self.processing_page
        if QImage.fromData(data).isNull():
            page.set_source_status("That file is not a readable picture.")
            return
        try:
            entry = self.source_history.remember(data, filename)
            self.desired_state = self.desired_store.update(
                source_identifier=entry.identifier
            )
        except (OSError, ValueError) as exc:
            page.set_source_status(f"Could not save that picture locally: {exc}")
            return
        self.pending_source = (data, filename)
        self._refresh_source_history()
        page.show_source_image(data, entry.filename)
        page.set_source_status(
            f"{entry.filename} is now the desired identity. Applying it to "
            "available processors…"
        )
        self._try_upload_source()
        self._apply_local_processor_state()

    def _try_upload_source(self) -> None:
        if self.pending_source is None or self.windows.upload_in_flight():
            return
        data, filename = self.pending_source
        refusal = self.windows.upload_source(data, filename)
        if refusal:
            self.processing_page.set_source_status(
                f"Saved locally; Windows sync is pending: {refusal}."
            )
            return
        self.source_upload_inflight = (data, filename)
        self.processing_page.set_source_controls_enabled(False)

    def _retry_source_upload(self) -> None:
        # Health reconnect is the wake-up path for an offline host.  While the
        # host remains reachable, this timer closes the gap where one transient
        # upload failure would otherwise leave a durable picture pending until
        # the next manager restart.
        if self.windows_was_reachable:
            self._try_upload_source()

    def _source_uploaded(self, name: str) -> None:
        self.source_retry_timer.stop()
        completed = self.source_upload_inflight
        self.source_upload_inflight = None
        if completed is not None and self.pending_source == completed:
            self.pending_source = None
        self.windows_config["source_configured"] = True
        if completed is not None:
            self.windows_config["source_identifier"] = hashlib.sha256(
                completed[0]
            ).hexdigest()
        self.processing_page.set_source_status(
            "Identity synchronized to Windows and retained in local history. "
            "The Arch processor uses the same active content-addressed picture."
        )
        self.append_log(f"source picture applied: {name or 'unknown'}")
        self.processing_page.set_source_controls_enabled(True)
        QTimer.singleShot(250, self.windows.request_config)
        if self.pending_source is not None:
            QTimer.singleShot(0, self._try_upload_source)

    def _source_upload_failed(self, detail: str) -> None:
        self.source_upload_inflight = None
        self.processing_page.set_source_status(
            f"Saved locally. Windows identity sync is pending: {detail}"
        )
        self.processing_page.set_source_controls_enabled(True)
        self.append_log(f"Windows source sync deferred: {detail}")
        if self.pending_source is not None and self.windows_was_reachable:
            self.source_retry_timer.start()

    # ------------------------------------------------------ desired topology

    def select_processor(self, processor: str) -> None:
        """Select the active engine while keeping both engines synchronized."""
        try:
            self.desired_state = self.desired_store.set_processor(processor)
        except ValueError as exc:
            self.append_log(str(exc))
            return
        self.desired_initialized = True
        self.processing_page.set_processor(processor)
        self.output_page.set_processor_processing_state(
            self.desired_state, pending=True
        )
        self._set_output_preview_source(local_processed=processor == PROCESSOR_ARCH)
        # Receiver source swaps between queues it already owns; the sink and
        # its public V4L2 identity remain intact.
        self.select_system_camera_policy(
            "local" if processor == PROCESSOR_ARCH else "windows"
        )
        self.pending_output_configuration = True
        self._apply_output_configuration()
        self.pending_windows_changes["processing_mode"] = processor_payload(
            self.desired_state, PROCESSOR_WINDOWS
        )["processing_mode"]
        self.windows_reconcile_requested = True
        self._post_windows_config()
        self._apply_local_processor_state()
        self._reconcile_phone_route()
        specification = PROCESSOR_SPECS[processor]
        self.append_log(
            f"desired processor → {specification.label} "
            f"({specification.model}/{specification.backend})"
        )

    def select_input(self, input_key: str) -> None:
        """Change semantic input without directly opening a camera device."""
        try:
            self.desired_state = self.desired_store.set_input(input_key)
        except ValueError as exc:
            self.append_log(str(exc))
            return
        self.desired_initialized = True
        self.input_page.set_input(input_key, status="switching")
        if input_key in (INPUT_PRERECORDED, INPUT_ASSEMBLER):
            self._sync_camera_form(force_values=True)
            # Assembler reuses the prerecorded transport: if an assembled
            # video is already loaded, resume the relay on selection.
            if self.desired_state.get("prerecorded_path"):
                self._apply_prerecorded_input()
            self._apply_local_processor_state()
            self._reconcile_phone_route()
            self._start_framing_preview()
        else:
            self._stop_framing_preview()
            self._stop_prerecorded_relay()
            self._sync_camera_form(force_values=True)
            self._apply_input_quality_defaults(input_key)
            # When switching between Android front/back cameras, automatically
            # apply the lens_facing change to the Android device
            if input_key in (INPUT_ANDROID_FRONT, INPUT_ANDROID_BACK):
                specification = INPUT_SPECS[input_key]
                if specification.stack == "android-camera2" and specification.lens_facing:
                    # Queue the camera configuration to apply the new lens_facing
                    QTimer.singleShot(100, lambda: self._apply_android_lens_facing(specification.lens_facing))
            self._apply_local_processor_state()
            self._reconcile_phone_route()
        self.append_log(f"desired input → {INPUT_SPECS[input_key].label}")

    def select_prerecorded_source(self, video_path: str) -> None:
        """Use a recorded/rendered video as the active camera input."""
        path = Path(video_path)
        if not path.exists():
            self.append_log(f"prerecorded source not found: {video_path}")
            return
        self.desired_state = self.desired_store.set_prerecorded_path(str(path))
        self.desired_state = self.desired_store.set_input(INPUT_PRERECORDED)
        self.desired_initialized = True
        self.input_page.set_prerecorded_path(str(path))
        self.input_page.set_input(INPUT_PRERECORDED, status="switching")
        self._apply_prerecorded_input()
        self._start_framing_preview()
        self.append_log(f"desired input → {INPUT_SPECS[INPUT_PRERECORDED].label} ({path.name})")

    def set_prerecorded_mode(self, mode: str) -> None:
        """Change playback mode for the prerecorded input."""
        if mode not in ("loop", "once", "freeze"):
            return
        self.desired_state = self.desired_store.set_prerecorded_mode(mode)
        self.desired_initialized = True
        self.input_page.set_prerecorded_mode(mode)
        self.append_log(f"prerecorded mode → {mode}")
        if (
            str(self.desired_state["input"]) == INPUT_PRERECORDED
            and self.prerecorded_relay_process is not None
        ):
            self._apply_prerecorded_input()

    def set_prerecorded_adjust(
        self,
        *,
        offset_x: int | None = None,
        offset_y: int | None = None,
        zoom: float | None = None,
    ) -> None:
        """Update prerecorded video framing. Written to the receiver adjust file."""
        self.desired_state = self.desired_store.set_prerecorded_adjust(
            offset_x=offset_x, offset_y=offset_y, zoom=zoom
        )
        self.desired_initialized = True
        self._write_prerecorded_adjust()

    def _write_prerecorded_adjust(self) -> None:
        """Persist the framing document so the receiver picks it up."""
        adjust = self.desired_state.get("prerecorded_adjust", {})
        path = self.state_dir / "prerecorded-adjust.json"
        try:
            path.write_text(json.dumps(adjust), encoding="utf-8")
        except OSError:
            pass

    def _write_prerecorded_playback(self) -> None:
        """Persist play/pause + seek + mode so the receiver applies it."""
        path = self.state_dir / "prerecorded-playback.json"
        try:
            path.write_text(
                json.dumps(
                    {
                        "paused": bool(self.prerecorded_paused),
                        "seek": self.prerecorded_seek,
                        "mode": str(
                            self.desired_state.get("prerecorded_mode", "loop")
                        ),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def toggle_prerecorded_paused(self) -> None:
        """Play/pause the prerecorded video."""
        self.prerecorded_paused = not self.prerecorded_paused
        self.input_page.set_prerecorded_paused(self.prerecorded_paused)
        self._write_prerecorded_playback()
        self.append_log(
            "prerecorded " + ("paused" if self.prerecorded_paused else "playing")
        )

    def seek_prerecorded(self, seconds: float) -> None:
        """Jump prerecorded playback to a position in seconds."""
        self.prerecorded_seek = max(0.0, float(seconds))
        self._write_prerecorded_playback()
        self.append_log(f"prerecorded seek → {self.prerecorded_seek:.1f}s")

    # --- assembler input -----------------------------------------------------

    def set_assembler_lib(self, path: str) -> None:
        """Pick the puppet segment library directory."""
        self.desired_state = self.desired_store.set_assembler_lib(path)
        self.desired_initialized = True
        self.input_page.set_assembler_lib(path)
        self._sync_camera_form(force_values=True)
        self.append_log(f"assembler library → {path}")

    def set_assembler_tokens(self, tokens: list) -> None:
        """Persist the composed prompt sequence."""
        clean = [str(t).strip() for t in tokens if str(t).strip()]
        self.desired_state = self.desired_store.set_assembler_tokens(clean)
        self.desired_initialized = True

    def assemble_and_load(self, tokens: list) -> None:
        """Run the segment assembler, then switch the camera input to it.

        The assembled video becomes the prerecorded_path, so every downstream
        piece -- file_relay decode, framing/pan/zoom, transport, phone return
        -- behaves exactly as it does for a chosen prerecorded file.
        """
        clean = [str(t).strip() for t in tokens if str(t).strip()]
        if not clean:
            self.input_page.set_assembler_status(
                "Sequence is empty — add at least one prompt token."
            )
            return
        lib = self.desired_state.get("assembler_lib")
        if not lib or not Path(str(lib)).is_dir():
            self.input_page.set_assembler_status(
                "Library directory missing — choose a valid library first."
            )
            return
        if self._assembler_process is not None and \
                self._assembler_process.state() != QProcess.ProcessState.NotRunning:
            self.input_page.set_assembler_status("Assembly already running…")
            return
        self.set_assembler_tokens(clean)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = (
            RENDERS_DIR / f"assembled_{timestamp}.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        self._assembler_output = output
        cmd = [
            sys.executable,
            local_helper("puppet_assemble.py"),
            "--lib", str(lib),
            "-o", str(output),
            *clean,
        ]
        self._assembler_process = QProcess(self)
        self._assembler_process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )
        self._assembler_process.finished.connect(self._assemble_finished)
        self._assembler_process.errorOccurred.connect(self._assemble_error)
        self._assembler_process.start(cmd[0], cmd[1:])
        self.input_page.set_assembler_status(
            f"Assembling {len(clean)} tokens…"
        )
        self.append_log(
            f"assembler: {len(clean)} tokens → {output.name}"
        )

    def _assemble_finished(self, exit_code: int, _status) -> None:
        output = getattr(self, "_assembler_output", None)
        if exit_code != 0:
            self.input_page.set_assembler_status(
                f"Assembly failed (rc={exit_code}) — see log."
            )
            return
        if output is None or not output.exists():
            self.input_page.set_assembler_status("Assembly produced no file.")
            return
        self.desired_state = self.desired_store.set_prerecorded_path(str(output))
        self.desired_state = self.desired_store.set_input(INPUT_ASSEMBLER)
        self.input_page.set_prerecorded_path(str(output))
        self._apply_prerecorded_input()
        self._start_framing_preview()
        self._reconcile_phone_route()
        self.input_page.set_assembler_status(
            f"Loaded {output.name} ({output.stat().st_size / 1e6:.1f} MB)."
        )
        self.append_log(f"assembler loaded → {output.name}")

    def _assemble_error(self, error) -> None:
        self.input_page.set_assembler_status(f"Assembly error: {error}")

    def _start_framing_preview(self) -> None:
        """Show the live framed-video preview on the Input tab."""
        self.input_page.clear_prerecorded_preview("starting framed preview…")
        if not self.framing_decoder.running:
            self.framing_decoder.start()

    def _stop_framing_preview(self) -> None:
        """Stop the framing preview decoder when leaving prerecorded input."""
        if self.framing_decoder.running:
            self.framing_decoder.stop()
        self.input_page.clear_prerecorded_preview("framed preview stopped")

    def _apply_prerecorded_input(self) -> None:
        """Start the prerecorded relay and ask the receiver to prefer it."""
        path = self.desired_state.get("prerecorded_path")
        if not path:
            self.input_page.input_status.setText(
                "No prerecorded video selected. Go to the Render tab and choose a file."
            )
            return
        self._start_prerecorded_relay(str(path), self.desired_state.get("prerecorded_mode", "loop"))
        self._write_prerecorded_adjust()
        # A fresh selection plays from the top, unpaused.
        self.prerecorded_paused = False
        self.prerecorded_seek = None
        self._write_prerecorded_playback()
        self.input_page.set_prerecorded_paused(False)
        self.select_system_camera_policy("prerecorded")

    def _start_prerecorded_relay(self, video_path: str, mode: str = "loop") -> None:
        """Stream the chosen MP4 to the receiver's local_prerecorded port."""
        if self.prerecorded_relay_process is not None:
            self._stop_prerecorded_relay(switch_to_auto=False)
        # Clean up orphan relays left behind by previous manager crashes.  A
        # dying relay's finally-block clears prerecorded-source.txt, which can
        # race the new relay's write and blank the source (receiver then sees
        # no prerecorded frames).  Kill first, wait for the clears to land,
        # THEN author the source path ourselves so the manager's write always
        # wins the race.
        subprocess.run(
            ["pkill", "-f", "prerecorded_relay.py"],
            check=False,
            capture_output=True,
        )
        self.prerecorded_relay_path = video_path
        self.prerecorded_relay_mode = mode
        self.prerecorded_relay_process = QProcess(self)
        self.prerecorded_relay_process.finished.connect(self._prerecorded_relay_finished)
        self.prerecorded_relay_process.errorOccurred.connect(self._prerecorded_relay_error)
        cmd = [
            sys.executable,
            local_helper("prerecorded_relay.py"),
            video_path,
            "--mode", mode,
        ]
        self.prerecorded_relay_process.start(cmd[0], cmd[1:])
        # Author the source path authoritatively after the killed relays have
        # run their finally-blocks, so a late clear cannot blank it.
        QTimer.singleShot(
            400, lambda p=video_path: self._author_prerecorded_source(p)
        )

    def _author_prerecorded_source(self, video_path: str) -> None:
        """Write the prerecorded source path, overriding any stale relay clear."""
        # Only if this is still the intended video (guards against a rapid
        # switch away landing this delayed write onto a different source).
        if self.prerecorded_relay_path != video_path:
            return
        try:
            (self.state_dir / "prerecorded-source.txt").write_text(video_path)
        except OSError:
            pass

    def _stop_prerecorded_relay(self, *, switch_to_auto: bool = True) -> None:
        """Stop the prerecorded relay and optionally let the receiver fall back to auto."""
        # Clear the source file first so the receiver's decoder stops on its
        # next iteration, even if the relay process takes time to die.
        try:
            Path("/run/deep-live-cam/prerecorded-source.txt").write_text("")
        except OSError:
            pass
        if self.prerecorded_relay_process is not None:
            process = self.prerecorded_relay_process
            self.prerecorded_relay_process = None
            self.prerecorded_relay_path = None
            self.prerecorded_relay_mode = None
            if process.state() != QProcess.ProcessState.NotRunning:
                process.terminate()
                QTimer.singleShot(1500, lambda: (
                    process.kill() if process.state() != QProcess.ProcessState.NotRunning else None
                ))
        if switch_to_auto:
            self.select_system_camera_policy("auto")

    def _prerecorded_relay_error(self, error: QProcess.ProcessError) -> None:
        self.append_log(f"prerecorded relay error: {error.name}")
        self.prerecorded_relay_process = None
        self.prerecorded_relay_path = None

    def _prerecorded_relay_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.prerecorded_relay_process is not None:
            self.prerecorded_relay_process = None
            self.prerecorded_relay_path = None

    def _apply_input_quality_defaults(self, input_key: str) -> None:
        """Send a conservative quality profile through the existing owner."""
        specification = INPUT_SPECS[input_key]
        if input_key == INPUT_ARCH_WEBCAM:
            values = profile_live_values(DEFAULT_CAMERA_PROFILE)
            values["profile"] = DEFAULT_CAMERA_PROFILE
        else:
            values = {
                "lens_facing": specification.lens_facing,
                "rotation": "auto",
                # Cover framing already fills the 16:9 transport. Keep the
                # sensor at native field of view by default; extra digital
                # zoom remains a live opt-in control in the Input tab.
                "zoom_percent": 100,
                "exposure_compensation": 0,
                "ae_lock": False,
                "awb_lock": False,
                "stabilization": "video",
            }
        self.input_page.set_camera_values(values)
        persist = input_key != INPUT_ARCH_WEBCAM
        if self.camera_helper.busy():
            # Latest semantic choice wins.  In particular, front -> back while
            # an old front request is running must retain the back lens across
            # an Android bridge restart rather than replay it session-only.
            self.pending_camera_request = (specification.stack, dict(values), persist)
            self.input_page.set_camera_status(
                "The latest input profile is queued behind one in-flight owner "
                "update and will be applied next.",
                state="working",
            )
            return
        self.camera_last_result_ok = None
        self.camera_last_result_detail = ""
        self.camera_persisting = persist
        self.input_page.set_camera_busy(True)
        self.input_page.set_camera_status(
            "Applying the measured low-loss capture profile through the "
            "already-running camera owner…",
            state="working",
        )
        self.camera_helper.run_python(
            local_helper("camera_adapters.py"),
            adapter_arguments(
                specification.stack,
                values,
                serial=self.android_serial,
                host=self.android_host,
                persist=input_key != INPUT_ARCH_WEBCAM,
            ),
        )

    def _desired_windows_device(self) -> str:
        # While Arch owns processing, park Windows on its Arch-returning slot.
        # That keeps the remote node synchronized and reachable without letting
        # it contend for the phone's single processed-return listener.
        if self.desired_state["processor"] == PROCESSOR_ARCH:
            return INPUT_SPECS[INPUT_ARCH_WEBCAM].device_id
        return INPUT_SPECS[str(self.desired_state["input"])].device_id

    def _current_phone_route_signature(self) -> tuple[Any, ...]:
        return route_signature(self.desired_state, self._desired_windows_device())

    def _prerecorded_delivering(self) -> bool:
        """True when the receiver reports the prerecorded input as live.

        The local phone relay reads the file_relay's 11009 tap; readiness is
        that the receiver has locked onto the prerecorded source and its frame
        counter is advancing (active input == local_prerecorded with a recent
        frame), so the relay is enabled only once real frames exist to forward.
        """
        receiver = getattr(self, "_latest_receiver", None)
        if not isinstance(receiver, dict):
            return False
        if receiver.get("active_input") != "local_prerecorded":
            return False
        inputs = receiver.get("inputs")
        entry = inputs.get("local_prerecorded") if isinstance(inputs, dict) else None
        if not isinstance(entry, dict):
            return False
        frames = entry.get("frames")
        age = entry.get("last_frame_age")
        return bool(
            isinstance(frames, (int, float))
            and frames > 0
            and isinstance(age, (int, float))
            and age < 3.0
        )

    def _phone_route_runtime_ready(self, target: str) -> bool:
        if target == RELAY_LOCAL:
            # Prerecorded is produced by the receiver's file_relay, not the
            # native phone processor.  Readiness is that the receiver is
            # actually delivering prerecorded frames to the local relay port,
            # not that the (parked) processor's return transport is open.
            if str(self.desired_state["input"]) in (
                    INPUT_PRERECORDED, INPUT_ASSEMBLER):
                return self._prerecorded_delivering()
            native_return = (
                self._latest_native_health.get("return")
                if isinstance(getattr(self, "_latest_native_health", None), dict)
                else None
            )
            return bool(
                isinstance(native_return, dict)
                and native_return.get("active") is True
                and native_return.get("transport_open") is True
                and windows_runtime_ready(
                    self.windows_devices, self._desired_windows_device()
                )
            )
        if (
            target == RELAY_OFF
            and not self.desired_state["outputs"][OUTPUT_ANDROID_PHONE]
        ):
            return True
        return windows_runtime_ready(
            self.windows_devices,
            self._desired_windows_device(),
            require_selected_stream=target == RELAY_WINDOWS,
        )

    def _run_phone_relay_control(
        self, source: str, signature: tuple[Any, ...]
    ) -> bool:
        if self.phone_relay_helper.busy():
            return False
        started = self.phone_relay_helper.run_python(
            local_helper("configure_phone_relay.py"),
            ["--source", source, "--socket", str(self.phone_relay_socket)],
        )
        if started:
            self.phone_relay_inflight_source = source
            self.phone_relay_inflight_signature = signature
        return started

    def _reconcile_phone_route(self) -> None:
        """Serialize every owner of Android's one processed-return listener.

        A topology change always closes the Arch relay first and waits for the
        synchronous acknowledgement.  Only then may Windows change slots.  The
        new relay source is enabled only after the Windows API reports the live
        router (not merely its persisted registry) on the requested device.
        """
        signature = self._current_phone_route_signature()
        target = desired_relay_source(self.desired_state)
        first_observation = self.phone_route_signature is None
        if signature != self.phone_route_signature:
            self.phone_route_signature = signature
            self.phone_route_quiesced_signature = None
            self.phone_route_applied_signature = None

        if self.phone_relay_helper.busy():
            return

        # A previously verified local route may keep working through a brief
        # management outage.  It is never established from an unverified
        # offline state. If Windows later reports the wrong live slot, close
        # the relay again before repair so its direct return cannot overlap.
        windows_route_ready = windows_runtime_ready(
            self.windows_devices, self._desired_windows_device()
        )
        if (
            self.registry_live
            and not windows_route_ready
            and not relay_is_closed(self.phone_relay_health)
        ):
            self.phone_route_quiesced_signature = None

        # On manager restart, an already-persisted correct intent may continue
        # without an artificial cut.  It is adopted only after the exact live
        # Windows runtime route is verified; an unknown/mismatched route is
        # closed before any attempt to repair it.
        if (
            first_observation
            and relay_desires(self.phone_relay_health, target)
            and (
                target != RELAY_OFF
                or relay_is_closed(self.phone_relay_health)
            )
            and self._phone_route_runtime_ready(target)
        ):
            self.phone_route_quiesced_signature = signature
            self.phone_route_applied_signature = signature
            return

        if self.phone_route_quiesced_signature != signature:
            if relay_is_closed(self.phone_relay_health):
                self.phone_route_quiesced_signature = signature
            else:
                self._run_phone_relay_control(RELAY_OFF, signature)
                return

        self._reconcile_windows_input()
        if (
            target != RELAY_OFF
            and self.registry_live
            and not windows_runtime_ready(
                self.windows_devices, self._desired_windows_device()
            )
        ):
            return
        if not self._phone_route_runtime_ready(target):
            return

        if target == RELAY_OFF:
            self.phone_route_applied_signature = signature
            return
        if relay_desires(self.phone_relay_health, target):
            self.phone_route_applied_signature = signature
            return
        self._run_phone_relay_control(target, signature)

    def _reconcile_windows_input(self) -> None:
        signature = self._current_phone_route_signature()
        if self.phone_route_quiesced_signature != signature:
            return
        desired_device = self._desired_windows_device()
        if not self.registry_live or self.windows.selection_in_flight():
            return
        if windows_runtime_ready(self.windows_devices, desired_device):
            return
        self.select_windows_slot(desired_device)

    def _phone_relay_finished(self, ok: bool, output: str) -> None:
        source = self.phone_relay_inflight_source
        signature = self.phone_relay_inflight_signature
        self.phone_relay_inflight_source = None
        self.phone_relay_inflight_signature = None
        document: dict[str, Any] | None = None
        try:
            value = json.loads(output) if output else None
            if isinstance(value, dict):
                document = value
        except (json.JSONDecodeError, TypeError, ValueError):
            document = None
        if ok and document is not None:
            self.phone_relay_health = document
            if source == RELAY_OFF and relay_is_closed(document):
                if signature == self._current_phone_route_signature():
                    self.phone_route_quiesced_signature = signature
                self.append_log("phone-return relay closed before route change")
            elif source and relay_desires(document, source):
                if signature == self._current_phone_route_signature():
                    self.phone_route_applied_signature = signature
                self.append_log(f"phone-return relay source → {source}")
        else:
            self.append_log(
                "phone-return relay sync pending: "
                + (output or "control socket unavailable")
            )
        QTimer.singleShot(0 if ok else 1200, self._reconcile_phone_route)

    def _phone_relay_unavailable(self, message: str) -> None:
        self.phone_relay_inflight_source = None
        self.phone_relay_inflight_signature = None
        self.append_log(f"phone-return relay control unavailable: {message}")
        QTimer.singleShot(1200, self._reconcile_phone_route)

    def set_output_enabled(self, output: str, enabled: bool) -> None:
        try:
            self.desired_state = self.desired_store.set_output(output, enabled)
        except (TypeError, ValueError) as exc:
            self.append_log(str(exc))
            return
        self.desired_initialized = True
        self.output_page.set_output_state(self.desired_state, pending=True)
        if output == OUTPUT_ARCH_CAMERA:
            self.pending_output_configuration = True
            self._apply_output_configuration()
        else:
            self.pending_android_output_configuration = True
            self._apply_android_output_configuration()
            self._reconcile_phone_route()

    def set_processor_processing_enabled(
        self, processor: str, enabled: bool
    ) -> None:
        """Atomically choose a host/mode, then reconcile every live endpoint."""
        previous_processor = str(self.desired_state["processor"])
        try:
            self.desired_state = self.desired_store.set_processor_processing(
                str(processor), bool(enabled)
            )
        except (TypeError, ValueError) as exc:
            self.append_log(str(exc))
            self.output_page.set_output_state(self.desired_state)
            return
        current_processor = str(self.desired_state["processor"])
        mode = str(self.desired_state["processing"]["processing_mode"])
        windows_mode = str(
            processor_payload(self.desired_state, PROCESSOR_WINDOWS)[
                "processing_mode"
            ]
        )
        self.desired_initialized = True
        self.processing_page.set_preset_index(0)
        self.processing_page.set_processor(current_processor)
        self.processing_page.apply_windows_config(
            dict(self.desired_state["processing"]), reset_preset=False
        )
        self.output_page.set_output_state(self.desired_state, pending=True)

        if previous_processor != current_processor:
            self._set_output_preview_source(
                local_processed=current_processor == PROCESSOR_ARCH
            )
            self.select_system_camera_policy(
                "local" if current_processor == PROCESSOR_ARCH else "windows"
            )
            self.pending_output_configuration = True
            self._apply_output_configuration()

        self.pending_windows_changes["processing_mode"] = windows_mode
        self.windows_reconcile_requested = True
        self._post_windows_config()
        self._apply_local_processor_state()
        self._reconcile_phone_route()
        target = PROCESSOR_SPECS[current_processor].label
        self.append_log(
            f"{target} face swapping → "
            f"{'enabled' if mode == 'face_swap' else 'passthrough'}"
        )

    def set_output_transform(self, mirror: bool, rotation: int) -> None:
        try:
            self.desired_state = self.desired_store.set_transform(
                mirror=mirror, rotation=rotation
            )
        except (TypeError, ValueError) as exc:
            self.append_log(str(exc))
            return
        self.desired_initialized = True
        self.output_page.set_output_state(self.desired_state, pending=True)
        self.pending_output_configuration = True
        self.pending_android_output_configuration = True
        self._apply_output_configuration()
        self._apply_android_output_configuration()

    def _desired_receiver_source(self) -> str:
        """The receiver source-mode the desired state implies.

        Prerecorded input takes precedence: its receiver priority excludes the
        live phone/webcam queues so a chosen video is not immediately preempted
        by the still-running Android processor.  Otherwise the processor choice
        selects the local (Arch) or Windows processed queue.
        """
        if self.desired_state["input"] in (INPUT_PRERECORDED, INPUT_ASSEMBLER):
            return "prerecorded"
        return (
            "local"
            if self.desired_state["processor"] == PROCESSOR_ARCH
            else "windows"
        )

    def _apply_output_configuration(self) -> None:
        if self.output_helper.busy():
            self.pending_output_configuration = True
            return
        transform = self.desired_state["output_transform"]
        source = self._desired_receiver_source()
        arguments = [
            "--mirror",
            "true" if transform["mirror"] else "false",
            "--rotation",
            str(transform["rotation"]),
            "--enabled",
            "true"
            if self.desired_state["outputs"][OUTPUT_ARCH_CAMERA]
            else "false",
            "--source",
            source,
            "--socket",
            str(self.state_dir / "receiver-control.sock"),
        ]
        if self.output_helper.run_python(
            local_helper("configure_receiver_output.py"), arguments
        ):
            self.output_configuration_inflight = True
            self.output_configuration_inflight_state = self._arch_output_signature()

    def _arch_output_signature(self) -> tuple[str, bool, bool, int]:
        transform = self.desired_state["output_transform"]
        return (
            self._desired_receiver_source(),
            bool(self.desired_state["outputs"][OUTPUT_ARCH_CAMERA]),
            bool(transform["mirror"]),
            int(transform["rotation"]),
        )

    @staticmethod
    def _arch_output_matches(
        document: object, desired: tuple[str, bool, bool, int]
    ) -> bool:
        if not isinstance(document, dict):
            return False
        transform = document.get("output_transform")
        if not isinstance(transform, dict):
            return False
        source, enabled, mirror, rotation = desired
        return (
            document.get("virtual_camera") == "/dev/deep-live-cam"
            and document.get("virtual_cameras") == ["/dev/deep-live-cam"]
            and document.get("source_mode", document.get("source")) == source
            and document.get("output_enabled") is enabled
            and transform.get("mirror") is mirror
            and transform.get("rotation") == rotation
        )

    def _output_configuration_finished(self, ok: bool, output: str) -> None:
        applied = self.output_configuration_inflight_state
        self.output_configuration_inflight_state = None
        self.output_configuration_inflight = False
        current = self._arch_output_signature()
        document: dict[str, Any] | None = None
        try:
            parsed = json.loads(output) if output else None
            if isinstance(parsed, dict):
                document = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            document = None
        effective = bool(
            ok
            and applied == current
            and self._arch_output_matches(document, current)
        )
        self.pending_output_configuration = not effective
        if effective:
            self.append_log("Arch output changed in place; sink session preserved")
        else:
            self.append_log(
                "Arch output change remains pending: the live receiver has not "
                "confirmed the newest desired source and transform"
            )
        self.output_page.set_output_state(
            self.desired_state, pending=self.pending_output_configuration
        )
        if self.pending_output_configuration:
            QTimer.singleShot(1000, self._apply_output_configuration)

    def _output_configuration_unavailable(self, message: str) -> None:
        self.output_configuration_inflight_state = None
        self.output_configuration_inflight = False
        self.pending_output_configuration = True
        self.append_log(f"Arch output helper unavailable: {message}")
        QTimer.singleShot(1000, self._apply_output_configuration)

    def _apply_android_output_configuration(self) -> None:
        """Apply when the deployed Android bridge exposes the hot contract.

        The helper is intentionally capability-detected so an older phone
        bridge keeps streaming while the desired state remains pending.
        """
        helper = local_helper("android_bridge.py")
        if self.android_output_helper.busy():
            self.pending_android_output_configuration = True
            return
        helper_path = Path(helper)
        if not helper_path.is_file():
            self.pending_android_output_configuration = True
            return
        if not self._android_output_supported(self.android_status):
            self.pending_android_output_configuration = True
            return
        if not (
            self.android_status.get("module_enabled")
            and self.android_status.get("output_selector_running")
        ):
            # A disabled/dead module is status-only.  Persisting into its root
            # files cannot revive it and makes an offline component look as if
            # it accepted a live change.
            self.pending_android_output_configuration = True
            return
        desired_signature = self._android_output_signature()
        if self._android_output_configuration_matches(
            self.android_status, desired_signature
        ):
            # The desired file is already durable.  If the selector/provider
            # is down, health polling—not repeated ADB writes—is what can prove
            # recovery.  Keep the UI pending until the live selector catches up.
            self.pending_android_output_configuration = not self._android_output_applied(
                self.android_status, desired_signature
            )
            return
        # The Android helper extension uses this command name. QProcess keeps
        # it asynchronous and no adb operation runs on the UI thread.
        transform = self.desired_state["output_transform"]
        arguments = [
            "configure-output",
            "--host",
            self.android_host,
            "--output-enabled",
            "true" if self.desired_state["outputs"][OUTPUT_ANDROID_PHONE] else "false",
            "--output-mirror",
            "true" if transform["mirror"] else "false",
            "--output-rotation",
            str(transform["rotation"]),
        ]
        if self.android_serial:
            arguments.extend(("--serial", self.android_serial))
        if self.android_output_helper.run_python(helper, arguments):
            self.android_output_inflight_state = desired_signature
            self.pending_android_output_configuration = False

    def _android_output_signature(self) -> tuple[bool, bool, int]:
        transform = self.desired_state["output_transform"]
        return (
            bool(self.desired_state["outputs"][OUTPUT_ANDROID_PHONE]),
            bool(transform["mirror"]),
            int(transform["rotation"]),
        )

    @staticmethod
    def _android_output_supported(document: object) -> bool:
        if (
            not isinstance(document, dict)
            or not document.get("available")
            or not document.get("module_installed")
        ):
            return False
        control = document.get("output_control")
        if isinstance(control, dict) and isinstance(control.get("supported"), bool):
            return bool(control["supported"])
        version = str(document.get("module_version") or "")
        match = re.search(r"(?:^|[^0-9])(\d+)\.(\d+)\.(\d+)(?:$|[^0-9])", version)
        if match is None:
            return False
        return tuple(int(part) for part in match.groups()) >= (0, 4, 6)

    @classmethod
    def _android_output_configuration_matches(
        cls, document: object, desired: tuple[bool, bool, int]
    ) -> bool:
        if not cls._android_output_supported(document):
            return False
        assert isinstance(document, dict)
        control = document.get("output_control")
        if not isinstance(control, dict) or not control.get("persisted"):
            return False
        return (
            control.get("enabled") is desired[0]
            and control.get("mirror") is desired[1]
            and control.get("rotation") == desired[2]
        )

    @classmethod
    def _android_output_applied(
        cls, document: object, desired: tuple[bool, bool, int]
    ) -> bool:
        if not cls._android_output_configuration_matches(document, desired):
            return False
        assert isinstance(document, dict)
        control = document.get("output_control")
        assert isinstance(control, dict)
        return bool(
            document.get("module_enabled")
            and document.get("output_selector_running")
            and control.get("applied")
        )

    @classmethod
    def _android_output_delivery_matches(
        cls, document: object, desired: tuple[bool, bool, int]
    ) -> bool:
        """True only when the requested media state is actually published."""
        if not cls._android_output_applied(document, desired):
            return False
        assert isinstance(document, dict)
        control = document.get("output_control")
        assert isinstance(control, dict)
        enabled = desired[0]
        if not document.get("camera_published"):
            return False
        if not enabled:
            # Disabled delivery intentionally keeps Camera2 alive on its stable
            # placeholder; no decoder worker is expected in that state.
            return control.get("effective_source") == "placeholder"
        return bool(
            control.get("effective_source") == "processed"
            and control.get("effective_worker_alive")
        )

    def _android_output_configuration_finished(self, ok: bool, output: str) -> None:
        applied = self.android_output_inflight_state
        self.android_output_inflight_state = None
        changed_while_applying = applied != self._android_output_signature()
        document: dict[str, Any] | None = None
        try:
            parsed = json.loads(output) if output else None
            if isinstance(parsed, dict):
                document = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            document = None
        effective = bool(
            ok
            and not changed_while_applying
            and self._android_output_applied(document, self._android_output_signature())
        )
        if document is not None:
            self.android_status = document
        self.pending_android_output_configuration = bool(
            not effective
        )
        if effective:
            self.append_log("Android output changed in place; Camera2 identity preserved")
        elif not ok:
            self.append_log(
                "Android output change failed and remains pending: "
                f"{output or 'helper reported a failure'}"
            )
        elif self._android_output_configuration_matches(
            document, self._android_output_signature()
        ):
            self.append_log(
                "Android output settings were saved; live selector confirmation "
                "is still pending"
            )
        else:
            self.append_log(
                "Android output response did not confirm the newest persisted "
                "settings; synchronization remains pending"
            )
        if self.pending_android_output_configuration:
            self.android_output_retry_timer.start()

    def _android_output_configuration_unavailable(self, message: str) -> None:
        self.android_output_inflight_state = None
        self.pending_android_output_configuration = True
        self.append_log(f"Android output helper unavailable: {message}")
        self.android_output_retry_timer.start()

    def _apply_local_processor_state(self) -> None:
        if self.local_processor_helper.busy():
            self.pending_local_processor_state = True
            return
        entry = self.source_history.active_entry()
        processing_payload = local_processor_payload(
            self.desired_state,
            self._latest_native_health,
        )
        arguments = [
            "--socket",
            str(self.state_dir / "processor-control.sock"),
            "--active",
            "true"
            if self.desired_state["processor"] == PROCESSOR_ARCH
            else "false",
            "--input",
            str(self.desired_state["input"]),
            "--processing-json",
            json.dumps(
                processing_payload,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "--revision",
            str(self.desired_state["revision"]),
        ]
        # The Arch target is fixed, even while it is the inactive standby.
        # Activation ownership and model configuration are separate controls.
        arguments.append("--activate-native256")
        if entry is not None:
            arguments.extend(("--source-path", str(entry.cache_path)))
        if self.local_processor_helper.run_python(
            local_helper("local_processor_control.py"), arguments
        ):
            self.pending_local_processor_state = False
            self.local_processor_inflight_revision = int(
                self.desired_state["revision"]
            )

    def _local_processor_finished(self, ok: bool, output: str) -> None:
        inflight_revision = self.local_processor_inflight_revision
        self.local_processor_inflight_revision = None
        document: dict[str, Any] | None = None
        try:
            value = json.loads(output) if output else None
            if isinstance(value, dict):
                document = value
        except (json.JSONDecodeError, TypeError, ValueError):
            document = None
        response_revision = document.get("revision") if document else None
        current_revision = int(self.desired_state["revision"])
        request_is_current = inflight_revision == current_revision
        response_matches_request = bool(
            inflight_revision is not None and response_revision == inflight_revision
        )
        synchronized = bool(
            ok
            and document is not None
            and document.get("ok") is True
            and document.get("in_sync") is True
            and request_is_current
            and response_matches_request
        )
        retry_immediately = False
        if synchronized:
            self.pending_local_processor_state = False
            self.append_log("Arch processor desired state synchronized")
        elif not request_is_current or (
            ok and document is not None and not response_matches_request
        ):
            # The UI changed while the helper was in flight. The stale reply
            # may have activated Arch, so immediately send the newest document
            # even when the newest owner is Windows. An online worker returning
            # a different revision gets the same treatment.
            self.pending_local_processor_state = True
            retry_immediately = True
            self.append_log("Ignoring stale Arch processor acknowledgement")
        elif self.desired_state["processor"] == PROCESSOR_ARCH:
            self.pending_local_processor_state = True
            self.append_log(
                f"Arch processor sync pending: {output or 'control socket unavailable'}"
            )
        else:
            # Failure to reach an already-inactive standby is a safe OFF. Do
            # not create an endless 1.5 second retry loop while Windows owns
            # processing; a service-health transition will reconcile it later.
            self.pending_local_processor_state = False
            self.append_log("Arch standby is offline; it remains safely off")
        self._reconcile_phone_route()
        if self.pending_local_processor_state:
            QTimer.singleShot(
                0 if retry_immediately else 1500,
                self._apply_local_processor_state,
            )

    def _local_processor_unavailable(self, message: str) -> None:
        self.local_processor_inflight_revision = None
        if self.desired_state["processor"] == PROCESSOR_ARCH:
            self.pending_local_processor_state = True
            self.append_log(f"Arch processor control unavailable: {message}")
            QTimer.singleShot(1500, self._apply_local_processor_state)
        else:
            self.pending_local_processor_state = False
            self.append_log("Arch standby is offline; it remains safely off")

    # ----------------------------------------------------------- windows config

    def windows_setting_changed(self, field: str, value: Any) -> None:
        try:
            self.desired_state = self.desired_store.set_processing(str(field), value)
        except (TypeError, ValueError) as exc:
            self.append_log(f"ignored invalid processing setting {field}: {exc}")
            return
        canonical = self.desired_state["processing"][str(field)]
        self.pending_windows_changes[str(field)] = processor_payload(
            self.desired_state, PROCESSOR_WINDOWS
        )[str(field)]
        self.desired_initialized = True
        self.windows_reconcile_requested = True
        self.processing_page.apply_windows_config(
            dict(self.desired_state["processing"]), reset_preset=False
        )
        if str(field) == "processing_mode":
            self.output_page.set_processor_processing_state(self.desired_state)
        self._post_windows_config()
        self._apply_local_processor_state()

    def apply_processing_preset(self, index: int) -> None:
        values = PRESETS.get(int(index))
        if values is None:
            return
        self.desired_state = self.desired_store.set_processing_values(values)
        self.desired_initialized = True
        effective = dict(self.desired_state["processing"])
        self.processing_page.apply_windows_config(effective, reset_preset=False)
        self.processing_page.set_preset_index(int(index))
        self.output_page.set_processor_processing_state(self.desired_state)
        windows_payload = processor_payload(self.desired_state, PROCESSOR_WINDOWS)
        self.pending_windows_changes.update(
            {field: windows_payload[field] for field in values}
        )
        self.windows_reconcile_requested = True
        self.append_log(f"applying processing preset {index}")
        self._post_windows_config()
        self._apply_local_processor_state()

    def _post_windows_config(self) -> None:
        if not self.pending_windows_changes or self.windows_inflight_changes:
            return
        payload = dict(self.pending_windows_changes)
        if self.windows.apply_config(payload):
            self.windows_inflight_changes = payload

    def _windows_config_applied(self, effective: dict[str, Any]) -> None:
        sent, self.windows_inflight_changes = self.windows_inflight_changes, {}
        self.windows_config = dict(effective)
        self.windows_confirmed_config = dict(effective)
        desired = processor_payload(self.desired_state, PROCESSOR_WINDOWS)
        for key, sent_value in sent.items():
            if desired.get(key) == sent_value:
                self.pending_windows_changes.pop(key, None)
        # The canonical response can reveal clamping or a deployment which
        # ignored a supported key; reconcile again instead of regressing UI.
        self.pending_windows_changes.update(
            reconciliation_payload(self.desired_state, effective)
        )
        self.windows_reconcile_requested = bool(self.pending_windows_changes)
        self.append_log(f"Windows settings updated: {', '.join(sorted(sent))}")
        if self.pending_windows_changes:
            self.windows_config_timer.start(350)

    def _windows_config_rejected(self, payload: dict[str, Any], error: str) -> None:
        self.windows_inflight_changes = {}
        self.pending_windows_changes.update(payload)
        self.windows_reconcile_requested = True
        self.append_log(f"Windows settings update failed: {error}")
        self.windows_config_timer.start(1200)

    def _windows_config_received(self, document: dict[str, Any]) -> None:
        self.windows_config = dict(document)
        self.windows_confirmed_config = dict(document)
        self._reconcile_windows_source_identity(document)
        if not self.desired_initialized:
            first_values = {
                key: document[key]
                for key in self.desired_state["processing"]
                if key in document
            }
            if first_values:
                self.desired_state = self.desired_store.set_processing_values(
                    first_values
                )
            self.desired_initialized = True
            self.processing_page.apply_windows_config(
                dict(self.desired_state["processing"])
            )
        differences = reconciliation_payload(self.desired_state, document)
        self.pending_windows_changes.update(differences)
        self.windows_reconcile_requested = bool(differences)
        self.append_log(
            "Windows processing settings synchronized"
            if not differences
            else f"Windows reconnected; reconciling {len(differences)} desired settings"
        )
        if differences:
            self._post_windows_config()
        self._try_upload_source()

    def _windows_config_failed(self, error: str) -> None:
        self.windows_reconcile_requested = True
        self.append_log(f"Windows settings unavailable: {error}")

    def _windows_health_received(self, document: Any) -> None:
        self.windows_health = document if isinstance(document, dict) else None
        self.windows_error = None if isinstance(document, dict) else "invalid health"
        reachable = isinstance(document, dict)
        if reachable and not self.windows_was_reachable:
            self.windows_reconcile_requested = True
            self.windows.request_config()
            self.windows.request_devices()
            self._queue_active_source_for_windows_reconnect()
            self._try_upload_source()
        self.windows_was_reachable = reachable

    def _windows_health_failed(self, error: str) -> None:
        self.windows_health = None
        self.windows_error = error
        self.windows_was_reachable = False
        self.windows_reconcile_requested = True

    # ---------------------------------------------------------------- slots

    def _windows_devices_received(self, document: dict[str, Any]) -> None:
        self.windows_devices = document
        self.registry_live = True
        self._reconcile_phone_route()

    def _windows_devices_failed(self, error: str) -> None:
        if not self.windows_devices.get("slots"):
            self._use_offline_device_registry()
        self.registry_live = False
        self.windows_error = self.windows_error or error

    def _use_offline_device_registry(self) -> None:
        self.windows_devices = offline_device_registry(
            selected_device_id=self.config.get("DEVICE_ID", "arch-webcam"),
            android_host=self.android_host,
            arch_host=self.arch_host,
            android_enabled=self.android_enabled,
        )
        self.registry_live = False

    def select_windows_slot(self, device_id: str) -> None:
        if not self.registry_live:
            self.routing_page.slot_contract.setText(
                "Windows route selection is offline, so the processed slot "
                "cannot be changed from here. Camera-owner controls below stay "
                "available."
            )
            return
        if windows_runtime_ready(self.windows_devices, device_id):
            return
        if self.windows.select_device(device_id):
            self.routing_page.slot_contract.setText(
                f"Switching the Windows input to {device_id}. Camera owners, "
                "the virtual-camera sink, and the operating-system camera "
                "identities are not restarted."
            )
            self.append_log(f"requesting Windows input → {device_id}")

    def _windows_selection_succeeded(self, document: dict[str, Any]) -> None:
        self.windows_devices = document
        self.registry_live = True
        self.append_log(
            f"Windows selected input → {document.get('selected_device_id')}"
        )
        self._reconcile_phone_route()

    def _windows_selection_failed(self, error: str) -> None:
        self.registry_live = False
        self.routing_page.slot_contract.setText(
            f"The slot selection was not applied: {error}. The previous route "
            "is still in place."
        )
        self.append_log(f"Windows slot selection failed: {error}")

    # ------------------------------------------------------- system camera

    def select_system_camera_policy(self, policy: str) -> None:
        if self.policy_helper.busy():
            return
        started = self.policy_helper.run_python(
            local_helper("select_receiver_source.py"),
            [policy, "--socket", str(self.state_dir / "receiver-control.sock")],
        )
        if started:
            self.pending_policy = policy
            self.routing_page.set_policy_enabled(False)
            self.routing_page.set_policy_result(
                f"Selecting {policy}: swapping already-owned frame queues…",
                success=True,
            )

    def _policy_finished(self, ok: bool, output: str) -> None:
        policy = self.pending_policy or "the requested policy"
        self.pending_policy = ""
        self.routing_page.set_policy_enabled(True)
        if ok:
            devices = " and ".join(str(device) for device in self.virtual_devices)
            self.routing_page.set_policy_result(
                f"Policy {policy} is now selected. The virtual-camera sink, "
                f"{devices or 'the camera nodes'}, and the capture owner were "
                "not restarted, and no application lost its camera.",
                success=True,
            )
        else:
            self.routing_page.set_policy_result(
                "The policy was not changed and the existing camera stream is "
                f"untouched. {output or 'the helper reported a failure'}",
                success=False,
            )
        if output:
            self.append_log(f"system-camera selector: {output}")

    def _policy_unavailable(self, message: str) -> None:
        self.pending_policy = ""
        self.routing_page.set_policy_enabled(True)
        self.routing_page.set_policy_result(
            "The system-camera selector could not start; the current stream "
            f"remains active. {message}",
            success=False,
        )
        self.append_log(message)

    # ------------------------------------------------------- capture owner

    def _camera_defaults(self) -> dict[str, Any]:
        return {
            "profile": self.config.get("CAMERA_PROFILE", CUSTOM_CAMERA_PROFILE),
            "capture_size": (
                f"{self.config.get('CAMERA_WIDTH', '1280')}x"
                f"{self.config.get('CAMERA_HEIGHT', '720')}"
            ),
            "brightness": self._config_int("CAMERA_BRIGHTNESS", 0),
            "contrast": self._config_int("CAMERA_CONTRAST", 32),
            "saturation": self._config_int("CAMERA_SATURATION", 64),
            "hue": self._config_int("CAMERA_HUE", 0),
            "gamma": self._config_int("CAMERA_GAMMA", 100),
            "gain": self._config_int("CAMERA_GAIN", 0),
            "sharpness": self._config_int("CAMERA_SHARPNESS", 3),
            "backlight_compensation": self._config_int("CAMERA_BACKLIGHT", 1),
            "power_line_frequency": self._config_int("CAMERA_POWER_LINE", 1),
            "auto_exposure": self._config_int("CAMERA_AUTO_EXPOSURE", 1) != 0,
            "exposure_time_absolute": self._config_int("CAMERA_EXPOSURE", 157),
            "exposure_dynamic_framerate": (
                self._config_int("CAMERA_EXPOSURE_DYNAMIC_FRAMERATE", 0) != 0
            ),
            "auto_white_balance": self._config_int("CAMERA_AUTO_WHITE_BALANCE", 1) != 0,
            "white_balance_temperature": self._config_int("CAMERA_WHITE_BALANCE", 4600),
        }

    def _selected_slot_document(self) -> dict[str, Any]:
        selected = self.windows_devices.get("selected_device_id")
        for slot in self.windows_devices.get("slots", []):
            if isinstance(slot, dict) and slot.get("device_id") == selected:
                return slot
        return {}

    def _sync_camera_form(self, *, force_values: bool = False) -> None:
        selected_input = str(self.desired_state.get("input", INPUT_ANDROID_FRONT))
        if selected_input == INPUT_PRERECORDED:
            if self.input_page.adapter_key() == ("local-prerecorded", "prerecorded-relay") and not force_values:
                return
            self.camera_live_timer.stop()
            self.input_page.rebuild_prerecorded_controls(
                path=self.desired_state.get("prerecorded_path"),
                mode=self.desired_state.get("prerecorded_mode", "loop"),
                offset_x=int(
                    self.desired_state.get("prerecorded_adjust", {}).get("offset_x", 0)
                ),
                offset_y=int(
                    self.desired_state.get("prerecorded_adjust", {}).get("offset_y", 0)
                ),
                zoom=float(
                    self.desired_state.get("prerecorded_adjust", {}).get("zoom", 1.0)
                ),
            )
            return
        if selected_input == INPUT_ASSEMBLER:
            if self.input_page.adapter_key() == ("local-prerecorded", "prerecorded-relay") and not force_values:
                return
            self.camera_live_timer.stop()
            self.input_page.rebuild_assembler_controls(
                lib=self.desired_state.get("assembler_lib"),
                tokens=list(self.desired_state.get("assembler_tokens", [])),
                offset_x=int(
                    self.desired_state.get("prerecorded_adjust", {}).get("offset_x", 0)
                ),
                offset_y=int(
                    self.desired_state.get("prerecorded_adjust", {}).get("offset_y", 0)
                ),
                zoom=float(
                    self.desired_state.get("prerecorded_adjust", {}).get("zoom", 1.0)
                ),
            )
            return
        specification = INPUT_SPECS[selected_input]
        device_id = specification.device_id
        stack = specification.stack
        if self.input_page.adapter_key() == (device_id, stack) and not force_values:
            if specification.lens_facing:
                self.input_page.set_camera_values(
                    {"lens_facing": specification.lens_facing}
                )
            return
        self.camera_live_timer.stop()
        defaults = self._camera_defaults()
        if stack == "android-camera2":
            reported = self.android_status.get("capture_metrics")
            if not isinstance(reported, dict):
                reported = self.android_status.get("controls")
            if isinstance(reported, dict):
                defaults.update(reported)
            defaults.update(
                {
                    "lens_facing": specification.lens_facing,
                    "rotation": defaults.get("rotation") or "auto",
                    "stabilization": defaults.get("stabilization") or "video",
                }
            )
        self.input_page.rebuild_camera_controls(
            device_id=device_id,
            label=specification.label,
            stack=stack,
            schema=camera_schema(stack),
            defaults=defaults,
        )

    def camera_control_changed(self, _key: str) -> None:
        self.camera_live_timer.start()

    def _preview_camera_configuration(self) -> None:
        self._start_camera_configuration(persist=False)

    def save_camera_configuration(self) -> None:
        self.camera_live_timer.stop()
        self._start_camera_configuration(persist=True)

    def _start_camera_configuration(self, *, persist: bool) -> None:
        page = self.routing_page
        adapter = page.adapter_key()
        if adapter is None:
            return
        values = page.camera_values()
        if not values:
            return
        _device_id, stack = adapter
        if self.camera_helper.busy():
            self.pending_camera_request = (stack, dict(values), persist)
            page.set_camera_status(
                "Save queued behind the current owner update."
                if persist
                else "Latest live preview queued behind the current owner update.",
                state="working",
            )
            return
        self._launch_camera_configuration(stack, values, persist=persist)

    def _launch_camera_configuration(
        self, stack: str, values: dict[str, Any], *, persist: bool
    ) -> None:
        page = self.routing_page
        self.camera_last_result_ok = None
        self.camera_last_result_detail = ""
        self.camera_persisting = persist
        page.set_camera_busy(True)
        page.set_camera_status(
            "Saving the current camera settings through the capture owner…"
            if persist
            else "Applying an unsaved preview through the capture owner…",
            state="working",
        )
        if persist and stack == "arch-v4l2":
            self.camera_helper.run("pkexec", arch_persist_arguments(values))
            return
        self.camera_helper.run_python(
            local_helper("camera_adapters.py"),
            adapter_arguments(
                stack,
                values,
                serial=self.android_serial,
                host=self.android_host,
                persist=persist,
            ),
        )

    def _replay_pending_camera_request(self) -> None:
        if self.camera_helper.busy() or self.pending_camera_request is None:
            return
        stack, values, persist = self.pending_camera_request
        self.pending_camera_request = None
        self._launch_camera_configuration(stack, values, persist=persist)

    def _camera_finished(self, ok: bool, output: str) -> None:
        page = self.routing_page
        persisted = self.camera_persisting
        self.camera_persisting = False
        page.set_camera_busy(False)
        effective = ""
        document: dict[str, Any] = {}
        try:
            parsed = json.loads(output) if output else {}
            if isinstance(parsed, dict):
                document = parsed
                effective = json.dumps(
                    document.get("controls") or document, indent=2, sort_keys=True
                )
        except (json.JSONDecodeError, ValueError):
            effective = output
        applied_ok = bool(ok and document.get("ok", True) is not False)
        detail = str(document.get("detail") or "").strip()
        capture_format = str(document.get("capture_format") or "").strip()
        live_applied = document.get("live_controls_applied")
        self.camera_last_result_ok = applied_ok
        self.camera_last_result_detail = detail or output
        if applied_ok and persisted:
            live_note = (
                " Live controls were applied to the current owner."
                if live_applied is True
                else " Live controls are staged for the next owner start."
                if live_applied is False
                else ""
            )
            format_note = (
                f" Capture format: {capture_format}." if capture_format else ""
            )
            page.set_camera_status(
                "Saved. "
                + (detail or "The capture owner accepted the configuration.")
                + live_note
                + format_note,
                state="saved",
                effective=effective,
            )
        elif applied_ok:
            page.set_camera_status(
                (detail or "Live preview updated through the capture owner.")
                + " These values are session-only until you save them.",
                state="preview",
                effective=effective,
            )
        else:
            page.set_camera_status(
                "The camera configuration was not applied and nothing changed: "
                f"{output or 'the helper reported a failure'}",
                state="failed",
            )
        if output:
            self.append_log(f"camera adapter: {output}")
        if self.pending_camera_request is not None:
            QTimer.singleShot(0, self._replay_pending_camera_request)

    def _camera_unavailable(self, message: str) -> None:
        self.camera_persisting = False
        self.camera_last_result_ok = False
        self.camera_last_result_detail = message
        self.routing_page.set_camera_busy(False)
        self.routing_page.set_camera_status(
            "The camera adapter could not start. No camera was opened or "
            f"changed. {message}",
            state="failed",
        )
        self.append_log(message)
        if self.pending_camera_request is not None:
            QTimer.singleShot(0, self._replay_pending_camera_request)

    def _apply_android_lens_facing(self, lens_facing: str) -> None:
        """Automatically apply lens_facing change when switching Android cameras."""
        if self.camera_helper.busy():
            # If camera helper is busy, try again later
            QTimer.singleShot(200, lambda: self._apply_android_lens_facing(lens_facing))
            return
        
        # Get current camera values from the form
        values = self.routing_page.camera_values()
        if not values:
            # Form not ready yet, use just lens_facing
            values = {"lens_facing": lens_facing}
        else:
            # Update lens_facing in existing values
            values["lens_facing"] = lens_facing
        
        # Apply the configuration with persist=True so it sticks
        self._launch_camera_configuration("android-camera2", values, persist=True)
        self.append_log(f"Applied lens_facing={lens_facing} to Android camera")

    # ----------------------------------------------------------------- android

    def _query_android_status(self) -> None:
        if not self.android_enabled or self.android_probe.busy():
            return
        now = time.monotonic()
        if now - self.android_last_query < 3.0:
            return
        self.android_last_query = now
        arguments = [
            "status",
            "--host",
            self.android_host,
            "--camera-id",
            self.android_camera_id,
        ]
        if self.android_serial:
            arguments.extend(["--serial", self.android_serial])
        self.android_probe.run(local_helper("android_bridge.py"), arguments)

    def _android_status_parsed(self, document: Any, error: str) -> None:
        self.android_status = document if isinstance(document, dict) else {}
        self.android_error = error or None
        self._sync_camera_form()
        desired = self._android_output_signature()
        if self._android_output_applied(self.android_status, desired):
            self.pending_android_output_configuration = False
            self.android_output_retry_timer.stop()
        else:
            self.pending_android_output_configuration = True
            if (
                self._android_output_supported(self.android_status)
                and not self._android_output_configuration_matches(
                    self.android_status, desired
                )
            ):
                QTimer.singleShot(0, self._apply_android_output_configuration)

    # ----------------------------------------------------------------- passive

    def open_phone_return_preview(self) -> None:
        if self.phone_return_preview_window is None:
            self.phone_return_preview_window = PhoneReturnPreviewWindow(
                self.android_native_preview_port,
                self.phone_relay_health_file,
                self,
                transform_supplier=lambda: dict(
                    self.desired_state["output_transform"]
                ),
            )
            self.phone_return_preview_window.setStyleSheet(self.styleSheet())
        self.phone_return_preview_window.show()
        self.phone_return_preview_window.raise_()
        self.phone_return_preview_window.activateWindow()

    def set_measurement_enabled(self, enabled: bool) -> None:
        self.quality_analysis_enabled = bool(enabled)
        if enabled:
            self.reset_measurements()
        else:
            self.append_log("passive measurement paused")

    def reset_measurements(self) -> None:
        self.raw_quality.reset()
        self.result_quality.reset()
        self.append_log("passive measurement sample window reset")

    def restart_readers(self) -> None:
        """Recycle this manager's own readers; nothing else is touched."""
        if self.output_decoder.running:
            self.output_decoder.restart()
        else:
            self.output_decoder.start()
        if self.input_decoder.running:
            self.input_decoder.restart()
        self.delayed_frames.clear()
        self.append_log("passive preview readers recycling")

    def copy_snapshot(self) -> None:
        QApplication.clipboard().setText(
            json.dumps(self.snapshot, indent=2, sort_keys=True, default=str)
        )
        self.append_log("diagnostic JSON copied to the clipboard")

    def append_log(self, line: str) -> None:
        self.system_page.append_log(f"{time.strftime('%H:%M:%S')}  {line}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        self.alignment_timer.stop()
        self.windows_config_timer.stop()
        self.camera_live_timer.stop()
        self.android_output_retry_timer.stop()
        self.windows.abort_all()
        self.input_decoder.stop()
        self.output_decoder.stop()
        self.framing_decoder.stop()
        if self.phone_return_preview_window is not None:
            self.phone_return_preview_window.close()
        for helper in (
            self.android_probe,
            self.policy_helper,
            self.output_helper,
            self.android_output_helper,
            self.local_processor_helper,
            self.phone_relay_helper,
            self.camera_helper,
        ):
            helper.terminate()
        self._stop_prerecorded_relay()
        event.accept()


# The historical class name is part of the installed entry point's contract and
# of the ownership tests, so it stays as an alias of the current shell.
TesterWindow = ManagerWindow
