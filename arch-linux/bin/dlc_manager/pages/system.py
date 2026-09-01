#!/usr/bin/env python3
"""System and diagnostics: plain-language status, the contract, and the log.

Problems are stated as "component + what happened + what to do", never as raw
JSON a reader has to decode. Runtime diagnostics are deliberately read-only;
the clean one-node v4l2loopback migration is handled by the installer at boot.
"""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..contracts import ARCH_LOCAL_RELAYS, WIRE_FPS, WIRE_HEIGHT, WIRE_WIDTH
from ..viewmodel import ManagerView
from ..widgets import (
    AlertList,
    Card,
    MetricRow,
    ResponsiveCardGrid,
    page_heading,
    scrollable,
)


LOG_LINE_LIMIT = 300


class SystemPage(QWidget):
    """Hidden read-only health, endpoint-contract, and event-log surface."""

    reconnectRequested = Signal()
    copySnapshotRequested = Signal()
    reloadWindowsRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(
            page_heading(
                "System and logs",
                "What each node is doing, the exact endpoint and device "
                "contract, and the actions that are safe to take from here.",
            )
        )

        self.alert_card = Card(
            "ATTENTION",
            "Every entry names the component that failed and one next action.",
            pill=True,
        )
        self.alerts = AlertList(limit=6, empty_text="No component reported a problem.")
        self.alert_card.add(self.alerts)
        layout.addWidget(self.alert_card)

        self.status_grid = ResponsiveCardGrid(minimum_card_width=330, spacing=10)
        self.android_card = Card("ANDROID PHONE", pill=True)
        self.android_summary = QLabel("")
        self.android_summary.setObjectName("hintText")
        self.android_summary.setWordWrap(True)
        self.android_identity = MetricRow("Expected device", "—")
        self.android_camera = MetricRow("Processed Camera2 output", "—")
        for widget in (
            self.android_summary,
            self.android_identity,
            self.android_camera,
        ):
            self.android_card.add(widget)

        self.windows_card = Card("WINDOWS PROCESSOR", pill=True)
        self.windows_summary = QLabel("")
        self.windows_summary.setObjectName("hintText")
        self.windows_summary.setWordWrap(True)
        self.windows_model = MetricRow("Loaded model", "—")
        self.windows_selection = MetricRow("Processing input", "—")
        for widget in (
            self.windows_summary,
            self.windows_model,
            self.windows_selection,
        ):
            self.windows_card.add(widget)

        self.arch_card = Card("ARCH NODE", pill=True)
        self.arch_summary = QLabel("")
        self.arch_summary.setObjectName("hintText")
        self.arch_summary.setWordWrap(True)
        self.arch_services = MetricRow("Capture and sink services", "—")
        self.arch_camera = MetricRow("Stable system camera", "—")
        for widget in (self.arch_summary, self.arch_services, self.arch_camera):
            self.arch_card.add(widget)

        self.status_grid.set_cards(
            [self.android_card, self.windows_card, self.arch_card]
        )
        layout.addWidget(self.status_grid)

        self.middle_grid = ResponsiveCardGrid(minimum_card_width=430, spacing=10)
        self.middle_grid.set_cards(
            [self._build_contract_card(), self._build_actions_card()]
        )
        layout.addWidget(self.middle_grid)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("diagnosticSplitter")
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setObjectName("diagnostics")
        self.stats_box.setFont(QFont("monospace", 10))
        self.stats_box.setAccessibleName("Detailed diagnostic snapshot")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("diagnostics")
        self.log_box.setFont(QFont("monospace", 9))
        self.log_box.document().setMaximumBlockCount(LOG_LINE_LIMIT)
        self.log_box.setAccessibleName("Bounded event log")
        self.splitter.addWidget(self.stats_box)
        self.splitter.addWidget(self.log_box)
        self.splitter.setSizes([840, 520])
        # Keep both panes usable inside the scrolling page instead of letting
        # them collapse when the window is at its 980x680 minimum.
        self.splitter.setMinimumHeight(260)
        layout.addWidget(self.splitter, 1)
        outer.addWidget(scrollable(panel))

    # ------------------------------------------------------------------ build

    def _build_contract_card(self) -> Card:
        card = Card(
            "ENDPOINT AND DEVICE CONTRACT",
            "The exact ports and paths this installation uses right now.",
        )
        self.contract_label = QLabel("")
        self.contract_label.setObjectName("readout")
        self.contract_label.setWordWrap(True)
        self.contract_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        card.add(self.contract_label)
        relays = QLabel(
            "\n".join(
                f"{relay.port}  {relay.title:<26}  {relay.owner}\n"
                f"       {relay.detail}"
                for relay in ARCH_LOCAL_RELAYS
            )
        )
        relays.setObjectName("hintText")
        relays.setWordWrap(True)
        relays.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card.add(relays)
        return card

    def _build_actions_card(self) -> Card:
        card = Card(
            "SAFE ACTIONS",
            "None of these change which process owns a camera.",
        )
        reconnect = QPushButton("Reconnect this manager's preview readers")
        reconnect.setToolTip(
            "Recycles only this window's ffmpeg decoders. Capture, processing, "
            "and the stable camera identities keep running."
        )
        reconnect.clicked.connect(self.reconnectRequested.emit)
        reload_windows = QPushButton("Reload settings from Windows")
        reload_windows.setToolTip(
            "Re-reads the processor configuration over the LAN JSON API."
        )
        reload_windows.clicked.connect(self.reloadWindowsRequested.emit)
        copy = QPushButton("Copy diagnostic JSON")
        copy.setToolTip("Copies the snapshot on the left to the clipboard.")
        copy.clicked.connect(self.copySnapshotRequested.emit)
        for button in (reconnect, reload_windows, copy):
            card.add(button)

        # Collect vertical slack at the bottom instead of letting a wrapped
        # label absorb it and float the buttons into the middle of the card.
        card.add_stretch()
        return card

    # ------------------------------------------------------------------ render

    def append_log(self, line: str) -> None:
        self.log_box.append(line)

    def set_snapshot_text(self, text: str) -> None:
        # Replacing the document would reset the scroll position, so only write
        # when the content actually changed.
        if self.stats_box.toPlainText() != text:
            bar = self.stats_box.verticalScrollBar()
            position = bar.value()
            at_bottom = position >= bar.maximum() - 2
            self.stats_box.setPlainText(text)
            bar.setValue(bar.maximum() if at_bottom else position)

    def render(self, view: ManagerView, *, ports: dict[str, int]) -> None:
        alerts: Sequence[Any] = view.alerts
        self.alerts.render(alerts)
        if self.alert_card.pill is not None:
            critical = view.critical_alerts()
            self.alert_card.pill.set_state(
                "ALL CLEAR"
                if not alerts
                else f"{len(alerts)} ITEM" + ("S" if len(alerts) != 1 else ""),
                "running"
                if not alerts
                else ("critical" if critical else "warning"),
            )

        android = view.android
        if self.android_card.pill is not None:
            self.android_card.pill.set_state(android.state_text, android.state)
        self.android_summary.setText(android.summary)
        self.android_identity.set_value(
            f"{android.model} at {android.host}"
            + (f" (serial {android.serial})" if android.serial else "")
        )
        self.android_camera.set_value(
            f"Camera2 {android.camera_id}: "
            + ("published" if android.camera_published else "not confirmed")
        )

        processor = view.processor
        if self.windows_card.pill is not None:
            self.windows_card.pill.set_state(
                processor.windows_state_text, processor.windows_state
            )
        self.windows_summary.setText(
            f"{view.windows_host}:8090 · {processor.windows_detail}"
        )
        self.windows_model.set_value(
            f"{processor.windows_active_model} (requested "
            f"{processor.windows_requested_model})"
        )
        self.windows_selection.set_value(
            f"{view.selected_device_id or 'none'} · mode {processor.windows_mode}"
            + (" · switching" if view.switching else "")
        )

        system = view.system_camera
        if self.arch_card.pill is not None:
            self.arch_card.pill.set_state(system.state_text, system.state)
        self.arch_summary.setText(system.detail)
        self.arch_services.set_value(
            " · ".join(f"{service.title}: {service.state_text}" for service in view.services)
        )
        self.arch_camera.set_value(
            f"{', '.join(system.devices) or 'not configured'} → "
            f"{system.active_label}"
        )

        self.contract_label.setText(
            f"Nodes      Android {view.hosts.get('android', '?')}  ·  Arch "
            f"{view.hosts.get('arch', '?')}  ·  Windows {view.hosts.get('windows', '?')}\n"
            f"Wire       H.264 / MPEG-TS / SRT  ·  {WIRE_WIDTH}×{WIRE_HEIGHT} at "
            f"{WIRE_FPS} FPS\n"
            f"Slots      slot N sends on 10000+2N; its return is always the "
            f"next port\n"
            f"Selected   {view.windows_host}:{view.selected_stream_port} "
            "(pulled by the receiver)\n"
            f"Capture    {view.capture_device or 'not resolved'}\n"
            f"Public     {', '.join(view.virtual_devices) or 'not configured'}\n"
            f"This app   raw {ports.get('raw', 0)}  ·  result "
            f"{ports.get('result', 0)}  ·  phone return "
            f"{ports.get('phone_return', 0)}"
        )
