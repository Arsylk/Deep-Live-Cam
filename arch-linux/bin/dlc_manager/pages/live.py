#!/usr/bin/env python3
"""Live operation: the selected result, its raw comparison, and route facts.

The hero pane is whatever is actually reaching the stable system camera, and
every claim on this page names the stream it came from.  Nothing here starts,
stops, or reopens a capture service or a camera device.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..viewmodel import ROUTE_WINDOWS_ANDROID, STREAM_RAW, STREAM_RESULT, ManagerView
from ..widgets import (
    Card,
    ResponsiveCardGrid,
    VideoPane,
    page_heading,
)


class LivePage(QWidget):
    """Workspace 1: watch the live result and confirm where it comes from."""

    phoneReturnRequested = Signal()
    reconnectRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(
            page_heading(
                "Live",
                "The large pane uses the same processed frames and applies the "
                "selected output orientation locally for display. Each camera "
                "sink encodes independently; neither preview opens a device.",
            )
        )

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("livePreviewSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.result_pane = VideoPane(
            "SELECTED RESULT",
            "resolving the active processed source…",
        )
        self.raw_pane = VideoPane(
            "ARCH RAW COMPARISON",
            "capture-owner diagnostic copy",
        )
        self.splitter.addWidget(self.result_pane)
        self.splitter.addWidget(self.raw_pane)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([900, 520])
        layout.addWidget(self.splitter, 1)

        self.tiles = ResponsiveCardGrid(minimum_card_width=205, spacing=9)
        self.route_tile = Card("Route", pill=True)
        self.policy_tile = Card("Camera policy", pill=True)
        self.input_tile = Card("Active input", pill=True)
        self.processor_tile = Card("Processor", pill=True)
        self.identity_tile = Card("Identity effect", pill=True)
        self.attention_tile = Card("Attention", pill=True)
        tiles = [
            self.route_tile,
            self.policy_tile,
            self.input_tile,
            self.processor_tile,
            self.identity_tile,
            self.attention_tile,
        ]
        for tile in tiles:
            # Every tile carries up to three wrapped lines of explanation, and
            # a grid row is only as tall as it is told to be.
            tile.setMinimumHeight(92)
        self.tiles.set_cards(tiles)
        layout.addWidget(self.tiles)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.phone_return_button = QPushButton("Open phone return preview")
        self.phone_return_button.setToolTip(
            "Open a passive view of the phone-relay frames with the selected "
            "output orientation. It opens no camera."
        )
        self.phone_return_button.clicked.connect(self.phoneReturnRequested.emit)
        self.reconnect_button = QPushButton("Reconnect preview readers")
        self.reconnect_button.setProperty("compact", True)
        self.reconnect_button.setToolTip(
            "Recycle this manager's own decoders only. Capture, processing, and "
            "the stable camera identities are untouched."
        )
        self.reconnect_button.clicked.connect(self.reconnectRequested.emit)
        actions.addStretch(1)
        actions.addWidget(self.phone_return_button)
        actions.addWidget(self.reconnect_button)
        layout.addLayout(actions)

    def render(self, view: ManagerView, *, phone_return_live: bool) -> None:
        result = view.stream(STREAM_RESULT)
        raw = view.stream(STREAM_RAW)
        self.result_pane.render(result)
        self.raw_pane.render(raw)

        self.route_tile.pill.set_state(view.route.badge, view.route.state)
        self.route_tile.set_detail(view.route.summary)

        self.policy_tile.pill.set_state(
            view.system_camera.configured_policy.upper(), "info"
        )
        self.policy_tile.set_detail(
            f"{view.system_camera.configured_label}. Fallback order: "
            + " → ".join(view.system_camera.fallback)
            if view.system_camera.fallback
            else "The receiver has not reported a fallback order yet."
        )

        self.input_tile.pill.set_state(
            view.system_camera.state_text, view.system_camera.state
        )
        self.input_tile.set_detail(view.system_camera.detail)

        processor = view.processor
        if view.route.windows_bypassed:
            model = f"{processor.local_model}/{processor.local_backend}"
            model_state = "running" if processor.local_running else "working"
            model_detail = processor.local_detail
        else:
            model = processor.windows_active_model
            model_state = "running" if processor.windows_reachable else "failed"
            model_detail = processor.windows_detail
        self.processor_tile.pill.set_state(
            model.split(" via ")[0].upper(), model_state
        )
        self.processor_tile.set_detail(model_detail)

        if view.route.windows_bypassed:
            identity_state = (
                "running" if processor.visual_effect_confirmed else "working"
            )
            identity_text = processor.identity_status.replace("-", " ").upper()
            identity_detail = processor.identity_detail
        else:
            identity_state = "unknown"
            identity_text = "NOT MEASURED"
            identity_detail = (
                "The Windows processor reports its own quality signals; this "
                "manager does not measure identity similarity for that route."
            )
        self.identity_tile.pill.set_state(identity_text, identity_state)
        self.identity_tile.set_detail(identity_detail)

        alerts = view.alerts
        critical = view.critical_alerts()
        if not alerts:
            self.attention_tile.pill.set_state("NOTHING TO REVIEW", "running")
            self.attention_tile.set_detail(
                "No component reported a problem in this refresh."
            )
        else:
            self.attention_tile.pill.set_state(
                f"{len(alerts)} ITEM" + ("S" if len(alerts) != 1 else ""),
                "critical" if critical else "warning",
            )
            worst = view.worst_alert()
            self.attention_tile.set_detail(
                f"{worst.component}: {worst.message}" if worst else ""
            )

        direct_windows_return = view.route.key == ROUTE_WINDOWS_ANDROID
        self.phone_return_button.setEnabled(not direct_windows_return)
        if direct_windows_return:
            self.phone_return_button.setText(
                "Direct Windows return · result shown above"
            )
            self.phone_return_button.setToolTip(
                "The paired Windows slot returns directly to Android, so Arch "
                "has no exact encoded phone-return relay. The selected result "
                "preview above shows the same processed frames."
            )
        else:
            self.phone_return_button.setText(
                "Open phone return preview · LIVE"
                if phone_return_live
                else "Open phone return preview · waiting"
            )
            self.phone_return_button.setToolTip(
                "Open a passive window showing the exact encoded stream sent "
                "to the phone by the exclusive Arch relay."
            )
