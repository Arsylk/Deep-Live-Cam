#!/usr/bin/env python3
"""Stable Arch and Android output controls with a passive live preview."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..desired_state import (
    OUTPUT_ANDROID_PHONE,
    OUTPUT_ARCH_CAMERA,
    PROCESSOR_ARCH,
    PROCESSOR_SPECS,
    PROCESSOR_WINDOWS,
)
from ..viewmodel import ManagerView
from ..widgets import (
    Card,
    MetricStrip,
    ResponsiveCardGrid,
    note_label,
    scrollable,
    setting_label,
)
from ..contracts import SCOPE_BOTH
from .live import LivePage


class OutputPage(LivePage):
    """Output tab; changes target frame workers, never camera registrations."""

    outputToggled = Signal(str, bool)
    transformChanged = Signal(bool, int)
    processorProcessingToggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading_output = False
        self.outputs: dict[str, QCheckBox] = {}
        self.output_pills: dict[str, object] = {}
        self.processor_processing: dict[str, QCheckBox] = {}
        self.processor_processing_status: dict[str, QLabel] = {}
        self.processor_processing_pills: dict[str, object] = {}
        heading = self.findChild(QLabel, "pageTitle")
        if heading is not None:
            heading.setText("Output")
        subtitles = self.findChildren(QLabel, "pageSubtitle")
        if subtitles:
            subtitles[0].setText(
                "Choose stable destinations, orient them, and preview the "
                "processed frames with the selected output orientation."
            )
        # Output is deliberately compact: one exact result preview and the two
        # decisions above it. Detailed route tiles belong to diagnostics, not
        # beside every ordinary output control.
        self.raw_pane.setVisible(False)
        self.tiles.setVisible(False)
        self.splitter.setSizes([1, 0])
        layout = self.layout()
        if isinstance(layout, QVBoxLayout):
            layout.insertWidget(1, self._build_output_controls())
            preview_index = layout.indexOf(self.splitter)
            layout.insertWidget(preview_index + 1, self._build_pipeline_metadata())
            self._wrap_in_scroll_area(layout)

    @staticmethod
    def _wrap_in_scroll_area(layout: QVBoxLayout) -> None:
        """Keep the preview useful at 980×680 without imposing a tall minimum."""
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(10)
        while layout.count():
            stretch = layout.stretch(0)
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            spacer = item.spacerItem()
            if widget is not None:
                panel_layout.addWidget(widget, stretch)
            elif child_layout is not None:
                panel_layout.addLayout(child_layout, stretch)
            elif spacer is not None:
                panel_layout.addItem(spacer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scrollable(panel))

    def _build_output_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)

        destinations = Card(
            "OUTPUT DESTINATIONS",
            "Both destinations may remain enabled. Disabling delivery preserves "
            "the registered camera identity and its open sessions.",
            pill=True,
        )
        self.destination_pill = destinations.pill
        target_grid = ResponsiveCardGrid(minimum_card_width=330, spacing=9)
        target_cards: list[QWidget] = []
        definitions = (
            (
                OUTPUT_ARCH_CAMERA,
                "Arch virtual webcam",
                "/dev/deep-live-cam · processed /dev/video42 output labelled Xiaomi Cam",
            ),
            (
                OUTPUT_ANDROID_PHONE,
                "Phone processed camera",
                "Processed return is continuously published as Camera2 ID 120. "
                "The scoped Xposed route substitutes it for the default front "
                "camera without changing that application's camera selection.",
            ),
        )
        for key, title, detail in definitions:
            card = Card(title.upper(), detail, pill=True)
            toggle = QCheckBox("Deliver processed video here")
            toggle.toggled.connect(
                lambda enabled, output_key=key: self._output_toggled(
                    output_key, enabled
                )
            )
            self.outputs[key] = toggle
            self.output_pills[key] = card.pill
            card.add(toggle)
            if key == OUTPUT_ANDROID_PHONE:
                self.phone_redirect_status = QLabel(
                    "Front-camera substitution has not been queried yet."
                )
                self.phone_redirect_status.setObjectName("hintText")
                self.phone_redirect_status.setWordWrap(True)
                card.add(self.phone_redirect_status)
            target_cards.append(card)
        target_grid.set_cards(target_cards)
        self.output_grid = target_grid
        destinations.add(target_grid)

        processing = Card(
            "FACE SWAP PROCESSOR",
            "Each switch is a complete action: it selects that host and turns "
            "face swapping on. Only one host can be on. Turn the active host "
            "off for passthrough without interrupting any camera session.",
            pill=True,
        )
        self.processing_pill = processing.pill
        processor_grid = ResponsiveCardGrid(minimum_card_width=330, spacing=9)
        processor_cards: list[QWidget] = []
        for processor_key, title in (
            (PROCESSOR_WINDOWS, "Windows 11 processing"),
            (PROCESSOR_ARCH, "Arch workstation processing"),
        ):
            specification = PROCESSOR_SPECS[processor_key]
            card = Card(
                title.upper(),
                f"{specification.model} · {specification.backend}. "
                f"{specification.detail}",
                pill=True,
            )
            toggle = QCheckBox(
                "Enable face swapping on "
                + ("Windows 11" if processor_key == PROCESSOR_WINDOWS else "Arch")
            )
            toggle.setAccessibleName(
                "Windows 11 face swapping"
                if processor_key == PROCESSOR_WINDOWS
                else "Arch workstation face swapping"
            )
            toggle.setAccessibleDescription(
                "Selects this processor and enables face swapping in one click. "
                "Turning it off keeps all input and output camera sessions open."
            )
            toggle.setToolTip(toggle.accessibleDescription())
            toggle.toggled.connect(
                lambda enabled, key=processor_key: self._processor_processing_toggled(
                    key, enabled
                )
            )
            status = QLabel("")
            status.setObjectName("hintText")
            status.setWordWrap(True)
            self.processor_processing[processor_key] = toggle
            self.processor_processing_status[processor_key] = status
            self.processor_processing_pills[processor_key] = card.pill
            card.add(toggle)
            card.add(status)
            processor_cards.append(card)
        processor_grid.set_cards(processor_cards)
        self.processor_processing_grid = processor_grid
        processing.add(processor_grid)

        transform = Card(
            "ORIENTATION",
            "The selected transform is part of the output contract and is "
            "applied to live frame workers without reopening a camera. "
            "Quarter turns fill the fixed camera frame with a centred crop; "
            "they never shrink the picture or add black bars.",
        )
        form = transform.add_form()
        self.mirror = QCheckBox("Mirror horizontally")
        self.mirror.toggled.connect(self._transform_changed)
        form.addRow(setting_label("Mirror", SCOPE_BOTH), self.mirror)
        self.rotation = QComboBox()
        for degrees in (0, 90, 180, 270):
            self.rotation.addItem(f"{degrees}°", degrees)
        self.rotation.currentIndexChanged.connect(self._transform_changed)
        form.addRow(setting_label("Rotate", SCOPE_BOTH), self.rotation)
        self.transform_status = QLabel("Waiting for the receiver…")
        self.transform_status.setObjectName("hintText")
        self.transform_status.setWordWrap(True)
        transform.add(self.transform_status)

        # Destinations need width for their two independent cards; orientation
        # is intentionally a compact second row instead of a same-height peer
        # that leaves half the page empty on a wide monitor.
        layout.addWidget(destinations)
        layout.addWidget(processing)
        layout.addWidget(transform)
        layout.addWidget(
            note_label(
                "Session safety: destination and orientation changes keep the "
                "Arch sink, Android provider, producer, and public camera IDs alive.",
                "info",
            )
        )
        return container

    def _build_pipeline_metadata(self) -> QWidget:
        """Compact, health-backed evidence immediately below the result feed."""
        card = Card(
            "LIVE PIPELINE EVIDENCE",
            "Reported by the selected processor. UNKNOWN means its health "
            "payload did not provide that evidence; video presence is never "
            "treated as proof of detection or a swap.",
            pill=True,
        )
        self.pipeline_evidence_pill = card.pill
        self.pipeline_metrics = MetricStrip(
            (
                "Face detected",
                "Face swapped",
                "Model / backend",
                "Generated face",
                "Processing FPS",
                "Error / waiting reason",
            ),
            spacing=12,
        )
        card.add(self.pipeline_metrics)
        return card

    def _output_toggled(self, key: str, enabled: bool) -> None:
        if not self._loading_output:
            self.outputToggled.emit(key, bool(enabled))

    def _transform_changed(self, _value: object = None) -> None:
        if not self._loading_output:
            self.transformChanged.emit(
                self.mirror.isChecked(), int(self.rotation.currentData() or 0)
            )

    def _processor_processing_toggled(self, processor: str, enabled: bool) -> None:
        if self._loading_output:
            return
        # Checkboxes are intentionally not a radio group because the active
        # processor may be turned off to select passthrough. Enforce the
        # one-active-host invariant without echoing the programmatic uncheck.
        if enabled:
            self._loading_output = True
            try:
                for key, toggle in self.processor_processing.items():
                    if key != processor:
                        toggle.setChecked(False)
            finally:
                self._loading_output = False
        self._update_processor_processing_status(
            processor if enabled else self._checked_processor(),
            enabled=bool(self._checked_processor()),
        )
        self.processorProcessingToggled.emit(str(processor), bool(enabled))

    def _checked_processor(self) -> str | None:
        return next(
            (
                key
                for key, toggle in self.processor_processing.items()
                if toggle.isChecked()
            ),
            None,
        )

    def _update_processor_processing_status(
        self, selected: str | None, *, enabled: bool, pending: bool = False
    ) -> None:
        for key, status in self.processor_processing_status.items():
            pill = self.processor_processing_pills[key]
            if key == selected and enabled:
                status.setText(
                    "Active target: face swapping is enabled. Settings are "
                    "hot-applied without reopening camera devices."
                )
                if pill is not None:
                    pill.set_state(
                        "APPLYING" if pending else "FACE SWAP ON",
                        "working" if pending else "running",
                    )
            elif key == selected:
                status.setText(
                    "Selected target, currently passthrough. Input and output "
                    "camera sessions remain connected."
                )
                if pill is not None:
                    pill.set_state(
                        "APPLYING" if pending else "PASSTHROUGH",
                        "working" if pending else "stopped",
                    )
            else:
                status.setText(
                    "Off. Click once to switch processing to this host and "
                    "enable face swapping."
                )
                if pill is not None:
                    pill.set_state("OFF", "off")
        if self.processing_pill is not None:
            if enabled and selected in PROCESSOR_SPECS:
                label = (
                    "WINDOWS 11" if selected == PROCESSOR_WINDOWS else "ARCH"
                )
                self.processing_pill.set_state(
                    f"{label} · " + ("APPLYING" if pending else "ON"),
                    "working" if pending else "running",
                )
            else:
                self.processing_pill.set_state(
                    "APPLYING" if pending else "PASSTHROUGH",
                    "working" if pending else "stopped",
                )

    def set_processor_processing_state(
        self, document: dict, *, pending: bool = False
    ) -> None:
        """Load the exclusive processor switches without emitting user events."""
        selected = str(document.get("processor", PROCESSOR_WINDOWS))
        processing = document.get("processing") or {}
        mode = processing.get("processing_mode")
        if mode is None and "processing_enabled" in processing:
            mode = (
                "face_swap" if bool(processing["processing_enabled"]) else "passthrough"
            )
        enabled = str(mode or "face_swap") == "face_swap"
        previous_loading = self._loading_output
        self._loading_output = True
        try:
            for key, toggle in self.processor_processing.items():
                toggle.setChecked(bool(enabled and key == selected))
        finally:
            self._loading_output = previous_loading
        self._update_processor_processing_status(
            selected, enabled=enabled, pending=pending
        )

    def set_output_state(self, document: dict, *, pending: bool = False) -> None:
        self._loading_output = True
        try:
            enabled = document.get("outputs") or {}
            for key, toggle in self.outputs.items():
                toggle.setChecked(bool(enabled.get(key, True)))
            transform = document.get("output_transform") or {}
            self.mirror.setChecked(bool(transform.get("mirror", False)))
            index = self.rotation.findData(int(transform.get("rotation", 0)))
            if index >= 0:
                self.rotation.setCurrentIndex(index)
            self.set_processor_processing_state(document, pending=pending)
        finally:
            self._loading_output = False
        count = sum(toggle.isChecked() for toggle in self.outputs.values())
        if self.destination_pill is not None:
            self.destination_pill.set_state(
                f"{count} ON" + (" · APPLYING" if pending else ""),
                "working" if pending else ("running" if count else "warning"),
            )
        self.transform_status.setText(
            f"Desired output: {'mirrored · ' if self.mirror.isChecked() else ''}"
            f"rotation {int(self.rotation.currentData() or 0)}°. "
            + ("Applying to live workers…" if pending else "Applied state retained.")
        )

    def render(self, view: ManagerView, *, phone_return_live: bool) -> None:
        super().render(view, phone_return_live=phone_return_live)
        processor = view.processor

        def yes_no_unknown(value: bool | None) -> str:
            if value is None:
                return "UNKNOWN"
            return "YES" if value else "NO"

        fps = (
            "UNKNOWN"
            if processor.selected_processing_fps is None
            else f"{processor.selected_processing_fps:.1f} FPS"
        )
        self.pipeline_metrics.update_values(
            (
                ("Face detected", yes_no_unknown(processor.selected_face_detected)),
                ("Face swapped", yes_no_unknown(processor.selected_face_swapped)),
                (
                    "Model / backend",
                    f"{processor.selected_model} / {processor.selected_backend}",
                ),
                (
                    "Generated face",
                    "UNKNOWN"
                    if processor.selected_render_resolution is None
                    else f"{processor.selected_render_resolution}×"
                    f"{processor.selected_render_resolution} px",
                ),
                ("Processing FPS", fps),
                ("Error / waiting reason", processor.selected_runtime_reason),
            )
        )
        if self.pipeline_evidence_pill is not None:
            self.pipeline_evidence_pill.set_state(
                (
                    "WINDOWS 11 HEALTH"
                    if processor.selected_processor == PROCESSOR_WINDOWS
                    else "ARCH HEALTH"
                ),
                processor.selected_runtime_state,
            )

    def set_delivery_status(
        self,
        desired: dict,
        receiver: dict,
        android: dict,
        *,
        receiver_service_active: bool = True,
        receiver_health_age: float | None = None,
    ) -> None:
        """Render desired/effective reconciliation without changing controls."""
        transform = desired.get("output_transform") or {}
        receiver = receiver if isinstance(receiver, dict) else {}
        android = android if isinstance(android, dict) else {}
        receiver_transform = receiver.get("output_transform") or {}
        if not isinstance(receiver_transform, dict):
            receiver_transform = {}
        arch_synced = bool(
            receiver
            and receiver.get("virtual_camera") == "/dev/deep-live-cam"
            and receiver.get("virtual_cameras") == ["/dev/deep-live-cam"]
            and receiver.get("source_mode", receiver.get("source"))
            == (
                "local"
                if desired.get("processor") == "arch"
                else "windows"
            )
            and receiver.get("output_enabled")
            == desired.get("outputs", {}).get(OUTPUT_ARCH_CAMERA)
            and receiver_transform.get("mirror") == transform.get("mirror")
            and receiver_transform.get("rotation") == transform.get("rotation")
        )
        arch_stream_live = bool(
            arch_synced
            and receiver_service_active
            and receiver_health_age is not None
            and receiver_health_age <= 3.0
            and receiver.get("status") == "streaming"
            and isinstance(receiver.get("sink_pid"), int)
            and receiver.get("sink_pid", 0) > 0
        )
        android_control = android.get("output_control") or {}
        if not isinstance(android_control, dict):
            android_control = {}
        phone_reachable = bool(android.get("available"))
        phone_supported = bool(android_control.get("supported"))
        phone_module_enabled = bool(android.get("module_enabled"))
        phone_selector_running = bool(android.get("output_selector_running"))
        phone_config_applied = bool(
            android_control.get("applied")
            and android_control.get("enabled")
            == desired.get("outputs", {}).get(OUTPUT_ANDROID_PHONE)
            and android_control.get("mirror") == transform.get("mirror")
            and android_control.get("rotation") == transform.get("rotation")
        )
        phone_requested = bool(
            desired.get("outputs", {}).get(OUTPUT_ANDROID_PHONE)
        )
        phone_stream_live = bool(
            phone_requested
            and phone_config_applied
            and android.get("camera_published")
            and android_control.get("effective_source") == "processed"
            and android_control.get("effective_worker_alive")
        )
        phone_delivery_off = bool(
            not phone_requested
            and phone_config_applied
            and android.get("camera_published")
            and android_control.get("effective_source") == "placeholder"
        )
        redirect = android.get("front_redirect") or {}
        if not isinstance(redirect, dict):
            redirect = {}
        if redirect.get("active") is True:
            redirect_text = (
                "Front-camera substitution is active for a scoped application "
                f"and resolves to Camera2 {redirect.get('processed_camera_id', '120')}."
            )
        elif redirect.get("package_installed"):
            redirect_text = (
                "The scoped Xposed front-camera route is installed. Activity is "
                "confirmed only while a scoped application opens its front camera."
            )
        else:
            redirect_text = (
                "Processed Camera2 delivery is separate from front-camera "
                "substitution; the scoped Xposed route is not verified."
            )
        self.phone_redirect_status.setText(redirect_text)
        arch_pill = self.output_pills.get(OUTPUT_ARCH_CAMERA)
        if arch_pill is not None:
            arch_requested = bool(
                desired.get("outputs", {}).get(OUTPUT_ARCH_CAMERA)
            )
            arch_pill.set_state(
                "STREAM LIVE"
                if arch_stream_live and arch_requested
                else "CONFIG SYNCED · WAITING"
                if arch_synced and arch_requested
                else "DELIVERY OFF · SYNCED"
                if arch_synced
                else "PENDING"
                if receiver
                else "OFFLINE",
                "running"
                if arch_stream_live or (arch_synced and not arch_requested)
                else ("working" if receiver else "warning"),
            )

        phone_pill = self.output_pills.get(OUTPUT_ANDROID_PHONE)
        if phone_pill is not None:
            if not phone_reachable:
                phone_text, phone_state = "OFFLINE", "warning"
            elif not android.get("module_installed"):
                phone_text, phone_state = "MODULE NOT INSTALLED", "failed"
            elif not phone_supported:
                phone_text, phone_state = "MODULE UPDATE REQUIRED", "failed"
            elif not phone_module_enabled:
                phone_text, phone_state = "MODULE DISABLED", "failed"
            elif not phone_selector_running:
                phone_text, phone_state = "SELECTOR OFFLINE", "failed"
            elif phone_delivery_off:
                phone_text, phone_state = "DELIVERY OFF · SYNCED", "running"
            elif not phone_requested and phone_config_applied:
                phone_text, phone_state = "DELIVERY OFF · WAITING", "working"
            elif phone_stream_live:
                phone_text, phone_state = "PROCESSED STREAM LIVE", "running"
            elif not phone_config_applied:
                phone_text, phone_state = "SETTINGS PENDING", "working"
            elif not android.get("camera_published"):
                phone_text, phone_state = "CAMERA OFFLINE", "failed"
            elif android_control.get("effective_source") == "raw":
                phone_text, phone_state = "RAW FALLBACK", "warning"
            else:
                phone_text, phone_state = "WAITING FOR PROCESSED STREAM", "working"
            phone_pill.set_state(phone_text, phone_state)
        self.transform_status.setText(
            f"Desired: {'mirrored · ' if transform.get('mirror') else ''}"
            f"rotation {int(transform.get('rotation', 0))}°. "
            f"Arch {'synced' if arch_synced else 'pending'} · phone "
            f"{'processed stream live' if phone_stream_live else 'delivery off' if phone_delivery_off else 'not live'}."
        )


__all__ = ["OutputPage"]
