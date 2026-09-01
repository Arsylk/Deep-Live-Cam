#!/usr/bin/env python3
"""Identity and processing: which face, which engine, and what it changes.

Two claims are kept apart on purpose.  Selecting a picture updates the one
durable identity used by the Windows INSwapper-128 and Arch Native-256
processors.  Neither is presented as proof that a face was actually replaced
— that lives in the evidence card, split into checkpoint qualification,
visual-effect evidence, and identity verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..contracts import (
    PROCESSOR_TARGETS,
    SCOPE_BOTH,
    TARGET_ARCH,
    TARGET_WINDOWS,
)
from ..desired_state import PROCESSOR_SPECS
from ..viewmodel import ManagerView
from ..widgets import (
    Card,
    MetricRow,
    StatusPill,
    ValueSlider,
    note_label,
    ResponsiveCardGrid,
    page_heading,
    scrollable,
    setting_label,
    thumbnail,
)


HISTORY_ICON = QSize(58, 58)
PREVIEW_SIZE = QSize(340, 220)


class ProcessingPage(QWidget):
    """Processor tab: engine selection, shared settings, and identity history."""

    sourcePictureRequested = Signal()
    historyPictureRequested = Signal(str)
    settingChanged = Signal(str, object)
    presetRequested = Signal(int)
    processorChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._target = TARGET_WINDOWS
        self._windows_available = True
        self._windows_reason = ""
        self.history_buttons: list[QToolButton] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 18)
        layout.setSpacing(10)
        layout.addWidget(
            page_heading(
                "Identity and processing",
                "Choose one processor and one identity. The same settings "
                "document follows you between processors and applies as soon "
                "as a control changes.",
            )
        )

        self.identity_card = self._build_identity_card()
        self.everyday_card = self._build_everyday_card()
        self.evidence_card = self._build_evidence_card()
        right = QWidget()
        right_column = QVBoxLayout(right)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(10)
        right_column.addWidget(self.everyday_card)
        right_column.addWidget(self.evidence_card)
        right_column.addStretch(1)
        self.top_grid = ResponsiveCardGrid(minimum_card_width=430, spacing=10)
        self.top_grid.set_cards([self.identity_card, right])
        layout.addWidget(self.top_grid)
        layout.addWidget(self._build_advanced_card())
        layout.addStretch(1)
        outer.addWidget(scrollable(panel))

    # ------------------------------------------------------------------ build

    def _build_identity_card(self) -> Card:
        card = Card(
            "IDENTITY SOURCE",
            "The picture below is what this manager last applied. Its two "
            "targets are reported separately.",
        )
        self.source_preview = QLabel("No source picture selected")
        self.source_preview.setObjectName("sourcePreview")
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_preview.setMinimumHeight(230)
        self.source_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        card.add(self.source_preview, 1)

        self.source_name = QLabel("Nothing applied from this control center yet")
        self.source_name.setObjectName("sourceName")
        self.source_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_name.setWordWrap(True)
        card.add(self.source_name)

        targets = QGridLayout()
        targets.setHorizontalSpacing(8)
        targets.setVerticalSpacing(4)
        windows_caption = QLabel("Windows processor upload")
        windows_caption.setObjectName("sectionLabel")
        self.windows_source_pill = StatusPill("CHECKING", "unknown")
        self.windows_source_detail = QLabel("")
        self.windows_source_detail.setObjectName("hintText")
        self.windows_source_detail.setWordWrap(True)
        local_caption = QLabel("Arch Native-256 source")
        local_caption.setObjectName("sectionLabel")
        self.local_source_pill = StatusPill("CHECKING", "unknown")
        self.local_source_detail = QLabel("")
        self.local_source_detail.setObjectName("hintText")
        self.local_source_detail.setWordWrap(True)
        targets.addWidget(windows_caption, 0, 0)
        targets.addWidget(self.windows_source_pill, 0, 1, Qt.AlignmentFlag.AlignRight)
        targets.addWidget(self.windows_source_detail, 1, 0, 1, 2)
        targets.addWidget(local_caption, 2, 0)
        targets.addWidget(self.local_source_pill, 2, 1, Qt.AlignmentFlag.AlignRight)
        targets.addWidget(self.local_source_detail, 3, 0, 1, 2)
        targets.setColumnStretch(0, 1)
        card.add_layout(targets)

        self.upload_source_button = QPushButton("Choose source picture…")
        self.upload_source_button.setProperty("emphasis", "primary")
        self.upload_source_button.clicked.connect(self.sourcePictureRequested.emit)
        card.add(self.upload_source_button)

        history_caption = QLabel("RECENT · CLICK ONCE TO SWITCH")
        history_caption.setObjectName("sectionLabel")
        card.add(history_caption)

        history_scroll = QScrollArea()
        history_scroll.setObjectName("sourceHistory")
        history_scroll.setWidgetResizable(True)
        history_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        history_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        history_scroll.setFixedHeight(112)
        container = QWidget()
        self.history_layout = QHBoxLayout(container)
        self.history_layout.setContentsMargins(3, 3, 3, 3)
        self.history_layout.setSpacing(6)
        history_scroll.setWidget(container)
        card.add(history_scroll)

        self.source_status = QLabel("Checking the Windows processor…")
        self.source_status.setObjectName("hintText")
        self.source_status.setWordWrap(True)
        card.add(self.source_status)
        return card

    def _build_everyday_card(self) -> Card:
        card = Card(
            "EVERYDAY CONTROLS",
            "The controls people change during a session. They are one shared "
            "desired state, not separate per-machine copies.",
        )
        target_row = QHBoxLayout()
        target_row.setSpacing(8)
        target_caption = QLabel("Processor target")
        target_caption.setObjectName("sectionLabel")
        self.processor_target = QComboBox()
        for key, label, _detail in PROCESSOR_TARGETS:
            self.processor_target.addItem(label, key)
        self.processor_target.currentIndexChanged.connect(self._target_changed)
        target_row.addWidget(target_caption)
        target_row.addWidget(self.processor_target, 1)
        card.add_layout(target_row)

        self.target_note = note_label("", "info")
        card.add(self.target_note)

        form = card.add_form()
        self.processing_preset = QComboBox()
        self.processing_preset.addItems(
            ["Custom", "Fast / low latency", "Balanced", "Quality"]
        )
        self.processing_preset.currentIndexChanged.connect(self._preset_changed)
        form.addRow(
            setting_label(
                "Preset",
                SCOPE_BOTH,
                "Sets several controls at once; each stays adjustable.",
            ),
            self.processing_preset,
        )

        self.processor_model = QLabel("")
        self.processor_model.setObjectName("readout")
        self.processor_model.setWordWrap(True)
        form.addRow(
            setting_label(
                "Fixed model",
                SCOPE_BOTH,
                "The processor choice owns the model; it is never silently "
                "changed by a preset.",
            ),
            self.processor_model,
        )
        # Compatibility alias for older diagnostics/tests. It is deliberately
        # not visible and is not a user setting anymore.
        self.win_swapper_model = QComboBox()
        self.win_swapper_model.setVisible(False)

        self.win_swapper_status = QLabel("Not loaded")
        self.win_swapper_status.setObjectName("hintText")
        form.addRow(
            setting_label(
                "Currently loaded",
                SCOPE_BOTH,
                "Reported by the processor, not requested by this manager.",
            ),
            self.win_swapper_status,
        )

        self.win_opacity = ValueSlider(0, 100, 100)
        self.win_opacity.setSuffix(" %")
        self.win_opacity.valueChanged.connect(
            lambda value: self._changed("opacity", value / 100.0)
        )
        form.addRow(setting_label("Face opacity", SCOPE_BOTH), self.win_opacity)

        self.win_color_match = ValueSlider(0, 100, 35)
        self.win_color_match.setSuffix(" %")
        self.win_color_match.valueChanged.connect(
            lambda value: self._changed("color_match_strength", value / 100.0)
        )
        form.addRow(setting_label("Colour match", SCOPE_BOTH), self.win_color_match)

        self.win_mouth_mask = ValueSlider(0, 100, 8)
        self.win_mouth_mask.setSuffix(" %")
        self.win_mouth_mask.valueChanged.connect(
            lambda value: self._changed("mouth_mask_size", float(value))
        )
        form.addRow(setting_label("Mouth mask", SCOPE_BOTH), self.win_mouth_mask)

        self.win_sharpness = ValueSlider(0, 50, 2)
        self.win_sharpness.setDisplayScale(10.0, 1)
        self.win_sharpness.valueChanged.connect(
            lambda value: self._changed("sharpness", value / 10.0)
        )
        form.addRow(setting_label("Sharpness", SCOPE_BOTH), self.win_sharpness)

        self.win_many_faces = QCheckBox("Process every detected face")
        self.win_many_faces.toggled.connect(
            lambda checked: self._changed("many_faces", checked)
        )
        form.addRow(setting_label("Multiple faces", SCOPE_BOTH), self.win_many_faces)

        return card

    def _build_advanced_card(self) -> Card:
        card = Card(
            "ADVANCED PROCESSING",
            "Detector cadence, tracking, enhancement, and the automatic output "
            "guard. These change how the processor behaves under load.",
        )
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_column = Card("Detection and tracking")
        tracking_form = left_column.add_form()

        self.win_tracking_enabled = QCheckBox(
            "Optical-flow tracking and miss hysteresis"
        )
        self.win_tracking_enabled.setChecked(True)
        self.win_tracking_enabled.toggled.connect(
            lambda checked: self._changed("tracking_enabled", checked)
        )
        tracking_form.addRow(
            setting_label(
                "Tracking",
                SCOPE_BOTH,
                "Predicts through a single detector miss instead of showing an "
                "unprocessed frame.",
            ),
            self.win_tracking_enabled,
        )
        self.win_detection_interval = ValueSlider(1, 5, 1)
        self.win_detection_interval.setSuffix(" frame(s)")
        self.win_detection_interval.valueChanged.connect(
            lambda value: self._changed("detection_interval", value)
        )
        tracking_form.addRow(
            setting_label("Detection cadence", SCOPE_BOTH, "1 is the most stable."),
            self.win_detection_interval,
        )
        self.win_detection_score = ValueSlider(10, 95, 45)
        self.win_detection_score.setSuffix(" %")
        self.win_detection_score.valueChanged.connect(
            lambda value: self._changed("minimum_detection_score", value / 100.0)
        )
        tracking_form.addRow(
            setting_label("Minimum confidence", SCOPE_BOTH),
            self.win_detection_score,
        )
        self.win_minimum_face_size = ValueSlider(32, 320, 64, 8)
        self.win_minimum_face_size.setSuffix(" px")
        self.win_minimum_face_size.valueChanged.connect(
            lambda value: self._changed("minimum_face_size", value)
        )
        tracking_form.addRow(
            setting_label("Minimum face size", SCOPE_BOTH),
            self.win_minimum_face_size,
        )
        self.win_tracking_smoothing = ValueSlider(0, 95, 65)
        self.win_tracking_smoothing.setSuffix(" %")
        self.win_tracking_smoothing.valueChanged.connect(
            lambda value: self._changed("tracking_smoothing", value / 100.0)
        )
        tracking_form.addRow(
            setting_label("Geometry smoothing", SCOPE_BOTH),
            self.win_tracking_smoothing,
        )
        self.win_tracking_grace = ValueSlider(0, 15, 5)
        self.win_tracking_grace.setSuffix(" frame(s)")
        self.win_tracking_grace.valueChanged.connect(
            lambda value: self._changed("tracking_grace_frames", value)
        )
        tracking_form.addRow(
            setting_label("Miss grace", SCOPE_BOTH), self.win_tracking_grace
        )
        left_layout.addWidget(left_column)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        enhancement = Card("Enhancement and delivery")
        enhancement_form = enhancement.add_form()
        self.win_enhancer = QComboBox()
        self.win_enhancer.addItem("None (fastest)", "none")
        self.win_enhancer.addItem("GFPGAN", "gfpgan")
        self.win_enhancer.currentIndexChanged.connect(
            lambda _index: self._changed("enhancer", self.win_enhancer.currentData())
        )
        enhancement_form.addRow(
            setting_label("Face enhancer", SCOPE_BOTH), self.win_enhancer
        )
        self.win_enable_interpolation = QCheckBox("Enable frame interpolation")
        self.win_enable_interpolation.toggled.connect(
            lambda checked: self._changed("enable_interpolation", checked)
        )
        enhancement_form.addRow(
            setting_label("Interpolation", SCOPE_BOTH),
            self.win_enable_interpolation,
        )
        self.win_interpolation_weight = ValueSlider(0, 100, 0)
        self.win_interpolation_weight.setSuffix(" %")
        self.win_interpolation_weight.valueChanged.connect(
            lambda value: self._changed("interpolation_weight", value / 100.0)
        )
        enhancement_form.addRow(
            setting_label("Interpolation strength", SCOPE_BOTH),
            self.win_interpolation_weight,
        )
        self.win_show_fps = QCheckBox("Draw the processing FPS on the output")
        self.win_show_fps.toggled.connect(
            lambda checked: self._changed("show_fps", checked)
        )
        enhancement_form.addRow(
            setting_label("Overlay", SCOPE_BOTH), self.win_show_fps
        )
        right_layout.addWidget(enhancement)

        repair = Card(
            "Detail and seam repair",
            "One shared post-swap profile for both hosts. Balanced defaults are "
            "the measured Windows baseline rather than processor-local state.",
        )
        repair_form = repair.add_form()
        self.win_repair_hf = ValueSlider(0, 50, 30)
        self.win_repair_hf.setSuffix(" %")
        self.win_repair_hf.valueChanged.connect(
            lambda value: self._changed("repair_hf_strength", value / 100.0)
        )
        repair_form.addRow(
            setting_label("Fine-detail restore", SCOPE_BOTH), self.win_repair_hf
        )
        self.win_repair_checkerboard = ValueSlider(0, 100, 40)
        self.win_repair_checkerboard.setSuffix(" %")
        self.win_repair_checkerboard.valueChanged.connect(
            lambda value: self._changed("repair_checkerboard", value / 100.0)
        )
        repair_form.addRow(
            setting_label("Checkerboard cleanup", SCOPE_BOTH),
            self.win_repair_checkerboard,
        )
        self.win_repair_wavelet = ValueSlider(0, 100, 50)
        self.win_repair_wavelet.setSuffix(" %")
        self.win_repair_wavelet.valueChanged.connect(
            lambda value: self._changed("repair_wavelet", value / 100.0)
        )
        repair_form.addRow(
            setting_label("Wavelet matching", SCOPE_BOTH), self.win_repair_wavelet
        )
        self.win_repair_camera_detail = ValueSlider(0, 400, 0)
        self.win_repair_camera_detail.setSuffix(" %")
        self.win_repair_camera_detail.valueChanged.connect(
            lambda value: self._changed("repair_camera_detail", value / 100.0)
        )
        repair_form.addRow(
            setting_label("Camera-resolution detail", SCOPE_BOTH),
            self.win_repair_camera_detail,
        )
        self.win_repair_boundary = QCheckBox("Use content-aware face boundary")
        self.win_repair_boundary.setChecked(True)
        self.win_repair_boundary.toggled.connect(
            lambda checked: self._changed("repair_boundary_mask", checked)
        )
        repair_form.addRow(
            setting_label("Semantic boundary", SCOPE_BOTH),
            self.win_repair_boundary,
        )
        self.win_repair_boundary_strength = ValueSlider(0, 100, 35)
        self.win_repair_boundary_strength.setSuffix(" %")
        self.win_repair_boundary_strength.valueChanged.connect(
            lambda value: self._changed(
                "repair_boundary_strength", value / 100.0
            )
        )
        repair_form.addRow(
            setting_label(
                "Boundary preservation",
                SCOPE_BOTH,
                "Keeps genuine camera pixels in the translucent face edge; "
                "the identity-bearing core remains unchanged.",
            ),
            self.win_repair_boundary_strength,
        )
        right_layout.addWidget(repair)

        guard = Card(
            "Automatic output guard",
            "Corrects bounded signal defects. It never chooses a camera source "
            "and never restarts a device.",
        )
        guard_form = guard.add_form()
        self.win_quality_mode = QComboBox()
        self.win_quality_mode.addItem("Monitor only", "monitor")
        self.win_quality_mode.addItem("Balanced correction", "balanced")
        self.win_quality_mode.addItem("Strict signal guard", "strict")
        self.win_quality_mode.currentIndexChanged.connect(
            lambda _index: self._changed(
                "quality_mode", self.win_quality_mode.currentData()
            )
        )
        guard_form.addRow(setting_label("Mode", SCOPE_BOTH), self.win_quality_mode)
        self.win_quality_auto_correct = QCheckBox("Bounded whole-frame correction")
        self.win_quality_auto_correct.setChecked(True)
        self.win_quality_auto_correct.toggled.connect(
            lambda checked: self._changed("quality_auto_correct", checked)
        )
        guard_form.addRow(
            setting_label("Correction", SCOPE_BOTH), self.win_quality_auto_correct
        )
        right_layout.addWidget(guard)


        self.advanced_grid = ResponsiveCardGrid(minimum_card_width=400, spacing=14)
        self.advanced_grid.set_cards([left, right])
        card.add(self.advanced_grid)
        return card

    def _build_evidence_card(self) -> Card:
        card = Card(
            "EVIDENCE FOR THE CURRENT RESULT",
            "Three separate questions. A processor being invoked, a visible "
            "change, and a verified identity are never merged into one claim.",
        )
        self.checkpoint_metric = MetricRow(
            "Checkpoint qualification",
            "unknown",
            "Whether the loaded model checkpoint is qualified for production.",
        )
        self.effect_metric = MetricRow(
            "Visual-effect evidence",
            "unknown",
            "Whether measured pixels inside the face core actually changed.",
        )
        self.verification_metric = MetricRow(
            "Identity verification",
            "not performed",
            "This manager never measures identity similarity, so a changed "
            "mask ring is not evidence of a verified swap.",
        )
        for metric in (
            self.checkpoint_metric,
            self.effect_metric,
            self.verification_metric,
        ):
            card.add(metric)
        return card

    # ----------------------------------------------------------------- signals

    def _changed(self, field: str, value: Any) -> None:
        if self._loading:
            return
        self.processing_preset.blockSignals(True)
        self.processing_preset.setCurrentIndex(0)
        self.processing_preset.blockSignals(False)
        self.settingChanged.emit(field, value)

    def _preset_changed(self, index: int) -> None:
        if self._loading or index <= 0:
            return
        self.presetRequested.emit(index)

    def _target_changed(self, _index: int) -> None:
        self._target = str(self.processor_target.currentData() or TARGET_WINDOWS)
        self._apply_enablement()
        if not self._loading:
            self.processorChanged.emit(self._target)

    # ------------------------------------------------------------------- state

    def setting_widgets(self) -> dict[str, QWidget]:
        """The single widget that owns each processor setting.

        One mapping, used both for enabling/disabling the form and by the
        off-screen contract test, is what keeps a second contradictory copy of
        a control from appearing on another page.
        """
        return {
            "opacity": self.win_opacity,
            "color_match_strength": self.win_color_match,
            "mouth_mask_size": self.win_mouth_mask,
            "sharpness": self.win_sharpness,
            "many_faces": self.win_many_faces,
            "tracking_enabled": self.win_tracking_enabled,
            "detection_interval": self.win_detection_interval,
            "minimum_detection_score": self.win_detection_score,
            "minimum_face_size": self.win_minimum_face_size,
            "tracking_smoothing": self.win_tracking_smoothing,
            "tracking_grace_frames": self.win_tracking_grace,
            "enhancer": self.win_enhancer,
            "enable_interpolation": self.win_enable_interpolation,
            "interpolation_weight": self.win_interpolation_weight,
            "show_fps": self.win_show_fps,
            "quality_mode": self.win_quality_mode,
            "quality_auto_correct": self.win_quality_auto_correct,
            "repair_hf_strength": self.win_repair_hf,
            "repair_checkerboard": self.win_repair_checkerboard,
            "repair_wavelet": self.win_repair_wavelet,
            "repair_camera_detail": self.win_repair_camera_detail,
            "repair_boundary_mask": self.win_repair_boundary,
            "repair_boundary_strength": self.win_repair_boundary_strength,
        }

    def windows_controls(self) -> list[QWidget]:
        return [self.processing_preset, *self.setting_widgets().values()]

    def set_windows_available(self, available: bool, reason: str = "") -> None:
        self._windows_available = bool(available)
        self._windows_reason = reason
        self._apply_enablement()

    def _apply_enablement(self) -> None:
        # Controls edit durable desired state even when the selected endpoint
        # is offline. Reconciliation, not widget enablement, represents reachability.
        editable = True
        for control in self.windows_controls():
            control.setEnabled(editable)
        detail = next(
            (text for key, _label, text in PROCESSOR_TARGETS if key == self._target),
            "",
        )
        specification = PROCESSOR_SPECS.get(self._target)
        if specification is not None:
            self.processor_model.setText(
                f"{specification.model} · {specification.backend}\n"
                f"{specification.detail}"
            )
        if self._target == TARGET_WINDOWS and not self._windows_available:
            self.target_note.setText(
                f"{detail} Windows is currently unreachable "
                f"({self._windows_reason or 'no response'}). Changes stay saved "
                "locally and will be pushed automatically when it returns."
            )
        elif self._target == TARGET_ARCH:
            self.target_note.setText(
                f"{detail} This is a development checkpoint; transport health "
                "must not be read as identity-quality verification."
            )
        else:
            self.target_note.setText(detail)

    def set_processor(self, target: str) -> None:
        """Select a processor without echoing a user-change signal."""
        index = self.processor_target.findData(target)
        if index < 0:
            return
        self._loading = True
        try:
            self.processor_target.setCurrentIndex(index)
            self._target = target
            self._apply_enablement()
        finally:
            self._loading = False

    def apply_windows_config(
        self, values: dict[str, Any], reset_preset: bool = True
    ) -> None:
        """Load processor-reported values without emitting change events."""
        self._loading = True
        try:
            if "opacity" in values:
                self.win_opacity.setValue(round(float(values["opacity"]) * 100))
            if "sharpness" in values:
                self.win_sharpness.setValue(round(float(values["sharpness"]) * 10))
            if "mouth_mask_size" in values:
                self.win_mouth_mask.setValue(round(float(values["mouth_mask_size"])))
            if "color_match_strength" in values:
                self.win_color_match.setValue(
                    round(float(values["color_match_strength"]) * 100)
                )
            if "interpolation_weight" in values:
                self.win_interpolation_weight.setValue(
                    round(float(values["interpolation_weight"]) * 100)
                )
            if "detection_interval" in values:
                self.win_detection_interval.setValue(int(values["detection_interval"]))
            if "tracking_smoothing" in values:
                self.win_tracking_smoothing.setValue(
                    round(float(values["tracking_smoothing"]) * 100)
                )
            if "tracking_grace_frames" in values:
                self.win_tracking_grace.setValue(int(values["tracking_grace_frames"]))
            if "minimum_detection_score" in values:
                self.win_detection_score.setValue(
                    round(float(values["minimum_detection_score"]) * 100)
                )
            if "minimum_face_size" in values:
                self.win_minimum_face_size.setValue(int(values["minimum_face_size"]))
            for field, widget, scale in (
                ("repair_hf_strength", self.win_repair_hf, 100),
                ("repair_checkerboard", self.win_repair_checkerboard, 100),
                ("repair_wavelet", self.win_repair_wavelet, 100),
                ("repair_camera_detail", self.win_repair_camera_detail, 100),
                (
                    "repair_boundary_strength",
                    self.win_repair_boundary_strength,
                    100,
                ),
            ):
                if field in values:
                    widget.setValue(round(float(values[field]) * scale))
            checkboxes = {
                "many_faces": self.win_many_faces,
                "show_fps": self.win_show_fps,
                "enable_interpolation": self.win_enable_interpolation,
                "quality_auto_correct": self.win_quality_auto_correct,
                "tracking_enabled": self.win_tracking_enabled,
                "repair_boundary_mask": self.win_repair_boundary,
            }
            for field, checkbox in checkboxes.items():
                if field in values:
                    checkbox.setChecked(bool(values[field]))
            for field, widget in (
                ("enhancer", self.win_enhancer),
                ("quality_mode", self.win_quality_mode),
            ):
                if field in values:
                    index = widget.findData(str(values[field]))
                    if index >= 0:
                        widget.setCurrentIndex(index)
            if reset_preset:
                self.processing_preset.setCurrentIndex(0)
        except (TypeError, ValueError):
            # A malformed value must not take the page down; the next refresh
            # reloads from the processor.
            pass
        finally:
            self._loading = False

    def set_preset_index(self, index: int) -> None:
        self._loading = True
        try:
            self.processing_preset.setCurrentIndex(index)
        finally:
            self._loading = False

    def show_source_image(self, data: bytes, filename: str) -> None:
        image = QImage.fromData(data)
        if image.isNull():
            self.source_preview.clear()
            self.source_preview.setText("Preview unavailable")
        else:
            self.source_preview.setPixmap(
                QPixmap.fromImage(image).scaled(
                    PREVIEW_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.source_name.setText(filename)

    def set_source_status(self, text: str) -> None:
        self.source_status.setText(text)

    def set_source_controls_enabled(self, enabled: bool) -> None:
        self.upload_source_button.setEnabled(enabled)
        for button in self.history_buttons:
            button.setEnabled(enabled)

    def rebuild_history(self, entries: list[Any], active_id: str | None) -> None:
        """Rebuild the recent-picture strip; called only when history changes."""
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.history_buttons = []
        if not entries:
            empty = QLabel("Pictures you apply will appear here.")
            empty.setObjectName("hintText")
            self.history_layout.addWidget(empty)
            self.history_layout.addStretch(1)
            return
        for entry in entries:
            button = QToolButton()
            button.setObjectName("historyItem")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setChecked(entry.identifier == active_id)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            short = (
                entry.filename
                if len(entry.filename) <= 14
                else f"{entry.filename[:11]}…"
            )
            button.setText(short)
            button.setToolTip(f"Apply {entry.filename}")
            button.setAccessibleName(f"Recent source picture {entry.filename}")
            pixmap = thumbnail(str(Path(entry.cache_path)), HISTORY_ICON)
            if not pixmap.isNull():
                button.setIcon(QIcon(pixmap))
                button.setIconSize(HISTORY_ICON)
            button.setFixedSize(88, 86)
            button.clicked.connect(
                lambda _checked=False, item_id=entry.identifier: (
                    self.historyPictureRequested.emit(item_id)
                )
            )
            self.history_layout.addWidget(button)
            self.history_buttons.append(button)
        self.history_layout.addStretch(1)

    # ------------------------------------------------------------------ render

    def render(self, view: ManagerView) -> None:
        processor = view.processor
        identity = view.identity
        self.win_swapper_status.setText(
            processor.windows_active_model
            if self._target == TARGET_WINDOWS
            else f"{processor.local_model} / {processor.local_backend} · "
            f"{processor.local_checkpoint}"
        )
        self.set_windows_available(
            processor.windows_reachable, processor.windows_detail
        )

        windows_states = {
            "applied": ("APPLIED", "running"),
            "pending": ("SYNCING", "working"),
            "unverified": ("UNVERIFIED", "warning"),
            "missing": ("NOT SET", "warning"),
            "unknown": ("UNKNOWN", "unknown"),
        }
        text, state = windows_states.get(
            identity.windows_state, ("UNKNOWN", "unknown")
        )
        self.windows_source_pill.set_state(text, state)
        self.windows_source_detail.setText(identity.windows_detail)

        local_states = {
            "unavailable": ("SERVICE NOT RUNNING", "stopped"),
            "restart-required": ("NOT LOADED YET", "warning"),
            "applying": ("APPLYING", "working"),
            "unconfirmed": ("NOT REPORTED", "unknown"),
        }
        text, state = local_states.get(identity.local_state, ("UNKNOWN", "unknown"))
        self.local_source_pill.set_state(text, state)
        self.local_source_detail.setText(identity.local_detail)

        self.checkpoint_metric.set_value(
            processor.local_checkpoint.upper()
            if processor.local_running
            else "LOCAL MODEL NOT RUNNING",
            processor.local_detail
            if processor.local_running
            else "Qualification applies to the local checkpoint; the Windows "
            "processor reports its own model separately.",
        )
        if not processor.local_running:
            self.effect_metric.set_value(
                "NOT MEASURED",
                "Pixel evidence is produced by the local processor only.",
            )
        elif processor.visual_effect_confirmed:
            self.effect_metric.set_value("OBSERVED", processor.identity_detail)
        else:
            self.effect_metric.set_value("NOT OBSERVED", processor.identity_detail)
        self.verification_metric.set_value(
            "VERIFIED" if processor.identity_verified else "NOT PERFORMED",
            "This manager never measures identity similarity. A changed mask "
            "ring or an invoked processor is not proof of a swap.",
        )
