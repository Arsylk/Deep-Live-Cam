#!/usr/bin/env python3
"""Semantic camera input selection and capture-owner configuration.

Transport slots are an implementation detail.  This page presents the things
a person can actually point at the processor while preserving the long-lived
camera owners and output identities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from ..desired_state import INPUT_ASSEMBLER, INPUT_PRERECORDED, INPUT_SPECS
from ..widgets import (
    Card,
    ElidedLabel,
    FramingPreview,
    ResponsiveCardGrid,
    StatusPill,
    note_label,
)
from .routing import RoutingPage


class InputPage(RoutingPage):
    """Input tab with one semantic choice and capability-driven settings."""

    inputSelected = Signal(str)
    prerecordedVideoSelected = Signal(str)
    prerecordedModeChanged = Signal(str)
    # offset_x, offset_y (px), zoom (float)
    prerecordedAdjustChanged = Signal(int, int, float)
    prerecordedPauseToggled = Signal()
    prerecordedSeekRequested = Signal(float)  # seconds
    # Assembler input: library dir, composed token list, assemble request.
    assemblerLibChanged = Signal(str)
    assemblerTokensChanged = Signal(list)
    assemblerAssembleRequested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading_input = False
        self.input_buttons: dict[str, QRadioButton] = {}
        self._prerecorded_path_label: QLabel | None = None
        self._prerecorded_choose_btn: QPushButton | None = None
        self._prerecorded_mode_combo: QComboBox | None = None
        self._prerecorded_offx_slider: QSlider | None = None
        self._prerecorded_offy_slider: QSlider | None = None
        self._prerecorded_zoom_spin: QDoubleSpinBox | None = None
        self._prerecorded_preview: FramingPreview | None = None
        self._prerecorded_grid_check = None
        self._prerecorded_play_btn: QPushButton | None = None
        self._prerecorded_seek_slider: QSlider | None = None
        self._prerecorded_time_label: QLabel | None = None
        self._prerecorded_duration = 0.0
        self._prerecorded_seeking = False
        self._prerecorded_adjust_loading = False
        self._assembler_lib_label: QLabel | None = None
        self._assembler_token_list: QListWidget | None = None
        self._assembler_digits_edit: QLineEdit | None = None
        self._assembler_idle_spin = None
        self._assembler_status: QLabel | None = None
        self._assembler_available: set[str] = set()

        heading = self.findChild(QLabel, "pageTitle")
        if heading is not None:
            heading.setText("Input")
        subtitles = self.findChildren(QLabel, "pageSubtitle")
        if subtitles:
            subtitles[0].setText(
                "Select the Arch webcam, one phone lens, or a prerecorded video, "
                "then configure that source in the settings card below."
            )

        # Retain RoutingPage's well-tested camera-control generator, but remove
        # its transport-slot and receiver-policy presentation from this UI.
        self.top_grid.setVisible(False)
        self.route_card.setVisible(False)
        self.camera_card.title.setText("INPUT DEVICE SETTINGS")
        self.camera_card.detail.setText(
            "Controls for the selected source. The manager never opens /dev/video* "
            "or Camera2 itself."
        )

        scroll = self.findChild(QScrollArea, "workspaceScroll")
        panel = scroll.widget() if scroll is not None else None
        layout = panel.layout() if panel is not None else None
        if isinstance(layout, QVBoxLayout):
            layout.insertWidget(1, self._build_input_card())

    def _build_input_card(self) -> Card:
        card = Card(
            "INPUT DEVICE",
            "Choose the physical source by meaning, not by transport port. "
            "Front and back share one phone transport and only change the "
            "Camera2 lens request.",
            pill=True,
        )
        self.input_state_pill = card.pill or StatusPill()
        self.input_group = QButtonGroup(self)
        self.input_group.setExclusive(True)
        cards: list[QWidget] = []
        for key, specification in INPUT_SPECS.items():
            option = Card(specification.label.upper(), specification.detail)
            button = QRadioButton("Use this input")
            button.setAccessibleName(f"Select {specification.label} input")
            button.toggled.connect(
                lambda checked, input_key=key: self._input_toggled(
                    input_key, checked
                )
            )
            self.input_group.addButton(button)
            self.input_buttons[key] = button
            option.add(button)
            endpoint = QLabel(
                f"Owner: {specification.stack} · route: {specification.device_id}"
            )
            endpoint.setObjectName("hintText")
            endpoint.setWordWrap(True)
            option.add(endpoint)
            cards.append(option)
        self.input_grid = ResponsiveCardGrid(minimum_card_width=265, spacing=9)
        self.input_grid.set_cards(cards)
        card.add(self.input_grid)
        self.input_status = note_label(
            "Selecting an input does not start or stop a camera owner, provider, "
            "virtual device, or output sink.",
            "info",
        )
        card.add(self.input_status)
        return card

    def rebuild_camera_controls(
        self,
        *,
        device_id: str,
        label: str,
        stack: str,
        schema: dict[str, Any],
        defaults: dict[str, Any],
    ) -> None:
        """Keep front/back cards as the sole semantic lens selector."""
        filtered = dict(schema)
        filtered["controls"] = [
            control
            for control in schema.get("controls", [])
            if control.get("key") != "lens_facing"
        ]
        super().rebuild_camera_controls(
            device_id=device_id,
            label=label,
            stack=stack,
            schema=filtered,
            defaults=defaults,
        )

    def rebuild_prerecorded_controls(
        self,
        *,
        path: str | None = None,
        mode: str = "loop",
        offset_x: int = 0,
        offset_y: int = 0,
        zoom: float = 1.0,
    ) -> None:
        """Populate the INPUT DEVICE SETTINGS card with prerecorded controls."""
        self._adapter_key = ("local-prerecorded", "prerecorded-relay")
        while self.camera_groups_layout.count():
            item = self.camera_groups_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.camera_controls = {}

        self.camera_title.setText(
            "Prerecorded video · loop-replayed through the receiver as a camera source."
        )

        self._prerecorded_path_label = ElidedLabel(
            f"Selected: {Path(path).name}" if path else "No video selected"
        )
        self._prerecorded_path_label.setObjectName("hintText")
        self.camera_groups_layout.addWidget(self._prerecorded_path_label)

        choose_btn = QPushButton("Choose video…")
        choose_btn.setToolTip("Select a recorded or rendered MP4")
        choose_btn.clicked.connect(self._choose_prerecorded_video)
        self.camera_groups_layout.addWidget(choose_btn)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        mode_label = QLabel("Playback:")
        mode_label.setObjectName("hintText")
        self._prerecorded_mode_combo = QComboBox()
        self._prerecorded_mode_combo.addItems(["loop", "once", "freeze"])
        self._prerecorded_mode_combo.setCurrentText(mode)
        self._prerecorded_mode_combo.setToolTip(
            "loop: repeat forever; once: play and stop; freeze: hold last frame"
        )
        self._prerecorded_mode_combo.currentTextChanged.connect(
            self.prerecordedModeChanged.emit
        )
        self._prerecorded_mode_combo.wheelEvent = lambda event: event.ignore()
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self._prerecorded_mode_combo, 1)
        self.camera_groups_layout.addWidget(mode_row)

        self._build_prerecorded_adjust(offset_x, offset_y, zoom)

        self.camera_apply.setEnabled(False)
        self.camera_state_pill.set_state("READY", "running")
        self.camera_status.setText(
            "Choose a video and a playback mode. The receiver will treat the looped "
            "stream as a normal camera input."
        )
        self.camera_measured.setText("")

    def _choose_prerecorded_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Prerecorded Video",
            "/var/lib/deep-live-cam/renders",
            "MP4 videos (*.mp4)",
        )
        if path:
            self.prerecordedVideoSelected.emit(path)

    # --- assembler input -----------------------------------------------------

    def rebuild_assembler_controls(
        self,
        *,
        lib: str | None = None,
        tokens: list[str] | None = None,
        mode: str = "loop",
        offset_x: int = 0,
        offset_y: int = 0,
        zoom: float = 1.0,
    ) -> None:
        """INPUT DEVICE SETTINGS card for the assembler input.

        Top half: library picker + segment availability + prompt-sequence
        composer.  Bottom half: the exact same framing preview (grid, drag,
        zoom) and transport (play/pause, seek, position) controls the
        prerecorded input uses -- the assembled video is played through the
        identical file_relay machinery.
        """
        self._adapter_key = ("local-prerecorded", "prerecorded-relay")
        while self.camera_groups_layout.count():
            item = self.camera_groups_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.camera_controls = {}

        self.camera_title.setText(
            "Assembler · compose a prompt sequence from the pre-rendered puppet "
            "library and replay it as a camera source."
        )

        # Library directory row.
        self._assembler_lib_label = ElidedLabel(
            f"Library: {lib}" if lib else "No library directory selected"
        )
        self._assembler_lib_label.setObjectName("hintText")
        self.camera_groups_layout.addWidget(self._assembler_lib_label)

        lib_btn = QPushButton("Choose library…")
        lib_btn.setToolTip(
            "Directory of concat-safe segments produced by the puppet "
            "recorder + library builder"
        )
        lib_btn.clicked.connect(self._choose_assembler_lib)
        self.camera_groups_layout.addWidget(lib_btn)

        # Segment availability, parsed from the directory.
        self._assembler_available: set[str] = set()
        if lib:
            lib_path = Path(lib)
            if lib_path.is_dir():
                self._assembler_available = {
                    p.stem for p in lib_path.glob("*.mp4")
                }
        missing = [
            name for name in ("turn_left", "turn_right", "look_up",
                              "look_down", "blink", "idle_0.5")
            if self._assembler_available
            and name not in self._assembler_available
        ]
        digits_present = [
            str(d) for d in range(10)
            if f"digits_{d}" in self._assembler_available
        ]
        if not self._assembler_available:
            availability = "Library not found or empty."
        else:
            availability = (
                f"segments: {len(self._assembler_available)} · "
                f"digits: {','.join(digits_present) or 'none'}"
                + (f" · MISSING: {','.join(missing)}" if missing else "")
            )
        self._assembler_avail_label = QLabel(availability)
        self._assembler_avail_label.setObjectName("hintText")
        self._assembler_avail_label.setWordWrap(True)
        self.camera_groups_layout.addWidget(self._assembler_avail_label)

        # Sequence composer.
        heading = QLabel("Prompt sequence")
        heading.setObjectName("hintText")
        self.camera_groups_layout.addWidget(heading)

        self._assembler_token_list = QListWidget()
        self._assembler_token_list.setMaximumHeight(110)
        self._assembler_token_list.setToolTip(
            "Ordered prompt tokens; assembled top to bottom"
        )
        for token in tokens or []:
            self._assembler_token_list.addItem(QListWidgetItem(token))
        self.camera_groups_layout.addWidget(self._assembler_token_list)

        actions_row = QWidget()
        actions = QHBoxLayout(actions_row)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)

        def add_action_button(text: str, token: str) -> None:
            btn = QPushButton(text)
            btn.setToolTip(f"Append '{token}'")
            btn.setEnabled(
                not self._assembler_available or token in self._assembler_available
            )
            btn.clicked.connect(lambda _=False, t=token: self._append_token(t))
            actions.addWidget(btn)

        add_action_button("Turn L", "turn_left")
        add_action_button("Turn R", "turn_right")
        add_action_button("Up", "look_up")
        add_action_button("Down", "look_down")
        add_action_button("Blink", "blink")
        self.camera_groups_layout.addWidget(actions_row)

        say_row = QWidget()
        say = QHBoxLayout(say_row)
        say.setContentsMargins(0, 0, 0, 0)
        say.setSpacing(4)
        self._assembler_digits_edit = QLineEdit()
        self._assembler_digits_edit.setPlaceholderText("digits e.g. 4-7-2")
        self._assembler_digits_edit.setToolTip(
            "Digits the puppet will say, hyphen separated"
        )
        say_btn = QPushButton("Say")
        say_btn.clicked.connect(self._append_say)
        idle_spin = QDoubleSpinBox()
        idle_spin.setRange(0.5, 8.0)
        idle_spin.setSingleStep(0.5)
        idle_spin.setValue(1.0)
        idle_spin.setSuffix("s")
        idle_spin.setDecimals(1)
        idle_spin.wheelEvent = lambda event: event.ignore()
        self._assembler_idle_spin = idle_spin
        idle_btn = QPushButton("Idle")
        idle_btn.clicked.connect(
            lambda: self._append_token(f"neutral {idle_spin.value():g}s")
        )
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove the selected token")
        remove_btn.clicked.connect(self._remove_selected_tokens)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._assembler_token_list.clear)
        clear_btn.clicked.connect(
            lambda: self.assemblerTokensChanged.emit(self.current_tokens())
        )
        say.addWidget(self._assembler_digits_edit, 1)
        say.addWidget(say_btn)
        say.addWidget(idle_spin)
        say.addWidget(idle_btn)
        say.addWidget(remove_btn)
        say.addWidget(clear_btn)
        self.camera_groups_layout.addWidget(say_row)

        self._assembler_token_list.itemChanged.connect(
            lambda: self.assemblerTokensChanged.emit(self.current_tokens())
        )

        assemble_btn = QPushButton("Assemble && load")
        assemble_btn.setToolTip(
            "Concatenate the sequence into one video and switch the camera "
            "input to it"
        )
        assemble_btn.clicked.connect(
            lambda: self.assemblerAssembleRequested.emit(self.current_tokens())
        )
        self.camera_groups_layout.addWidget(assemble_btn)

        self._assembler_status = QLabel(" ")
        self._assembler_status.setObjectName("hintText")
        self._assembler_status.setWordWrap(True)
        self.camera_groups_layout.addWidget(self._assembler_status)

        # Same framing preview + transport as the prerecorded input.
        self._prerecorded_mode_combo = None  # mode fixed to loop for assembler
        self._build_prerecorded_adjust(offset_x, offset_y, zoom)
        if self._prerecorded_play_btn is not None:
            self._prerecorded_play_btn.setText("Pause")

        self.camera_apply.setEnabled(False)
        self.camera_state_pill.set_state("READY", "running")
        self.camera_status.setText(
            "Compose a sequence, press Assemble & load. The receiver treats "
            "the result like any prerecorded camera input."
        )
        self.camera_measured.setText("")

    def current_tokens(self) -> list[str]:
        if self._assembler_token_list is None:
            return []
        return [
            self._assembler_token_list.item(i).text()
            for i in range(self._assembler_token_list.count())
        ]

    def _append_token(self, token: str) -> None:
        if self._assembler_token_list is None:
            return
        self._assembler_token_list.addItem(QListWidgetItem(token))
        self.assemblerTokensChanged.emit(self.current_tokens())

    def _append_say(self) -> None:
        digits = (self._assembler_digits_edit.text() if
                  self._assembler_digits_edit is not None else "").strip()
        if not digits:
            return
        self._append_token(f"say {digits}")
        if self._assembler_digits_edit is not None:
            self._assembler_digits_edit.clear()

    def _remove_selected_tokens(self) -> None:
        if self._assembler_token_list is None:
            return
        for item in self._assembler_token_list.selectedItems():
            self._assembler_token_list.takeItem(
                self._assembler_token_list.row(item)
            )
        self.assemblerTokensChanged.emit(self.current_tokens())

    def _choose_assembler_lib(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Puppet Library Directory",
            "/var/lib/deep-live-cam",
        )
        if path:
            self.assemblerLibChanged.emit(path)

    def set_assembler_lib(self, path: str | None) -> None:
        if self._assembler_lib_label is not None and path:
            self._assembler_lib_label.setText(f"Library: {path}")

    def set_assembler_status(self, text: str) -> None:
        if self._assembler_status is not None:
            self._assembler_status.setText(text)

    def set_prerecorded_path(self, path: str | None) -> None:
        if self._prerecorded_path_label is not None:
            self._prerecorded_path_label.setText(
                f"Selected: {Path(path).name}" if path else "No video selected"
            )

    def set_prerecorded_mode(self, mode: str) -> None:
        if self._prerecorded_mode_combo is not None:
            self._prerecorded_mode_combo.setCurrentText(mode)

    def _build_prerecorded_adjust(
        self, offset_x: int, offset_y: int, zoom: float
    ) -> None:
        """Offset (left/right, up/down) and zoom controls for the video framing.

        Offsets shift the video inside the locked output box; black fills any
        area no longer covered by the source.  Zoom scales the video about its
        centre.  Values stream live to the receiver as they change.
        """
        self._prerecorded_adjust_loading = True

        heading = QLabel("Framing")
        heading.setObjectName("hintText")
        self.camera_groups_layout.addWidget(heading)

        # Live what-you-see preview of the framed video.  Dragging it pans the
        # video and keeps the sliders in sync.
        self._prerecorded_preview = FramingPreview(1280, 720)
        self._prerecorded_preview.set_offset(int(offset_x), int(offset_y))
        self._prerecorded_preview.offsetDragged.connect(
            self._on_preview_dragged
        )
        self._prerecorded_preview.offsetPreview.connect(
            self._on_preview_dragging
        )
        self.camera_groups_layout.addWidget(self._prerecorded_preview)

        drag_hint = QLabel("Drag the preview to reposition; black fills empty area.")
        drag_hint.setObjectName("hintText")
        drag_hint.setWordWrap(True)
        self.camera_groups_layout.addWidget(drag_hint)

        # Positioning grid toggle.
        self._prerecorded_grid_check = QCheckBox("Show positioning grid")
        self._prerecorded_grid_check.setChecked(True)
        self._prerecorded_grid_check.setToolTip(
            "Rule-of-thirds guides and a centre cross over the output box"
        )
        self._prerecorded_grid_check.toggled.connect(
            lambda on: self._prerecorded_preview.set_grid_visible(on)
            if self._prerecorded_preview is not None
            else None
        )
        self.camera_groups_layout.addWidget(self._prerecorded_grid_check)

        # Transport: play/pause + seek + time readout.
        transport = QWidget()
        t_layout = QHBoxLayout(transport)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(6)
        self._prerecorded_play_btn = QPushButton("Pause")
        self._prerecorded_play_btn.setToolTip("Play or pause the prerecorded video")
        self._prerecorded_play_btn.clicked.connect(self.prerecordedPauseToggled.emit)
        self._prerecorded_seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._prerecorded_seek_slider.setRange(0, 1000)
        self._prerecorded_seek_slider.setValue(0)
        self._prerecorded_seek_slider.setToolTip("Scrub to a position in the video")
        self._prerecorded_seek_slider.wheelEvent = lambda event: event.ignore()
        self._prerecorded_seek_slider.sliderPressed.connect(
            self._on_seek_pressed
        )
        self._prerecorded_seek_slider.sliderReleased.connect(
            self._on_seek_released
        )
        self._prerecorded_time_label = QLabel("0:00 / 0:00")
        self._prerecorded_time_label.setObjectName("hintText")
        t_layout.addWidget(self._prerecorded_play_btn)
        t_layout.addWidget(self._prerecorded_seek_slider, 1)
        t_layout.addWidget(self._prerecorded_time_label)
        self.camera_groups_layout.addWidget(transport)

        grid_row = QWidget()
        grid = QGridLayout(grid_row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        # Horizontal offset: negative = left, positive = right.
        offx_label = QLabel("Left / Right")
        offx_label.setObjectName("hintText")
        self._prerecorded_offx_slider = QSlider(Qt.Orientation.Horizontal)
        self._prerecorded_offx_slider.setRange(-960, 960)
        self._prerecorded_offx_slider.setValue(int(offset_x))
        self._prerecorded_offx_slider.setToolTip(
            "Shift the video horizontally; black fills the exposed edge"
        )
        self._prerecorded_offx_slider.wheelEvent = lambda event: event.ignore()
        self._prerecorded_offx_value = QLabel(f"{int(offset_x):+d}px")
        self._prerecorded_offx_value.setObjectName("hintText")
        self._prerecorded_offx_slider.valueChanged.connect(
            lambda value: (
                self._prerecorded_offx_value.setText(f"{value:+d}px"),
                self._emit_prerecorded_adjust(),
            )
        )
        grid.addWidget(offx_label, 0, 0)
        grid.addWidget(self._prerecorded_offx_slider, 0, 1)
        grid.addWidget(self._prerecorded_offx_value, 0, 2)

        # Vertical offset: negative = up, positive = down.
        offy_label = QLabel("Up / Down")
        offy_label.setObjectName("hintText")
        self._prerecorded_offy_slider = QSlider(Qt.Orientation.Horizontal)
        self._prerecorded_offy_slider.setRange(-540, 540)
        self._prerecorded_offy_slider.setValue(int(offset_y))
        self._prerecorded_offy_slider.setToolTip(
            "Shift the video vertically; black fills the exposed edge"
        )
        self._prerecorded_offy_slider.wheelEvent = lambda event: event.ignore()
        self._prerecorded_offy_value = QLabel(f"{int(offset_y):+d}px")
        self._prerecorded_offy_value.setObjectName("hintText")
        self._prerecorded_offy_slider.valueChanged.connect(
            lambda value: (
                self._prerecorded_offy_value.setText(f"{value:+d}px"),
                self._emit_prerecorded_adjust(),
            )
        )
        grid.addWidget(offy_label, 1, 0)
        grid.addWidget(self._prerecorded_offy_slider, 1, 1)
        grid.addWidget(self._prerecorded_offy_value, 1, 2)

        # Zoom about centre.
        zoom_label = QLabel("Zoom")
        zoom_label.setObjectName("hintText")
        self._prerecorded_zoom_spin = QDoubleSpinBox()
        self._prerecorded_zoom_spin.setRange(0.25, 4.0)
        self._prerecorded_zoom_spin.setSingleStep(0.05)
        self._prerecorded_zoom_spin.setDecimals(2)
        self._prerecorded_zoom_spin.setValue(float(zoom))
        self._prerecorded_zoom_spin.setSuffix("x")
        self._prerecorded_zoom_spin.setToolTip(
            "Scale the video about its centre; below 1.0 shows black borders"
        )
        self._prerecorded_zoom_spin.wheelEvent = lambda event: event.ignore()
        self._prerecorded_zoom_spin.valueChanged.connect(
            lambda _value: self._emit_prerecorded_adjust()
        )
        grid.addWidget(zoom_label, 2, 0)
        grid.addWidget(self._prerecorded_zoom_spin, 2, 1, 1, 2)

        self.camera_groups_layout.addWidget(grid_row)

        reset_btn = QPushButton("Reset framing")
        reset_btn.setToolTip("Recentre the video and reset zoom to 1.0x")
        reset_btn.clicked.connect(self._reset_prerecorded_adjust)
        self.camera_groups_layout.addWidget(reset_btn)

        self._prerecorded_adjust_loading = False

    def _on_preview_dragging(self, offset_x: int, offset_y: int) -> None:
        """Live sync during a drag.

        The receiver applies framing over zmq now (no decoder restart), so it is
        safe to push every intermediate drag position: the receiver's throttled
        poll turns each into a live crop update and the output pans smoothly.
        """
        if (
            self._prerecorded_offx_slider is None
            or self._prerecorded_offy_slider is None
        ):
            return
        clamped_x = max(
            self._prerecorded_offx_slider.minimum(),
            min(self._prerecorded_offx_slider.maximum(), int(offset_x)),
        )
        clamped_y = max(
            self._prerecorded_offy_slider.minimum(),
            min(self._prerecorded_offy_slider.maximum(), int(offset_y)),
        )
        self._prerecorded_adjust_loading = True
        self._prerecorded_offx_slider.setValue(clamped_x)
        self._prerecorded_offx_value.setText(f"{clamped_x:+d}px")
        self._prerecorded_offy_slider.setValue(clamped_y)
        self._prerecorded_offy_value.setText(f"{clamped_y:+d}px")
        self._prerecorded_adjust_loading = False
        self._emit_prerecorded_adjust()

    def _on_preview_dragged(self, offset_x: int, offset_y: int) -> None:
        """Drag released: commit the final offset to the receiver once."""
        if (
            self._prerecorded_offx_slider is None
            or self._prerecorded_offy_slider is None
        ):
            return
        # Clamp to slider bounds so the value the receiver gets matches the UI.
        clamped_x = max(
            self._prerecorded_offx_slider.minimum(),
            min(self._prerecorded_offx_slider.maximum(), int(offset_x)),
        )
        clamped_y = max(
            self._prerecorded_offy_slider.minimum(),
            min(self._prerecorded_offy_slider.maximum(), int(offset_y)),
        )
        # Setting the sliders triggers their valueChanged handlers, which emit
        # the framing change once; guard the loading flag so we do not double-
        # emit while both sliders update.
        self._prerecorded_adjust_loading = True
        self._prerecorded_offx_slider.setValue(clamped_x)
        self._prerecorded_offx_value.setText(f"{clamped_x:+d}px")
        self._prerecorded_offy_slider.setValue(clamped_y)
        self._prerecorded_offy_value.setText(f"{clamped_y:+d}px")
        self._prerecorded_adjust_loading = False
        self._emit_prerecorded_adjust()

    def _emit_prerecorded_adjust(self) -> None:
        if self._prerecorded_adjust_loading:
            return
        if (
            self._prerecorded_offx_slider is None
            or self._prerecorded_offy_slider is None
            or self._prerecorded_zoom_spin is None
        ):
            return
        if self._prerecorded_preview is not None:
            self._prerecorded_preview.set_offset(
                self._prerecorded_offx_slider.value(),
                self._prerecorded_offy_slider.value(),
            )
        self.prerecordedAdjustChanged.emit(
            self._prerecorded_offx_slider.value(),
            self._prerecorded_offy_slider.value(),
            self._prerecorded_zoom_spin.value(),
        )

    def _reset_prerecorded_adjust(self) -> None:
        self._prerecorded_adjust_loading = True
        if self._prerecorded_offx_slider is not None:
            self._prerecorded_offx_slider.setValue(0)
            self._prerecorded_offx_value.setText("+0px")
        if self._prerecorded_offy_slider is not None:
            self._prerecorded_offy_slider.setValue(0)
            self._prerecorded_offy_value.setText("+0px")
        if self._prerecorded_zoom_spin is not None:
            self._prerecorded_zoom_spin.setValue(1.0)
        if self._prerecorded_preview is not None:
            self._prerecorded_preview.set_offset(0, 0)
        self._prerecorded_adjust_loading = False
        self._emit_prerecorded_adjust()

    def set_prerecorded_preview_frame(self, image: object) -> None:
        """Push a decoded framing-preview frame onto the surface (if present)."""
        if self._prerecorded_preview is not None:
            self._prerecorded_preview.set_image(image)

    def clear_prerecorded_preview(self, message: str | None = None) -> None:
        if self._prerecorded_preview is not None:
            self._prerecorded_preview.clear_image(message)

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def set_prerecorded_paused(self, paused: bool) -> None:
        if self._prerecorded_play_btn is not None:
            self._prerecorded_play_btn.setText("Play" if paused else "Pause")

    def set_prerecorded_playback(
        self,
        *,
        position: float | None,
        duration: float | None,
        paused: bool,
    ) -> None:
        """Update the transport bar from the receiver's live playback state."""
        self.set_prerecorded_paused(paused)
        if duration and duration > 0:
            self._prerecorded_duration = float(duration)
        # Do not fight the user while they are scrubbing.
        if self._prerecorded_seeking or self._prerecorded_seek_slider is None:
            return
        if position is not None and self._prerecorded_duration > 0:
            frac = max(0.0, min(1.0, position / self._prerecorded_duration))
            self._prerecorded_seek_slider.setValue(int(frac * 1000))
        if self._prerecorded_time_label is not None:
            pos_txt = self._format_time(position or 0.0)
            dur_txt = self._format_time(self._prerecorded_duration)
            self._prerecorded_time_label.setText(f"{pos_txt} / {dur_txt}")

    def _on_seek_pressed(self) -> None:
        self._prerecorded_seeking = True

    def _on_seek_released(self) -> None:
        self._prerecorded_seeking = False
        if self._prerecorded_seek_slider is None or self._prerecorded_duration <= 0:
            return
        frac = self._prerecorded_seek_slider.value() / 1000.0
        self.prerecordedSeekRequested.emit(frac * self._prerecorded_duration)

    def _input_toggled(self, key: str, checked: bool) -> None:
        if checked and not self._loading_input:
            self.inputSelected.emit(key)

    def set_input(self, key: str, *, status: str = "desired") -> None:
        button = self.input_buttons.get(key)
        if button is None:
            return
        self._loading_input = True
        try:
            button.setChecked(True)
        finally:
            self._loading_input = False
        specification = INPUT_SPECS[key]
        if status == "switching":
            self.input_state_pill.set_state("APPLYING", "working")
        elif status == "offline":
            self.input_state_pill.set_state("SAVED · PENDING", "warning")
        elif status == "failed":
            self.input_state_pill.set_state("SETTINGS FAILED", "failed")
        elif status == "mismatch":
            self.input_state_pill.set_state("LENS SYNC PENDING", "warning")
        else:
            self.input_state_pill.set_state("SELECTED", "running")
        self.input_status.setText(
            f"{specification.label} · {specification.detail}. The desired choice "
            "is durable and is reconciled to a processor when it is available."
        )


__all__ = ["InputPage"]
