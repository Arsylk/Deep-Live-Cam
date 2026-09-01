#!/usr/bin/env python3
"""Cameras and routing: three independent decisions, never one selector.

1. Which of the five client slots Windows processes.
2. What the stable Arch system camera publishes.
3. Which capture owner this page is configuring.

Selecting a slot or a policy swaps already-owned frame queues.  Nothing on this
page starts, stops, or reopens a camera device, a sink, or a service.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from camera_profiles import (
    ARCH_CAMERA_PROFILES,
    CUSTOM_CAMERA_PROFILE,
    DEFAULT_CAMERA_PROFILE,
    camera_profile,
    profile_live_values,
)

from ..contracts import (
    SCOPE_CAPTURE_OWNER,
    SLOT_COUNT,
    SYSTEM_CAMERA_POLICIES,
    group_camera_controls,
)
from ..viewmodel import ManagerView
from ..widgets import (
    Card,
    ResponsiveCardGrid,
    SlotCard,
    StatusPill,
    TopologyNode,
    ValueSlider,
    note_label,
    page_heading,
    scrollable,
    setting_label,
)



def measured_default_summary() -> str:
    """Describe the measured Arch default without duplicating its values."""
    profile = camera_profile(DEFAULT_CAMERA_PROFILE)
    capture = profile["capture"]
    controls = profile["controls"]
    return (
        f"Measured default '{DEFAULT_CAMERA_PROFILE}': "
        f"{str(capture['input_format']).upper()} {capture['width']}×"
        f"{capture['height']} at {capture['fps']} FPS · brightness "
        f"{controls['brightness']}, contrast {controls['contrast']}, saturation "
        f"{controls['saturation']}, hue {controls['hue']}, gamma "
        f"{controls['gamma']} · gain {controls['gain']}, sharpness "
        f"{controls['sharpness']}, backlight compensation "
        f"{controls['backlight_compensation']} · "
        f"{'50' if controls['power_line_frequency'] == 1 else '60'} Hz "
        "power-line filtering · automatic exposure "
        f"{'on' if controls['auto_exposure'] else 'off'}, dynamic frame rate "
        f"{'on' if controls['exposure_dynamic_framerate'] else 'off'} · "
        "automatic white balance "
        f"{'on' if controls['auto_white_balance'] else 'off'} · manual fallback "
        f"exposure {controls['exposure_time_absolute']} and white balance "
        f"{controls['white_balance_temperature']} K."
    )


class RoutingPage(QWidget):
    """Workspace 3: slot selection, output policy, and capture-owner settings."""

    slotSelected = Signal(str)
    policySelected = Signal(str)
    cameraControlChanged = Signal(str)
    cameraSaveRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading_policy = False
        self._adapter_key: tuple[str, str] | None = None
        self.camera_controls: dict[str, QWidget] = {}
        self.policy_buttons: dict[str, QRadioButton] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 18)
        layout.setSpacing(10)
        layout.addWidget(
            page_heading(
                "Cameras and routing",
                "Windows input selection, the stable system-camera policy, and "
                "capture-owner settings are three separate decisions.",
            )
        )

        self.slots_card = self._build_slots_card()
        self.policy_card = self._build_policy_card()
        self.route_card = self._build_route_card()
        self.camera_card = self._build_camera_card()
        # Side by side on a wide display, stacked at the 980-wide minimum,
        # rather than clipped by a scroll area that cannot scroll sideways.
        self.top_grid = ResponsiveCardGrid(minimum_card_width=440, spacing=10)
        self.top_grid.set_cards([self.slots_card, self.policy_card])
        layout.addWidget(self.top_grid)
        layout.addWidget(self.route_card)
        layout.addWidget(self.camera_card)
        layout.addStretch(1)
        outer.addWidget(scrollable(panel))

    # ------------------------------------------------------------------ build

    def _build_slots_card(self) -> Card:
        card = Card(
            "WINDOWS INPUT · FIVE ISOLATED SLOTS",
            "Only the selected client is processed on Windows. Every slot keeps "
            "its own fixed transport pair, so an unselected client stays "
            "usable on its own local fallback.",
            pill=True,
        )
        self.slot_grid = ResponsiveCardGrid(minimum_card_width=290, spacing=9)
        self.slot_cards = [SlotCard(index) for index in range(SLOT_COUNT)]
        for slot_card in self.slot_cards:
            slot_card.clicked.connect(
                lambda _checked=False, card=slot_card: self._slot_clicked(card)
            )
        self.slot_grid.set_cards(self.slot_cards)
        card.add(self.slot_grid)
        self.slot_contract = QLabel("Loading the five-slot registry from Windows…")
        self.slot_contract.setObjectName("hintText")
        self.slot_contract.setWordWrap(True)
        card.add(self.slot_contract)
        return card

    def _build_policy_card(self) -> Card:
        card = Card(
            "ARCH STABLE SYSTEM CAMERA",
            "Chooses what /dev/deep-live-cam publishes. The receiver "
            "swaps between frame queues it already owns.",
            pill=True,
        )
        self.policy_group = QButtonGroup(self)
        self.policy_group.setExclusive(True)
        for policy in SYSTEM_CAMERA_POLICIES:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(1)
            button = QRadioButton(policy.label)
            button.setToolTip(policy.summary)
            button.toggled.connect(
                lambda checked, key=policy.key: self._policy_toggled(key, checked)
            )
            summary = QLabel(policy.summary)
            summary.setObjectName("hintText")
            summary.setWordWrap(True)
            row_layout.addWidget(button)
            row_layout.addWidget(summary)
            self.policy_group.addButton(button)
            self.policy_buttons[policy.key] = button
            card.add(row)

        self.policy_state = QLabel("Reading the stable-camera receiver…")
        self.policy_state.setObjectName("sectionLabel")
        self.policy_state.setWordWrap(True)
        card.add(self.policy_state)
        self.policy_result = QLabel("")
        self.policy_result.setObjectName("infoBox")
        self.policy_result.setWordWrap(True)
        self.policy_result.setVisible(False)
        card.add(self.policy_result)
        self.policy_identity = note_label("", "info")
        card.add(self.policy_identity)
        card.add_stretch()
        return card

    def _build_route_card(self) -> Card:
        card = Card(
            "ROUTE AND TOPOLOGY",
            "Where frames are actually flowing right now, including bypass and "
            "fallback states.",
            pill=True,
        )
        self.nodes = {
            "android": TopologyNode("ANDROID", ""),
            "windows": TopologyNode("WINDOWS", ""),
            "arch": TopologyNode("ARCH", ""),
        }
        # A row on a wide display, a column at the minimum width. Three fixed
        # side-by-side nodes would push the whole page past a scroll area that
        # deliberately cannot scroll sideways.
        self.node_grid = ResponsiveCardGrid(minimum_card_width=250, spacing=8)
        self.node_grid.set_cards(
            [self.nodes["android"], self.nodes["windows"], self.nodes["arch"]]
        )
        card.add(self.node_grid)
        flow = QLabel("Android  →  Windows  →  Arch")
        flow.setObjectName("topologyLink")
        card.add(flow)
        self.route_summary = QLabel("")
        self.route_summary.setObjectName("hintText")
        self.route_summary.setWordWrap(True)
        card.add(self.route_summary)
        self.route_warning = note_label("", "note")
        card.add(self.route_warning)
        return card

    def _build_camera_card(self) -> Card:
        card = Card(
            "SELECTED CAPTURE OWNER",
            "Controls are generated from the capabilities the selected client "
            "advertises. This manager never opens the camera; the owning "
            "process applies each value.",
            pill=True,
        )
        self.camera_title = QLabel("Waiting for the selected client's capabilities…")
        self.camera_title.setObjectName("sectionLabel")
        self.camera_title.setWordWrap(True)
        card.add(self.camera_title)

        self.camera_groups = QWidget()
        self.camera_groups_layout = QVBoxLayout(self.camera_groups)
        self.camera_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_groups_layout.setSpacing(10)
        card.add(self.camera_groups)

        # The card header carries the only state pill for this card, so a
        # second one cannot drift out of step with it.
        self.camera_state_pill = card.pill or StatusPill("SAVED VALUES", "running")
        self.camera_state_pill.set_state("SAVED VALUES", "running")
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.camera_apply = QPushButton("Save current settings")
        self.camera_apply.setProperty("emphasis", "primary")
        self.camera_apply.setToolTip(
            "Persist the values above through the existing capture owner. "
            "Resolution is staged for the capture owner's next natural start; "
            "Save never reopens an active input."
        )
        self.camera_apply.clicked.connect(self.cameraSaveRequested.emit)
        actions.addStretch(1)
        actions.addWidget(self.camera_apply)
        card.add_layout(actions)

        self.camera_status = QLabel(
            "Slider and selector changes preview live through the capture "
            "owner and stay session-only until they are saved."
        )
        self.camera_status.setObjectName("hintText")
        self.camera_status.setWordWrap(True)
        card.add(self.camera_status)

        self.camera_effective = QLabel("")
        self.camera_effective.setObjectName("readout")
        self.camera_effective.setWordWrap(True)
        self.camera_effective.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.camera_effective.setVisible(False)
        card.add(self.camera_effective)

        self.camera_measured = QLabel(measured_default_summary())
        self.camera_measured.setObjectName("hintText")
        self.camera_measured.setWordWrap(True)
        card.add(self.camera_measured)
        return card

    # ---------------------------------------------------------------- signals

    def _slot_clicked(self, card: SlotCard) -> None:
        if card.device_id:
            self.slotSelected.emit(card.device_id)

    def _policy_toggled(self, key: str, checked: bool) -> None:
        if checked and not self._loading_policy:
            self.policySelected.emit(key)

    # ------------------------------------------------------- camera adapter UI

    def adapter_key(self) -> tuple[str, str] | None:
        return self._adapter_key

    def rebuild_camera_controls(
        self,
        *,
        device_id: str,
        label: str,
        stack: str,
        schema: dict[str, Any],
        defaults: dict[str, Any],
    ) -> None:
        """Generate one form per semantic group from an adapter's schema."""
        self._adapter_key = (device_id, stack)
        while self.camera_groups_layout.count():
            item = self.camera_groups_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.camera_controls = {}

        controls = schema.get("controls") or []
        self.camera_title.setText(
            f"{label} · {stack} · settings are applied by the capture owner, "
            "which keeps the operating-system camera registered throughout."
        )
        if not controls:
            empty = QLabel(
                "This client advertises no camera-control adapter. Its stream "
                "stays usable; there is simply nothing to configure from here."
            )
            empty.setObjectName("hintText")
            empty.setWordWrap(True)
            self.camera_groups_layout.addWidget(empty)
            self.camera_apply.setEnabled(False)
            self.camera_state_pill.set_state("NO ADAPTER", "unavailable")
            return

        self.camera_apply.setEnabled(True)
        for group, members in group_camera_controls(controls):
            card = Card(group.title, group.detail)
            form = card.add_form()
            for specification in members:
                key = str(specification["key"])
                widget = self._build_control(specification, defaults)
                self.camera_controls[key] = widget
                form.addRow(
                    setting_label(
                        str(specification["label"]),
                        SCOPE_CAPTURE_OWNER,
                        _control_hint(key),
                    ),
                    widget,
                )
            self.camera_groups_layout.addWidget(card)
        self.camera_measured.setVisible(stack == "arch-v4l2")

    def _build_control(
        self, specification: dict[str, Any], defaults: dict[str, Any]
    ) -> QWidget:
        key = str(specification["key"])
        default = defaults.get(key, specification.get("default"))
        kind = specification.get("kind")
        if kind == "boolean":
            widget: QWidget = QCheckBox()
            widget.setChecked(bool(default))
            widget.toggled.connect(
                lambda _checked, control_key=key: self._control_changed(control_key)
            )
        elif kind == "choice":
            picker = QComboBox()
            for choice in specification.get("choices", []):
                picker.addItem(_choice_label(key, choice), choice)
            index = picker.findData(default)
            if index >= 0:
                picker.setCurrentIndex(index)
            picker.currentIndexChanged.connect(
                lambda _index, control_key=key: self._control_changed(control_key)
            )
            widget = picker
        else:
            widget = ValueSlider(
                int(specification["minimum"]),
                int(specification["maximum"]),
                int(default),
            )
            widget.valueChanged.connect(
                lambda _value, control_key=key: self._control_changed(control_key)
            )
        return widget

    def _control_changed(self, key: str) -> None:
        stack = self._adapter_key[1] if self._adapter_key else ""
        if stack == "arch-v4l2":
            profile_widget = self.camera_controls.get("profile")
            if isinstance(profile_widget, QComboBox):
                if key == "profile":
                    profile = str(profile_widget.currentData() or "")
                    if profile in ARCH_CAMERA_PROFILES:
                        # A named profile is a bundle: load every component at
                        # once instead of leaving a half-applied mixture.
                        values = profile_live_values(profile)
                        values["profile"] = profile
                        self.set_camera_values(values)
                        self.camera_status.setText(
                            f"Loaded the measured '{profile}' profile for live "
                            "preview. Save to persist it."
                        )
                else:
                    custom = profile_widget.findData(CUSTOM_CAMERA_PROFILE)
                    if custom >= 0 and profile_widget.currentIndex() != custom:
                        profile_widget.blockSignals(True)
                        profile_widget.setCurrentIndex(custom)
                        profile_widget.blockSignals(False)
                        self.camera_status.setText(
                            "One component changed, so the profile is now "
                            "Custom. Save to persist these values."
                        )
        self.camera_state_pill.set_state("SESSION PREVIEW · NOT SAVED", "working")
        self.cameraControlChanged.emit(key)

    def camera_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, widget in self.camera_controls.items():
            if isinstance(widget, ValueSlider):
                values[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentData()
        return values

    def set_camera_values(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            widget = self.camera_controls.get(key)
            if widget is None:
                continue
            widget.blockSignals(True)
            try:
                if isinstance(widget, ValueSlider):
                    widget.setValue(int(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
            except (TypeError, ValueError):
                continue
            finally:
                widget.blockSignals(False)

    def set_camera_status(
        self, text: str, *, state: str, effective: str = ""
    ) -> None:
        self.camera_status.setText(text)
        self.camera_state_pill.set_state(
            {
                "saved": "SAVED VALUES",
                "preview": "SESSION PREVIEW · NOT SAVED",
                "working": "APPLYING…",
                "failed": "NOT APPLIED",
            }.get(state, state.upper()),
            {
                "saved": "running",
                "preview": "working",
                "working": "working",
                "failed": "failed",
            }.get(state, "unknown"),
        )
        self.camera_effective.setText(effective)
        self.camera_effective.setVisible(bool(effective))

    def set_camera_busy(self, busy: bool) -> None:
        self.camera_apply.setEnabled(not busy and bool(self.camera_controls))

    def set_policy_result(self, text: str, *, success: bool) -> None:
        self.policy_result.setText(text)
        self.policy_result.setObjectName("infoBox" if success else "noteBox")
        self.policy_result.setVisible(bool(text))
        self.policy_result.style().unpolish(self.policy_result)
        self.policy_result.style().polish(self.policy_result)

    def set_policy_enabled(self, enabled: bool) -> None:
        for button in self.policy_buttons.values():
            button.setEnabled(enabled)

    # ------------------------------------------------------------------ render

    def render(self, view: ManagerView) -> None:
        for card, slot in zip(self.slot_cards, view.slots):
            card.render(slot)
        if self.slots_card.pill is not None:
            self.slots_card.pill.set_state(
                "SWITCHING" if view.switching else (
                    "LIVE REGISTRY" if view.registry_live else "OFFLINE COPY"
                ),
                "working" if view.switching else (
                    "running" if view.registry_live else "unknown"
                ),
            )
        self.slot_contract.setText(
            f"Windows is processing {view.selected_device_id or 'no client'}. "
            f"Slot N sends on {10_000}+2N and its return is always the next "
            f"port. The shared selected stream is "
            f"{view.windows_host}:{view.selected_stream_port}."
            + (
                " Windows is unreachable, so selection is disabled and the last "
                "known registry is shown."
                if not view.registry_live
                else ""
            )
        )

        system = view.system_camera
        self._loading_policy = True
        try:
            button = self.policy_buttons.get(system.configured_policy)
            if button is not None and not button.isChecked():
                button.setChecked(True)
        finally:
            self._loading_policy = False
        if self.policy_card.pill is not None:
            self.policy_card.pill.set_state(system.state_text, system.state)
        self.policy_state.setText(
            f"Configured policy: {system.configured_label}\n"
            f"Actual active input: {system.active_label}\n"
            f"Stable nodes: {', '.join(system.devices) or 'not configured'}"
        )
        self.policy_identity.setText(system.identity_note)

        for key, node in self.nodes.items():
            match = next((item for item in view.route.nodes if item.key == key), None)
            if match is not None:
                node.render(match)
        if self.route_card.pill is not None:
            self.route_card.pill.set_state(view.route.badge, view.route.state)
        self.route_summary.setText(f"{view.route.summary}\n{view.route.detail}")
        warning = view.route.warning or (
            "All clients are isolated: an unselected sender stays ready without "
            "competing for the selected input."
        )
        self.route_warning.setText(warning)


def _choice_label(key: str, choice: Any) -> str:
    text = str(choice)
    if key == "capture_size":
        return text
    if key == "profile":
        return "Custom (unsaved mix)" if text == CUSTOM_CAMERA_PROFILE else (
            ARCH_CAMERA_PROFILES.get(text, {}).get("label", text)
        )
    if key == "rotation" and text.isdigit():
        return f"{text}°"
    return text.replace("_", " ").title()


_CONTROL_HINTS = {
    "profile": "Loads every value in this card at once.",
    "capture_size": "Save stages this for the owner's next natural start.",
    "power_line_frequency": "0 disabled · 1 = 50 Hz · 2 = 60 Hz.",
    "exposure_time_absolute": "Fallback used while automatic exposure is off.",
    "white_balance_temperature": "Fallback used while automatic white balance is off.",
    "exposure_dynamic_framerate": "Off keeps the 30 FPS delivery contract.",
    "zoom_percent": "Live centred sensor zoom; 100% keeps the full field of view.",
}


def _control_hint(key: str) -> str:
    return _CONTROL_HINTS.get(key, "")
