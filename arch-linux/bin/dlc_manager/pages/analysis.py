#!/usr/bin/env python3
"""Analyze: comparison controls, passive metrics, and the active baseline.

Everything on this page is measurement.  The comparison delay and the sampler
change only what this window shows; they never touch capture, processing, or
camera ownership.  Signal metrics are labelled as signal metrics, and the
limits of full-reference scores on an intentional identity edit are stated
rather than implied.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..baseline import BaselineView, load_active_baseline
from ..contracts import SCOPE_COMPARISON
from ..quality import METRIC_ROWS, readiness, region_text
from ..viewmodel import ManagerView
from ..widgets import (
    Card,
    MetricRow,
    ValueSlider,
    note_label,
    wrap_label,
    ResponsiveCardGrid,
    page_heading,
    scrollable,
    setting_label,
)


class AnalysisPage(QWidget):
    """Workspace 4: passive comparison and the reproducible baseline."""

    comparisonChanged = Signal()
    measurementToggled = Signal(bool)
    resetRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 18)
        layout.setSpacing(10)
        layout.addWidget(
            page_heading(
                "Analyze",
                "Comparison and measurement only. Nothing here changes a "
                "camera, a processor, or the stable system-camera route.",
            )
        )

        self.comparison_card = self._build_comparison_card()
        self.measurement_card = self._build_measurement_card()
        self.baseline_card = self._build_baseline_card()
        self.top_grid = ResponsiveCardGrid(minimum_card_width=430, spacing=10)
        self.top_grid.set_cards([self.comparison_card, self.measurement_card])
        layout.addWidget(self.top_grid)
        layout.addWidget(self.baseline_card)
        layout.addWidget(self._build_caveat_card())
        layout.addStretch(1)
        outer.addWidget(scrollable(panel))

    # ------------------------------------------------------------------ build

    def _build_comparison_card(self) -> Card:
        card = Card(
            "COMPARISON VIEW",
            "Aligns the raw pane with the processed result so the two can be "
            "judged on the same moment.",
            scope=SCOPE_COMPARISON,
        )
        form = card.add_form()
        self.align_views = QCheckBox("Delay the raw pane")
        self.align_views.setChecked(True)
        self.align_views.toggled.connect(self._alignment_toggled)
        form.addRow(
            setting_label(
                "Alignment",
                SCOPE_COMPARISON,
                "Buffers already-decoded preview frames in this window only.",
            ),
            self.align_views,
        )
        self.raw_delay = ValueSlider(0, 2000, 350, 10)
        self.raw_delay.setSuffix(" ms")
        self.raw_delay.valueChanged.connect(lambda _value: self.comparisonChanged.emit())
        form.addRow(
            setting_label(
                "Raw preview delay",
                SCOPE_COMPARISON,
                "Matches the round-trip latency you want to compare against.",
            ),
            self.raw_delay,
        )
        card.add(
            note_label(
                "The camera pipeline is never delayed. Only this window's raw "
                "pane is held back.",
                "info",
            )
        )
        return card

    def _build_measurement_card(self) -> Card:
        card = Card(
            "PASSIVE MEASUREMENT",
            "Sampled from frames this window already decoded for display.",
            pill=True,
            scope=SCOPE_COMPARISON,
        )
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.measure_toggle = QCheckBox("Measure both previews")
        self.measure_toggle.setChecked(True)
        self.measure_toggle.toggled.connect(self.measurementToggled.emit)
        self.reset_button = QPushButton("Reset the sample window")
        self.reset_button.setProperty("compact", True)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        controls.addWidget(self.measure_toggle, 1)
        controls.addWidget(self.reset_button)
        card.add_layout(controls)

        self.region_labels: dict[str, QLabel] = {}
        self.metric_widgets: dict[str, dict[str, MetricRow]] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        headers = {"raw": "RAW INPUT", "result": "PROCESSED RESULT"}
        for column, (key, title) in enumerate(headers.items()):
            caption = QLabel(title)
            caption.setObjectName("sectionLabel")
            grid.addWidget(caption, 0, column)
            region = QLabel("waiting for samples")
            region.setObjectName("hintText")
            region.setWordWrap(True)
            grid.addWidget(region, 1, column)
            self.region_labels[key] = region
            self.metric_widgets[key] = {}
        for row, (metric_key, label, _template, meaning) in enumerate(
            METRIC_ROWS, start=2
        ):
            for column, key in enumerate(headers):
                widget = MetricRow(label, "—", meaning if column == 0 else "")
                grid.addWidget(widget, row, column)
                self.metric_widgets[key][metric_key] = widget
        card.add_layout(grid)

        self.comparison_summary = QLabel("")
        self.comparison_summary.setObjectName("readout")
        self.comparison_summary.setWordWrap(True)
        wrap_label(self.comparison_summary)
        card.add(self.comparison_summary)

        self.processor_quality = QLabel("")
        self.processor_quality.setObjectName("hintText")
        self.processor_quality.setWordWrap(True)
        wrap_label(self.processor_quality)
        card.add(self.processor_quality)
        return card

    def _build_baseline_card(self) -> Card:
        card = Card(
            "ACTIVE REPRODUCIBLE BASELINE",
            "The registered reference run future candidates are measured "
            "against.",
            pill=True,
        )
        self.baseline_identity = QLabel("")
        self.baseline_identity.setObjectName("sectionLabel")
        self.baseline_identity.setWordWrap(True)
        card.add(self.baseline_identity)
        self.baseline_context = QLabel("")
        self.baseline_context.setObjectName("hintText")
        self.baseline_context.setWordWrap(True)
        card.add(self.baseline_context)
        self.baseline_metrics: list[MetricRow] = []
        for _label, _value, _meaning in BaselineView(available=False).metrics():
            row = MetricRow(_label, _value, _meaning)
            self.baseline_metrics.append(row)
            card.add(row)
        self.baseline_limits = note_label("", "note")
        card.add(self.baseline_limits)
        return card

    def _build_caveat_card(self) -> Card:
        card = Card(
            "WHAT THESE NUMBERS DO NOT MEAN",
            "Read before comparing two configurations.",
        )
        for text in (
            "VMAF, SSIM, and PSNR are full-reference signal metrics. On a face "
            "swap they also penalize the intended identity edit, so a lower "
            "score is not evidence of a worse swap.",
            "Detail, edge, and blockiness values describe signal preservation "
            "in the decoded preview. They are not a facial-realism score.",
            "A definitive comparison between two pipelines requires replaying "
            "the same frozen decoded reference corpus through both. Live "
            "numbers from two different moments are not comparable.",
            "A processor being invoked, or a mask ring changing, is not "
            "evidence that an identity was replaced.",
        ):
            label = QLabel(f"·  {text}")
            label.setObjectName("hintText")
            label.setWordWrap(True)
            card.add(label)
        return card

    # ---------------------------------------------------------------- signals

    def _alignment_toggled(self, checked: bool) -> None:
        self.raw_delay.setEnabled(checked)
        self.comparisonChanged.emit()

    def comparison_delay_ms(self) -> int:
        return self.raw_delay.value() if self.align_views.isChecked() else 0

    # ------------------------------------------------------------------ render

    def load_baseline(self, baseline: BaselineView | None = None) -> None:
        """Render the baseline once; it only changes when a run is registered."""
        value = baseline if baseline is not None else load_active_baseline()
        pill = self.baseline_card.pill
        if pill is not None:
            pill.set_state(
                "REGISTERED" if value.available else "NOT REGISTERED",
                "running" if value.available else "unknown",
            )
        self.baseline_identity.setText(
            value.identifier if value.available else "No baseline is registered."
        )
        if value.available:
            self.baseline_context.setText(
                f"Camera profile {value.camera_profile} · {value.camera_mode}\n"
                f"Model {value.model} / {value.backend}\n"
                f"Pairing: {value.pairing}\n{value.run_path}"
            )
        else:
            self.baseline_context.setText(value.error)
        for row, (label, shown, meaning) in zip(
            self.baseline_metrics, value.metrics()
        ):
            row.label.setText(label.upper())
            row.set_value(shown, meaning)
        self.baseline_limits.setText(
            "Comparison limits — identity: "
            f"{value.interpretation.get('identity', '')}. Full-reference "
            f"metrics: {value.interpretation.get('full_reference_metrics', '')}. "
            "A definitive cross-pipeline comparison "
            f"{value.interpretation.get('definitive_cross_pipeline_comparison', '')}."
        )

    def render(
        self,
        view: ManagerView,
        *,
        raw_metrics: dict[str, Any],
        result_metrics: dict[str, Any],
        enabled: bool,
    ) -> None:
        pill = self.measurement_card.pill
        if not enabled:
            if pill is not None:
                pill.set_state("PAUSED", "stopped")
            self.comparison_summary.setText(
                "Measurement is paused. The previews keep decoding; only the "
                "sampler is idle."
            )
            return

        samples = {"raw": raw_metrics, "result": result_metrics}
        states = []
        for key, values in samples.items():
            state, text = readiness(values)
            states.append((state, text))
            self.region_labels[key].setText(
                f"{text.title()} — measured over {region_text(values)}"
                if values.get("samples")
                else text.title()
            )
            for metric_key, _label, template, _meaning in METRIC_ROWS:
                raw_value = values.get(metric_key)
                widget = self.metric_widgets[key][metric_key]
                if raw_value is None:
                    widget.set_value("—")
                else:
                    try:
                        widget.set_value(template.format(value=float(raw_value)))
                    except (TypeError, ValueError):
                        widget.set_value("—")
        if pill is not None:
            worst = min(
                states,
                key=lambda item: {"unavailable": 0, "waiting": 1, "working": 2}.get(
                    item[0], 3
                ),
            )
            pill.set_state(worst[1], worst[0])

        self.comparison_summary.setText(_comparison_text(raw_metrics, result_metrics))
        self.processor_quality.setText(_processor_quality_text(view))


def _ratio(numerator: Any, denominator: Any) -> str:
    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return "—"
    if bottom <= 0:
        return "—"
    return f"{100.0 * top / bottom:.0f} %"


def _comparison_text(raw: dict[str, Any], result: dict[str, Any]) -> str:
    if not raw.get("samples") or not result.get("samples"):
        return (
            "Both panes need samples before a comparison is meaningful. The "
            "raw pane is unavailable on routes where phone frames go straight "
            "to Windows."
        )
    return (
        "Result relative to raw — fine detail "
        f"{_ratio(result.get('detail_laplacian'), raw.get('detail_laplacian'))}, "
        "edge strength "
        f"{_ratio(result.get('edge_energy'), raw.get('edge_energy'))}. "
        "These describe how much signal survived the round trip, not how "
        "convincing the face looks."
    )


def _processor_quality_text(view: ManagerView) -> str:
    processor = view.processor
    if processor.local_running:
        return (
            f"Local processor: {processor.local_model}/{processor.local_backend} "
            f"at {processor.local_fps:.1f} unique FPS · checkpoint "
            f"{processor.local_checkpoint} · identity evidence "
            f"{processor.identity_status} ({processor.identity_detail})."
        )
    if processor.windows_reachable:
        return (
            f"Windows processor: {processor.windows_active_model} · mode "
            f"{processor.windows_mode}. Its own quality gate is reported on "
            "System and logs."
        )
    return "No processor is reporting quality signals right now."
