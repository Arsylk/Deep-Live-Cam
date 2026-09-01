#!/usr/bin/env python3
"""Reusable presentation components shared by every page.

There is exactly one implementation of a status pill, a metric row, a card, a
slot card, and a value slider.  Pages compose these; they never grow a second,
contradictory copy of the same control.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFontMetrics,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .contracts import SCOPE_LABELS, SCOPE_TOOLTIPS
from .theme import state_tokens
from .viewmodel import Alert, SlotView, StreamView


class StatusPill(QLabel):
    """A compact state chip that always pairs colour with a word and a glyph.

    ``elide=True`` is for chips whose text is long and runtime-driven (the
    header ribbon).  Those must not be able to push the window past its
    980x680 minimum, so they shrink with an ellipsis and keep the full state
    in the tooltip.
    """

    def __init__(
        self, text: str = "…", state: str = "unknown", *, elide: bool = False
    ) -> None:
        super().__init__()
        self.setObjectName("statusPill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._elide = bool(elide)
        self._full = ""
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored if elide else QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        self.set_state(text, state)

    def set_state(self, text: str, state: str) -> None:
        background, border, foreground, glyph = state_tokens(state)
        self._state = state
        self._full = f"{glyph} {text}"
        self.setStyleSheet(
            f"background:{background}; color:{foreground}; border:1px solid {border};"
            "border-radius:4px; padding:3px 8px; font-family:monospace;"
            "font-weight:700; font-size:9px;"
        )
        self.setToolTip(f"{text} ({state})")
        self.setAccessibleName(f"{text} ({state})")
        self._apply_text()

    def state(self) -> str:
        return self._state

    def full_text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        if not self._elide:
            return super().minimumSizeHint()
        return QSize(46, super().minimumSizeHint().height())

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._elide:
            self._apply_text()

    def _apply_text(self) -> None:
        if not self._elide:
            QLabel.setText(self, self._full)
            return
        available = self.width() - 18
        if available <= 24:
            QLabel.setText(self, self._full)
            return
        QLabel.setText(
            self,
            QFontMetrics(self.font()).elidedText(
                self._full, Qt.TextElideMode.ElideRight, available
            ),
        )


class ScopeBadge(QLabel):
    """States which engine a setting reaches, in the row that carries it."""

    def __init__(self, scope: str) -> None:
        super().__init__(SCOPE_LABELS.get(scope, scope.upper()))
        self.setObjectName("scopeBadge")
        self.setToolTip(SCOPE_TOOLTIPS.get(scope, ""))
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


def wrap_label(label: QLabel) -> None:
    """Give a wrapped label exactly the height its text needs.

    Without ``heightForWidth`` the label is clipped; without a fixed vertical
    policy it instead absorbs every spare pixel of its card and pushes the
    controls below it into the middle of empty space.
    """
    policy = label.sizePolicy()
    policy.setHeightForWidth(True)
    policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
    label.setSizePolicy(policy)


class ElidedLabel(QLabel):
    """A label that elides instead of forcing the whole window wider.

    Endpoints, model names, and device lists are long and change at runtime.
    Letting them drive the layout minimum would make the 980x680 window
    impossible, so they shrink with an ellipsis and keep the full text in the
    tooltip and in :meth:`full_text`.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full = str(text)
        self.setToolTip(self._full)
        self._elide()
        # The preferred width changed, so the surrounding layout has to be
        # asked again; otherwise the first (placeholder) text sizes the column
        # for the rest of the session.
        self.updateGeometry()

    def full_text(self) -> str:
        return self._full

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        # ensurePolished() applies the stylesheet font first; measuring with
        # the unpolished font under-reports and the label elides needlessly.
        self.ensurePolished()
        metrics = QFontMetrics(self.font())
        return QSize(
            metrics.horizontalAdvance(self._full) + 6,
            super().sizeHint().height(),
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(24, super().minimumSizeHint().height())

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        available = self.width() - 2
        if available <= 16:
            QLabel.setText(self, self._full)
            return
        QLabel.setText(
            self,
            QFontMetrics(self.font()).elidedText(
                self._full, Qt.TextElideMode.ElideRight, available
            ),
        )


class MetricRow(QWidget):
    """One labelled value, optionally with a plain-language hint below it."""

    def __init__(self, label: str, value: str = "—", hint: str = "") -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        # Both halves elide: a narrow pane must squeeze the strip rather than
        # let its items overlap each other.
        self.label = ElidedLabel(label.upper())
        self.label.setObjectName("metricLabel")
        self.value = ElidedLabel(value)
        self.value.setObjectName("metricValue")
        self.value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.hint = QLabel(hint)
        self.hint.setObjectName("metricHint")
        self.hint.setWordWrap(True)
        wrap_label(self.hint)
        self.hint.setVisible(bool(hint))
        layout.addWidget(self.label)
        layout.addWidget(self.value)
        layout.addWidget(self.hint)

    def set_value(self, value: str, hint: str | None = None) -> None:
        self.value.setText(value)
        if hint is not None:
            self.hint.setText(hint)
            self.hint.setVisible(bool(hint))


class MetricStrip(QWidget):
    """A horizontal run of metric rows with stable ordering."""

    def __init__(self, labels: Sequence[str], spacing: int = 18) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        self.rows: dict[str, MetricRow] = {}
        for label in labels:
            row = MetricRow(label)
            self.rows[label] = row
            layout.addWidget(row)
        layout.addStretch(1)

    def update_values(self, values: Iterable[tuple[str, str]]) -> None:
        for label, value in values:
            row = self.rows.get(label)
            if row is not None:
                row.set_value(value)


class Card(QFrame):
    """A titled surface with an optional detail line and trailing pill."""

    def __init__(
        self,
        title: str,
        detail: str = "",
        *,
        pill: bool = False,
        scope: str | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(13, 11, 13, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.title = ElidedLabel(title)
        self.title.setObjectName("cardTitle")
        header.addWidget(self.title, 1)
        if scope is not None:
            header.addWidget(ScopeBadge(scope))
        self.pill = StatusPill("…") if pill else None
        if self.pill is not None:
            header.addWidget(self.pill)
        outer.addLayout(header)

        self.detail = QLabel(detail)
        self.detail.setObjectName("cardDetail")
        self.detail.setWordWrap(True)
        wrap_label(self.detail)
        self.detail.setVisible(bool(detail))
        outer.addWidget(self.detail)

        self.content = QVBoxLayout()
        self.content.setSpacing(8)
        outer.addLayout(self.content)
        self._outer = outer

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.content.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout: QLayout) -> QLayout:
        self.content.addLayout(layout)
        return layout

    def add_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(7)
        self.content.addLayout(form)
        return form

    def add_stretch(self, stretch: int = 1) -> None:
        self.content.addStretch(stretch)

    def set_detail(self, text: str) -> None:
        self.detail.setText(text)
        self.detail.setVisible(bool(text))


def setting_label(text: str, scope: str, hint: str = "") -> QWidget:
    """Build the label side of a settings row: name, engine scope, and reason."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 2, 0, 0)
    layout.setSpacing(2)
    top = QHBoxLayout()
    top.setSpacing(6)
    name = QLabel(text)
    name.setWordWrap(True)
    top.addWidget(name)
    top.addWidget(ScopeBadge(scope))
    top.addStretch(1)
    layout.addLayout(top)
    if hint:
        note = QLabel(hint)
        note.setObjectName("metricHint")
        note.setWordWrap(True)
        layout.addWidget(note)
    return widget


class ValueSlider(QWidget):
    """Slider with explicit minimum/current/maximum labels.

    The application-wide wheel guard covers the embedded ``QSlider``; the value
    stays editable by dragging or with the keyboard arrows.
    """

    valueChanged = Signal(int)

    def __init__(
        self, minimum: int, maximum: int, value: int, step: int = 1
    ) -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.suffix = ""
        self.display_divisor = 1.0
        self.display_decimals = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(max(step, (maximum - minimum) // 10))
        self.slider.setValue(value)
        self.slider.setMinimumWidth(140)
        labels = QHBoxLayout()
        labels.setContentsMargins(1, 0, 1, 0)
        self.minimum_label = QLabel(str(minimum))
        self.minimum_label.setObjectName("rangeEndpoint")
        self.value_label = QLabel()
        self.value_label.setObjectName("rangeValue")
        self.maximum_label = QLabel(str(maximum))
        self.maximum_label.setObjectName("rangeEndpoint")
        labels.addWidget(self.minimum_label)
        labels.addStretch(1)
        labels.addWidget(self.value_label)
        labels.addStretch(1)
        labels.addWidget(self.maximum_label)
        layout.addWidget(self.slider)
        layout.addLayout(labels)
        self.slider.valueChanged.connect(self._update_value)
        self.slider.valueChanged.connect(self.valueChanged.emit)
        self._update_value(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def setSuffix(self, suffix: str) -> None:
        self.suffix = suffix
        self._update_value(self.value())

    def setDisplayScale(self, divisor: float, decimals: int = 0) -> None:
        self.display_divisor = divisor
        self.display_decimals = decimals
        self.minimum_label.setText(self._format_value(self.minimum))
        self.maximum_label.setText(self._format_value(self.maximum))
        self._update_value(self.value())

    def _format_value(self, value: int) -> str:
        return f"{value / self.display_divisor:.{self.display_decimals}f}"

    def _update_value(self, value: int) -> None:
        self.value_label.setText(f"{self._format_value(value)}{self.suffix}")


class VideoSurface(QLabel):
    """Paint the newest frame without making a pixmap part of layout state.

    ``QLabel.setPixmap()`` makes the pixmap dimensions its size hint.  A
    portrait frame inside the manager's resizable/scrollable output page can
    therefore cause a resize -> rescale -> new size-hint feedback loop.  It
    also eagerly allocates a newly scaled pixmap for every decoded frame even
    when Qt is already behind painting the previous one.

    Keeping the source ``QImage`` here and drawing it in ``paintEvent`` leaves
    geometry independent of frame orientation.  ``update()`` requests are
    naturally coalesced by Qt, so under load only the newest frame is painted.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._frame: QImage | None = None

    def set_frame(self, image: QImage) -> None:
        self._frame = image
        QLabel.setText(self, "")
        self.update()

    def clear_frame(self, message: str) -> None:
        self._frame = None
        QLabel.clear(self)
        QLabel.setText(self, message)
        self.update()

    def set_waiting_text(self, message: str) -> None:
        if self._frame is None:
            QLabel.setText(self, message)
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        # Deliberately independent of source frame geometry.  In particular,
        # switching 640x360 to 360x640 must not alter the surrounding layout.
        return QSize(320, 180)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(192, 108)

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        # QLabel still paints the stylesheet background/border and, while no
        # frame is available, the centered waiting text.
        super().paintEvent(event)
        frame = self._frame
        if (
            frame is None
            or frame.isNull()
            or self.width() < 2
            or self.height() < 2
        ):
            return
        bounds = self.contentsRect()
        fitted = frame.size().scaled(
            bounds.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        left = bounds.left() + (bounds.width() - fitted.width()) // 2
        top = bounds.top() + (bounds.height() - fitted.height()) // 2
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(
            QRect(left, top, fitted.width(), fitted.height()),
            frame,
        )


class VideoPane(QFrame):
    """A passive preview surface with a labelled, honest header and footer."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self.setObjectName("videoPane")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 11, 12, 11)
        layout.setSpacing(7)

        header = QHBoxLayout()
        header.setSpacing(7)
        # The title elides and the comparison chip sits on its own row, so a
        # long source name cannot widen the minimum window.
        self.title_label = ElidedLabel(title)
        self.title_label.setObjectName("paneTitle")
        header.addWidget(self.title_label, 1)
        self.status_pill = StatusPill("STARTING", "waiting")
        header.addWidget(self.status_pill)
        layout.addLayout(header)

        # Single elided lines with the full text in the tooltip: pane chrome
        # must not grow tall enough to squeeze the picture out of the pane.
        self.subtitle_label = ElidedLabel(subtitle)
        self.subtitle_label.setObjectName("paneSubtitle")
        layout.addWidget(self.subtitle_label)

        delay_row = QHBoxLayout()
        delay_row.setContentsMargins(0, 0, 0, 0)
        self.delay_pill = StatusPill("DELAYED", "working")
        self.delay_pill.setVisible(False)
        delay_row.addWidget(self.delay_pill)
        delay_row.addStretch(1)
        layout.addLayout(delay_row)

        self.video = VideoSurface("WAITING FOR FRAMES")
        self.video.setObjectName("videoSurface")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(192, 108)
        self.video.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.video, 1)

        self.metrics = MetricStrip(
            ("Decoded FPS", "Frames", "Drops", "Last frame"), spacing=16
        )
        layout.addWidget(self.metrics)

        self.stats = ElidedLabel("decoder starting…")
        self.stats.setObjectName("paneSubtitle")
        layout.addWidget(self.stats)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.video.set_frame(image)

    def clear_image(self, message: str) -> None:
        self._image = None
        self.video.clear_frame(message)

    def set_waiting(self, message: str) -> None:
        if self._image is None:
            self.video.set_waiting_text(message)

    def set_heading(self, title: str, subtitle: str) -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)

    def render(self, stream: StreamView) -> None:
        """Apply one normalized stream description to every visible field."""
        self.status_pill.set_state(stream.state_text, stream.state)
        self.delay_pill.setVisible(stream.delayed)
        if stream.delayed:
            self.delay_pill.set_state(
                f"DELAYED {stream.delayed_ms} ms FOR COMPARISON", "working"
            )
        self.set_heading(
            stream.title,
            f"{stream.source}  ·  model {stream.model}  ·  {stream.endpoint}",
        )
        self.metrics.update_values(stream.metrics())
        self.stats.setText(stream.note)

class SlotCard(QToolButton):
    """One Windows client slot: identity, capability, endpoints, readiness."""

    def __init__(self, slot: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot = slot
        self.device_id: str | None = None
        self.setObjectName("slotCard")
        self.setCheckable(True)
        self.setAutoRaise(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(104)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.identity = ElidedLabel(f"SLOT {slot}")
        self.identity.setObjectName("nodeTitle")
        self.pill = StatusPill("…")
        header.addWidget(self.identity, 1)
        header.addWidget(self.pill)
        layout.addLayout(header)
        self.capability = QLabel("")
        self.capability.setObjectName("cardDetail")
        self.capability.setWordWrap(True)
        self.endpoint = QLabel("")
        self.endpoint.setObjectName("cardDetail")
        self.endpoint.setWordWrap(True)
        self.problem = QLabel("")
        self.problem.setObjectName("alertMessage")
        self.problem.setWordWrap(True)
        self.problem.setVisible(False)
        for widget in (self.capability, self.endpoint, self.problem):
            layout.addWidget(widget)
        for child in (
            self.identity,
            self.pill,
            self.capability,
            self.endpoint,
            self.problem,
        ):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def render(self, view: SlotView) -> None:
        self.device_id = view.device_id
        self.identity.setText(f"SLOT {view.slot}  ·  {view.label.upper()}")
        self.capability.setText(f"{view.stack} — {view.capability}")
        self.endpoint.setText(view.endpoint)
        self.pill.set_state(view.state_text, view.state)
        self.problem.setText(view.error or "")
        self.problem.setVisible(bool(view.error))
        self.setProperty("routeState", view.state)
        self.setChecked(view.selected)
        self.setEnabled(view.selectable or view.selected)
        if view.configured:
            self.setToolTip(
                f"Select {view.identity} as the only Windows processing input. "
                "Every other client keeps its own local fallback; no camera is "
                "opened, closed, or reassigned."
            )
        else:
            self.setToolTip(
                f"Slot {view.slot} reserves ports {view.input_port} and "
                f"{view.return_port} but has no camera owner configured."
            )
        self.style().unpolish(self)
        self.style().polish(self)


class ResponsiveCardGrid(QWidget):
    """Re-flow fixed-height cards into as many columns as the width allows."""

    def __init__(
        self,
        minimum_card_width: int = 320,
        spacing: int = 10,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.minimum_card_width = int(minimum_card_width)
        self._cards: list[QWidget] = []
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)

    def set_cards(self, cards: Sequence[QWidget]) -> None:
        self._cards = list(cards)
        self._columns = 0
        # Before the first resize the width is zero, and flowing everything
        # into a single column would make the initial vertical minimum huge.
        # Assume a typical workspace width and let resizeEvent correct it.
        self._reflow(self.columns_for(self.width() or 860))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Never force the window wider than one card.

        The grid re-flows on resize, so a wide arrangement must not become a
        floor that stops the window shrinking back to 980x680.
        """
        hint = self._grid.minimumSize()
        return QSize(min(self.minimum_card_width, hint.width()), hint.height())

    def columns_for(self, width: int) -> int:
        spacing = self._grid.horizontalSpacing()
        usable = max(0, int(width))
        columns = max(
            1,
            (usable + spacing) // (self.minimum_card_width + spacing),
        )
        return int(min(max(1, columns), max(1, len(self._cards))))

    def columns(self) -> int:
        return self._columns

    def _reflow(self, columns: int) -> None:
        if columns == self._columns or not self._cards:
            return
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        for index, card in enumerate(self._cards):
            self._grid.addWidget(card, index // columns, index % columns)
        for column in range(self._grid.columnCount()):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Never demand more than one card's width.

        The current column count is a consequence of the available width; if it
        also became the layout minimum, widening the window once would stop it
        from being narrowed again.
        """
        return QSize(self.minimum_card_width, super().minimumSizeHint().height())

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._reflow(self.columns_for(self.width()))


class TopologyNode(QFrame):
    """One node of the compact Android / Windows / Arch route diagram."""

    def __init__(self, title: str, detail: str) -> None:
        super().__init__()
        self.setObjectName("topologyNode")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(4)
        header = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setObjectName("nodeTitle")
        self.pill = StatusPill("…")
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.pill)
        layout.addLayout(header)
        self.detail = QLabel(detail)
        self.detail.setObjectName("cardDetail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

    def render(self, node: Any) -> None:
        self.title.setText(node.title)
        self.detail.setText(node.detail)
        self.pill.set_state(node.state_text, node.state)


class AlertRow(QFrame):
    """A failure with its owning component and one concrete next action."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("alertRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 8, 11, 8)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.component = QLabel()
        self.component.setObjectName("alertComponent")
        self.component.setWordWrap(True)
        self.pill = StatusPill("…")
        header.addWidget(self.component, 1)
        header.addWidget(self.pill)
        layout.addLayout(header)
        self.message = QLabel()
        self.message.setObjectName("alertMessage")
        self.message.setWordWrap(True)
        self.action = QLabel()
        self.action.setObjectName("alertAction")
        self.action.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addWidget(self.action)

    def render(self, alert: Alert) -> None:
        self.component.setText(alert.component)
        self.message.setText(alert.message)
        self.action.setText(f"Next: {alert.next_action}")
        self.pill.set_state(alert.severity.upper(), alert.severity)
        self.setProperty("severity", alert.severity)
        self.style().unpolish(self)
        self.style().polish(self)


class AlertList(QWidget):
    """A bounded list of alert rows, reusing widgets across refreshes."""

    def __init__(self, limit: int = 6, empty_text: str = "No problems reported.") -> None:
        super().__init__()
        self.limit = limit
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.empty = QLabel(empty_text)
        self.empty.setObjectName("hintText")
        layout.addWidget(self.empty)
        self.rows = [AlertRow() for _ in range(limit)]
        for row in self.rows:
            row.setVisible(False)
            layout.addWidget(row)
        self.overflow = QLabel()
        self.overflow.setObjectName("hintText")
        self.overflow.setVisible(False)
        layout.addWidget(self.overflow)

    def render(self, alerts: Sequence[Alert]) -> None:
        self.empty.setVisible(not alerts)
        for index, row in enumerate(self.rows):
            if index < len(alerts):
                row.render(alerts[index])
                row.setVisible(True)
            else:
                row.setVisible(False)
        extra = max(0, len(alerts) - self.limit)
        self.overflow.setText(
            f"{extra} more item(s) not shown; see the detailed snapshot below."
            if extra
            else ""
        )
        self.overflow.setVisible(bool(extra))


def page_heading(title: str, description: str) -> QWidget:
    frame = QWidget()
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 4)
    layout.setSpacing(2)
    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    subtitle = QLabel(description)
    subtitle.setObjectName("pageSubtitle")
    subtitle.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(subtitle)
    return frame


def scrollable(panel: QWidget) -> QScrollArea:
    """Wrap a page body so it can scroll vertically without nesting tabs."""
    scroll = QScrollArea()
    scroll.setObjectName("workspaceScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(panel)
    return scroll


def note_label(text: str, kind: str = "note") -> QLabel:
    label = QLabel(text)
    label.setObjectName("noteBox" if kind == "note" else "infoBox")
    label.setWordWrap(True)
    wrap_label(label)
    return label


def thumbnail(path: str, size: QSize) -> QPixmap:
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class FramingPreview(QFrame):
    """Live preview of the framed prerecorded video with click-drag repositioning.

    The surface shows whatever the receiver's framing relay currently produces
    (already zoomed/offset), so it is an honest what-you-see preview rather than
    a client-side re-render.  Dragging with the cursor pans the video: the pixel
    delta on screen is converted to source pixels (using the ratio of the source
    width to the displayed width) and emitted as a cumulative offset via
    ``offsetDragged`` so the caller can push it to the receiver live.
    """

    # Cumulative source-space offset (offset_x, offset_y) in pixels.
    offsetDragged = Signal(int, int)
    # Live, per-move offset while a drag is in progress (for label/slider sync
    # without thrashing the receiver).
    offsetPreview = Signal(int, int)

    def __init__(
        self,
        source_width: int,
        source_height: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("videoPane")
        self._image: QImage | None = None
        self._source_width = int(source_width)
        self._source_height = int(source_height)
        self._offset_x = 0
        self._offset_y = 0
        self._dragging = False
        self._drag_start: tuple[float, float] | None = None
        self._drag_origin: tuple[int, int] | None = None
        # Screen-space delta applied optimistically while dragging so the
        # preview pans instantly; the real framed frame from the receiver
        # replaces it on release.
        self._pan_screen: tuple[float, float] = (0.0, 0.0)
        self._waiting = "waiting for framed preview…"
        self._show_grid = True
        self.setMinimumSize(320, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setMouseTracking(False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(384, 432)

    def set_offset(self, offset_x: int, offset_y: int) -> None:
        """Adopt an externally-set offset (e.g. from the sliders or a reset)."""
        self._offset_x = int(offset_x)
        self._offset_y = int(offset_y)

    def set_grid_visible(self, visible: bool) -> None:
        """Toggle the rule-of-thirds + centre-cross positioning guides."""
        self._show_grid = bool(visible)
        self.update()

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def clear_image(self, message: str | None = None) -> None:
        self._image = None
        if message is not None:
            self._waiting = message
        self.update()

    def _displayed_rect(self) -> QRect | None:
        frame = self._image
        if frame is None or frame.isNull():
            return None
        bounds = self.contentsRect()
        if bounds.width() < 2 or bounds.height() < 2:
            return None
        fitted = frame.size().scaled(
            bounds.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        left = bounds.left() + (bounds.width() - fitted.width()) // 2
        top = bounds.top() + (bounds.height() - fitted.height()) // 2
        return QRect(left, top, fitted.width(), fitted.height())

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        rect = self._displayed_rect()
        painter = QPainter(self)
        if rect is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(
                self.contentsRect(),
                Qt.AlignmentFlag.AlignCenter,
                self._waiting,
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        assert self._image is not None
        # While dragging, translate the decoded frame by the in-progress screen
        # delta so the pan feels instant.  The receiver output only changes on
        # release, so this optimistic shift is what makes the drag responsive.
        draw_rect = rect
        if self._dragging:
            dx, dy = self._pan_screen
            draw_rect = rect.translated(int(dx), int(dy))
        painter.drawImage(draw_rect, self._image)
        if self._show_grid:
            self._paint_grid(painter, rect)

    def _paint_grid(self, painter: QPainter, rect: QRect) -> None:
        """Camera-style framing overlay fixed to the locked output box.

        Designed to be genuinely useful for placing a face and readable over any
        footage, not a flat white tic-tac-toe:
          * rule-of-thirds lines drawn as a dark halo + light line so they stay
            visible on both bright and dark video;
          * small dots at the four thirds intersections (where a face's eyes
            naturally sit) instead of lines cutting through the subject;
          * a compact centre reticle (short ticks + gap) rather than a full
            cross that obscures the face;
          * corner brackets marking the output-box edge instead of a solid
            border ring.
        The grid tracks the displayed frame rect, so it is a stable reference
        for where /dev/deep-live-cam's locked frame sits while the video pans.
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        left, top = rect.left(), rect.top()
        w, h = rect.width(), rect.height()
        right, bottom = left + w, top + h
        thirds_x = (left + w // 3, left + 2 * w // 3)
        thirds_y = (top + h // 3, top + 2 * h // 3)

        halo = QPen(QColor(0, 0, 0, 110))
        halo.setWidthF(2.6)
        line = QPen(QColor(235, 238, 245, 120))
        line.setWidthF(1.0)

        # Rule-of-thirds: draw the dark halo first, then the light line on top.
        for pen in (halo, line):
            painter.setPen(pen)
            for x in thirds_x:
                painter.drawLine(x, top, x, bottom)
            for y in thirds_y:
                painter.drawLine(left, y, right, y)

        # Power-point dots at the thirds intersections (eye-line targets).
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120))
        for x in thirds_x:
            for y in thirds_y:
                painter.drawEllipse(QRect(x - 3, y - 3, 6, 6))
        painter.setBrush(QColor(255, 255, 255, 210))
        for x in thirds_x:
            for y in thirds_y:
                painter.drawEllipse(QRect(x - 2, y - 2, 4, 4))

        # Centre reticle: short ticks with a gap, not a full cross.
        cx, cy = left + w // 2, top + h // 2
        tick = max(6, min(w, h) // 24)
        gap = max(3, tick // 2)
        reticle_halo = QPen(QColor(0, 0, 0, 120))
        reticle_halo.setWidthF(2.6)
        reticle = QPen(QColor(120, 200, 255, 220))
        reticle.setWidthF(1.2)
        for pen in (reticle_halo, reticle):
            painter.setPen(pen)
            painter.drawLine(cx - gap - tick, cy, cx - gap, cy)
            painter.drawLine(cx + gap, cy, cx + gap + tick, cy)
            painter.drawLine(cx, cy - gap - tick, cx, cy - gap)
            painter.drawLine(cx, cy + gap, cx, cy + gap + tick)

        # Corner brackets marking the locked output-box edge.
        blen = max(12, min(w, h) // 12)
        bracket_halo = QPen(QColor(0, 0, 0, 130))
        bracket_halo.setWidthF(3.4)
        bracket = QPen(QColor(255, 255, 255, 200))
        bracket.setWidthF(1.6)
        # Inset by 1px so the strokes sit fully inside the frame.
        l, t, r, b = left + 1, top + 1, right - 1, bottom - 1
        corners = (
            ((l, t), (1, 1)), ((r, t), (-1, 1)),
            ((l, b), (1, -1)), ((r, b), (-1, -1)),
        )
        for pen in (bracket_halo, bracket):
            painter.setPen(pen)
            for (px, py), (sx, sy) in corners:
                painter.drawLine(px, py, px + sx * blen, py)
                painter.drawLine(px, py, px, py + sy * blen)

        painter.restore()

    def _source_scale(self, rect: QRect) -> float:
        """Source pixels per displayed pixel along the width."""
        if rect.width() <= 0:
            return 1.0
        return self._source_width / float(rect.width())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        rect = self._displayed_rect()
        if rect is None:
            return
        self._dragging = True
        self._drag_start = (event.position().x(), event.position().y())
        self._drag_origin = (self._offset_x, self._offset_y)
        self._pan_screen = (0.0, 0.0)
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._dragging or self._drag_start is None:
            return
        rect = self._displayed_rect()
        if rect is None:
            return
        scale = self._source_scale(rect)
        dx_screen = event.position().x() - self._drag_start[0]
        dy_screen = event.position().y() - self._drag_start[1]
        self._pan_screen = (dx_screen, dy_screen)
        assert self._drag_origin is not None
        self._offset_x = int(round(self._drag_origin[0] + dx_screen * scale))
        self._offset_y = int(round(self._drag_origin[1] + dy_screen * scale))
        # Update labels/sliders live, but do NOT push to the receiver mid-drag
        # (that would restart the decoder on every move and stutter badly).
        self.offsetPreview.emit(self._offset_x, self._offset_y)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        was_dragging = self._dragging
        self._dragging = False
        self._drag_start = None
        self._drag_origin = None
        self._pan_screen = (0.0, 0.0)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        if was_dragging:
            # Commit the final offset to the receiver once, on release.
            self.offsetDragged.emit(self._offset_x, self._offset_y)
        self.update()
