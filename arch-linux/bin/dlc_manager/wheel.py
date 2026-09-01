#!/usr/bin/env python3
"""Application-wide guard that stops incidental scrolling from editing values."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QWidget,
)


class WheelValueGuard(QObject):
    """Keep incidental page scrolling from editing value controls.

    Qt normally sends a wheel event to the widget under the pointer, even when
    that widget was never selected.  That makes it very easy to change a camera
    parameter, route, or model while merely scrolling a settings page.  A value
    wheel input is always forwarded to the surrounding scroll area.  Values
    remain editable by click/drag, keyboard arrows, or direct text entry.

    Matching the Qt base classes also covers controls this file has never seen:
    sliders and dials, integer/double/date spin boxes, and combo boxes, whether
    they were written by hand or generated from a camera adapter schema.
    """

    _VALUE_CONTROL_TYPES = (QAbstractSlider, QAbstractSpinBox, QComboBox)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._forwarding = False

    @classmethod
    def _value_control(cls, watched: QObject) -> QWidget | None:
        widget = watched if isinstance(watched, QWidget) else None
        while widget is not None:
            if isinstance(widget, cls._VALUE_CONTROL_TYPES):
                return widget
            widget = widget.parentWidget()
        return None

    @staticmethod
    def _scroll_area(control: QWidget) -> QAbstractScrollArea | None:
        widget = control.parentWidget()
        while widget is not None:
            if isinstance(widget, QAbstractScrollArea):
                return widget
            widget = widget.parentWidget()
        return None

    @staticmethod
    def _forward_wheel(
        scroll_area: QAbstractScrollArea,
        event: QWheelEvent,
    ) -> None:
        pixel = event.pixelDelta()
        angle = event.angleDelta()
        horizontal = abs(pixel.x()) > abs(pixel.y()) or (
            pixel.isNull() and abs(angle.x()) > abs(angle.y())
        )
        bar = (
            scroll_area.horizontalScrollBar()
            if horizontal
            else scroll_area.verticalScrollBar()
        )
        pixel_delta = pixel.x() if horizontal else pixel.y()
        angle_delta = angle.x() if horizontal else angle.y()
        if pixel_delta:
            movement = -pixel_delta
        elif angle_delta:
            movement = round(-angle_delta / 120.0 * max(1, bar.singleStep()) * 3)
        else:
            return
        bar.setValue(bar.value() + movement)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if self._forwarding:
            return False
        control = self._value_control(watched)
        if control is None:
            return False
        if event.type() != QEvent.Type.Wheel:
            return False
        if isinstance(event, QWheelEvent):
            scroll_area = self._scroll_area(control)
            if scroll_area is not None:
                self._forwarding = True
                try:
                    self._forward_wheel(scroll_area, event)
                finally:
                    self._forwarding = False
        event.accept()
        return True
