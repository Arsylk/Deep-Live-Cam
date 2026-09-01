from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QFocusEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
MODULE_PATH = BIN / "tester.py"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location("arch_tester_wheel_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tester = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tester)


def _application() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def _wheel_event(widget: QWidget, delta: int = -120) -> QWheelEvent:
    position = QPointF(widget.rect().center())
    global_position = QPointF(widget.mapToGlobal(widget.rect().center()))
    return QWheelEvent(
        position,
        global_position,
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _control_case(kind: str) -> tuple[QWidget, object, object]:
    if kind == "slider":
        control = QSlider(Qt.Orientation.Horizontal)
        control.setRange(0, 10)
        control.setValue(5)
        return control, control.value, 5
    if kind == "dial":
        control = QDial()
        control.setRange(0, 10)
        control.setValue(5)
        return control, control.value, 5
    if kind == "spin":
        control = QSpinBox()
        control.setRange(0, 10)
        control.setValue(5)
        return control, control.value, 5
    if kind == "double-spin":
        control = QDoubleSpinBox()
        control.setRange(0.0, 10.0)
        control.setValue(5.0)
        return control, control.value, 5.0
    if kind == "combo":
        control = QComboBox()
        control.addItems(["one", "two", "three"])
        control.setCurrentIndex(1)
        return control, control.currentIndex, 1
    raise AssertionError(kind)


def _scroll_fixture(control: QWidget) -> tuple[QScrollArea, QPushButton]:
    scroll = QScrollArea()
    scroll.resize(320, 180)
    scroll.setWidgetResizable(True)
    panel = QWidget()
    panel.setMinimumHeight(1100)
    layout = QVBoxLayout(panel)
    layout.addWidget(control)
    focus_target = QPushButton("Safe focus target")
    layout.addWidget(focus_target)
    spacer = QLabel("page content")
    spacer.setFixedHeight(900)
    layout.addWidget(spacer)
    scroll.setWidget(panel)
    scroll.show()
    return scroll, focus_target


def test_hover_wheel_never_edits_wheel_sensitive_value_controls():
    app = _application()
    guard = tester.WheelValueGuard(app)
    app.installEventFilter(guard)
    try:
        for kind in ("slider", "dial", "spin", "double-spin", "combo"):
            control, read_value, original_value = _control_case(kind)
            scroll, focus_target = _scroll_fixture(control)
            app.processEvents()
            focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
            scroll.verticalScrollBar().setValue(0)
            app.processEvents()

            QApplication.sendEvent(control, _wheel_event(control))

            assert read_value() == original_value, kind
            assert scroll.verticalScrollBar().value() > 0, kind
            scroll.close()
            scroll.deleteLater()
            app.processEvents()
    finally:
        app.removeEventFilter(guard)


def test_wheel_stays_disabled_after_deliberate_control_focus():
    app = _application()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 10)
    slider.setValue(5)
    scroll, _focus_target = _scroll_fixture(slider)
    guard = tester.WheelValueGuard(app)
    app.installEventFilter(guard)
    try:
        app.processEvents()
        scroll.verticalScrollBar().setValue(0)
        QApplication.sendEvent(
            slider,
            QFocusEvent(
                QFocusEvent.Type.FocusIn,
                Qt.FocusReason.MouseFocusReason,
            ),
        )
        app.processEvents()

        QApplication.sendEvent(slider, _wheel_event(slider))

        assert slider.value() == 5
        assert scroll.verticalScrollBar().value() > 0
    finally:
        app.removeEventFilter(guard)
        scroll.close()
        scroll.deleteLater()
        app.processEvents()


def test_automatic_focus_does_not_make_hover_wheel_dangerous():
    app = _application()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 10)
    slider.setValue(5)
    scroll, _focus_target = _scroll_fixture(slider)
    guard = tester.WheelValueGuard(app)
    app.installEventFilter(guard)
    try:
        app.processEvents()
        scroll.verticalScrollBar().setValue(0)
        QApplication.sendEvent(
            slider,
            QFocusEvent(
                QFocusEvent.Type.FocusIn,
                Qt.FocusReason.ActiveWindowFocusReason,
            ),
        )
        app.processEvents()

        QApplication.sendEvent(slider, _wheel_event(slider))

        assert slider.value() == 5
        assert scroll.verticalScrollBar().value() > 0
    finally:
        app.removeEventFilter(guard)
        scroll.close()
        scroll.deleteLater()
        app.processEvents()


def test_spinbox_child_is_recognized_as_part_of_guarded_control():
    app = _application()
    spin = QSpinBox()
    spin.setRange(0, 10)
    spin.setValue(5)
    scroll, focus_target = _scroll_fixture(spin)
    guard = tester.WheelValueGuard(app)
    app.installEventFilter(guard)
    try:
        app.processEvents()
        focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
        scroll.verticalScrollBar().setValue(0)
        app.processEvents()

        line_edit = spin.lineEdit()
        QApplication.sendEvent(line_edit, _wheel_event(line_edit))

        assert spin.value() == 5
        assert scroll.verticalScrollBar().value() > 0
    finally:
        app.removeEventFilter(guard)
        scroll.close()
        scroll.deleteLater()
        app.processEvents()


def _camera_page():
    """Build a real camera-owner form from the adapter schema."""
    from camera_adapters import camera_schema
    from dlc_manager.pages.routing import RoutingPage

    page = RoutingPage()
    page.rebuild_camera_controls(
        device_id="arch-webcam",
        label="Arch USB webcam",
        stack="arch-v4l2",
        schema=camera_schema("arch-v4l2"),
        defaults={},
    )
    return page


def _guarded_value(widget):
    from dlc_manager.widgets import ValueSlider

    if isinstance(widget, ValueSlider):
        return widget.slider, widget.value()
    if isinstance(widget, QComboBox):
        return widget, widget.currentIndex()
    return widget, widget.isChecked()


def _current_value(widget):
    from dlc_manager.widgets import ValueSlider

    if isinstance(widget, ValueSlider):
        return widget.value()
    if isinstance(widget, QComboBox):
        return widget.currentIndex()
    return widget.isChecked()


def test_dynamically_generated_camera_controls_are_also_wheel_safe():
    """The guard is global, so adapter-generated controls cannot bypass it."""
    app = _application()
    guard = tester.WheelValueGuard(app)
    app.installEventFilter(guard)
    page = _camera_page()
    page.resize(520, 320)
    page.show()
    app.processEvents()
    try:
        assert page.camera_controls, "the arch adapter advertises controls"
        for key, widget in page.camera_controls.items():
            target, original = _guarded_value(widget)
            QApplication.sendEvent(target, _wheel_event(target))
            app.processEvents()
            assert _current_value(widget) == original, key
    finally:
        app.removeEventFilter(guard)
        page.close()
        page.deleteLater()
        app.processEvents()


def test_wheeling_a_generated_control_scrolls_its_page_instead():
    app = _application()
    guard = tester.WheelValueGuard(app)
    app.installEventFilter(guard)
    page = _camera_page()
    page.resize(480, 260)
    page.show()
    app.processEvents()
    scroll = page.findChild(QScrollArea, "workspaceScroll")
    try:
        assert scroll is not None
        scroll.verticalScrollBar().setValue(0)
        app.processEvents()
        widget = page.camera_controls["brightness"]
        target, original = _guarded_value(widget)

        QApplication.sendEvent(target, _wheel_event(target))
        app.processEvents()

        assert page.camera_controls["brightness"].value() == original
        assert scroll.verticalScrollBar().value() > 0
    finally:
        app.removeEventFilter(guard)
        page.close()
        page.deleteLater()
        app.processEvents()
