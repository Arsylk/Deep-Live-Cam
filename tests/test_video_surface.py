from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QObject, QSize  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ARCH_BIN = ROOT / "arch-linux" / "bin"
sys.path.insert(0, str(ARCH_BIN))

from dlc_manager.widgets import VideoPane  # noqa: E402


def _application() -> QApplication:
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def _frame(size: QSize) -> QImage:
    image = QImage(size, QImage.Format.Format_RGB888)
    image.fill(0x205080)
    return image


class _EventCounter(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.paints = 0
        self.resizes = 0

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Paint:
            self.paints += 1
        elif event.type() == QEvent.Type.Resize:
            self.resizes += 1
        return False


def test_video_surface_geometry_does_not_follow_frame_orientation():
    application = _application()
    pane = VideoPane("RESULT", "test stream")
    pane.resize(900, 500)
    counter = _EventCounter()
    pane.video.installEventFilter(counter)
    pane.show()
    application.processEvents()

    expected_hint = pane.video.sizeHint()
    counter.resizes = 0
    for _ in range(12):
        pane.set_image(_frame(QSize(640, 360)))
        application.processEvents()
        assert pane.video.sizeHint() == expected_hint
        pane.set_image(_frame(QSize(360, 640)))
        application.processEvents()
        assert pane.video.sizeHint() == expected_hint

    # Frame orientation is paint state, not a QLabel pixmap size hint.  It
    # cannot feed back into the scroll-page/splitter geometry.
    assert pane.video.pixmap().isNull()
    assert counter.resizes == 0
    pane.close()
    pane.deleteLater()
    application.processEvents()


def test_video_surface_coalesces_frames_until_qt_can_paint():
    application = _application()
    pane = VideoPane("RESULT", "test stream")
    pane.resize(900, 500)
    counter = _EventCounter()
    pane.video.installEventFilter(counter)
    pane.show()
    application.processEvents()
    counter.paints = 0

    for _ in range(120):
        pane.set_image(_frame(QSize(360, 640)))

    # No scaled pixmap or immediate paint is produced on the decoder signal.
    assert counter.paints == 0
    application.processEvents()
    # Some headless Qt backends do not expose the window and therefore defer
    # the paint entirely; exposed backends still coalesce the whole burst to a
    # single paint.
    assert counter.paints <= 1
    pane.close()
    pane.deleteLater()
    application.processEvents()
