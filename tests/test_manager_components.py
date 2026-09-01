"""Reusable manager components: grids, pills, log discipline, and baselines."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from camera_adapters import camera_schema  # noqa: E402
from dlc_manager import services, theme  # noqa: E402
from dlc_manager.baseline import default_pointer_path, load_active_baseline  # noqa: E402
from dlc_manager.contracts import (  # noqa: E402
    CAMERA_CONTROL_GROUPS,
    group_camera_controls,
    local_mpegts_preview_command,
)
from dlc_manager.decoders import LogRateLimiter  # noqa: E402
from dlc_manager.health import default_android_native_health_file  # noqa: E402
from dlc_manager.viewmodel import SlotView  # noqa: E402
from dlc_manager.widgets import (  # noqa: E402
    ElidedLabel,
    FramingPreview,
    ResponsiveCardGrid,
    SlotCard,
    StatusPill,
)


def _application() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_manager_defaults_to_the_system_processor_health_snapshot():
    assert default_android_native_health_file() == Path(
        "/run/deep-live-cam/android-phone-processed-health.json"
    )


def test_windows_source_request_identifies_exact_upload_bytes():
    app = _application()
    client = services.WindowsControlClient("192.0.2.1")
    data = b"exact source upload bytes\x00\xff"
    try:
        request = client._source_upload_request(data, "portrait.png", "image/png")
        assert bytes(request.rawHeader("X-Source-Identifier")) == hashlib.sha256(
            data
        ).hexdigest().encode("ascii")
        assert bytes(request.rawHeader("X-Filename")) == b"portrait.png"
    finally:
        client.deleteLater()
        app.processEvents()


# ------------------------------------------------------------- preview command


def test_a_preview_command_only_ever_reads_a_loopback_relay():
    rendered = " ".join(local_mpegts_preview_command(11_007))

    assert "udp://127.0.0.1:11007?" in rendered
    assert "/dev/" not in rendered
    assert "v4l2" not in rendered
    assert "srt://" not in rendered
    assert rendered.count(" -i ") == 1


@pytest.mark.parametrize("port", [0, -1, 70_000])
def test_a_preview_command_refuses_an_impossible_port(port):
    with pytest.raises(ValueError):
        local_mpegts_preview_command(port)


def test_preview_fits_the_source_with_the_same_cover_crop_as_the_output():
    # The preview must frame the source exactly as the stable camera publishes
    # it: scale-to-cover then centre-crop, never an aspect-breaking stretch.
    rendered = " ".join(local_mpegts_preview_command(11_007))

    assert "force_original_aspect_ratio=increase" in rendered
    assert "crop=640:360" in rendered
    assert "scale=640:360:flags" not in rendered


# ------------------------------------------------------------- responsive grid


def test_a_card_grid_chooses_its_columns_from_the_available_width():
    app = _application()
    grid = ResponsiveCardGrid(minimum_card_width=320, spacing=10)
    grid.set_cards([QLabel(str(index)) for index in range(5)])
    try:
        assert grid.columns_for(1500) == 4
        assert grid.columns_for(760) == 2
        assert grid.columns_for(400) == 1
        # Never more columns than there are cards, and never fewer than one.
        assert grid.columns_for(10_000) == 5
        assert grid.columns_for(0) == 1
    finally:
        grid.deleteLater()
        app.processEvents()


def test_a_card_grid_actually_reflows_when_it_is_widened():
    app = _application()
    host = QWidget()
    layout = QVBoxLayout(host)
    grid = ResponsiveCardGrid(minimum_card_width=320, spacing=10)
    layout.addWidget(grid)
    grid.set_cards([QLabel(str(index)) for index in range(5)])
    host.resize(1600, 300)
    host.show()
    app.processEvents()
    try:
        wide = grid.columns()
        assert wide >= 3, grid.width()
        positions = {
            grid.layout().getItemPosition(index)[:2] for index in range(5)
        }
        assert len(positions) == 5
    finally:
        host.close()
        host.deleteLater()
        app.processEvents()


def test_a_card_grid_never_demands_more_than_one_card_of_width():
    app = _application()
    grid = ResponsiveCardGrid(minimum_card_width=330)
    grid.set_cards([QLabel(str(index)) for index in range(5)])
    grid.resize(1600, 200)
    app.processEvents()

    assert grid.minimumSizeHint().width() <= 330
    grid.deleteLater()
    app.processEvents()


# --------------------------------------------------------------- status states


def test_every_state_pairs_colour_with_a_word_and_a_glyph():
    app = _application()
    pill = StatusPill()
    try:
        for state in theme.STATE_TOKENS:
            pill.set_state("READY", state)
            glyph, word = pill.text().split(" ", 1)
            assert glyph == theme.state_glyph(state)
            assert word == "READY"
            assert "color:" in pill.styleSheet()
    finally:
        pill.deleteLater()
        app.processEvents()


def test_a_slot_card_shows_identity_capability_endpoint_and_readiness():
    app = _application()
    card = SlotCard(1)
    view = SlotView(
        slot=1,
        device_id="arch-webcam",
        label="Arch USB webcam",
        stack="arch-v4l2",
        input_port=10_002,
        return_port=10_003,
        return_host="192.168.1.11",
        configured=True,
        enabled=True,
        selected=True,
        selectable=True,
        state="active",
        state_text="ACTIVE · LIVE",
        endpoint="SRT 10002 → Windows → 192.168.1.11:10003",
        capability="V4L2 capture owner · full camera controls",
    )
    try:
        card.render(view)

        assert "SLOT 1" in card.identity.text()
        assert "ARCH USB WEBCAM" in card.identity.text()
        assert "arch-v4l2" in card.capability.text()
        assert card.endpoint.text() == view.endpoint
        assert card.pill.text().endswith("ACTIVE · LIVE")
        assert card.isChecked() is True
        assert card.property("routeState") == "active"
        assert "no camera is opened" in card.toolTip()
    finally:
        card.deleteLater()
        app.processEvents()


def test_an_elided_label_keeps_the_full_text_available():
    app = _application()
    label = ElidedLabel()
    label.resize(60, 20)
    try:
        label.setText("Local INSwapper-128 result → /dev/deep-live-cam")

        assert label.full_text().endswith("/dev/deep-live-cam")
        assert label.toolTip() == label.full_text()
        assert label.minimumSizeHint().width() <= 24
    finally:
        label.deleteLater()
        app.processEvents()


# --------------------------------------------------------------- log discipline


def test_repeated_decoder_startup_warnings_are_aggregated_once():
    limiter = LogRateLimiter()
    limiter.restart(0.0)

    shown = [
        limiter.observe("raw: [h264 @ 0x1] non-existing PPS 0 referenced", 0.1 * step)
        for step in range(1, 12)
    ]

    assert shown == [None] * 11
    drained = limiter.drain(60.0)
    assert len(drained) == 1
    assert "aggregated" in drained[0]


def test_an_actionable_failure_is_shown_immediately():
    limiter = LogRateLimiter()
    limiter.restart(0.0)

    first = limiter.observe("raw: Connection refused", 0.2)
    repeat = limiter.observe("raw: Connection refused", 0.4)

    assert first == "raw: Connection refused"
    assert repeat is None
    assert any("repeated" in line for line in limiter.drain(60.0))


def test_distinct_failures_are_not_collapsed_together():
    limiter = LogRateLimiter()
    limiter.restart(0.0)

    assert limiter.observe("raw: Connection refused", 5.0) is not None
    assert limiter.observe("raw: Invalid data found when processing input", 5.1) is not None


# ------------------------------------------------------- camera control groups


def test_adapter_controls_are_grouped_semantically():
    grouped = group_camera_controls(camera_schema("arch-v4l2")["controls"])
    titles = [group.key for group, _members in grouped]

    assert titles == ["profile", "tone", "exposure"]
    profile_keys = [
        control["key"] for group, members in grouped for control in members
        if group.key == "profile"
    ]
    assert profile_keys == ["profile", "capture_size"]


def test_an_android_adapter_uses_its_own_groups():
    grouped = group_camera_controls(camera_schema("android-camera2")["controls"])
    keys = {group.key for group, _members in grouped}

    assert "optics" in keys
    assert "exposure" in keys


def test_an_unknown_control_still_renders_in_a_trailing_group():
    grouped = group_camera_controls(
        [
            {"key": "brightness", "label": "Brightness", "kind": "integer"},
            {"key": "future_thing", "label": "Future thing", "kind": "boolean"},
        ]
    )

    assert [group.key for group, _members in grouped] == ["tone", "other"]
    assert grouped[-1][1][0]["key"] == "future_thing"


def test_every_declared_group_has_a_plain_language_description():
    for group in CAMERA_CONTROL_GROUPS:
        assert group.title and group.detail


# -------------------------------------------------------------------- baseline


def test_the_registered_baseline_is_readable_with_its_limitations():
    baseline = load_active_baseline(default_pointer_path(ROOT))

    assert baseline.available is True
    assert baseline.identifier == "arch-webcam-inswapper128-natural-indoor-v1"
    assert baseline.camera_profile == "natural-indoor"
    assert baseline.model == "inswapper-128"
    assert baseline.frame_count == 50
    assert baseline.interpretation["identity"]
    assert "signal" in baseline.interpretation["full_reference_metrics"]
    labels = [label for label, _value, _meaning in baseline.metrics()]
    assert labels == [
        "Processing rate",
        "Face stability",
        "Detector misses",
        "Seam colour delta",
        "Compared frames",
    ]
    assert all(meaning for _label, _value, meaning in baseline.metrics())


def test_a_missing_baseline_pointer_is_explained_not_faked(tmp_path):
    baseline = load_active_baseline(tmp_path / "absent.json")

    assert baseline.available is False
    assert "absent.json" in baseline.error


# ------------------------------------------------------------ framing preview


def test_framing_preview_drag_maps_screen_pixels_to_source_offset():
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QImage, QMouseEvent

    app = _application()
    # Source is 1280 wide; render surface is forced to 640 so the drag scale
    # factor is exactly 2 source px per screen px.
    preview = FramingPreview(1280, 720)
    preview.resize(640, 360)
    frame = QImage(640, 360, QImage.Format.Format_RGB888)
    frame.fill(0x808080)
    preview.set_image(frame)
    try:
        committed: list[tuple[int, int]] = []
        live: list[tuple[int, int]] = []
        preview.offsetDragged.connect(lambda x, y: committed.append((x, y)))
        preview.offsetPreview.connect(lambda x, y: live.append((x, y)))

        def press(x, y):
            return QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPointF(x, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )

        def move(x, y):
            return QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPointF(x, y),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )

        def release(x, y):
            return QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPointF(x, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )

        preview.mousePressEvent(press(100, 100))
        preview.mouseMoveEvent(move(150, 120))  # +50px, +20px on screen
        # Live preview fires during the drag, but nothing is committed yet.
        assert live and committed == []
        # 640-wide surface fitting a 640x360 frame → scale 1280/640 = 2.
        assert live[-1] == (100, 40)

        preview.mouseReleaseEvent(release(150, 120))
        # Exactly one commit on release, matching the final live value.
        assert committed == [(100, 40)]
    finally:
        preview.deleteLater()
        app.processEvents()


def test_framing_preview_external_offset_and_clear_are_honoured():
    from PySide6.QtGui import QImage

    app = _application()
    preview = FramingPreview(1280, 720)
    try:
        preview.set_offset(120, -40)
        # A decoded frame is accepted and retained.
        frame = QImage(640, 360, QImage.Format.Format_RGB888)
        frame.fill(0)
        preview.set_image(frame)
        assert preview._image is not None
        # Clearing drops the frame and shows the waiting message.
        preview.clear_image("stopped")
        assert preview._image is None
    finally:
        preview.deleteLater()
        app.processEvents()


def test_framing_preview_grid_toggle_is_on_by_default_and_switchable():
    app = _application()
    preview = FramingPreview(1280, 720)
    try:
        # The positioning grid is a first-paint default so the user always has
        # a reference; it can be turned off without touching the frame.
        assert preview._show_grid is True
        preview.set_grid_visible(False)
        assert preview._show_grid is False
        preview.set_grid_visible(True)
        assert preview._show_grid is True
    finally:
        preview.deleteLater()
        app.processEvents()
