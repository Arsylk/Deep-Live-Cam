from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QColor, QImage


ROOT = Path(__file__).resolve().parents[1]
ARCH_BIN = ROOT / "arch-linux" / "bin"
sys.path.insert(0, str(ARCH_BIN))

from dlc_manager.preview_transform import transform_preview_image  # noqa: E402


def _image() -> QImage:
    image = QImage(2, 3, QImage.Format.Format_RGB32)
    colors = ("red", "green", "blue", "yellow", "magenta", "cyan")
    for index, name in enumerate(colors):
        image.setPixelColor(index % 2, index // 2, QColor(name))
    return image


def _square_image() -> QImage:
    image = QImage(2, 2, QImage.Format.Format_RGB32)
    for index, name in enumerate(("red", "green", "blue", "yellow")):
        image.setPixelColor(index % 2, index // 2, QColor(name))
    return image


def _names(image: QImage) -> list[list[str]]:
    return [
        [image.pixelColor(x, y).name() for x in range(image.width())]
        for y in range(image.height())
    ]


def test_rotation_then_final_coordinate_mirror_matches_output_contract():
    transformed = transform_preview_image(_square_image(), rotation=90, mirror=True)

    assert (transformed.width(), transformed.height()) == (2, 2)
    assert _names(transformed) == [
        [QColor("red").name(), QColor("blue").name()],
        [QColor("green").name(), QColor("yellow").name()],
    ]


def test_quarter_turn_cover_crop_keeps_geometry_and_has_no_letterbox_pixels():
    image = QImage(16, 9, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))

    for rotation in (90, 270):
        transformed = transform_preview_image(image, rotation=rotation)

        assert (transformed.width(), transformed.height()) == (16, 9)
        assert all(
            transformed.pixelColor(x, y) == QColor("white")
            for y in range(transformed.height())
            for x in range(transformed.width())
        )


def test_identity_transform_returns_original_image_pixels():
    image = _image()
    transformed = transform_preview_image(image)
    assert _names(transformed) == _names(image)
