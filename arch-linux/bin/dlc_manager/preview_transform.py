#!/usr/bin/env python3
"""Pure display transform matching the stable output-coordinate contract."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QTransform


def transform_preview_image(
    image: QImage, *, mirror: bool = False, rotation: int = 0
) -> QImage:
    """Rotate clockwise, centre-crop to fill, then mirror final coordinates.

    Preview geometry stays identical to the decoded frame geometry, matching
    the fixed virtual-camera output contract. In particular, a 90/270 degree
    selection cannot turn a landscape preview into a small letterboxed
    portrait thumbnail.
    """
    if not isinstance(mirror, bool):
        raise TypeError("mirror must be a boolean")
    if isinstance(rotation, bool) or rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if image.isNull() or (rotation == 0 and not mirror):
        return image
    target_width = image.width()
    target_height = image.height()
    result = image
    if rotation:
        result = result.transformed(
            QTransform().rotate(rotation),
            Qt.TransformationMode.FastTransformation,
        )
    if (result.width(), result.height()) != (target_width, target_height):
        # Crop the rotated source to the destination aspect *before* scaling.
        # Scaling a 720x1280 quarter-turn to cover 1280x720 first would create
        # a wasteful 1280x2276 intermediate for every preview frame.
        source_width = result.width()
        source_height = result.height()
        if source_width * target_height > source_height * target_width:
            crop_width = max(
                1,
                min(
                    source_width,
                    round(source_height * target_width / target_height),
                ),
            )
            result = result.copy(
                (source_width - crop_width) // 2,
                0,
                crop_width,
                source_height,
            )
        else:
            crop_height = max(
                1,
                min(
                    source_height,
                    round(source_width * target_height / target_width),
                ),
            )
            result = result.copy(
                0,
                (source_height - crop_height) // 2,
                source_width,
                crop_height,
            )
        result = result.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
    if mirror:
        result = result.flipped(Qt.Orientation.Horizontal)
    return result


__all__ = ["transform_preview_image"]
