#!/usr/bin/env python3
"""Measured, named profiles for the Arch-owned physical camera.

Profiles are bundles rather than implicit image filters.  The values below are
the V4L2 settings measured on the Sonix camera, plus the capture mode required
to reproduce the measurement.  Manual exposure and white-balance values are
retained as safe fallbacks while their automatic modes are enabled.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CUSTOM_CAMERA_PROFILE = "custom"
DEFAULT_CAMERA_PROFILE = "natural-indoor"

ARCH_CAMERA_PROFILES: dict[str, dict[str, Any]] = {
    DEFAULT_CAMERA_PROFILE: {
        "id": DEFAULT_CAMERA_PROFILE,
        "label": "Natural indoor (measured)",
        "description": (
            "Neutral indoor color and exposure measured on the Sonix webcam; "
            "keeps 30 FPS and avoids digital oversharpening."
        ),
        "measured": True,
        "measurement_revision": "v4",
        "capture": {
            "input_format": "mjpeg",
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
        "controls": {
            "brightness": -12,
            "contrast": 12,
            "saturation": 40,
            "hue": 0,
            "gamma": 95,
            "gain": 0,
            "sharpness": 2,
            "backlight_compensation": 0,
            "power_line_frequency": 1,
            "auto_exposure": True,
            "exposure_time_absolute": 157,
            "exposure_dynamic_framerate": False,
            "auto_white_balance": True,
            "white_balance_temperature": 4600,
        },
    },
}


_CONTROL_TO_CONFIG = {
    "brightness": "CAMERA_BRIGHTNESS",
    "contrast": "CAMERA_CONTRAST",
    "saturation": "CAMERA_SATURATION",
    "hue": "CAMERA_HUE",
    "gamma": "CAMERA_GAMMA",
    "gain": "CAMERA_GAIN",
    "sharpness": "CAMERA_SHARPNESS",
    "backlight_compensation": "CAMERA_BACKLIGHT",
    "power_line_frequency": "CAMERA_POWER_LINE",
    "auto_exposure": "CAMERA_AUTO_EXPOSURE",
    "exposure_time_absolute": "CAMERA_EXPOSURE",
    "exposure_dynamic_framerate": "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE",
    "auto_white_balance": "CAMERA_AUTO_WHITE_BALANCE",
    "white_balance_temperature": "CAMERA_WHITE_BALANCE",
}


def camera_profile(name: str) -> dict[str, Any]:
    """Return an isolated profile value so callers cannot mutate the registry."""

    try:
        return deepcopy(ARCH_CAMERA_PROFILES[name])
    except KeyError as exc:
        raise ValueError(f"unknown camera profile: {name}") from exc


def profile_live_values(name: str) -> dict[str, int | bool | str]:
    """Return adapter-shaped values for previewing a named profile live."""

    profile = camera_profile(name)
    capture = profile["capture"]
    return {
        "capture_size": f"{capture['width']}x{capture['height']}",
        **profile["controls"],
    }


def profile_config_values(name: str) -> dict[str, str]:
    """Return the complete persistent environment bundle for a profile."""

    profile = camera_profile(name)
    capture = profile["capture"]
    values = {
        "CAMERA_PROFILE": name,
        "CAMERA_INPUT_FORMAT": str(capture["input_format"]),
        "CAMERA_WIDTH": str(capture["width"]),
        "CAMERA_HEIGHT": str(capture["height"]),
        "CAMERA_FPS": str(capture["fps"]),
    }
    for control, value in profile["controls"].items():
        config_key = _CONTROL_TO_CONFIG[control]
        if isinstance(value, bool):
            values[config_key] = "1" if value else "0"
        else:
            values[config_key] = str(value)
    return values


def profile_descriptors() -> list[dict[str, Any]]:
    """Return JSON-safe profile descriptions for capability-driven clients."""

    return [camera_profile(name) for name in ARCH_CAMERA_PROFILES]
