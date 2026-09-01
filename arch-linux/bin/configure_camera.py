#!/usr/bin/env python3
"""Persist camera settings without interrupting the current capture owner.

Image controls are handed to the sender that already owns the camera. Capture
format changes are staged for its next natural start; this helper never
restarts a service or opens the camera node itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from camera_adapters import CameraAdapterError, apply_arch_controls
from common import DEFAULT_CONFIG, DEFAULT_STATE_DIR, load_env_file
from camera_profiles import (
    ARCH_CAMERA_PROFILES,
    CUSTOM_CAMERA_PROFILE,
    DEFAULT_CAMERA_PROFILE,
    profile_config_values,
)


CAPTURE_SIZES = {
    "1920x1080", "1280x720", "1024x576", "864x480", "800x448", "640x360", "432x240"
}
RANGES = {
    "CAMERA_BRIGHTNESS": (-64, 64),
    "CAMERA_CONTRAST": (0, 64),
    "CAMERA_SATURATION": (0, 128),
    "CAMERA_HUE": (-40, 40),
    "CAMERA_GAMMA": (72, 500),
    "CAMERA_GAIN": (0, 100),
    "CAMERA_SHARPNESS": (0, 6),
    "CAMERA_BACKLIGHT": (0, 2),
    "CAMERA_POWER_LINE": (0, 2),
    "CAMERA_EXPOSURE": (1, 5000),
    "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE": (0, 1),
    "CAMERA_WHITE_BALANCE": (2800, 6500),
}


def bounded(name: str, value: int) -> int:
    minimum, maximum = RANGES[name]
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def update_config(changes: dict[str, str]) -> None:
    path = DEFAULT_CONFIG
    lines = path.read_text(encoding="utf-8").splitlines()
    found: set[str] = set()
    output: list[str] = []
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in changes:
                output.append(f"{key}={changes[key]}")
                found.add(key)
                continue
        output.append(line)
    for key, value in changes.items():
        if key not in found:
            output.append(f"{key}={value}")

    descriptor, temporary = tempfile.mkstemp(prefix=".deep-live-cam-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(output) + "\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def owner_controls(config: dict[str, str]) -> dict[str, int | bool | str]:
    """Translate the persisted environment into the owner's public schema."""

    defaults = profile_config_values(DEFAULT_CAMERA_PROFILE)
    return {
        "profile": config.get("CAMERA_PROFILE", CUSTOM_CAMERA_PROFILE),
        "capture_size": (
            f"{config.get('CAMERA_WIDTH', '1280')}x"
            f"{config.get('CAMERA_HEIGHT', '720')}"
        ),
        "brightness": int(config.get("CAMERA_BRIGHTNESS", defaults["CAMERA_BRIGHTNESS"])),
        "contrast": int(config.get("CAMERA_CONTRAST", defaults["CAMERA_CONTRAST"])),
        "saturation": int(config.get("CAMERA_SATURATION", defaults["CAMERA_SATURATION"])),
        "hue": int(config.get("CAMERA_HUE", "0")),
        "gamma": int(config.get("CAMERA_GAMMA", defaults["CAMERA_GAMMA"])),
        "gain": int(config.get("CAMERA_GAIN", "0")),
        "sharpness": int(config.get("CAMERA_SHARPNESS", defaults["CAMERA_SHARPNESS"])),
        "backlight_compensation": int(
            config.get("CAMERA_BACKLIGHT", defaults["CAMERA_BACKLIGHT"])
        ),
        "power_line_frequency": int(config.get("CAMERA_POWER_LINE", "1")),
        "auto_exposure": config.get("CAMERA_AUTO_EXPOSURE", "1") != "0",
        "exposure_time_absolute": int(config.get("CAMERA_EXPOSURE", "157")),
        "exposure_dynamic_framerate": (
            config.get(
                "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE",
                defaults["CAMERA_EXPOSURE_DYNAMIC_FRAMERATE"],
            )
            != "0"
        ),
        "auto_white_balance": config.get("CAMERA_AUTO_WHITE_BALANCE", "1") != "0",
        "white_balance_temperature": int(
            config.get("CAMERA_WHITE_BALANCE", "4600")
        ),
    }


def apply_controls(config: dict[str, str]) -> dict[str, object]:
    """Apply hot controls through the existing owner, never through V4L2."""

    state_directory = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    socket_path = state_directory / "sender-control.sock"
    return apply_arch_controls(owner_controls(config), socket_path=socket_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(ARCH_CAMERA_PROFILES))
    parser.add_argument("--capture-size", choices=sorted(CAPTURE_SIZES))
    parser.add_argument("--brightness", type=int)
    parser.add_argument("--contrast", type=int)
    parser.add_argument("--saturation", type=int)
    parser.add_argument("--hue", type=int)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--gain", type=int)
    parser.add_argument("--sharpness", type=int)
    parser.add_argument("--backlight", type=int)
    parser.add_argument("--power-line", type=int)
    parser.add_argument("--auto-exposure", type=int, choices=(0, 1))
    parser.add_argument("--exposure", type=int)
    parser.add_argument(
        "--exposure-dynamic-framerate", type=int, choices=(0, 1)
    )
    parser.add_argument("--auto-white-balance", type=int, choices=(0, 1))
    parser.add_argument("--white-balance", type=int)
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("This helper must be run through pkexec or sudo.", file=sys.stderr)
        return 1

    changes: dict[str, str] = {}
    format_changed = False
    current_config = load_env_file(DEFAULT_CONFIG)
    explicit_arguments = [
        args.capture_size,
        args.brightness,
        args.contrast,
        args.saturation,
        args.hue,
        args.gamma,
        args.gain,
        args.sharpness,
        args.backlight,
        args.power_line,
        args.auto_exposure,
        args.exposure,
        args.exposure_dynamic_framerate,
        args.auto_white_balance,
        args.white_balance,
    ]
    if args.profile and any(value is not None for value in explicit_arguments):
        parser.error("--profile cannot be combined with individual camera settings")
    if args.profile:
        changes.update(profile_config_values(args.profile))
        format_changed = any(
            current_config.get(key) != changes[key]
            for key in (
                "CAMERA_INPUT_FORMAT",
                "CAMERA_WIDTH",
                "CAMERA_HEIGHT",
                "CAMERA_FPS",
            )
        )
    if args.capture_size:
        width, height = args.capture_size.split("x", 1)
        changes["CAMERA_WIDTH"] = width
        changes["CAMERA_HEIGHT"] = height
        format_changed = (
            current_config.get("CAMERA_WIDTH", "1280") != width
            or current_config.get("CAMERA_HEIGHT", "720") != height
        )
    mappings = {
        "brightness": "CAMERA_BRIGHTNESS", "contrast": "CAMERA_CONTRAST",
        "saturation": "CAMERA_SATURATION", "hue": "CAMERA_HUE", "gamma": "CAMERA_GAMMA",
        "gain": "CAMERA_GAIN", "sharpness": "CAMERA_SHARPNESS", "backlight": "CAMERA_BACKLIGHT",
        "power_line": "CAMERA_POWER_LINE", "exposure": "CAMERA_EXPOSURE",
        "exposure_dynamic_framerate": "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE",
        "white_balance": "CAMERA_WHITE_BALANCE",
    }
    for argument, config_name in mappings.items():
        value = getattr(args, argument)
        if value is not None:
            changes[config_name] = str(bounded(config_name, value))
    if args.auto_exposure is not None:
        changes["CAMERA_AUTO_EXPOSURE"] = str(args.auto_exposure)
    if args.auto_white_balance is not None:
        changes["CAMERA_AUTO_WHITE_BALANCE"] = str(args.auto_white_balance)
    if not args.profile and changes:
        changes["CAMERA_PROFILE"] = CUSTOM_CAMERA_PROFILE
    if not changes:
        parser.error("at least one setting is required")

    update_config(changes)
    config = load_env_file(DEFAULT_CONFIG)
    result: dict[str, object] = {
        "ok": True,
        "persisted": True,
        "capture_format": (
            "staged-next-owner-start" if format_changed else "unchanged"
        ),
        "owner_restarted": False,
    }
    try:
        applied = apply_controls(config)
        result["live_controls_applied"] = True
        result["controls"] = applied.get("controls", owner_controls(config))
    except CameraAdapterError as exc:
        # Persistence succeeded. The sender will read the complete saved bundle
        # on its next natural start, but this helper must not start it merely to
        # make a Save button look successful.
        result["live_controls_applied"] = False
        result["live_detail"] = str(exc)
    if format_changed:
        result["detail"] = (
            "image controls saved; capture format is staged for the next "
            "natural capture-owner start; no active session was restarted"
        )
    else:
        result["detail"] = (
            "settings saved and hot controls sent to the existing capture "
            "owner; no active session was restarted"
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
