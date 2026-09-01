#!/usr/bin/env python3
"""Capability-driven camera configuration adapters for client-owned capture.

Adapters send allow-listed settings to the process that already owns a camera.
They never open a physical or virtual camera node themselves.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import socket
from pathlib import Path
from typing import Any

from camera_profiles import (
    ARCH_CAMERA_PROFILES,
    CUSTOM_CAMERA_PROFILE,
    DEFAULT_CAMERA_PROFILE,
    profile_descriptors,
    profile_live_values,
)


@dataclass(frozen=True)
class CameraControl:
    key: str
    label: str
    kind: str
    minimum: int | None = None
    maximum: int | None = None
    default: int | bool | str | None = None
    choices: tuple[str, ...] = ()


ARCH_CONTROLS = (
    CameraControl(
        "profile",
        "Camera profile",
        "choice",
        default=DEFAULT_CAMERA_PROFILE,
        choices=(*ARCH_CAMERA_PROFILES, CUSTOM_CAMERA_PROFILE),
    ),
    CameraControl(
        "capture_size",
        "Capture resolution",
        "choice",
        default="1280x720",
        choices=(
            "1920x1080",
            "1280x720",
            "1024x576",
            "864x480",
            "800x448",
            "640x360",
            "432x240",
        ),
    ),
    CameraControl("brightness", "Brightness", "integer", -64, 64, -12),
    CameraControl("contrast", "Contrast", "integer", 0, 64, 12),
    CameraControl("saturation", "Saturation", "integer", 0, 128, 40),
    CameraControl("hue", "Hue", "integer", -40, 40, 0),
    CameraControl("gamma", "Gamma", "integer", 72, 500, 95),
    CameraControl("gain", "Gain", "integer", 0, 100, 0),
    CameraControl("sharpness", "Sharpness", "integer", 0, 6, 2),
    CameraControl(
        "backlight_compensation", "Backlight compensation", "integer", 0, 2, 0
    ),
    CameraControl("power_line_frequency", "Anti-flicker", "integer", 0, 2, 1),
    CameraControl("auto_exposure", "Automatic exposure", "boolean", default=True),
    CameraControl("exposure_time_absolute", "Exposure", "integer", 1, 5000, 157),
    CameraControl(
        "exposure_dynamic_framerate",
        "Allow exposure to reduce frame rate",
        "boolean",
        default=False,
    ),
    CameraControl(
        "auto_white_balance", "Automatic white balance", "boolean", default=True
    ),
    CameraControl(
        "white_balance_temperature", "White balance", "integer", 2800, 6500, 4600
    ),
)

ANDROID_CONTROLS = (
    CameraControl(
        "lens_facing",
        "Lens",
        "choice",
        default="front",
        choices=("front", "back"),
    ),
    CameraControl(
        "rotation",
        "Orientation (clockwise)",
        "choice",
        default="auto",
        choices=("auto", "0", "90", "180", "270"),
    ),
    CameraControl(
        "zoom_percent",
        "Digital zoom",
        "integer",
        100,
        300,
        100,
    ),
    CameraControl(
        "exposure_compensation", "Exposure compensation", "integer", -12, 12, 0
    ),
    CameraControl("ae_lock", "Lock automatic exposure", "boolean", default=False),
    CameraControl("awb_lock", "Lock automatic white balance", "boolean", default=False),
    CameraControl(
        "stabilization",
        "Stabilization",
        "choice",
        default="video",
        choices=("off", "video", "optical"),
    ),
)

SCHEMAS = {
    "arch-v4l2": ARCH_CONTROLS,
    "android-camera2": ANDROID_CONTROLS,
    "generic-srt": (),
}


class CameraAdapterError(RuntimeError):
    pass


def camera_schema(stack: str) -> dict[str, Any]:
    if stack not in SCHEMAS:
        stack = "generic-srt"
    schema = {
        "stack": stack,
        "ownership": "client",
        "manager_opens_camera": False,
        "controls": [asdict(control) for control in SCHEMAS[stack]],
    }
    if stack == "arch-v4l2":
        schema["profiles"] = profile_descriptors()
    return schema


def normalize_controls(stack: str, values: object) -> dict[str, int | bool | str]:
    if stack not in SCHEMAS:
        raise CameraAdapterError(f"unsupported camera stack: {stack}")
    if not isinstance(values, dict):
        raise CameraAdapterError("controls must be an object")
    specifications = {control.key: control for control in SCHEMAS[stack]}
    normalized: dict[str, int | bool | str] = {}
    for key, raw in values.items():
        specification = specifications.get(key)
        if specification is None:
            raise CameraAdapterError(f"{stack} does not support {key}")
        if specification.kind == "boolean":
            if not isinstance(raw, bool):
                raise CameraAdapterError(f"{key} must be true or false")
            normalized[key] = raw
        elif specification.kind == "choice":
            value = str(raw)
            if value not in specification.choices:
                raise CameraAdapterError(
                    f"{key} must be one of {', '.join(specification.choices)}"
                )
            normalized[key] = value
        else:
            if isinstance(raw, bool):
                raise CameraAdapterError(f"{key} must be an integer")
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise CameraAdapterError(f"{key} must be an integer") from exc
            if (
                specification.minimum is None
                or specification.maximum is None
                or not specification.minimum <= value <= specification.maximum
            ):
                raise CameraAdapterError(
                    f"{key} must be between {specification.minimum} and "
                    f"{specification.maximum}"
                )
            normalized[key] = value
    return normalized


def apply_arch_controls(
    values: dict[str, int | bool | str],
    socket_path: Path = Path("/run/deep-live-cam/sender-control.sock"),
) -> dict[str, Any]:
    controls = normalize_controls("arch-v4l2", values)
    selected_profile = controls.pop("profile", None)
    if selected_profile and selected_profile != CUSTOM_CAMERA_PROFILE:
        controls = {
            **profile_live_values(str(selected_profile)),
            **controls,
        }
    # Resolution requires reopening the capture owner and is therefore only
    # acted on by the explicit persistent-save path in the native manager.
    live_controls = {
        key: value for key, value in controls.items() if key != "capture_size"
    }
    if live_controls.get("auto_exposure") is True:
        live_controls.pop("exposure_time_absolute", None)
    if live_controls.get("auto_white_balance") is True:
        live_controls.pop("white_balance_temperature", None)
    if not live_controls:
        result = {
            "ok": True,
            "controls": controls,
            "detail": "resolution will apply when current settings are saved",
        }
        if selected_profile:
            result["profile"] = selected_profile
        return result
    request = json.dumps({"controls": live_controls}).encode("utf-8")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(4.0)
    try:
        client.connect(str(socket_path))
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while len(response) <= 65_536:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    except OSError as exc:
        raise CameraAdapterError(
            "the Arch capture owner is unavailable; no camera was opened by the manager"
        ) from exc
    finally:
        client.close()
    try:
        result = json.loads(response)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CameraAdapterError("invalid response from the Arch capture owner") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise CameraAdapterError(str(result.get("error") or result.get("detail") or "apply failed"))
    if selected_profile:
        result["profile"] = selected_profile
    return result


def apply_android_controls(
    values: dict[str, int | bool | str],
    *,
    serial: str = "",
    host: str = "",
    persist: bool = False,
) -> dict[str, Any]:
    controls = normalize_controls("android-camera2", values)
    from android_bridge import configure_camera

    configure_camera(controls, serial, host, persist=persist)
    return {
        "ok": True,
        "controls": controls,
        "persisted": persist,
        "detail": "sent to Camera2 owner",
    }


def apply_configuration(
    stack: str,
    values: dict[str, Any],
    *,
    serial: str = "",
    host: str = "",
    persist: bool = False,
    socket_path: Path = Path("/run/deep-live-cam/sender-control.sock"),
) -> dict[str, Any]:
    if stack == "arch-v4l2":
        return apply_arch_controls(values, socket_path)
    if stack == "android-camera2":
        return apply_android_controls(
            values, serial=serial, host=host, persist=persist
        )
    raise CameraAdapterError(
        "this device has no camera-control adapter; its stream remains usable"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True, choices=sorted(SCHEMAS))
    parser.add_argument("--controls-json", required=True)
    parser.add_argument("--serial", default="")
    parser.add_argument("--host", default="")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="persist Android settings; live preview is the default",
    )
    parser.add_argument("--socket", default="/run/deep-live-cam/sender-control.sock")
    args = parser.parse_args()
    try:
        values = json.loads(args.controls_json)
        result = apply_configuration(
            args.stack,
            values,
            serial=args.serial,
            host=args.host,
            persist=args.persist,
            socket_path=Path(args.socket),
        )
    except (CameraAdapterError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
