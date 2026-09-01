#!/usr/bin/env python3
"""Inspect and control the companion Android camera bridge over local ADB."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


PACKAGE = "dev.vcam.app"
ACTIVITY = f"{PACKAGE}/.MainActivity"
SERVICE = f"{PACKAGE}/.CameraBridgeService"
DEFAULT_CAMERA_ID = "120"
OUTPUT_CONTROL_FILE = "/data/adb/android-vcam-output.conf"
OUTPUT_STATE_FILE = "/data/local/tmp/android-vcam-output.state"
ACTIONS = (
    "status",
    "start",
    "stop",
    "restart",
    "configure",
    "configure-output",
)


class AndroidBridgeError(RuntimeError):
    pass


def _run(command: Sequence[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AndroidBridgeError(str(exc)) from exc


def parse_adb_devices(output: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        values = {"serial": parts[0], "state": parts[1]}
        for part in parts[2:]:
            if ":" in part:
                key, value = part.split(":", 1)
                values[key] = value
        devices.append(values)
    return devices


def _adb_path() -> str:
    path = shutil.which("adb")
    if not path:
        raise AndroidBridgeError("adb is not installed")
    return path


def _adb(adb: str, serial: str, *args: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return _run([adb, "-s", serial, *args], timeout=timeout)


def _select_device(
    adb: str,
    preferred_serial: str = "",
    expected_host: str = "",
) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    result = _run([adb, "devices", "-l"], timeout=4.0)
    devices = parse_adb_devices(result.stdout)
    if preferred_serial:
        selected = next(
            (device for device in devices if device.get("serial") == preferred_serial),
            None,
        )
        if selected is not None:
            return selected, devices
    if expected_host:
        matching_host = [
            device
            for device in devices
            if device.get("state") == "device"
            and device.get("serial", "").split(":", 1)[0] == expected_host
        ]
        if len(matching_host) == 1:
            return matching_host[0], devices
    ready = [device for device in devices if device.get("state") == "device"]
    return (ready[0] if len(ready) == 1 else None), devices


def _section(output: str, name: str, following: str | None = None) -> str:
    marker = f"__{name}__"
    if marker not in output:
        return ""
    value = output.split(marker, 1)[1]
    if following and f"__{following}__" in value:
        value = value.split(f"__{following}__", 1)[0]
    return value.strip()


def _bool_value(values: dict[str, str], key: str) -> bool:
    return values.get(key) == "1"


def _int_value(
    values: dict[str, str], key: str, default: int | None = None
) -> int | None:
    try:
        return int(values[key])
    except (KeyError, TypeError, ValueError):
        return default


def _module_supports_output_control(version: str | None) -> bool:
    if not version:
        return False
    match = re.search(r"(?:^|\D)(\d+)\.(\d+)\.(\d+)(?:\D|$)", version)
    if not match:
        return False
    return tuple(int(part) for part in match.groups()) >= (0, 4, 6)


def _key_values(output: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for line in output.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def _vcam_log_message(line: str) -> str:
    """Remove common logcat prefixes while retaining legacy plain messages."""
    stripped = line.strip()
    brief = re.search(r"\b[VDIWEF]/VCamBridge(?:\(\s*\d+\))?:\s*(.*)$", stripped)
    if brief:
        return brief.group(1).strip()
    threadtime = re.search(
        r"\b[VDIWEF]\s+VCamBridge(?:\(\s*\d+\))?:?\s*(.*)$", stripped
    )
    if threadtime:
        return threadtime.group(1).strip()
    legacy = re.search(r"\b[VDIWEF]\s+VCamBridge:\s*(.*)$", stripped)
    return legacy.group(1).strip() if legacy else stripped


def _parse_legacy_capture_metrics(output: str) -> dict[str, Any]:
    metric_line = next(
        (line for line in reversed(output.splitlines()) if "encoded=" in line),
        "",
    )
    metric_pattern = re.compile(
        r"encoded=(?P<encoded>\d+)\s+fps=(?P<fps>[\d.]+)\s+"
        r"bitrate=(?P<bitrate>[\d.]+)Mbps\s+"
        r"captureInterval=(?P<interval>[\d.]+)ms\s+"
        r"jitter=(?P<jitter>[\d.]+)ms\s+dropped~(?P<drops>\d+)"
        r"(?:\s+exposure=(?P<exposure>[\d.]+)ms\s+ISO=(?P<iso>\d+)"
        r"\s+exposureJitter=(?P<exposure_jitter>[\d.]+)EV"
        r"\s+awbJitter=(?P<awb_jitter>[\d.]+)"
        r"\s+ae=(?P<ae>\S+)\s+awb=(?P<awb>\S+))?"
        r"(?:\s+rotation=(?P<rotation>auto|0|90|180|270)"
        r"\s+effectiveRotation=(?P<effective_rotation>\d+)"
        r"\s+rendered=(?P<rendered>\d+)"
        r"(?:\s+textureRotation=(?P<texture_rotation>\d+)"
        r"\s+shaderRotation=(?P<shader_rotation>\d+))?)?"
    )
    metric_match = metric_pattern.search(metric_line)
    if not metric_match:
        return {}
    values = metric_match.groupdict()
    return {
        "encoded_frames": int(values["encoded"]),
        "encoded_fps": float(values["fps"]),
        "bitrate_mbps": float(values["bitrate"]),
        "sensor_interval_ms": float(values["interval"]),
        "sensor_jitter_ms": float(values["jitter"]),
        "estimated_drops": int(values["drops"]),
        "exposure_ms": (
            float(values["exposure"])
            if values.get("exposure") is not None
            else None
        ),
        "sensitivity_iso": (
            int(values["iso"]) if values.get("iso") is not None else None
        ),
        "exposure_jitter_ev": (
            float(values["exposure_jitter"])
            if values.get("exposure_jitter") is not None
            else None
        ),
        "awb_gain_jitter": (
            float(values["awb_jitter"])
            if values.get("awb_jitter") is not None
            else None
        ),
        "ae_state": values.get("ae"),
        "awb_state": values.get("awb"),
        "rotation": values.get("rotation"),
        "effective_rotation_degrees": (
            int(values["effective_rotation"])
            if values.get("effective_rotation") is not None
            else None
        ),
        "rendered_frames": (
            int(values["rendered"])
            if values.get("rendered") is not None
            else None
        ),
        "texture_rotation_degrees": (
            int(values["texture_rotation"])
            if values.get("texture_rotation") is not None
            else None
        ),
        "shader_rotation_degrees": (
            int(values["shader_rotation"])
            if values.get("shader_rotation") is not None
            else None
        ),
    }


def _parse_merged_telemetry_block(lines: Sequence[str]) -> tuple[dict[str, Any], int]:
    """Parse one merged-app telemetry block and return values plus completeness."""
    metrics: dict[str, Any] = {}
    sections = 0
    for line in lines:
        message = _vcam_log_message(line)
        overview = re.search(
            r"\bup=(?P<uptime>[\d.]+)s\s+lens=(?P<lens>front|back)\s+"
            r"rot=(?P<rotation>auto|0|90|180|270)"
            r"\((?P<effective_rotation>\d+)\)\s+"
            r"stab=(?P<stabilization>\S+)\s+"
            r"(?:zoom=(?P<zoom>[\d.]+)x/(?P<max_zoom>[\d.]+)x\s+)?"
            r"exp=(?P<exposure>[+-]?\d+)\s+"
            r"ae=(?P<ae>\S+)\s+awb=(?P<awb>\S+)",
            message,
        )
        if overview:
            values = overview.groupdict()
            metrics.update(
                {
                    "uptime_seconds": float(values["uptime"]),
                    "lens_facing": values["lens"],
                    "rotation": values["rotation"],
                    "effective_rotation_degrees": int(values["effective_rotation"]),
                    "stabilization": values["stabilization"],
                    "zoom_ratio": (
                        float(values["zoom"]) if values.get("zoom") else None
                    ),
                    "maximum_zoom_ratio": (
                        float(values["max_zoom"])
                        if values.get("max_zoom")
                        else None
                    ),
                    "zoom_percent": (
                        round(float(values["zoom"]) * 100)
                        if values.get("zoom")
                        else None
                    ),
                    "exposure_compensation": int(values["exposure"]),
                    "ae_state": values["ae"],
                    "awb_state": values["awb"],
                }
            )
            sections += 1
            continue

        camera = re.search(
            r"\bcam\s*:\s*id=(?P<id>\S+)\s+state=(?P<state>\S+)\s+"
            r"cap=(?P<frames>\d+)\((?P<fps>[\d.]+)fps\)\s+"
            r"int=(?P<interval>[\d.]+)ms\s+jit=(?P<jitter>[\d.]+)ms",
            message,
        )
        if camera:
            values = camera.groupdict()
            metrics.update(
                {
                    "camera_id": values["id"],
                    "camera_state": values["state"],
                    "captured_frames": int(values["frames"]),
                    "captured_fps": float(values["fps"]),
                    "sensor_interval_ms": float(values["interval"]),
                    "sensor_jitter_ms": float(values["jitter"]),
                }
            )
            sections += 1
            continue

        encoder = re.search(
            r"\benc\s*:\s*(?P<encoder>.+?)\s+(?P<state>\S+)\s+"
            r"(?P<width>\d+)x(?P<height>\d+)@(?P<target_fps>\d+)\s+"
            r"frames=(?P<frames>\d+)\((?P<fps>[\d.]+)fps\)\s+"
            r"tx=(?P<tx>\S+)",
            message,
        )
        if encoder:
            values = encoder.groupdict()
            metrics.update(
                {
                    "encoder": values["encoder"],
                    "encoder_state": values["state"],
                    "width": int(values["width"]),
                    "height": int(values["height"]),
                    "target_fps": int(values["target_fps"]),
                    "encoded_frames": int(values["frames"]),
                    "encoded_fps": float(values["fps"]),
                    "encoded_tx": values["tx"],
                }
            )
            sections += 1
            continue

        renderer = re.search(
            r"\bgl\s*:\s*egl=(?P<egl>\S+)\s+frames=(?P<frames>\d+)\s+"
            r"swapErr=(?P<errors>\d+)\s+err=(?P<error>.*)$",
            message,
        )
        if renderer:
            values = renderer.groupdict()
            metrics.update(
                {
                    "egl_state": values["egl"],
                    "rendered_frames": int(values["frames"]),
                    "renderer_swap_errors": int(values["errors"]),
                    "renderer_error": values["error"],
                }
            )
            continue

        transport = re.search(
            r"\btcp\s*:\s*:(?P<port>\d+)\s+listen=(?P<listen>yes|no)\s+"
            r"binds=(?P<binds>\d+)\s+conns=(?P<connections>\d+)\s+"
            r"client=(?P<client>yes|no)\s+up=(?P<client_up>\S+)\s+"
            r"tx=(?P<tx>\S+)\s+(?P<rate>[\d.]+)KB/s\s+"
            r"wrErr=(?P<write_errors>\d+)",
            message,
        )
        if transport:
            values = transport.groupdict()
            rate_kbps = float(values["rate"])
            metrics.update(
                {
                    "tcp_port": int(values["port"]),
                    "tcp_listening": values["listen"] == "yes",
                    "tcp_bind_attempts": int(values["binds"]),
                    "tcp_connections": int(values["connections"]),
                    "tcp_client_connected": values["client"] == "yes",
                    "tcp_client_uptime": values["client_up"],
                    "tcp_tx": values["tx"],
                    "tcp_rate_kbytes_s": rate_kbps,
                    "bitrate_mbps": round(rate_kbps * 8.0 / 1024.0, 3),
                    "tcp_write_errors": int(values["write_errors"]),
                }
            )
            sections += 1

    if "captured_frames" in metrics and "encoded_frames" in metrics:
        metrics["estimated_drops"] = max(
            0, int(metrics["captured_frames"]) - int(metrics["encoded_frames"])
        )
    return metrics, sections


def parse_capture_metrics(output: str) -> dict[str, Any]:
    """Parse the newest complete merged-app block, then fall back to legacy."""
    blocks: list[list[str]] = []
    current: list[str] = []
    saw_marker = False
    for line in output.splitlines():
        message = _vcam_log_message(line)
        if message == "telemetry":
            if current:
                blocks.append(current)
            current = []
            saw_marker = True
            continue
        if saw_marker or re.match(r"^(?:up=|cam\s*:|enc\s*:|gl\s*:|tcp\s*:)", message):
            current.append(message)
    if current:
        blocks.append(current)

    candidates = [_parse_merged_telemetry_block(block) for block in blocks]
    for metrics, sections in reversed(candidates):
        if sections >= 4:
            return metrics
    if candidates:
        metrics, sections = max(
            enumerate(candidates), key=lambda item: (item[1][1], item[0])
        )[1]
        if sections:
            return metrics
    return _parse_legacy_capture_metrics(output)


def collect_status(
    preferred_serial: str = "",
    expected_host: str = "",
    camera_id: str = DEFAULT_CAMERA_ID,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "management": "adb",
        "video_transport": "lan",
        "serial": preferred_serial or None,
        "model": None,
        "host": expected_host or None,
        "adb_state": "unavailable",
        "app_installed": False,
        "app_version": None,
        "bridge_running": False,
        "capture_metrics": {},
        "module_installed": False,
        "module_enabled": False,
        "module_version": None,
        "network_sender_running": False,
        "capture_fanout_running": False,
        "return_receiver_running": False,
        "output_selector_running": False,
        "provider_running": False,
        "camera_node_ready": False,
        "camera_id": str(camera_id),
        "camera_published": False,
        "front_redirect": {
            "package_installed": False,
            "active": None,
            "processed_camera_id": str(camera_id),
        },
        "output_control": {
            "supported": False,
            "enabled": True,
            "mirror": False,
            "rotation": 0,
            "revision": 0,
            "persisted": False,
            "effective_source": None,
            "effective_worker_alive": False,
            "effective_revision": None,
            "applied": False,
        },
        "error": None,
    }
    try:
        adb = _adb_path()
        selected, devices = _select_device(adb, preferred_serial, expected_host)
    except AndroidBridgeError as exc:
        status["error"] = str(exc)
        return status

    status["devices"] = devices
    if selected is None:
        if preferred_serial:
            status["error"] = f"configured Android device {preferred_serial} is not connected"
        elif len([item for item in devices if item.get("state") == "device"]) > 1:
            status["error"] = "multiple Android devices are connected; configure ANDROID_ADB_SERIAL"
        else:
            status["error"] = "no Android device is connected over ADB"
        return status

    serial = selected.get("serial", "")
    state = selected.get("state", "unknown")
    status.update(
        {
            "serial": serial,
            "model": selected.get("model"),
            "adb_state": state,
            "available": state == "device",
        }
    )
    if state != "device":
        status["error"] = f"Android device is {state}"
        return status

    # One remote shell minimizes UI polling overhead while keeping every
    # command and package name fixed (no user-provided shell fragments).
    shell_command = (
        "printf '__MODEL__\\n'; getprop ro.product.model; "
        "printf '__IP__\\n'; ip -brief address show wlan0 2>/dev/null; "
        f"printf '__APP__\\n'; pm path {PACKAGE} 2>/dev/null; "
        f"printf '__VERSION__\\n'; dumpsys package {PACKAGE} 2>/dev/null | "
        "sed -n 's/^[[:space:]]*versionName=//p'; "
        f"printf '__SERVICE__\\n'; dumpsys activity services {PACKAGE} 2>/dev/null; "
        "printf '__METRICS__\\n'; logcat -d -v brief -s VCamBridge:D '*:S' "
        "2>/dev/null | tail -n 80"
    )
    detail = _adb(adb, serial, "shell", shell_command, timeout=6.0)
    if detail.returncode == 0:
        model = _section(detail.stdout, "MODEL", "IP").splitlines()
        if model and model[0].strip():
            status["model"] = model[0].strip()
        ip_text = _section(detail.stdout, "IP", "APP")
        match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}(?=/\d+)", ip_text)
        if match:
            status["host"] = match.group(0)
        app_text = _section(detail.stdout, "APP", "VERSION")
        status["app_installed"] = "package:" in app_text
        versions = _section(detail.stdout, "VERSION", "SERVICE").splitlines()
        if versions:
            status["app_version"] = versions[0].strip() or None
        service_text = _section(detail.stdout, "SERVICE", "METRICS")
        status["bridge_running"] = SERVICE in service_text and "(nothing)" not in service_text
        status["capture_metrics"] = parse_capture_metrics(
            _section(detail.stdout, "METRICS")
        )
    else:
        status["error"] = detail.stderr.strip() or "could not inspect the Android bridge app"

    root_command = (
        "alive_file() { [ -s \"$1\" ] || return 1; "
        "p=$(cat \"$1\" 2>/dev/null); [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null; }; "
        "[ -d /data/adb/modules/android-vcam ] && echo module_installed=1 || echo module_installed=0; "
        "[ -d /data/adb/modules/android-vcam ] && "
        "[ ! -e /data/adb/modules/android-vcam/disable ] && "
        "echo module_enabled=1 || echo module_enabled=0; "
        "sed -n 's/^version=//p' /data/adb/modules/android-vcam/module.prop 2>/dev/null | "
        "sed 's/^/module_version=/'; "
        "alive_file /data/local/tmp/android-vcam-sender.pid && echo sender=1 || echo sender=0; "
        "alive_file /data/local/tmp/android-vcam-capture.pid && echo capture=1 || echo capture=0; "
        "alive_file /data/local/tmp/android-vcam-return.pid && echo return=1 || echo return=0; "
        "alive_file /data/local/tmp/android-vcam-output-selector.pid && echo output=1 || echo output=0; "
        "alive_file /data/local/tmp/android-vcam-provider.pid && echo provider=1 || echo provider=0; "
        "pm path dev.vcam.camlog >/dev/null 2>&1 && echo front_redirect_package=1 || echo front_redirect_package=0; "
        "[ -e /dev/video20 ] && echo camera_node=1 || echo camera_node=0; "
        f"output_control={shlex.quote(OUTPUT_CONTROL_FILE)}; "
        f"output_state={shlex.quote(OUTPUT_STATE_FILE)}; "
        "read_output_value() { sed -n \"s/^$2=//p\" \"$1\" 2>/dev/null | tail -n 1; }; "
        "[ -r \"$output_control\" ] && echo output_config_present=1 || echo output_config_present=0; "
        "value=$(read_output_value \"$output_control\" enabled); echo output_enabled=${value:-1}; "
        "value=$(read_output_value \"$output_control\" mirror); echo output_mirror=${value:-0}; "
        "value=$(read_output_value \"$output_control\" rotation); echo output_rotation=${value:-0}; "
        "value=$(read_output_value \"$output_control\" revision); echo output_revision=${value:-0}; "
        "value=$(read_output_value \"$output_state\" enabled); echo effective_enabled=${value:-}; "
        "value=$(read_output_value \"$output_state\" mirror); echo effective_mirror=${value:-}; "
        "value=$(read_output_value \"$output_state\" rotation); echo effective_rotation=${value:-}; "
        "value=$(read_output_value \"$output_state\" revision); echo effective_revision=${value:-}; "
        "value=$(read_output_value \"$output_state\" source); echo effective_source=${value:-}; "
        "value=$(read_output_value \"$output_state\" worker_alive); echo effective_worker_alive=${value:-0}"
    )
    root = _adb(
        adb,
        serial,
        "shell",
        f"su -c {shlex.quote(root_command)}",
        timeout=5.0,
    )
    if root.returncode == 0:
        values = _key_values(root.stdout)
        output_revision = _int_value(values, "output_revision", 0) or 0
        if output_revision < 0:
            output_revision = 0
        effective_revision = _int_value(values, "effective_revision")
        if effective_revision is not None and effective_revision < 0:
            effective_revision = None
        output_enabled = values.get("output_enabled", "1") == "1"
        output_mirror = values.get("output_mirror", "0") == "1"
        output_rotation = _int_value(values, "output_rotation", 0) or 0
        if output_rotation not in (0, 90, 180, 270):
            output_rotation = 0
        output_control_supported = _module_supports_output_control(
            values.get("module_version")
        )
        effective_matches = bool(
            output_control_supported
            and _bool_value(values, "output")
            and effective_revision is not None
            and effective_revision == output_revision
            and values.get("effective_enabled") == ("1" if output_enabled else "0")
            and values.get("effective_mirror") == ("1" if output_mirror else "0")
            and _int_value(values, "effective_rotation") == output_rotation
        )
        status.update(
            {
                "module_installed": _bool_value(values, "module_installed"),
                "module_enabled": _bool_value(values, "module_enabled"),
                "module_version": values.get("module_version"),
                "network_sender_running": _bool_value(values, "sender"),
                "capture_fanout_running": _bool_value(values, "capture"),
                "return_receiver_running": _bool_value(values, "return"),
                "output_selector_running": _bool_value(values, "output"),
                "provider_running": _bool_value(values, "provider"),
                "camera_node_ready": _bool_value(values, "camera_node"),
                "front_redirect": {
                    "package_installed": _bool_value(
                        values, "front_redirect_package"
                    ),
                    # Hooks live inside target application processes; inactive
                    # simply means no scoped app has an open front session.
                    "active": None,
                    "processed_camera_id": str(camera_id),
                },
                "output_control": {
                    "supported": output_control_supported,
                    "enabled": output_enabled,
                    "mirror": output_mirror,
                    "rotation": output_rotation,
                    "revision": output_revision,
                    "persisted": _bool_value(values, "output_config_present"),
                    "effective_source": values.get("effective_source") or None,
                    "effective_worker_alive": _bool_value(
                        values, "effective_worker_alive"
                    ),
                    "effective_revision": effective_revision,
                    "applied": effective_matches,
                },
            }
        )
    else:
        status["root_status_error"] = root.stderr.strip() or "Android module status unavailable"
    status["camera_published"] = bool(
        status["module_enabled"]
        and status["provider_running"]
        and status["camera_node_ready"]
    )
    return status


def _require_device(preferred_serial: str, expected_host: str = "") -> tuple[str, str]:
    adb = _adb_path()
    selected, _devices = _select_device(adb, preferred_serial, expected_host)
    if selected is None or selected.get("state") != "device":
        target = preferred_serial or "configured phone"
        raise AndroidBridgeError(f"{target} is not available over ADB")
    return adb, selected["serial"]


def _start_camera_bridge(adb: str, serial: str) -> None:
    """Start the merged app's non-exported foreground camera owner as root."""
    arguments = [
        "am",
        "start-foreground-service",
        "--user",
        "0",
        "-n",
        SERVICE,
        "-a",
        f"{PACKAGE}.START",
    ]
    root_command = " ".join(shlex.quote(argument) for argument in arguments)
    started = _adb(
        adb,
        serial,
        "shell",
        f"su -c {shlex.quote(root_command)}",
        timeout=10.0,
    )
    output = "\n".join((started.stdout, started.stderr)).strip()
    if started.returncode != 0 or "Error:" in output or "Exception" in output:
        raise AndroidBridgeError(output or "could not start Android camera bridge")


def control(action: str, preferred_serial: str = "", expected_host: str = "") -> None:
    if action not in ("start", "stop", "restart"):
        raise AndroidBridgeError(f"unsupported action: {action}")
    adb, serial = _require_device(preferred_serial, expected_host)
    if action in ("stop", "restart"):
        stopped = _adb(adb, serial, "shell", "am", "force-stop", PACKAGE, timeout=5.0)
        if stopped.returncode != 0:
            raise AndroidBridgeError(stopped.stderr.strip() or "could not stop Android bridge")
    if action == "restart":
        time.sleep(0.4)
    if action in ("start", "restart"):
        _start_camera_bridge(adb, serial)
        # Keep the historical convenience of opening the dashboard, but the
        # camera service above is authoritative. A locked display must not turn
        # a successful headless bridge start into a failed control operation.
        try:
            _adb(
                adb,
                serial,
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                ACTIVITY,
                timeout=10.0,
            )
        except AndroidBridgeError:
            pass


def configure_camera(
    values: dict[str, int | bool | str],
    preferred_serial: str = "",
    expected_host: str = "",
    *,
    persist: bool = False,
) -> None:
    """Hand capture settings to the already-owning Camera2 service over ADB."""
    allowed = {
        "lens_facing",
        "exposure_compensation",
        "ae_lock",
        "awb_lock",
        "stabilization",
        "rotation",
        "zoom_percent",
    }
    if not values or set(values) - allowed:
        raise AndroidBridgeError("invalid or empty Android camera configuration")
    arguments = [
        "am", "start-foreground-service", "--user", "0",
        "-n", SERVICE, "-a", f"{PACKAGE}.CONFIGURE",
        "--ez", "persist", "true" if persist else "false",
    ]
    if "lens_facing" in values:
        lens = str(values["lens_facing"])
        if lens not in ("front", "back"):
            raise AndroidBridgeError("lens_facing must be front or back")
        arguments.extend(("--es", "lens_facing", lens))
    if "exposure_compensation" in values:
        exposure = int(values["exposure_compensation"])
        if not -12 <= exposure <= 12:
            raise AndroidBridgeError("exposure_compensation must be between -12 and 12")
        arguments.extend(("--ei", "exposure_compensation", str(exposure)))
    if "ae_lock" in values:
        if not isinstance(values["ae_lock"], bool):
            raise AndroidBridgeError("ae_lock must be true or false")
        arguments.extend(("--ez", "ae_lock", "true" if values["ae_lock"] else "false"))
    if "awb_lock" in values:
        if not isinstance(values["awb_lock"], bool):
            raise AndroidBridgeError("awb_lock must be true or false")
        arguments.extend(
            ("--ez", "awb_lock", "true" if values["awb_lock"] else "false")
        )
    if "stabilization" in values:
        stabilization = str(values["stabilization"])
        if stabilization not in ("off", "video", "optical"):
            raise AndroidBridgeError(
                "stabilization must be off, video, or optical"
            )
        arguments.extend(("--es", "stabilization", stabilization))
    if "rotation" in values:
        rotation = str(values["rotation"])
        if rotation not in ("auto", "0", "90", "180", "270"):
            raise AndroidBridgeError(
                "rotation must be auto, 0, 90, 180, or 270"
            )
        arguments.extend(("--es", "rotation", rotation))
    if "zoom_percent" in values:
        zoom_percent = int(values["zoom_percent"])
        if not 100 <= zoom_percent <= 300:
            raise AndroidBridgeError("zoom_percent must be between 100 and 300")
        arguments.extend(("--ei", "zoom_percent", str(zoom_percent)))
    adb, serial = _require_device(preferred_serial, expected_host)
    running = _adb(
        adb,
        serial,
        "shell",
        "dumpsys",
        "activity",
        "services",
        PACKAGE,
        timeout=5.0,
    )
    if running.returncode != 0 or SERVICE not in running.stdout:
        raise AndroidBridgeError(
            "the Camera2 owner is not running; no camera was started or stopped"
        )
    # The service is not exported to ordinary apps. Root invokes this fixed,
    # allow-listed command; camera video still never traverses ADB.
    root_command = " ".join(arguments)
    configured = _adb(
        adb,
        serial,
        "shell",
        f"su -c {shlex.quote(root_command)}",
        timeout=10.0,
    )
    output = "\n".join((configured.stdout, configured.stderr)).strip()
    if configured.returncode != 0 or "Error:" in output or "Exception" in output:
        raise AndroidBridgeError(output or "could not configure Camera2 owner")


def configure_output(
    values: dict[str, bool | int],
    preferred_serial: str = "",
    expected_host: str = "",
) -> dict[str, bool | int]:
    """Persist a hot Camera2-120 output policy without touching camera owners.

    The module's long-lived selector observes this atomically replaced file.
    It may replace only its decoder/FIFO writer; the provider, producer,
    /dev/video20, Camera2 ID 120, and already-open app sessions remain alive.
    Repeating an already-effective request is idempotent and does not advance
    the revision or restart the decoder worker.
    """
    allowed = {"enabled", "mirror", "rotation"}
    if not values or set(values) - allowed:
        raise AndroidBridgeError("invalid or empty Android output configuration")
    for key in ("enabled", "mirror"):
        if key in values and not isinstance(values[key], bool):
            raise AndroidBridgeError(f"{key} must be true or false")
    if "rotation" in values:
        rotation = values["rotation"]
        if isinstance(rotation, bool) or rotation not in (0, 90, 180, 270):
            raise AndroidBridgeError("rotation must be 0, 90, 180, or 270")

    requested_enabled = (
        "1" if values.get("enabled") is True
        else "0" if values.get("enabled") is False
        else ""
    )
    requested_mirror = (
        "1" if values.get("mirror") is True
        else "0" if values.get("mirror") is False
        else ""
    )
    requested_rotation = str(values["rotation"]) if "rotation" in values else ""

    # Every interpolated value above has already been reduced to a fixed token.
    # The remote shell still validates both persisted and requested values,
    # merges omitted settings, and only advances its revision on a real change.
    root_script = f"""
control={shlex.quote(OUTPUT_CONTROL_FILE)}
read_value() {{ sed -n "s/^$1=//p" "$control" 2>/dev/null | tail -n 1; }}
present=0
[ -r "$control" ] && present=1
enabled=$(read_value enabled)
mirror=$(read_value mirror)
rotation=$(read_value rotation)
revision=$(read_value revision)
case "$enabled" in 0|1) ;; *) enabled=1 ;; esac
case "$mirror" in 0|1) ;; *) mirror=0 ;; esac
case "$rotation" in 0|90|180|270) ;; *) rotation=0 ;; esac
case "$revision" in ''|*[!0-9]*) revision=0 ;; esac
requested_enabled={requested_enabled!r}
requested_mirror={requested_mirror!r}
requested_rotation={requested_rotation!r}
[ -z "$requested_enabled" ] || enabled=$requested_enabled
[ -z "$requested_mirror" ] || mirror=$requested_mirror
[ -z "$requested_rotation" ] || rotation=$requested_rotation
case "$enabled" in 0|1) ;; *) exit 64 ;; esac
case "$mirror" in 0|1) ;; *) exit 64 ;; esac
case "$rotation" in 0|90|180|270) ;; *) exit 64 ;; esac
old_enabled=$(read_value enabled)
old_mirror=$(read_value mirror)
old_rotation=$(read_value rotation)
changed=0
if [ "$present" != 1 ] || [ "$old_enabled" != "$enabled" ] || [ "$old_mirror" != "$mirror" ] || [ "$old_rotation" != "$rotation" ]; then
    changed=1
    revision=$((revision + 1))
    tmp="${{control}}.tmp.$$"
    umask 077
    {{
        printf 'version=1\\n'
        printf 'enabled=%s\\n' "$enabled"
        printf 'mirror=%s\\n' "$mirror"
        printf 'rotation=%s\\n' "$rotation"
        printf 'revision=%s\\n' "$revision"
    }} >"$tmp" || exit 73
    chmod 0600 "$tmp" || exit 73
    mv -f "$tmp" "$control" || exit 73
fi
printf 'enabled=%s\\n' "$enabled"
printf 'mirror=%s\\n' "$mirror"
printf 'rotation=%s\\n' "$rotation"
printf 'revision=%s\\n' "$revision"
printf 'changed=%s\\n' "$changed"
printf 'persisted=1\\n'
""".strip()

    adb, serial = _require_device(preferred_serial, expected_host)
    configured = _adb(
        adb,
        serial,
        "shell",
        f"su -c {shlex.quote(root_script)}",
        timeout=10.0,
    )
    output = "\n".join((configured.stdout, configured.stderr)).strip()
    if configured.returncode != 0:
        raise AndroidBridgeError(output or "could not configure Android output")
    response = _key_values(configured.stdout)
    required = {"enabled", "mirror", "rotation", "revision", "changed", "persisted"}
    if not required.issubset(response):
        raise AndroidBridgeError("Android output control returned an incomplete response")
    try:
        rotation = int(response["rotation"])
        revision = int(response["revision"])
    except ValueError as exc:
        raise AndroidBridgeError("Android output control returned invalid values") from exc
    if (
        response["enabled"] not in ("0", "1")
        or response["mirror"] not in ("0", "1")
        or rotation not in (0, 90, 180, 270)
        or revision < 0
        or response["changed"] not in ("0", "1")
        or response["persisted"] != "1"
    ):
        raise AndroidBridgeError("Android output control returned invalid values")
    return {
        "enabled": response["enabled"] == "1",
        "mirror": response["mirror"] == "1",
        "rotation": rotation,
        "revision": revision,
        "changed": response["changed"] == "1",
        "persisted": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS, nargs="?", default="status")
    parser.add_argument("--serial", default="", help="preferred ADB serial")
    parser.add_argument("--host", default="", help="expected phone LAN address")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--lens-facing", choices=("front", "back"))
    parser.add_argument("--exposure-compensation", type=int)
    parser.add_argument("--ae-lock", choices=("true", "false"))
    parser.add_argument("--awb-lock", choices=("true", "false"))
    parser.add_argument("--stabilization", choices=("off", "video", "optical"))
    parser.add_argument("--rotation", choices=("auto", "0", "90", "180", "270"))
    parser.add_argument("--zoom-percent", type=int)
    parser.add_argument("--output-enabled", choices=("true", "false"))
    parser.add_argument("--output-mirror", choices=("true", "false"))
    parser.add_argument("--output-rotation", type=int, choices=(0, 90, 180, 270))
    args = parser.parse_args()
    try:
        if args.action == "configure":
            values: dict[str, int | bool | str] = {}
            if args.lens_facing is not None:
                values["lens_facing"] = args.lens_facing
            if args.exposure_compensation is not None:
                values["exposure_compensation"] = args.exposure_compensation
            if args.ae_lock is not None:
                values["ae_lock"] = args.ae_lock == "true"
            if args.awb_lock is not None:
                values["awb_lock"] = args.awb_lock == "true"
            if args.stabilization is not None:
                values["stabilization"] = args.stabilization
            if args.rotation is not None:
                values["rotation"] = args.rotation
            if args.zoom_percent is not None:
                values["zoom_percent"] = args.zoom_percent
            configure_camera(values, args.serial, args.host)
        elif args.action == "configure-output":
            output_values: dict[str, bool | int] = {}
            if args.output_enabled is not None:
                output_values["enabled"] = args.output_enabled == "true"
            if args.output_mirror is not None:
                output_values["mirror"] = args.output_mirror == "true"
            if args.output_rotation is not None:
                output_values["rotation"] = args.output_rotation
            configure_output(output_values, args.serial, args.host)
        elif args.action != "status":
            control(args.action, args.serial, args.host)
        status = collect_status(args.serial, args.host, args.camera_id)
    except AndroidBridgeError as exc:
        print(f"deep-live-cam Android bridge: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(status, sort_keys=True))
    if args.action != "status" and not status.get("available"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
