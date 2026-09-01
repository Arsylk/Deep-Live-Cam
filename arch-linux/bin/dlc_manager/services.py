#!/usr/bin/env python3
"""Adapters for everything outside this process.

All of it is asynchronous from the UI thread and bounded by a timeout: the LAN
JSON API on Windows, the narrow local helper scripts, and systemd state.  A
disappearing endpoint degrades the interface; it never blocks or crashes it.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from PySide6.QtCore import QByteArray, QObject, QProcess, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


INSTALLED_HELPER_DIRECTORY = Path("/usr/local/lib/deep-live-cam-arch")

SENDER_UNIT = "deep-live-cam-sender.service"
RECEIVER_UNIT = "deep-live-cam-receiver.service"


def local_helper(name: str) -> str:
    """Resolve a helper next to this package, then in the installed layout."""
    adjacent = Path(__file__).resolve().parents[1] / name
    if adjacent.exists():
        return str(adjacent)
    return str(INSTALLED_HELPER_DIRECTORY / name)


def unit_state(unit: str) -> dict[str, str]:
    """Read one unit's state without ever changing it."""
    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                unit,
                "--property=ActiveState",
                "--property=SubState",
                "--property=UnitFileState",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "ActiveState": "unknown",
            "SubState": "unknown",
            "UnitFileState": "unknown",
        }
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return {
        "ActiveState": values.get("ActiveState", "unknown"),
        "SubState": values.get("SubState", "unknown"),
        "UnitFileState": values.get("UnitFileState", "unknown"),
    }


class SystemdProbe:
    """Cache unit states briefly so a 1 Hz refresh is not 1 Hz of subprocesses."""

    def __init__(self, ttl: float = 0.9) -> None:
        self.ttl = float(ttl)
        self._cache: dict[str, tuple[float, dict[str, str]]] = {}

    def state(self, unit: str) -> dict[str, str]:
        now = time.monotonic()
        cached = self._cache.get(unit)
        if cached is not None and now - cached[0] < self.ttl:
            return cached[1]
        values = unit_state(unit)
        self._cache[unit] = (now, values)
        return values

    def states(self, units: Mapping[str, str]) -> dict[str, dict[str, str]]:
        return {key: self.state(unit) for key, unit in units.items()}


class HelperProcess(QObject):
    """One single-flight child process with merged output.

    A second request while one is running is refused rather than queued
    silently, so a slider drag cannot spawn a pile of privileged helpers.
    """

    finished = Signal(bool, str)
    failedToStart = Signal(str)

    def __init__(self, name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self._process: QProcess | None = None

    def busy(self) -> bool:
        return (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        )

    def run(self, program: str, arguments: Sequence[str]) -> bool:
        if self._process is not None:
            return False
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self._process = process
        process.start(program, list(arguments))
        return True

    def run_python(self, script: str, arguments: Sequence[str]) -> bool:
        return self.run(sys.executable, [script, *arguments])

    def terminate(self) -> None:
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(1000)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process = self._process
        output = ""
        if process is not None:
            output = bytes(process.readAll()).decode("utf-8", "replace").strip()
            process.deleteLater()
        self._process = None
        self.finished.emit(exit_code == 0, output)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        process = self._process
        message = process.errorString() if process is not None else "could not start"
        if process is not None:
            process.deleteLater()
        self._process = None
        self.failedToStart.emit(message)


class JsonProcess(QObject):
    """A helper whose stdout is a JSON document (the Android status bridge)."""

    parsed = Signal(object, str)

    def __init__(self, name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self._process: QProcess | None = None

    def busy(self) -> bool:
        return self._process is not None

    def run(self, script: str, arguments: Sequence[str]) -> bool:
        if self._process is not None:
            return False
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self._process = process
        process.start(sys.executable, [script, *arguments])
        return True

    def terminate(self) -> None:
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(1000)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process = self._process
        out = ""
        err = ""
        if process is not None:
            out = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
            err = bytes(process.readAllStandardError()).decode("utf-8", "replace").strip()
            process.deleteLater()
        self._process = None
        try:
            value = json.loads(out)
            if not isinstance(value, dict):
                raise ValueError("status was not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.parsed.emit(None, err or f"invalid {self.name} status: {exc}")
            return
        error = str(value.get("error")) if value.get("error") else None
        if exit_code != 0 and not error:
            error = err or f"{self.name} helper exited {exit_code}"
        self.parsed.emit(value, error or "")

    def _error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        process = self._process
        message = (
            process.errorString()
            if process is not None
            else f"could not start the {self.name} helper"
        )
        if process is not None:
            process.deleteLater()
        self._process = None
        self.parsed.emit(None, message)


class WindowsControlClient(QObject):
    """Asynchronous client for the private LAN JSON API on Windows TCP 8090.

    This is a LAN control endpoint, not an internet service: when it is
    unreachable the manager keeps working against local state.
    """

    healthReceived = Signal(object)
    healthFailed = Signal(str)
    configReceived = Signal(dict)
    configFailed = Signal(str)
    configApplied = Signal(dict)
    configRejected = Signal(dict, str)
    devicesReceived = Signal(dict)
    devicesFailed = Signal(str)
    selectionSucceeded = Signal(dict)
    selectionFailed = Signal(str)
    sourceUploaded = Signal(str)
    sourceUploadFailed = Signal(str)

    MAXIMUM_SOURCE_BYTES = 20 * 1024 * 1024

    def __init__(self, host: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.host = host
        self.network = QNetworkAccessManager(self)
        self._health: QNetworkReply | None = None
        self._config: QNetworkReply | None = None
        self._post: QNetworkReply | None = None
        self._devices: QNetworkReply | None = None
        self._select: QNetworkReply | None = None
        self._upload: QNetworkReply | None = None
        self._pending_payload: dict[str, Any] = {}
        self._pending_source: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.host) and self.host != "CHANGE_ME"

    def selection_in_flight(self) -> bool:
        return self._select is not None

    def upload_in_flight(self) -> bool:
        return self._upload is not None

    def _url(self, path: str) -> QUrl:
        return QUrl(f"http://{self.host}:8090{path}")

    def _request(self, path: str, timeout_ms: int) -> QNetworkRequest:
        request = QNetworkRequest(self._url(path))
        request.setTransferTimeout(timeout_ms)
        return request

    def request_health(self) -> None:
        if self._health is not None or not self.configured:
            return
        self._health = self.network.get(self._request("/healthz", 1800))
        self._health.finished.connect(self._health_finished)

    def _health_finished(self) -> None:
        reply, self._health = self._health, None
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                value = json.loads(bytes(reply.readAll()).decode("utf-8"))
                self.healthReceived.emit(value if isinstance(value, dict) else None)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.healthFailed.emit(f"invalid health JSON: {exc}")
        else:
            self.healthFailed.emit(reply.errorString())
        reply.deleteLater()

    def request_config(self) -> None:
        if self._config is not None or not self.configured:
            return
        self._config = self.network.get(self._request("/api/config", 2500))
        self._config.finished.connect(self._config_finished)

    def _config_finished(self) -> None:
        reply, self._config = self._config, None
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                value = json.loads(bytes(reply.readAll()).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("response is not an object")
                self.configReceived.emit(value)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.configFailed.emit(f"could not read Windows settings: {exc}")
        else:
            self.configFailed.emit(reply.errorString())
        reply.deleteLater()

    def apply_config(self, payload: Mapping[str, Any]) -> bool:
        if self._post is not None or not payload or not self.configured:
            return False
        self._pending_payload = dict(payload)
        request = self._request("/api/config", 3500)
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )
        body = QByteArray(
            json.dumps(self._pending_payload, separators=(",", ":")).encode("utf-8")
        )
        self._post = self.network.post(request, body)
        self._post.finished.connect(self._post_finished)
        return True

    def _post_finished(self) -> None:
        reply, self._post = self._post, None
        payload, self._pending_payload = self._pending_payload, {}
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            # The endpoint clamps/normalizes values. Its response is the
            # effective state; falling back to the request keeps compatibility
            # with the oldest service build that returned an empty body.
            effective = dict(payload)
            body = bytes(reply.readAll())
            if body:
                try:
                    value = json.loads(body.decode("utf-8"))
                    if isinstance(value, dict):
                        effective = value
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self.configRejected.emit(
                        payload, "Windows returned invalid settings JSON"
                    )
                    reply.deleteLater()
                    return
            self.configApplied.emit(effective)
        else:
            self.configRejected.emit(payload, reply.errorString())
        reply.deleteLater()

    def request_devices(self) -> None:
        if self._devices is not None or not self.configured:
            return
        self._devices = self.network.get(self._request("/api/devices", 2000))
        self._devices.finished.connect(self._devices_finished)

    def _devices_finished(self) -> None:
        reply, self._devices = self._devices, None
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                value = json.loads(bytes(reply.readAll()))
                if not isinstance(value, dict) or not isinstance(
                    value.get("slots"), list
                ):
                    raise ValueError("slot registry was not an object")
                self.devicesReceived.emit(value)
            except (json.JSONDecodeError, ValueError) as exc:
                self.devicesFailed.emit(f"invalid Windows slot registry: {exc}")
        else:
            self.devicesFailed.emit(reply.errorString())
        reply.deleteLater()

    def select_device(self, device_id: str) -> bool:
        if self._select is not None or not self.configured or not device_id:
            return False
        request = self._request("/api/devices/select", 8000)
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )
        self._select = self.network.post(
            request, QByteArray(json.dumps({"device_id": device_id}).encode("utf-8"))
        )
        self._select.finished.connect(self._select_finished)
        return True

    def _select_finished(self) -> None:
        reply, self._select = self._select, None
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                value = json.loads(bytes(reply.readAll()))
                if not isinstance(value, dict):
                    raise ValueError("selection response was not an object")
                self.selectionSucceeded.emit(value)
            except (json.JSONDecodeError, ValueError) as exc:
                self.selectionFailed.emit(f"invalid selection response: {exc}")
        else:
            self.selectionFailed.emit(reply.errorString())
        reply.deleteLater()

    def upload_source(self, data: bytes, filename: str) -> str:
        """Start a source-picture upload; returns "" or a refusal reason."""
        if self._upload is not None:
            return "a source picture change is already running"
        if not self.configured:
            return "the Windows processor is not configured"
        if not data or len(data) > self.MAXIMUM_SOURCE_BYTES:
            return "the picture must be between 1 byte and 20 MB"
        safe_name = (
            Path(filename).name.strip().replace("\r", " ").replace("\n", " ")
            or "source-picture"
        )[:240]
        mime_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        request = self._source_upload_request(data, safe_name, mime_type)
        self._pending_source = safe_name
        self._upload = self.network.post(request, QByteArray(data))
        self._upload.finished.connect(self._upload_finished)
        return ""

    def _source_upload_request(
        self, data: bytes, safe_name: str, mime_type: str
    ) -> QNetworkRequest:
        """Build the authenticated-by-content source request."""
        request = self._request("/api/source", 15000)
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, mime_type)
        request.setRawHeader(b"X-Filename", safe_name.encode("utf-8", "replace"))
        # This identifies the exact bytes sent over the wire, not the file name
        # and not the JPEG Windows creates after decoding the upload.  Windows
        # validates it before accepting the source and persists the relationship
        # to its re-encoded JPEG separately.
        request.setRawHeader(
            b"X-Source-Identifier", hashlib.sha256(data).hexdigest().encode("ascii")
        )
        return request

    def _upload_finished(self) -> None:
        reply, self._upload = self._upload, None
        name, self._pending_source = self._pending_source, ""
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            self.sourceUploaded.emit(name)
        else:
            detail = bytes(reply.readAll()).decode("utf-8", "replace").strip()
            self.sourceUploadFailed.emit(detail or reply.errorString())
        reply.deleteLater()

    def abort_all(self) -> None:
        for reply in (
            self._health,
            self._config,
            self._post,
            self._devices,
            self._select,
            self._upload,
        ):
            if reply is not None:
                reply.abort()


def offline_device_registry(
    *,
    selected_device_id: str,
    android_host: str,
    arch_host: str,
    android_enabled: bool,
) -> dict[str, Any]:
    """The five-slot contract to show while Windows is unreachable.

    Client-owned camera controls stay usable offline, so the registry shape is
    reproduced locally instead of leaving the page blank.
    """
    return {
        "selected_device_id": selected_device_id,
        "offline": True,
        "slots": [
            {
                "slot": 0,
                "device_id": "android-phone",
                "label": "Android phone",
                "stack": "android-camera2",
                "configured": True,
                "enabled": bool(android_enabled),
                "return_host": android_host,
                "input_port": 10_000,
                "return_port": 10_001,
            },
            {
                "slot": 1,
                "device_id": "arch-webcam",
                "label": "Arch USB webcam",
                "stack": "arch-v4l2",
                "configured": True,
                "enabled": True,
                "return_host": arch_host,
                "input_port": 10_002,
                "return_port": 10_003,
            },
            *[
                {
                    "slot": slot,
                    "device_id": None,
                    "label": "Unassigned",
                    "stack": "generic-srt",
                    "configured": False,
                    "enabled": False,
                    "input_port": 10_000 + slot * 2,
                    "return_port": 10_001 + slot * 2,
                }
                for slot in range(2, 5)
            ],
        ],
    }


def arch_persist_arguments(values: Mapping[str, Any]) -> list[str]:
    """Build ``configure_camera.py`` arguments for a persistent Arch save.

    An unmodified named profile is saved as that profile so the stored
    configuration keeps its identity; an edited profile degrades to the
    explicit value list.
    """
    from camera_profiles import ARCH_CAMERA_PROFILES, profile_live_values

    arguments = [local_helper("configure_camera.py")]
    selected = str(values.get("profile") or "")
    if selected in ARCH_CAMERA_PROFILES:
        expected = profile_live_values(selected)
        if all(values.get(key) == value for key, value in expected.items()):
            arguments.extend(("--profile", selected))
            return arguments
    for key, argument in (
        ("capture_size", "capture-size"),
        ("brightness", "brightness"),
        ("contrast", "contrast"),
        ("saturation", "saturation"),
        ("hue", "hue"),
        ("gamma", "gamma"),
        ("gain", "gain"),
        ("sharpness", "sharpness"),
        ("backlight_compensation", "backlight"),
        ("power_line_frequency", "power-line"),
        ("exposure_time_absolute", "exposure"),
        ("white_balance_temperature", "white-balance"),
    ):
        if key in values:
            arguments.extend((f"--{argument}", str(values[key])))
    for key, argument in (
        ("auto_exposure", "auto-exposure"),
        ("exposure_dynamic_framerate", "exposure-dynamic-framerate"),
        ("auto_white_balance", "auto-white-balance"),
    ):
        if key in values:
            arguments.extend((f"--{argument}", "1" if values[key] else "0"))
    return arguments


def adapter_arguments(
    stack: str,
    values: Mapping[str, Any],
    *,
    serial: str = "",
    host: str = "",
    persist: bool = False,
) -> list[str]:
    """Build ``camera_adapters.py`` arguments for a capability-driven stack."""
    arguments = [
        "--stack",
        stack,
        "--controls-json",
        json.dumps(dict(values), separators=(",", ":"), sort_keys=True),
    ]
    if stack == "android-camera2":
        if serial:
            arguments.extend(("--serial", serial))
        if host:
            arguments.extend(("--host", host))
        if persist:
            arguments.append("--persist")
    return arguments
