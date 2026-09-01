"""Native-client JSON control API for the Windows processing service.

This module intentionally serves no browser camera UI.  The Arch desktop app
is the management surface and consumes these LAN-only JSON endpoints without
opening any physical or virtual camera device.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np

import modules.globals


MAX_UPLOAD = 20 * 1024 * 1024
MAX_JSON = 256 * 1024
ENHANCERS = {"none": None, "gfpgan": "face_enhancer"}
PROCESSING_MODES = {"face_swap", "passthrough"}
SWAPPER_MODELS = {
    "auto",
    "inswapper-128",
    "instyle-256",
    "simswap-512",
    "native-256",
}
SOURCE_IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _enhancer_available(name: str) -> bool:
    if name == "none":
        return True
    if name == "gfpgan":
        return (
            Path(__file__).resolve().parent.parent / "models" / "gfpgan-1024.onnx"
        ).exists()
    return False


class ControlState:
    def __init__(
        self,
        directory: str,
        health: Callable[[], dict[str, Any]],
        devices: Callable[[], dict[str, Any]],
        select_device: Callable[[str], dict[str, Any]],
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.directory / "settings.json"
        self.source_path = self.directory / "source.jpg"
        self.source_metadata_path = self.directory / "source.metadata.json"
        self.health = health
        self.devices = devices
        self.select_device = select_device
        self.lock = threading.Lock()
        self.source_identifier: str | None = None
        self.load()

    def config(self) -> dict[str, Any]:
        self.source_identifier = self._verified_source_identifier()
        enhancer = "none"
        for name, key in ENHANCERS.items():
            if key and modules.globals.fp_ui.get(key, False):
                enhancer = name
        processing_mode = (
            "face_swap" if modules.globals.processing_enabled else "passthrough"
        )
        return {
            "processing_mode": processing_mode,
            "swapper_model": modules.globals.swapper_model,
            "swapper_backend": modules.globals.swapper_backend,
            "active_swapper_model": modules.globals.active_swapper_model,
            "active_swapper_backend": modules.globals.active_swapper_backend,
            "active_swapper_resolution": modules.globals.active_swapper_resolution,
            "opacity": modules.globals.opacity,
            "sharpness": modules.globals.sharpness,
            "mouth_mask_size": modules.globals.mouth_mask_size,
            "interpolation_weight": modules.globals.interpolation_weight,
            "many_faces": modules.globals.many_faces,
            "live_mirror": modules.globals.live_mirror,
            "show_fps": modules.globals.show_fps,
            "enable_interpolation": modules.globals.enable_interpolation,
            # Compatibility fields for older native clients.
            "processing_enabled": modules.globals.processing_enabled,
            "processing_off_output": modules.globals.processing_off_output,
            "quality_mode": modules.globals.quality_mode,
            "quality_auto_correct": modules.globals.quality_auto_correct,
            "tracking_enabled": modules.globals.tracking_enabled,
            "detection_interval": modules.globals.detection_interval,
            "tracking_smoothing": modules.globals.tracking_smoothing,
            "tracking_grace_frames": modules.globals.tracking_grace_frames,
            "minimum_detection_score": modules.globals.minimum_detection_score,
            "minimum_face_size": modules.globals.minimum_face_size,
            "color_match_strength": modules.globals.color_match_strength,
            "repair_hf_strength": modules.globals.repair_hf_strength,
            "repair_checkerboard": modules.globals.repair_checkerboard,
            "repair_wavelet": modules.globals.repair_wavelet,
            "repair_boundary_mask": modules.globals.repair_boundary_mask,
            "repair_boundary_strength": modules.globals.repair_boundary_strength,
            # Kept only so an older client can read/write its old state file;
            # the runtime deliberately performs no target-texture transfer.
            "repair_skin_texture": modules.globals.repair_skin_texture,
            "repair_camera_detail": modules.globals.repair_camera_detail,
            "enhancer": enhancer,
            "source_configured": bool(modules.globals.source_path),
            # None means that the source predates the identity contract, its
            # metadata is incomplete, or source.jpg has changed on disk.
            "source_identifier": self.source_identifier,
        }

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _verified_source_identifier(self) -> str | None:
        """Return an upload ID only while metadata still matches source.jpg."""
        if not self.source_path.is_file() or not self.source_metadata_path.is_file():
            return None
        try:
            metadata = json.loads(self.source_metadata_path.read_text("utf-8"))
            if not isinstance(metadata, dict):
                return None
            identifier = metadata.get("source_identifier")
            stored_hash = metadata.get("stored_jpeg_sha256")
            if not isinstance(
                identifier, str
            ) or not SOURCE_IDENTIFIER_PATTERN.fullmatch(identifier):
                return None
            if not isinstance(
                stored_hash, str
            ) or not SOURCE_IDENTIFIER_PATTERN.fullmatch(stored_hash):
                return None
            actual_hash = self._sha256(self.source_path.read_bytes())
            if not hmac.compare_digest(stored_hash, actual_hash):
                return None
            return identifier
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def apply(self, data: dict[str, Any], persist: bool = True) -> None:
        if not isinstance(data, dict):
            raise ValueError("settings must be a JSON object")
        limits = {
            "opacity": (0.0, 1.0),
            "sharpness": (0.0, 5.0),
            "mouth_mask_size": (0.0, 100.0),
            "interpolation_weight": (0.0, 1.0),
            "tracking_smoothing": (0.0, 0.95),
            "minimum_detection_score": (0.1, 0.95),
            "color_match_strength": (0.0, 1.0),
            "repair_hf_strength": (0.0, 0.5),
            "repair_checkerboard": (0.0, 1.0),
            "repair_wavelet": (0.0, 1.0),
            "repair_boundary_strength": (0.0, 1.0),
            "repair_skin_texture": (0.0, 0.3),
            "repair_camera_detail": (0.0, 4.0),
        }
        integer_limits = {
            "detection_interval": (1, 5),
            "tracking_grace_frames": (0, 15),
            "minimum_face_size": (32, 512),
        }
        with self.lock:
            requested_mode = data.get("processing_mode")
            if requested_mode is not None:
                requested_mode = str(requested_mode)
                if requested_mode not in PROCESSING_MODES:
                    raise ValueError(
                        "processing_mode must be face_swap or passthrough"
                    )
                modules.globals.processing_enabled = requested_mode == "face_swap"
                modules.globals.processing_off_output = "passthrough"
            if "swapper_model" in data:
                requested_model = str(data["swapper_model"])
                if requested_model not in SWAPPER_MODELS:
                    raise ValueError(
                        "swapper_model must be auto, inswapper-128, instyle-256, "
                        "simswap-512, or native-256"
                    )
                modules.globals.swapper_model = requested_model
            for key, (low, high) in limits.items():
                if key in data:
                    setattr(
                        modules.globals,
                        key,
                        max(low, min(high, float(data[key]))),
                    )
            for key, (low, high) in integer_limits.items():
                if key in data:
                    setattr(
                        modules.globals,
                        key,
                        max(low, min(high, int(data[key]))),
                    )
            for key in (
                "processing_enabled",
                "many_faces",
                "live_mirror",
                "show_fps",
                "enable_interpolation",
                "quality_auto_correct",
                "tracking_enabled",
                "repair_boundary_mask",
            ):
                if key in data and requested_mode is None:
                    setattr(modules.globals, key, bool(data[key]))
                elif key != "processing_enabled" and key in data:
                    setattr(modules.globals, key, bool(data[key]))
            if data.get("quality_mode") in ("monitor", "balanced", "strict"):
                modules.globals.quality_mode = data["quality_mode"]
            if (
                requested_mode is None
                and data.get("processing_off_output") in ("passthrough", "black")
            ):
                modules.globals.processing_off_output = data[
                    "processing_off_output"
                ]
            modules.globals.mouth_mask = modules.globals.mouth_mask_size > 0
            if "enhancer" in data:
                requested = str(data["enhancer"]).lower()
                if requested not in ENHANCERS:
                    raise ValueError(f"unknown enhancer: {requested}")
                selected = (
                    ENHANCERS.get(requested)
                    if _enhancer_available(requested)
                    else None
                )
                for key in ENHANCERS.values():
                    if key:
                        modules.globals.fp_ui[key] = key == selected
            if persist:
                temporary = self.settings_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(self.config(), indent=2), encoding="utf-8"
                )
                os.replace(temporary, self.settings_path)

    def load(self) -> None:
        if self.settings_path.exists():
            try:
                self.apply(
                    json.loads(self.settings_path.read_text("utf-8")), False
                )
            except Exception as exc:
                print(f"[control] ignoring invalid settings: {exc}", flush=True)
        if self.source_path.exists():
            modules.globals.source_path = str(self.source_path)
        # Legacy files and interrupted two-file replacements deliberately load
        # as unverified.  config() repeats this check to catch later tampering.
        self.source_identifier = self._verified_source_identifier()

    def upload(self, body: bytes, source_identifier: str) -> str:
        if not body or len(body) > MAX_UPLOAD:
            raise ValueError("image must be between 1 byte and 20 MB")
        if not isinstance(
            source_identifier, str
        ) or not SOURCE_IDENTIFIER_PATTERN.fullmatch(source_identifier):
            raise ValueError("X-Source-Identifier must be a lowercase SHA-256 digest")
        actual_identifier = self._sha256(body)
        if not hmac.compare_digest(source_identifier, actual_identifier):
            raise ValueError("X-Source-Identifier does not match the uploaded bytes")
        image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 64:
            raise ValueError("not a valid image, or image is too small")
        with self.lock:
            temporary = self.source_path.with_suffix(".tmp.jpg")
            if not cv2.imwrite(
                str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise ValueError("could not store image")
            stored_jpeg_sha256 = self._sha256(temporary.read_bytes())
            metadata_temporary = self.source_metadata_path.with_suffix(".tmp.json")
            metadata_temporary.write_text(
                json.dumps(
                    {
                        "source_identifier": source_identifier,
                        "stored_jpeg_sha256": stored_jpeg_sha256,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            # Replacing the image first is deliberate: a crash between
            # replacements leaves old metadata paired with a new JPEG, which
            # validation treats as unverified instead of advertising a stale ID.
            os.replace(temporary, self.source_path)
            os.replace(metadata_temporary, self.source_metadata_path)
            modules.globals.source_path = str(self.source_path)
            self.source_identifier = source_identifier
        return source_identifier


def make_handler(state: ControlState):
    class Handler(BaseHTTPRequestHandler):
        def _local_client(self) -> bool:
            try:
                address = ipaddress.ip_address(self.client_address[0])
                return address.is_private or address.is_loopback
            except ValueError:
                return False

        def _json(self, status: int, value: Any) -> None:
            body = json.dumps(value).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # Native clients may cancel an obsolete poll while an
                # inference backend is warming up. This is not a service
                # failure and should not flood the diagnostic log.
                return

        def _read(self, maximum: int) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > maximum:
                raise ValueError("request too large")
            return self.rfile.read(length)

        def do_GET(self) -> None:
            if not self._local_client():
                self._json(403, {"error": "LAN access only"})
                return
            path = urlparse(self.path).path
            if path == "/":
                self._json(
                    200,
                    {
                        "service": "deep-live-cam-network",
                        "management": "native-client",
                        "health": "/healthz",
                    },
                )
            elif path == "/api/config":
                self._json(200, state.config())
            elif path == "/api/devices":
                self._json(200, state.devices())
            elif path == "/api/source" and state.source_path.exists():
                body = state.source_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/healthz":
                self._json(200, state.health())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local_client():
                self._json(403, {"error": "LAN access only"})
                return
            try:
                path = urlparse(self.path).path
                if path == "/api/config":
                    state.apply(json.loads(self._read(MAX_JSON)))
                    self._json(200, state.config())
                elif path == "/api/devices/select":
                    request = json.loads(self._read(MAX_JSON))
                    if not isinstance(request, dict):
                        raise ValueError("selection must be a JSON object")
                    device_id = request.get("device_id")
                    if not isinstance(device_id, str):
                        raise ValueError("device_id is required")
                    self._json(200, state.select_device(device_id))
                elif path == "/api/source":
                    identifier = state.upload(
                        self._read(MAX_UPLOAD),
                        self.headers.get("X-Source-Identifier", ""),
                    )
                    self._json(
                        200, {"ok": True, "source_identifier": identifier}
                    )
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(
                f"[http] {self.client_address[0]} {fmt % args}", flush=True
            )

    return Handler


def serve(
    state: ControlState, host: str, port: int
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    threading.Thread(
        target=server.serve_forever, name="control-http", daemon=True
    ).start()
    return server
