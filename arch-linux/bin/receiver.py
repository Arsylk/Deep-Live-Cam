#!/usr/bin/env python3
"""Receive processed SRT video and continuously feed a V4L2 loopback camera."""

from __future__ import annotations

import json
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common import (
    env_float,
    env_int,
    install_signal_handlers,
    resolve_virtual_devices,
    sd_notify,
    srt_query,
    state_dir,
    stop_process,
    write_state,
)


SOURCE_PRIORITIES = {
    "auto": (
        "local_processed",
        "local_prerecorded",
        "processed_return",
        "selected_stream",
        "local_raw",
    ),
    "local": ("local_processed", "local_prerecorded", "local_raw"),
    "windows": ("processed_return", "selected_stream", "local_raw"),
    "raw": ("local_raw",),
    "prerecorded": ("local_prerecorded",),
}

OUTPUT_ROTATIONS = (0, 90, 180, 270)
DEFAULT_SOURCE_STATE_FILE = Path(
    "/var/lib/deep-live-cam/receiver-source.json"
)


def _environment_source_mode() -> str:
    selected = os.environ.get("RECEIVER_SOURCE", "auto").strip().lower()
    if selected not in SOURCE_PRIORITIES:
        raise ValueError(
            "RECEIVER_SOURCE must be auto, local, windows, or raw"
        )
    return selected


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    """Replace one service-owned intent file without a partial-write window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    payload = json.dumps(dict(document), sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _environment_boolean(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _validate_output_transform(mirror: object, rotation: object) -> tuple[bool, int]:
    if not isinstance(mirror, bool):
        raise ValueError("mirror must be true or false")
    if not isinstance(rotation, int) or isinstance(rotation, bool):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if rotation not in OUTPUT_ROTATIONS:
        raise ValueError("rotation must be 0, 90, 180, or 270")
    return mirror, rotation


def _unpack_yuyv422(frame: bytes, width: int, height: int) -> tuple[np.ndarray, ...]:
    packed = np.frombuffer(frame, dtype=np.uint8).reshape(height, width // 2, 4)
    y = np.empty((height, width), dtype=np.uint8)
    y[:, 0::2] = packed[:, :, 0]
    y[:, 1::2] = packed[:, :, 2]
    u = np.repeat(packed[:, :, 1], 2, axis=1)
    v = np.repeat(packed[:, :, 3], 2, axis=1)
    return y, u, v


def _pack_yuyv422(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> bytes:
    height, width = y.shape
    packed = np.empty((height, width // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = y[:, 0::2]
    packed[:, :, 2] = y[:, 1::2]
    # A 90-degree rotation turns vertically-subsampled source relationships
    # into horizontal ones. Re-subsample chroma for valid packed YUYV422.
    packed[:, :, 1] = (
        (
            u[:, 0::2].astype(np.uint16)
            + u[:, 1::2].astype(np.uint16)
            + 1
        )
        // 2
    ).astype(np.uint8)
    packed[:, :, 3] = (
        (
            v[:, 0::2].astype(np.uint16)
            + v[:, 1::2].astype(np.uint16)
            + 1
        )
        // 2
    ).astype(np.uint8)
    return packed.tobytes()


def _cover_plane(
    plane: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Nearest-neighbour aspect fill with a centred crop.

    Output cameras keep one negotiated geometry for their entire lifetime. A
    quarter-turn changes the frame aspect ratio, so preserving all pixels
    would necessarily introduce letterboxing. The output contract instead
    matches a camera app's ``centerCrop`` behaviour: scale until both output
    dimensions are covered and discard the equal overflow on opposite edges.
    This guarantees that an orientation change can never create embedded
    black bars or a thumbnail-sized picture.

    The output-to-source map is calculated directly rather than allocating a
    potentially very large intermediate (a 16:9 quarter-turn cover is over
    three times taller than its destination).
    """
    source_height, source_width = plane.shape
    if (source_width, source_height) == (output_width, output_height):
        return plane

    scale = max(output_width / source_width, output_height / source_height)
    visible_width = output_width / scale
    visible_height = output_height / scale
    source_left = (source_width - visible_width) / 2.0
    source_top = (source_height - visible_height) / 2.0
    x_indices = np.minimum(
        (source_left + (np.arange(output_width) + 0.5) / scale).astype(int),
        source_width - 1,
    )
    y_indices = np.minimum(
        (source_top + (np.arange(output_height) + 0.5) / scale).astype(int),
        source_height - 1,
    )
    return plane[np.ix_(y_indices, x_indices)]


def transform_yuyv422(
    frame: bytes,
    width: int,
    height: int,
    *,
    mirror: bool = False,
    rotation: int = 0,
) -> bytes:
    """Apply a hot output transform while retaining fixed packed-YUYV geometry.

    Rotation is clockwise. Mirroring is horizontal in the final output
    coordinate system. A quarter turn uses an aspect-preserving centre crop
    back to the original dimensions. It therefore fills the output without
    embedded bars while consumers never observe a format renegotiation.
    """
    mirror, rotation = _validate_output_transform(mirror, rotation)
    if width <= 0 or height <= 0 or width % 2:
        raise ValueError("YUYV422 output width must be a positive even number")
    expected = width * height * 2
    if len(frame) != expected:
        raise ValueError(f"expected {expected} YUYV422 bytes, got {len(frame)}")
    if rotation == 0 and not mirror:
        return frame

    # Preserve exact source chroma for the common non-quarter-turn cases.
    packed = np.frombuffer(frame, dtype=np.uint8).reshape(height, width // 2, 4)
    if rotation == 0 and mirror:
        return packed[:, ::-1, :][..., [2, 1, 0, 3]].tobytes()
    if rotation == 180 and not mirror:
        return packed[::-1, ::-1, :][..., [2, 1, 0, 3]].tobytes()
    if rotation == 180 and mirror:
        return packed[::-1, :, :].tobytes()

    y, u, v = _unpack_yuyv422(frame, width, height)
    quarter_turns = rotation // 90
    y = np.rot90(y, k=-quarter_turns)
    u = np.rot90(u, k=-quarter_turns)
    v = np.rot90(v, k=-quarter_turns)
    y = _cover_plane(y, width, height)
    u = _cover_plane(u, width, height)
    v = _cover_plane(v, width, height)
    if mirror:
        y = y[:, ::-1]
        u = u[:, ::-1]
        v = v[:, ::-1]
    return _pack_yuyv422(y, u, v)


def env_port(name: str, default: int) -> int:
    value = env_int(name, default)
    if value > 65_535:
        raise ValueError(f"{name} must be at most 65535, got {value}")
    return value


class Receiver:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.decoders: dict[str, subprocess.Popen[bytes]] = {}
        self.sink: subprocess.Popen[bytes] | None = None
        self.active_input: str | None = None
        self.control_server: socket.socket | None = None
        self.control_thread: threading.Thread | None = None
        self.network_frames = 0
        self.output_frames = 0
        self.decoder_restarts = 0
        self.sink_restarts = 0
        self.virtual_lock_attempted = False
        self.started = time.monotonic()
        # Prerecorded playback position tracking.  When the file_relay decoder
        # (re)starts we record the wall clock and the seek it started from; the
        # live position is seek_base + elapsed, wrapped at the clip duration.
        self._prerecorded_started_at: float | None = None
        self._prerecorded_seek_base = 0.0
        self._prerecorded_duration: float | None = None
        self._prerecorded_duration_path: str | None = None
        self._prerecorded_position_value: float | None = None

        self.virtual_cameras = resolve_virtual_devices(os.environ)
        self.width = env_int("VIDEO_WIDTH", 1280)
        self.height = env_int("VIDEO_HEIGHT", 720)
        self.fps = env_int("VIDEO_FPS", 30)
        self.latency_us = env_int("SRT_LATENCY_US", 100_000)
        device_slot = env_int("DEVICE_SLOT", 1, 0)
        if device_slot > 4:
            raise ValueError("DEVICE_SLOT must be between 0 and 4")
        expected_return = 10_001 + device_slot * 2
        direct_port = env_port("WINDOWS_RETURN_PORT", expected_return)
        if direct_port != expected_return:
            raise ValueError(
                f"WINDOWS_RETURN_PORT must be {expected_return} for slot {device_slot}"
            )
        windows_host = os.environ.get("WINDOWS_HOST", "192.168.1.35").strip()
        legacy_broadcast_port = env_port("WINDOWS_BROADCAST_PORT", 10_010)
        broadcast_port = env_port(
            "WINDOWS_SELECTED_STREAM_PORT", legacy_broadcast_port
        )
        local_preview_port = env_port("LOCAL_PREVIEW_PORT", 11_000)
        manager_output_preview_port = env_port(
            "MANAGER_OUTPUT_PREVIEW_PORT", 11_003
        )
        local_processed_port = env_port("LOCAL_PROCESSED_PORT", 11_006)
        local_processed_preview_port = env_port(
            "LOCAL_PROCESSED_PREVIEW_PORT", 11_007
        )
        local_prerecorded_port = env_port("LOCAL_PRERECORDED_PORT", 11_010)
        local_phone_relay_source_port = env_port("LOCAL_PHONE_RELAY_SOURCE_PORT", 11_009)
        windows_phone_relay_port = env_port(
            "WINDOWS_PHONE_RELAY_SOURCE_PORT", 11_008
        )
        # A dedicated encoded tap for the Input tab's framing preview.  It must
        # be distinct from the Output tab's result preview (manager_output_
        # preview_port) so the two previews never split each other's datagrams.
        prerecorded_preview_port = env_port(
            "PRERECORDED_PREVIEW_PORT", 11_011
        )
        if len(
            {
                local_preview_port,
                manager_output_preview_port,
                local_processed_port,
                local_processed_preview_port,
                local_prerecorded_port,
                windows_phone_relay_port,
                prerecorded_preview_port,
            }
        ) != 7 or direct_port in {
            local_preview_port,
            manager_output_preview_port,
            local_processed_port,
            local_processed_preview_port,
            local_prerecorded_port,
            windows_phone_relay_port,
            prerecorded_preview_port,
        }:
            raise ValueError(
                "WINDOWS_RETURN_PORT and all local receiver ports must be distinct"
            )
        self.source_intent_path = Path(
            os.environ.get(
                "RECEIVER_SOURCE_STATE_FILE",
                str(DEFAULT_SOURCE_STATE_FILE),
            )
        )
        self.source_mode, self.source_restore_error = (
            self._restore_source_mode()
        )
        self.input_priority = SOURCE_PRIORITIES[self.source_mode]
        self.control_socket_path = state_dir() / "receiver-control.sock"
        self.output_state_path = state_dir() / "receiver-output.json"
        self.output_enabled = _environment_boolean("OUTPUT_ENABLED", True)
        self.output_mirror = _environment_boolean("OUTPUT_MIRROR", False)
        output_rotation = env_int("OUTPUT_ROTATION", 0, 0)
        self.output_mirror, self.output_rotation = _validate_output_transform(
            self.output_mirror, output_rotation
        )
        self.output_transform_revision = 0
        self._restore_output_transform()
        self.input_specs: dict[str, dict[str, str | int]] = {
            "local_processed": {
                "kind": "udp",
                "host": "127.0.0.1",
                "port": local_processed_port,
                # The processor emits an IDR (with in-band SPS/PPS) about twice
                # a second.  The probe only has to span one keyframe interval
                # (~0.5s) to lock on without the "non-existing PPS 0" failure,
                # so keep it tight: a smaller probe means less startup and
                # reconnect latency than the old 2s/2MB window.
                "probesize": 800_000,
                "analyzeduration": 600_000,
                "preview_port": local_processed_preview_port,
            },
            "local_prerecorded": {
                "kind": "file_relay",
                "host": "127.0.0.1",
                "port": local_prerecorded_port,
                "probesize": 800_000,
                "analyzeduration": 600_000,
                "preview_port": manager_output_preview_port,
                "framing_preview_port": prerecorded_preview_port,
                # Prerecorded feeds the phone through the same local relay port
                # (11009) the native phone processor uses.  The two never write
                # it at once: the processor only opens its 11009 return while a
                # phone camera session is active, and prerecorded input parks
                # the processor, so the file_relay owns 11009 exclusively while
                # prerecorded is the selected source.
                "phone_relay_port": local_phone_relay_source_port,
            },
            "processed_return": {"kind": "srt_listener", "port": direct_port},
            "selected_stream": {
                "kind": "srt_caller",
                "host": windows_host,
                "port": broadcast_port,
                "preview_port": manager_output_preview_port,
                # A second, dedicated copy prevents two UDP readers from
                # splitting the manager's preview datagrams.  The exclusive
                # phone-return relay is the only consumer of this endpoint.
                "phone_relay_port": windows_phone_relay_port,
            },
            "local_raw": {"kind": "udp", "host": "127.0.0.1", "port": local_preview_port},
        }
        self.input_stats: dict[str, dict[str, object]] = {
            name: {
                **spec,
                "frames": 0,
                "restarts": 0,
                "last_frame_at": None,
                "frame": None,
            }
            for name, spec in self.input_specs.items()
        }
        self.stale_seconds = env_float("RECEIVER_STALE_SECONDS", 1.0, 0.1)
        self.retry_seconds = env_float("RETRY_SECONDS", 2.0, 0.2)
        self.frame_bytes = self.width * self.height * 2  # packed YUYV422
        self.placeholder = bytes((32, 128, 32, 128)) * (self.width * self.height // 2)
        self.disabled_placeholder = bytes((16, 128, 16, 128)) * (
            self.width * self.height // 2
        )
        self._rng = np.random.default_rng()
        self._noise_pool = self._make_noise_pool()

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed")

    def _restore_source_mode(self) -> tuple[str, str | None]:
        """Restore durable GUI intent, using the environment only if absent.

        A corrupt durable document must not reactivate an unrelated processor
        through a stale environment default. Raw is the receiver's fail-safe:
        it preserves the stable V4L2 sink while making no processed-route claim.
        """
        try:
            document = json.loads(
                self.source_intent_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return _environment_source_mode(), None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return (
                "raw",
                f"durable receiver source ignored (fail-safe raw): {error}",
            )
        try:
            if not isinstance(document, Mapping):
                raise ValueError("state is not a JSON object")
            if document.get("schema_version") != 1:
                raise ValueError("state schema_version must be 1")
            selected = document.get("source")
            if selected not in SOURCE_PRIORITIES:
                raise ValueError(
                    "state source must be auto, local, windows, or raw"
                )
            return str(selected), None
        except (TypeError, ValueError) as error:
            return (
                "raw",
                f"durable receiver source ignored (fail-safe raw): {error}",
            )

    def _persist_source_mode_locked(self, source: str) -> None:
        _atomic_json(
            self.source_intent_path,
            {"schema_version": 1, "source": source},
        )

    def _restore_output_transform(self) -> None:
        """Restore the last hot transform across a receiver service restart."""
        try:
            document = json.loads(self.output_state_path.read_text(encoding="utf-8"))
            enabled = document.get("enabled", self.output_enabled)
            if not isinstance(enabled, bool):
                raise ValueError("enabled must be true or false")
            mirror, rotation = _validate_output_transform(
                document.get("mirror"), document.get("rotation")
            )
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, AttributeError, ValueError):
            # Runtime state is a convenience cache; invalid state must never
            # prevent the stable camera producer from starting.
            return
        self.output_enabled = enabled
        self.output_mirror = mirror
        self.output_rotation = rotation
        revision = document.get("revision", 0)
        if (
            isinstance(revision, int)
            and not isinstance(revision, bool)
            and revision >= 0
        ):
            self.output_transform_revision = revision

    def _persist_output_transform(self) -> bool:
        try:
            write_state(
                "receiver-output",
                {
                    "enabled": self.output_enabled,
                    "mirror": self.output_mirror,
                    "rotation": self.output_rotation,
                    "revision": self.output_transform_revision,
                },
            )
        except OSError:
            return False
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.control_server is not None:
            try:
                self.control_server.close()
            except OSError:
                pass
        with self.lock:
            decoders = list(self.decoders.values())
        for decoder in decoders:
            stop_process(decoder)
        stop_process(self.sink)

    def _handle_control_connection(self, connection: socket.socket) -> None:
        try:
            payload = connection.recv(4097)
            if not payload or len(payload) > 4096:
                raise ValueError("receiver configuration request is empty or too large")
            request = json.loads(payload.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("expected a JSON object")

            has_source = "source" in request
            requested: str | None = None
            if has_source:
                requested = str(request["source"]).strip().lower()
                if requested not in SOURCE_PRIORITIES:
                    raise ValueError("source must be auto, local, windows, or raw")

            transform_request = request.get("transform")
            if transform_request is not None and not isinstance(transform_request, dict):
                raise ValueError("transform must be a JSON object")
            transform_values = dict(transform_request or {})
            for key in ("mirror", "rotation"):
                if key in request:
                    if key in transform_values and transform_values[key] != request[key]:
                        raise ValueError(f"conflicting {key} values")
                    transform_values[key] = request[key]
            unknown_transform = set(transform_values) - {"mirror", "rotation"}
            if unknown_transform:
                unsupported = ", ".join(sorted(str(key) for key in unknown_transform))
                raise ValueError(f"unsupported output transform: {unsupported}")
            has_transform = bool(transform_values)
            has_enabled = "enabled" in request
            enabled_request = request.get("enabled", self.output_enabled)
            if has_enabled and not isinstance(enabled_request, bool):
                raise ValueError("enabled must be true or false")
            if not has_source and not has_transform and not has_enabled:
                raise ValueError("request must include source, transform, or enabled")

            with self.lock:
                mirror = transform_values.get("mirror", self.output_mirror)
                rotation = transform_values.get("rotation", self.output_rotation)
                mirror, rotation = _validate_output_transform(mirror, rotation)
                output_changed = (
                    enabled_request != self.output_enabled
                    or mirror != self.output_mirror
                    or rotation != self.output_rotation
                )
                if requested is not None:
                    # Persist before publishing the hot selection. If storage
                    # fails, the request is rejected and the existing source,
                    # active frame, decoder workers, and camera sink stay put.
                    self._persist_source_mode_locked(requested)
                    self.source_mode = requested
                    self.input_priority = SOURCE_PRIORITIES[requested]
                    self.active_input = None
                    self.source_restore_error = None
                if output_changed:
                    self.output_enabled = enabled_request
                    self.output_mirror = mirror
                    self.output_rotation = rotation
                    self.output_transform_revision += 1
                priority = list(self.input_priority)
                source_mode = self.source_mode
                output_transform = {
                    "mirror": self.output_mirror,
                    "rotation": self.output_rotation,
                    "revision": self.output_transform_revision,
                }
                output_enabled = self.output_enabled
                sink = self.sink
                sink_pid = (
                    None if sink is None or sink.poll() is not None else sink.pid
                )
                sink_restarts = self.sink_restarts
            persisted = not output_changed or self._persist_output_transform()
            response = {
                "ok": True,
                "source": source_mode,
                "priority": priority,
                "source_persisted": requested is not None,
                "source_intent_file": str(self.source_intent_path),
                "virtual_camera": str(self.virtual_cameras[0]),
                "virtual_cameras": [
                    str(camera) for camera in self.virtual_cameras
                ],
                "output_transform": output_transform,
                "output_enabled": output_enabled,
                "persisted": persisted,
                "sink_pid": sink_pid,
                "sink_restarts": sink_restarts,
                "detail": "configuration applied without restarting the camera sink",
            }
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        try:
            connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
        except OSError:
            pass

    def control_loop(self) -> None:
        self.control_socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.control_socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.control_server = server
        server.bind(str(self.control_socket_path))
        os.chmod(self.control_socket_path, 0o666)
        server.listen(4)
        server.settimeout(1.0)
        try:
            while not self.stop_event.is_set():
                try:
                    connection, _address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with connection:
                    connection.settimeout(3.0)
                    self._handle_control_connection(connection)
        finally:
            try:
                self.control_socket_path.unlink()
            except FileNotFoundError:
                pass

    def start_control_server(self) -> None:
        self.control_thread = threading.Thread(
            target=self.control_loop,
            name="receiver-source-control",
            daemon=True,
        )
        self.control_thread.start()

    def decoder_command(self, spec: dict[str, str | int]) -> list[str]:
        kind = str(spec["kind"])
        port = int(spec["port"])
        if kind in ("srt", "srt_listener", "srt_caller"):
            query = srt_query(
                {
                    "mode": "caller" if kind == "srt_caller" else "listener",
                    "transtype": "live",
                    "latency": self.latency_us,
                    "timeout": 5_000_000,
                    "tlpktdrop": 1,
                    "messageapi": 1,
                    "pkt_size": 1316,
                    **(
                        {"connect_timeout": 3000}
                        if kind == "srt_caller"
                        else {}
                    ),
                }
            )
            source = (
                f"srt://{spec['host']}:{port}?{query}"
                if kind == "srt_caller"
                else f"srt://0.0.0.0:{port}?{query}"
            )
        elif kind == "udp":
            source = (
                f"udp://{spec['host']}:{port}?reuse=1&fifo_size=1000000&"
                "overrun_nonfatal=1"
            )
        elif kind == "fifo":
            fifo_path = str(spec["fifo_path"])
            os.makedirs(os.path.dirname(fifo_path), exist_ok=True)
            if not os.path.exists(fifo_path):
                os.mkfifo(fifo_path)
            source = fifo_path
        elif kind == "file_relay":
            # file_relay reads from a video file path stored in a state file.
            # The actual command is built in decoder_loop, not here.
            source = "pipe:0"  # placeholder, not used
        else:
            raise ValueError(f"unsupported receiver input kind: {kind}")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-analyzeduration",
            str(spec.get("analyzeduration", 0)),
            "-probesize",
            str(spec.get("probesize", 32768)),
            "-f",
            "mpegts",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            # Phone-accurate fit: scale until the source covers the fixed output
            # box (force_original_aspect_ratio=increase), then centre-crop the
            # equal overflow. This matches a camera app's centerCrop /
            # SCALER_CROP so a portrait or differently-proportioned source is
            # never stretched into the landscape box and never letterboxed.
            # The output geometry stays locked, so consumers see no format
            # renegotiation when the selected source's proportions change.
            #
            # Rotation is handled by ffmpeg's default -autorotate, which applies
            # any display-matrix side data before the filtergraph runs, so the
            # input frames are already upright here.  A manual rotate filter is
            # NOT used: the rotate filter's angle expression cannot read frame
            # metadata (metadata() is undefined there), so an expression like
            # rotate='angle=...metadata(rotate)...' fails to parse with
            # "Invalid argument" and silently drops every frame.
            (
                f"scale={self.width}:{self.height}:"
                f"force_original_aspect_ratio=increase:flags=fast_bilinear,"
                f"crop={self.width}:{self.height},"
                f"fps={self.fps},format=yuyv422"
            ),
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuyv422",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        copy_ports = [spec.get("preview_port"), spec.get("phone_relay_port")]
        for preview_port in copy_ports:
            if preview_port is None:
                continue
            command.extend(
                [
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    "-bsf:v",
                    "dump_extra=freq=keyframe",
                    "-mpegts_flags",
                    "resend_headers",
                    "-muxdelay",
                    "0",
                    "-muxpreload",
                    "0",
                    "-flush_packets",
                    "1",
                    "-f",
                    "mpegts",
                    f"udp://127.0.0.1:{int(preview_port)}?pkt_size=1316",
                ]
            )
        return command

    def sink_command(self) -> list[str]:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuyv422",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
        ]
        for camera in self.virtual_cameras:
            command += [
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                "yuyv422",
                "-f",
                "v4l2",
                str(camera),
            ]
        return command

    def state(self, status: str, detail: str = "") -> dict[str, object]:
        now = time.monotonic()
        with self.lock:
            network_frames = self.network_frames
            active_input = self.active_input
            source_mode = self.source_mode
            input_priority = list(self.input_priority)
            output_transform = {
                "mirror": self.output_mirror,
                "rotation": self.output_rotation,
                "revision": self.output_transform_revision,
            }
            output_enabled = self.output_enabled
            inputs = {
                name: {
                    "port": values["port"],
                    "kind": values["kind"],
                    "host": values.get("host"),
                    "frames": values["frames"],
                    "restarts": values["restarts"],
                    "last_frame_age": (
                        None
                        if values["last_frame_at"] is None
                        else round(now - float(values["last_frame_at"]), 3)
                    ),
                    "pid": (
                        self.decoders[name].pid
                        if name in self.decoders and self.decoders[name].poll() is None
                        else None
                    ),
                }
                for name, values in self.input_stats.items()
            }
            active_last = (
                self.input_stats.get(active_input, {}).get("last_frame_at")
                if active_input
                else None
            )
            age = None if active_last is None else round(now - float(active_last), 3)
            decoder_pid = (
                inputs.get(active_input, {}).get("pid")
                if active_input is not None
                else next((item["pid"] for item in inputs.values() if item["pid"]), None)
            )
        return {
            "status": status,
            "detail": detail,
            "virtual_camera": str(self.virtual_cameras[0]),
            "virtual_cameras": [str(camera) for camera in self.virtual_cameras],
            "network_frames": network_frames,
            "output_frames": self.output_frames,
            "last_network_frame_age": age,
            "active_input": active_input,
            "source_mode": source_mode,
            "source_intent_file": str(self.source_intent_path),
            "source_restore_error": self.source_restore_error,
            "input_priority": input_priority,
            "output_transform": output_transform,
            "output_enabled": output_enabled,
            "receiver_control_socket": str(self.control_socket_path),
            "inputs": inputs,
            "decoder_restarts": self.decoder_restarts,
            "sink_restarts": self.sink_restarts,
            "uptime_seconds": round(now - self.started, 1),
            "decoder_pid": decoder_pid,
            "sink_pid": None if self.sink is None else self.sink.pid,
            "prerecorded": self._prerecorded_status(),
        }

    def _prerecorded_status(self) -> dict[str, object]:
        """Playback state for the GUI seek/transport bar."""
        position, duration = self._prerecorded_position()
        paused, _seek = self._prerecorded_playback()
        return {
            "position": None if position is None else round(position, 2),
            "duration": None if duration is None else round(duration, 2),
            "paused": paused,
        }

    def decoder_loop(self, input_name: str, spec: dict[str, str | int]) -> None:
        try:
            self._decoder_loop_inner(input_name, spec)
        except Exception:
            pass

    def _decoder_loop_inner(self, input_name: str, spec: dict[str, str | int]) -> None:
        while not self.stop_event.is_set():
            if str(spec.get("kind")) == "file_relay":
                command = self._file_relay_command(spec)
                if command is None:
                    if not self.stop_event.wait(2.0):
                        continue
                    break
                # Remember the source path this decoder was built for so a
                # video change recycles it promptly.  Framing (offset/zoom)
                # edits are applied live over zmq instead — see below — so they
                # never restart the decoder and the output stays smooth.
                relay_source = self._prerecorded_source_signature()
                relay_adjust = self._prerecorded_adjust_signature()
                relay_seek = self._prerecorded_playback_seek_signature()
                relay_mode = self._prerecorded_mode()
                # Record the playback baseline so the published position is
                # seek_base + elapsed while this decoder session runs.
                with self.lock:
                    self._prerecorded_seek_base = relay_seek or 0.0
                    self._prerecorded_started_at = time.monotonic()
                self._ensure_prerecorded_duration(
                    self._prerecorded_source_signature()
                )
            else:
                command = self.decoder_command(spec)
                relay_source = None
                relay_adjust = None
                relay_seek = None
            decoder = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0)
            with self.lock:
                self.decoders[input_name] = decoder
                self.decoder_restarts += 1
                self.input_stats[input_name]["restarts"] = int(
                    self.input_stats[input_name]["restarts"] or 0
                ) + 1
            buffer = bytearray()
            received_any = False
            last_decoded_at = time.monotonic()
            last_signature_check = time.monotonic()
            assert decoder.stdout is not None
            descriptor = decoder.stdout.fileno()
            os.set_blocking(descriptor, False)

            try:
                while not self.stop_event.is_set() and decoder.poll() is None:
                    # A file_relay produces frames continuously, so the select
                    # below is almost always readable; poll source + framing on
                    # a throttled interval here.  A video change restarts the
                    # decoder; a framing change is pushed live over zmq to the
                    # crop@live filter so the output pans/zooms without a
                    # restart (no black gap).
                    if (
                        relay_source is not None
                        and time.monotonic() - last_signature_check >= 0.2
                    ):
                        last_signature_check = time.monotonic()
                        if self._prerecorded_source_signature() != relay_source:
                            stop_process(decoder)
                            break
                        # A seek is a deliberate jump: restart decode at the
                        # new point (input -ss).  Pan/zoom stay live via zmq.
                        if self._prerecorded_playback_seek_signature() != relay_seek:
                            stop_process(decoder)
                            break
                        # A mode change (loop<->once<->freeze) alters the
                        # stream_loop flag, so it needs a decode restart.
                        if self._prerecorded_mode() != relay_mode:
                            stop_process(decoder)
                            break
                        current_adjust = self._prerecorded_adjust_signature()
                        if current_adjust != relay_adjust:
                            relay_adjust = current_adjust
                            self._send_prerecorded_framing()
                    # Pause: stop consuming frames so ffmpeg blocks on pipe
                    # backpressure (paused at the current point), and keep the
                    # last shown frame fresh so the sink holds it instead of
                    # falling back to the placeholder.
                    paused, _seek = self._prerecorded_playback()
                    if paused:
                        with self.lock:
                            if self.input_stats[input_name]["frame"] is not None:
                                self.input_stats[input_name]["last_frame_at"] = (
                                    time.monotonic()
                                )
                        self.stop_event.wait(0.1)
                        continue
                    readable, _, _ = select.select([descriptor], [], [], 0.5)
                    if not readable:
                        if (
                            spec["kind"] in ("udp", "srt_caller")
                            and received_any
                            and time.monotonic() - last_decoded_at > 3.0
                        ):
                            # UDP/MPEG-TS has no connection close to wake the
                            # decoder when a publisher restarts. Recycling the
                            # passive decoder re-acquires fresh stream headers.
                            stop_process(decoder)
                            break
                        continue
                    try:
                        chunk = os.read(descriptor, 1 << 20)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while len(buffer) >= self.frame_bytes:
                        frame = bytes(buffer[: self.frame_bytes])
                        del buffer[: self.frame_bytes]
                        with self.lock:
                            received_at = time.monotonic()
                            received_any = True
                            last_decoded_at = received_at
                            self.network_frames += 1
                            self.input_stats[input_name]["frames"] = int(
                                self.input_stats[input_name]["frames"] or 0
                            ) + 1
                            self.input_stats[input_name]["last_frame_at"] = received_at
                            self.input_stats[input_name]["frame"] = frame
            finally:
                stop_process(decoder)
                with self.lock:
                    if self.decoders.get(input_name) is decoder:
                        self.decoders.pop(input_name, None)
            # End-of-file handling for a file_relay.  Loop mode never reaches
            # here (ffmpeg loops internally with -stream_loop -1).  once/freeze
            # DID play through: freeze holds the last decoded frame on screen;
            # once lets it fall idle.  In both we must NOT immediately respawn
            # ffmpeg (that is the old black-flash re-loop); instead idle until
            # the GUI changes the source, seek, or mode.
            if relay_source is not None and not self.stop_event.is_set():
                end_mode = self._prerecorded_mode()
                if end_mode in ("once", "freeze"):
                    baseline = (
                        self._prerecorded_source_signature(),
                        self._prerecorded_playback_seek_signature(),
                        end_mode,
                    )
                    while not self.stop_event.is_set():
                        if end_mode == "freeze":
                            # Keep the last frame "live" so the sink holds it
                            # instead of falling back to the placeholder.
                            with self.lock:
                                if self.input_stats[input_name]["frame"] is not None:
                                    self.input_stats[input_name][
                                        "last_frame_at"
                                    ] = time.monotonic()
                        current = (
                            self._prerecorded_source_signature(),
                            self._prerecorded_playback_seek_signature(),
                            self._prerecorded_mode(),
                        )
                        if current != baseline:
                            break  # GUI changed something -> rebuild decoder
                        if self.stop_event.wait(0.2):
                            break
                    continue
            if not self.stop_event.wait(self.retry_seconds):
                continue

    def _prerecorded_source_signature(self) -> str:
        """The current prerecorded video path (a change here restarts decode)."""
        try:
            return (state_dir() / "prerecorded-source.txt").read_text().strip()
        except (FileNotFoundError, PermissionError):
            return ""

    def _prerecorded_adjust_signature(self) -> str:
        """The current framing document text (a change is applied live via zmq)."""
        try:
            return (state_dir() / "prerecorded-adjust.json").read_text().strip()
        except (FileNotFoundError, PermissionError):
            return ""

    def _prerecorded_playback(self) -> tuple[bool, float | None]:
        """Read playback state: (paused, seek_seconds_or_None)."""
        paused = False
        seek: float | None = None
        try:
            document = json.loads(
                (state_dir() / "prerecorded-playback.json").read_text(
                    encoding="utf-8"
                )
            )
            if isinstance(document, dict):
                paused = bool(document.get("paused", False))
                raw_seek = document.get("seek")
                if raw_seek is not None:
                    seek = max(0.0, float(raw_seek))
        except (FileNotFoundError, PermissionError, ValueError, TypeError):
            pass
        return paused, seek

    def _prerecorded_playback_seek_signature(self) -> float | None:
        """Only the seek target (a change restarts decode at the new point)."""
        return self._prerecorded_playback()[1]

    def _prerecorded_mode(self) -> str:
        """Playback mode: loop (default) | once | freeze."""
        try:
            document = json.loads(
                (state_dir() / "prerecorded-playback.json").read_text(
                    encoding="utf-8"
                )
            )
            if isinstance(document, dict):
                mode = str(document.get("mode", "loop"))
                if mode in ("loop", "once", "freeze"):
                    return mode
        except (FileNotFoundError, PermissionError, ValueError, TypeError):
            pass
        return "loop"

    def _ensure_prerecorded_duration(self, video_path: str) -> None:
        """Probe the clip duration once (cached per source path)."""
        if not video_path:
            return
        if self._prerecorded_duration_path == video_path:
            return
        duration: float | None = None
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                check=False, capture_output=True, text=True, timeout=5.0,
            )
            value = result.stdout.strip()
            if value:
                duration = max(0.0, float(value))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            duration = None
        with self.lock:
            self._prerecorded_duration = duration
            self._prerecorded_duration_path = video_path

    def _prerecorded_position(self) -> tuple[float | None, float | None]:
        """Current playback position and duration in seconds (None if unknown).

        While playing, the position advances with wall clock from the seek base.
        The last computed value is cached so pausing freezes it exactly where
        playback stopped rather than snapping back to the seek point.
        """
        with self.lock:
            started = self._prerecorded_started_at
            base = self._prerecorded_seek_base
            duration = self._prerecorded_duration
            cached = self._prerecorded_position_value
        if started is None:
            return cached, duration
        paused, _seek = self._prerecorded_playback()
        if paused:
            return cached, duration
        position = base + (time.monotonic() - started)
        if duration and duration > 0:
            position = position % duration
        with self.lock:
            self._prerecorded_position_value = position
        return position, duration

    def _send_prerecorded_framing(self) -> None:
        """Push the current offset/zoom to the running decoder's crop@live.

        Uses zmq to update the crop window in place so /dev/deep-live-cam pans
        and zooms without restarting ffmpeg.  Best-effort: if the command fails
        the previous framing simply persists until the next poll.
        """
        try:
            import zmq  # noqa: PLC0415 - optional, only for live framing
        except ImportError:
            return
        offset_x, offset_y, zoom = self._prerecorded_framing()
        crop_w, crop_h, crop_x, crop_y = self._prerecorded_crop_params(
            offset_x, offset_y, zoom
        )
        context = zmq.Context.instance()
        sock = context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 300)
        sock.setsockopt(zmq.SNDTIMEO, 300)
        try:
            sock.connect(self.PRERECORDED_ZMQ_ENDPOINT)
            for command, value in (
                ("w", crop_w),
                ("h", crop_h),
                ("x", crop_x),
                ("y", crop_y),
            ):
                sock.send_string(f"crop@live {command} {value}")
                try:
                    sock.recv_string()
                except zmq.error.Again:
                    break
        except zmq.ZMQError:
            pass
        finally:
            sock.close()

    def _prerecorded_framing(self) -> tuple[int, int, float]:
        """Read the prerecorded framing document (offset px, zoom about centre)."""
        offset_x = 0
        offset_y = 0
        zoom = 1.0
        adjust_file = state_dir() / "prerecorded-adjust.json"
        try:
            document = json.loads(adjust_file.read_text(encoding="utf-8"))
            if isinstance(document, dict):
                offset_x = int(document.get("offset_x", 0))
                offset_y = int(document.get("offset_y", 0))
                zoom = float(document.get("zoom", 1.0))
        except (FileNotFoundError, PermissionError, ValueError, TypeError):
            pass
        return offset_x, offset_y, min(4.0, max(0.25, zoom))

    # Framing pipeline geometry.  The source is scaled once to cover the locked
    # WxH box, padded with a generous black margin on every side, then a live
    # crop pulls the visible window out and scales it back to WxH.  Because the
    # scale and pad are fixed, zoom and pan are just live crop w/h/x/y updates
    # (sent over zmq) and never restart the decoder — so /dev/deep-live-cam pans
    # and zooms smoothly.  PAD must exceed the largest pan the UI allows.
    PRERECORDED_PAD = 1280
    PRERECORDED_ZMQ_ENDPOINT = "ipc:///run/deep-live-cam/prerecorded-zmq.sock"

    def _prerecorded_crop_params(
        self, offset_x: int, offset_y: int, zoom: float
    ) -> tuple[int, int, int, int]:
        """Crop window (w, h, x, y) in padded-canvas pixels for a framing.

        zoom scales about the centre: a larger zoom means a smaller crop window
        (more magnification).  offset_x/offset_y are in OUTPUT pixels (what the
        slider shows); one output pixel is 1/zoom canvas pixels, so a positive
        offset shifts the video right/down by moving the crop window left/up.
        """
        w = self.width
        h = self.height
        pad = self.PRERECORDED_PAD
        zoom = min(4.0, max(0.25, zoom))
        cw = max(2, (int(round(w / zoom)) // 2) * 2)
        ch = max(2, (int(round(h / zoom)) // 2) * 2)
        x = int(round(pad + (w - cw) / 2 - offset_x / zoom))
        y = int(round(pad + (h - ch) / 2 - offset_y / zoom))
        # Keep the window inside the padded canvas so crop never errors.
        max_x = w + 2 * pad - cw
        max_y = h + 2 * pad - ch
        x = max(0, min(max_x, x))
        y = max(0, min(max_y, y))
        return cw, ch, x, y

    def _prerecorded_filter_complex(self, split_outputs: int) -> str:
        """Framing graph with a live-commandable crop for smooth zoom + pan.

        scale-to-cover -> pad black margins -> zmq -> crop@live -> scale to WxH
        -> split.  The scale and pad are fixed, so the manager changes framing
        by sending crop@live w/h/x/y commands over zmq without restarting the
        decoder; anywhere the crop window falls on the black pad shows black.
        """
        offset_x, offset_y, zoom = self._prerecorded_framing()
        w = self.width
        h = self.height
        pad = self.PRERECORDED_PAD
        cw = w + 2 * pad
        ch = h + 2 * pad
        crop_w, crop_h, crop_x, crop_y = self._prerecorded_crop_params(
            offset_x, offset_y, zoom
        )
        labels = "".join(f"[v{index}]" for index in range(split_outputs))
        # zmq binds a unix-socket (IPC) endpoint rather than TCP: it avoids the
        # TCP resolver (which crashed under the sandboxed service) and stays
        # within AF_UNIX.  Its colons/slashes are doubly-escaped so the
        # filter-args parser (which strips one backslash level) sees the real
        # ipc:///path address.
        endpoint = self.PRERECORDED_ZMQ_ENDPOINT.replace(
            ":", chr(92) + chr(92) + ":"
        )
        return (
            f"[0:v:0]scale={w}:{h}:"
            f"force_original_aspect_ratio=increase:flags=fast_bilinear,"
            f"crop@fit={w}:{h},"
            f"pad={cw}:{ch}:{pad}:{pad}:color=black,"
            f"zmq=bind_address={endpoint},"
            f"crop@live=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y}:exact=1,"
            f"scale={w}:{h}:flags=fast_bilinear,"
            f"fps={self.fps},format=yuyv422,split={split_outputs}{labels}"
        )

    def _file_relay_command(self, spec: dict[str, str | int]) -> list[str] | None:
        """Build a decode command for a prerecorded video file."""
        source_file = state_dir() / "prerecorded-source.txt"
        try:
            video_path = source_file.read_text().strip()
        except (FileNotFoundError, PermissionError):
            return None
        if not video_path or not Path(video_path).exists():
            return None
        _paused, seek = self._prerecorded_playback()
        mode = self._prerecorded_mode()
        copy_ports = [
            port
            for port in (
                spec.get("preview_port"),
                spec.get("phone_relay_port"),
                spec.get("framing_preview_port"),
            )
            if port is not None
        ]
        # One framing pass, then split so the raw virtual-camera feed and every
        # MPEG-TS preview/phone tap all show the identical adjusted framing.
        split_outputs = 1 + len(copy_ports)
        filter_complex = self._prerecorded_filter_complex(split_outputs)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            "-re",
            "-fflags", "+genpts",
        ]
        # Loop mode loops INSIDE this single ffmpeg process (-stream_loop -1),
        # so end-of-file wraps back to frame 0 with no process restart and no
        # black gap.  once/freeze play through a single time; the decode loop
        # then holds the last frame (freeze) or falls idle (once).  -stream_loop
        # must precede -i.
        if mode == "loop":
            command.extend(["-stream_loop", "-1"])
        # A deliberate seek starts decode at the requested second (input-side
        # -ss is fast and keyframe-accurate enough for scrubbing).
        if seek is not None and seek > 0:
            command.extend(["-ss", f"{seek:.3f}"])
        command.extend([
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[v0]",
            "-an",
            "-c:v", "rawvideo",
            "-pix_fmt", "yuyv422",
            "-f", "rawvideo",
            "pipe:1",
        ])
        # H264 MPEG-TS preview/phone outputs carrying the same framed picture.
        # x264 baseline needs 4:2:0, so convert each tap from the shared 4:2:2
        # framed stream before encoding (the raw virtual-camera feed stays
        # yuyv422 for the sink).
        for index, port in enumerate(copy_ports, start=1):
            command.extend([
                "-map", f"[v{index}]",
                "-an",
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-profile:v", "baseline",
                "-b:v", "5M",
                "-maxrate", "5M",
                "-bufsize", "1M",
                "-g", "15",
                "-bf", "0",
                "-bsf:v", "dump_extra=freq=keyframe",
                "-mpegts_flags", "resend_headers",
                "-muxdelay", "0",
                "-muxpreload", "0",
                "-flush_packets", "1",
                "-f", "mpegts",
                f"udp://127.0.0.1:{int(port)}?pkt_size=1316",
            ])
        return command

    def prepare_virtual_camera(self) -> None:
        """Unlock any stale format lock and pre-set the format if possible.

        With exclusive_caps=1 the capture side rejects S_FMT until a producer
        is attached, so the authoritative format set happens when the sink
        starts; the lock is applied afterwards by lock_virtual_camera(). The
        output-side S_FMT here is best-effort for driver versions that accept
        it before a producer exists.
        """
        fmt = f"width={self.width},height={self.height},pixelformat=YUYV"
        commands = []
        for camera in self.virtual_cameras:
            commands += [
                ["v4l2-ctl", f"--device={camera}", "--set-ctrl=keep_format=0"],
                ["v4l2-ctl", f"--device={camera}", f"--set-fmt-video-out={fmt}"],
            ]
        for command in commands:
            try:
                subprocess.run(command, check=False, capture_output=True, timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def lock_virtual_camera(self) -> None:
        """Lock the producer-established format against consumer changes."""
        self.virtual_lock_attempted = True
        for camera in self.virtual_cameras:
            command = [
                "v4l2-ctl",
                f"--device={camera}",
                "--set-ctrl=keep_format=1,sustain_framerate=1",
            ]
            try:
                subprocess.run(command, check=False, capture_output=True, timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        camera = self.virtual_cameras[0]
        try:
            result = subprocess.run(
                ["v4l2-ctl", f"--device={camera}", "--get-fmt-video"],
                check=False, capture_output=True, text=True, timeout=3.0,
            )
            if f"{self.width}/{self.height}" not in result.stdout:
                print(
                    "deep-live-cam receiver: virtual camera format was not established: "
                    + "; ".join(result.stdout.strip().splitlines()[:2]),
                    file=sys.stderr,
                )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def start_sink(self) -> bool:
        if not all(camera.exists() for camera in self.virtual_cameras):
            return False
        self.prepare_virtual_camera()
        self.sink = subprocess.Popen(self.sink_command(), stdin=subprocess.PIPE, bufsize=0)
        self.sink_restarts += 1
        self.virtual_lock_attempted = False
        return True

    def _make_noise_pool(self, count: int = 32) -> list[np.ndarray]:
        """Triangular-distribution temporal dither frames (sigma ~1.6 LSB)."""
        rng = np.random.default_rng()
        pool = []
        for _ in range(count):
            noise = rng.integers(-2, 3, size=self.frame_bytes, dtype=np.int16)
            noise += rng.integers(-2, 3, size=self.frame_bytes, dtype=np.int16)
            pool.append(noise)
        return pool

    def _dither(self, frame: bytes) -> bytes:
        """Temporal dither for placeholder/repeated frames: a real sensor
        never emits bit-identical frames, so static placeholder output must
        not be bit-stable either."""
        index = int(self._rng.integers(0, len(self._noise_pool)))
        shift = int(self._rng.integers(0, max(1, self.frame_bytes // 8)))
        noise = np.roll(self._noise_pool[index], shift)
        pixels = np.frombuffer(frame, dtype=np.uint8).astype(np.int16)
        return np.clip(pixels + noise, 0, 255).astype(np.uint8).tobytes()

    def current_frame(self) -> tuple[bytes, bool, float | None, str | None]:
        now = time.monotonic()
        with self.lock:
            youngest_stale: float | None = None
            for input_name in self.input_priority:
                values = self.input_stats[input_name]
                timestamp = values["last_frame_at"]
                frame = values["frame"]
                if timestamp is None or frame is None:
                    continue
                age = now - float(timestamp)
                if youngest_stale is None or age < youngest_stale:
                    youngest_stale = age
                if age <= self.stale_seconds:
                    self.active_input = input_name
                    return bytes(frame), True, age, input_name
            self.active_input = None
        return self.placeholder, False, youngest_stale, None

    def write_frame(self, frame: bytes) -> None:
        """A raw pipe write may be partial; preserve exact frame boundaries."""
        assert self.sink is not None and self.sink.stdin is not None
        remaining = memoryview(frame)
        while remaining and not self.stop_event.is_set():
            written = self.sink.stdin.write(remaining)
            if written is None or written <= 0:
                raise BrokenPipeError("virtual-camera FFmpeg stopped accepting frames")
            remaining = remaining[written:]

    def transform_frame(self, frame: bytes) -> bytes:
        with self.lock:
            mirror = self.output_mirror
            rotation = self.output_rotation
        return transform_yuyv422(
            frame,
            self.width,
            self.height,
            mirror=mirror,
            rotation=rotation,
        )

    def render_output_frame(self, frame: bytes, *, streaming: bool) -> bytes:
        """Return one sink frame without changing sink ownership or cadence."""
        with self.lock:
            enabled = self.output_enabled
            mirror = self.output_mirror
            rotation = self.output_rotation
        if not enabled:
            return self.disabled_placeholder
        source_frame = frame if streaming else self._dither(frame)
        return transform_yuyv422(
            source_frame,
            self.width,
            self.height,
            mirror=mirror,
            rotation=rotation,
        )

    def _ensure_prerecorded_state_files(self) -> None:
        """Pre-create the prerecorded control files world-writable.

        The receiver runs as root but the GUI manager and the prerecorded relay
        run as the desktop user.  /run/deep-live-cam is root-owned, so those
        processes can only update these files if the receiver creates them 0666
        first.  Without this the manager's framing writes fail silently with
        Permission denied and offset/zoom edits never reach the decoder.
        """
        directory = state_dir()
        for name, initial in (
            ("prerecorded-source.txt", ""),
            ("prerecorded-adjust.json", '{"offset_x": 0, "offset_y": 0, "zoom": 1.0}'),
            ("prerecorded-playback.json", '{"paused": false, "seek": null}'),
        ):
            path = directory / name
            try:
                if not path.exists():
                    path.write_text(initial, encoding="utf-8")
                os.chmod(path, 0o666)
            except OSError:
                pass

    def run(self) -> int:
        install_signal_handlers(self.stop)
        self._ensure_prerecorded_state_files()
        self.start_control_server()
        decoder_threads = [
            threading.Thread(
                target=self.decoder_loop,
                args=(name, spec),
                name=f"srt-decoder-{name}",
                daemon=True,
            )
            for name, spec in self.input_specs.items()
        ]
        for decoder_thread in decoder_threads:
            decoder_thread.start()
        port_summary = ", ".join(
            f"{name}={spec['kind']} {spec['port']}"
            for name, spec in self.input_specs.items()
        )
        sd_notify(f"READY=1\nSTATUS=receiver supervisor started ({port_summary})")

        interval = 1.0 / self.fps
        deadline = time.monotonic()
        last_state = 0.0
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.sink is None or self.sink.poll() is not None:
                    stop_process(self.sink)
                    self.sink = None
                    if not self.start_sink():
                        if now - last_state >= 2.0:
                            waiting = ", ".join(str(camera) for camera in self.virtual_cameras)
                            write_state("receiver", self.state("waiting_virtual_camera", waiting))
                            sd_notify(f"WATCHDOG=1\nSTATUS=waiting for {waiting}")
                            last_state = now
                        self.stop_event.wait(0.5)
                        deadline = time.monotonic()
                        continue

                frame, streaming, age, active_input = self.current_frame()
                try:
                    # Fresh network frames already carry temporal dither from
                    # the Windows output leg; only synthetic/repeated frames
                    # need it here.
                    self.write_frame(
                        self.render_output_frame(frame, streaming=streaming)
                    )
                    self.output_frames += 1
                    if not self.virtual_lock_attempted:
                        self.lock_virtual_camera()
                except (BrokenPipeError, OSError):
                    stop_process(self.sink)
                    self.sink = None
                    continue

                now = time.monotonic()
                if now - last_state >= 2.0:
                    with self.lock:
                        output_enabled = self.output_enabled
                    if not output_enabled:
                        status = "disabled"
                        detail = "neutral output; camera sink remains active"
                    elif streaming:
                        status = "streaming"
                        detail = f"{active_input} frame age {age:.3f}s"
                    else:
                        status = "placeholder"
                        detail = "waiting for processed return, selected Windows SRT stream, or local raw feed"
                    write_state("receiver", self.state(status, detail))
                    sd_notify(f"WATCHDOG=1\nSTATUS={status}: {detail}")
                    last_state = now

                deadline += interval
                delay = deadline - time.monotonic()
                if delay > 0:
                    self.stop_event.wait(delay)
                elif delay < -interval:
                    deadline = time.monotonic()
        finally:
            self.stop()
            for decoder_thread in decoder_threads:
                decoder_thread.join(timeout=5.0)
            if self.control_thread is not None:
                self.control_thread.join(timeout=3.0)
            write_state("receiver", self.state("stopped"))
        return 0


def main() -> int:
    try:
        return Receiver().run()
    except (ValueError, RuntimeError) as exc:
        print(f"deep-live-cam receiver: {exc}", file=sys.stderr)
        sd_notify(f"STATUS=configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
