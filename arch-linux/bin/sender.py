#!/usr/bin/env python3
"""Supervise physical-camera capture and the SRT sender to Windows."""

from __future__ import annotations

import os
import json
import selectors
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import (
    env_float,
    env_int,
    install_signal_handlers,
    resolve_capture_device,
    sd_notify,
    srt_query,
    state_dir,
    stop_process,
    write_state,
)
from camera_profiles import (
    ARCH_CAMERA_PROFILES,
    CUSTOM_CAMERA_PROFILE,
    DEFAULT_CAMERA_PROFILE,
    profile_config_values,
)


class Sender:
    PROFILE_DEFAULTS = profile_config_values(DEFAULT_CAMERA_PROFILE)
    CONTROL_RANGES = {
        "brightness": (-64, 64),
        "contrast": (0, 64),
        "saturation": (0, 128),
        "hue": (-40, 40),
        "gamma": (72, 500),
        "gain": (0, 100),
        "sharpness": (0, 6),
        "backlight_compensation": (0, 2),
        "power_line_frequency": (0, 2),
        "exposure_time_absolute": (1, 5000),
        "exposure_dynamic_framerate": (0, 1),
        "white_balance_temperature": (2800, 6500),
    }

    def __init__(self) -> None:
        self.stop_requested = False
        self.process: subprocess.Popen[bytes] | None = None
        self.network_process: subprocess.Popen[bytes] | None = None
        self.network_thread: threading.Thread | None = None
        self.restarts = 0
        self.network_restarts = 0
        self.started = time.monotonic()
        self.last_frame_at: float | None = None
        self.network_last_frame_at: float | None = None
        self.frame_count = 0

        self.windows_host = os.environ.get("WINDOWS_HOST", "").strip()
        self.camera = Path(os.environ.get("PHYSICAL_CAMERA", "").strip())
        self.width = env_int("VIDEO_WIDTH", 1280)
        self.height = env_int("VIDEO_HEIGHT", 720)
        self.fps = env_int("VIDEO_FPS", 30)
        self.camera_width = env_int("CAMERA_WIDTH", self.width)
        self.camera_height = env_int("CAMERA_HEIGHT", self.height)
        self.camera_fps = env_int("CAMERA_FPS", self.fps)
        self.camera_profile = os.environ.get(
            "CAMERA_PROFILE", DEFAULT_CAMERA_PROFILE
        ).strip()
        if self.camera_profile not in {
            CUSTOM_CAMERA_PROFILE,
            *ARCH_CAMERA_PROFILES,
        }:
            raise ValueError(f"unsupported CAMERA_PROFILE: {self.camera_profile}")
        if self.camera_profile in ARCH_CAMERA_PROFILES:
            expected_profile = profile_config_values(self.camera_profile)
            mismatches = [
                f"{key}={os.environ[key]} (expected {expected})"
                for key, expected in expected_profile.items()
                if key in os.environ and os.environ[key] != expected
            ]
            if mismatches:
                raise ValueError(
                    f"CAMERA_PROFILE={self.camera_profile} does not match: "
                    + ", ".join(mismatches)
                )
        self.bitrate = os.environ.get("VIDEO_BITRATE", "4M")
        self.input_format = os.environ.get("CAMERA_INPUT_FORMAT", "mjpeg")
        self.encoder = os.environ.get("SENDER_ENCODER", "auto").lower()
        self.preset = os.environ.get("X264_PRESET", "ultrafast")
        self.vaapi_device = os.environ.get(
            "VAAPI_DEVICE", "/dev/dri/renderD128"
        )
        self.latency_us = env_int("SRT_LATENCY_US", 100_000)
        self.device_id = os.environ.get("DEVICE_ID", "arch-webcam").strip()
        self.device_slot = env_int("DEVICE_SLOT", 1, 0)
        if self.device_slot > 4:
            raise ValueError("DEVICE_SLOT must be between 0 and 4")
        self.windows_input_port = env_int(
            "WINDOWS_INPUT_PORT", 10_000 + self.device_slot * 2
        )
        expected_input_port = 10_000 + self.device_slot * 2
        if self.windows_input_port != expected_input_port:
            raise ValueError(
                f"WINDOWS_INPUT_PORT must be {expected_input_port} for slot "
                f"{self.device_slot}"
            )
        self.preview_port = env_int("LOCAL_PREVIEW_PORT", 11_000)
        self.manager_preview_port = env_int("MANAGER_PREVIEW_PORT", 11_001)
        self.network_source_port = env_int("NETWORK_SOURCE_PORT", 11_002)
        self.native_processor_source_port = env_int(
            "NATIVE_PROCESSOR_SOURCE_PORT", 11_005
        )
        if len(
            {
                self.preview_port,
                self.manager_preview_port,
                self.network_source_port,
                self.native_processor_source_port,
            }
        ) != 4:
            raise ValueError(
                "LOCAL_PREVIEW_PORT, MANAGER_PREVIEW_PORT, and "
                "NETWORK_SOURCE_PORT, and NATIVE_PROCESSOR_SOURCE_PORT "
                "must be distinct"
            )
        self.startup_timeout = env_float("SENDER_STARTUP_TIMEOUT", 20.0, 3.0)
        self.stall_timeout = env_float("SENDER_STALL_TIMEOUT", 10.0, 3.0)
        self.retry_seconds = env_float("RETRY_SECONDS", 2.0, 0.2)
        self.shadow = os.environ.get("SHADOW_ORIGINAL", "0") == "1"
        self.active_camera = self.camera
        self.controls_applied = False
        self.health_url = f"http://{self.windows_host}:8090/healthz"
        self.camera_control_overrides: dict[str, int | bool] = {}
        self.camera_control_lock = threading.Lock()
        self.control_socket_path = state_dir() / "sender-control.sock"
        self.control_server: socket.socket | None = None
        self.control_thread: threading.Thread | None = None

        if not self.windows_host or self.windows_host == "CHANGE_ME":
            raise ValueError("WINDOWS_HOST is not configured in /etc/deep-live-cam-arch.conf")
        if not str(self.camera):
            raise ValueError("PHYSICAL_CAMERA is not configured")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed")
        self.encoder = self.select_encoder(self.encoder)

    def select_encoder(self, requested: str) -> str:
        """Select a working low-latency encoder with a safe CPU fallback."""
        if requested not in ("auto", "libx264", "h264_vaapi"):
            raise ValueError(
                "SENDER_ENCODER must be auto, h264_vaapi, or libx264"
            )
        if requested == "libx264":
            return requested

        device = Path(self.vaapi_device)
        if device.exists():
            probe = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-vaapi_device", str(device),
                "-f", "lavfi", "-i",
                f"color=size={self.width}x{self.height}:rate={self.fps}",
                "-frames:v", "1", "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi", "-profile:v", "high",
                "-rc_mode", "CBR", "-async_depth", "1",
                "-quality", "1",
                "-b:v", self.bitrate, "-maxrate", self.bitrate,
                "-bufsize", self.bitrate,
                "-f", "null", "-",
            ]
            result = subprocess.run(
                probe, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=10.0, check=False,
            )
            if result.returncode == 0:
                return "h264_vaapi"
            detail = result.stderr.decode("utf-8", "replace").strip()
        else:
            detail = f"{device} does not exist"

        if requested == "h264_vaapi":
            raise RuntimeError(f"VAAPI encoder unavailable: {detail}")
        print(
            f"deep-live-cam sender: VAAPI unavailable ({detail}); "
            "using libx264",
            file=sys.stderr,
        )
        return "libx264"

    def stop(self) -> None:
        self.stop_requested = True
        if self.control_server is not None:
            try:
                self.control_server.close()
            except OSError:
                pass
        stop_process(self.network_process)
        stop_process(self.process)

    def state(self, status: str, detail: str = "") -> dict[str, object]:
        now = time.monotonic()
        return {
            "status": status,
            "detail": detail,
            "camera": str(self.camera),
            "capture": str(self.active_camera),
            "camera_mode": f"{self.camera_width}x{self.camera_height}@{self.camera_fps}",
            "camera_profile": self.camera_profile,
            # This is the owner's validated control vector, not a device
            # query. Passive diagnostics can fingerprint it without opening
            # or otherwise competing for the physical camera.
            "camera_controls": self._camera_controls(),
            "encoder": self.encoder,
            "bitrate": self.bitrate,
            "windows_host": self.windows_host,
            "device_id": self.device_id,
            "device_slot": self.device_slot,
            "windows_input_port": self.windows_input_port,
            "local_preview": f"udp://127.0.0.1:{self.preview_port}",
            "manager_preview": (
                f"udp://127.0.0.1:{self.manager_preview_port}"
            ),
            "network_source": (
                f"udp://127.0.0.1:{self.network_source_port}"
            ),
            "native_processor_source": (
                f"udp://127.0.0.1:{self.native_processor_source_port}"
            ),
            "network_pid": (
                None
                if self.network_process is None
                else self.network_process.pid
            ),
            "network_connected": (
                self.network_last_frame_at is not None
                and now - self.network_last_frame_at < 5.0
            ),
            "network_restarts": self.network_restarts,
            "camera_control_socket": str(self.control_socket_path),
            "frames": self.frame_count,
            "last_frame_age": None if self.last_frame_at is None else round(now - self.last_frame_at, 3),
            "restarts": self.restarts,
            "uptime_seconds": round(now - self.started, 1),
            "pid": None if self.process is None else self.process.pid,
        }

    def notify(self, status: str, detail: str = "") -> None:
        write_state("sender", self.state(status, detail))
        sd_notify(f"WATCHDOG=1\nSTATUS={status}: {detail}".rstrip())

    def wait_for_camera(self) -> bool:
        announced = ""
        while not self.stop_requested:
            target = resolve_capture_device(os.environ)
            if target.exists():
                self.active_camera = target
                return True
            self.controls_applied = False
            if self.shadow and self.camera.exists():
                status = "waiting_shadow"
                detail = f"waiting for preserved source node {target}"
            else:
                status = "waiting_camera"
                detail = f"waiting for {self.camera}"
            if announced != status:
                self.notify(status, detail)
                announced = status
            sd_notify("WATCHDOG=1")
            time.sleep(1.0)
        return False

    def _camera_controls(self) -> dict[str, str]:
        defaults = self.PROFILE_DEFAULTS
        controls = {
            "brightness": os.environ.get("CAMERA_BRIGHTNESS", defaults["CAMERA_BRIGHTNESS"]),
            "contrast": os.environ.get("CAMERA_CONTRAST", defaults["CAMERA_CONTRAST"]),
            "saturation": os.environ.get("CAMERA_SATURATION", defaults["CAMERA_SATURATION"]),
            "hue": os.environ.get("CAMERA_HUE", defaults["CAMERA_HUE"]),
            "gamma": os.environ.get("CAMERA_GAMMA", defaults["CAMERA_GAMMA"]),
            "gain": os.environ.get("CAMERA_GAIN", defaults["CAMERA_GAIN"]),
            "sharpness": os.environ.get("CAMERA_SHARPNESS", defaults["CAMERA_SHARPNESS"]),
            "backlight_compensation": os.environ.get(
                "CAMERA_BACKLIGHT", defaults["CAMERA_BACKLIGHT"]
            ),
            "power_line_frequency": os.environ.get(
                "CAMERA_POWER_LINE", defaults["CAMERA_POWER_LINE"]
            ),
            "exposure_dynamic_framerate": (
                "1"
                if os.environ.get(
                    "CAMERA_EXPOSURE_DYNAMIC_FRAMERATE",
                    defaults["CAMERA_EXPOSURE_DYNAMIC_FRAMERATE"],
                )
                != "0"
                else "0"
            ),
        }
        auto_exposure = os.environ.get("CAMERA_AUTO_EXPOSURE", "1") != "0"
        controls["auto_exposure"] = "3" if auto_exposure else "1"
        if not auto_exposure:
            controls["exposure_time_absolute"] = os.environ.get("CAMERA_EXPOSURE", "157")
        auto_white_balance = os.environ.get("CAMERA_AUTO_WHITE_BALANCE", "1") != "0"
        controls["white_balance_automatic"] = "1" if auto_white_balance else "0"
        if not auto_white_balance:
            controls["white_balance_temperature"] = os.environ.get("CAMERA_WHITE_BALANCE", "4600")

        with self.camera_control_lock:
            overrides = dict(self.camera_control_overrides)
        if "auto_exposure" in overrides:
            automatic = bool(overrides.pop("auto_exposure"))
            controls["auto_exposure"] = "3" if automatic else "1"
            if automatic:
                controls.pop("exposure_time_absolute", None)
                overrides.pop("exposure_time_absolute", None)
        if "auto_white_balance" in overrides:
            automatic = bool(overrides.pop("auto_white_balance"))
            controls["white_balance_automatic"] = (
                "1" if automatic else "0"
            )
            if automatic:
                controls.pop("white_balance_temperature", None)
                overrides.pop("white_balance_temperature", None)
        if "exposure_dynamic_framerate" in overrides:
            dynamic_framerate = bool(
                overrides.pop("exposure_dynamic_framerate")
            )
            controls["exposure_dynamic_framerate"] = (
                "1" if dynamic_framerate else "0"
            )
        controls.update({key: str(value) for key, value in overrides.items()})
        return controls

    def apply_camera_controls(self) -> tuple[bool, str]:
        controls = self._camera_controls()

        command = [
            "v4l2-ctl",
            f"--device={self.active_camera}",
            "--set-ctrl=" + ",".join(f"{name}={value}" for name, value in controls.items()),
        ]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"deep-live-cam sender: could not apply camera controls: {exc}", file=sys.stderr)
            return False, str(exc)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"v4l2-ctl exited {result.returncode}"
            print(f"deep-live-cam sender: camera controls: {detail}", file=sys.stderr)
            return False, detail
        else:
            self.controls_applied = True
            return True, "applied by the active capture owner"

    def _validated_control_request(self, request: object) -> dict[str, int | bool]:
        if not isinstance(request, dict) or not isinstance(request.get("controls"), dict):
            raise ValueError("expected a controls object")
        validated: dict[str, int | bool] = {}
        for key, raw in request["controls"].items():
            if key in (
                "auto_exposure",
                "auto_white_balance",
                "exposure_dynamic_framerate",
            ):
                if not isinstance(raw, bool):
                    raise ValueError(f"{key} must be true or false")
                validated[key] = raw
                continue
            if key not in self.CONTROL_RANGES:
                raise ValueError(f"unsupported camera control: {key}")
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be an integer")
            value = int(raw)
            minimum, maximum = self.CONTROL_RANGES[key]
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
            validated[key] = value
        return validated

    def _handle_control_connection(self, connection: socket.socket) -> None:
        try:
            payload = connection.recv(65_537)
            if not payload or len(payload) > 65_536:
                raise ValueError("camera configuration request is empty or too large")
            controls = self._validated_control_request(json.loads(payload.decode("utf-8")))
            with self.camera_control_lock:
                # Each manager request is a complete selected-device state.
                # Replacement prevents an inactive manual exposure/WB value
                # from surviving after its automatic mode is re-enabled.
                self.camera_control_overrides = dict(controls)
            succeeded, detail = self.apply_camera_controls()
            response = {"ok": succeeded, "detail": detail, "controls": controls}
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
        while not self.stop_requested:
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(3.0)
                self._handle_control_connection(connection)
        try:
            self.control_socket_path.unlink()
        except FileNotFoundError:
            pass

    def start_control_server(self) -> None:
        self.control_thread = threading.Thread(
            target=self.control_loop,
            name="camera-control-owner",
            daemon=True,
        )
        self.control_thread.start()

    def remote_input_frames(self) -> int | None:
        """Input frame counter reported by the Windows health endpoint.

        Unidirectional SRT does not reliably detect a silently dead peer:
        the local ffmpeg keeps encoding into a socket whose listener is
        gone. The health endpoint is the ground truth for whether frames
        actually arrive.
        """
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(self.health_url, timeout=2.0) as response:
                health = json.load(response)
                if health.get("selected_device_id") != self.device_id:
                    return None
                return int(health["input"]["frames"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def command(self) -> list[str]:
        local_preview = f"udp://127.0.0.1:{self.preview_port}?pkt_size=1316"
        manager_preview = (
            f"udp://127.0.0.1:{self.manager_preview_port}?pkt_size=1316"
        )
        network_source = (
            f"udp://127.0.0.1:{self.network_source_port}?pkt_size=1316"
        )
        native_processor_source = (
            f"udp://127.0.0.1:{self.native_processor_source_port}?pkt_size=1316"
        )
        # Camera capture is deliberately LAN-independent. The four local
        # outputs have one consumer each: the stable-camera mux, native
        # manager, independently reconnecting Windows transport worker, and
        # the local native face processor that returns video to Android.
        preview_slave = f"[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]{local_preview}"
        manager_slave = (
            "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]"
            f"{manager_preview}"
        )
        network_slave = (
            "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]"
            f"{network_source}"
        )
        native_processor_slave = (
            "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]"
            f"{native_processor_source}"
        )
        tee_target = (
            f"{preview_slave}|{manager_slave}|{network_slave}|"
            f"{native_processor_slave}"
        )
        # A 0.5s GOP halves the worst-case keyframe wait for any consumer
        # (Windows processor, local preview taps) that joins or reconnects,
        # trading a little bitrate for faster recovery on a live stream.
        gop = max(1, self.fps // 2)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-fflags",
            "nobuffer",
        ]
        if self.encoder == "h264_vaapi":
            command.extend(["-vaapi_device", self.vaapi_device])
        command.extend([
            "-thread_queue_size",
            "2",
            "-f",
            "v4l2",
            "-input_format",
            self.input_format,
            "-framerate",
            str(self.camera_fps),
            "-video_size",
            f"{self.camera_width}x{self.camera_height}",
            "-i",
            str(self.active_camera),
            "-an",
            "-map",
            "0:v:0",
            "-vf",
            (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease:flags=fast_bilinear,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={self.fps},"
                + ("format=nv12,hwupload" if self.encoder == "h264_vaapi" else "format=yuv420p")
            ),
            "-c:v",
            self.encoder,
        ])
        if self.encoder == "h264_vaapi":
            command.extend([
                "-profile:v", "high", "-rc_mode", "CBR",
                "-async_depth", "1", "-quality", "1",
            ])
        else:
            command.extend([
                "-preset", self.preset, "-tune", "zerolatency",
            ])
        command.extend([
            "-b:v",
            self.bitrate,
            "-maxrate",
            self.bitrate,
            "-bufsize",
            self.bitrate,
            "-g",
            str(gop),
            "-bf",
            "0",
        ])
        if self.encoder == "libx264":
            command.extend([
                "-keyint_min", str(gop), "-sc_threshold", "0",
                "-x264-params", "nal-hrd=cbr:force-cfr=1:repeat-headers=1",
            ])
        command.extend([
            "-flush_packets",
            "1",
            "-bsf:v",
            "dump_extra=freq=keyframe",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "tee",
            tee_target,
        ])
        return command

    def network_command(self) -> list[str]:
        query = srt_query(
            {
                "mode": "caller",
                "transtype": "live",
                "messageapi": 1,
                "latency": self.latency_us,
                "connect_timeout": 3000,
                "timeout": 5_000_000,
                "tlpktdrop": 1,
                "pkt_size": 1316,
            }
        )
        source = (
            f"udp://127.0.0.1:{self.network_source_port}?"
            "fifo_size=1000000&overrun_nonfatal=1"
        )
        target = f"srt://{self.windows_host}:{self.windows_input_port}?{query}"
        return [
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
            "1000000",
            "-probesize",
            "1000000",
            "-f",
            "mpegts",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-flush_packets",
            "1",
            "-f",
            "mpegts",
            target,
        ]

    def network_loop(self) -> None:
        """Reconnect transport without ever recycling physical capture."""
        while not self.stop_requested:
            launched_at = time.monotonic()
            last_remote: int | None = None
            last_advance_at = launched_at
            try:
                self.network_process = subprocess.Popen(
                    self.network_command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )
                self.network_restarts += 1
                while (
                    not self.stop_requested
                    and self.network_process.poll() is None
                ):
                    remote = self.remote_input_frames()
                    now = time.monotonic()
                    if remote is None:
                        last_remote = None
                        last_advance_at = now
                    elif last_remote is None or remote > last_remote:
                        last_remote = remote
                        last_advance_at = now
                        self.network_last_frame_at = now
                    elif (
                        now - launched_at > 8.0
                        and now - last_advance_at > 8.0
                    ):
                        print(
                            "deep-live-cam sender: Windows input is stale; "
                            "recycling only the network worker",
                            file=sys.stderr,
                        )
                        stop_process(self.network_process)
                        break
                    if self.stop_requested:
                        break
                    time.sleep(2.0)
            except OSError as exc:
                print(
                    f"deep-live-cam sender: network worker failed: {exc}",
                    file=sys.stderr,
                )
            finally:
                stop_process(self.network_process)
                self.network_process = None
            if not self.stop_requested:
                time.sleep(self.retry_seconds)

    def supervise_once(self) -> None:
        command = self.command()
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0)
        self.restarts += 1
        launched_at = time.monotonic()
        last_progress = launched_at
        last_state = 0.0
        attempt_frame = 0
        buffer = bytearray()
        selector = selectors.DefaultSelector()
        assert self.process.stdout is not None
        selector.register(self.process.stdout, selectors.EVENT_READ)
        self.notify("connecting", f"ffmpeg pid {self.process.pid}")

        try:
            while not self.stop_requested and self.process.poll() is None:
                now = time.monotonic()
                for key, _events in selector.select(timeout=1.0):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        continue
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw_line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        line = raw_line.decode("utf-8", "replace").strip()
                        if line.startswith("frame="):
                            try:
                                frame = int(line.split("=", 1)[1].strip())
                            except ValueError:
                                continue
                            if frame > attempt_frame:
                                self.frame_count += frame - attempt_frame
                                attempt_frame = frame
                                self.last_frame_at = now
                                last_progress = now

                timeout = self.stall_timeout if self.last_frame_at is not None else self.startup_timeout
                if now - last_progress > timeout:
                    self.notify("stalled", f"no FFmpeg progress for {timeout:.0f}s; recycling")
                    stop_process(self.process)
                    break
                if now - last_state >= 2.0:
                    status = "streaming" if self.last_frame_at and now - self.last_frame_at < 3 else "connecting"
                    self.notify(status, f"ffmpeg pid {self.process.pid}")
                    last_state = now
        finally:
            selector.close()
            return_code = self.process.poll()
            stop_process(self.process)
            self.process = None
            if not self.stop_requested:
                self.notify("reconnecting", f"ffmpeg exited with {return_code}")

    def run(self) -> int:
        install_signal_handlers(self.stop)
        self.start_control_server()
        self.network_thread = threading.Thread(
            target=self.network_loop,
            name="windows-transport",
            daemon=True,
        )
        self.network_thread.start()
        sd_notify("READY=1\nSTATUS=sender supervisor started")
        while not self.stop_requested:
            if not self.wait_for_camera():
                break
            if not self.controls_applied:
                self.apply_camera_controls()
            self.supervise_once()
            if not self.stop_requested:
                time.sleep(self.retry_seconds)
        write_state("sender", self.state("stopped"))
        if self.control_thread is not None:
            self.control_thread.join(timeout=3.0)
        if self.network_thread is not None:
            self.network_thread.join(timeout=5.0)
        return 0


def main() -> int:
    try:
        return Sender().run()
    except (ValueError, RuntimeError) as exc:
        print(f"deep-live-cam sender: {exc}", file=sys.stderr)
        sd_notify(f"STATUS=configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
