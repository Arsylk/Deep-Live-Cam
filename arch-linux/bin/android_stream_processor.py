#!/usr/bin/env python3
"""Process an owner-produced camera stream on Arch and return it to Android.

The process owns only local/network stream endpoints.  It never opens an Arch
or Android camera device.  The intended installed route is:

    Android front-camera owner -> SRT :10001 -> LiveProcessor -> Android :10001

The return worker closes its SRT connection when decoded input becomes stale,
which lets the Android stable-camera mux fall back without changing its
Camera2 device identity.  The Arch webcam owner's loopback feed remains an
explicit fallback input mode.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Running a file below ``arch-linux/bin`` would otherwise put only that
# directory on sys.path.  Import the repository packages by their real,
# unambiguous names; do not rely on the Windows deployment overlay which copies
# ``windows/modules`` into the top-level ``modules`` package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import modules.globals  # noqa: E402
from modules import imread_unicode  # noqa: E402
from modules.face_analyser import get_one_face  # noqa: E402
from modules.pipeline_benchmark import PairedBenchmarkRecorder  # noqa: E402
from windows.modules.live_processor import LiveProcessor  # noqa: E402
from windows.modules.live_stream import (  # noqa: E402
    LatestFrame,
    SrtInput,
    SrtOutput,
    _drain_stderr,
    _record_cadence,
)


PROVIDER_NAMES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}

QUALIFIED_MODEL_STATUSES = {"production", "qualified"}

CONTROL_MAX_BYTES = 256 * 1024
SEMANTIC_INPUTS = {"arch-webcam", "android-front", "android-back"}
MODEL_TARGET = {"swapper_model": "native-256", "swapper_backend": "ncnn"}


def parse_boolean_argument(value: str) -> bool:
    """Parse an explicit true/false CLI value without Python truthiness."""
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")

# Keep this contract in lock-step with the LAN controller on Windows.  Model
# and delivery transforms are deliberately not part of it: the former is a
# processor capability, while the latter belongs to the stable output sink.
PROCESSING_FLOAT_LIMITS = {
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
    # Deprecated compatibility field; accepted but intentionally not used.
    "repair_skin_texture": (0.0, 0.3),
    "repair_camera_detail": (0.0, 4.0),
}
PROCESSING_INTEGER_LIMITS = {
    "detection_interval": (1, 5),
    "tracking_grace_frames": (0, 15),
    "minimum_face_size": (32, 512),
}
PROCESSING_BOOLEAN_FIELDS = {
    "processing_enabled",
    "many_faces",
    "live_mirror",
    "show_fps",
    "enable_interpolation",
    "quality_auto_correct",
    "tracking_enabled",
    "repair_boundary_mask",
}
PROCESSING_CHOICE_FIELDS = {
    "processing_mode": {"face_swap", "passthrough"},
    "processing_off_output": {"passthrough", "black"},
    "quality_mode": {"monitor", "balanced", "strict"},
    "enhancer": {"none", "gfpgan"},
}
PROCESSING_FIELDS = frozenset(
    set(PROCESSING_FLOAT_LIMITS)
    | set(PROCESSING_INTEGER_LIMITS)
    | PROCESSING_BOOLEAN_FIELDS
    | set(PROCESSING_CHOICE_FIELDS)
)

# The USB camera exposes both a bogus IEC958 profile (which captures digital
# zeroes) and its real analogue microphone.  Keep the physical source explicit
# so a desktop/Bluetooth default change cannot silently substitute another mic.
DEFAULT_WEBCAM_AUDIO_SOURCE = (
    "alsa_input.usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_"
    "SN0001-02.analog-stereo"
)


def runtime_directory() -> Path:
    """Return the per-user runtime directory used by the other Arch workers."""
    configured = os.environ.get("XDG_RUNTIME_DIR")
    return Path(configured) if configured else Path(f"/run/user/{os.getuid()}")


def default_control_socket() -> Path:
    return runtime_directory() / "deep-live-cam" / "processor-control.sock"


def discover_native256_manifest() -> Path | None:
    """Find a user-installed native-256 pack without embedding a home path.

    An explicit ``DLC_NATIVE256_MANIFEST`` always wins.  The fallback searches
    only the application's XDG data directory and validates the small manifest
    header; no model is downloaded or opened here.
    """
    explicit = os.environ.get("DLC_NATIVE256_MANIFEST")
    if explicit:
        return Path(explicit).expanduser()
    data_home = Path(
        os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    ).expanduser()
    root = data_home / "deep-live-cam" / "models"
    preferred = root / "dlc_swap256m-development-20260814" / "manifest.json"
    candidates = [preferred, *sorted(root.glob("*/manifest.json"))]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("model_id") == "dlc_swap256m"
            and document.get("input_size") == [256, 256]
        ):
            return candidate
    return None


def semantic_input(value: object) -> str:
    """Normalize the legacy Android transport name to a lens-aware input."""
    selected = str(value or "").strip().lower()
    if selected == "android-srt":
        return "android-front"
    if selected not in SEMANTIC_INPUTS:
        raise ValueError(
            "input must be arch-webcam, android-front, or android-back"
        )
    return selected


def current_processing_config() -> dict[str, Any]:
    """Return the canonical, effective shared processing configuration."""
    enhancer = (
        "gfpgan"
        if bool(modules.globals.fp_ui.get("face_enhancer", False))
        else "none"
    )
    processing_enabled = bool(modules.globals.processing_enabled)
    return {
        "processing_mode": "face_swap" if processing_enabled else "passthrough",
        "opacity": float(modules.globals.opacity),
        "sharpness": float(modules.globals.sharpness),
        "mouth_mask_size": float(modules.globals.mouth_mask_size),
        "interpolation_weight": float(modules.globals.interpolation_weight),
        "many_faces": bool(modules.globals.many_faces),
        "live_mirror": bool(modules.globals.live_mirror),
        "show_fps": bool(modules.globals.show_fps),
        "enable_interpolation": bool(modules.globals.enable_interpolation),
        "processing_enabled": processing_enabled,
        "processing_off_output": str(modules.globals.processing_off_output),
        "quality_mode": str(modules.globals.quality_mode),
        "quality_auto_correct": bool(modules.globals.quality_auto_correct),
        "tracking_enabled": bool(modules.globals.tracking_enabled),
        "detection_interval": int(modules.globals.detection_interval),
        "tracking_smoothing": float(modules.globals.tracking_smoothing),
        "tracking_grace_frames": int(modules.globals.tracking_grace_frames),
        "minimum_detection_score": float(
            modules.globals.minimum_detection_score
        ),
        "minimum_face_size": int(modules.globals.minimum_face_size),
        "color_match_strength": float(modules.globals.color_match_strength),
        "repair_hf_strength": float(modules.globals.repair_hf_strength),
        "repair_checkerboard": float(modules.globals.repair_checkerboard),
        "repair_wavelet": float(modules.globals.repair_wavelet),
        "repair_boundary_mask": bool(modules.globals.repair_boundary_mask),
        "repair_boundary_strength": float(
            modules.globals.repair_boundary_strength
        ),
        "repair_skin_texture": float(modules.globals.repair_skin_texture),
        "repair_camera_detail": float(modules.globals.repair_camera_detail),
        "enhancer": enhancer,
        "source_path": modules.globals.source_path,
    }


def normalize_processing_patch(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a partial shared-config request before mutating globals."""
    unknown = set(values) - PROCESSING_FIELDS
    if unknown:
        raise ValueError(
            "unsupported processing setting(s): " + ", ".join(sorted(unknown))
        )
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if key in PROCESSING_FLOAT_LIMITS:
            if isinstance(value, bool):
                raise ValueError(f"{key} must be a number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
            low, high = PROCESSING_FLOAT_LIMITS[key]
            normalized[key] = max(low, min(high, number))
        elif key in PROCESSING_INTEGER_LIMITS:
            if isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
            number = int(value)
            low, high = PROCESSING_INTEGER_LIMITS[key]
            normalized[key] = max(low, min(high, number))
        elif key in PROCESSING_BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
            normalized[key] = value
        else:
            selected = str(value)
            allowed = PROCESSING_CHOICE_FIELDS[key]
            if selected not in allowed:
                raise ValueError(
                    f"{key} must be one of {', '.join(sorted(allowed))}"
                )
            normalized[key] = selected
    return normalized


def native256_target_status(manifest_path: Path | None) -> dict[str, Any]:
    """Inspect the fixed Arch model target without loading an inference model."""
    status: dict[str, Any] = {
        "model_target": MODEL_TARGET["swapper_model"],
        "backend_target": MODEL_TARGET["swapper_backend"],
        "model_id": None,
        "manifest": str(manifest_path) if manifest_path else None,
        "available": False,
        "quality_status": "unavailable",
        "qualified": False,
        "warning": "native-256 model pack is not installed",
    }
    if manifest_path is None:
        return status
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("manifest root is not an object")
        quality = str(document.get("quality_status", "unknown"))
        status.update(
            {
                "model_id": document.get("model_id"),
                "quality_status": quality,
                "qualified": quality in QUALIFIED_MODEL_STATUSES,
            }
        )
        from modules.native256_ncnn_swapper import native256_ncnn_available

        status["available"] = native256_ncnn_available(
            manifest_path=manifest_path,
            require_qualified=False,
        )
        if status["available"]:
            status["warning"] = (
                None
                if status["qualified"]
                else (
                    f"checkpoint is {quality}/unqualified; transport and "
                    "inference availability do not establish swap quality"
                )
            )
        else:
            status["warning"] = (
                "native-256 manifest, hash-pinned model assets, or the ncnn "
                "bridge is unavailable"
            )
    except Exception as error:
        status["warning"] = f"native-256 target inspection failed: {error}"
    return status


def owner_camera_profile(
    path: Path | None = None,
) -> dict[str, object] | None:
    """Read the capture owner's declared controls without opening the camera."""
    state_path = path or Path(
        os.environ.get(
            "DLC_ARCH_SENDER_STATE", "/run/deep-live-cam/sender.json"
        )
    )
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    controls = document.get("camera_controls")
    if not isinstance(controls, dict):
        return None
    return {
        "profile": document.get("camera_profile"),
        "capture": document.get("capture"),
        "camera_mode": document.get("camera_mode"),
        "controls": dict(controls),
        "source": "capture-owner-state",
    }


def identity_swap_health(model_quality: str, quality: object) -> dict:
    """Describe evidence of a swap without confusing invocation with effect.

    A native-256 development bundle is allowed to run when explicitly chosen,
    but it must never be represented as a verified face swap.  Pixel evidence
    is deliberately conservative and face-core based so an alpha-mask ring is
    not mistaken for an identity change.
    """
    metrics = quality if isinstance(quality, dict) else {}
    face_value = metrics.get("face", {})
    face = face_value if isinstance(face_value, dict) else {}
    checkpoint_status = str(model_quality or "unknown").strip().lower()
    checkpoint_qualified = checkpoint_status in QUALIFIED_MODEL_STATUSES
    processing_active = bool(metrics.get("processing_active", True))
    attempted = bool(metrics.get("swap_applied", face.get("swap_applied", False)))
    face_measurable = bool(face.get("available", False))

    def nonnegative_float(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return round(number, 3) if math.isfinite(number) and number >= 0.0 else None

    core_delta = nonnegative_float(face.get("core_mean_absolute_delta"))
    core_changed = nonnegative_float(face.get("core_changed_pixels_percent"))
    visual_effect = bool(
        attempted
        and face_measurable
        and core_delta is not None
        and core_changed is not None
        and core_delta >= 1.5
        and core_changed >= 10.0
    )

    if not processing_active:
        status = "disabled"
        detail = "face processing is disabled"
    elif not checkpoint_qualified:
        status = "unqualified-checkpoint"
        detail = (
            f"checkpoint quality is {checkpoint_status}; transport may be live "
            "but an identity swap is not verified"
        )
    elif not attempted:
        status = "waiting-for-face"
        detail = "no face-swap invocation has been observed"
    elif not face_measurable:
        status = "effect-unmeasurable"
        detail = "a swap was invoked but no comparable face crop is available"
    elif not visual_effect:
        status = "no-measurable-effect"
        detail = "the detected face core is effectively unchanged"
    else:
        status = "visual-effect-detected"
        detail = (
            "a qualified checkpoint produced a measurable face-core change; "
            "identity similarity is not measured"
        )

    return {
        "status": status,
        "detail": detail,
        "checkpoint_qualified": checkpoint_qualified,
        "attempted": attempted,
        "face_measurable": face_measurable,
        "core_mean_absolute_delta": core_delta,
        "core_changed_pixels_percent": core_changed,
        "visual_effect_confirmed": bool(checkpoint_qualified and visual_effect),
        # Pixel deltas can establish an effect, not that the requested identity
        # is present.  A future embedding comparison may promote this field.
        "identity_change_verified": False,
    }


def active_source_image() -> str | None:
    """Resolve the manager's active, content-addressed source picture."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    directory = base / "deep-live-cam" / "source-history"
    try:
        document = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    active_id = document.get("active_id")
    items = document.get("items")
    if not isinstance(active_id, str) or not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get("id") != active_id:
            continue
        cache_name = item.get("cache_name")
        if not isinstance(cache_name, str) or Path(cache_name).name != cache_name:
            return None
        candidate = directory / cache_name
        return str(candidate.resolve()) if candidate.is_file() else None
    return None


def _has_nvidia_device() -> bool:
    """Detect NVIDIA hardware without trusting an unusable ORT provider list."""
    if Path("/proc/driver/nvidia/version").is_file():
        return True
    drm = Path("/sys/class/drm")
    for vendor in drm.glob("card*/device/vendor"):
        try:
            if vendor.read_text(encoding="ascii").strip().lower() == "0x10de":
                return True
        except OSError:
            continue
    return False


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def resolve_execution_providers(
    requested: str,
    available: Iterable[str],
    *,
    system: str | None = None,
    has_nvidia: bool | None = None,
) -> list[str]:
    """Return an ordered, actually available ORT provider list.

    Some CUDA-flavoured ORT wheels expose CUDA and TensorRT providers even on
    hosts with no NVIDIA GPU.  Selecting either on the RX 570 causes a slow
    initialization failure before ORT falls back.  Linux auto selection only
    chooses CUDA when matching hardware is present; ncnn/Vulkan remains
    independent and is selected through ``--swapper-backend``.
    """
    providers = list(dict.fromkeys(str(item) for item in available))
    operating_system = system or platform.system()
    nvidia = _has_nvidia_device() if has_nvidia is None else has_nvidia

    if requested != "auto":
        name = PROVIDER_NAMES[requested]
        if name not in providers:
            raise ValueError(
                f"requested ONNX provider {name} is unavailable; installed: "
                f"{providers or ['none']}"
            )
        selected = [name]
    elif operating_system == "Linux":
        preferred = ["ROCMExecutionProvider", "OpenVINOExecutionProvider"]
        if nvidia:
            preferred.insert(0, "CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")
        selected = [name for name in preferred if name in providers][:1]
    else:
        preferred = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected = [name for name in preferred if name in providers][:1]

    if not selected:
        raise ValueError(
            "no supported ONNX Runtime execution provider is available; "
            f"installed: {providers or ['none']}"
        )
    if "CPUExecutionProvider" in providers and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected


def _validated_ip(value: str, label: str) -> str:
    try:
        return str(ipaddress.ip_address(value))
    except ValueError as error:
        raise ValueError(f"{label} must be a literal IPv4 or IPv6 address") from error


def _validated_port(value: int, label: str) -> int:
    if not 1 <= value <= 65_535:
        raise ValueError(f"{label} must be between 1 and 65535")
    return value


def srt_listener_url(host: str, port: int, latency_us: int) -> str:
    host = _validated_ip(host, "input host")
    port = _validated_port(port, "input port")
    return (
        f"srt://{host}:{port}?mode=listener&transtype=live&messageapi=1&"
        f"pkt_size=1316&latency={latency_us}&tlpktdrop=1"
    )


def srt_caller_url(host: str, port: int, latency_us: int) -> str:
    host = _validated_ip(host, "Android host")
    port = _validated_port(port, "return port")
    return (
        f"srt://{host}:{port}?mode=caller&transtype=live&messageapi=1&"
        f"pkt_size=1316&latency={latency_us}&tlpktdrop=1&"
        "connect_timeout=3000&timeout=5000000"
    )


def local_preview_url(host: str, port: int) -> str:
    """Build a loopback-only MPEG-TS preview destination.

    The preview is deliberately not a LAN broadcast.  It is a passive copy of
    the return encoder output for the local manager and must never become a
    second network-facing camera endpoint.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("preview host must be a literal loopback IPv4 address") from error
    if address.version != 4 or not address.is_loopback:
        raise ValueError("preview host must be a literal loopback IPv4 address")
    port = _validated_port(port, "preview port")
    return f"udp://{address}:{port}?pkt_size=1316"


def local_owner_feed_url(port: int) -> str:
    """Build the private webcam-owner feed consumed by the local processor."""
    port = _validated_port(port, "webcam source port")
    return (
        f"udp://127.0.0.1:{port}?"
        "fifo_size=1000000&overrun_nonfatal=1"
    )


def tee_mux_target(primary_url: str, *secondary_urls: str) -> str:
    """Return one encoded fan-out for the phone and independent local taps."""
    slaves = [
        "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]" + primary_url
    ]
    slaves.extend(
        "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]" + url
        for url in secondary_urls
        if url
    )
    return "|".join(slaves)


def ffmpeg_has_srt(ffmpeg: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-protocols"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    protocols = {line.strip() for line in result.stdout.splitlines()}
    return result.returncode == 0 and "srt" in protocols


class LinuxSrtOutput(SrtOutput):
    """SRT output with a real Linux VAAPI path and software fallback."""

    def __init__(
        self,
        *args,
        vaapi_device: str = "/dev/dri/renderD128",
        preview_url: str | None = None,
        system_camera_url: str | None = None,
        audio_source: str | None = None,
        audio_bitrate: str = "128k",
        audio_delay_ms: int = 220,
        active: bool = True,
        **kwargs,
    ):
        # Delivery ownership is independent from the worker lifetime.  The
        # worker remains hot while inactive, but it is forbidden from opening
        # the phone/local return transport until ownership is granted again.
        self._delivery_lock = threading.RLock()
        self._delivery_changed = threading.Event()
        self._active = bool(active)
        self.vaapi_device = vaapi_device
        self.preview_url = preview_url
        self.system_camera_url = system_camera_url
        self.audio_source = audio_source
        self.audio_bitrate = audio_bitrate
        self.audio_delay_ms = audio_delay_ms
        super().__init__(*args, **kwargs)

    @property
    def active(self) -> bool:
        with self._delivery_lock:
            return self._active

    @property
    def transport_open(self) -> bool:
        with self._delivery_lock:
            process = self.process
            return process is not None and process.poll() is None

    def set_active(self, active: bool) -> bool:
        """Hot-toggle ownership of every return sink without stopping work."""
        if not isinstance(active, bool):
            raise TypeError("active must be a boolean")
        with self._delivery_lock:
            changed = self._active != active
            self._active = active
            if not active:
                # Returning from this call guarantees that FFmpeg no longer
                # owns the SRT/preview/system-camera delivery endpoints.
                super()._terminate()
                self.frames.clear()
            self._delivery_changed.set()
            return changed

    def _probe_vaapi(self) -> bool:
        if platform.system() != "Linux" or not Path(self.vaapi_device).exists():
            return False
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-vaapi_device",
            self.vaapi_device,
            "-f",
            "lavfi",
            "-i",
            f"color=size={self.width}x{self.height}:rate={self.fps}",
            "-vf",
            "format=nv12,hwupload",
            "-frames:v",
            "1",
            "-c:v",
            "h264_vaapi",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _select_encoder(self) -> str:
        if platform.system() == "Linux":
            if self._probe_vaapi():
                print(
                    f"[{self.label}] encoder selected: h264_vaapi "
                    f"({self.vaapi_device})",
                    flush=True,
                )
                return "h264_vaapi"
            print(
                f"[{self.label}] VAAPI unavailable; using libx264",
                flush=True,
            )
            return "libx264"
        return super()._select_encoder()

    def _ffmpeg_command(self) -> list[str]:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
        ]
        if self.encoder == "h264_vaapi":
            command.extend([
                "-vaapi_device",
                self.vaapi_device,
            ])
        command.extend([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
        ])
        if self.audio_source:
            command.extend([
                "-thread_queue_size",
                "512",
                "-f",
                "pulse",
                "-i",
                self.audio_source,
            ])
        if self.encoder == "h264_vaapi":
            command.extend([
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-profile:v",
                "high",
                "-rc_mode",
                "CBR",
                "-async_depth",
                "1",
                "-quality",
                "1",
                # Force an IDR every GOP so a late-joining consumer (the Arch
                # receiver reading the loopback copy) can recover quickly.
                # Without this VAAPI emits a single IDR at startup and only
                # P-frames afterwards, so any consumer that connects after the
                # first frame never sees SPS/PPS and fails with
                # "non-existing PPS 0 referenced", delivering zero frames.
                "-idr_interval",
                "0",
                "-b:v",
                self.bitrate,
                "-maxrate",
                self.bitrate,
                "-bufsize",
                self.bitrate,
            ])
        else:
            command.extend([
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-b:v",
                self.bitrate,
                "-maxrate",
                self.bitrate,
                "-bufsize",
                self.bitrate,
                "-x264-params",
                "scenecut=0:force-cfr=1",
                "-pix_fmt",
                "yuv420p",
            ])
        command.extend([
            "-g",
            str(max(1, self.fps // 2)),
            "-bf",
            "0",
            # Guarantee a real keyframe about twice a second regardless of
            # encoder rate-control heuristics.  A 0.5s interval halves the
            # worst-case wait for a late-joining consumer (the receiver's local
            # decoder) to lock on, so a reconnect recovers faster; combined
            # with dump_extra below it also makes SPS/PPS recur in-band.
            "-force_key_frames",
            "expr:gte(t,n_forced*0.5)",
            "-bsf:v",
            "dump_extra=freq=keyframe",
        ])
        if self.audio_source:
            command.extend([
                "-c:a",
                "aac",
                "-profile:a",
                "aac_low",
                "-b:a",
                self.audio_bitrate,
                "-ar",
                "48000",
                "-ac",
                "1",
                "-af",
                (
                    "aresample=async=1:first_pts=0,"
                    f"adelay={self.audio_delay_ms}:all=1"
                ),
            ])
        secondary_urls = tuple(
            url for url in (self.preview_url, self.system_camera_url) if url
        )
        if secondary_urls:
            command.extend([
                "-map",
                "0:v:0",
            ])
            if self.audio_source:
                command.extend(["-map", "1:a:0"])
            else:
                command.append("-an")
            command.extend([
                "-flush_packets",
                "1",
                "-f",
                "tee",
                "-use_fifo",
                "1",
                "-fifo_options",
                (
                    "attempt_recovery=1:recover_any_error=1:"
                    "recovery_wait_time=1:restart_with_keyframe=1:"
                    "drop_pkts_on_overflow=1"
                ),
                tee_mux_target(self.url, *secondary_urls),
            ])
        else:
            command.extend([
                "-map",
                "0:v:0",
            ])
            if self.audio_source:
                command.extend(["-map", "1:a:0"])
            else:
                command.append("-an")
            command.extend([
                "-flush_packets",
                "1",
                "-f",
                "mpegts",
                "-mpegts_flags",
                "resend_headers",
                self.url,
            ])
        return command

    def _start_ffmpeg(self) -> subprocess.Popen:
        with self._delivery_lock:
            if not self._active:
                raise _DeliveryInactive
            command = self._ffmpeg_command()
            print(f"[{self.label}] opening {self.url}", flush=True)
            if self.preview_url:
                print(
                    f"[{self.label}] exact encoded preview tee: {self.preview_url}",
                    flush=True,
                )
            if self.system_camera_url:
                print(
                    f"[{self.label}] system-camera copy: {self.system_camera_url}",
                    flush=True,
                )
            if self.audio_source:
                print(
                    f"[{self.label}] webcam microphone: {self.audio_source} -> "
                    f"AAC-LC mono/48k ({self.audio_bitrate}, delay "
                    f"{self.audio_delay_ms} ms)",
                    flush=True,
                )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            threading.Thread(
                target=_drain_stderr,
                args=(process, f"{self.label}-ffmpeg"),
                daemon=True,
            ).start()
            return process

    def run(self) -> None:
        """Run the normal CFR sender behind a hot delivery-ownership gate."""
        latest = None
        latest_at = 0.0
        period = 1.0 / max(1, self.fps)
        deadline = time.perf_counter()
        while not self.stop_event.is_set():
            if not self.active:
                latest = None
                latest_at = 0.0
                self._terminate()
                self.frames.clear()
                self._delivery_changed.clear()
                if not self.active:
                    self._delivery_changed.wait(0.1)
                continue

            fresh_frame = False
            try:
                while True:
                    latest = self.frames.get(
                        timeout=0.01 if latest is None else 0.0
                    )
                    latest_at = time.monotonic()
                    self.source_frames += 1
                    fresh_frame = True
            except queue.Empty:
                pass
            if latest is None:
                if (
                    self.process is not None
                    and latest_at
                    and time.monotonic() - latest_at > self.stale_seconds
                ):
                    self._terminate()
                self.stop_event.wait(0.02)
                continue
            if time.monotonic() - latest_at > self.stale_seconds:
                latest = None
                self._terminate()
                continue
            if not isinstance(latest, np.ndarray):
                print(f"[{self.label}] dropped invalid frame payload", flush=True)
                latest = None
                continue
            if self.process is None or self.process.poll() is not None:
                self._terminate()
                try:
                    # Keep the spawn *and* assignment inside the ownership
                    # lock.  Otherwise deactivation could observe ``None``
                    # between return and assignment and leave a newly spawned
                    # sender running after the control call completed.
                    with self._delivery_lock:
                        if not self._active:
                            raise _DeliveryInactive
                        self.process = self._start_ffmpeg()
                    self.connections += 1
                    self._last_frame_monotonic = 0.0
                except _DeliveryInactive:
                    continue
                except Exception as exc:
                    print(f"[{self.label}] {exc}", flush=True)
                    self.stop_event.wait(1.0)
                    continue
            now = time.perf_counter()
            if deadline < now:
                deadline = now
            delay = deadline - now
            if delay > 0:
                self.stop_event.wait(delay)
            step = period
            if self.jitter:
                step = period * (1.0 + float(self._rng.uniform(-0.02, 0.02)))
            try:
                process = self.process
                if process is None or process.stdin is None:
                    raise BrokenPipeError("FFmpeg input pipe closed")
                payload = self._dither(latest) if self.dither else latest
                if not payload.flags.c_contiguous:
                    payload = np.ascontiguousarray(payload)
                remaining = memoryview(payload).cast("B")
                while remaining:
                    written = process.stdin.write(remaining)
                    if not written:
                        raise BrokenPipeError("FFmpeg input pipe closed")
                    remaining = remaining[written:]
                self.sent += 1
                if not fresh_frame:
                    self.repeated_frames += 1
                self.last_frame_at = time.time()
                _record_cadence(self, time.monotonic(), self.fps)
                deadline += step
            except (BrokenPipeError, OSError) as exc:
                if self.active:
                    print(
                        f"[{self.label}] receiver unavailable: {exc}",
                        flush=True,
                    )
                    self.disconnects += 1
                self._terminate()

    def _terminate(self) -> None:
        with self._delivery_lock:
            super()._terminate()


class _DeliveryInactive(RuntimeError):
    """Internal non-error used when ownership changes during sender startup."""


class AndroidStreamService:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        input_factory=SrtInput,
        output_factory=LinuxSrtOutput,
        processor_factory=LiveProcessor,
    ) -> None:
        self.args = args
        self.input_factory = input_factory
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self.input_frames = LatestFrame()
        self.output_frames = LatestFrame()
        self.input_lock = threading.RLock()
        self.control_lock = threading.RLock()
        self._input_stop_event = threading.Event()
        self._input_switching = False
        self._workers_started = False
        self.desired_active = bool(getattr(args, "active", True))
        self.input_generation = 0
        self.input_restarts = 0
        self._input_counters = {
            "received": 0,
            "connections": 0,
            "disconnects": 0,
        }
        self.effective_input = semantic_input(
            getattr(args, "semantic_input", getattr(args, "input_mode", None))
        )
        self.desired_input = self.effective_input
        self.input = input_factory(
            args.input_url,
            args.width,
            args.height,
            self.input_frames,
            self._input_stop_event,
            label=args.input_label,
            expected_fps=args.fps,
            route_token=self.input_generation,
        )
        output_kwargs = {
            "label": "android-return-slot-0",
            "stale_seconds": args.stale_seconds,
            "encoder": None if args.encoder == "auto" else args.encoder,
        }
        if output_factory is LinuxSrtOutput:
            output_kwargs["vaapi_device"] = args.vaapi_device
            output_kwargs["preview_url"] = args.preview_url
            output_kwargs["system_camera_url"] = args.system_camera_url
            output_kwargs["audio_source"] = (
                args.return_audio_source if args.return_audio else None
            )
            output_kwargs["audio_bitrate"] = args.return_audio_bitrate
            output_kwargs["audio_delay_ms"] = args.return_audio_delay_ms
            output_kwargs["active"] = self.desired_active
        self.output = output_factory(
            args.output_url,
            args.width,
            args.height,
            args.fps,
            args.bitrate,
            self.output_frames,
            self.stop_event,
            **output_kwargs,
        )
        output_active_setter = getattr(self.output, "set_active", None)
        if callable(output_active_setter):
            output_active_setter(self.desired_active)
        else:
            # Test/embedding adapters may not implement transport ownership,
            # but canonical state must still remain representable.
            self.output.active = self.desired_active
        self.processor = processor_factory(
            self.input_frames,
            self.output_frames,
            self.stop_event,
            args.fps,
            args.state_dir,
        )
        self.benchmark = PairedBenchmarkRecorder(
            args.state_dir,
            context_supplier=self._benchmark_context,
            start_callback=getattr(
                self.processor, "reset_benchmark_window", lambda: None
            ),
        )
        self.processor.frame_observer = self.benchmark.observe
        self.control_socket_path = Path(
            getattr(args, "control_socket", default_control_socket())
        ).expanduser()
        manifest_value = getattr(args, "native256_manifest", None)
        self.native256_manifest = (
            Path(manifest_value).expanduser() if manifest_value else None
        )
        self.native_target = native256_target_status(self.native256_manifest)
        self.desired_processing = current_processing_config()
        self.desired_revision = 0
        self.control_server: socket.socket | None = None
        self.control_thread: threading.Thread | None = None
        self.control_last_error: str | None = None
        self.control_updated_at = time.time()
        self._closed = False

    def _route_for_input(self, selected: str) -> dict[str, str]:
        selected = semantic_input(selected)
        if selected == "arch-webcam":
            return {
                "url": local_owner_feed_url(
                    int(getattr(self.args, "webcam_source_port", 11_005))
                ),
                "label": "arch-webcam-owner-feed",
                "transport": "arch-webcam",
                "route_id": "arch-webcam-processed-to-android",
            }
        return {
            "url": srt_listener_url(
                str(getattr(self.args, "input_host", "0.0.0.0")),
                int(getattr(self.args, "input_port", 10_001)),
                int(getattr(self.args, "latency_us", 100_000)),
            ),
            # Front/back selection is owned by the already-open Camera2 owner.
            # Both lenses intentionally share one stable transport decoder.
            "label": "android-slot-0",
            "transport": "android-srt",
            "route_id": f"{selected}-processed-to-android",
        }

    def _new_input_worker(
        self, route: Mapping[str, str], stop_event: threading.Event
    ) -> Any:
        return self.input_factory(
            route["url"],
            self.args.width,
            self.args.height,
            self.input_frames,
            stop_event,
            label=route["label"],
            expected_fps=self.args.fps,
            route_token=self.input_generation,
        )

    def _remember_input_counters(self, worker: object) -> None:
        for key in self._input_counters:
            self._input_counters[key] += int(getattr(worker, key, 0) or 0)

    def _switch_input(self, requested: str) -> bool:
        """Change only the decoder when the selected transport changes.

        The output encoder, processor object, virtual-camera receiver, and
        Android Camera2 owner are never touched here.  Front/back use the same
        phone transport, so that semantic change only resets tracking; the GUI
        separately asks the persistent Camera2 owner to change lens.
        """
        selected = semantic_input(requested)
        route = self._route_for_input(selected)
        with self.input_lock:
            self.desired_input = selected
            old_semantic = self.effective_input
            current_url = str(getattr(self.input, "url", ""))
            if current_url == route["url"]:
                self.effective_input = selected
                self.args.input_mode = route["transport"]
                self.args.input_url = route["url"]
                self.args.input_label = route["label"]
                self.args.route_id = route["route_id"]
                if selected != old_semantic:
                    # Change the token read by SrtInput.  LiveProcessor sees it
                    # with the next frame and resets tracking on its own thread.
                    self.input_generation += 1
                    if hasattr(self.input, "route_token"):
                        self.input.route_token = self.input_generation
                return False

            self._input_switching = True
            old_worker = self.input
            old_stop = self._input_stop_event
            was_started = self._workers_started
            try:
                old_stop.set()
                old_worker.close()
                if old_worker.is_alive():
                    old_worker.join(timeout=3.0)
                self._remember_input_counters(old_worker)
                self.input_frames.clear()
                self.input_generation += 1
                replacement_stop = threading.Event()
                replacement = self._new_input_worker(route, replacement_stop)
                self.input = replacement
                self._input_stop_event = replacement_stop
                self.effective_input = selected
                self.args.input_mode = route["transport"]
                self.args.input_url = route["url"]
                self.args.input_label = route["label"]
                self.args.route_id = route["route_id"]
                self.input_restarts += 1
                if was_started:
                    replacement.start()
                return True
            finally:
                self._input_switching = False

    def _apply_processing(self, patch: Mapping[str, Any]) -> None:
        values = dict(patch)
        requested_mode = values.get("processing_mode")
        if requested_mode is not None:
            values["processing_enabled"] = requested_mode == "face_swap"
            values["processing_off_output"] = "passthrough"
        elif "processing_enabled" in values:
            values["processing_mode"] = (
                "face_swap" if values["processing_enabled"] else "passthrough"
            )

        for key in PROCESSING_FLOAT_LIMITS | PROCESSING_INTEGER_LIMITS:
            if key in values:
                setattr(modules.globals, key, values[key])
        for key in PROCESSING_BOOLEAN_FIELDS:
            if key in values:
                setattr(modules.globals, key, values[key])
        for key in ("processing_off_output", "quality_mode"):
            if key in values:
                setattr(modules.globals, key, values[key])
        if requested_mode is not None:
            modules.globals.processing_enabled = requested_mode == "face_swap"
            modules.globals.processing_off_output = "passthrough"
        modules.globals.mouth_mask = modules.globals.mouth_mask_size > 0

        if "enhancer" in values:
            requested = values["enhancer"]
            model = REPOSITORY_ROOT / "models" / "gfpgan-1024.onnx"
            available = requested == "none" or model.is_file()
            enabled = requested == "gfpgan" and available
            modules.globals.fp_ui["face_enhancer"] = enabled
            processors = [
                name
                for name in modules.globals.frame_processors
                if name != "face_enhancer"
            ]
            if enabled:
                processors.append("face_enhancer")
            modules.globals.frame_processors = processors

        self.desired_processing.update(values)

    def _effective_active(self) -> bool:
        return bool(getattr(self.output, "active", self.desired_active))

    def _set_active(self, active: bool) -> None:
        setter = getattr(self.output, "set_active", None)
        if callable(setter):
            setter(active)
        else:
            self.output.active = active
        self.desired_active = active

    def _apply_model_target(self) -> None:
        # Re-inspect on demand so installing a bundle does not require a
        # service restart.  Discovery remains XDG/env based and never reaches
        # the network.
        if self.native256_manifest is None:
            self.native256_manifest = discover_native256_manifest()
        if (
            not self.native_target.get("available")
            or self.native_target.get("manifest")
            != (
                str(self.native256_manifest)
                if self.native256_manifest is not None
                else None
            )
        ):
            self.native_target = native256_target_status(
                self.native256_manifest
            )
        if not self.native_target["available"]:
            raise ValueError(str(self.native_target["warning"]))
        assert self.native256_manifest is not None
        os.environ["DLC_NATIVE256_MANIFEST"] = str(self.native256_manifest)
        modules.globals.swapper_model = MODEL_TARGET["swapper_model"]
        modules.globals.swapper_backend = MODEL_TARGET["swapper_backend"]
        self.args.swapper_model = MODEL_TARGET["swapper_model"]
        self.args.swapper_backend = MODEL_TARGET["swapper_backend"]
        self.args.model_quality = self.native_target["quality_status"]

    def control_snapshot(self) -> dict[str, Any]:
        with self.input_lock:
            effective_input = self.effective_input
            desired_input = self.desired_input
            input_generation = self.input_generation
            input_restarts = self.input_restarts
        effective_processing = current_processing_config()
        desired_processing = dict(self.desired_processing)
        processing_in_sync = all(
            effective_processing.get(key) == value
            for key, value in desired_processing.items()
        )
        active_model = str(modules.globals.active_swapper_model)
        active_backend = str(modules.globals.active_swapper_backend)
        requested_model = str(modules.globals.swapper_model)
        requested_backend = str(modules.globals.swapper_backend)
        expected_model_id = self.native_target.get("model_id")
        model_configured = bool(
            requested_model == MODEL_TARGET["swapper_model"]
            and requested_backend == MODEL_TARGET["swapper_backend"]
        )
        model_ready = bool(
            model_configured
            and active_backend == MODEL_TARGET["swapper_backend"]
            and active_model in {"native-256", expected_model_id}
        )
        effective_active = self._effective_active()
        active_in_sync = self.desired_active == effective_active
        return {
            "ok": True,
            "revision": self.desired_revision,
            "updated_at": self.control_updated_at,
            "desired": {
                "active": self.desired_active,
                "input": desired_input,
                "processing": desired_processing,
                "model": dict(MODEL_TARGET),
            },
            "effective": {
                "active": effective_active,
                "input": effective_input,
                "processing": effective_processing,
                "model": {
                    "swapper_model": requested_model,
                    "swapper_backend": requested_backend,
                    "active_swapper_model": active_model,
                    "active_swapper_backend": active_backend,
                    "configured": model_configured,
                    "ready": model_ready,
                },
            },
            "in_sync": bool(
                active_in_sync
                and desired_input == effective_input
                and processing_in_sync
                and model_configured
            ),
            "sync": {
                "active": active_in_sync,
                "input": desired_input == effective_input,
                "processing": processing_in_sync,
                "model_configuration": model_configured,
                "model_runtime_ready": model_ready,
            },
            "input_generation": input_generation,
            "input_restarts": input_restarts,
            "input_switching": self._input_switching,
            "model_target": dict(self.native_target),
            "capabilities": {
                "inputs": sorted(SEMANTIC_INPUTS),
                "processing_fields": sorted(PROCESSING_FIELDS),
                "enhancers": [
                    "none",
                    *(
                        ["gfpgan"]
                        if (
                            REPOSITORY_ROOT / "models" / "gfpgan-1024.onnx"
                        ).is_file()
                        else []
                    ),
                ],
                "source_path": True,
                "active": True,
                "hot_delivery_ownership": True,
                "inactive_keeps_workers": True,
                "model_fixed": True,
                "hot_input_decoder": True,
                "owns_camera_device": False,
                "restarts_output_on_input_change": False,
            },
            "socket": str(self.control_socket_path),
            "last_error": self.control_last_error,
        }

    def apply_control_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("control request must be a JSON object")
        operation = str(request.get("op", "set"))
        if operation not in {"get", "set"}:
            raise ValueError("op must be get or set")
        if operation == "get":
            return self.control_snapshot()

        nested = request.get("processing", request.get("settings", {}))
        if nested is None:
            nested = {}
        if not isinstance(nested, Mapping):
            raise ValueError("processing must be a JSON object")
        processing_request = dict(nested)
        for field in PROCESSING_FIELDS:
            if field in request:
                processing_request[field] = request[field]
        nested_model = processing_request.pop("swapper_model", None)
        nested_backend = processing_request.pop("swapper_backend", None)
        source_value = request.get(
            "source_path", processing_request.pop("source_path", None)
        )
        processing_patch = normalize_processing_patch(processing_request)

        active_present = "active" in request
        active_requested = request.get("active")
        if active_present and not isinstance(active_requested, bool):
            raise ValueError("active must be a boolean")

        input_requested = request.get("input")
        selected_input = (
            semantic_input(input_requested) if input_requested is not None else None
        )

        model_value = request.get("swapper_model", nested_model)
        backend_value = request.get("swapper_backend", nested_backend)
        target_value = request.get("model_target")
        if isinstance(target_value, Mapping):
            model_value = target_value.get("swapper_model", model_value)
            backend_value = target_value.get("swapper_backend", backend_value)
        elif target_value is not None:
            model_value = target_value
        change_model = model_value is not None or backend_value is not None
        if model_value is not None and str(model_value) != "native-256":
            raise ValueError("the Arch processor model is fixed to native-256")
        if backend_value is not None and str(backend_value) != "ncnn":
            raise ValueError("the Arch processor backend is fixed to ncnn/Vulkan")

        source_path: str | None | object = source_value
        if source_value is not None:
            candidate = Path(str(source_value)).expanduser().resolve()
            if not candidate.is_file():
                raise ValueError(f"source image does not exist: {candidate}")
            source_path = str(candidate)

        revision_value = request.get("revision")
        if revision_value is not None:
            if isinstance(revision_value, bool):
                raise ValueError("revision must be a non-negative integer")
            revision = int(revision_value)
            if revision < 0:
                raise ValueError("revision must be a non-negative integer")
        else:
            revision = self.desired_revision + 1

        with self.control_lock:
            if change_model:
                # Validate the complete hash-pinned pack before replacing the
                # current model request.  Failure preserves the working model.
                if self.native256_manifest is None:
                    self.native256_manifest = discover_native256_manifest()
                self.native_target = native256_target_status(
                    self.native256_manifest
                )
                if not self.native_target["available"]:
                    raise ValueError(str(self.native_target["warning"]))
            if selected_input is not None:
                self._switch_input(selected_input)
            self._apply_processing(processing_patch)
            if source_value is not None:
                assert isinstance(source_path, str)
                modules.globals.source_path = source_path
                self.desired_processing["source_path"] = source_path
            if change_model:
                self._apply_model_target()
            if active_present:
                assert isinstance(active_requested, bool)
                self._set_active(active_requested)
            self.desired_revision = revision
            self.control_updated_at = time.time()
            self.control_last_error = None
            return self.control_snapshot()

    def _handle_control_connection(self, connection: socket.socket) -> None:
        try:
            chunks = bytearray()
            while len(chunks) <= CONTROL_MAX_BYTES:
                chunk = connection.recv(min(65_536, CONTROL_MAX_BYTES + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
                if b"\n" in chunk:
                    break
            if not chunks:
                raise ValueError("control request is empty")
            if len(chunks) > CONTROL_MAX_BYTES:
                raise ValueError("control request is too large")
            payload = bytes(chunks).split(b"\n", 1)[0]
            request = json.loads(payload.decode("utf-8"))
            response = self.apply_control_request(request)
        except Exception as error:
            self.control_last_error = f"{type(error).__name__}: {error}"
            response = self.control_snapshot()
            response.update({"ok": False, "error": str(error)})
        try:
            connection.sendall(
                json.dumps(response, sort_keys=True).encode("utf-8") + b"\n"
            )
        except OSError:
            pass

    def _control_loop(self) -> None:
        server = self.control_server
        assert server is not None
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
        self.control_socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.control_socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.control_socket_path))
            os.chmod(self.control_socket_path, 0o666)
            server.listen(8)
            server.settimeout(1.0)
        except Exception:
            server.close()
            raise
        self.control_server = server
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="arch-processor-control",
            daemon=True,
        )
        self.control_thread.start()

    def _model_artifacts(self) -> list[str]:
        if modules.globals.swapper_model != "native-256":
            model_root = REPOSITORY_ROOT / "models" / "ncnn"
            return [
                str(model_root / "inswapper_128.ncnn.param"),
                str(model_root / "inswapper_128.ncnn.bin"),
                str(model_root / "inswapper_128_emap.npy"),
            ]
        manifest = self.native256_manifest
        artifacts = [
            str(
                Path(
                    os.environ.get(
                        "DLC_NCNN_LIBRARY",
                        REPOSITORY_ROOT
                        / "arch-linux"
                        / "ncnn"
                        / "libdeep_live_cam_ncnn.so",
                    )
                )
            )
        ]
        if manifest is None:
            return artifacts
        artifacts.insert(0, str(manifest))
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            relative_files = [
                document["identity_map"]["file"],
                document["identity_conditioner"]["file"],
                document["swapper"]["file"],
                document["ncnn"]["identity_conditioner"]["param_file"],
                document["ncnn"]["identity_conditioner"]["bin_file"],
                document["ncnn"]["swapper"]["param_file"],
                document["ncnn"]["swapper"]["bin_file"],
            ]
            artifacts.extend(str(manifest.parent / name) for name in relative_files)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            # Target inspection already exposes the detailed failure.  Keep the
            # manifest/library in benchmark provenance even if it changed.
            pass
        return artifacts

    def _benchmark_context(self) -> dict:
        """Freeze the effective pipeline contract at capture start."""
        source = str(modules.globals.source_path or "")
        context = {
            "schema_version": "1.0",
            "pipeline": "arch-live-processor",
            "route": getattr(self.args, "route_id", None),
            "input_mode": self.effective_input,
            "input_url": getattr(self.args, "input_url", None),
            "output_url": getattr(self.args, "output_url", None),
            "system_camera_url": getattr(
                self.args, "system_camera_url", None
            ),
            "resolution": [
                int(getattr(self.args, "width", 0)),
                int(getattr(self.args, "height", 0)),
            ],
            "delivery_fps": int(getattr(self.args, "fps", 0)),
            "model": modules.globals.swapper_model,
            "backend": modules.globals.swapper_backend,
            "model_quality": getattr(self.args, "model_quality", None),
            "execution_providers": list(getattr(self.args, "providers", [])),
            "quality_mode": modules.globals.quality_mode,
            "detection_interval": modules.globals.detection_interval,
            "tracking_grace_frames": modules.globals.tracking_grace_frames,
            "minimum_detection_score": modules.globals.minimum_detection_score,
            "source_artifact": source or None,
            "model_artifacts": self._model_artifacts(),
            "code_artifacts": [
                str(Path(__file__).resolve()),
                str(REPOSITORY_ROOT / "windows" / "modules" / "live_processor.py"),
                str(REPOSITORY_ROOT / "modules" / "quality_pipeline.py"),
                str(REPOSITORY_ROOT / "modules" / "face_tracking.py"),
                str(REPOSITORY_ROOT / "modules" / "ncnn_swapper.py"),
                str(
                    REPOSITORY_ROOT
                    / "modules"
                    / "processors"
                    / "frame"
                    / "face_swapper.py"
                ),
            ],
            "repository_root": str(REPOSITORY_ROOT),
            "host": platform.node(),
            "platform": platform.platform(),
            "opens_camera_device": False,
        }
        if self.effective_input == "arch-webcam":
            context["camera_profile"] = owner_camera_profile()
        return context

    def snapshot(self) -> dict:
        now = time.time()

        def age(timestamp: float) -> float | None:
            return None if not timestamp else round(max(0.0, now - timestamp), 3)

        quality = {}
        processor_quality = getattr(self.processor, "quality", None)
        quality_snapshot = getattr(processor_quality, "snapshot", None)
        if callable(quality_snapshot):
            candidate = quality_snapshot()
            if isinstance(candidate, dict):
                quality = candidate
        model_quality = str(getattr(self.args, "model_quality", "unknown"))
        effective_active = self._effective_active()
        effective_processing = current_processing_config()
        identity_quality = dict(quality)
        # Before the first frame, QualityPipeline has no observation from which
        # to infer this state. The configured mode is authoritative and keeps a
        # fresh standby health document from claiming that swapping is active.
        identity_quality["processing_active"] = bool(
            effective_processing["processing_enabled"]
        )

        with self.input_lock:
            input_worker = self.input
            effective_input = self.effective_input
            input_frames = self._input_counters["received"] + int(
                getattr(input_worker, "received", 0) or 0
            )
            input_connections = self._input_counters["connections"] + int(
                getattr(input_worker, "connections", 0) or 0
            )
            input_disconnects = self._input_counters["disconnects"] + int(
                getattr(input_worker, "disconnects", 0) or 0
            )
            input_url = str(getattr(input_worker, "url", self.args.input_url))
            input_last_frame_at = float(
                getattr(input_worker, "last_frame_at", 0.0) or 0.0
            )
            input_alive = input_worker.is_alive()

        return {
            "state": "stopping" if self.stop_event.is_set() else "running",
            "active": effective_active,
            "uptime_seconds": round(now - self.started_at, 1),
            "route": self.args.route_id,
            "input": {
                "source": effective_input,
                "transport_mode": self.args.input_mode,
                "url": input_url,
                "frames": input_frames,
                "last_frame_age": age(input_last_frame_at),
                "connections": input_connections,
                "disconnects": input_disconnects,
                "worker_alive": input_alive,
                "generation": self.input_generation,
                "restarts": self.input_restarts,
            },
            "processing": {
                # ``mode`` matches the Windows health schema; the control
                # snapshot below retains the full canonical configuration.
                "mode": effective_processing["processing_mode"],
                "processing_enabled": effective_processing[
                    "processing_enabled"
                ],
                "processing_off_output": effective_processing[
                    "processing_off_output"
                ],
                "frames": self.processor.processed,
                "fps": round(self.processor.actual_fps, 2),
                "last_frame_age": age(self.processor.last_frame_at),
                "last_error": self.processor.last_error,
                "worker_alive": self.processor.is_alive(),
                "model": modules.globals.active_swapper_model,
                "backend": modules.globals.active_swapper_backend,
                "quality_status": model_quality,
                "target": dict(self.native_target),
                "identity_swap": identity_swap_health(
                    model_quality, identity_quality
                ),
                "timings_ms": {
                    name: round(value, 3)
                    for name, value in self.processor.timings_ms.items()
                },
            },
            "return": {
                "active": effective_active,
                "transport_open": bool(
                    getattr(self.output, "transport_open", False)
                ),
                "url": self.args.output_url,
                "frames": self.output.sent,
                "source_frames": self.output.source_frames,
                "repeated_frames": self.output.repeated_frames,
                "last_frame_age": age(self.output.last_frame_at),
                "connections": self.output.connections,
                "disconnects": self.output.disconnects,
                "worker_alive": self.output.is_alive(),
                "encoder": self.output.encoder,
                "preview_url": getattr(self.output, "preview_url", None),
                "system_camera_url": getattr(
                    self.output, "system_camera_url", None
                ),
                "audio": {
                    "enabled": bool(getattr(self.output, "audio_source", None)),
                    "source": getattr(self.output, "audio_source", None),
                    "codec": "aac-lc",
                    "sample_rate": 48_000,
                    "channels": 1,
                    "bitrate": getattr(self.output, "audio_bitrate", None),
                    "delay_ms": getattr(self.output, "audio_delay_ms", None),
                },
            },
            "benchmark": {
                **self.benchmark.status(),
                "observer_error": getattr(
                    self.processor, "benchmark_last_error", None
                ),
            },
            "control": self.control_snapshot(),
        }

    def _publish_health(self) -> None:
        document = self.snapshot()
        print(f"[health] {json.dumps(document, sort_keys=True)}", flush=True)
        if not self.args.health_file:
            return
        destination = Path(self.args.health_file).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def run(self) -> None:
        print(
            f"[route] {self.args.input_mode} {self.args.input_url} -> "
            f"{self.args.swapper_model}/{self.args.swapper_backend} -> "
            f"{self.args.output_url}",
            flush=True,
        )
        self.processor.start()
        self.input.start()
        self.output.start()
        self._workers_started = True
        self.start_control_server()
        next_status = time.monotonic()
        while not self.stop_event.wait(0.25):
            with self.input_lock:
                input_worker = self.input
                input_switching = self._input_switching
            workers = [
                ("processor", self.processor),
                ("return", self.output),
            ]
            if not input_switching:
                workers.insert(0, ("input", input_worker))
            for label, worker in workers:
                if not worker.is_alive():
                    detail = getattr(worker, "last_error", None)
                    raise RuntimeError(
                        f"{label} worker exited unexpectedly"
                        + (f": {detail}" if detail else "")
                    )
            if self.args.status_interval > 0 and time.monotonic() >= next_status:
                self._publish_health()
                next_status = time.monotonic() + self.args.status_interval

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        self._workers_started = False
        if self.control_server is not None:
            try:
                self.control_server.close()
            except OSError:
                pass
        with self.input_lock:
            self._input_stop_event.set()
            input_worker = self.input
            input_worker.close()
        self.output.close()
        for worker in (input_worker, self.output, self.processor):
            if worker.is_alive():
                worker.join(timeout=3.0)
        if self.control_thread is not None and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        try:
            self.control_socket_path.unlink()
        except FileNotFoundError:
            pass
        self.benchmark.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.environ.get("DLC_SOURCE_IMAGE") or active_source_image(),
        help=(
            "local source-face image (default: DLC_SOURCE_IMAGE, then the "
            "manager's active local source-history entry)"
        ),
    )
    parser.add_argument("--input-host", default="0.0.0.0")
    parser.add_argument("--input-port", type=int, default=10_001)
    parser.add_argument(
        "--input-mode",
        choices=("arch-webcam", "android-srt", "android-front", "android-back"),
        default="android-srt",
        help=(
            "android-front/android-back are semantic lens selections over the "
            "same persistent Camera2-owner transport; android-srt is the "
            "legacy alias for android-front; arch-webcam consumes the "
            "canonical sender's private loopback copy without opening a camera"
        ),
    )
    parser.add_argument(
        "--webcam-source-port",
        type=int,
        default=11_005,
        help="dedicated loopback MPEG-TS copy produced by the Arch webcam owner",
    )
    parser.add_argument("--android-host", default="192.168.1.12")
    parser.add_argument("--return-port", type=int, default=10_001)
    parser.add_argument(
        "--phone-relay-port",
        type=int,
        default=0,
        help=(
            "dedicated loopback MPEG-TS input for the exclusive phone-return "
            "relay; 0 retains the standalone direct-SRT output"
        ),
    )
    parser.add_argument(
        "--preview-host",
        default="127.0.0.1",
        help="loopback IPv4 address for the exact encoded return preview",
    )
    parser.add_argument(
        "--preview-port",
        type=int,
        default=11_004,
        help="local MPEG-TS/UDP preview port; 0 disables the preview tee",
    )
    parser.add_argument(
        "--system-camera-port",
        type=int,
        default=11_006,
        help=(
            "dedicated loopback MPEG-TS copy for the stable Arch virtual "
            "camera receiver; 0 disables the local processed source"
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="8M")
    parser.add_argument(
        "--return-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mux the Arch webcam microphone into the Android return stream",
    )
    parser.add_argument(
        "--return-audio-source",
        default=os.environ.get(
            "DLC_WEBCAM_AUDIO_SOURCE", DEFAULT_WEBCAM_AUDIO_SOURCE
        ),
        help="explicit Pulse/PipeWire webcam source (never follows desktop default)",
    )
    parser.add_argument("--return-audio-bitrate", default="128k")
    parser.add_argument(
        "--return-audio-delay-ms",
        type=int,
        default=220,
        help="delay microphone audio to align it with face-processing latency",
    )
    parser.add_argument(
        "--latency-us",
        type=int,
        # The SRT receiver buffer on the phone input is a fixed additive
        # latency.  Honour the deployment's SRT_LATENCY_US so the whole stack
        # shares one setting instead of the processor silently pinning 100ms.
        default=int(os.environ.get("SRT_LATENCY_US", "100000")),
    )
    parser.add_argument("--stale-seconds", type=float, default=1.0)
    parser.add_argument(
        "--active",
        type=parse_boolean_argument,
        default=True,
        metavar="{true,false}",
        help=(
            "own and open the return delivery transport (default: true); "
            "false keeps the stream workers hot without competing for the "
            "phone or local output endpoints; inference is controlled "
            "independently by --processing-mode"
        ),
    )
    parser.add_argument(
        "--processing-mode",
        choices=("face_swap", "passthrough"),
        default="face_swap",
        help=(
            "initial processor mode (default: face_swap); passthrough keeps "
            "the decoder and control worker available without loading or "
            "running detector/swap inference"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--execution-provider",
        choices=("auto", "cpu", "cuda", "rocm", "openvino"),
        default="auto",
        help="ONNX provider for face analysis; ncnn swap inference is independent",
    )
    parser.add_argument(
        "--swapper-model",
        choices=("native-256", "inswapper-128", "auto"),
        default="inswapper-128",
        help=(
            "face-swap model family; the default is the locally verified "
            "production INSwapper model, while native-256 remains an explicit "
            "development-only option until a qualified checkpoint is installed"
        ),
    )
    parser.add_argument(
        "--swapper-backend",
        choices=("ncnn", "ort", "auto"),
        default="ncnn",
    )
    discovered_manifest = discover_native256_manifest()
    parser.add_argument(
        "--native256-manifest",
        default=str(discovered_manifest) if discovered_manifest else None,
        help=(
            "local hash-pinned native-256 manifest used by the fixed Arch "
            "model target (default: DLC_NATIVE256_MANIFEST, then XDG data)"
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_vaapi", "libx264"),
        default="auto",
    )
    parser.add_argument("--vaapi-device", default="/dev/dri/renderD128")
    parser.add_argument("--detection-interval", type=int, default=1)
    parser.add_argument("--tracking-grace-frames", type=int, default=8)
    parser.add_argument("--minimum-detection-score", type=float, default=0.35)
    parser.add_argument(
        "--quality-mode", choices=("monitor", "balanced", "strict"), default="balanced"
    )
    parser.add_argument(
        "--state-dir",
        default=str(REPOSITORY_ROOT / "arch-linux" / "runtime" / "android-phone-processed"),
    )
    parser.add_argument("--health-file")
    parser.add_argument(
        "--control-socket",
        default=str(default_control_socket()),
        help="Unix socket for hot processor/input control",
    )
    parser.add_argument("--status-interval", type=float, default=5.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments and print the route without opening model or sockets",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "validate the configured bootstrap locally, then exit without "
            "opening sockets; passthrough deliberately leaves inference unloaded"
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        if args.source:
            args.source = str(Path(args.source).expanduser().resolve())
        elif not args.dry_run:
            raise ValueError("--source (or DLC_SOURCE_IMAGE) is required")
        if args.source and not Path(args.source).is_file():
            raise ValueError(f"source image does not exist: {args.source}")
        if args.width < 64 or args.height < 64:
            raise ValueError("width and height must be at least 64")
        if not 1 <= args.fps <= 120:
            raise ValueError("FPS must be between 1 and 120")
        if not 1 <= args.cpu_threads <= 32:
            raise ValueError("CPU threads must be between 1 and 32")
        if args.latency_us < 20_000:
            raise ValueError("SRT latency must be at least 20000 microseconds")
        if args.stale_seconds < 0.2:
            raise ValueError("stale seconds must be at least 0.2")
        if args.status_interval < 0:
            raise ValueError("status interval must not be negative")
        if args.return_audio and not args.return_audio_source.strip():
            raise ValueError("return audio source must not be empty")
        if not 0 <= args.return_audio_delay_ms <= 2_000:
            raise ValueError("return audio delay must be between 0 and 2000 ms")
        if not 1 <= args.detection_interval <= 30:
            raise ValueError("detection interval must be between 1 and 30")
        if not 0 <= args.tracking_grace_frames <= 120:
            raise ValueError("tracking grace frames must be between 0 and 120")
        if not 0.0 <= args.minimum_detection_score <= 1.0:
            raise ValueError("minimum detection score must be between 0 and 1")
        if args.native256_manifest:
            manifest = Path(args.native256_manifest).expanduser().resolve()
            if not manifest.is_file():
                raise ValueError(f"native-256 manifest does not exist: {manifest}")
            args.native256_manifest = str(manifest)
        args.semantic_input = semantic_input(args.input_mode)
        if args.input_mode == "arch-webcam":
            args.input_url = local_owner_feed_url(args.webcam_source_port)
            args.input_label = "arch-webcam-owner-feed"
            args.route_id = "arch-webcam-processed-to-android"
        else:
            args.input_url = srt_listener_url(
                args.input_host, args.input_port, args.latency_us
            )
            args.input_label = "android-slot-0"
            args.route_id = "android-camera-processed-to-android"
        args.output_url = (
            local_preview_url("127.0.0.1", args.phone_relay_port)
            if args.phone_relay_port
            else srt_caller_url(args.android_host, args.return_port, args.latency_us)
        )
        args.preview_url = (
            None
            if args.preview_port == 0
            else local_preview_url(args.preview_host, args.preview_port)
        )
        args.system_camera_url = (
            None
            if args.system_camera_port == 0
            else local_preview_url("127.0.0.1", args.system_camera_port)
        )
        reserved_local_ports = {11_000, 11_001, 11_002, 11_003}
        if (
            args.input_mode == "arch-webcam"
            and args.webcam_source_port in reserved_local_ports
        ):
            raise ValueError(
                "webcam source port must use its dedicated owner copy, not "
                "reserved local ports 11000-11003"
            )
        endpoint_ports = {
            args.input_port,
            args.return_port,
            args.webcam_source_port,
        }
        if args.phone_relay_port:
            if args.phone_relay_port in reserved_local_ports | endpoint_ports:
                raise ValueError(
                    "phone relay port must be distinct from input/return and "
                    "reserved local ports 11000-11003"
                )
            endpoint_ports.add(args.phone_relay_port)
        if args.preview_port and args.preview_port in reserved_local_ports | endpoint_ports:
            raise ValueError(
                "preview port must be distinct from input/return and reserved "
                "local ports 11000-11003"
            )
        if args.system_camera_port and args.system_camera_port in (
            reserved_local_ports | endpoint_ports | {args.preview_port}
        ):
            raise ValueError(
                "system camera port must be distinct from preview, input/return, "
                "and reserved local ports 11000-11003"
            )
        # The listener is local and the return port is on Android.  They may
        # intentionally use the same number (for example when a host firewall
        # temporarily admits only local UDP/10001).
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("ffmpeg is not available on PATH")
        if not ffmpeg_has_srt(ffmpeg):
            raise ValueError("the installed ffmpeg does not support SRT")
        if args.encoder == "h264_vaapi" and not Path(args.vaapi_device).exists():
            raise ValueError(f"VAAPI device does not exist: {args.vaapi_device}")
        args.providers = resolve_execution_providers(
            args.execution_provider, available_onnx_providers()
        )
    except ValueError as error:
        parser.error(str(error))


def configure_pipeline(args: argparse.Namespace) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    try:
        import cv2

        cv2.setNumThreads(args.cpu_threads)
    except (ImportError, AttributeError):
        pass
    modules.globals.execution_providers = list(args.providers)
    modules.globals.execution_threads = args.cpu_threads
    modules.globals.swapper_model = args.swapper_model
    modules.globals.swapper_backend = args.swapper_backend
    modules.globals.source_path = args.source
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.headless = True
    modules.globals.map_faces = False
    modules.globals.many_faces = False
    modules.globals.processing_enabled = args.processing_mode == "face_swap"
    # A standby worker must forward its input internally rather than create a
    # black frame. Delivery ownership remains an independent ``--active``
    # concern, so inactive standby still keeps the decoder/control path hot.
    modules.globals.processing_off_output = "passthrough"
    modules.globals.quality_mode = args.quality_mode
    modules.globals.tracking_enabled = True
    modules.globals.detection_interval = args.detection_interval
    modules.globals.tracking_grace_frames = args.tracking_grace_frames
    modules.globals.minimum_detection_score = args.minimum_detection_score
    if args.native256_manifest:
        os.environ["DLC_NATIVE256_MANIFEST"] = args.native256_manifest


def preflight_pipeline(args: argparse.Namespace) -> None:
    if args.processing_mode == "passthrough":
        # Do not turn a non-owning standby into an inference workload merely
        # to prove that its configured model can load. The first face-swap
        # frame performs normal lazy initialization after a hot control switch.
        args.model_quality = "not-loaded"
        print(
            "[model] standby: detector and face-swap model left unloaded",
            flush=True,
        )
        return
    source_face = get_one_face(imread_unicode(args.source))
    if source_face is None:
        raise RuntimeError(f"no face was detected in source image: {args.source}")
    from modules.processors.frame import face_swapper

    active_swapper = face_swapper.get_face_swapper()
    if active_swapper is None:
        raise RuntimeError(
            f"could not initialize {args.swapper_model}/{args.swapper_backend}; "
            "check the local manifest, hashes, ncnn assets, and bridge library"
        )
    args.model_quality = str(
        getattr(active_swapper, "quality_status", "production")
    )
    print(
        f"[model] ready: {modules.globals.active_swapper_model}/"
        f"{modules.globals.active_swapper_backend}; ONNX providers={args.providers}",
        flush=True,
    )


def dry_run_document(args: argparse.Namespace) -> dict:
    return {
        "active": args.active,
        "processing_mode": args.processing_mode,
        "input": args.input_url,
        "input_mode": args.input_mode,
        "semantic_input": args.semantic_input,
        "route": args.route_id,
        "return": args.output_url,
        "preview": args.preview_url,
        "system_camera": args.system_camera_url,
        "source": args.source,
        "swapper_model": args.swapper_model,
        "swapper_backend": args.swapper_backend,
        "model_quality": getattr(args, "model_quality", "not-loaded"),
        "model_target": native256_target_status(
            Path(args.native256_manifest) if args.native256_manifest else None
        ),
        "onnx_providers": args.providers,
        "encoder": args.encoder,
        "return_audio": {
            "enabled": args.return_audio,
            "source": args.return_audio_source if args.return_audio else None,
            "codec": "aac-lc",
            "sample_rate": 48_000,
            "channels": 1,
            "bitrate": args.return_audio_bitrate,
            "delay_ms": args.return_audio_delay_ms,
        },
        "opens_camera_device": False,
        "control_socket": args.control_socket,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.dry_run:
        print(json.dumps(dry_run_document(args), indent=2, sort_keys=True))
        return 0
    configure_pipeline(args)
    try:
        preflight_pipeline(args)
    except Exception as error:
        print(f"[preflight] {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.preflight_only:
        print(json.dumps(dry_run_document(args), indent=2, sort_keys=True))
        return 0

    service = AndroidStreamService(args)

    def request_stop(*_unused) -> None:
        service.stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        service.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"[service] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
