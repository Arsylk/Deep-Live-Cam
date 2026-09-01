from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "arch-linux" / "bin" / "android_stream_processor.py"
SPEC = importlib.util.spec_from_file_location("android_stream_processor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stream_processor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream_processor)

CONTROL_CLIENT_PATH = ROOT / "arch-linux" / "bin" / "local_processor_control.py"
CONTROL_CLIENT_SPEC = importlib.util.spec_from_file_location(
    "local_processor_control", CONTROL_CLIENT_PATH
)
assert CONTROL_CLIENT_SPEC is not None and CONTROL_CLIENT_SPEC.loader is not None
local_processor_control = importlib.util.module_from_spec(CONTROL_CLIENT_SPEC)
CONTROL_CLIENT_SPEC.loader.exec_module(local_processor_control)


def test_linux_auto_provider_ignores_cuda_wheel_without_nvidia_hardware():
    selected = stream_processor.resolve_execution_providers(
        "auto",
        [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        system="Linux",
        has_nvidia=False,
    )

    assert selected == ["CPUExecutionProvider"]


def test_linux_auto_provider_prefers_rocm_and_keeps_cpu_fallback():
    selected = stream_processor.resolve_execution_providers(
        "auto",
        ["CPUExecutionProvider", "ROCMExecutionProvider"],
        system="Linux",
        has_nvidia=False,
    )

    assert selected == ["ROCMExecutionProvider", "CPUExecutionProvider"]


def test_explicit_unavailable_provider_fails_instead_of_changing_backend():
    with pytest.raises(ValueError, match="CUDAExecutionProvider is unavailable"):
        stream_processor.resolve_execution_providers(
            "cuda",
            ["CPUExecutionProvider"],
            system="Linux",
            has_nvidia=False,
        )


def test_local_listener_and_remote_return_may_use_same_port_number():
    listener = stream_processor.srt_listener_url("0.0.0.0", 10001, 100_000)
    returned = stream_processor.srt_caller_url("192.168.1.12", 10001, 100_000)

    assert listener.startswith("srt://0.0.0.0:10001?mode=listener")
    assert returned.startswith("srt://192.168.1.12:10001?mode=caller")
    assert "messageapi=1" in listener
    assert "messageapi=1" in returned


def test_arch_webcam_owner_feed_is_loopback_only():
    source = stream_processor.local_owner_feed_url(11005)

    assert source.startswith("udp://127.0.0.1:11005?")
    assert "fifo_size=1000000" in source
    assert "/dev/video" not in source


def test_arch_webcam_rejects_a_shared_reserved_owner_port(monkeypatch, capsys):
    monkeypatch.setattr(stream_processor, "active_source_image", lambda: None)
    parser = stream_processor.build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--input-mode",
            "arch-webcam",
            "--webcam-source-port",
            "11001",
        ]
    )

    with pytest.raises(SystemExit) as error:
        stream_processor.validate_args(parser, args)

    assert error.value.code == 2
    assert "must use its dedicated owner copy" in capsys.readouterr().err


def test_development_checkpoint_never_reports_a_confirmed_swap():
    status = stream_processor.identity_swap_health(
        "development",
        {
            "processing_active": True,
            "swap_applied": True,
            "face": {
                "available": True,
                "core_mean_absolute_delta": 20.0,
                "core_changed_pixels_percent": 90.0,
            },
        },
    )

    assert status["status"] == "unqualified-checkpoint"
    assert status["checkpoint_qualified"] is False
    assert status["visual_effect_confirmed"] is False
    assert status["identity_change_verified"] is False


def test_qualified_checkpoint_requires_measurable_face_core_change():
    unchanged = stream_processor.identity_swap_health(
        "qualified",
        {
            "processing_active": True,
            "swap_applied": True,
            "face": {
                "available": True,
                "core_mean_absolute_delta": 0.2,
                "core_changed_pixels_percent": 2.0,
            },
        },
    )
    changed = stream_processor.identity_swap_health(
        "qualified",
        {
            "processing_active": True,
            "swap_applied": True,
            "face": {
                "available": True,
                "core_mean_absolute_delta": 4.0,
                "core_changed_pixels_percent": 25.0,
            },
        },
    )

    assert unchanged["status"] == "no-measurable-effect"
    assert unchanged["visual_effect_confirmed"] is False
    assert changed["status"] == "visual-effect-detected"
    assert changed["visual_effect_confirmed"] is True
    assert changed["identity_change_verified"] is False


def test_preview_url_is_loopback_only_and_uses_a_dedicated_port():
    preview = stream_processor.local_preview_url("127.0.0.1", 11004)

    assert preview == "udp://127.0.0.1:11004?pkt_size=1316"
    with pytest.raises(ValueError, match="loopback IPv4"):
        stream_processor.local_preview_url("192.168.1.11", 11004)
    with pytest.raises(ValueError, match="loopback IPv4"):
        stream_processor.local_preview_url("::1", 11004)


def test_owner_camera_profile_reads_declared_state_without_device_access(tmp_path):
    state = tmp_path / "sender.json"
    state.write_text(
        json.dumps(
            {
                "camera_profile": "natural-indoor",
                "capture": "/run/deep-live-cam/source/video0",
                "camera_mode": "1280x720@30",
                "camera_controls": {
                    "brightness": "-12",
                    "exposure_dynamic_framerate": "0",
                },
            }
        ),
        encoding="utf-8",
    )

    profile = stream_processor.owner_camera_profile(state)

    assert profile == {
        "profile": "natural-indoor",
        "capture": "/run/deep-live-cam/source/video0",
        "camera_mode": "1280x720@30",
        "controls": {
            "brightness": "-12",
            "exposure_dynamic_framerate": "0",
        },
        "source": "capture-owner-state",
    }


@pytest.mark.parametrize("encoder", ["h264_vaapi", "libx264"])
def test_linux_return_uses_one_encoder_for_phone_and_exact_local_preview(encoder):
    phone = stream_processor.srt_caller_url("192.168.1.12", 10001, 100_000)
    preview = stream_processor.local_preview_url("127.0.0.1", 11004)
    output = stream_processor.LinuxSrtOutput(
        phone,
        1280,
        720,
        30,
        "8M",
        stream_processor.LatestFrame(),
        threading.Event(),
        encoder=encoder,
        preview_url=preview,
    )

    command = output._ffmpeg_command()
    tee_target = command[-1]
    assert command.count("-i") == 1
    assert command.count("-c:v") == 1
    assert command[command.index("-bsf:v") + 1] == "dump_extra=freq=keyframe"
    assert command.count("tee") == 1
    assert tee_target.count(phone) == 1
    assert tee_target.count(preview) == 1
    assert "onfail=ignore" in tee_target.split("|")[0]
    assert "onfail=ignore" in tee_target.split("|")[1]
    rendered = " ".join(command)
    assert "/dev/video" not in rendered
    assert "192.168.1.35" not in rendered
    assert ":11003" not in rendered
    assert ":10003" not in rendered


@pytest.mark.parametrize("encoder", ["h264_vaapi", "libx264"])
def test_linux_return_fans_out_to_phone_preview_and_stable_camera(encoder):
    phone = stream_processor.srt_caller_url("192.168.1.12", 10001, 100_000)
    preview = stream_processor.local_preview_url("127.0.0.1", 11004)
    system_camera = stream_processor.local_preview_url("127.0.0.1", 11006)
    output = stream_processor.LinuxSrtOutput(
        phone,
        1280,
        720,
        30,
        "8M",
        stream_processor.LatestFrame(),
        threading.Event(),
        encoder=encoder,
        preview_url=preview,
        system_camera_url=system_camera,
    )

    command = output._ffmpeg_command()
    target = command[-1]
    assert command.count("-c:v") == 1
    assert command.count("tee") == 1
    assert target.count(phone) == 1
    assert target.count(preview) == 1
    assert target.count(system_camera) == 1
    assert len(target.split("|")) == 3
    assert all("onfail=ignore" in part for part in target.split("|")[1:])
    assert "/dev/video" not in " ".join(command)


def test_linux_return_without_preview_preserves_plain_primary_mux():
    phone = stream_processor.srt_caller_url("192.168.1.12", 10001, 100_000)
    output = stream_processor.LinuxSrtOutput(
        phone,
        1280,
        720,
        30,
        "8M",
        stream_processor.LatestFrame(),
        threading.Event(),
        encoder="libx264",
        preview_url=None,
    )

    command = output._ffmpeg_command()
    assert "tee" not in command
    assert command[-1] == phone
    assert command[-3:-1] == ["-mpegts_flags", "resend_headers"]


@pytest.mark.parametrize("encoder", ["h264_vaapi", "libx264"])
def test_linux_return_muxes_explicit_webcam_microphone_into_both_outputs(encoder):
    phone = stream_processor.srt_caller_url("192.168.1.12", 10001, 100_000)
    preview = stream_processor.local_preview_url("127.0.0.1", 11004)
    source = stream_processor.DEFAULT_WEBCAM_AUDIO_SOURCE
    output = stream_processor.LinuxSrtOutput(
        phone,
        1280,
        720,
        30,
        "8M",
        stream_processor.LatestFrame(),
        threading.Event(),
        encoder=encoder,
        preview_url=preview,
        audio_source=source,
        audio_bitrate="128k",
        audio_delay_ms=220,
    )

    command = output._ffmpeg_command()
    assert command.count("-i") == 2
    assert command[command.index("-f", command.index("pipe:0")) + 1] == "pulse"
    assert source in command
    assert command.count("-map") == 2
    assert "0:v:0" in command
    assert "1:a:0" in command
    assert "-an" not in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "1"
    audio_filter = command[command.index("-af") + 1]
    assert "aresample=async=1:first_pts=0" in audio_filter
    assert "adelay=220:all=1" in audio_filter
    assert command[-1] == stream_processor.tee_mux_target(phone, preview)
    assert command[command.index("-use_fifo") + 1] == "1"
    assert "attempt_recovery=1" in command[command.index("-fifo_options") + 1]


def test_dry_run_supports_firewall_workaround_and_opens_no_camera(monkeypatch, capsys):
    monkeypatch.setattr(stream_processor.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(stream_processor, "ffmpeg_has_srt", lambda _path: True)
    monkeypatch.setattr(
        stream_processor,
        "available_onnx_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(stream_processor, "_has_nvidia_device", lambda: False)

    result = stream_processor.main(
        [
            "--dry-run",
            "--input-mode",
            "android-srt",
            "--input-port",
            "10001",
            "--android-host",
            "192.168.1.12",
            "--return-port",
            "10001",
            "--detection-interval",
            "2",
            "--swapper-backend",
            "ncnn",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert '"opens_camera_device": false' in output
    assert '"processing_mode": "face_swap"' in output
    assert '"CPUExecutionProvider"' in output
    assert "srt://0.0.0.0:10001?mode=listener" in output
    assert "srt://192.168.1.12:10001?mode=caller" in output


def test_dry_run_defaults_to_phone_front_listener_without_opening_camera(
    monkeypatch, capsys
):
    monkeypatch.setattr(stream_processor.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(stream_processor, "ffmpeg_has_srt", lambda _path: True)
    monkeypatch.setattr(
        stream_processor,
        "available_onnx_providers",
        lambda: ["CPUExecutionProvider"],
    )

    result = stream_processor.main(["--dry-run"])

    output = capsys.readouterr().out
    assert result == 0
    assert '"input_mode": "android-srt"' in output
    assert '"route": "android-camera-processed-to-android"' in output
    assert '"swapper_model": "inswapper-128"' in output
    assert "srt://0.0.0.0:10001?mode=listener" in output
    assert "srt://192.168.1.12:10001?mode=caller" in output
    assert '"opens_camera_device": false' in output


class FakeInput:
    def __init__(self, url, *_args, **kwargs):
        self.url = url
        self.label = kwargs["label"]
        self.received = 0
        self.last_frame_at = 0.0
        self.connections = 0
        self.disconnects = 0
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started

    def close(self):
        self.started = False

    def join(self, timeout=None):
        del timeout


class FakeOutput:
    def __init__(self, url, *_args, **kwargs):
        self.url = url
        self.label = kwargs["label"]
        self.encoder = kwargs["encoder"] or "fake-auto"
        self.sent = 0
        self.source_frames = 0
        self.repeated_frames = 0
        self.last_frame_at = 0.0
        self.connections = 0
        self.disconnects = 0
        self.started = False
        self.active = True
        self.transport_open = False
        self.active_changes = []

    def set_active(self, active):
        self.active = active
        self.active_changes.append(active)
        if not active:
            self.transport_open = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started

    def close(self):
        self.started = False

    def join(self, timeout=None):
        del timeout


class FakeProcessor:
    def __init__(self, *_args):
        self.processed = 0
        self.actual_fps = 0.0
        self.last_frame_at = 0.0
        self.last_error = None
        self.timings_ms = {}
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started

    def join(self, timeout=None):
        del timeout


def test_service_wires_slot_zero_without_opening_a_camera(tmp_path):
    args = argparse.Namespace(
        input_url=stream_processor.srt_listener_url("0.0.0.0", 10000, 100_000),
        input_mode="android-srt",
        input_label="android-slot-0",
        route_id="android-camera-processed-to-android",
        output_url=stream_processor.srt_caller_url("192.168.1.12", 10001, 100_000),
        width=1280,
        height=720,
        fps=30,
        bitrate="8M",
        stale_seconds=1.0,
        encoder="libx264",
        vaapi_device="/dev/dri/renderD128",
        state_dir=str(tmp_path),
        status_interval=0.0,
        health_file=None,
        swapper_model="native-256",
        swapper_backend="ncnn",
        model_quality="development",
    )
    service = stream_processor.AndroidStreamService(
        args,
        input_factory=FakeInput,
        output_factory=FakeOutput,
        processor_factory=FakeProcessor,
    )

    assert service.input.url.startswith("srt://0.0.0.0:10000?mode=listener")
    assert service.output.url.startswith("srt://192.168.1.12:10001?mode=caller")
    assert service.output.encoder == "libx264"
    assert isinstance(service.stop_event, threading.Event)
    swap = service.snapshot()["processing"]["identity_swap"]
    assert swap["status"] == "unqualified-checkpoint"
    assert swap["visual_effect_confirmed"] is False


def runtime_args(tmp_path, **overrides):
    values = {
        "source": None,
        "input_host": "0.0.0.0",
        "input_port": 10001,
        "input_url": stream_processor.srt_listener_url(
            "0.0.0.0", 10001, 100_000
        ),
        "input_mode": "android-srt",
        "semantic_input": "android-front",
        "input_label": "android-slot-0",
        "route_id": "android-camera-processed-to-android",
        "webcam_source_port": 11005,
        "latency_us": 100_000,
        "output_url": stream_processor.srt_caller_url(
            "192.168.1.12", 10001, 100_000
        ),
        "width": 1280,
        "height": 720,
        "fps": 30,
        "bitrate": "8M",
        "stale_seconds": 1.0,
        "encoder": "libx264",
        "vaapi_device": "/dev/dri/renderD128",
        "state_dir": str(tmp_path),
        "status_interval": 0.0,
        "health_file": None,
        "swapper_model": "inswapper-128",
        "swapper_backend": "ncnn",
        "model_quality": "production",
        "control_socket": str(tmp_path / "processor-control.sock"),
        "native256_manifest": None,
        "active": True,
        "processing_mode": "face_swap",
        "cpu_threads": 1,
        "providers": ["CPUExecutionProvider"],
        "quality_mode": "balanced",
        "detection_interval": 1,
        "tracking_grace_frames": 8,
        "minimum_detection_score": 0.35,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TrackingInput(FakeInput):
    instances = []

    def __init__(self, url, *args, **kwargs):
        super().__init__(url, *args, **kwargs)
        self.stop_event = args[3]
        self.route_token = kwargs.get("route_token")
        type(self).instances.append(self)


def make_runtime_service(tmp_path, *, input_factory=TrackingInput, **overrides):
    return stream_processor.AndroidStreamService(
        runtime_args(tmp_path, **overrides),
        input_factory=input_factory,
        output_factory=FakeOutput,
        processor_factory=FakeProcessor,
    )


def test_runtime_processing_control_is_immediate_and_canonical(
    tmp_path, monkeypatch
):
    for name in (
        "detection_interval",
        "tracking_grace_frames",
        "quality_mode",
        "processing_off_output",
        "mouth_mask",
    ):
        monkeypatch.setattr(
            stream_processor.modules.globals,
            name,
            getattr(stream_processor.modules.globals, name),
        )
    monkeypatch.setattr(stream_processor.modules.globals, "opacity", 1.0)
    monkeypatch.setattr(stream_processor.modules.globals, "sharpness", 0.0)
    monkeypatch.setattr(stream_processor.modules.globals, "processing_enabled", True)
    service = make_runtime_service(tmp_path)
    output = service.output
    processor = service.processor

    response = service.apply_control_request(
        {
            "revision": 19,
            "processing": {
                "processing_mode": "passthrough",
                "opacity": 0.42,
                "sharpness": 0.7,
                "detection_interval": 3,
                "tracking_grace_frames": 9,
                "quality_mode": "strict",
            },
        }
    )

    assert response["ok"] is True
    assert response["revision"] == 19
    assert response["desired"]["processing"]["opacity"] == 0.42
    assert response["effective"]["processing"]["opacity"] == 0.42
    assert response["effective"]["processing"]["processing_mode"] == "passthrough"
    assert stream_processor.modules.globals.detection_interval == 3
    assert stream_processor.modules.globals.tracking_grace_frames == 9
    assert service.output is output
    assert service.processor is processor
    service.close()


def test_semantic_input_switch_rebuilds_only_decoder(tmp_path):
    TrackingInput.instances.clear()
    service = make_runtime_service(tmp_path)
    service.input.start()
    service.output.start()
    service.processor.start()
    service._workers_started = True
    first_input = service.input
    output = service.output
    processor = service.processor

    lens_response = service.apply_control_request({"input": "android-back"})

    assert lens_response["effective"]["input"] == "android-back"
    assert service.input is first_input
    assert service.input_restarts == 0
    assert service.input.route_token == 1

    route_response = service.apply_control_request({"input": "arch-webcam"})

    assert route_response["effective"]["input"] == "arch-webcam"
    assert service.input is not first_input
    assert first_input.started is False
    assert first_input.stop_event.is_set()
    assert service.input.started is True
    assert service.input.route_token == 2
    assert service.input_restarts == 1
    assert service.output is output
    assert service.output.is_alive()
    assert service.processor is processor
    assert service.processor.is_alive()
    service.close()


def test_fixed_native256_target_is_applied_without_restarting_stream_workers(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    available = {
        "model_target": "native-256",
        "backend_target": "ncnn",
        "model_id": "dlc_swap256m",
        "manifest": str(manifest),
        "available": True,
        "quality_status": "development",
        "qualified": False,
        "warning": "checkpoint is development/unqualified",
    }
    monkeypatch.setattr(
        stream_processor, "native256_target_status", lambda _path: dict(available)
    )
    for name in ("swapper_model", "swapper_backend"):
        monkeypatch.setattr(
            stream_processor.modules.globals,
            name,
            getattr(stream_processor.modules.globals, name),
        )
    service = make_runtime_service(tmp_path, native256_manifest=str(manifest))
    output = service.output
    processor = service.processor

    response = service.apply_control_request(
        {"swapper_model": "native-256", "swapper_backend": "ncnn"}
    )

    assert response["ok"] is True
    assert response["desired"]["model"] == {
        "swapper_model": "native-256",
        "swapper_backend": "ncnn",
    }
    assert response["effective"]["model"]["swapper_model"] == "native-256"
    assert response["model_target"]["quality_status"] == "development"
    assert response["model_target"]["qualified"] is False
    assert response["in_sync"] is True
    assert response["sync"]["model_configuration"] is True
    assert response["sync"]["model_runtime_ready"] is False
    assert service.args.model_quality == "development"
    assert service.output is output
    assert service.processor is processor
    service.close()


def test_unavailable_native256_request_preserves_working_model(tmp_path, monkeypatch):
    unavailable = {
        "model_target": "native-256",
        "backend_target": "ncnn",
        "model_id": None,
        "manifest": None,
        "available": False,
        "quality_status": "unavailable",
        "qualified": False,
        "warning": "native pack missing",
    }
    monkeypatch.setattr(stream_processor, "discover_native256_manifest", lambda: None)
    monkeypatch.setattr(
        stream_processor, "native256_target_status", lambda _path: dict(unavailable)
    )
    monkeypatch.setattr(
        stream_processor.modules.globals, "swapper_model", "inswapper-128"
    )
    service = make_runtime_service(tmp_path)

    with pytest.raises(ValueError, match="native pack missing"):
        service.apply_control_request(
            {"swapper_model": "native-256", "swapper_backend": "ncnn"}
        )

    assert stream_processor.modules.globals.swapper_model == "inswapper-128"
    service.close()


def test_unix_control_socket_returns_canonical_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stream_processor.modules.globals,
        "opacity",
        stream_processor.modules.globals.opacity,
    )
    service = make_runtime_service(tmp_path)
    service.start_control_server()
    try:
        response = local_processor_control.request_processor(
            {"op": "set", "processing": {"opacity": 0.31}},
            socket_path=service.control_socket_path,
        )
    finally:
        service.close()

    assert response["ok"] is True
    assert response["effective"]["processing"]["opacity"] == 0.31
    assert response["capabilities"]["hot_input_decoder"] is True
    assert response["capabilities"]["restarts_output_on_input_change"] is False


def test_passthrough_preflight_does_not_load_detector_or_swapper(
    monkeypatch, capsys
):
    calls = {"image": 0, "source_detector": 0, "swapper": 0}

    def forbidden(name):
        def invoke(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must remain unloaded in passthrough")

        return invoke

    from modules.processors.frame import face_swapper

    monkeypatch.setattr(stream_processor, "imread_unicode", forbidden("image"))
    monkeypatch.setattr(
        stream_processor, "get_one_face", forbidden("source_detector")
    )
    monkeypatch.setattr(
        face_swapper, "get_face_swapper", forbidden("swapper")
    )
    args = argparse.Namespace(processing_mode="passthrough")

    stream_processor.preflight_pipeline(args)

    assert calls == {"image": 0, "source_detector": 0, "swapper": 0}
    assert args.model_quality == "not-loaded"
    assert "detector and face-swap model left unloaded" in capsys.readouterr().out


def test_inactive_passthrough_bootstrap_keeps_workers_and_control_hot_without_inference(
    tmp_path, monkeypatch
):
    live_runtime = importlib.import_module(stream_processor.LiveProcessor.__module__)
    calls = {
        "source_detector": 0,
        "target_detector": 0,
        "processor_loader": 0,
        "swapper_loader": 0,
        "swap": 0,
    }

    def forbidden(name):
        def invoke(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} ran in passthrough standby")

        return invoke

    from modules.processors.frame import face_swapper

    monkeypatch.setattr(
        live_runtime, "get_one_face", forbidden("source_detector")
    )
    monkeypatch.setattr(
        live_runtime, "detect_many_faces_fast", forbidden("target_detector")
    )
    monkeypatch.setattr(
        live_runtime,
        "get_frame_processors_modules",
        forbidden("processor_loader"),
    )
    monkeypatch.setattr(
        face_swapper, "get_face_swapper", forbidden("swapper_loader")
    )
    monkeypatch.setattr(face_swapper, "swap_face", forbidden("swap"))

    configured_globals = (
        "execution_providers",
        "execution_threads",
        "swapper_model",
        "swapper_backend",
        "source_path",
        "frame_processors",
        "headless",
        "map_faces",
        "many_faces",
        "processing_enabled",
        "processing_off_output",
        "quality_mode",
        "tracking_enabled",
        "detection_interval",
        "tracking_grace_frames",
        "minimum_detection_score",
        "active_swapper_model",
        "active_swapper_backend",
        "live_mirror",
    )
    for name in configured_globals:
        monkeypatch.setattr(
            stream_processor.modules.globals,
            name,
            getattr(stream_processor.modules.globals, name),
        )
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
    monkeypatch.setenv("KMP_BLOCKTIME", "0")

    releases = []

    def release_without_loading(processor):
        releases.append(True)
        processor.gpu_models_released = True
        stream_processor.modules.globals.active_swapper_model = "not-loaded"
        stream_processor.modules.globals.active_swapper_backend = "not-loaded"

    monkeypatch.setattr(
        live_runtime.LiveProcessor,
        "_release_gpu_models",
        release_without_loading,
    )

    args = runtime_args(
        tmp_path,
        active=False,
        processing_mode="passthrough",
        width=64,
        height=64,
    )
    stream_processor.configure_pipeline(args)
    stream_processor.preflight_pipeline(args)
    service = stream_processor.AndroidStreamService(
        args,
        input_factory=TrackingInput,
        output_factory=FakeOutput,
        processor_factory=stream_processor.LiveProcessor,
    )
    frame = stream_processor.np.full((64, 64, 3), 73, dtype="uint8")

    try:
        service.processor.start()
        service.input.start()
        service.output.start()
        service._workers_started = True
        service.start_control_server()
        service.input_frames.put(frame)
        forwarded = service.output_frames.get(timeout=2.0)
        control = local_processor_control.request_processor(
            {"op": "get"}, socket_path=service.control_socket_path
        )
        health = service.snapshot()

        assert stream_processor.np.array_equal(forwarded, frame)
        assert calls == {
            "source_detector": 0,
            "target_detector": 0,
            "processor_loader": 0,
            "swapper_loader": 0,
            "swap": 0,
        }
        assert releases
        assert service.processor.last_error is None
        assert service.processor.processed >= 1
        assert service.input.is_alive()
        assert service.processor.is_alive()
        assert service.output.is_alive()
        assert service.control_thread is not None
        assert service.control_thread.is_alive()
        assert control["effective"]["active"] is False
        assert control["effective"]["processing"]["processing_mode"] == "passthrough"
        assert health["processing"]["mode"] == "passthrough"
        assert health["processing"]["processing_enabled"] is False
        assert health["processing"]["identity_swap"]["status"] == "disabled"
        assert health["return"]["active"] is False
        assert health["return"]["transport_open"] is False
        assert health["return"]["worker_alive"] is True
    finally:
        service.close()


def test_hot_active_control_preserves_all_workers_and_exposes_canonical_state(
    tmp_path,
):
    service = make_runtime_service(tmp_path)
    service.input.start()
    service.output.start()
    service.processor.start()
    service._workers_started = True
    workers = (service.input, service.output, service.processor)

    inactive = service.apply_control_request({"active": False, "revision": 41})

    assert (service.input, service.output, service.processor) == workers
    assert all(worker.is_alive() for worker in workers)
    assert inactive["revision"] == 41
    assert inactive["desired"]["active"] is False
    assert inactive["effective"]["active"] is False
    assert inactive["sync"]["active"] is True
    assert inactive["capabilities"]["active"] is True
    assert inactive["capabilities"]["hot_delivery_ownership"] is True
    assert service.snapshot()["active"] is False
    assert service.snapshot()["return"]["active"] is False

    active = service.apply_control_request({"active": True})

    assert (service.input, service.output, service.processor) == workers
    assert all(worker.is_alive() for worker in workers)
    assert active["desired"]["active"] is True
    assert active["effective"]["active"] is True
    assert active["sync"]["active"] is True
    assert service.output.active_changes[-2:] == [False, True]
    service.close()


@pytest.mark.parametrize("invalid", ["false", 0, 1, None, [], {}])
def test_hot_active_control_rejects_non_boolean_values(tmp_path, invalid):
    service = make_runtime_service(tmp_path)

    with pytest.raises(ValueError, match="active must be a boolean"):
        service.apply_control_request({"active": invalid})

    assert service.desired_active is True
    assert service.output.active is True
    service.close()


class _FakeReturnProcess:
    def __init__(self, started, wrote, terminated):
        self._terminated = False
        self._wrote = wrote
        self._terminated_event = terminated
        self.stdin = self._Sink(self)
        self.stderr = io.BytesIO()
        started.set()

    class _Sink:
        def __init__(self, process):
            self.process = process
            self.closed = False

        def write(self, payload):
            if self.closed or self.process._terminated:
                raise BrokenPipeError("closed")
            self.process._wrote.set()
            return len(payload)

        def close(self):
            self.closed = True

    def poll(self):
        return 0 if self._terminated else None

    def terminate(self):
        self._terminated = True
        self._terminated_event.set()

    def wait(self, timeout=None):
        del timeout
        return 0

    def kill(self):
        self.terminate()


def test_linux_return_inactive_closes_transport_but_keeps_sender_thread_hot(
    monkeypatch,
):
    started = threading.Event()
    wrote = threading.Event()
    terminated = threading.Event()
    process_instances = []

    def fake_popen(*_args, **_kwargs):
        process = _FakeReturnProcess(started, wrote, terminated)
        process_instances.append(process)
        return process

    monkeypatch.setattr(stream_processor.subprocess, "Popen", fake_popen)
    frames = stream_processor.LatestFrame()
    stop = threading.Event()
    output = stream_processor.LinuxSrtOutput(
        "srt://192.168.1.12:10001?mode=caller",
        4,
        4,
        30,
        "1M",
        frames,
        stop,
        encoder="libx264",
        active=False,
    )
    output.start()
    try:
        frames.put(stream_processor.np.zeros((4, 4, 3), dtype="uint8"))
        assert not started.wait(0.15)
        assert output.is_alive()
        assert output.transport_open is False

        output.set_active(True)
        frames.put(stream_processor.np.zeros((4, 4, 3), dtype="uint8"))
        assert started.wait(1.0)
        assert wrote.wait(1.0)
        assert output.transport_open is True

        output.set_active(False)
        assert terminated.wait(1.0)
        assert output.transport_open is False
        assert output.is_alive()
        assert len(process_instances) == 1
    finally:
        stop.set()
        output.set_active(False)
        output.join(timeout=1.0)

    assert not output.is_alive()


def test_local_control_cli_emits_boolean_active_request(monkeypatch, capsys):
    captured = {}

    def fake_request(request, *, socket_path):
        captured.update(request)
        captured["socket_path"] = str(socket_path)
        return {"ok": True}

    monkeypatch.setattr(local_processor_control, "request_processor", fake_request)

    assert (
        local_processor_control.main(
            ["--socket", "/tmp/test-processor.sock", "--active", "false"]
        )
        == 0
    )
    assert captured == {
        "op": "set",
        "active": False,
        "socket_path": "/tmp/test-processor.sock",
    }
    assert '"ok": true' in capsys.readouterr().out


def test_direct_cli_defaults_active_but_system_service_starts_inactive():
    defaults = stream_processor.build_parser().parse_args([])
    assert defaults.active is True
    assert defaults.processing_mode == "face_swap"
    service_unit = (
        ROOT
        / "arch-linux"
        / "systemd"
        / "deep-live-cam-phone-processed.service"
    ).read_text(encoding="utf-8")
    assert (
        "android_stream_processor.py --active false "
        "--processing-mode passthrough"
    ) in service_unit
