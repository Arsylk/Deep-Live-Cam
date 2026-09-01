from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
MODULE_PATH = BIN / "tester.py"
sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location("arch_tester_preview", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tester = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tester)


def test_phone_return_preview_decoder_is_local_and_camera_free():
    command = tester.local_mpegts_preview_command(11004)
    rendered = " ".join(command)

    assert "udp://127.0.0.1:11004?" in rendered
    assert command.count("-i") == 1
    assert "/dev/video" not in rendered
    assert "srt://" not in rendered
    assert "192.168.1." not in rendered
    assert ":11003" not in rendered


def test_phone_return_preview_health_requires_fresh_exact_tee():
    health = {
        "state": "running",
        "return": {
            "preview_url": "udp://127.0.0.1:11004?pkt_size=1316",
            "worker_alive": True,
            "last_frame_age": 0.02,
        },
    }

    assert tester.android_native_preview_fresh(health, health_age=0.1)
    assert not tester.android_native_preview_fresh(health, health_age=30.0)
    health["return"]["last_frame_age"] = 3.0
    assert not tester.android_native_preview_fresh(health, health_age=0.1)


def test_phone_return_preview_never_substitutes_windows_preview():
    health = {
        "state": "running",
        "return": {
            "preview_url": "udp://127.0.0.1:11003?pkt_size=1316",
            "worker_alive": True,
            "last_frame_age": 0.01,
        },
    }

    assert not tester.android_native_preview_fresh(health, health_age=0.1)


def test_main_preview_switches_between_windows_and_local_receiver_relays():
    class FakeDecoder:
        def __init__(self):
            self.running = True
            self.command: list[str] = []
            self.starts = 0

        def set_command(self, command: list[str]) -> None:
            self.running = False
            self.command = list(command)

        def start(self) -> None:
            self.running = True
            self.starts += 1

    class FakePane:
        def __init__(self):
            self.heading: tuple[str, str] | None = None
            self.waiting: str | None = None

        def set_heading(self, title: str, detail: str) -> None:
            self.heading = (title, detail)

        def clear_image(self, detail: str) -> None:
            self.waiting = detail

    decoder = FakeDecoder()
    pane = FakePane()
    window = SimpleNamespace(
        local_processed_preview_port=11007,
        output_preview_port=11003,
        active_output_preview_port=11003,
        output_decoder=decoder,
        output_pane=pane,
        windows_host="192.168.1.35",
        selected_stream_port=10010,
    )
    window._output_command = MethodType(
        tester.TesterWindow._output_command, window
    )

    tester.TesterWindow._set_output_preview_source(
        window, local_processed=True
    )

    assert window.active_output_preview_port == 11007
    assert "udp://127.0.0.1:11007?" in " ".join(decoder.command)
    assert pane.heading is not None
    assert pane.heading[0] == "LOCAL NATIVE-256 OUTPUT"
    assert pane.waiting == "WAITING FOR LOCAL NATIVE-256"
    assert decoder.running is True
    assert decoder.starts == 1

    tester.TesterWindow._set_output_preview_source(
        window, local_processed=False
    )

    assert window.active_output_preview_port == 11003
    assert "udp://127.0.0.1:11003?" in " ".join(decoder.command)
    assert pane.heading is not None
    assert pane.heading[0] == "SELECTED WINDOWS STREAM"
    assert pane.waiting == "WAITING FOR SELECTED WINDOWS STREAM"
    assert decoder.running is True
    assert decoder.starts == 2


def test_phone_front_native_route_has_an_explicit_ui_identity():
    health = {
        "state": "running",
        "route": "android-camera-processed-to-android",
        "processing": {"model": "dlc_swap256m"},
        "return": {
            "preview_url": "udp://127.0.0.1:11004?pkt_size=1316",
            "worker_alive": True,
            "last_frame_age": 0.02,
        },
    }

    assert tester.android_native_phone_route_fresh(health, health_age=0.1)
    assert (
        tester.android_native_route_title(health)
        == "PHONE FRONT → ARCH NATIVE 256 → CAMERA2 120"
    )


def test_phone_front_production_route_has_an_explicit_ui_identity():
    health = {
        "state": "running",
        "route": "android-camera-processed-to-android",
        "processing": {"model": "inswapper-128"},
        "return": {
            "preview_url": "udp://127.0.0.1:11004?pkt_size=1316",
            "worker_alive": True,
            "last_frame_age": 0.02,
        },
    }

    assert tester.android_native_phone_route_fresh(health, health_age=0.1)
    assert (
        tester.android_native_route_title(health)
        == "PHONE FRONT → ARCH INSWAPPER 128 → CAMERA2 120"
    )


def test_webcam_production_route_has_an_explicit_ui_identity():
    health = {
        "state": "running",
        "route": "arch-webcam-processed-to-android",
        "processing": {"model": "inswapper-128"},
        "return": {
            "preview_url": "udp://127.0.0.1:11004?pkt_size=1316",
            "worker_alive": True,
            "last_frame_age": 0.02,
        },
    }

    assert (
        tester.android_native_route_title(health)
        == "ARCH WEBCAM → INSWAPPER 128 → PHONE"
    )
