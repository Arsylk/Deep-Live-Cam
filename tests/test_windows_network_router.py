from __future__ import annotations

import threading
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windows.modules.device_slots import DeviceRegistry
from windows.modules.network_router import SlotRouter


class FakeWorker:
    instances: list["FakeWorker"] = []

    def __init__(self, url, *_args, **kwargs):
        self.url = url
        self.label = kwargs.get("label", "broadcast")
        self.route_token = kwargs.get("route_token")
        self.encoder = kwargs.get("encoder") or "fake-h264"
        self.received = 0
        self.sent = 0
        self.last_frame_at = 0.0
        self.started = False
        self.closed = False
        FakeWorker.instances.append(self)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def is_alive(self):
        return self.started and not self.closed

    def join(self, timeout=None):
        del timeout


class FakeProcessor:
    def __init__(self, *_args, **_kwargs):
        self.started = False
        self.processed = 0
        self.last_frame_at = 0.0

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started

    def join(self, timeout=None):
        del timeout


def build_router(tmp_path):
    FakeWorker.instances.clear()
    return SlotRouter(
        DeviceRegistry(tmp_path),
        width=1280,
        height=720,
        fps=30,
        bitrate="8M",
        latency_us=100_000,
        state_dir=str(tmp_path),
        stop=threading.Event(),
        input_factory=FakeWorker,
        output_factory=FakeWorker,
        broadcast_factory=FakeWorker,
        processor_factory=FakeProcessor,
    )


def test_windows_supervisor_pins_the_processor_owned_model():
    supervisor = (ROOT / "windows" / "run-network-service.cmd").read_text(
        encoding="utf-8"
    )

    assert "--swapper-model inswapper-128" in supervisor


def test_windows_activation_ships_face_swapper_runtime_contract():
    activation = (ROOT / "windows" / "activate-multidevice.ps1").read_text(
        encoding="utf-8"
    )

    assert "'modules\\swapper_contract.py'" in activation


def test_router_starts_only_selected_slot_pair(tmp_path):
    router = build_router(tmp_path)

    router.start()

    snapshot = router.snapshot()
    assert snapshot["selected_device_id"] == "android-phone"
    assert snapshot["selected_slot"] == 0
    assert snapshot["input"]["port"] == 10000
    assert snapshot["return"]["port"] == 10001
    assert "srt://192.168.1.35:10010" in snapshot["selected_stream"]["url"]
    assert "mode=caller" in snapshot["selected_stream"]["url"]
    assert "srt://0.0.0.0:10010" in snapshot["selected_stream"]["listen_url"]
    assert snapshot["broadcast"] is snapshot["selected_stream"]
    broadcast_worker = next(
        worker for worker in FakeWorker.instances if worker.label == "broadcast"
    )
    assert "mode=listener" in broadcast_worker.url
    assert snapshot["input"]["cadence"] == {
        "interval_ms_ema": 0.0,
        "jitter_ms_ema": 0.0,
        "max_interval_ms": 0.0,
        "estimated_drops": 0,
    }
    route_workers = [worker for worker in FakeWorker.instances if worker.label != "broadcast"]
    assert len([worker for worker in route_workers if worker.is_alive()]) == 2


def test_selection_closes_old_network_workers_but_keeps_processor(tmp_path):
    router = build_router(tmp_path)
    router.start()
    processor = router.processor
    old_input, old_output = router.input, router.output

    router.select("arch-webcam")

    snapshot = router.snapshot()
    assert router.processor is processor
    assert processor.is_alive()
    assert old_input.closed is True
    assert old_output.closed is True
    assert snapshot["selected_device_id"] == "arch-webcam"
    assert snapshot["input"]["port"] == 10002
    assert snapshot["return"]["port"] == 10003
    assert router.input.route_token == 1
    active_route_workers = [
        worker
        for worker in FakeWorker.instances
        if worker.label != "broadcast" and worker.is_alive()
    ]
    assert active_route_workers == [router.input, router.output]
