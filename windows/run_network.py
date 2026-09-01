#!/usr/bin/env python3
"""Five-slot headless Deep-Live-Cam router and processing service."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import threading
import time
from pathlib import Path

# Keep CPU-side image preparation responsive on laptop-class CPUs. ONNX and
# OpenCV otherwise each create a full-size thread pool, which caused >700% CPU
# use and starved the native control API during continuous inference.
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("DLC_CPU_THREADS", "4"))
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import psutil

# Reuse run.py's Windows CUDA DLL registration before importing ONNX modules.
import run as _bootstrap  # noqa: F401

import modules.globals
from modules.pipeline_benchmark import PairedBenchmarkRecorder
from modules.device_slots import (
    BROADCAST_LISTEN_URL,
    BROADCAST_PORT,
    BROADCAST_URL,
    DeviceRegistry,
)
from modules.network_router import SlotRouter
from modules.remote_control import ControlState, serve


class NetworkService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started_at = time.time()
        self.state = "starting"
        self.stop = threading.Event()

        modules.globals.execution_providers = [
            *(
                ["TensorrtExecutionProvider"]
                if args.execution_provider == "tensorrt"
                else []
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        modules.globals.execution_threads = args.cpu_threads
        modules.globals.swapper_model = args.swapper_model
        modules.globals.swapper_backend = "ort"
        try:
            import cv2

            cv2.setNumThreads(args.cpu_threads)
        except (ImportError, AttributeError):
            pass
        modules.globals.frame_processors = ["face_swapper"]
        modules.globals.headless = True
        modules.globals.map_faces = False
        modules.globals.max_memory = 16

        self.registry = DeviceRegistry(
            args.state_dir,
            android_host=args.android_host,
            arch_host=args.arch_host,
        )
        self.benchmark = PairedBenchmarkRecorder(
            args.state_dir,
            context_supplier=self._benchmark_context,
            start_callback=self._reset_benchmark_window,
        )
        self.router = SlotRouter(
            self.registry,
            width=args.width,
            height=args.height,
            fps=args.fps,
            bitrate=args.bitrate,
            latency_us=args.latency_us,
            state_dir=args.state_dir,
            stop=self.stop,
            frame_observer=self.benchmark.observe,
        )
        self.control = ControlState(
            args.state_dir,
            self.health,
            self.devices,
            self.select_device,
        )
        self.http = serve(self.control, args.control_host, args.control_port)

    def _reset_benchmark_window(self) -> None:
        processor = self.router.processor
        if processor is None:
            raise RuntimeError("processor is not ready for benchmark capture")
        processor.reset_benchmark_window()

    def _benchmark_context(self) -> dict:
        root = Path(__file__).resolve().parent
        source = Path(self.args.state_dir) / "source.jpg"
        return {
            "schema_version": "1.0",
            "pipeline": "windows-network-live",
            "route": self.router.snapshot(),
            "configuration": self.control.config(),
            "resolution": [self.args.width, self.args.height],
            "delivery_fps": self.args.fps,
            "execution_provider": self.args.execution_provider,
            "source_artifact": str(source) if source.is_file() else None,
            "model_artifacts": [str(root / "models" / "inswapper_128.onnx")],
            "code_artifacts": [
                str(Path(__file__).resolve()),
                str(root / "modules" / "live_processor.py"),
                str(root / "modules" / "quality_pipeline.py"),
                str(root / "modules" / "face_tracking.py"),
                str(root / "modules" / "processors" / "frame" / "face_swapper.py"),
            ],
            "repository_root": str(root),
            "opens_camera_device": False,
        }

    def devices(self) -> dict:
        document = self.registry.snapshot()
        route = self.router.snapshot()
        document["runtime"] = route
        for slot in document["slots"]:
            slot["runtime"] = (
                {
                    "input": route["input"],
                    "return": route["return"],
                }
                if slot["selected"]
                else {
                    "input": {"streaming": False, "worker_alive": False},
                    "return": {"streaming": False, "worker_alive": False},
                }
            )
        return document

    def select_device(self, device_id: str) -> dict:
        self.router.select(device_id)
        return self.devices()

    def health(self) -> dict:
        now = time.time()
        route = self.router.snapshot()
        proc = self.router.processor
        process_age = (
            None if not proc or not proc.last_frame_at else now - proc.last_frame_at
        )
        input_streaming = bool(route["input"].get("streaming"))
        return_streaming = bool(route["return"].get("streaming"))
        selected_streaming = bool(route["selected_stream"].get("streaming"))
        if self.router.switching:
            display_state = "switching-device"
        elif input_streaming and modules.globals.processing_enabled:
            display_state = "streaming-face-swap"
        elif input_streaming:
            display_state = "streaming-passthrough"
        elif self.state == "running":
            display_state = "waiting-for-selected-device"
        else:
            display_state = self.state
        processor_alive = bool(proc and proc.is_alive())
        processing_mode = (
            "face_swap" if modules.globals.processing_enabled else "passthrough"
        )
        output = dict(route["return"])
        return {
            "healthy": (
                self.state not in ("failed", "stopping")
                and (proc is None or processor_alive)
            ),
            "streaming": input_streaming and (
                return_streaming or selected_streaming
            ),
            "state": display_state,
            "uptime_seconds": round(now - self.started_at, 1),
            "source_configured": bool(modules.globals.source_path),
            "selected_device_id": route["selected_device_id"],
            "selected_slot": route["selected_slot"],
            "route_generation": route["generation"],
            "switching": route["switching"],
            "input": route["input"],
            "processing": {
                "mode": processing_mode,
                "swapper_model": modules.globals.swapper_model,
                "swapper_backend": modules.globals.swapper_backend,
                "active_swapper_model": modules.globals.active_swapper_model,
                "active_swapper_backend": modules.globals.active_swapper_backend,
                "active_swapper_resolution": modules.globals.active_swapper_resolution,
                "frames": proc.processed if proc else 0,
                "fps": proc.actual_fps if proc else 0.0,
                "last_frame_age": process_age,
                "worker_alive": processor_alive,
                "last_error": proc.last_error if proc else None,
                "timings_ms": (
                    {
                        name: round(value, 3)
                        for name, value in proc.timings_ms.items()
                    }
                    if proc
                    else {}
                ),
                "quality": proc.quality.snapshot() if proc else {},
                "tracking": proc.tracker.snapshot() if proc else {},
                "gpu_models_released": (
                    proc.gpu_models_released if proc else False
                ),
                "enabled": modules.globals.processing_enabled,
                "off_output": modules.globals.processing_off_output,
            },
            "return": route["return"],
            # Kept for clients from the single-route generation.
            "output": output,
            "selected_stream": route["selected_stream"],
            # Compatibility alias for clients from the multicast generation.
            "broadcast": route["selected_stream"],
            "fanout": route["fanout"],
            "last_switch_at": route["last_switch_at"],
            "last_switch_error": route["last_switch_error"],
            "benchmark": {
                **self.benchmark.status(),
                "observer_error": (
                    getattr(proc, "benchmark_last_error", None) if proc else None
                ),
            },
        }

    def start(self) -> None:
        print(
            f"[control] native API http://{self.args.control_host}:"
            f"{self.args.control_port}",
            flush=True,
        )
        print(f"[selected-stream] {BROADCAST_URL}", flush=True)
        self._remove_stale_ffmpeg()
        self.router.start()
        self.state = "running"
        self._watchdog()

    def _remove_stale_ffmpeg(self) -> None:
        """Remove only workers belonging to an earlier router instance."""
        # Match both the current SRT listener and the retired multicast output
        # so an upgrade cannot leave either encoder on the selected-stream
        # port orphaned.
        markers = [
            f"0.0.0.0:{BROADCAST_PORT}",
            f"239.255.77.77:{BROADCAST_PORT}",
            BROADCAST_LISTEN_URL,
        ]
        for slot in self.registry.slots():
            if not slot.configured:
                continue
            markers.extend(
                (
                    slot.input_url(self.args.latency_us),
                    slot.return_url(self.args.latency_us),
                )
            )
        for process in psutil.process_iter(("pid", "name", "cmdline")):
            try:
                if (process.info["name"] or "").lower() != "ffmpeg.exe":
                    continue
                command = " ".join(process.info["cmdline"] or [])
                if not any(marker in command for marker in markers):
                    continue
                print(
                    f"[startup] removing stale router FFmpeg PID {process.pid}",
                    flush=True,
                )
                process.terminate()
                try:
                    process.wait(timeout=2)
                except psutil.TimeoutExpired:
                    process.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

    def _watchdog(self) -> None:
        previous_input = previous_processed = 0
        return_previous_processed = previous_return = 0
        process_stalled_at = return_stalled_at = time.monotonic()
        input_recycled_at = 0.0
        while not self.stop.wait(2.0):
            proc = self.router.processor
            inp = self.router.input
            out = self.router.output
            if proc is None or not proc.is_alive():
                print(
                    "[watchdog] processing worker exited; requesting full restart",
                    flush=True,
                )
                os._exit(75)
            if inp and inp.received != previous_input:
                previous_input = inp.received
                if proc.processed != previous_processed:
                    previous_processed = proc.processed
                    process_stalled_at = time.monotonic()
                elif time.monotonic() - process_stalled_at > 300:
                    print(
                        "[watchdog] CUDA processing is hung; requesting full restart",
                        flush=True,
                    )
                    os._exit(75)
            else:
                process_stalled_at = time.monotonic()

            if proc.processed != return_previous_processed:
                return_previous_processed = proc.processed
                if out and out.sent != previous_return:
                    previous_return = out.sent
                    return_stalled_at = time.monotonic()
                elif time.monotonic() - return_stalled_at > 20:
                    print(
                        "[watchdog] selected return is hung; recycling it",
                        flush=True,
                    )
                    self.router.recycle_return()
                    return_stalled_at = time.monotonic()
            else:
                return_stalled_at = time.monotonic()

            if (
                inp
                and inp.last_frame_at
                and time.time() - inp.last_frame_at > 20
                and time.monotonic() - input_recycled_at > 20
            ):
                print(
                    "[watchdog] selected input is stale; recycling its listener",
                    flush=True,
                )
                self.router.recycle_input()
                input_recycled_at = time.monotonic()

    def close(self) -> None:
        if self.stop.is_set():
            return
        self.state = "stopping"
        self.router.close()
        self.benchmark.close()
        self.http.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="10M")
    parser.add_argument(
        "--execution-provider",
        choices=("cuda", "tensorrt"),
        default="cuda",
        help=(
            "CUDA is the low-startup, route-switch-safe default. TensorRT is "
            "an opt-in benchmark backend and uses the persistent engine cache."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=max(1, int(os.environ.get("DLC_CPU_THREADS", "4"))),
    )
    parser.add_argument(
        "--swapper-model",
        choices=("auto", "inswapper-128", "instyle-256", "simswap-512", "native-256"),
        default="auto",
        help=(
            "Select a local swap model. Explicit native-256 permits development "
            "bundles; auto selects it only after qualification."
        ),
    )
    parser.add_argument("--latency-us", type=int, default=100_000)
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=8090)
    parser.add_argument("--android-host", default="192.168.1.12")
    parser.add_argument("--arch-host", default="192.168.1.11")
    parser.add_argument(
        "--state-dir",
        default=str(Path(__file__).parent / "runtime" / "network-live"),
    )
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is not available on PATH")
    if args.width < 64 or args.height < 64 or not 1 <= args.fps <= 120:
        parser.error("invalid width, height, or FPS")
    if not 1 <= args.cpu_threads <= 32:
        parser.error("--cpu-threads must be between 1 and 32")
    if args.latency_us < 20_000:
        parser.error("--latency-us must be at least 20000")
    return args


def main() -> int:
    args = parse_args()
    service = NetworkService(args)
    signal.signal(signal.SIGINT, lambda *_: service.close())
    signal.signal(signal.SIGTERM, lambda *_: service.close())
    try:
        service.start()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        service.state = "failed"
        print(f"[service] fatal error: {exc}", flush=True)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
