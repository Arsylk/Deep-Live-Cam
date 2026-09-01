"""Single-selection route supervisor for the Windows processing node."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .device_slots import (
    BROADCAST_LISTEN_URL,
    BROADCAST_URL,
    DeviceRegistry,
    DeviceSlot,
)
from .live_stream import (
    FrameFanout,
    LatestFrame,
    SrtInput,
    SrtOutput,
    UdpBroadcastOutput,
)


class SlotRouter:
    """Own exactly one selected input and its paired return transport.

    The processor and selected-stream publisher are long-lived. Selection replaces
    only the old slot's network input/return workers, so camera ownership stays
    entirely on each client and the control API never disappears.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: str,
        latency_us: int,
        state_dir: str,
        stop: threading.Event,
        input_factory: Callable[..., SrtInput] = SrtInput,
        output_factory: Callable[..., SrtOutput] = SrtOutput,
        broadcast_factory: Callable[..., UdpBroadcastOutput] = UdpBroadcastOutput,
        processor_factory: Callable[..., Any] | None = None,
        frame_observer: Callable[..., None] | None = None,
    ) -> None:
        self.registry = registry
        self.width, self.height, self.fps = width, height, fps
        self.bitrate, self.latency_us = bitrate, latency_us
        self.state_dir, self.stop_event = state_dir, stop
        self.input_factory = input_factory
        self.output_factory = output_factory
        self.broadcast_factory = broadcast_factory
        if processor_factory is None:
            from .live_processor import LiveProcessor

            processor_factory = LiveProcessor
        self.processor_factory = processor_factory
        self.frame_observer = frame_observer

        self.input_frames = LatestFrame()
        self.return_frames = LatestFrame()
        self.broadcast_frames = LatestFrame()
        self._route_token = registry.generation
        self.fanout = FrameFanout(
            {"return": self.return_frames, "broadcast": self.broadcast_frames},
            self.route_token,
        )
        self.processor: Any | None = None
        self.broadcast: UdpBroadcastOutput | None = None
        self.input: SrtInput | None = None
        self.output: SrtOutput | None = None
        self.active_slot: DeviceSlot | None = None
        self.route_stop: threading.Event | None = None
        self.switching = False
        self.last_switch_at = 0.0
        self.last_switch_error: str | None = None
        self._lock = threading.RLock()
        self._selection_lock = threading.Lock()

    def route_token(self) -> int:
        return self._route_token

    def start(self) -> None:
        with self._selection_lock:
            if self.processor is not None:
                return
            self.processor = self.processor_factory(
                self.input_frames,
                self.fanout,
                self.stop_event,
                self.fps,
                self.state_dir,
            )
            if self.frame_observer is not None:
                self.processor.frame_observer = self.frame_observer
            self.broadcast = self.broadcast_factory(
                BROADCAST_LISTEN_URL,
                self.width,
                self.height,
                self.fps,
                self.bitrate,
                self.broadcast_frames,
                self.stop_event,
                stale_seconds=1.0,
            )
            self.processor.start()
            self.broadcast.start()
            self._activate(self.registry.selected(), self.registry.generation)

    def select(self, device_id: str) -> dict[str, Any]:
        """Persist and activate a configured slot without touching its camera."""
        with self._selection_lock:
            requested = self.registry.resolve(device_id)
            if self.active_slot and self.active_slot.device_id == requested.device_id:
                return self.snapshot()
            selected, _changed = self.registry.select(device_id)
            try:
                self._activate(selected, self.registry.generation)
                self.last_switch_error = None
            except Exception as exc:
                self.last_switch_error = f"{type(exc).__name__}: {exc}"
                raise
            return self.snapshot()

    def _activate(self, slot: DeviceSlot, token: int) -> None:
        # Constructors are deliberately cheap when an encoder is supplied.
        # Prepare the new objects before tearing down the current route.
        route_stop = threading.Event()
        new_input = self.input_factory(
            slot.input_url(self.latency_us),
            self.width,
            self.height,
            self.input_frames,
            route_stop,
            label=f"slot-{slot.slot}-{slot.device_id}",
            route_token=token,
            expected_fps=self.fps,
        )
        selected_encoder = self.broadcast.encoder if self.broadcast else None
        new_output = self.output_factory(
            slot.return_url(self.latency_us),
            self.width,
            self.height,
            self.fps,
            self.bitrate,
            self.return_frames,
            route_stop,
            label=f"return-slot-{slot.slot}",
            stale_seconds=1.0,
            encoder=selected_encoder,
        )

        self.switching = True
        # Invalidate old in-flight processing before closing its workers.
        self._route_token = token
        self._stop_active_route()
        self.input_frames.clear()
        self.fanout.clear()
        if self.broadcast is not None:
            # Flush FFmpeg's encoded old-route buffers.  Its thread remains
            # alive and opens a fresh selected-stream encoder on the next frame.
            self.broadcast.close()
        with self._lock:
            self.route_stop = route_stop
            self.input = new_input
            self.output = new_output
            self.active_slot = slot
            self.last_switch_at = time.time()
        new_input.start()
        new_output.start()
        self.switching = False
        print(
            f"[router] selected {slot.device_id} slot={slot.slot} "
            f"input=UDP/{slot.input_port} return={slot.return_host}:{slot.return_port}",
            flush=True,
        )

    def _stop_active_route(self) -> None:
        with self._lock:
            stop = self.route_stop
            inp, out = self.input, self.output
            self.route_stop = None
            self.input = None
            self.output = None
            self.active_slot = None
        if stop is not None:
            stop.set()
        if inp is not None:
            inp.close()
        if out is not None:
            out.close()
        for worker in (inp, out):
            if worker is not None and worker.is_alive():
                worker.join(timeout=3.0)

    @staticmethod
    def _worker_health(worker: Any, now: float, counter: str) -> dict[str, Any]:
        last = float(getattr(worker, "last_frame_at", 0.0) or 0.0) if worker else 0.0
        age = None if not last else now - last
        health = {
            "frames": int(getattr(worker, counter, 0) or 0) if worker else 0,
            "last_frame_age": age,
            "streaming": age is not None and age < 2.5,
            "worker_alive": bool(worker and worker.is_alive()),
        }
        if worker is not None:
            health["cadence"] = {
                "interval_ms_ema": round(
                    float(getattr(worker, "interval_ms_ema", 0.0) or 0.0), 3
                ),
                "jitter_ms_ema": round(
                    float(getattr(worker, "jitter_ms_ema", 0.0) or 0.0), 3
                ),
                "max_interval_ms": round(
                    float(getattr(worker, "max_interval_ms", 0.0) or 0.0), 3
                ),
                "estimated_drops": int(
                    getattr(worker, "estimated_drops", 0) or 0
                ),
            }
            frame_queue = getattr(worker, "frames", None)
            published = max(
                0,
                int(getattr(frame_queue, "published", 0) or 0)
                - int(getattr(worker, "_queue_published_start", 0) or 0),
            )
            overwritten = max(
                0,
                int(getattr(frame_queue, "overwritten", 0) or 0)
                - int(getattr(worker, "_queue_overwritten_start", 0) or 0),
            )
            health["queue"] = {
                "published": published,
                "overwrites": overwritten,
                "overwrite_percent": round(
                    100.0 * overwritten / max(1, published), 3
                ),
            }
            if hasattr(worker, "connections"):
                health["transport"] = {
                    "connections": int(getattr(worker, "connections", 0) or 0),
                    "disconnects": int(getattr(worker, "disconnects", 0) or 0),
                }
            if hasattr(worker, "repeated_frames"):
                sent = int(getattr(worker, "sent", 0) or 0)
                repeated = int(getattr(worker, "repeated_frames", 0) or 0)
                health["cadence"].update(
                    {
                        "source_frames": int(
                            getattr(worker, "source_frames", 0) or 0
                        ),
                        "repeated_frames": repeated,
                        "repeat_percent": round(
                            100.0 * repeated / max(1, sent), 3
                        ),
                    }
                )
        return health

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            slot, inp, out = self.active_slot, self.input, self.output
        input_health = self._worker_health(inp, now, "received")
        return_health = self._worker_health(out, now, "sent")
        broadcast_health = self._worker_health(self.broadcast, now, "sent")
        if slot:
            input_health.update(
                {"url": slot.input_url(self.latency_us), "port": slot.input_port}
            )
            return_health.update(
                {
                    "url": slot.return_url(self.latency_us),
                    "host": slot.return_host,
                    "port": slot.return_port,
                    "encoder": getattr(out, "encoder", None),
                }
            )
        broadcast_health.update(
            {
                "url": BROADCAST_URL,
                "listen_url": BROADCAST_LISTEN_URL,
                "host": "192.168.1.35",
                "port": 10_010,
                "encoder": getattr(self.broadcast, "encoder", None),
            }
        )
        result = {
            "selected_device_id": slot.device_id if slot else None,
            "selected_slot": slot.slot if slot else None,
            "generation": self._route_token,
            "switching": self.switching,
            "last_switch_at": self.last_switch_at or None,
            "last_switch_error": self.last_switch_error,
            "input": input_health,
            "return": return_health,
            "selected_stream": broadcast_health,
            "broadcast": broadcast_health,
            "fanout": {
                "published": self.fanout.published,
                "dropped_old_route": self.fanout.dropped_old_route,
            },
        }
        return result

    def recycle_input(self) -> None:
        with self._lock:
            worker = self.input
        if worker is not None:
            worker.close()

    def recycle_return(self) -> None:
        with self._lock:
            worker = self.output
        if worker is not None:
            worker.close()

    def close(self) -> None:
        self.stop_event.set()
        with self._selection_lock:
            self._route_token += 1
            self._stop_active_route()
            if self.broadcast is not None:
                self.broadcast.close()
            for worker in (self.broadcast, self.processor):
                if worker is not None and worker.is_alive():
                    worker.join(timeout=5.0)
