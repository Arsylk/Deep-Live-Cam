"""Reusable latest-frame Deep-Live-Cam processor for network live mode."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import time
import traceback
import gc
from typing import Any, Callable

import cv2
import numpy as np

import modules.globals
from modules import imread_unicode
from modules.face_analyser import detect_many_faces_fast, ensure_landmarks, get_one_face
from modules.face_tracking import TemporalFaceTracker, TrackingResult
from modules.gpu_processing import gpu_flip
from .live_stream import FrameFanout, LatestFrame, RoutedFrame
from modules.processors.frame.core import get_frame_processors_modules
from modules.quality_pipeline import QualityPipeline


class LiveProcessor(threading.Thread):
    def __init__(self, source: LatestFrame,
                 destination: LatestFrame | FrameFanout,
                 stop: threading.Event, input_fps: int,
                 quality_state_dir: str | None = None) -> None:
        super().__init__(name="live-processor", daemon=True)
        self.source, self.destination = source, destination
        self.stop_event, self.input_fps = stop, input_fps
        self.processed = 0
        self.actual_fps = 0.0
        self.last_frame_at = 0.0
        self.last_error: str | None = None
        # A benchmark recorder is a passive observer, never a second consumer
        # of ``source``. Services may attach it after construction so existing
        # factories and route switching remain unchanged.
        self.frame_observer: Callable[
            [np.ndarray, np.ndarray, dict[str, Any]], None
        ] | None = None
        self.benchmark_last_error: str | None = None
        self.gpu_models_released = False
        self.quality = QualityPipeline(quality_state_dir)
        self.tracker = TemporalFaceTracker(input_fps)
        self.model_idle_seconds = max(
            30.0, float(os.environ.get("DLC_MODEL_IDLE_SECONDS", "180"))
        )
        self.timings_ms = {
            "detection": 0.0,
            "tracking": 0.0,
            "swap": 0.0,
            "postprocess": 0.0,
            "processors": 0.0,
            "quality": 0.0,
            "total": 0.0,
        }

    def _record_timing(self, name: str, elapsed_seconds: float) -> None:
        """Maintain a low-noise exponential moving average for health stats."""
        value = elapsed_seconds * 1000.0
        previous = self.timings_ms.get(name, 0.0)
        self.timings_ms[name] = value if previous == 0.0 else previous * 0.9 + value * 0.1

    def _observe_pair(
        self,
        reference: np.ndarray,
        processed: np.ndarray,
        *,
        route_token: int | None,
        processing_active: bool,
        tracking: dict[str, Any] | None = None,
    ) -> None:
        observer = self.frame_observer
        if observer is None:
            return
        try:
            # Quality metrics remain at the root so the existing
            # ``stability_report.py`` can compare a captured run directly.
            metadata = self.quality.snapshot()
            metadata["benchmark"] = {
                "processor_frame": self.processed,
                "route_token": route_token,
                "processing_active": processing_active,
                "processing_fps": round(self.actual_fps, 6),
                "timings_ms": {
                    name: round(value, 6)
                    for name, value in self.timings_ms.items()
                },
                "tracking": dict(tracking or {}),
            }
            observer(reference, processed, metadata)
            self.benchmark_last_error = None
        except Exception as exc:
            # Recording is diagnostic and must never interrupt the camera.
            self.benchmark_last_error = f"{type(exc).__name__}: {exc}"

    def reset_benchmark_window(self) -> None:
        """Reset diagnostic aggregates without touching active tracking."""
        self.quality.reset_metrics_window()
        self.tracker.reset_metrics_window()

    def _release_gpu_models(self) -> None:
        """Release optional live ONNX sessions when entering standby."""
        if self.gpu_models_released:
            return
        try:
            import modules.face_analyser as analyser
            import modules.processors.frame.face_swapper as swapper
            analyser.FACE_ANALYSER = None
            current_swapper = swapper.FACE_SWAPPER
            close_swapper = getattr(current_swapper, "close", None)
            if callable(close_swapper):
                close_swapper()
            swapper.FACE_SWAPPER = None
            swapper.FACE_SWAPPER_KEY = None
            modules.globals.active_swapper_model = "not-loaded"
            modules.globals.active_swapper_backend = "not-loaded"
            modules.globals.active_swapper_resolution = 0
            swapper.PREVIOUS_FRAME_RESULT = None
            swapper.FACE_DETECTION_CACHE.clear()
            graph = getattr(swapper, "_cuda_graph_session", None)
            if isinstance(graph, dict):
                for key in ("session", "io_binding", "ort_input", "ort_latent"):
                    graph[key] = None
                graph["recorded"] = False
            reset_temporal_state = getattr(swapper, "reset_temporal_state", None)
            if callable(reset_temporal_state):
                reset_temporal_state()
            self.tracker.reset(reset_counters=False)
            for module_name, attribute in (
                ("modules.processors.frame.face_enhancer", "FACE_ENHANCER"),
                ("modules.processors.frame.face_enhancer_gpen256", "ENHANCER"),
                ("modules.processors.frame.face_enhancer_gpen512", "ENHANCER"),
            ):
                try:
                    module = __import__(module_name, fromlist=[attribute])
                    setattr(module, attribute, None)
                except ImportError:
                    pass
            gc.collect()
            self.gpu_models_released = True
            print("[processor] standby: CUDA inference models released", flush=True)
        except Exception as exc:
            self.last_error = f"Standby release warning: {exc}"
            print(f"[processor] {self.last_error}", flush=True)

    def run(self) -> None:
        try:
            self._run_pipeline()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[processor] fatal worker error: {self.last_error}", flush=True)
            traceback.print_exc()

    def _run_pipeline(self) -> None:
        source_face = None
        source_path = None
        source_stamp = None
        previous_route_token = None
        previous_many_faces = bool(modules.globals.many_faces)
        processors_key = None
        processors = []
        count, started = 0, time.perf_counter()
        black_frame = None
        last_input_at = time.monotonic()

        while not self.stop_event.is_set():
            try:
                packet = self.source.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_input_at >= self.model_idle_seconds:
                    self._release_gpu_models()
                continue
            route_token = None
            if isinstance(packet, RoutedFrame):
                frame = packet.frame
                route_token = packet.route_token
            elif isinstance(packet, np.ndarray):
                frame = packet
            else:
                self.last_error = "Dropped an invalid routed frame payload"
                continue
            frame_started = time.perf_counter()
            last_input_at = time.monotonic()
            if route_token != previous_route_token:
                self.tracker.reset(reset_counters=False)
                previous_route_token = route_token
            if modules.globals.live_mirror:
                frame = gpu_flip(frame, 1)
            self.quality.capture_source(frame)

            if not modules.globals.processing_enabled:
                self._release_gpu_models()
                source_face = source_path = source_stamp = None
                self.tracker.reset(reset_counters=False)
                if modules.globals.processing_off_output == "black":
                    if black_frame is None or black_frame.shape != frame.shape:
                        black_frame = np.zeros_like(frame)
                    result = black_frame
                else:
                    result = frame
                quality_started = time.perf_counter()
                result = self.quality.process(
                    result, fallback=frame, processing_active=False
                )
                self._record_timing(
                    "quality", time.perf_counter() - quality_started
                )
                count += 1
                self.processed += 1
                elapsed = time.perf_counter() - started
                if elapsed >= 1.0:
                    self.actual_fps = count / elapsed
                    count, started = 0, time.perf_counter()
                self.destination.put(result, route_token=route_token)
                self.last_frame_at = time.time()
                frame_elapsed = time.perf_counter() - frame_started
                self._record_timing("total", frame_elapsed)
                self.quality.record_pipeline_timing(frame_elapsed)
                self._observe_pair(
                    frame,
                    result,
                    route_token=route_token,
                    processing_active=False,
                )
                continue

            self.gpu_models_released = False

            current_path = modules.globals.source_path
            current_stamp = None
            if current_path:
                try:
                    source_status = Path(current_path).stat()
                    current_stamp = (
                        current_path,
                        source_status.st_mtime_ns,
                        source_status.st_size,
                    )
                except OSError:
                    current_stamp = (current_path, None, None)
            if current_path and current_stamp != source_stamp:
                candidate = get_one_face(imread_unicode(current_path))
                if candidate is not None:
                    source_face, source_path, source_stamp = (
                        candidate,
                        current_path,
                        current_stamp,
                    )
                    try:
                        import modules.processors.frame.face_swapper as swapper
                        reset_temporal_state = getattr(
                            swapper, "reset_temporal_state", None
                        )
                        if callable(reset_temporal_state):
                            reset_temporal_state()
                    except ImportError:
                        pass
                    print(f"[processor] source face loaded: {current_path}", flush=True)

            many_faces = bool(modules.globals.many_faces)
            if many_faces != previous_many_faces:
                self.tracker.reset(reset_counters=False)
                previous_many_faces = many_faces

            tracking_enabled = bool(
                getattr(modules.globals, "tracking_enabled", True)
            )
            detection_interval = max(
                1, int(getattr(modules.globals, "detection_interval", 1))
            )
            detection_ran = (
                not tracking_enabled
                or self.tracker.should_detect(
                    detection_interval, many_faces=many_faces
                )
            )
            detections = None
            if detection_ran:
                detection_started = time.perf_counter()
                detections = detect_many_faces_fast(frame) or []
                # Landmark inference is tied to fresh detections.  On skipped
                # or missed detector frames the tracker propagates all points
                # with the same affine motion as the five alignment points.
                if modules.globals.mouth_mask and detections:
                    ensure_landmarks(frame, detections)
                self._record_timing(
                    "detection", time.perf_counter() - detection_started
                )

            tracking_started = time.perf_counter()
            tracking: TrackingResult = self.tracker.update(
                frame,
                detections,
                detection_ran=detection_ran,
                enabled=tracking_enabled,
                smoothing=float(
                    getattr(modules.globals, "tracking_smoothing", 0.65)
                ),
                grace_frames=int(
                    getattr(modules.globals, "tracking_grace_frames", 5)
                ),
                minimum_score=float(
                    getattr(modules.globals, "minimum_detection_score", 0.45)
                ),
                minimum_size=float(
                    getattr(modules.globals, "minimum_face_size", 64)
                ),
                many_faces=many_faces,
            )
            self._record_timing(
                "tracking", time.perf_counter() - tracking_started
            )
            faces = tracking.faces
            cached_many = faces if many_faces else None
            cached_target = tracking.primary if not many_faces else None

            result = frame
            current_processors_key = tuple(modules.globals.frame_processors)
            if current_processors_key != processors_key:
                processors = get_frame_processors_modules(
                    modules.globals.frame_processors
                )
                processors_key = current_processors_key
            processors_started = time.perf_counter()
            for processor in processors:
                if processor.NAME == "DLC.FACE-SWAPPER":
                    swap_started = time.perf_counter()
                    boxes = []
                    paste_regions = []
                    if source_face is not None and cached_many:
                        result = result.copy()
                        for target in cached_many:
                            result = processor.swap_face(
                                source_face,
                                target,
                                result,
                                paste_regions,
                            )
                            if getattr(target, "bbox", None) is not None:
                                boxes.append(target.bbox.astype(int))
                    elif source_face is not None and cached_target is not None:
                        # The optimized paste-back path mutates its destination
                        # array. Preserve `frame` as the genuine camera input
                        # for tracker fades, quality comparison, and fallback;
                        # otherwise output and fallback alias each other and a
                        # pass-through (or a real swap) is falsely measured as
                        # zero pixel change.
                        result = result.copy()
                        result = processor.swap_face(
                            source_face,
                            cached_target,
                            result,
                            paste_regions,
                        )
                        if getattr(cached_target, "bbox", None) is not None:
                            boxes.append(cached_target.bbox.astype(int))
                    if boxes and tracking.swap_alpha < 0.999:
                        # Fade only the pixels changed by the swap (the rest of
                        # the frame is identical), avoiding an original/swap
                        # toggle on a short detector miss.
                        result = cv2.addWeighted(
                            frame,
                            1.0 - tracking.swap_alpha,
                            result,
                            tracking.swap_alpha,
                            0.0,
                        )
                    self._record_timing(
                        "swap", time.perf_counter() - swap_started
                    )
                    postprocess_started = time.perf_counter()
                    result = processor.apply_post_processing(
                        result,
                        boxes,
                        motion_matrix=tracking.motion_matrix,
                        reference_frame=frame,
                        paste_regions=paste_regions,
                        paste_alpha_scale=tracking.swap_alpha,
                    )
                    self._record_timing(
                        "postprocess",
                        time.perf_counter() - postprocess_started,
                    )
                elif processor.NAME.startswith("DLC.FACE-ENHANCER"):
                    key = processor.__name__.split(".")[-1]
                    if modules.globals.fp_ui.get(key, False):
                        try:
                            result = processor.process_frame(
                                None, result, detected_faces=faces
                            )
                        except Exception as exc:
                            # Enhancement is optional. A missing/broken model
                            # must never take down the face-swap stream.
                            modules.globals.fp_ui[key] = False
                            self.last_error = (
                                f"Disabled {key}: {type(exc).__name__}: {exc}"
                            )
                            print(f"[processor] {self.last_error}", flush=True)
                else:
                    result = processor.process_frame(source_face, result)
            self._record_timing(
                "processors", time.perf_counter() - processors_started
            )

            count += 1
            self.processed += 1
            elapsed = time.perf_counter() - started
            if elapsed >= 1.0:
                self.actual_fps = count / elapsed
                count, started = 0, time.perf_counter()
            if modules.globals.show_fps:
                cv2.putText(result, f"FPS: {self.actual_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            quality_started = time.perf_counter()
            result = self.quality.process(
                result,
                fallback=frame,
                processing_active=True,
                face_bbox=(
                    getattr(tracking.primary, "bbox", None)
                    if tracking.primary is not None
                    else None
                ),
                tracking=self.tracker.snapshot(),
                swap_applied=bool(source_face is not None and faces),
            )
            self._record_timing(
                "quality", time.perf_counter() - quality_started
            )
            self.destination.put(result, route_token=route_token)
            self.last_frame_at = time.time()
            frame_elapsed = time.perf_counter() - frame_started
            self._record_timing("total", frame_elapsed)
            self.quality.record_pipeline_timing(frame_elapsed)
            self._observe_pair(
                frame,
                result,
                route_token=route_token,
                processing_active=True,
                tracking=self.tracker.snapshot(),
            )
