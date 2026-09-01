from __future__ import annotations

import queue
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windows.modules.live_stream import FrameFanout, LatestFrame, _record_cadence


def test_fanout_gives_every_consumer_the_same_frame():
    route_token = 4
    returned = LatestFrame()
    broadcast = LatestFrame()
    fanout = FrameFanout(
        {"return": returned, "broadcast": broadcast}, lambda: route_token
    )
    frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))

    fanout.put(frame, route_token=4)

    assert returned.get(timeout=0.01) is frame
    assert broadcast.get(timeout=0.01) is frame
    assert fanout.published == 1


def test_late_old_route_result_is_not_returned_or_broadcast():
    current = {"token": 8}
    returned = LatestFrame()
    broadcast = LatestFrame()
    fanout = FrameFanout(
        {"return": returned, "broadcast": broadcast},
        lambda: current["token"],
    )

    fanout.put(np.zeros((1, 1, 3), np.uint8), route_token=7)

    for frames in (returned, broadcast):
        try:
            frames.get(timeout=0.001)
        except queue.Empty:
            pass
        else:
            raise AssertionError("old-route frame leaked to an output")
    assert fanout.dropped_old_route == 1


def test_latest_frame_counts_overwrites_without_building_latency():
    frames = LatestFrame()
    first = np.zeros((1, 1, 3), np.uint8)
    second = np.ones((1, 1, 3), np.uint8)

    frames.put(first)
    frames.put(second)

    assert frames.get(timeout=0.01) is second
    assert frames.published == 2
    assert frames.overwritten == 1


def test_cadence_measurement_tracks_jitter_and_missing_frames():
    class Worker:
        _last_frame_monotonic = 0.0
        interval_ms_ema = 0.0
        jitter_ms_ema = 0.0
        max_interval_ms = 0.0
        estimated_drops = 0

    worker = Worker()
    _record_cadence(worker, 10.0, 30)
    _record_cadence(worker, 10.033, 30)
    _record_cadence(worker, 10.133, 30)

    assert 35.0 < worker.interval_ms_ema < 40.0
    assert worker.jitter_ms_ema > 3.0
    assert worker.max_interval_ms == pytest.approx(100.0)
    assert worker.estimated_drops == 2
