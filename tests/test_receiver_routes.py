from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import receiver  # noqa: E402
import configure_receiver_output  # noqa: E402
import select_receiver_source  # noqa: E402


def configure_small_receiver(
    monkeypatch,
    *,
    manager_output_port: str = "11003",
    local_processed_port: str = "11006",
    local_processed_preview_port: str = "11007",
    receiver_source: str = "auto",
    source_state_file: str | None = None,
):
    monkeypatch.setenv("VIDEO_WIDTH", "2")
    monkeypatch.setenv("VIDEO_HEIGHT", "2")
    monkeypatch.setenv("VIDEO_FPS", "1")
    monkeypatch.setenv("VIRTUAL_CAMERAS", "/dev/video-test")
    monkeypatch.setenv("DEVICE_SLOT", "1")
    monkeypatch.setenv("WINDOWS_RETURN_PORT", "10003")
    monkeypatch.setenv("WINDOWS_HOST", "192.168.1.35")
    monkeypatch.setenv("WINDOWS_SELECTED_STREAM_PORT", "10010")
    monkeypatch.setenv("ARCH_HOST", "192.168.1.11")
    monkeypatch.setenv("LOCAL_PREVIEW_PORT", "11000")
    monkeypatch.setenv("MANAGER_OUTPUT_PREVIEW_PORT", manager_output_port)
    monkeypatch.setenv("LOCAL_PROCESSED_PORT", local_processed_port)
    monkeypatch.setenv(
        "LOCAL_PROCESSED_PREVIEW_PORT", local_processed_preview_port
    )
    monkeypatch.setenv("RECEIVER_SOURCE", receiver_source)
    if source_state_file is None:
        source_state_file = (
            f"/tmp/deep-live-cam-receiver-test-{os.getpid()}-"
            f"{id(monkeypatch)}.json"
        )
        Path(source_state_file).unlink(missing_ok=True)
    monkeypatch.setenv("RECEIVER_SOURCE_STATE_FILE", source_state_file)
    monkeypatch.setattr(receiver.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    return receiver.Receiver()


def publish(worker, name: str, frame: bytes, age: float = 0.0):
    worker.input_stats[name]["frame"] = frame
    worker.input_stats[name]["last_frame_at"] = time.monotonic() - age


def yuyv_frame(luma: np.ndarray, *, u: int = 100, v: int = 150) -> bytes:
    height, width = luma.shape
    assert width % 2 == 0
    packed = np.empty((height, width // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = luma[:, 0::2]
    packed[:, :, 1] = u
    packed[:, :, 2] = luma[:, 1::2]
    packed[:, :, 3] = v
    return packed.tobytes()


def unpack_luma(frame: bytes, width: int, height: int) -> np.ndarray:
    packed = np.frombuffer(frame, dtype=np.uint8).reshape(height, width // 2, 4)
    luma = np.empty((height, width), dtype=np.uint8)
    luma[:, 0::2] = packed[:, :, 0]
    luma[:, 1::2] = packed[:, :, 2]
    return luma


def test_receiver_has_isolated_local_return_broadcast_and_raw_inputs(monkeypatch):
    worker = configure_small_receiver(monkeypatch)

    assert worker.input_priority == (
        "local_processed",
        "local_prerecorded",
        "processed_return",
        "selected_stream",
        "local_raw",
    )
    local = " ".join(worker.decoder_command(worker.input_specs["local_processed"]))
    direct = " ".join(worker.decoder_command(worker.input_specs["processed_return"]))
    broadcast = " ".join(worker.decoder_command(worker.input_specs["selected_stream"]))
    raw = " ".join(worker.decoder_command(worker.input_specs["local_raw"]))
    assert "srt://0.0.0.0:10003?" in direct
    assert "udp://127.0.0.1:11006?" in local
    assert "udp://127.0.0.1:11007?" in local
    assert "-c:v copy" in local
    assert "messageapi=1" in direct
    assert "srt://192.168.1.35:10010?" in broadcast
    assert "mode=caller" in broadcast
    assert "connect_timeout=3000" in broadcast
    assert "udp://127.0.0.1:11003?" in broadcast
    assert "-c:v copy" in broadcast
    assert "udp://127.0.0.1:11000?" in raw


def test_decoder_fits_every_source_with_phone_accurate_cover_crop(monkeypatch):
    # A bare scale=W:H stretches a portrait or off-aspect source into the
    # landscape box. The receiver must instead scale-to-cover then centre-crop
    # (a camera app's centerCrop / SCALER_CROP) so the selected source's real
    # proportions are retained without letterboxing and without renegotiating
    # the locked output geometry.
    worker = configure_small_receiver(monkeypatch)
    for name in worker.input_specs:
        rendered = " ".join(worker.decoder_command(worker.input_specs[name]))
        assert "force_original_aspect_ratio=increase" in rendered
        assert f"crop={worker.width}:{worker.height}" in rendered
        # The old unconditional stretch must be gone from the decode filter.
        assert f"scale={worker.width}:{worker.height}:flags" not in rendered


def test_prerecorded_framing_uses_a_live_commandable_crop_over_black(
    monkeypatch, tmp_path
):
    # A prerecorded clip must be framable: zoom about centre and shift by an
    # offset, with black filling anywhere the source no longer covers the
    # locked box.  The framing is a fixed scale-to-cover + black pad + a
    # live-commandable crop (driven over zmq) + scale back, so offset/zoom
    # edits never restart the decoder.
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(monkeypatch)
    (tmp_path / "prerecorded-adjust.json").write_text(
        '{"offset_x": 3, "offset_y": -2, "zoom": 0.5}', encoding="utf-8"
    )

    graph = worker._prerecorded_filter_complex(2)

    w, h = worker.width, worker.height
    pad = worker.PRERECORDED_PAD
    # Fixed scale-to-cover, black pad margins, a zmq broker, and a named live
    # crop the manager can command without a restart.
    assert f"scale={w}:{h}:force_original_aspect_ratio=increase" in graph
    assert f"pad={w + 2 * pad}:{h + 2 * pad}:{pad}:{pad}:color=black" in graph
    assert "zmq=bind_address=" in graph
    assert "crop@live=" in graph
    assert "split=2[v0][v1]" in graph
    # The initial crop window matches the framing math for these values.
    cw, ch, cx, cy = worker._prerecorded_crop_params(3, -2, 0.5)
    assert f"crop@live=w={cw}:h={ch}:x={cx}:y={cy}" in graph


def test_prerecorded_crop_params_zoom_and_offset_math():
    monkeypatch = None  # not needed; construct via the small receiver helper
    import types

    worker = types.SimpleNamespace(
        width=1280, height=720, PRERECORDED_PAD=1280
    )
    from receiver import Receiver

    crop = Receiver._prerecorded_crop_params
    pad = 1280
    # Neutral: full 1280x720 window at the exact centre of the padded canvas.
    assert crop(worker, 0, 0, 1.0) == (1280, 720, pad, pad)
    # Zoom 2x magnifies: the crop window halves, still centred.
    cw, ch, cx, cy = crop(worker, 0, 0, 2.0)
    assert (cw, ch) == (640, 360)
    assert cx == pad + (1280 - 640) // 2
    assert cy == pad + (720 - 360) // 2
    # Zoom 0.5 shows black: the window is larger than the cover box.
    cw, ch, _cx, _cy = crop(worker, 0, 0, 0.5)
    assert (cw, ch) == (2560, 1440)
    # Positive offset shifts the video right => window moves left by offset/zoom.
    _cw, _ch, cx, _cy = crop(worker, 400, 0, 1.0)
    assert cx == pad + 0 - 400  # (W-cw)/2 == 0 at zoom 1


def test_prerecorded_source_and_adjust_signatures_are_independent(
    monkeypatch, tmp_path
):
    # A video change must restart the decoder; a framing change must not (it is
    # applied live over zmq).  The two signatures are therefore separate.
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(monkeypatch)
    (tmp_path / "prerecorded-source.txt").write_text("/tmp/a.mp4")
    (tmp_path / "prerecorded-adjust.json").write_text('{"zoom": 1.0}')

    src0 = worker._prerecorded_source_signature()
    adj0 = worker._prerecorded_adjust_signature()

    # Framing change: adjust signature moves, source stays.
    (tmp_path / "prerecorded-adjust.json").write_text('{"zoom": 2.0}')
    assert worker._prerecorded_source_signature() == src0
    assert worker._prerecorded_adjust_signature() != adj0

    # Video change: source signature moves.
    (tmp_path / "prerecorded-source.txt").write_text("/tmp/b.mp4")
    assert worker._prerecorded_source_signature() != src0


def test_prerecorded_playback_parsing_and_seek_signature(monkeypatch, tmp_path):
    # Play/pause + seek come from a small JSON document.  A seek change is a
    # deliberate jump (restart at -ss); pause is handled in the decode loop.
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(monkeypatch)
    pb = tmp_path / "prerecorded-playback.json"

    # Absent document defaults to playing from the start.
    assert worker._prerecorded_playback() == (False, None)
    assert worker._prerecorded_playback_seek_signature() is None

    pb.write_text('{"paused": true, "seek": 12.5}')
    paused, seek = worker._prerecorded_playback()
    assert paused is True
    assert seek == 12.5
    assert worker._prerecorded_playback_seek_signature() == 12.5

    # A pause-only change must NOT move the seek signature (no restart).
    pb.write_text('{"paused": false, "seek": 12.5}')
    assert worker._prerecorded_playback_seek_signature() == 12.5


def test_prerecorded_seek_is_applied_as_input_ss(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")  # existence is all _file_relay_command checks
    (tmp_path / "prerecorded-source.txt").write_text(str(video))
    (tmp_path / "prerecorded-playback.json").write_text('{"seek": 7.5}')

    spec = {
        "kind": "file_relay", "host": "127.0.0.1", "port": 11010,
        "preview_port": 11003, "framing_preview_port": 11011,
    }
    command = worker._file_relay_command(spec)
    assert command is not None
    # -ss appears before -i so the seek is an input option (fast + accurate).
    assert "-ss" in command
    assert command.index("-ss") < command.index("-i")
    assert "7.500" in command


def test_prerecorded_loop_is_gapless_via_stream_loop(monkeypatch, tmp_path):
    # A looping clip must loop INSIDE one ffmpeg process (-stream_loop -1) so
    # end-of-file wraps to frame 0 with no process restart and no black flash.
    # once/freeze play through a single time and must NOT add -stream_loop.
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    (tmp_path / "prerecorded-source.txt").write_text(str(video))
    pb = tmp_path / "prerecorded-playback.json"
    spec = {
        "kind": "file_relay", "host": "127.0.0.1", "port": 11010,
        "preview_port": 11003, "framing_preview_port": 11011,
    }

    pb.write_text('{"mode": "loop"}')
    cmd = worker._file_relay_command(spec)
    assert "-stream_loop" in cmd
    # -stream_loop must precede -i (it is an input option).
    assert cmd.index("-stream_loop") < cmd.index("-i")
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"

    for mode in ("once", "freeze"):
        pb.write_text(f'{{"mode": "{mode}"}}')
        assert "-stream_loop" not in worker._file_relay_command(spec), mode

    # Default (no mode key) loops.
    pb.write_text('{}')
    assert worker._prerecorded_mode() == "loop"
    assert "-stream_loop" in worker._file_relay_command(spec)


def test_prerecorded_emits_a_phone_relay_tap_on_11009(monkeypatch, tmp_path):
    # Sending prerecorded video to the phone requires the file_relay to write
    # the local phone-relay port (11009), which the windows_phone_relay reads
    # as its "local" source and SRT-calls to the phone.  The receiver's
    # local_prerecorded spec must carry that port, and the built command must
    # emit an mpegts tap to it.
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(monkeypatch)
    spec = worker.input_specs["local_prerecorded"]
    assert spec.get("phone_relay_port") == 11009

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    (tmp_path / "prerecorded-source.txt").write_text(str(video))
    (tmp_path / "prerecorded-playback.json").write_text('{"mode": "loop"}')
    command = " ".join(worker._file_relay_command(spec))
    # The phone tap lands on 11009 as an mpegts udp output.
    assert "udp://127.0.0.1:11009" in command
    assert "mpegts" in command


def test_packaged_config_declares_the_complete_receiver_route():
    config = (
        ARCH_BIN.parent / "config" / "deep-live-cam-arch.conf"
    ).read_text(encoding="utf-8")

    assert "NATIVE_PROCESSOR_SOURCE_PORT=11005" in config
    assert "LOCAL_PROCESSED_PORT=11006" in config
    assert "LOCAL_PROCESSED_PREVIEW_PORT=11007" in config
    assert "RECEIVER_SOURCE=windows" in config
    assert (
        "RECEIVER_SOURCE_STATE_FILE=/var/lib/deep-live-cam/receiver-source.json"
        in config
    )


def test_receiver_unit_and_installer_own_durable_source_state():
    unit = (
        ARCH_BIN.parent / "systemd" / "deep-live-cam-receiver.service"
    ).read_text(encoding="utf-8")
    installer = (ARCH_BIN.parent / "install.sh").read_text(encoding="utf-8")
    narrow_deploy = (ARCH_BIN / "deploy_live_webcam_stack.py").read_text(
        encoding="utf-8"
    )

    assert "StateDirectory=deep-live-cam" in unit
    assert "StateDirectoryMode=0755" in unit
    assert (
        "RECEIVER_SOURCE_STATE_FILE=/var/lib/deep-live-cam/receiver-source.json"
        in installer
    )
    assert "set_config RECEIVER_SOURCE windows" in installer
    assert '"RECEIVER_SOURCE": "windows"' in narrow_deploy
    assert (
        '"/var/lib/deep-live-cam/receiver-source.json"' in narrow_deploy
    )


def test_manager_output_relay_cannot_share_raw_mux_port(monkeypatch):
    try:
        configure_small_receiver(monkeypatch, manager_output_port="11000")
    except ValueError as exc:
        assert "must be distinct" in str(exc)
    else:
        raise AssertionError("manager output relay reused the raw mux port")


def test_receiver_rejects_out_of_range_and_cross_protocol_port_collisions(
    monkeypatch,
):
    with pytest.raises(
        ValueError, match="LOCAL_PROCESSED_PORT must be at most 65535"
    ):
        configure_small_receiver(monkeypatch, local_processed_port="65536")

    with pytest.raises(
        ValueError,
        match="WINDOWS_RETURN_PORT and all local receiver ports must be distinct",
    ):
        configure_small_receiver(monkeypatch, local_processed_port="10003")


def test_unix_selector_uses_the_receivers_custom_state_directory(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    worker = configure_small_receiver(
        monkeypatch,
        source_state_file=str(tmp_path / "receiver-source.json"),
    )
    worker.start_control_server()
    socket_path = tmp_path / "receiver-control.sock"
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        result = select_receiver_source.select_source("raw", socket_path)
        assert result["ok"] is True
        assert result["source"] == "raw"
        assert result["priority"] == ["local_raw"]
        assert worker.source_mode == "raw"
        assert worker.input_priority == ("local_raw",)
        assert worker.sink is None
    finally:
        worker.stop()
        if worker.control_thread is not None:
            worker.control_thread.join(timeout=2.0)


def test_windows_source_selection_survives_receiver_reconstruction_without_sink_restart(
    monkeypatch, tmp_path
):
    runtime = tmp_path / "runtime"
    intent = tmp_path / "state" / "receiver-source.json"
    monkeypatch.setenv("STATE_DIR", str(runtime))
    worker = configure_small_receiver(
        monkeypatch,
        receiver_source="local",
        source_state_file=str(intent),
    )

    class Sink:
        pid = 4242

        @staticmethod
        def poll():
            return None

    sink = Sink()
    worker.sink = sink
    worker.sink_restarts = 6
    worker.start_control_server()
    deadline = time.monotonic() + 2.0
    while not worker.control_socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        response = select_receiver_source.select_source(
            "windows", worker.control_socket_path
        )
        assert response["source"] == "windows"
        assert response["priority"] == [
            "processed_return",
            "selected_stream",
            "local_raw",
        ]
        assert response["source_persisted"] is True
        assert worker.sink is sink
        assert worker.sink_restarts == 6
        assert json.loads(intent.read_text(encoding="utf-8")) == {
            "schema_version": 1,
            "source": "windows",
        }
    finally:
        # The fake models sink identity only; never pass it to stop_process.
        worker.sink = None
        worker.stop()
        if worker.control_thread is not None:
            worker.control_thread.join(timeout=2.0)

    restarted = configure_small_receiver(
        monkeypatch,
        receiver_source="local",
        source_state_file=str(intent),
    )
    assert restarted.source_mode == "windows"
    assert restarted.input_priority == (
        "processed_return",
        "selected_stream",
        "local_raw",
    )
    assert restarted.source_restore_error is None


def test_missing_durable_source_uses_environment_fallback(monkeypatch, tmp_path):
    worker = configure_small_receiver(
        monkeypatch,
        receiver_source="windows",
        source_state_file=str(tmp_path / "missing" / "receiver-source.json"),
    )

    assert worker.source_mode == "windows"
    assert worker.source_restore_error is None


@pytest.mark.parametrize(
    "payload",
    (
        "{not-json",
        "[]",
        '{"schema_version":2,"source":"windows"}',
        '{"schema_version":1,"source":"unknown"}',
    ),
)
def test_corrupt_durable_source_fails_safe_to_raw(
    monkeypatch, tmp_path, payload
):
    intent = tmp_path / "receiver-source.json"
    intent.write_text(payload, encoding="utf-8")

    worker = configure_small_receiver(
        monkeypatch,
        # Corrupt durable intent must not reactivate this environment default.
        receiver_source="windows",
        source_state_file=str(intent),
    )
    health = worker.state("streaming")

    assert worker.source_mode == "raw"
    assert worker.input_priority == ("local_raw",)
    assert health["source_mode"] == "raw"
    assert "fail-safe raw" in str(health["source_restore_error"])


def test_direct_processed_return_has_highest_priority(monkeypatch):
    worker = configure_small_receiver(monkeypatch)
    raw = b"r" * worker.frame_bytes
    broadcast = b"b" * worker.frame_bytes
    direct = b"d" * worker.frame_bytes
    publish(worker, "local_raw", raw)
    publish(worker, "selected_stream", broadcast)
    publish(worker, "processed_return", direct)

    current, streaming, age, active_input = worker.current_frame()

    assert current == direct
    assert streaming is True
    assert age is not None and age < 0.1
    assert active_input == "processed_return"


def test_local_processed_has_highest_auto_priority(monkeypatch):
    worker = configure_small_receiver(monkeypatch)
    local = b"l" * worker.frame_bytes
    publish(worker, "local_raw", b"r" * worker.frame_bytes)
    publish(worker, "selected_stream", b"b" * worker.frame_bytes)
    publish(worker, "processed_return", b"d" * worker.frame_bytes)
    publish(worker, "local_processed", local)

    current, streaming, _age, active_input = worker.current_frame()

    assert current == local
    assert streaming is True
    assert active_input == "local_processed"


def test_source_modes_switch_cached_frames_without_restarting_sink(monkeypatch):
    worker = configure_small_receiver(monkeypatch)
    publish(worker, "local_raw", b"r" * worker.frame_bytes)
    publish(worker, "selected_stream", b"b" * worker.frame_bytes)
    publish(worker, "processed_return", b"d" * worker.frame_bytes)
    publish(worker, "local_processed", b"l" * worker.frame_bytes)

    for mode, expected in (
        ("local", "local_processed"),
        ("windows", "processed_return"),
        ("raw", "local_raw"),
        ("auto", "local_processed"),
    ):
        with worker.lock:
            worker.source_mode = mode
            worker.input_priority = receiver.SOURCE_PRIORITIES[mode]
        _frame, streaming, _age, active_input = worker.current_frame()
        assert streaming is True
        assert active_input == expected


def test_android_broadcast_becomes_systemwide_when_direct_return_is_absent(monkeypatch):
    worker = configure_small_receiver(monkeypatch)
    raw = b"r" * worker.frame_bytes
    broadcast = b"b" * worker.frame_bytes
    publish(worker, "local_raw", raw)
    publish(worker, "selected_stream", broadcast)

    current, streaming, _age, active_input = worker.current_frame()

    assert current == broadcast
    assert streaming is True
    assert active_input == "selected_stream"


def test_local_raw_fallback_keeps_same_virtual_camera_alive(monkeypatch):
    worker = configure_small_receiver(monkeypatch)
    raw = b"r" * worker.frame_bytes
    publish(worker, "processed_return", b"d" * worker.frame_bytes, age=3.0)
    publish(worker, "selected_stream", b"b" * worker.frame_bytes, age=3.0)
    publish(worker, "local_raw", raw)

    current, streaming, _age, active_input = worker.current_frame()

    assert current == raw
    assert streaming is True
    assert active_input == "local_raw"


def test_slot_port_mismatch_is_rejected(monkeypatch):
    monkeypatch.setenv("VIDEO_WIDTH", "2")
    monkeypatch.setenv("VIDEO_HEIGHT", "2")
    monkeypatch.setenv("VIDEO_FPS", "1")
    monkeypatch.setenv("VIRTUAL_CAMERAS", "/dev/video-test")
    monkeypatch.setenv("DEVICE_SLOT", "1")
    monkeypatch.setenv("WINDOWS_RETURN_PORT", "10001")
    monkeypatch.setattr(receiver.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    try:
        receiver.Receiver()
    except ValueError as exc:
        assert "must be 10003" in str(exc)
    else:
        raise AssertionError("mismatched slot return port was accepted")


def test_yuyv_mirror_and_half_turn_preserve_exact_pixel_and_chroma_layout():
    frame = bytes(
        (
            10, 100, 20, 150, 30, 110, 40, 160,
            50, 120, 60, 170, 70, 130, 80, 180,
        )
    )

    mirrored = receiver.transform_yuyv422(frame, 4, 2, mirror=True)
    half_turn = receiver.transform_yuyv422(frame, 4, 2, rotation=180)
    half_turn_mirrored = receiver.transform_yuyv422(
        frame, 4, 2, mirror=True, rotation=180
    )

    assert mirrored == bytes(
        (
            40, 110, 30, 160, 20, 100, 10, 150,
            80, 130, 70, 180, 60, 120, 50, 170,
        )
    )
    assert half_turn == bytes(
        (
            80, 130, 70, 180, 60, 120, 50, 170,
            40, 110, 30, 160, 20, 100, 10, 150,
        )
    )
    assert half_turn_mirrored == bytes(
        (
            50, 120, 60, 170, 70, 130, 80, 180,
            10, 100, 20, 150, 30, 110, 40, 160,
        )
    )


def test_quarter_turn_is_clockwise_and_mirror_uses_final_output_coordinates():
    luma = np.arange(1, 17, dtype=np.uint8).reshape(4, 4)
    frame = yuyv_frame(luma)

    clockwise = receiver.transform_yuyv422(frame, 4, 4, rotation=90)
    clockwise_mirrored = receiver.transform_yuyv422(
        frame, 4, 4, rotation=90, mirror=True
    )

    expected = np.rot90(luma, k=-1)
    np.testing.assert_array_equal(unpack_luma(clockwise, 4, 4), expected)
    np.testing.assert_array_equal(
        unpack_luma(clockwise_mirrored, 4, 4), expected[:, ::-1]
    )
    assert len(clockwise) == 4 * 4 * 2


def test_quarter_turn_fills_and_center_crops_to_fixed_output_dimensions():
    frame = yuyv_frame(
        np.array(((10, 20, 30, 40), (50, 60, 70, 80)), dtype=np.uint8)
    )

    transformed = receiver.transform_yuyv422(frame, 4, 2, rotation=270)

    assert len(transformed) == len(frame)
    np.testing.assert_array_equal(
        unpack_luma(transformed, 4, 2),
        np.array(((30, 30, 70, 70), (20, 20, 60, 60)), dtype=np.uint8),
    )


def test_output_control_is_hot_persistent_and_does_not_replace_sink(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    source_intent = tmp_path / "receiver-source.json"
    worker = configure_small_receiver(
        monkeypatch, source_state_file=str(source_intent)
    )

    class Sink:
        pid = 4242

        @staticmethod
        def poll():
            return None

    sink = Sink()
    worker.sink = sink
    worker.sink_restarts = 7
    worker.start_control_server()
    socket_path = tmp_path / "receiver-control.sock"
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        result = configure_receiver_output.configure_output(
            mirror=True,
            rotation=270,
            source="raw",
            socket_path=socket_path,
        )
        assert result["ok"] is True
        assert result["source"] == "raw"
        assert result["output_transform"] == {
            "mirror": True,
            "rotation": 270,
            "revision": 1,
        }
        assert result["output_enabled"] is True
        assert result["sink_pid"] == 4242
        assert result["sink_restarts"] == 7
        assert worker.sink is sink
        assert worker.sink_restarts == 7
        assert worker.input_priority == ("local_raw",)
        assert worker.state("streaming")["output_transform"] == {
            "mirror": True,
            "rotation": 270,
            "revision": 1,
        }
        assert worker.state("streaming")["output_enabled"] is True
        persisted = (tmp_path / "receiver-output.json").read_text(encoding="utf-8")
        assert '"enabled": true' in persisted
        assert '"mirror": true' in persisted
        assert '"rotation": 270' in persisted

        live_frame = yuyv_frame(np.arange(4, dtype=np.uint8).reshape(2, 2))
        disabled = configure_receiver_output.configure_output(
            enabled=False,
            socket_path=socket_path,
        )
        assert disabled["output_enabled"] is False
        assert disabled["output_transform"]["revision"] == 2
        assert disabled["sink_pid"] == 4242
        assert disabled["sink_restarts"] == 7
        assert worker.sink is sink
        assert worker.render_output_frame(live_frame, streaming=True) == bytes(
            (16, 128, 16, 128)
        ) * 2

        enabled = configure_receiver_output.configure_output(
            enabled=True,
            socket_path=socket_path,
        )
        assert enabled["output_enabled"] is True
        assert enabled["output_transform"]["revision"] == 3
        assert enabled["sink_pid"] == 4242
        assert worker.sink is sink
        assert worker.sink_restarts == 7
        assert worker.render_output_frame(live_frame, streaming=True) == (
            receiver.transform_yuyv422(
                live_frame, 2, 2, mirror=True, rotation=270
            )
        )
    finally:
        # The fake sink only models identity and liveness; do not pass it to
        # the process termination helper during test cleanup.
        worker.sink = None
        worker.stop()
        if worker.control_thread is not None:
            worker.control_thread.join(timeout=2.0)

    restarted = configure_small_receiver(
        monkeypatch, source_state_file=str(source_intent)
    )
    assert restarted.output_enabled is True
    assert restarted.output_mirror is True
    assert restarted.output_rotation == 270
    assert restarted.output_transform_revision == 3


def test_disabled_output_ignores_live_content_but_keeps_fixed_yuyv_geometry(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_ENABLED", "false")
    worker = configure_small_receiver(monkeypatch)
    live_frame = yuyv_frame(np.arange(4, dtype=np.uint8).reshape(2, 2))

    first = worker.render_output_frame(live_frame, streaming=True)
    second = worker.render_output_frame(bytes(reversed(live_frame)), streaming=True)

    black = bytes((16, 128, 16, 128)) * 2
    assert first == black
    assert second == black
    assert len(first) == worker.frame_bytes
    assert worker.output_frames == 0


def test_output_helper_requires_at_least_one_hot_setting(tmp_path):
    with pytest.raises(ValueError, match="source, mirror, rotation, or enabled"):
        configure_receiver_output.configure_output(socket_path=tmp_path / "unused.sock")


@pytest.mark.parametrize(
    ("mirror", "rotation"),
    ((False, 0), (True, 0), (False, 90), (True, 180), (False, 270)),
)
def test_every_output_transform_retains_exact_yuyv422_geometry(mirror, rotation):
    frame = yuyv_frame(np.arange(32, dtype=np.uint8).reshape(4, 8))

    transformed = receiver.transform_yuyv422(
        frame, 8, 4, mirror=mirror, rotation=rotation
    )

    assert isinstance(transformed, bytes)
    assert len(transformed) == 8 * 4 * 2


def test_invalid_output_transform_is_rejected_without_touching_frame():
    frame = yuyv_frame(np.arange(8, dtype=np.uint8).reshape(2, 4))

    with pytest.raises(ValueError, match="rotation must be"):
        receiver.transform_yuyv422(frame, 4, 2, rotation=45)
    with pytest.raises(ValueError, match="rotation must be"):
        receiver.transform_yuyv422(frame, 4, 2, rotation=90.5)
    with pytest.raises(ValueError, match="mirror must be"):
        receiver.transform_yuyv422(frame, 4, 2, mirror=1)
    with pytest.raises(ValueError, match="expected 16 YUYV422 bytes"):
        receiver.transform_yuyv422(frame[:-1], 4, 2, mirror=True)
