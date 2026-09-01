#!/usr/bin/env python3
"""Exclusive Arch-side owner for processed video returned to Android.

Windows has one paired return per input slot.  Slot 0 can return directly to the
phone, but slot 1 returns to Arch.  This service closes that otherwise-missing
matrix cell by relaying an independent encoded copy of the selected Windows
stream to Android.  It also accepts the local processor's encoded phone leg.

Only this process opens an Arch-originated SRT caller to the phone.  Source
changes first close the old FFmpeg process and only then acknowledge the
control request, so the manager can safely sequence it against Windows' direct
slot-0 return without ever exposing two callers to Android's single listener.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from common import (
    env_float,
    env_int,
    install_signal_handlers,
    sd_notify,
    srt_query,
    state_dir,
    stop_process,
    write_state,
)


SOURCES = {"off", "local", "windows"}
MAX_CONTROL_BYTES = 16 * 1024
STATE_PUBLISH_INTERVAL_SECONDS = 1.0


def _environment_source() -> str:
    value = os.environ.get("WINDOWS_PHONE_RELAY_SOURCE", "off").strip().lower()
    if value not in SOURCES:
        raise ValueError("WINDOWS_PHONE_RELAY_SOURCE must be off, local, or windows")
    return value


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    """Replace one tiny service-owned state file without a partial-write window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(dict(document), sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _udp_input(port: int) -> str:
    return (
        f"udp://127.0.0.1:{port}?reuse=1&fifo_size=1000000&"
        "overrun_nonfatal=1"
    )


def _udp_output(port: int) -> str:
    return f"udp://127.0.0.1:{port}?pkt_size=1316"


def _srt_output(host: str, port: int, latency_us: int) -> str:
    query = srt_query(
        {
            "mode": "caller",
            "transtype": "live",
            "messageapi": 1,
            "latency": latency_us,
            "connect_timeout": 3000,
            "timeout": 5_000_000,
            "tlpktdrop": 1,
            "pkt_size": 1316,
        }
    )
    return f"srt://{host}:{port}?{query}"


def _tee_target(*urls: str) -> str:
    return "|".join(
        "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]" + url
        for url in urls
    )


class WindowsPhoneRelay:
    """Persistent control owner with a replaceable transport-only worker."""

    def __init__(self) -> None:
        self.windows_port = env_int("WINDOWS_PHONE_RELAY_SOURCE_PORT", 11_008)
        self.local_port = env_int("LOCAL_PHONE_RELAY_SOURCE_PORT", 11_009)
        self.preview_port = env_int("ANDROID_NATIVE_PREVIEW_PORT", 11_004)
        self.android_host = os.environ.get("ANDROID_HOST", "192.168.1.12").strip()
        self.android_port = env_int("ANDROID_NATIVE_RETURN_PORT", 10_001)
        self.latency_us = env_int("SRT_LATENCY_US", 100_000, 20_000)
        self.retry_seconds = env_float("RETRY_SECONDS", 2.0, 0.2)
        self.stale_seconds = env_float("PHONE_RELAY_STALE_SECONDS", 3.0, 0.5)
        self.control_socket_path = Path(
            os.environ.get(
                "PHONE_RETURN_RELAY_CONTROL_SOCKET",
                str(state_dir() / "phone-return-relay-control.sock"),
            )
        )
        self.intent_path = Path(
            os.environ.get(
                "PHONE_RETURN_RELAY_STATE_FILE",
                "/var/lib/deep-live-cam/phone-return-relay.json",
            )
        )
        if not self.android_host:
            raise ValueError("ANDROID_HOST must not be empty")
        ports = {
            self.windows_port,
            self.local_port,
            self.preview_port,
            self.android_port,
        }
        if len(ports) != 4:
            raise ValueError("phone relay source, preview, and return ports must be distinct")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed")

        self.stop_event = threading.Event()
        self.changed = threading.Event()
        self.lock = threading.RLock()
        self.source, self.revision, restore_error = self._restore_intent()
        self.process: subprocess.Popen[bytes] | None = None
        self.process_source = "off"
        self.restarts = 0
        self.frames = 0
        self.last_frame_at = 0.0
        self.last_error: str | None = restore_error
        self.started_at = time.time()
        self.control_server: socket.socket | None = None
        self.control_thread: threading.Thread | None = None

    def _restore_intent(self) -> tuple[str, int, str | None]:
        """Restore durable GUI intent, failing closed on malformed state."""
        if not self.intent_path.exists():
            return _environment_source(), 0, None
        try:
            document = json.loads(self.intent_path.read_text(encoding="utf-8"))
            if not isinstance(document, Mapping):
                raise ValueError("state is not a JSON object")
            source = document.get("source")
            revision = document.get("revision")
            if source not in SOURCES:
                raise ValueError("state source must be off, local, or windows")
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 0
            ):
                raise ValueError("state revision must be a non-negative integer")
            return str(source), revision, None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            return "off", 0, f"durable relay state ignored (fail-closed): {error}"

    def _persist_intent_locked(self, source: str, revision: int) -> None:
        _atomic_json(
            self.intent_path,
            {"schema_version": 1, "source": source, "revision": revision},
        )

    def source_port(self, source: str) -> int:
        if source == "windows":
            return self.windows_port
        if source == "local":
            return self.local_port
        raise ValueError("source must be local or windows")

    def command(self, source: str) -> list[str]:
        source_url = _udp_input(self.source_port(source))
        phone_url = _srt_output(
            self.android_host, self.android_port, self.latency_us
        )
        preview_url = _udp_output(self.preview_port)
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
            "500000",
            "-probesize",
            "500000",
            "-max_delay",
            "0",
            "-f",
            "mpegts",
            "-i",
            source_url,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-bsf:v",
            "dump_extra=freq=keyframe",
            "-flush_packets",
            "1",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "tee",
            "-use_fifo",
            "1",
            "-fifo_options",
            (
                "attempt_recovery=1:recover_any_error=1:recovery_wait_time=1:"
                "restart_with_keyframe=1:drop_pkts_on_overflow=1"
            ),
            _tee_target(phone_url, preview_url),
        ]

    def _terminate_locked(self) -> None:
        process = self.process
        self.process = None
        self.process_source = "off"
        # A control request must fit inside the GUI helper's four-second
        # timeout even when FFmpeg ignores SIGTERM.  stop_process waits once
        # before and once after SIGKILL, hence 0.4s keeps the close bounded.
        stop_process(process, grace=0.4)

    def set_source(self, source: str, *, revision: int | None = None) -> dict[str, Any]:
        normalized = str(source).strip().lower()
        if normalized not in SOURCES:
            raise ValueError("source must be off, local, or windows")
        if revision is not None and (isinstance(revision, bool) or revision < 0):
            raise ValueError("revision must be a non-negative integer")
        with self.lock:
            target_revision = self.revision + 1 if revision is None else int(revision)
            if target_revision < self.revision:
                raise ValueError(
                    f"stale revision {target_revision}; current revision is {self.revision}"
                )
            if target_revision == self.revision and normalized != self.source:
                raise ValueError(
                    "a revision already applied cannot select a different source"
                )
            changed = normalized != self.source
            if changed:
                # This synchronous close is the ownership guarantee consumed by
                # the manager before it lets Windows select its direct phone slot.
                self._terminate_locked()
            if changed or target_revision != self.revision:
                self._persist_intent_locked(normalized, target_revision)
            if changed:
                self.source = normalized
                self.last_frame_at = 0.0
            self.revision = target_revision
            self.changed.set()
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, Any]:
        now = time.time()
        process = self.process
        worker_alive = process is not None and process.poll() is None
        frame_age = (
            None
            if not self.last_frame_at
            else round(max(0.0, now - self.last_frame_at), 3)
        )
        return {
            "state": "stopping" if self.stop_event.is_set() else "running",
            "source": self.source,
            "effective_source": self.process_source if worker_alive else "off",
            "revision": self.revision,
            "worker_alive": worker_alive,
            "transport_open": worker_alive,
            "pid": process.pid if worker_alive else None,
            "frames": self.frames,
            "last_frame_age": frame_age,
            "streaming": bool(
                worker_alive
                and frame_age is not None
                and frame_age <= self.stale_seconds
            ),
            "restarts": self.restarts,
            "last_error": self.last_error,
            "input_ports": {
                "windows": self.windows_port,
                "local": self.local_port,
            },
            "output": {
                "host": self.android_host,
                "port": self.android_port,
                "preview_port": self.preview_port,
            },
            "intent_file": str(self.intent_path),
            "uptime_seconds": round(max(0.0, now - self.started_at), 1),
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.snapshot_locked()

    def _handle_control(self, connection: socket.socket) -> None:
        try:
            payload = connection.recv(MAX_CONTROL_BYTES + 1)
            if not payload or len(payload) > MAX_CONTROL_BYTES:
                raise ValueError("relay control request is empty or too large")
            request = json.loads(payload.split(b"\n", 1)[0].decode("utf-8"))
            if not isinstance(request, Mapping):
                raise ValueError("expected a JSON object")
            operation = str(request.get("op", "get"))
            if operation == "get":
                response = self.snapshot()
            elif operation == "set":
                revision_value = request.get("revision")
                if isinstance(revision_value, bool):
                    raise ValueError("revision must be a non-negative integer")
                revision = None if revision_value is None else int(revision_value)
                response = self.set_source(
                    str(request.get("source", "")), revision=revision
                )
            else:
                raise ValueError("op must be get or set")
            response = {"ok": True, **response}
        except Exception as error:
            response = {"ok": False, "error": str(error)}
        try:
            connection.sendall(json.dumps(response, sort_keys=True).encode() + b"\n")
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
        # The session manager is an unprivileged desktop process and is not a
        # member of the service account's group.  The protocol is strictly
        # allow-listed (get/set source only), matching the receiver/processor
        # control sockets, so mode 0666 is deliberate.
        os.chmod(self.control_socket_path, 0o666)
        server.listen(4)
        server.settimeout(0.5)
        while not self.stop_event.is_set():
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                self._handle_control(connection)

    def stop(self) -> None:
        self.stop_event.set()
        self.changed.set()
        server = self.control_server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        with self.lock:
            self._terminate_locked()

    def _publish_state(self) -> dict[str, Any]:
        """Refresh runtime health and keep systemd's watchdog alive."""
        snapshot = self.snapshot()
        write_state("phone-return-relay", snapshot)
        sd_notify(
            "WATCHDOG=1\nSTATUS="
            f"{snapshot['source']}: "
            f"{'streaming' if snapshot['streaming'] else 'waiting'}"
        )
        return snapshot

    def _observe_progress(self, process: subprocess.Popen[bytes], source: str) -> None:
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        seen_frame = 0
        launched = time.monotonic()
        last_progress = launched
        last_state = 0.0
        try:
            while not self.stop_event.is_set() and process.poll() is None:
                with self.lock:
                    if source != self.source or process is not self.process:
                        break
                for key, _events in selector.select(timeout=0.25):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        continue
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("frame="):
                            continue
                        try:
                            frame = int(line.split("=", 1)[1].strip())
                        except ValueError:
                            continue
                        if frame > seen_frame:
                            delta = frame - seen_frame
                            seen_frame = frame
                            last_progress = time.monotonic()
                            with self.lock:
                                self.frames += delta
                                self.last_frame_at = time.time()
                now = time.monotonic()
                if now - last_state >= STATE_PUBLISH_INTERVAL_SECONDS:
                    # A healthy active worker can remain in this loop forever.
                    # Publish from inside it so systemd does not mistake useful
                    # continuous streaming for a wedged main process.
                    self._publish_state()
                    last_state = now
                if seen_frame and now - last_progress > self.stale_seconds:
                    self.last_error = f"{source} input stalled"
                    break
                if not seen_frame and now - launched > 10.0:
                    self.last_error = f"{source} input produced no frames"
                    break
        finally:
            selector.close()

    def run(self) -> int:
        install_signal_handlers(self.stop)
        self.control_thread = threading.Thread(
            target=self.control_loop, name="phone-relay-control", daemon=True
        )
        self.control_thread.start()
        sd_notify("READY=1\nSTATUS=phone-return relay ready; source off")
        last_state = 0.0
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    source = self.source
                if source == "off":
                    self.changed.clear()
                    if source == self.source:
                        self.changed.wait(0.25)
                else:
                    process: subprocess.Popen[bytes] | None = None
                    try:
                        with self.lock:
                            if source != self.source or self.stop_event.is_set():
                                continue
                            self.process = subprocess.Popen(
                                self.command(source),
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                bufsize=0,
                            )
                            process = self.process
                            self.process_source = source
                            self.restarts += 1
                            self.last_error = None
                        self._observe_progress(process, source)
                    except OSError as error:
                        self.last_error = str(error)
                    finally:
                        with self.lock:
                            if process is not None and self.process is process:
                                self._terminate_locked()
                    if not self.stop_event.is_set() and source == self.source:
                        self.stop_event.wait(self.retry_seconds)

                now = time.monotonic()
                if now - last_state >= 1.0:
                    self._publish_state()
                    last_state = now
        finally:
            self.stop()
            if self.control_thread is not None:
                self.control_thread.join(timeout=2.0)
            try:
                self.control_socket_path.unlink()
            except FileNotFoundError:
                pass
            write_state("phone-return-relay", self.snapshot())
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and print the inactive topology",
    )
    args = parser.parse_args()
    try:
        relay = WindowsPhoneRelay()
    except (ValueError, RuntimeError) as error:
        print(f"deep-live-cam phone relay: {error}", file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(relay.snapshot(), indent=2, sort_keys=True))
        return 0
    return relay.run()


if __name__ == "__main__":
    raise SystemExit(main())
