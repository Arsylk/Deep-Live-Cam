"""Persistent five-slot network-camera registry for the Windows processor.

The registry owns names and endpoints only.  It never opens a camera device;
capture and stable virtual-camera publication remain responsibilities of the
per-device clients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any


SLOT_COUNT = 5
FIRST_DEVICE_PORT = 10_000
BROADCAST_PORT = 10_010
BROADCAST_HOST = "192.168.1.35"
# Kept as a compatibility alias for callers that displayed the old multicast
# sender address. It is now the address clients connect to, not an FFmpeg
# ``localaddr`` option.
BROADCAST_LOCAL_ADDRESS = BROADCAST_HOST
BROADCAST_LISTEN_URL = (
    f"srt://0.0.0.0:{BROADCAST_PORT}?mode=listener&transtype=live&"
    "messageapi=1&pkt_size=1316&latency=100000&tlpktdrop=1&timeout=5000000"
)
BROADCAST_URL = (
    f"srt://{BROADCAST_HOST}:{BROADCAST_PORT}?mode=caller&transtype=live&"
    "messageapi=1&pkt_size=1316&latency=100000&tlpktdrop=1&"
    "connect_timeout=3000&timeout=5000000"
)
DEVICE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\."
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$"
)
STACKS = {"android-camera2", "arch-v4l2", "generic-srt"}


class DeviceRegistryError(ValueError):
    """Raised when a slot document violates the transport contract."""


def input_port(slot: int) -> int:
    if not 0 <= slot < SLOT_COUNT:
        raise DeviceRegistryError(f"slot must be between 0 and {SLOT_COUNT - 1}")
    return FIRST_DEVICE_PORT + slot * 2


def return_port(slot: int) -> int:
    return input_port(slot) + 1


def _validated_host(value: str) -> str:
    host = str(value).strip()
    if not host:
        raise DeviceRegistryError("a configured device requires return_host")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        if HOSTNAME_PATTERN.fullmatch(host):
            return host
    raise DeviceRegistryError(f"invalid return_host: {value!r}")


@dataclass(frozen=True)
class DeviceSlot:
    slot: int
    device_id: str | None = None
    label: str = "Unassigned"
    stack: str = "generic-srt"
    return_host: str | None = None
    enabled: bool = False
    camera: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        input_port(self.slot)
        if self.device_id is None:
            if self.enabled or self.return_host:
                raise DeviceRegistryError(
                    f"slot {self.slot}: an unassigned slot cannot be enabled"
                )
            return
        if not DEVICE_ID_PATTERN.fullmatch(self.device_id):
            raise DeviceRegistryError(f"slot {self.slot}: invalid device_id")
        if self.stack not in STACKS:
            raise DeviceRegistryError(
                f"slot {self.slot}: stack must be one of {sorted(STACKS)}"
            )
        _validated_host(self.return_host or "")
        if not isinstance(self.camera, dict):
            raise DeviceRegistryError(f"slot {self.slot}: camera must be an object")

    @property
    def configured(self) -> bool:
        return self.device_id is not None

    @property
    def input_port(self) -> int:
        return input_port(self.slot)

    @property
    def return_port(self) -> int:
        return return_port(self.slot)

    def input_url(self, latency_us: int) -> str:
        return (
            f"srt://0.0.0.0:{self.input_port}?mode=listener&transtype=live&"
            f"messageapi=1&pkt_size=1316&latency={latency_us}&tlpktdrop=1"
        )

    def return_url(self, latency_us: int) -> str:
        host = _validated_host(self.return_host or "")
        return (
            f"srt://{host}:{self.return_port}?mode=caller&transtype=live&"
            f"messageapi=1&pkt_size=1316&latency={latency_us}&tlpktdrop=1&"
            "connect_timeout=3000&timeout=5000000"
        )

    def public(self, selected: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "configured": self.configured,
                "selected": selected,
                "input_port": self.input_port,
                "return_port": self.return_port,
            }
        )
        return value


def default_slots(
    android_host: str = "192.168.1.12",
    arch_host: str = "192.168.1.11",
) -> list[DeviceSlot]:
    """Defaults preserve the currently-live Android 10000/10001 path."""
    configured = [
        DeviceSlot(
            0,
            "android-phone",
            "Android phone",
            "android-camera2",
            android_host,
            True,
            {
                "adapter": "android-camera2",
                "camera_id": "front",
                "virtual_camera_id": "120",
            },
        ),
        DeviceSlot(
            1,
            "arch-webcam",
            "Arch USB webcam",
            "arch-v4l2",
            arch_host,
            True,
            {
                "adapter": "arch-v4l2",
                "stable_device": "/dev/video0",
            },
        ),
    ]
    configured.extend(DeviceSlot(slot) for slot in range(2, SLOT_COUNT))
    return configured


class DeviceRegistry:
    VERSION = 1

    def __init__(
        self,
        directory: str | Path,
        *,
        android_host: str = "192.168.1.12",
        arch_host: str = "192.168.1.11",
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "devices.json"
        self._lock = threading.RLock()
        self._slots = default_slots(android_host, arch_host)
        self._selected_device_id = "android-phone"
        self._generation = 0
        self._load()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def selected_device_id(self) -> str:
        with self._lock:
            return self._selected_device_id

    def slots(self) -> tuple[DeviceSlot, ...]:
        with self._lock:
            return tuple(self._slots)

    def selected(self) -> DeviceSlot:
        return self.resolve(self.selected_device_id)

    def resolve(self, device_id: str) -> DeviceSlot:
        with self._lock:
            match = next(
                (slot for slot in self._slots if slot.device_id == device_id),
                None,
            )
        if match is None or not match.configured:
            raise DeviceRegistryError(f"unknown device_id: {device_id!r}")
        if not match.enabled:
            raise DeviceRegistryError(f"device {device_id!r} is disabled")
        return match

    def select(self, device_id: str) -> tuple[DeviceSlot, bool]:
        selected = self.resolve(str(device_id))
        with self._lock:
            changed = selected.device_id != self._selected_device_id
            if changed:
                self._selected_device_id = selected.device_id or ""
                self._generation += 1
                self._persist_locked()
            return selected, changed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            selected = self._selected_device_id
            selected_stream = {
                "protocol": "mpegts-srt",
                "host": BROADCAST_HOST,
                "port": BROADCAST_PORT,
                "windows_role": "listener",
                "client_role": "caller",
                "url": BROADCAST_URL,
            }
            return {
                "version": self.VERSION,
                "selected_device_id": selected,
                "generation": self._generation,
                "selected_stream": selected_stream,
                # Compatibility alias for native clients deployed before the
                # transport changed from multicast UDP to pull-based SRT.
                "broadcast": dict(selected_stream),
                "slots": [
                    slot.public(slot.device_id == selected) for slot in self._slots
                ],
            }

    def _load(self) -> None:
        if not self.path.exists():
            self._persist_locked()
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if int(document.get("version", 0)) != self.VERSION:
                raise DeviceRegistryError("unsupported devices.json version")
            raw_slots = document.get("slots")
            if not isinstance(raw_slots, list) or len(raw_slots) != SLOT_COUNT:
                raise DeviceRegistryError(f"devices.json must define {SLOT_COUNT} slots")
            slots = [
                DeviceSlot(
                    slot=int(raw["slot"]),
                    device_id=raw.get("device_id"),
                    label=str(raw.get("label", "Unassigned")),
                    stack=str(raw.get("stack", "generic-srt")),
                    return_host=raw.get("return_host"),
                    enabled=bool(raw.get("enabled", False)),
                    camera=dict(raw.get("camera") or {}),
                )
                for raw in raw_slots
            ]
            if sorted(slot.slot for slot in slots) != list(range(SLOT_COUNT)):
                raise DeviceRegistryError("slot numbers must be unique and contiguous")
            identifiers = [slot.device_id for slot in slots if slot.device_id]
            if len(identifiers) != len(set(identifiers)):
                raise DeviceRegistryError("device_id values must be unique")
            selected = str(document.get("selected_device_id", ""))
            if not any(
                slot.device_id == selected and slot.enabled for slot in slots
            ):
                raise DeviceRegistryError("selected_device_id is not enabled")
            self._slots = sorted(slots, key=lambda slot: slot.slot)
            self._selected_device_id = selected
            self._generation = max(0, int(document.get("generation", 0)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DeviceRegistryError(f"invalid {self.path}: {exc}") from exc

    def _persist_locked(self) -> None:
        document = {
            "version": self.VERSION,
            "selected_device_id": self._selected_device_id,
            "generation": self._generation,
            "slots": [asdict(slot) for slot in self._slots],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".devices.", suffix=".json", dir=self.directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
