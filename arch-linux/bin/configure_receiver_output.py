#!/usr/bin/env python3
"""Hot-configure the stable Arch camera output without restarting its sink."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


SOURCES = ("auto", "local", "windows", "raw", "prerecorded")
ROTATIONS = (0, 90, 180, 270)


def _send_request(request: dict[str, object], socket_path: Path) -> dict[str, object]:
    payload = json.dumps(request).encode("utf-8")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(4.0)
    try:
        client.connect(str(socket_path))
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while len(response) <= 65_536:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    finally:
        client.close()
    result = json.loads(response)
    if not isinstance(result, dict):
        raise RuntimeError("receiver returned a non-object response")
    if result.get("ok") is not True:
        raise RuntimeError(str(result.get("error") or "output configuration failed"))
    return result


def configure_output(
    *,
    socket_path: Path,
    mirror: bool | None = None,
    rotation: int | None = None,
    enabled: bool | None = None,
    source: str | None = None,
) -> dict[str, object]:
    if mirror is not None and not isinstance(mirror, bool):
        raise ValueError("mirror must be true or false")
    if rotation is not None and (
        not isinstance(rotation, int) or isinstance(rotation, bool)
    ):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if rotation is not None and rotation not in ROTATIONS:
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    request: dict[str, object] = {}
    transform: dict[str, object] = {}
    if mirror is not None:
        transform["mirror"] = mirror
    if rotation is not None:
        transform["rotation"] = rotation
    if transform:
        request["transform"] = transform
    if enabled is not None:
        request["enabled"] = enabled
    if source is not None:
        normalized_source = source.strip().lower()
        if normalized_source not in SOURCES:
            raise ValueError(
                "source must be auto, local, windows, raw, or prerecorded"
            )
        request["source"] = normalized_source
    if not request:
        raise ValueError("source, mirror, rotation, or enabled is required")
    return _send_request(request, socket_path)


def parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true/false, on/off, or 1/0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=parse_boolean)
    parser.add_argument("--rotation", type=int, choices=ROTATIONS)
    parser.add_argument("--enabled", type=parse_boolean)
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path("/run/deep-live-cam/receiver-control.sock"),
    )
    args = parser.parse_args()
    try:
        result = configure_output(
            mirror=args.mirror,
            rotation=args.rotation,
            enabled=args.enabled,
            source=args.source,
            socket_path=args.socket,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
