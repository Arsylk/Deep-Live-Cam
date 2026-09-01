#!/usr/bin/env python3
"""Configure the persistent Arch-to-Android processed-return relay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence


SOURCES = ("off", "local", "windows")
MAX_RESPONSE = 64 * 1024


def request_relay(
    request: Mapping[str, Any], *, socket_path: str | Path, timeout: float = 4.0
) -> dict[str, Any]:
    payload = json.dumps(dict(request), separators=(",", ":")).encode("utf-8")
    chunks = bytearray()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(Path(socket_path)))
        client.sendall(payload + b"\n")
        client.shutdown(socket.SHUT_WR)
        while len(chunks) <= MAX_RESPONSE:
            chunk = client.recv(min(4096, MAX_RESPONSE + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
            if b"\n" in chunk:
                break
    if not chunks or len(chunks) > MAX_RESPONSE:
        raise RuntimeError("phone relay returned an invalid response size")
    response = json.loads(bytes(chunks).split(b"\n", 1)[0])
    if not isinstance(response, dict):
        raise RuntimeError("phone relay returned a non-object response")
    if response.get("ok") is not True:
        raise RuntimeError(str(response.get("error") or "phone relay rejected state"))
    return response


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument(
        "--socket",
        default="/run/deep-live-cam/phone-return-relay-control.sock",
    )
    parser.add_argument("--revision", type=int)
    args = parser.parse_args(argv)
    request: dict[str, Any] = {"op": "set", "source": args.source}
    if args.revision is not None:
        request["revision"] = args.revision
    try:
        response = request_relay(request, socket_path=args.socket)
    except Exception as error:
        parser.exit(1, f"phone relay control failed: {error}\n")
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
