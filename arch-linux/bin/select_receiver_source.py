#!/usr/bin/env python3
"""Select the stable Arch camera's source without restarting camera owners."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


SOURCES = ("auto", "local", "windows", "raw", "prerecorded")


def select_source(source: str, socket_path: Path) -> dict[str, object]:
    request = json.dumps({"source": source}).encode("utf-8")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(4.0)
    try:
        client.connect(str(socket_path))
        client.sendall(request)
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
        raise RuntimeError(str(result.get("error") or "selection failed"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path("/run/deep-live-cam/receiver-control.sock"),
    )
    args = parser.parse_args()
    try:
        result = select_source(args.source, args.socket)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
