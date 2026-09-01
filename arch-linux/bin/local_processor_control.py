#!/usr/bin/env python3
"""Small native client for the Arch processor's Unix control socket.

The desktop manager can import :func:`request_processor` directly.  The CLI is
also useful for diagnostics and deliberately has no browser/network fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence


MAX_RESPONSE = 1024 * 1024


def default_socket_path() -> Path:
    configured = os.environ.get("XDG_RUNTIME_DIR")
    runtime = Path(configured) if configured else Path(f"/run/user/{os.getuid()}")
    return runtime / "deep-live-cam" / "processor-control.sock"


def request_processor(
    request: Mapping[str, Any],
    *,
    socket_path: Path | str | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Send one JSON request and return the processor's canonical state."""
    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")
    path = Path(socket_path or default_socket_path()).expanduser()
    payload = json.dumps(dict(request), separators=(",", ":")).encode("utf-8")
    if len(payload) > 256 * 1024:
        raise ValueError("request is too large")
    chunks = bytearray()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(payload + b"\n")
        client.shutdown(socket.SHUT_WR)
        while len(chunks) <= MAX_RESPONSE:
            chunk = client.recv(min(65_536, MAX_RESPONSE + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
            if b"\n" in chunk:
                break
    if not chunks:
        raise RuntimeError("processor returned an empty response")
    if len(chunks) > MAX_RESPONSE:
        raise RuntimeError("processor response is too large")
    document = json.loads(bytes(chunks).split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError("processor response is not a JSON object")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=str(default_socket_path()))
    parser.add_argument(
        "--input",
        choices=("arch-webcam", "android-front", "android-back"),
    )
    parser.add_argument(
        "--active",
        choices=("true", "false"),
        help="hot-toggle ownership of the local processor's return delivery",
    )
    parser.add_argument(
        "--processing-json",
        help="partial processing configuration as a JSON object",
    )
    parser.add_argument("--source-path")
    parser.add_argument(
        "--activate-native256",
        action="store_true",
        help="request the fixed native-256 ncnn/Vulkan model target",
    )
    parser.add_argument("--revision", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    request: dict[str, Any] = {"op": "get"}
    if any(
        value is not None
        for value in (
            args.active,
            args.input,
            args.processing_json,
            args.source_path,
        )
    ) or args.activate_native256:
        request = {"op": "set"}
        if args.active is not None:
            request["active"] = args.active == "true"
        if args.input:
            request["input"] = args.input
        if args.processing_json:
            try:
                processing = json.loads(args.processing_json)
            except json.JSONDecodeError as error:
                parser.error(f"invalid --processing-json: {error}")
            if not isinstance(processing, dict):
                parser.error("--processing-json must be a JSON object")
            request["processing"] = processing
        if args.source_path:
            request["source_path"] = args.source_path
        if args.activate_native256:
            request.update(
                {"swapper_model": "native-256", "swapper_backend": "ncnn"}
            )
    if args.revision is not None:
        request["revision"] = args.revision
    try:
        response = request_processor(request, socket_path=args.socket)
    except Exception as error:
        parser.exit(1, f"local processor control failed: {error}\n")
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
