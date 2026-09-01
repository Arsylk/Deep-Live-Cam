#!/usr/bin/env python3
"""Report local pipeline state and the Windows control-plane health."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, DEFAULT_STATE_DIR, load_env_file
from android_bridge import collect_status as collect_android_status
from pipeline_topology import ROUTE_ANDROID, ROUTE_ARCH, infer_topology, stream_is_fresh


def service_state(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "is-active", name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def windows_health(host: str) -> tuple[dict[str, Any] | None, str | None]:
    if not host or host == "CHANGE_ME":
        return None, "WINDOWS_HOST is not configured"
    url = f"http://{host}:8090/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2.0) as response:
            return json.load(response), None
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, str(exc)


def collect() -> dict[str, Any]:
    config = load_env_file(DEFAULT_CONFIG)
    state_dir = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    host = config.get("WINDOWS_HOST", "")
    remote, remote_error = windows_health(host)
    services = {
        "sender": service_state("deep-live-cam-sender.service"),
        "receiver": service_state("deep-live-cam-receiver.service"),
        "local_processor": service_state("deep-live-cam-phone-processed.service"),
        "phone_return_relay": service_state(
            "deep-live-cam-phone-return-relay.service"
        ),
    }
    android = collect_android_status(
        config.get("ANDROID_ADB_SERIAL", ""),
        config.get("ANDROID_HOST", ""),
        config.get("ANDROID_CAMERA_ID", "120"),
    ) if config.get("ANDROID_BRIDGE_ENABLED", "1") == "1" else {}
    topology = infer_topology(
        remote,
        arch_host=config.get("ARCH_HOST", "192.168.1.11"),
        android_host=config.get("ANDROID_HOST", "192.168.1.12"),
        arch_sender_active=services["sender"] == "active",
        android_sender_active=(
            bool(android.get("bridge_running") and android.get("network_sender_running"))
            if android.get("available")
            else None
        ),
    )
    return {
        "windows_host": host or None,
        "services": services,
        "sender": read_json(state_dir / "sender.json"),
        "receiver": read_json(state_dir / "receiver.json"),
        "phone_return_relay": read_json(state_dir / "phone-return-relay.json"),
        "shadow": read_json(state_dir / "shadow.json"),
        "windows": remote,
        "windows_error": remote_error,
        "android": android,
        "selected_stream": {
            "protocol": "srt",
            "host": host,
            "port": int(
                config.get(
                    "WINDOWS_SELECTED_STREAM_PORT",
                    config.get("WINDOWS_BROADCAST_PORT", "10010"),
                )
            ),
            "client_role": "caller",
            "streaming": bool(
                ((remote or {}).get("selected_stream") or (remote or {}).get("broadcast") or {}).get("streaming")
            ),
        },
        "topology": topology.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    args = parser.parse_args()
    report = collect()
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"Sender service:   {report['services']['sender']}")
        print(f"Receiver service: {report['services']['receiver']}")
        print(f"Local processor:  {report['services']['local_processor']}")
        print(f"Phone relay:      {report['services']['phone_return_relay']}")
        topology = report["topology"]
        print(f"Selected route:   {topology['selected']} ({topology['summary']})")
        sender = report["sender"] or {}
        receiver = report["receiver"] or {}
        print(f"Sender state:     {sender.get('status', 'unavailable')}")
        print(
            f"Receiver state:   {receiver.get('status', 'unavailable')} "
            f"(input={receiver.get('active_input') or 'waiting'})"
        )
        relay = report["phone_return_relay"] or {}
        print(
            f"Phone return:     source={relay.get('source', 'unavailable')} "
            f"effective={relay.get('effective_source', 'off')} "
            f"streaming={bool(relay.get('streaming'))}"
        )
        shadow = report["shadow"] or {}
        print(f"Shadow state:     {shadow.get('status', 'unavailable')}")
        if report["windows"] is not None:
            print(f"Windows health:   reachable (healthy={report['windows'].get('healthy')}, streaming={report['windows'].get('streaming')})")
            print("Native controls:  deep-live-cam-tester")
        else:
            print(f"Windows health:   unreachable ({report['windows_error']})")
        android = report["android"] or {}
        print(
            "Android node:    "
            + (
                f"ADB connected, bridge={'running' if android.get('bridge_running') else 'stopped'}, "
                f"Camera2 {android.get('camera_id', '120')}={'published' if android.get('camera_published') else 'not ready'}"
                if android.get("available")
                else f"ADB management unavailable ({android.get('error', 'not configured')})"
            )
        )
        if topology.get("warning"):
            print(f"Route warning:   {topology['warning']}")
    selected = report["topology"].get("selected")
    windows_input = (report["windows"] or {}).get("input") or {}
    if selected == ROUTE_ANDROID:
        route_ok = stream_is_fresh(windows_input) and not report["topology"].get("conflict")
    elif selected == ROUTE_ARCH:
        route_ok = (
            report["services"]["sender"] == "active"
            and report["services"]["receiver"] == "active"
            and stream_is_fresh(windows_input)
            and not report["topology"].get("conflict")
        )
    else:
        route_ok = False
    return 0 if route_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
