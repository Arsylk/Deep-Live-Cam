# Windows five-slot processor overlay

Copy these files over the matching `C:\Deep-Live-Cam` files on the Windows 11
CUDA host. The router persists its registry in
`runtime\network-live\devices.json` and defaults to:

- slot 0 `android-phone`: UDP 10000 in, UDP 10001 back to `192.168.1.12`;
- slot 1 `arch-webcam`: UDP 10002 in, UDP 10003 back to `192.168.1.11`;
- slots 2–4: reserved/unassigned;
- selected output SRT listener: `192.168.1.35:10010`.

The scheduled task continues to launch `run-network-service.cmd`, but the
command no longer embeds one device's return address. Manual selection occurs
through `POST /api/devices/select` and swaps only the selected SRT workers.
The CUDA processor, JSON API, and selected-stream publisher remain alive.

The network service defaults to ONNX Runtime CUDA on this RTX 2060. The swap
model still uses its static-shape CUDA graph, while detector/source-analysis
sessions avoid TensorRT's multi-minute first-use builds and large route-switch
memory spike. TensorRT remains available for controlled benchmarking with
`run_network.py --execution-provider tensorrt`; its engine cache remains under
`runtime\trt-cache`. CPU-side OpenCV/OMP work is capped at four threads by
default (`--cpu-threads`) so inference cannot starve the control API.

The service exposes no browser camera page or HTTPS certificate workflow. TCP
8090 is a LAN-only JSON API used by the native Arch manager. See
[`CAMERA_NETWORK_PROTOCOL.md`](../CAMERA_NETWORK_PROTOCOL.md) for the complete
client and failure contract.

Source-picture synchronization is content exact. `POST /api/source` requires
`X-Source-Identifier` to be the lowercase SHA-256 digest of the uploaded body;
a forged or malformed identifier is rejected without replacing the current
source. The service atomically persists `source.jpg` and
`source.metadata.json`, verifies the stored normalized JPEG on restart, and
returns `source_identifier` from `/api/config` only while both files agree.
The Arch manager uses that identifier—not the weaker `source_configured`
boolean—to decide whether the currently selected history picture is synced.

This directory is an overlay, not proof of the running Windows state. It must
be copied and activated on `192.168.1.35`; until that host's live API reports
the expected configuration and source identifier, the manager should display
saved/pending or offline.

## Temporal-stability pipeline

The live processor now detects on every frame in the Balanced and Quality
profiles and reconciles each detection with optical-flow motion prediction.
It keeps the selected person by overlap/proximity, propagates both five-point
and 106-point mouth landmarks, and gradually fades through a short miss instead
of alternating between processed and original frames. Optional interpolation
is motion-compensated and restricted to a feathered face region.

`GET /healthz` exposes `processing.tracking` plus face-local measurements under
`processing.quality.face`. A compact five-second history is written to
`runtime\network-live\quality-history.jsonl`; compare controlled baseline and
candidate runs with `python tools\stability_report.py --baseline OLD.jsonl
--candidate NEW.jsonl`.

Input, return, and selected-stream health also expose actual frame interval, cadence
jitter, estimated transport gaps, queue overwrites, output repeats, and
connection/disconnection counts. Reconnect gaps begin a new cadence window, so
they are not misreported as thousands of missing camera frames. The
return publisher uses a bounded constant-rate hold to keep wall-clock delivery
and MPEG-TS timestamps aligned through short inference-time variations, then
closes after one stale second to preserve the client fallback contract.

Route teardown keeps immutable FFmpeg pipe handles inside each worker. This
prevents a concurrent native-client route selection from turning the expected
closed-pipe event into a `None.stdout` worker exception.
