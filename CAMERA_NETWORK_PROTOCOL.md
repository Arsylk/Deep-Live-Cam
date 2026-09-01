# Multi-device camera routing contract

Version 1 fixes ownership, ports, switching, and failure behavior for the
Windows processor and up to five camera clients. The Windows control service
never opens a client camera. Each client owns its capture device and one stable
OS-visible output camera for its entire lifetime.

## Reserved endpoints

| Slot | Default client | Client → Windows | Windows → same client |
|---:|---|---:|---:|
| 0 | Android phone | SRT/UDP 10000 | SRT/UDP 10001 |
| 1 | Arch USB webcam | SRT/UDP 10002 | SRT/UDP 10003 |
| 2 | Unassigned | SRT/UDP 10004 | SRT/UDP 10005 |
| 3 | Unassigned | SRT/UDP 10006 | SRT/UDP 10007 |
| 4 | Unassigned | SRT/UDP 10008 | SRT/UDP 10009 |

For every slot, `return_port = input_port + 1`. Video is H.264 in MPEG-TS,
1280×720 at 30 FPS. SRT uses live/message mode with a 100 ms latency target.

### Selected Android slot-0 route

The active phone route uses the reserved Windows slot directly:

```text
Android front Camera2 -> Windows SRT UDP 10000 -> CUDA face swap
Android Camera2 120   <- Windows SRT UDP 10001 <- processed return
```

The phone keeps its physical capture owner and Camera2 ID 120 registered for
the entire route. If the Windows return is stale, the Android-side mux uses the
same phone encoder as a raw fallback without changing camera identity. The
standalone Arch INSwapper-128 service remains an explicit alternative, not a
simultaneous producer for Android port 10001.

The selected processed stream is also exposed by Windows as an MPEG-TS/SRT
listener at `srt://192.168.1.35:10010`. The Arch receiver initiates this
connection, which works reliably through host firewall and VPN policy without
depending on Wi-Fi multicast forwarding. It is not a device return port and it
is never used as a source of camera ownership or control.

## Selection and isolation

- Exactly one slot is selected on Windows.
- Windows creates an input listener and paired return sender only for that
  slot. Unselected clients keep capturing locally and their SRT callers retry.
- Manual selection is `POST /api/devices/select` with
  `{"device_id":"..."}`. The control server and CUDA processor stay alive.
- Every decoded input frame carries a route generation. Results from a prior
  generation are discarded, and all one-frame queues plus the selected-stream
  encoder are flushed during a switch. A frame from device A therefore cannot
  be returned to device B.
- Return and selected-stream outputs use a real 30 FPS wall clock. They may hold the
  most recent processed frame across a short inference variance, but expose
  every repeat in health telemetry and close after one second of input
  staleness so a failed upstream still triggers client fallback.

## Stable client camera rule

Each platform client implements a two-source mux behind one persistent camera
identity and one fixed format:

1. Use the paired Windows return while frames are fresh.
2. Fall back to the same device's local raw frames when the return is absent or
   stale.
3. Never unregister, rename, renumber, or renegotiate the OS camera during that
   change.

On Arch, one long-lived receiver owns the single upstream `v4l2loopback`
output `/dev/video42`, published through the stable alias
`/dev/deep-live-cam` and ordinary application-facing card label `Razer Kiyo`.
The physical USB webcam retains its original nodes, driver, and applications.
A socket-controlled policy selects the dedicated local-model result on UDP
11006, Windows returns, or raw bypass without restarting the sink. Automatic
mode picks local processed, direct slot return, selected Windows SRT stream,
then the local USB raw feed. The receiver relays the selected Windows stream
to localhost UDP 11003 and the local-model result to UDP 11007 for passive
manager previews. The manager also reads the capture owner's separate raw copy
on localhost UDP 11001; it never opens a physical or virtual V4L2 node or
competes with the system mux on UDP 11000.
The physical-camera encoder publishes a third private copy on UDP 11002 to an
independent copy-only SRT worker and a fourth on UDP 11005 to the local model.
Windows reconnects therefore never reopen or stall the camera capture owner.

On Android, Camera2 ID 120 and its external provider remain registered. A
single FIFO producer switches between the paired processed return and a local
copy of the phone encoder. Starting/stopping the manager or changing the
Windows selection never opens or closes the phone's physical camera.

## Configuration boundaries

The native manager talks to three separate interfaces:

- Windows processing adapter: face swap or passthrough, source image, blend,
  enhancement, and quality settings through TCP 8090.
- Arch V4L2 adapter: camera controls are handed to the already-owning sender;
  preview code never opens the camera.
- Android Camera2 adapter: supported capture-request controls are handed to the
  already-owning bridge service; ADB is management-only, not video transport.

Generic SRT slots advertise no camera controls unless their client provides an
adapter. Unsupported controls are hidden rather than guessed.

## APIs and persisted state

- `GET /api/devices`: all five slots, selected device, fixed ports, selected-stream
  endpoint, and selected-route runtime state.
- `POST /api/devices/select`: select one enabled configured device.
- `GET/POST /api/config`: Windows processing settings. `processing_mode` is
  explicitly `face_swap` or `passthrough`.
- `GET/POST /api/source`: current source picture. The client sends the SHA-256
  digest of the exact uploaded bytes in `X-Source-Identifier`; Windows rejects
  a mismatch, atomically stores the image and metadata, and reports that
  identifier from `/api/config` only while the persisted pair verifies. The
  manager calls the identity synchronized only when this value equals its
  currently selected content-addressed history entry.
- `GET /healthz`: selected input, paired return, selected SRT stream, processing, and
  generation health.

The health document contains transport interval/jitter, estimated gaps, queue
overwrites, held-frame counts, and a temporal tracker snapshot (detection
misses, flow/hold frames, face pixel density, detector-correction jitter,
reacquisitions, and swap transitions). Face-local diagnostics include excess
flicker, detail ratio, luma/chroma mismatch, boundary stability, and a labeled
stability score. Full processing latency and diagnostic-gate overhead are
reported separately. These measurements do not alter routing ownership.

Windows persists slot registration in
`runtime/network-live/devices.json`. The desktop application is the intended
control surface; the HTTP service is a LAN JSON API and serves no web camera
interface.
