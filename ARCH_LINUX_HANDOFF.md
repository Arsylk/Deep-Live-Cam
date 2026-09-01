# Deep-Live-Cam LAN Pipeline — Arch Linux Handoff

Generated on 2026-08-02 for transfer from the Windows 11 CUDA device to the
Arch Linux camera device.

This file contains the complete user-visible conversation so far, followed by
a consolidated technical handoff for implementing the Arch Linux side.

> **Current architecture note (2026-08-11):** The historical single-port and
> browser-control material below is superseded by
> [`CAMERA_NETWORK_PROTOCOL.md`](CAMERA_NETWORK_PROTOCOL.md). The current design
> reserves five isolated input/return pairs (UDP 10000–10009), selects exactly
> one on Windows, and publishes the selected result through an SRT listener on
> `192.168.1.35:10010`, pulled by the Arch receiver.
> Android is slot 0 and Arch is slot 1. Every client performs raw/processed
> failover behind one persistent OS camera identity. Day-to-day operation is
> entirely through `deep-live-cam-tester`; TCP 8090 is its internal LAN JSON API
> and serves no browser camera page.

> **Quality/stability overhaul (2026-08-11):** The current tree adds
> detector-plus-optical-flow face tracking, miss hysteresis, motion-compensated
> face-local smoothing, temporally bounded LAB color matching, seam-safe
> sharpening, true constant-rate returns, queue/cadence telemetry, and
> face-local A/B metrics. Android bridge v0.3.3 and module v0.4.0 are deployed
> on `192.168.1.12`. The bridge now applies a sensor-aware, aspect-preserving
> GPU orientation transform with persisted Auto/0/90/180/270 controls. Measured
> capture is 29.9 FPS / 10.04 Mbps with 0.00 ms sensor jitter and no estimated
> drops. The first updated Windows
> overlay was deployed and exercised live. That run exposed and led to fixes
> for route-teardown races, unbounded flow grace, CPU-heavy full-frame
> compositing, SRT payload-mode mismatch, and TensorRT first-use stalls. The
> second benchmark revision passes 63 tests and is ready for activation with
> `tools/deploy_windows_multidevice.py --activate`; completing its live A/B run
> still requires `192.168.1.35` to be powered on and reachable over SSH.

> **DroidCam coexistence (2026-08-12):** The Arch installer now reserves a
> third endpoint at `/dev/video50` and exposes it as `/dev/droidcam`. Project
> wrappers for both `droidcam` and `droidcam-cli` always select that endpoint,
> while the processed shadow pair remains `/dev/video0` and `/dev/video1`.
> The shadow manager accepts auxiliary loopbacks while every loopback remains
> in the standard virtual sysfs hierarchy.

> **Arch kernel safety repair (2026-08-12):** The custom v4l2loopback package
> is now `0.15.4-3` / DKMS `0.15.4.safe1`. It retains only the QUERYCAP
> `driver_name` and `bus_info` overrides. The custom `parent_device` parameter,
> writable `parent` sysfs attribute, USB traversal, and runtime `device_move()`
> code were removed because they created unsafe USB lifetime and power-ordering
> dependencies during shadow-mode UVC rebinds. The safe source builds and has
> matching vermagic on stock Arch 7.1.8, Zen 7.1.8, and CachyOS 7.1.8. rEFInd's
> active `extra_kernel_version_strings` list was also repaired to put longer
> CachyOS variants first and include `linux-zen`; each of the three kernels now
> resolves to its matching initramfs. Rollback files are under
> `/var/backups/deep-live-cam-kernel-fix-20260812`.

## Conversation transcript

### User

> i am implementing a DeepCamLive camera pipeline where:
> - input camera is on different PC with arch linux
> - input video gets encoded and sent over lan to this device with win11 and cuda
> - this device runs a live processing and transforming of the video feed with the use of DeepCamLive project
> - this device sends the processed video back over lan to the arch linux PC
> - the arch linux PC has a camera device which is neatly encapsulating that entire flow
>
> this is the win11 with cuda device, plan what you will need to implement minimum unencessary friction or complexity, fully working as in specs system on this device only for now

### Assistant

The Windows side should become a bounded-latency SRT video appliance:

```text
Arch camera
  → H.264/MPEG-TS over SRT
  → Windows FFmpeg decoder
  → existing Deep-Live-Cam CUDA pipeline
  → Windows FFmpeg/NVENC encoder
  → H.264/MPEG-TS over SRT
  → Arch receiver/virtual camera
```

SRT was selected because the installed Windows FFmpeg supports it, it provides
packet-loss recovery suitable for a LAN, and it does not require deploying an
RTSP or WebRTC service.

Confirmed Windows environment:

- NVIDIA GeForce RTX 2060, 6 GB
- ONNX Runtime 1.26 with CUDA and TensorRT providers
- FFmpeg 8.1.2 with SRT, NVDEC, and NVENC
- `h264_nvenc`, `hevc_nvenc`, and `av1_nvenc` available
- Existing live pipeline already uses capture, processing, and output queues
- Existing live input was limited to local DirectShow cameras
- Existing processed frames terminated at the Qt preview

Proposed wire contract:

| Direction | Default endpoint | SRT role |
|---|---|---|
| Arch → Windows | `srt://0.0.0.0:10000` | Windows listens |
| Windows → Arch | `srt://ARCH_IP:10001` | Windows calls |

Both streams use H.264 in MPEG-TS, video only, fixed resolution and frame rate,
and approximately 100 ms SRT latency. The initial target is 1280×720 at 30 FPS
and 4 Mbps.

The Windows plan was to add FFmpeg-backed SRT input and NVENC output, keep
one-frame overwrite queues to bound latency, expose GUI/CLI configuration, add
clean failure recovery, diagnostics, and later test using synthetic streams.

### User

> ignore the entire testing side for now, just go for a working prototype first, remember that setting the various params that DeepCamLive (such as the face to swap with, slider values, etc) needs to be confdigurable from the arch linux PC side

### Assistant

The prototype direction was changed to expose a browser control panel from the
Windows process. This lets the Arch PC upload a source face and change live
Deep-Live-Cam settings without interacting with the Windows desktop.

The remotely relevant controls identified in the existing application were:

- opacity/transparency;
- sharpness;
- original-mouth mask size;
- many-face mode;
- input mirroring;
- FPS overlay;
- temporal interpolation/smoothing and its weight;
- enhancer selection (the prototype exposes None and GFPGAN).

Network address, resolution, FPS, and bitrate remain launch-time parameters in
the prototype because changing those requires restarting the FFmpeg stream legs.

### User

> also have in mind that this device will likely be intended to be hands off with no way to manually restart if it crashes, so have proper healtchecks and auto resume solutions implemented to make it never go offline or inujsable

### Assistant

Hands-off recovery was added as an operational requirement. The service exposes
machine-readable health, reconnects failed FFmpeg legs, continues waiting when
either peer disappears, and includes a watchdog plus Windows Task Scheduler
startup/restart support.

The remote settings and uploaded source face persist under
`runtime/network-live`. Health distinguishes an alive service with a missing
Arch peer from a hung Windows pipeline. A peer outage causes reconnection. A
hung decoder or encoder is recycled. A CUDA-processing hang terminates Python
so the outer launcher can restart the whole service.

### Assistant implementation result

The Windows prototype was implemented with:

- SRT input on UDP port 10000;
- CUDA Deep-Live-Cam processing;
- NVENC H.264 SRT output to Arch on UDP port 10001;
- browser control panel on TCP port 8090;
- remote source-face upload;
- remote processing controls;
- persisted face and settings;
- `/healthz` JSON endpoint;
- automatic FFmpeg reconnection;
- watchdog recovery for stalled input, output, or CUDA processing;
- persistent command launcher restart loop;
- Task Scheduler boot installer;
- private-network Windows Firewall configuration.

Manual Windows launch:

```powershell
.\run-network-service.cmd 192.168.1.20
```

The address is replaced with the Arch PC's LAN address. From Arch, the control
panel is opened at:

```text
http://WINDOWS_IP:8090/
```

Hands-off Windows installation, from elevated PowerShell:

```powershell
.\install-network-autostart.ps1 -ArchIp 192.168.1.20
```

Health monitoring endpoint:

```text
http://WINDOWS_IP:8090/healthz
```

Basic Python compilation and CLI checks passed. Full streaming tests were
deliberately not implemented or run at the user's request.

### User

> export this entire converwsation into a single file to be picked up on the arch linux side

## Current Windows implementation

The relevant repository files on Windows are:

| File | Purpose |
|---|---|
| `run_network.py` | Headless service entry point, health state, watchdog, lifecycle |
| `modules/live_stream.py` | FFmpeg SRT input/output and latest-frame buffers |
| `modules/live_processor.py` | Existing Deep-Live-Cam frame transformation adapted to network frames |
| `modules/remote_control.py` | Browser UI, JSON configuration API, source upload, persistence |
| `run-network-service.cmd` | Infinite restart loop around the Python process |
| `install-network-autostart.ps1` | Task Scheduler and Windows Firewall setup |
| `NETWORK_PROTOTYPE.md` | Short Windows operating guide |

The Windows service is started with the Arch address as its required output
destination. Defaults are:

```text
Input:       srt://0.0.0.0:10000?mode=listener&latency=100000
Output:      srt://ARCH_IP:10001?mode=caller&latency=100000
Video:       1280x720, 30 FPS
Codec:       H.264 via NVENC
Container:   MPEG-TS
Bitrate:     4M CBR
GOP:         30 frames
B-frames:    disabled
Pixel format:yuv420p
Audio:       none
Control:     HTTP TCP 8090
```

The numerical SRT `latency` value is expressed in microseconds by FFmpeg;
`100000` represents 100 ms.

## Required Arch Linux implementation

The Arch side must provide three functions:

```text
physical camera → SRT sender to Windows:10000
Windows SRT output → decoder → V4L2 virtual camera
supervision/autostart for both processes and the virtual device
```

The simplest intended Linux stack is:

- FFmpeg;
- `v4l2loopback` kernel module;
- two systemd services, or one small supervised service containing both FFmpeg
  legs;
- optional health polling of the Windows `/healthz` endpoint.

### Expected outbound sender

The Arch sender should capture the physical V4L2 camera, convert/scale to the
fixed contract, encode H.264 with a low-latency configuration, mux MPEG-TS, and
call the Windows listener:

```text
/dev/PHYSICAL_CAMERA
  → 1280x720 BGR/YUV at 30 FPS
  → low-latency H.264
  → MPEG-TS
  → srt://WINDOWS_IP:10000?mode=caller&latency=100000
```

The exact Linux encoder can be selected based on Arch hardware. `libx264` is the
portable baseline. A hardware encoder may be substituted if present, but that
is not required by the Windows contract.

### Expected inbound receiver

The Arch receiver must listen before the Windows caller connects:

```text
srt://0.0.0.0:10001?mode=listener&latency=100000
  → MPEG-TS/H.264 decode
  → fixed 1280x720 at 30 FPS
  → V4L2-compatible pixel format such as yuv420p or yuyv422
  → /dev/videoN provided by v4l2loopback
```

The resulting `/dev/videoN` is the camera exposed to ordinary Arch
applications. Give it a stable label such as `DeepLiveCam` and preferably a
stable symlink rather than relying on a numerical `/dev/videoN` assignment.

Configure `v4l2loopback` with `exclusive_caps=1` for compatibility with browser
and conferencing applications. Whether the sink needs `yuyv422` or another raw
format should be determined from the consumer application's V4L2 expectations.

### Recovery expectations

The Arch implementation should mirror the Windows hands-off behavior:

- load `v4l2loopback` at boot;
- start the SRT receiver before, or independently of, the sender;
- use systemd `Restart=always` with a short restart delay;
- avoid unbounded queues and favor current frames over old frames;
- tolerate the Windows machine being absent or rebooting;
- automatically reconnect both FFmpeg legs;
- keep the virtual camera device present even during network interruption;
- ideally publish the most recent frame or a neutral placeholder while the
  processed stream is unavailable;
- optionally poll `http://WINDOWS_IP:8090/healthz` for observability, but do not
  restart Arch components solely because `streaming` is false;
- distinguish Windows-control-plane reachability from active video flow.

No software-only design can guarantee literal zero downtime through power,
kernel, hardware, or network failure. The intended requirement is automatic
recovery without a person touching the Windows device.

## Windows HTTP interface available to Arch

### Browser panel

```text
GET http://WINDOWS_IP:8090/
```

### Read configuration

```text
GET http://WINDOWS_IP:8090/api/config
```

The response includes:

```json
{
  "opacity": 1.0,
  "sharpness": 0.0,
  "mouth_mask_size": 0.0,
  "interpolation_weight": 0.0,
  "many_faces": false,
  "live_mirror": false,
  "show_fps": false,
  "enable_interpolation": true,
  "enhancer": "none",
  "source_configured": false
}
```

### Change configuration

```text
POST http://WINDOWS_IP:8090/api/config
Content-Type: application/json
```

Supported fields and ranges:

| Field | Type/range |
|---|---|
| `opacity` | number, 0.0–1.0 |
| `sharpness` | number, 0.0–5.0 |
| `mouth_mask_size` | number, 0–100 |
| `interpolation_weight` | number, 0.0–1.0 |
| `many_faces` | boolean |
| `live_mirror` | boolean |
| `show_fps` | boolean |
| `enable_interpolation` | boolean |
| `enhancer` | `none` or `gfpgan` |

### Upload a source face

```text
POST http://WINDOWS_IP:8090/api/source
Content-Type: image/jpeg
X-Filename: source.jpg

<raw image bytes>
```

The maximum upload size is 20 MB. The browser control panel performs this call
directly. The uploaded image is persisted on Windows.

### Read health

```text
GET http://WINDOWS_IP:8090/healthz
```

Representative response:

```json
{
  "healthy": true,
  "streaming": true,
  "state": "streaming",
  "uptime_seconds": 120.5,
  "source_configured": true,
  "input": {
    "url": "srt://0.0.0.0:10000?mode=listener&latency=100000",
    "frames": 3510,
    "last_frame_age": 0.02
  },
  "processing": {
    "frames": 870,
    "fps": 7.4,
    "last_frame_age": 0.08
  },
  "output": {
    "url": "srt://ARCH_IP:10001?mode=caller&latency=100000",
    "frames": 3490,
    "last_frame_age": 0.01
  }
}
```

`healthy: true` means the Windows service itself remains operational.
`streaming: false` may simply mean an Arch sender or receiver is not connected.

## Important prototype constraints

- Video only; audio is not transported.
- The stream is fixed at launch-time resolution and FPS.
- The browser/API currently has no authentication. Use only on a trusted LAN or
  add firewall restrictions/authentication before exposing it elsewhere.
- Face mapping for individually identified target faces is not remotely exposed;
  the prototype supports the normal single-source mode and many-face toggle.
- When CUDA processing is slower than 30 FPS, Windows repeats the newest
  processed frame at the fixed output cadence rather than accumulating latency.
- The source and processed queues retain only the newest frame.
- Full cross-machine streaming tests have not yet been performed.

## Information still needed on the Arch side

Before finalizing its systemd units and FFmpeg commands, inspect or decide:

1. The stable physical-camera path, preferably `/dev/v4l/by-id/...`.
2. Camera-supported resolutions, frame rates, and input pixel formats.
3. The desired stable virtual-camera device name/path.
4. Whether the Arch machine has a usable hardware H.264 encoder.
5. The Windows and Arch static/reserved LAN addresses.
6. Which applications will consume the virtual camera and their accepted V4L2
   formats.

This is the boundary at which the Arch-side work should continue.
