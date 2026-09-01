# Deep-Live-Cam Arch slot client and native manager

This directory contains the Arch/CachyOS client for the five-slot Windows
router described in [the network contract](../CAMERA_NETWORK_PROTOCOL.md).
Windows is `192.168.1.35`; this machine is slot 1 at `192.168.1.11`.

## Fixed routes

```text
Arch Sonix camera owner (sender)
  ├─ H.264/MPEG-TS/UDP → localhost UDP 11000 (system raw fallback; puppet-recorder record port)
  ├─ H.264/MPEG-TS/UDP → localhost UDP 11001 (manager-only preview)
  ├─ H.264/MPEG-TS/UDP → localhost UDP 11002 (network-worker input)
  └─ H.264/MPEG-TS/UDP → localhost UDP 11005 (local model input)

Phone capture owner (dev.vcam.app, native portrait 720x1280@30 hardware H.264)
  ├─ H.264/SRT/UDP → Arch 0.0.0.0:10001 (single phone listener; camera in)
  └─ on-device: app → TCP 127.0.0.1:10020 → local ffmpeg → SRT; mic relay 10025

Local Native-256/NCNN-Vulkan processor (input now native portrait 720x1280)
  ├─ dedicated phone-relay source → localhost UDP 11009
  ├─ exact encoded system source → localhost UDP 11006 (native 720x1280)
  │    └─ receiver copy-only relay → localhost UDP 11007 for manager comparison
  └─ exact encoded preview tap   → localhost UDP 11012 (phone cam; recorder "Phone")

Exclusive Arch phone-return relay (source off | local | windows)
  ├─ selected source → Android SRT UDP 10001 (phone return)
  └─ same encoded network stream → localhost UDP 11004 for passive preview

Independent copy-only network worker
  └─ H.264/MPEG-TS/SRT → Windows UDP 10002

Windows selected slot 1
  └─ processed H.264/MPEG-TS/SRT → Arch UDP 10003

Windows selected-stream listener
  └─ Arch pulls processed H.264/MPEG-TS/SRT ← 192.168.1.35:10010
     ├─ receiver copy-only relay → localhost UDP 11003 for manager preview
     └─ dedicated phone-relay source → localhost UDP 11008

Receiver file_relay (prerecorded / assembler)
  ├─ reads the MP4 file directly (UDP 11010 is reserved, intentionally unused)
  ├─ transformed system-camera feed → /dev/deep-live-cam (locked 1280x720 YUYV)
  ├─ manager output preview tap → localhost UDP 11003
  ├─ framing preview tap → localhost UDP 11011 (Input tab drag/zoom preview)
  └─ phone relay taps → 11008 (windows mode) / 11009 (local mode)

Puppet recorder (guided takes)
  └─ live preview of the take → localhost UDP 11013 (encoder tee; while recording)
```

### Verified port map (live-verified 2026-09-01)

Every UDP port has exactly one producer; each tap has one named consumer.
UDPortrait readers compete for unicast datagrams — never attach a second
reader to a port that already has one (kill stale ffmpeg readers first).

| Port | Transport | Producer → consumer | Notes |
|---|---|---|---|
| 10000 | SRT | Arch → Windows slot 0 input | slot N uses 10000+2N |
| 10001 | SRT | Phone → Arch camera in; Arch → phone return | the phone's single SRT endpoint (input + return) |
| 10002 | SRT | Arch → Windows slot 1 input (webcam) | WINDOWS_INPUT_PORT |
| 10003 | SRT | Windows → Arch processed return | listener on Arch |
| 10010 | SRT/UDP | Arch → Windows selected stream; local tap on 127.0.0.1:10010 for the phone relay | dual use |
| 10020 | TCP (phone) | capture app → on-device ffmpeg | phone-local |
| 11000 | UDP | sender → puppet-recorder record port | webcam raw fallback |
| 11001 | UDP | sender → manager raw preview | webcam |
| 11002 | UDP | sender → network worker input | |
| 11003 | UDP | receiver → manager output preview | |
| 11004 | UDP | phone-return relay → passive preview | writer only while the relay runs |
| 11005 | UDP | sender → local processor (webcam input) | |
| 11006 | UDP | processor → receiver (local_processed) | native phone dims 720x1280 |
| 11007 | UDP | receiver → manager processed preview | copy of 11006 content |
| 11008 | UDP | receiver → phone-return relay (windows source) | reader only in windows mode |
| 11009 | UDP | processor → phone-return relay (local source) | reader only in local mode |
| 11010 | — | RESERVED, intentionally unused | file_relay reads the MP4 directly |
| 11011 | UDP | receiver → manager framing preview (Input tab) | reader while prerecorded/assembler input is active |
| 11012 | UDP | processor → phone camera preview tap | recorder "Phone" source; native 720x1280 |
| 11013 | UDP | recorder encoder tee → its own preview pane | writer only while a take records |

The long-lived receiver writes the 1280×720 YUYV/30 feed to the independent
`/dev/deep-live-cam` virtual camera (`/dev/video42`). Inputs that are not
already 1280×720 (the phone's native 720×1280 portrait stream, prerecorded
files of any shape) are fitted into that locked output at delivery — the only
point in the chain where the phone's native stream is reshaped. The manager
reconciles its Processor selection to a receiver source policy without
restarting the sink or either camera owner:

- **Local Native-256:** local processed UDP 11006, then raw UDP 11000.
- **Windows:** direct return UDP 10003, selected SRT stream, then raw UDP 11000.
- **Automatic:** local processed, direct return, selected stream, then raw.
- **Raw bypass:** raw UDP 11000 only.

The checked-in and installed defaults are `LOCAL_PROCESSED_PORT=11006`,
`LOCAL_PROCESSED_PREVIEW_PORT=11007`, and `RECEIVER_SOURCE=windows`.
That environment value is only the first-start fallback. Successful source
changes are atomically persisted in
`/var/lib/deep-live-cam/receiver-source.json`, so a receiver reconstruction or
service restart restores the manager's last selection. A malformed state file
fails safely to raw bypass instead of selecting a processor implicitly.

Selection is an allow-listed request to
`/run/deep-live-cam/receiver-control.sock`; all decoders remain passive and
the public V4L2 identities stay open. Changing processors only changes the
receiver's input priority; it does not replace or restart the camera sink.

This means an app never loses or reselects its camera when Windows changes
devices, processing is toggled, or the LAN fails. When the Android phone is
selected, its processed SRT stream becomes systemwide on Arch through those
same nodes. When no Windows result is fresh, the Arch camera immediately shows
its own raw frames behind the same identity.

## Ownership and busy-device behavior

- `deep-live-cam-sender.service` is the only physical Sonix camera owner.
- Desktop camera discovery exposes only the processed `Xiaomi Cam` endpoint.
  The physical Sonix video and metadata nodes are reserved to the
  `deep-live-cam` service account, so a browser cannot choose the owner's busy
  input by mistake. This does not rename, unbind, or interrupt that input.
- Its capture encoder never restarts merely because Windows is unavailable or
  the selected slot changes; a separate copy-only worker owns SRT retries.
- `deep-live-cam-receiver.service` is the only V4L2 loopback producer.
- The native manager reads dedicated receiver/capture-owner relays on localhost
  ports 11003 and 11001. It never shares the system mux's port 11000. It
  never opens a physical or virtual camera.
- Camera settings use `camera_adapters.py`. The Arch adapter sends an
  allow-listed JSON request to `/run/deep-live-cam/sender-control.sock`; the
  already-owning sender applies it. The Android adapter hands supported values
  to the already-owning Camera2 bridge service over management-only ADB.
- Camera sliders and choices preview through those owners after a short
  debounce. They remain session-only until **Save current settings** is used.
  Arch capture resolution is selectable, but Save only stages it for the
  capture owner's next natural start. Save never restarts or reopens an active
  camera session; hot image controls are sent through the existing owner.
- The UI contains no stream-service stop/restart buttons. Route selection and
  processing changes therefore cannot produce `device busy` by taking camera
  ownership away from an application.
- The clean topology also installs
  `/etc/modprobe.d/deep-live-cam-disable-v4l2loopback-dc.conf`. It neutralizes
  stale distro/DroidCam requests for the obsolete `v4l2loopback_dc` fork so
  only the configured upstream `v4l2loopback` instance can allocate a virtual
  camera node at boot.

## Prerecorded video workflow

The system supports recording a camera source, rendering it offline with
unlimited time per frame for perfect face-swap quality, and replaying it as a
live input source. The prerecorded video appears in the manager alongside phone
front/back/arch webcam as a selectable system camera policy.

See **[PRERECORDED_WORKFLOW.md](PRERECORDED_WORKFLOW.md)** for the complete
guide, including:

- Recording from any camera (`record_and_render.py record`)
- Offline rendering with no time constraints (`record_and_render.py render`)
- Streaming to the receiver in a loop (`prerecorded_relay.py --loop`)
- Selecting "Prerecorded video" in the manager's routing page

The receiver decodes the prerecorded stream using the same phone-accurate
transform as live sources (rotate-to-upright, aspect-correct scale,
center-crop) so the output stays locked at 1280×720 with no distortion or
letterboxing.

Quick example:

```bash
# Record 10 seconds, render offline, stream in a loop
python3 arch-linux/bin/record_and_render.py all \
    --device /dev/video0 \
    --duration 10.0 \
    --face path/to/face.jpg

# Then select "Prerecorded video" in the manager
```

## Native manager

Launch the existing desktop entry or run:

```bash
deep-live-cam-tester
```

The system desktop entry remains linked to `/usr/local/bin/deep-live-cam-tester`;
a development checkout may install a standard per-user entry that launches its
current `arch-linux/bin/tester.py` directly. Both paths start the same
application. `--self-check` stays a non-GUI configuration diagnostic and also
reports the manager modules it found.

The window has exactly three primary tabs across the top: **Processor**,
**Input**, and **Output**. The header ribbon continuously summarizes the same
three decisions as `INPUT → PROCESSOR → SYSTEM CAMERA`; transport slots,
service processes, and diagnostic pages are not mixed into ordinary controls.
There is no settings dock and there are no nested tabs. The minimum usable
window is 980 × 680; card grids re-flow into more columns on larger displays.

### Processor tab

The processor selector is a machine choice, not a free-form model selector:

- **Windows 11 remote** uses the production INSwapper-128 ONNX model on CUDA.
- **This Arch workstation** uses the native-256 semantic-mask model through
  ncnn/Vulkan on the RX 570. That installed checkpoint is explicitly labelled
  `development` and unqualified; a live transport or visible pixel change is
  not presented as proof of identity quality.

The model/backend readout is therefore fixed by the selected host and cannot be
silently changed by a preset. Both processors expose the same processing form:
preset, opacity, colour match, mouth mask, sharpness, multiple faces,
tracking and detector controls, enhancement/interpolation, FPS overlay, and the
bounded output-quality guard. The shared advanced form also owns the detail,
checkerboard, wavelet, semantic-boundary, and skin-texture repair strengths;
these are not hidden per-host defaults. Changing one control updates the single desired
configuration immediately; it does not create a Windows copy and an Arch copy
that can drift apart.

The same tab contains the identity picture selector. It shows the currently
selected picture and a horizontally scrollable, one-click history of
content-addressed recent pictures. Choosing a picture makes it the durable
local identity first, applies its unique cached path to the Arch processor, and
uploads it to Windows when Windows is reachable. A failed upload stays pending
and retries instead of reverting the picture shown by the manager. Checkpoint
qualification, measured face-core change, and actual identity verification
remain three separate evidence statements.

The Windows upload carries the SHA-256 digest of the exact transmitted bytes
as `X-Source-Identifier`. Windows rejects a mismatched body, atomically stores
the normalized image plus verification metadata, and reports the original
upload identifier only while that pair still verifies. The manager therefore
labels the source synchronized only when the remote identifier equals the
currently selected history entry; `source_configured=true` by itself is not
accepted as proof that the right picture is active.

### Input tab

Input selection is semantic and has exactly three choices: **Arch USB webcam**,
**Phone front camera**, and **Phone back camera**. UDP/SRT ports and the five
Windows slots remain implementation details. Selecting an input reconciles the
same choice to both processor routes without opening `/dev/video*`, Camera2, a
virtual camera, or an application-owned camera session.

The Arch webcam receives the measured natural-indoor capture profile. Phone
inputs receive conservative Camera2 defaults for 1280×720/30, hardware H.264,
automatic rotation/exposure/white balance, and video stabilization where the
owner reports support. These are best-effort low-loss defaults applied through
the process that already owns the camera. Front and back share the persistent
phone transport; changing lens does not replace the decoder or public camera
identity. Switching between the phone transport and the Arch-owner relay
rebuilds only the processor's private input decoder, never its output encoder or
either camera owner.

Below the three choices, capture controls are generated from `camera_schema()`
for the selected owner and grouped by meaning: profile/capture, tone/colour,
exposure/white balance, and orientation/lens/stabilization where supported.
Live changes are debounced through that owner. The manager itself never probes
the physical camera and therefore does not introduce `device busy` failures.

The Android front/back adapter also advertises a live 100–300% centred sensor
zoom. At 100%, a quarter-turn phone image uses an aspect-correct centre-crop to
fill the 1280×720 transport. It is never squeezed into pillar bars, so Windows
receives the useful face pixels at full canvas scale instead of detecting a
small image and enlarging the 128px swap afterward. Selecting a phone input
restores the lossless 100% sensor default; additional zoom is explicit and does
not reopen Camera2 or reconnect the stream.

### Output tab

There are two independent destinations and both may remain enabled:

- **Arch virtual webcam** is the one safe custom `v4l2loopback` device at
  `/dev/video42`, exposed through `/dev/deep-live-cam` with the application-facing
  card label `Xiaomi Cam`. Its QUERYCAP driver and bus strings mirror the
  connected USB webcam; the physical input keeps its own nodes and lifecycle.
- **Phone processed camera** continuously returns the processed stream through
  the Android bridge as stable Camera2 ID 120. The separately scoped Xposed
  route can substitute ID 120 when a selected application requests the default
  front camera; the manager reports that hook as installed or active only when
  the phone can actually prove it.

The same tab provides horizontal mirroring and rotations of 0°, 90°, 180°,
and 270°. It also owns the explicit **Enable face swapping** checkbox. Turning
that checkbox off stores shared `processing_mode=passthrough` and immediately
asks the selected processor to forward unchanged frames; turning it on stores
`processing_mode=face_swap`. Neither action interrupts transport or replaces a
camera session. Destination toggles and transforms are sent to the already-running
Arch receiver and Android bridge. They change live frame delivery without
unregistering a camera, restarting the Arch sink, restarting the Android
provider/producer, or changing public camera IDs, so applications with an open
camera session keep that session. The preview below the controls is a passive
decode of the exact selected result and never opens a camera device.

### Durable state and reconnect behavior

The authoritative manager document is
`$XDG_CONFIG_HOME/deep-live-cam/manager-state.json` (normally
`~/.config/deep-live-cam/manager-state.json`). It atomically stores the selected
processor, semantic input, enabled outputs, output transform, active source
identifier, shared processing values, and a monotonic revision. A click updates
this document and the visible control first, then asynchronously reconciles the
available local and remote workers.

An offline processor is shown as saved/pending, but all settings remain
editable. When Windows comes back, the manager compares its reported effective
configuration with the durable desired document, posts only differences, and
retries the active source picture. A late or older Windows response cannot
replace newer local intent or move the controls. The Arch processor exposes the
same desired/effective/revision contract through
`$XDG_RUNTIME_DIR/deep-live-cam/processor-control.sock`; input and processing
changes are applied at runtime without restarting its output worker.

### Hidden diagnostics

Passive analysis and system/log widgets still exist as controller-owned
implementation surfaces, but they are deliberately not fourth and fifth tabs.
They continue to receive the immutable refresh snapshot, baseline/quality
measurements, bounded event log, and endpoint state used by status summaries and
off-screen contract tests. Detailed non-GUI inspection remains available via
`deep-live-cam-tester --self-check`, `deep-live-cam-status --json`, and the
runtime health documents. Ordinary operation needs only the three primary tabs.

### Interaction rules

Mouse-wheel input over a slider, dial, spin box, or selector scrolls the page
instead of changing the value — including controls generated from a camera
adapter schema, because the guard is installed on the application. Values stay
editable by click/drag, keyboard arrows, and direct text entry. Focus is
visible on every interactive class, state is always a word plus a glyph rather
than colour alone, and a health refresh never moves a control, resets an edit,
steals focus, or changes the page or splitter sizes.

### Internals

`bin/tester.py` is the entry point; the application is the `bin/dlc_manager`
package installed beside it:

```text
dlc_manager/desired_state.py durable intent, validation, revisions, reconciliation
dlc_manager/contracts.py     ports, policies, scopes, camera-control groups
dlc_manager/health.py        reading and interpreting service health documents
dlc_manager/viewmodel.py     one immutable ManagerView per refresh
dlc_manager/services.py      Windows JSON API, helper processes, systemd probe
dlc_manager/decoders.py      passive relay decoders and log rate limiting
dlc_manager/quality.py       passive measurement of decoded previews
dlc_manager/baseline.py      the registered reproducible baseline and its limits
dlc_manager/wheel.py         the application-wide wheel guard
dlc_manager/widgets.py       pills, cards, metric rows, wheel-safe controls
dlc_manager/theme.py         design tokens and the single stylesheet
dlc_manager/phone_preview.py the passive phone-return window
dlc_manager/phone_route.py   exclusive return ownership truth table
dlc_manager/preview_transform.py output-oriented preview transform
dlc_manager/pages/processing.py  Processor tab
dlc_manager/pages/input.py       Input tab and owner-backed camera controls
dlc_manager/pages/output.py      Output tab, transforms, and exact preview
dlc_manager/pages/analysis.py    hidden passive quality implementation
dlc_manager/pages/system.py      hidden snapshot and bounded-log implementation
dlc_manager/shell.py             tabs, refresh loop, durable action wiring
local_processor_control.py       Arch desired/effective Unix-socket client
windows_phone_relay.py           persistent exclusive Android SRT caller
configure_phone_relay.py         allow-listed relay control client
```

Pages never read raw service JSON: a refresh collects documents, builds one
`ManagerView`, and hands that value to each page. Network and subprocess work
is asynchronous and bounded by timeouts, so an unreachable Windows host or a
missing helper degrades the interface instead of blocking it.

It uses the LAN JSON API on Windows TCP 8090 internally. No browser page,
certificate, web camera permission, or internet connection is required.

## Local phone-front processing route

`bin/android_stream_processor.py` is the headless phone-return route for a
local face-swap model on the RX 570. The Android Camera2 owner sends its front
camera to Arch SRT port 10001. The processor opens neither an Android nor an
Arch camera device:

```text
Android front Camera2 -> SRT :10001 -> INSwapper-128/ncnn
                    local relay source UDP :11009 <----------┤
             stable Arch cameras <- local UDP :11006 <------┘
                         Camera2 120 <- Android SRT :10001 <- exclusive relay
             output-oriented preview <- local UDP :11004 <---┘
```

The processor uses one encoder and sends a dedicated copy to UDP 11009. A
persistent relay is the only Arch process allowed to call Android SRT 10001;
it selects `local`, `windows`, or `off` with close-before-open ownership. Its
UDP 11004 tee carries the same network stream, and the preview applies the
desired rotation/mirror locally for display. On the manager's Output tab, press
**Open phone return preview** to open it. The reader never opens a camera, runs
inference, starts another encoder, or affects the Android connection.
Closing the preview window only stops its local decoder. Port 11003 remains
reserved for the selected Windows-stream preview, so the two sources cannot
interleave. The receiver independently relays the selected local-model source
from UDP 11006 to UDP 11007 for the main Output preview; this avoids sharing the
phone-return preview port.

The identity source defaults to the manager's active local source-history
picture (or `DLC_SOURCE_IMAGE`). The default and installed persistent service
use the production INSwapper-128/ncnn path with phone-front input:

```bash
.venv/bin/python arch-linux/bin/android_stream_processor.py \
  --input-mode android-srt --input-host 0.0.0.0 --input-port 10001 \
  --android-host 192.168.1.12 \
  --swapper-model inswapper-128 --swapper-backend ncnn \
  --detection-interval 2 \
  --phone-relay-port 11009 --preview-port 0 --system-camera-port 11006
```

Native-256 remains available only as an explicit development experiment until
a trained checkpoint passes its release gates. It is never substituted for the
production model merely because its graph can execute.

Selecting **This Arch workstation** in the manager is that explicit request:
the manager activates the installed native-256/ncnn target through the local
control socket and keeps its `development/unqualified` warning visible. The
service's standalone safe default remains INSwapper-128; a missing or invalid
native-256 pack is reported as pending and never triggers a silent fallback
while the Arch processor is the selected target.

`arch-webcam` remains available as an explicit fallback using the sender's
private UDP 11005 copy, but it is not selected by the installed service. Use
`--preflight-only` to load and validate the active source, model, ncnn
bridge, and Linux execution-provider choice without binding a network port.
Use `--dry-run` for argument/route validation only. The `android-srt` local
listener and remote return intentionally use the same port number because they
are on different hosts.

The installer places a system unit named
`deep-live-cam-phone-processed.service`, but deliberately does not enable or
start it: a valid local source picture and qualified model pack must exist
first. This keeps an unavailable Arch processor truthful instead of creating a
restart loop. The separate `deep-live-cam-phone-return-relay.service` is enabled
with the other pipeline services when installation is run with `--start`; it
restores durable relay intent and defaults fail-closed when state is absent or
corrupt. Inspect both without opening a camera:

```bash
systemctl status deep-live-cam-phone-processed.service
systemctl status deep-live-cam-phone-return-relay.service
jq . /run/deep-live-cam/android-phone-processed-health.json
jq . /run/deep-live-cam/phone-return-relay.json
```

On Linux, automatic ONNX provider selection does not mistake CUDA providers
from an installed CUDA-flavoured wheel for usable hardware on an AMD-only
host. Face analysis therefore uses a real ROCm/OpenVINO provider when present,
or CPU otherwise; face-swap inference remains on ncnn/Vulkan. Output
encoding prefers `/dev/dri/renderD128` through H.264 VAAPI and falls back to
low-latency `libx264`. The SRT return closes after stale input so Android can
fall back to raw frames behind the same Camera2 identity.

## Install

### Source, installed, and live state

The files in this checkout describe and test the intended topology; their
presence does **not** prove that `/etc`, the running Windows service, or the
phone has been upgraded. The boundaries are deliberate:

- Launching the per-user `.desktop` entry runs this checkout's current manager
  immediately. The system entry runs the copy under `/usr/local` and changes
  only after installation/deployment.
- The installed `deep-live-cam-phone-processed.service` currently uses this
  checkout's `/opt/github/Deep-Live-Cam/.venv`, models, and processor entry
  point. Installing its unit does not bundle or qualify a local model pack, so
  keep the checkout at that path (or update the unit as part of a controlled
  deployment).
- `install.sh` installs the Arch manager/services and the one-node loopback
  configuration. If another `v4l2loopback` topology is already loaded, it
  stages `/etc/deep-live-cam-arch.conf.nextboot` and leaves all open camera
  descriptors alone; `/dev/video42` is not live until the controlled reboot.
- The Windows source under `windows/` must be copied/activated on
  `192.168.1.35`. Until its API reports the desired settings revision and exact
  source identifier, the UI correctly remains pending rather than claiming
  synchronization.
- The Android source and `android/android-vcam-module-v0.4.9.zip` must be
  deployed to the phone. A ZIP in this checkout is not evidence that Camera2
  ID 120 or the scoped front-camera redirect is installed or active.

Use the manager pills, `deep-live-cam-tester --self-check`, service health
documents, and device/provider inspection after deployment to establish live
state. Unit tests establish source/contract readiness only.

```bash
sudo ./arch-linux/install.sh --windows-host 192.168.1.35
```

The installer migrates legacy shared ports to the slot-1 contract, installs the
manager behind the existing `.desktop` entry, configures one upstream
v4l2loopback output, and allows only the Windows host into the slot-1 return port UDP 10003
when a local firewall is active. Port 10010 is outbound from Arch because Arch
initiates the SRT selected-stream connection.

The machine-specific configuration is `/etc/deep-live-cam-arch.conf`. Runtime
health is under `/run/deep-live-cam/` and can be shown with:

```bash
deep-live-cam-status
deep-live-cam-status --json
```

## Clean virtual-camera identity

The default is one safe `v4l2loopback-custom` device at `/dev/video42`, with
the card label `Xiaomi Cam` and stable udev alias `/dev/deep-live-cam`. Its
QUERYCAP identity reports the physical webcam's `uvcvideo` driver and
`usb-0000:02:00.0-4` bus. The
physical USB webcam keeps its original nodes, by-id/by-path links, driver and
power lifecycle, but its video nodes are reserved to the dedicated sender
account. WirePlumber disables every other V4L2/libcamera camera, and the
udev access rule also prevents direct-V4L2 browsers from selecting the Sonix
capture or metadata node. Nothing unbinds the USB camera, takes over
`video0`/`video1`, or creates `video50`.

`exclusive_caps=1` is retained for WebRTC compatibility. Before a producer
opens video42 it truthfully advertises only V4L2 output capability; after the
receiver opens it, it becomes a capture camera. The receiver's bounded,
root-scoped post-start helper verifies `/dev/video42`, the cloned `uvcvideo`
driver, USB bus string, and `Xiaomi Cam` card name, then emits one `change`
uevent. This closes
the boot race where WirePlumber could retain a device object without creating
the browser-visible `Video/Source` node.

To deploy only this browser-camera policy without running the full installer:

```bash
sudo install -Dm0755 arch-linux/bin/publish_virtual_camera.py \
  /usr/local/lib/deep-live-cam-arch/publish_virtual_camera.py
sudo install -Dm0644 arch-linux/systemd/deep-live-cam-receiver.service \
  /etc/systemd/system/deep-live-cam-receiver.service
sudo install -Dm0644 arch-linux/config/72-deep-live-cam-camera-access.rules \
  /etc/udev/rules.d/72-deep-live-cam-camera-access.rules
sudo install -Dm0644 arch-linux/config/51-deep-live-cam-camera-policy.conf \
  /etc/wireplumber/wireplumber.conf.d/51-deep-live-cam-camera-policy.conf
sudo systemctl daemon-reload
sudo udevadm control --reload
sudo udevadm trigger --action=change --subsystem-match=video4linux
sudo udevadm settle
sudo /usr/local/lib/deep-live-cam-arch/publish_virtual_camera.py \
  --device /dev/video42 --timeout 15
systemctl --user restart wireplumber
```

The udev permission update does not invalidate the sender's existing physical
camera descriptor. Fully close and reopen the browser after the WirePlumber
restart because browsers cache `enumerateDevices()` results. An already-open
browser camera session is intentionally not killed; it must be closed once by
the application. The policy deliberately hides all additional desktop cameras,
including newly attached USB/libcamera devices. Remove the two policy files,
reload udev, restart WirePlumber, and replug/trigger the physical webcam to
restore general-purpose camera discovery.

The safe custom module clones the identity surfaces applications normally read
through `VIDIOC_QUERYCAP`, while leaving the device in virtual sysfs. The
retired USB-parent patch is intentionally not restored: it coupled the virtual
device lifetime to UVC unplug/rebind and was the part implicated in unreliable
Zen/CachyOS boot and shutdown behavior.

When upstream `v4l2loopback-dkms` or an older custom package is detected, the
installer replaces it with `v4l2loopback-custom` and writes the one-node boot
configuration. If the old module is loaded, it is never unloaded in place.
The desired application configuration is saved as
`/etc/deep-live-cam-arch.conf.nextboot`, an early-boot oneshot commits it, and
the existing services and camera file descriptors are left untouched.  Reboot
before manually restarting either camera service.  The reboot removes all
volatile old loopback nodes and restores normal udev links for the USB camera.

The former takeover helper remains only for an explicit rollback using
`install.sh --legacy-shadow`.  The configuration requires both
`LEGACY_SHADOW=1` and `SHADOW_ORIGINAL=1`; a stale single setting cannot invoke
USB unbind or node hiding.

On rEFInd systems with stock, Zen, and CachyOS kernels in the same ESP
directory, use longest-first matching so each kernel receives its own image:

```text
extra_kernel_version_strings linux-cachyos-lts,linux-cachyos-bore,linux-cachyos-bmq,linux-cachyos,linux-zen,linux-lts,linux
```

Putting `linux` before these variants makes it consume every later name;
omitting `linux-zen` leaves that kernel without an unambiguous initramfs match.

## DroidCam cleanup

Deep-Live-Cam no longer allocates a local DroidCam `/dev/video50` endpoint and
does not override the system `droidcam`, `droidcam-cli`, or DroidCam desktop
entry.  The Android phone output uses the network return and Android camera
stack, so it is independent of the single Arch virtual camera.  During
migration the installer removes only wrappers and desktop files that an older
Deep-Live-Cam release installed; package- or user-owned DroidCam files are left
alone.
