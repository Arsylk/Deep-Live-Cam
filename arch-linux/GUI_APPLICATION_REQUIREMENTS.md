# Deep-Live-Cam native manager: technical and UX requirements

## 1. Objective

Build a cohesive native desktop application for operating the complete
Deep-Live-Cam camera system from Arch Linux. This is an implementation task,
not a mock-up or a design-only exercise.

The application must make a complicated multi-node video system understandable
without hiding important state. It owns one shared desired processing
configuration and reconciles that same configuration to whichever fixed
processor is selected; an offline processor remains visibly pending and is
resynchronized when it returns. It replaces the historical side dock, nested
tabs, and long mixed-purpose settings pages with a deliberate information
architecture suitable for daily use.

The existing entry point is `arch-linux/bin/tester.py`. A ground-up internal
refactor into focused modules is encouraged, provided the installed launcher,
`--self-check`, configuration, protocol contracts, and existing functionality
remain compatible.

## 2. Runtime and deployment requirements

- Use Python and PySide6. This is a native application, not an HTML/browser UI.
- It must start through both existing desktop-entry paths:
  - the per-user entry that launches the repository `tester.py`;
  - `/usr/local/bin/deep-live-cam-tester`, which resolves to the installed app.
- It must run fully without internet access, an embedded browser, web-camera
  permission, cloud assets, telemetry, or a web account.
- The Windows control endpoint at `http://192.168.1.35:8090` is a private LAN
  JSON API, not an internet dependency. Unreachable Windows must degrade to a
  useful local/offline UI instead of blocking startup.
- Read `/etc/deep-live-cam-arch.conf`, honor a custom `STATE_DIR`, and retain
  `--self-check` as a non-GUI configuration diagnostic.
- Minimum usable window: 980 x 680. The layout must also use 1500 x 900 and
  larger displays effectively. Pages may scroll vertically, but the entire app
  must not be one giant settings scroll.
- Startup, page changes, and settings-page scrolling must never start/stop a
  capture service or open a camera device.

## 3. Non-negotiable camera-ownership invariants

The GUI is a controller and passive observer. It must never directly open any
physical or virtual V4L2 camera:

- Physical capture owner: `deep-live-cam-sender.service`.
- Stable system-camera writer: `deep-live-cam-receiver.service`.
- Stable Arch output identity: `/dev/deep-live-cam` (the one upstream
  `v4l2loopback` node `/dev/video42`, card label `Xiaomi Cam`).
- The physical USB webcam keeps its package-owned nodes, by-id/by-path links,
  and kernel driver, but desktop camera discovery cannot select it. Its video
  nodes are reserved to the capture-owner service so browsers see only the
  processed Xiaomi Cam output and never collide with that owner.
- Preview readers consume only the dedicated localhost MPEG-TS relays.
- Camera settings are sent to the existing device capture owner through the
  camera adapter/helper contract.
- Save atomically persists the selected capture bundle and applies only hot
  controls through that owner. Format changes are staged for its next natural
  start; Save must not invoke systemd, reopen a device, or interrupt a session.
- Switching raw, local processed, Windows processed, or automatic policy swaps
  already-owned frame queues. It must not restart the sender, receiver, virtual
  camera sink, or an application using `/dev/deep-live-cam`.
- Raw and processed transitions must retain the same device identity, format,
  1280 x 720 dimensions, and 30 FPS delivery contract.
- Do not restore legacy general-purpose Start/Stop/Restart camera buttons that
  can cause `device busy`. Recovery actions may reconnect passive GUI decoders.
  Privileged repair must use the existing narrow helper and clearly explain
  its scope.

## 4. Transport and routing contract

The controller must model these as distinct concepts even though normal users
choose them through only the three semantic tabs:

1. Which of up to five client slots Windows processes.
2. What stream the Arch stable system camera currently displays.
3. What capture owner/settings target is currently selected.

Never merge their runtime state into one ambiguous “device” status.

### Five device slots

- Slot `N` sends to Windows on `10000 + 2N`.
- Its processed return is always `input port + 1`.
- Slots 0 through 4 therefore reserve `10000/10001` through `10008/10009`.
- Only one input is processed on Windows at a time.
- Windows publishes its selected valid stream separately on port `10010`.
- Hidden diagnostics must retain each slot's identity, stack/capabilities,
  endpoints, readiness, active selection, and stale/error state. Slot numbers
  and ports must not replace the three semantic choices on the Input tab.

### Arch-local ports

- `11000`: sender raw fallback for the stable receiver.
- `11001`: independent raw preview for the manager.
- `11002`: private sender copy for the Windows transport worker.
- `11003`: receiver-owned preview relay for selected Windows output.
- `11004`: exact encoded copy of the stream returned to the phone.
- `11005`: dedicated camera input to local Native-256.
- `11006`: local Native-256 result into the stable receiver.
- `11007`: receiver-owned preview relay for local processed output.
- `11008`: selected Windows result for the exclusive phone-return relay.
- `11009`: local processor result for the exclusive phone-return relay.

### System-camera policy

Retain and accurately describe these receiver policies in status/diagnostics:

- `local`: local Native-256 first, raw webcam fallback;
- `windows`: Windows result first, selected-stream pull second, raw fallback;
- `auto`: best fresh processed source, then raw fallback;
- `raw`: raw webcam only.

The selected processor derives the ordinary receiver policy; it is not a
second user-facing processor selector. Use `select_receiver_source.py` and its
Unix socket. Show both configured policy and actual active input. A successful
change must explicitly say that the sink and camera identities were not
restarted.

## 5. Required functional areas

The application has exactly three full-window primary tabs: **Processor**,
**Input**, and **Output**. Passive analysis and diagnostics may remain hidden
controller surfaces, but they are not additional primary navigation and there
is no settings dock or nested tab hierarchy.

### A. Live operation

- Make the selected processed/system result the hero preview on the Output
  tab.
- Show a smaller raw/input comparison when that stream exists.
- Label every pane with the real source, model, endpoint, freshness, decoded
  FPS, frame count, drops, and whether it is delayed for comparison.
- Provide a passive pop-out preview of the exact stream sent to the phone.
- Surface route, system-camera policy, active input, processor/model, identity
  effect status, and critical warnings without opening Diagnostics.
- Page navigation must not stop either preview decoder.
- Joining H.264 mid-GOP may briefly lack SPS/PPS. Do not flood the visible log
  with repeated identical startup decoder messages; aggregate or rate-limit
  them while retaining actionable failures.

### B. Identity and processing

- Always display the currently selected source picture.
- Provide at least eight recent source pictures with one-click switching.
- Show filename, target, applied/pending/error state, and clear active styling.
- Give each history entry a content identifier. Windows source upload must
  carry the SHA-256 of the exact transmitted bytes, and Windows must persist
  and report that identifier only when its image/metadata pair verifies. Do
  not display synchronized merely because Windows reports that *some* source
  is configured.
- Explicitly distinguish the Windows processor source upload from the local
  Native-256 source-history image. Do not claim a source is active locally
  unless the local processor has actually loaded it.
- The processor choice has two fixed targets: Windows 11 uses production
  INSwapper-128 ONNX/CUDA; Arch uses Native-256 semantic/ncnn-Vulkan and keeps
  its development/unqualified warning visible. Model choice is not a third
  user-controlled setting.
- Group everyday controls together: processor target, preset, opacity, color
  match, mouth mask, sharpness, and multi-face. The one explicit **Enable face
  swapping** checkbox belongs to Output and maps to the shared face-swap versus
  passthrough mode; it must not have a second copy on Processor. Mirror and
  rotation also belong to Output, not to a processor copy of the configuration.
- Group advanced processing separately: enhancer, interpolation, detector
  cadence/confidence/minimum size, optical-flow tracking, smoothing, grace,
  automatic output guard, and the detail/checkerboard/wavelet/semantic-boundary/
  skin-texture repair profile. These values belong to the one shared desired
  document and must be reconciled to either processor rather than retained as
  host-local defaults.
- Every setting must state which engine it affects: Windows, local Native-256, both,
  or comparison-view only.
- Disabled/unavailable settings must remain understandable and say why.
- Never represent “processor invoked” or a changed mask ring as verified
  identity similarity. Preserve the current distinction between checkpoint
  qualification, visual-effect evidence, and identity verification.

### C. Cameras and routing

- Present exactly three semantic input choices: Arch USB webcam, phone front,
  and phone back. The five-slot Windows registry and its port pairs are an
  internal transport contract, not primary input navigation.
- Keep Input selection visually separate from Processor and Output choices.
- Show a compact route/topology view for Android, Windows, and Arch, including
  bypass and fallback states.
- Generate capture-owner controls from `camera_schema()` and the selected
  adapter rather than hard-coding one device's capabilities.
- Organize camera controls semantically: profile/capture, tone/color,
  exposure/white balance, orientation/lens/stabilization where available.
- Clearly distinguish session-only live preview changes from persisted values.
- Named profile selection must load the whole profile atomically. Editing one
  component changes the visible state to Custom.
- Saving camera configuration must show progress, success/failure, and the
  effective owner-reported values.
- The measured Arch default is `natural-indoor`:
  - MJPEG 1280 x 720 at 30 FPS;
  - brightness -12, contrast 12, saturation 40, hue 0, gamma 95;
  - gain 0, sharpness 2, backlight compensation 0;
  - 50 Hz power-line filtering;
  - automatic exposure, dynamic frame rate off;
  - automatic white balance;
  - manual fallback exposure 157 and white balance 4600 K.

### D. Analyze and baselines

- Keep comparison-view delay and passive measurement controls separate from
  actual camera/processor settings.
- Show useful raw and processed qualitative metrics with plain-language labels,
  sample readiness, and caveats.
- Display the active reproducible baseline from
  `arch-linux/runtime/android-phone-processed/benchmarks/active-baseline.json`.
- Current baseline:
  `arch-webcam-inswapper128-natural-indoor-v1`.
- Show its camera profile, model/backend, FPS, stability, detection misses,
  seam metric, frame count, and comparison limitations.
- Do not turn VMAF/SSIM/PSNR into a facial-realism score. They measure signal
  preservation and also penalize the intended identity edit.
- Make clear that a definitive comparison requires replaying the same frozen
  decoded reference corpus.

### E. System and diagnostics

- Summarize Android, Windows, and Arch status without requiring users to parse
  raw JSON.
- Provide an inspectable detailed snapshot and bounded event log.
- Include the endpoint/device contract and exact current paths.
- Keep safe actions such as reconnecting GUI readers and copying diagnostic
  JSON close to the relevant status.
- Deduplicate/rate-limit repetitive FFmpeg startup warnings.
- Error states must include the failing component and a useful next action.

## 6. Interaction requirements

- Mouse-wheel events over `QAbstractSlider`, all `QAbstractSpinBox` variants,
  dials, and `QComboBox` must never change their value—even when focused.
- Such wheel events should scroll the containing page when it can scroll.
- Values remain editable with click/drag, keyboard arrows, and direct text
  input where applicable.
- Apply this as a global rule so dynamically generated camera-adapter controls
  cannot bypass it.
- Maintain keyboard navigation and visible focus state.
- Do not rely only on red/green; pair color with text/icon/state labels.
- Avoid ambiguous toggles. Use explicit verbs and explain consequence/scope.
- Avoid modal dialogs for routine status. Confirm only genuinely destructive or
  privileged repair operations.
- Preserve view/page selection and sensible splitter sizes during refreshes.
- Frequent health refreshes must not move controls, reset user edits, steal
  keyboard focus, or scroll a page unexpectedly.

## 7. Internal architecture expectations

The current `tester.py` is large and stateful. A robust refactor should prefer:

- a small application shell and navigation controller;
- separate primary page widgets for Processor, Input, and Output, with passive
  analysis/system widgets retained only as hidden implementation surfaces;
- a normalized immutable-ish view model assembled from sender, receiver,
  Android, native-processor, and Windows health;
- render functions that update each page from that view model;
- service/API adapters separate from widget construction;
- reusable status pills/cards, metric rows, and wheel-safe value controls;
- no parallel contradictory copies of the same setting widget;
- no retained dead “legacy” GUI builders after the replacement is proven.

Do not rewrite the transport or inference system merely to make the GUI easier.
Preserve existing helper and API contracts unless a tested compatibility layer
is added.

## 8. State and API inputs

At minimum, preserve support for:

- `${STATE_DIR}/sender.json`;
- `${STATE_DIR}/receiver.json`;
- `${STATE_DIR}/receiver-control.sock`;
- `/run/deep-live-cam/android-phone-processed-health.json`;
- local systemd service state;
- Android status through the existing non-blocking helper/process;
- Windows `/healthz`, configuration, source upload, five-slot registry, and
  selection endpoints through asynchronous `QNetworkAccessManager` calls.

All network and subprocess activity must remain asynchronous from the UI
thread, bounded by timeouts, and safe if an endpoint disappears.

## 9. Performance and reliability

- Keep the UI responsive while video is decoding and health is refreshing.
- Bound image/history memory and log length.
- Do not create a new decoder on every refresh or page change.
- Avoid duplicate consumers of unicast preview ports. Each visible manager feed
  has a dedicated receiver/sender relay.
- Closing the manager stops only its decoder/subprocess children.
- A failed Windows/API/ADB query must not disturb the live local camera route.
- Never alter sender, receiver, camera-control, or system-device state during
  tests or off-screen visual checks.

## 10. Verification and acceptance criteria

Implementation is complete only when all of the following hold:

1. `python arch-linux/bin/tester.py --self-check` succeeds.
2. The window can be instantiated off-screen at 980 x 680 and 1500 x 900.
3. Every primary page can be selected without exceptions or decoder restarts.
4. Live previews remain alive across page navigation.
5. Off-screen tests assert the page/category contract and ownership-safe UI.
6. Wheel-event tests cover sliders, dials, integer/double spin boxes, combo
   boxes, focused controls, and dynamically generated controls.
7. Existing source history, route switching, phone return preview, camera
   adapters, and Windows control tests continue to pass.
8. The full repository test suite passes.
9. Python compilation and desktop-file validation pass.
10. `install.sh` and the narrow deployment helper install every new module.
11. No test or visual validation directly opens `/dev/video*` or the physical
    camera.
12. The implementation updates `arch-linux/README.md` to match the final UI.

## 11. Worktree and implementation boundaries

- The worktree is intentionally dirty and contains the active camera/network
  implementation. Preserve all existing unrelated changes; do not reset,
  clean, checkout, or delete them.
- Work directly in this checkout so the current untracked Arch/Android/Windows
  implementation remains visible.
- Do not restart live services, change camera controls, invoke privileged
  deployment, open real camera nodes, or modify the Android/Windows endpoints.
- You may run read-only status checks, unit tests, compilation, `--self-check`,
  and `QT_QPA_PLATFORM=offscreen` UI tests.
- Implement the application, tests, installer integration, and documentation;
  do not stop at a plan or audit.
