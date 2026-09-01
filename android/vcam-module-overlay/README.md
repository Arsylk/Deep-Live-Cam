# Android virtual-camera processor client v0.4.9

This is the single-camera, physical-provider-safe build for the Apollo phone.
One Camera2 hardware encoder supplies independent transports to Windows slot 0
at `192.168.1.35:10000` and the Arch local processor at
`192.168.1.11:10001`. The selected processor returns video to the phone's SRT
listener on port `10001`.

## Camera topology

- The module creates exactly one loopback node: `/dev/video20`.
- The AOSP external provider exposes that node as external Camera2 ID `120`.
- `/dev/video0`, `/dev/video1`, and every Qualcomm camera node remain unchanged.
- Android init remains the sole owner of the Qualcomm physical-camera provider.
- The module never issues `stop` or `kill` against the Qualcomm provider and
  never mounts over a physical camera device node.

The installable ZIP uses the validated provider, producer, fallback image, and
`v4l2loopback.ko` from v0.3.0. Apollo does not project module files into its
separate read-only vendor partition, so the early hook retains v0.3.0's two
known-good vendor-file binds: the combined IR/external VINTF fragment and the
external-camera implementation. Neither bind touches camera device nodes or
the Qualcomm provider.

At installation only, `customize.sh` repairs the one known stale state from the
experimental dual build: if `persist.vendor.camera.disable` is exactly `1`, it
is changed to `0` before reboot. Other values and all other properties are left
untouched; there is no recurring property override.

## Transport isolation

- `PHONE_CAPTURE_ENABLED=1` runs one `bridge-capture.sh` process and the merged
  app's scoped physical Camera2 owner. Front/back changes stay inside that
  owner and retain the same encoder TCP endpoint and transport workers.
- `bridge-capture.sh` is the only TCP reader of the hardware H.264 encoder. Its
  tee writes complete MPEG-TS copies to raw fallback UDP `10022`, Windows-feed
  UDP `10023`, and Arch-feed UDP `10027`.
- Before the tee, `extract_extradata` caches MediaCodec's in-band H.264 SPS/PPS
  and `dump_extra` prepends that cached configuration to every keyframe. A
  sender or decoder which reconnects in the middle of the continuous capture
  stream can therefore start at the next IDR without `non-existing PPS 0`
  errors. The Apollo Termux FFmpeg 8.1.1 build was checked for H.264-capable
  `extract_extradata`, `setts`, and `dump_extra` filters.
- `bridge-sender.sh` consumes only UDP `10023` and owns the Windows SRT caller
  to `192.168.1.35:10000`.
- `bridge-arch-sender.sh` consumes only UDP `10027` and owns the Arch SRT
  caller to `192.168.1.11:10001`.
- The two SRT workers have separate processes, logs, progress files, status
  files, and exponential retry state. A Windows outage cannot restart or
  consume packets from the Arch path, and an Arch outage cannot affect slot 0.
  Their atomic status files are
  `/data/local/tmp/android-vcam-windows-sender.state` and
  `/data/local/tmp/android-vcam-arch-sender.state`.
- `bridge-return.sh` receives processed frames and publishes a decoded-frame
  heartbeat while copying the optional AAC track to a private A/V feed.
- `bridge-audio.sh` decodes that AAC track to 48 kHz mono PCM on localhost
  UDP 10025. The scoped Xposed module consumes it only while the redirected
  front camera is active; it is never played through the phone speaker.
- `bridge-output-selector.sh` is the only writer to Camera2 ID 120. It selects
  processed video only while frames advance. With phone capture disabled, a
  return outage shows the module fallback frame without changing camera
  identity or format; phone-input mode may instead fall back to its raw feed.
- `/data/adb/android-vcam-output.conf` is the durable, atomically replaced
  output policy. Enable/disable, horizontal mirror, and 0/90/180/270-degree
  rotation only replace the FIFO decoder worker. The provider, `/dev/video20`,
  producer, Camera2 ID 120 registration, and already-open camera sessions stay
  alive; disabled output continuously resolves to the producer's placeholder.
  Every transform center-fills the fixed 1280x720 Camera2 surface while
  preserving aspect ratio. A 90/270-degree rotation therefore crops the excess
  portrait axis instead of shrinking the picture into black padding.

Camera2 ID 120 remains 1280×720 at 30 FPS across return-stream loss and
recovery. The unified APK preview selects ID 120 first and falls back only to
another external camera, never to a physical phone camera.

The boot service retries the unified APK's two foreground services for up to
60 seconds after Termux becomes available. This closes the short interval in
which credential-encrypted files are visible but Android user 0/package
services are not yet ready, without restarting the provider or Camera2 120.

The Arch manager calls the same narrow helper that is also available from the
command line:

```sh
python arch-linux/bin/android_bridge.py configure-output \
  --host 192.168.1.12 --output-enabled true \
  --output-mirror true --output-rotation 90
```

The command is idempotent: repeating an unchanged policy leaves its revision
and decoder worker untouched. Module versions before v0.4.6 do not consume the
control file, so the UI must treat a missing effective revision as pending
rather than claiming that the setting was applied.

## Output-transform hot update

Install only a changed `bridge-output-common.sh` and restart only its selector
and decoder worker with:

```sh
android/vcam-module-overlay/deploy-output-transform-hot.sh [ADB_SERIAL]
```

The serial argument is optional. The helper uses `ANDROID_ADB_SERIAL` next and
otherwise selects the first authorized adb device. It atomically installs the
filter helper and verifies the replacement selector; it does not restart the
provider, producer, unified app, physical-camera capture, or Camera2 ID 120.

## Capture-only v0.4.9 hot update

The SPS/PPS repair does not require a Camera2, MediaCodec, external-provider,
producer, return-path, or sender restart. Stage only the four capture-update
files, then run the narrow device-side helper as root:

```sh
adb -s b83607a5 shell mkdir -p /data/local/tmp/android-vcam-v049-capture
adb -s b83607a5 push \
  android/vcam-module-overlay/bridge-capture.sh \
  android/vcam-module-overlay/module.prop \
  android/vcam-module-overlay/README.md \
  android/vcam-module-overlay/deploy-capture-hot.sh \
  /data/local/tmp/android-vcam-v049-capture/
adb -s b83607a5 shell su -c \
  '/system/bin/sh /data/local/tmp/android-vcam-v049-capture/deploy-capture-hot.sh'
```

The helper first verifies the installed FFmpeg filters, atomically installs
the staged files, and replaces only `android-vcam-capture.pid` plus its direct
FFmpeg child. The scoped app keeps the physical camera and hardware encoder
open; its existing accept loop sends cached SPS/PPS and requests a sync frame
for the new TCP client. Camera2 ID 120 remains registered throughout.

Verify without opening or restarting either camera:

```sh
adb -s b83607a5 shell su -c \
  'cat /data/adb/modules/android-vcam/module.prop; cat /data/local/tmp/android-vcam-capture.pid; tail -n 20 /data/local/tmp/android-vcam-capture-ffmpeg.log'
```
