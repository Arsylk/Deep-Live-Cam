# Prerecorded Video Pipeline

Record a camera source, render it offline with face-swap (unlimited time per frame), and stream the result as a looping input source to the receiver's system camera.

## Overview

The `record_and_render.py` tool provides three stages:

1. **Record** — capture from any V4L2 camera to MP4
2. **Render** — apply face-swap with no time constraints via `run.py --headless`
3. **Stream** — loop the rendered video to `udp://127.0.0.1:11010` (receiver's `LOCAL_PRERECORDED_PORT`)

The receiver treats the prerecorded stream as `local_prerecorded` — a standard input source that can be selected via the system camera policy or the manager GUI.

## Usage

### All-in-one (record, render, stream)

```bash
cd /opt/github/Deep-Live-Cam
arch-linux/bin/record_and_render.py all \
  --device /dev/video0 \
  --duration 10.0 \
  --face path/to/face.jpg
```

This records 10 seconds from `/dev/video0`, renders it with `face.jpg`, then streams the result in a loop. Press `Ctrl+C` to stop streaming.

### Stage by stage

**Record:**
```bash
arch-linux/bin/record_and_render.py record \
  --device /dev/video0 \
  --duration 10.0 \
  --output recorded.mp4
```

**Render:**
```bash
arch-linux/bin/record_and_render.py render \
  --input recorded.mp4 \
  --face path/to/face.jpg \
  --output rendered.mp4 \
  --settings processing_settings.json
```

The `--settings` JSON file can contain any `run.py` CLI flags as keys:
```json
{
  "face_mask_blur": 0.3,
  "face_mask_padding": [0, 0, 0, 0],
  "execution_thread_count": 4
}
```

**Stream:**
```bash
arch-linux/bin/record_and_render.py stream \
  --input rendered.mp4 \
  --port 11010
```

Streams to the receiver's `LOCAL_PRERECORDED_PORT` (default 11010) in a loop.

## Selecting the prerecorded source

### Via the manager GUI

1. Open the manager: `arch-linux/bin/tester.py`
2. Go to **System** workspace
3. Set **System camera policy** to **"Prerecorded video"**

The `/dev/deep-live-cam` device will now show the looping rendered video.

### Via device_control.py

```bash
arch-linux/bin/device_control.py source prerecorded
```

### Via receiver control socket

```bash
echo '{"mode": "prerecorded"}' | nc -U /run/user/$(id -u)/deep-live-cam-receiver-source.sock
```

## Pipeline details

- **Recording**: uses `ffmpeg -f v4l2` with MJPEG input, encodes to H.264 with CRF 18 (high quality).
- **Rendering**: calls `run.py --headless` with CPU execution provider for quality over speed. Each frame gets unlimited processing time.
- **Streaming**: `ffmpeg -re -stream_loop -1` sends the rendered video to UDP 11010 as MPEG-TS, matching the receiver's expected transport.
- **Receiver integration**: `local_prerecorded` is wired into `SOURCE_PRIORITIES` for the `local` and `prerecorded` policies. The decoder applies the same phone-accurate aspect-correct cover-crop transform as all other sources.

## Example: Xiaomi phone front camera as a prerecorded source

Assuming the phone front camera is currently the active sender source:

```bash
# Record 15 seconds of the phone front camera
arch-linux/bin/record_and_render.py record \
  --device /dev/video0 \
  --duration 15.0 \
  --output phone_front.mp4

# Render with a target face
arch-linux/bin/record_and_render.py render \
  --input phone_front.mp4 \
  --face target_face.jpg \
  --output phone_front_swapped.mp4

# Stream it
arch-linux/bin/record_and_render.py stream \
  --input phone_front_swapped.mp4
```

Now switch the system camera policy to "Prerecorded video" via the manager, and `/dev/deep-live-cam` will display the looping rendered result with the same portrait orientation and aspect ratio as the live phone camera.

## Notes

- The prerecorded source has the same 1280×720 locked output geometry as all other receiver inputs. The decoder's phone-accurate transform (rotate-to-upright + aspect-correct cover-crop) preserves the real proportions of the recorded source.
- Rendering is CPU-only by default for maximum quality. For faster preview rendering, edit `record_and_render.py` and change `--execution-provider cpu` to `cuda` or `coreml`.
- The stream loops indefinitely until stopped with `Ctrl+C`.
- The receiver's `local_prerecorded` decoder restarts every ~3 seconds when the UDP stream is absent, so you can stop/start streaming without restarting the receiver.
