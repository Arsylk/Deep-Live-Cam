# Prerecorded Video Workflow

The prerecorded video system lets you record a camera source, render it offline with perfect face-swap quality (no time constraints), and replay it as a live input source that appears alongside phone front/back/arch webcam in the manager.

## Overview

Three components work together:

1. **record_and_render.py** — record from any camera, render offline with unlimited time per frame
2. **prerecorded_relay.py** — stream the rendered video to the receiver's `local_prerecorded` port (11010)
3. **Deep-Live-Cam Manager** — select "Prerecorded video" as the system camera policy

## Quick Start

### All-in-one command

Record 10 seconds from your webcam, render with face-swap, and stream in a loop:

```bash
python3 arch-linux/bin/record_and_render.py all \
    --device /dev/video0 \
    --duration 10.0 \
    --face path/to/face.jpg
```

This records, renders, and immediately starts streaming. Press Ctrl+C to stop the stream.

### Step-by-step workflow

#### 1. Record from a camera

Record 15 seconds from the Arch webcam:

```bash
python3 arch-linux/bin/record_and_render.py record \
    --device /dev/video0 \
    --duration 15.0 \
    --output raw_recording.mp4
```

Or record from the phone (via `/dev/deep-live-cam` when the receiver is publishing phone front/back):

```bash
python3 arch-linux/bin/record_and_render.py record \
    --device /dev/deep-live-cam \
    --duration 15.0 \
    --output phone_recording.mp4
```

#### 2. Render offline with face-swap

Render with unlimited time per frame for maximum quality:

```bash
python3 arch-linux/bin/record_and_render.py render \
    --input raw_recording.mp4 \
    --face path/to/face.jpg \
    --output rendered_result.mp4
```

**Optional:** Pass a JSON settings file for custom processing options:

```bash
# settings.json example:
{
    "face_swapper": "inswapper_128",
    "face_enhancer": "gfpgan_1.4",
    "frame_enhancer": "realesrgan_x2plus"
}

python3 arch-linux/bin/record_and_render.py render \
    --input raw_recording.mp4 \
    --face path/to/face.jpg \
    --output rendered_result.mp4 \
    --settings settings.json
```

Rendering runs via `run.py --headless` with CPU execution provider for quality over speed. On Arch hardware with no time constraints, this produces the best possible result.

#### 3. Stream to the receiver

Start streaming the rendered video in a loop:

```bash
python3 arch-linux/bin/prerecorded_relay.py rendered_result.mp4 --loop
```

Or stream once without looping:

```bash
python3 arch-linux/bin/prerecorded_relay.py rendered_result.mp4
```

The relay streams to `udp://127.0.0.1:11010` (the receiver's `LOCAL_PRERECORDED_PORT`).

#### 4. Select in the manager

1. Open **Deep-Live-Cam Manager** (the native GUI)
2. Navigate to **Cameras and routing** (workspace 3)
3. In the **ARCH STABLE SYSTEM CAMERA** card, select **Prerecorded video**
4. The system camera (`/dev/deep-live-cam`) now publishes the looping rendered video

The prerecorded source appears in the manager as a stable input alongside:
- Phone front camera
- Phone back camera  
- Arch raw webcam
- Local Native-256 processed result
- Windows return

## Port Assignment

| Port  | Purpose                                    |
|-------|--------------------------------------------|
| 11010 | Prerecorded video input (MPEG-TS/UDP)     |
| 11009 | Local prerecorded preview (MPEG-TS/UDP)   |

The receiver decodes the prerecorded stream using the same phone-accurate transform as live sources:
1. Rotate to upright based on metadata
2. Scale to cover the fixed 1280×720 output box (aspect-correct)
3. Center-crop back to 1280×720

This matches option C: per-source rotate-to-upright + aspect-correct cover-crop into the locked box. No distortion, no letterboxing, looks like a phone flip.

## Source Priority

When the receiver is in `auto` or `local` mode, the prerecorded source has priority:

- **auto**: `local_processed` → **local_prerecorded** → `processed_return` → `selected_stream` → `local_raw`
- **local**: `local_processed` → **local_prerecorded** → `local_raw`
- **prerecorded**: **local_prerecorded** only

The `prerecorded` policy locks the system camera to the prerecorded input exclusively.

## Use Cases

### Perfect quality demo reel

Record yourself, render offline with maximum enhancement, and loop it during a demo:

```bash
# Record 30 seconds
python3 arch-linux/bin/record_and_render.py record \
    --device /dev/video0 \
    --duration 30.0 \
    --output demo.mp4

# Render with all enhancers
python3 arch-linux/bin/record_and_render.py render \
    --input demo.mp4 \
    --face target_face.jpg \
    --output demo_rendered.mp4

# Stream in a loop
python3 arch-linux/bin/prerecorded_relay.py demo_rendered.mp4 --loop
```

Then select **Prerecorded video** in the manager. The system camera now shows your perfect render on repeat.

### Regression testing

Capture a known-good input, render a reference result, and replay it to verify processing pipeline changes:

```bash
# One-time capture and render
python3 arch-linux/bin/record_and_render.py all \
    --device /dev/video0 \
    --duration 5.0 \
    --face reference_face.jpg

# Later: replay the same input for regression testing
python3 arch-linux/bin/prerecorded_relay.py rendered_temp.mp4 --loop
```

### Phone orientation testing

Record from the phone in different orientations (front camera portrait, back camera landscape), render each, and verify the system camera handles rotation correctly:

```bash
# Record phone front (portrait)
python3 arch-linux/bin/record_and_render.py record \
    --device /dev/deep-live-cam \
    --duration 10.0 \
    --output phone_front.mp4

# Render
python3 arch-linux/bin/record_and_render.py render \
    --input phone_front.mp4 \
    --face face.jpg \
    --output phone_front_rendered.mp4

# Stream and verify in manager
python3 arch-linux/bin/prerecorded_relay.py phone_front_rendered.mp4 --loop
```

## Implementation Notes

### Decoder transform (receiver.py lines 655-669)

The decoder already implements the phone-accurate transform:

```python
f"rotate='angle=PI*metadata(rotate)/180:ow=rotw(iw):oh=roth(ih):c=none',"
f"scale={self.width}:{self.height}:"
f"force_original_aspect_ratio=increase:flags=fast_bilinear,"
f"crop={self.width}:{self.height},"
f"fps={self.fps},format=yuyv422"
```

This ensures:
- Portrait phone feeds are rotated upright first
- Aspect ratio is preserved during scale
- Center-crop to the fixed 1280×720 box (no stretch, no letterbox)

### Recording format

`record_and_render.py` records in H.264/MP4 with CRF 18 (visually lossless) to preserve quality for offline rendering. The relay re-encodes to MPEG-TS/UDP for streaming to the receiver.

### Rendering quality

Offline rendering uses `--execution-provider cpu` by default. On Arch hardware with no real-time constraints, this produces the highest quality result. GPU providers can be specified via processing settings JSON if needed.

### Manager integration

The manager's **ARCH STABLE SYSTEM CAMERA** card (routing page) already includes the "Prerecorded video" policy. No GUI changes were needed — the infrastructure was already present in `contracts.py` and `receiver.py`.

## Troubleshooting

### Relay not appearing in manager

Check that the relay is streaming:

```bash
# Terminal 1: start relay
python3 arch-linux/bin/prerecorded_relay.py video.mp4 --loop

# Terminal 2: verify UDP traffic
ss -u -n | grep 11010
```

If no UDP traffic appears, verify the video file exists and is readable.

### System camera shows placeholder

The receiver falls back to a placeholder when no frames arrive for 1 second (configurable via `RECEIVER_STALE_SECONDS`). Verify:

1. Relay is running (`ps aux | grep prerecorded_relay`)
2. Port 11010 is not blocked
3. Video file is valid (`ffprobe video.mp4`)

### Wrong aspect ratio

The decoder's phone-accurate transform handles rotation and aspect. If the output looks stretched:

1. Verify the receiver is running the latest decoder code (lines 655-669)
2. Check that the video file has correct rotation metadata (`ffprobe -show_streams video.mp4 | grep rotate`)
3. If metadata is missing, add it: `ffmpeg -i input.mp4 -metadata:s:v rotate=90 -c copy output.mp4`

### Rendering too slow

Offline rendering prioritizes quality. To speed it up:

1. Use GPU execution provider: `--execution-provider cuda` (if available)
2. Reduce frame enhancer settings in the JSON config
3. Render a shorter clip
4. Accept lower quality for testing (use the `all` command which renders and streams immediately)

## Future Enhancements

Potential improvements (not yet implemented):

- GUI button in the manager to start/stop relay processes
- Playlist support (sequential or random video selection)
- Real-time quality metrics comparison (live vs prerecorded)
- Batch rendering of multiple sources
- WebUI for remote recording/rendering control
