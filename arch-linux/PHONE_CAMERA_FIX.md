# Phone Camera Quality & Orientation Fix

## Problem Statement

The system camera (`/dev/deep-live-cam`) was stretching and cropping phone camera feeds:

1. **Aspect distortion** — portrait phone feeds (9:16) were stretched horizontally into the locked 16:9 output box, squashing faces.
2. **Center-crop zoom** — rotated feeds were center-cropped back to 1280×720, discarding most of the frame.
3. **Inconsistent quality** — switching between front/back cameras changed distortion severity because they have different native resolutions/orientations.

Root cause: the receiver held one fixed V4L2 format (1280×720 landscape) and force-fit every source into that box with no aspect preservation.

## Solution Implemented (Option C — Phone-Accurate)

**Per-source rotate-to-upright + aspect-correct cover-crop into the locked output box.**

### What Changed

1. **Fixed decoder stretch** (`receiver.py:decoder_command()`)
   - Added `force_original_aspect_ratio=decrease,pad=1280:720:-1:-1:color=black` to preserve aspect ratio
   - Sources are now pillarboxed/letterboxed instead of stretched
   - Matches the existing Arch `sender.py` path

2. **Rotate-to-upright transform** (`receiver.py:transform_yuyv422()`)
   - Phone feeds are rotated to upright orientation first (based on EXIF/metadata)
   - Then aspect-correct cover-cropped into 1280×720
   - Result: looks like a phone flip — portrait content stays portrait-shaped within the box

3. **Output format unchanged**
   - `/dev/deep-live-cam` remains locked at 1280×720 landscape
   - No V4L2 format renegotiation — open consumers never see a resolution change
   - Same device stability, better input fidelity

### Visual Result

- Front camera (portrait 9:16) → rotated upright, pillarboxed to 16:9, no face squash
- Back camera (landscape or portrait depending on phone) → rotated upright, correct aspect
- Arch webcam (landscape 16:9) → unchanged, fills entire frame
- All sources: real proportions preserved, looks like flipping a phone

### Files Modified

- `arch-linux/bin/receiver.py` — decoder aspect preservation + rotate-to-upright transform
- No changes needed to sender, android_stream_processor, or device_control

### Verification

Run the receiver with a phone source and verify:
```bash
python arch-linux/bin/receiver.py
```

Check the system camera output:
```bash
ffplay /dev/deep-live-cam
# or
mpv av://v4l2:/dev/deep-live-cam
```

Expected: portrait phone content appears upright with black bars on sides, no distortion.

---

## Pre-Recorded Video Rendering (New Feature)

Added a "Render" workspace to the native manager for offline high-quality face swapping.

### Workflow

1. **Record** from any camera source (phone front/back/arch webcam) to a video file
2. **Pre-render** with no time constraints — same `swap_face()` path as live but unlimited time per frame
3. **Replay** the rendered video as a receiver input source (loops indefinitely)
4. **Display** through the system camera as "Pre-recorded Video" source

### Why

Live face swapping is constrained by frame time (must finish before next frame arrives). Pre-rendering allows:
- Higher quality settings
- Multiple processing passes
- Post-processing effects
- Stable output for demos/testing

### UI Location

Native manager → **Render** tab (new workspace)

### Controls

- **Source**: Select camera input (Phone Front, Phone Back, Arch Webcam)
- **Duration**: Recording length in seconds (5-300s, default 30s)
- **Face**: Browse button to select source face image (same picker as Identity tab)
- **Quality**: Fast / Balanced / High
- **Record**: Captures video from selected source
- **Render**: Processes recorded video with face swap
- **Add to Sources**: Makes rendered video available as a receiver input

### Files Added

- `arch-linux/bin/dlc_manager/pages/render.py` — new Render workspace widget
- Updated `arch-linux/bin/dlc_manager/shell.py` — integrated Render tab into navigation

### Technical Details

Recording path:
```
Selected camera → FFmpeg capture → /tmp/dlc_recording_XXXXXX.mp4
```

Render path:
```
/tmp/dlc_recording_XXXXXX.mp4 + face image
  → run.py --face <face> --quality <quality>
  → /tmp/dlc_rendered_XXXXXX.mp4
```

Replay path:
```
/tmp/dlc_rendered_XXXXXX.mp4
  → receiver.py input source (loops via -stream_loop -1)
  → /dev/deep-live-cam as "Pre-recorded Video"
```

The rendered video is automatically added to the receiver's input source registry and appears in the "Cameras / routing" workspace.

### Face Picker

Uses the same `QFileDialog.getOpenFileName()` pattern as the Identity tab:
- Title: "Select source picture"
- Filter: "Image (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All (*)"
- Matches existing manager UX

### Verification

1. Open Deep-Live-Cam Manager (already runs as root via pkexec)
2. Navigate to **Render** tab
3. Select a camera source
4. Click "Record" → wait for duration
5. Select a face image
6. Click "Render" → wait for processing
7. Click "Add to Sources"
8. Navigate to **Cameras** tab
9. Select "Pre-recorded Video" as input
10. See rendered loop on system camera

---

## Desktop Entry

Already configured to run as root via `pkexec`:

```desktop
Exec=pkexec /usr/local/bin/deep-live-cam-tester
```

Location: `arch-linux/config/deep-live-cam-tester.desktop`

No changes needed — privilege elevation was already in place.

---

## Testing

### Phone Camera Quality

```bash
# Start receiver with phone connected
python arch-linux/bin/receiver.py

# In another terminal, check output
ffplay /dev/deep-live-cam

# Switch between phone front/back in the manager
# Verify: no face squashing, correct aspect ratio, upright orientation
```

### Pre-Recorded Rendering

```bash
# Launch manager
pkexec /usr/local/bin/deep-live-cam-tester

# Or via desktop entry
gtk-launch deep-live-cam-tester

# Follow UI workflow in Render tab
# Verify: recording captures, render processes, replay loops
```

### Integration Test

```bash
# Record from phone front camera
# Render with a face swap
# Add to sources
# Route to system camera
# Open in browser/meeting app
# Verify: stable loop, correct quality, no distortion
```

---

## Future Enhancements

Possible extensions (not implemented):

1. **Render queue** — batch multiple recordings
2. **Render presets** — save quality/face combinations
3. **Render history** — keep recent renders accessible
4. **Render preview** — scrub through render before adding to sources
5. **Dynamic output format** — let output resolution follow active source (would require V4L2 renegotiation, breaks "never renegotiate" invariant)

Current implementation prioritizes stability (fixed output format) and simplicity (single render at a time).
