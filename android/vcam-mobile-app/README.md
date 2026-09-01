# Deep Live Mobile prototype

`dev.vcam.mobile` is an offline, standalone Android implementation of the
first mobile milestone. It owns a physical Camera2 camera only while its
activity is visible, performs SCRFD detection, ArcFace source embedding,
INSwapper inference, temporal landmark smoothing, LAB color matching, feathered
paste-back, and face-local sharpening, and displays the result directly.

It intentionally has no `INTERNET`, storage, microphone, foreground-service,
root, KernelSU, or virtual-camera permission. It does not modify or communicate
with `dev.vcam.bridge`, Windows, Arch, DroidCam, or camera 120.

## Build and install

```bash
./fetch-dependencies.sh
./prepare-models.sh
./build.sh
./install.sh
```

The dependency and model preparation steps need host files/network only once.
The installed application is entirely offline. Large models are installed into
the app-private model directory over ADB rather than embedded in the APK, which
keeps APK installation reliable and permits checksum verification before use.
The installer stages a complete pack, publishes its manifest last, and swaps
directories atomically so an interrupted transfer cannot replace a working
pack with a partial one. This milestone build is debug-signed so `install.sh`
can use Android's scoped `run-as` access; the application itself never invokes
ADB or root.

The app requests Android's declared public `libOpenCL.so` vendor interface and
uses a strict one-node probe to prove QNN GPU is genuinely available. Each full
model is then admitted to QNN only if its complete graph is accepted with CPU
fallback disabled. A rejected model alone is reopened as a four-thread ONNX
Runtime CPU session; it does not pull the other models off the GPU. The live
metrics state the active detector/swapper provider honestly. The recognizer is
loaded only for a newly imported source. Its final 512-float latent is stored
beside the app-private history entry, so restoring or selecting any recent
source does not reopen the 166 MiB recognizer.

The swapper has a QNN FP16 copy to reduce GPU bandwidth and resident memory.
Its FP32 copy remains the CPU fallback; model preparation checks both files and
the phone verifies every model checksum before creating a session.

## Current prototype scope

- 1280x720 physical-camera capture, rotated/mirrored for the portrait preview.
- Raw/processed toggle and hold-to-compare.
- Local-only source picker with five persistent recent pictures.
- Local MP4 recording of whichever mode is displayed.
- Camera, inference, stage latency, memory, thermal, and dropped-frame metrics.
- Capacity-one inference queue and source/camera generation guards.
- Clean raw fallback on missing model, source, accelerator, or face.
- Whole-stack camera availability preflight: an existing bridge/camera owner is
  never evicted; the preview waits and starts automatically when it is released.
- Camera and QNN/ORT sessions are released when the activity is backgrounded.

The tracker is deliberately labeled `adaptive EMA/fade`: it uses capture-time
aware landmark smoothing and a bounded missed-detection fade, not optical flow.
System camera output is outside this milestone.

## Measured Snapdragon 865 baseline

The complete APK was exercised on the connected Xiaomi M2007J3SY (`apollo`,
Android 16) at 1280x720. Camera capture sustains approximately 29–31 FPS while
the unchanged large InsightFace models produce about 0.9–1.1 processed FPS:
roughly 225–350 ms detection, 625–750 ms swap/post-processing, and
0.86–1.12 seconds end to end. Thermal status remained 0 during validation.

The first visible session needs roughly 25–45 seconds to compile the strict QNN
GPU graphs; raw preview remains live throughout. Active PSS is currently about
1.0–1.1 GiB because ONNX Runtime retains host initializers while QNN owns a GPU
copy. Backgrounding releases those sessions and measured PSS falls to about
170–190 MiB, with no active camera client. This is a functional quality
baseline, not real-time performance; shared external QNN contexts and/or a
mobile-specific swapper are the next optimization milestone.
