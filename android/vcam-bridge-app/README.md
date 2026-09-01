# Android Camera2 slot client

`dev.vcam.bridge` is the camera owner for the Android slot. It captures and
hardware-encodes the phone camera once, while the KernelSU client handles SRT
transport and the stable Camera2 ID 120 output.

Version 0.3.3 uses a 10 Mbps, constant-rate, B-frame-free AVC capture and adds
allow-listed controls for lens selection, exposure compensation, AE/AWB locks,
video/optical stabilization, and orientation. The default `auto` orientation
uses the selected lens's Camera2 sensor orientation; manual clockwise choices
are `0`, `90`, `180`, and `270` degrees. A zero-copy OpenGL ES stage applies
the turn and aspect-fits portrait video into the fixed 1280x720 transport frame
without stretching or cropping it.

Sensor timestamps, capture jitter, estimated drops, exposure/ISO drift,
white-balance gain drift, effective rotation, and rendered-frame count are
logged for the native Arch manager. The manager never opens the phone camera
and ADB never carries video. Preview changes update the active capture and GPU
renderer immediately; only the manager's explicit save action updates the
app's persistent settings.

Build with `./build.sh`. The resulting artifact is
`build/vcam-bridge.apk`.
