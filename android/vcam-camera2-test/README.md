# Camera2 120 live test

This small Android diagnostic app opens the processed external Camera2 device
`120`, reports frame/FPS/luma health, and renders the same live stream directly
below the status text. An opaque warning covers the preview whenever no fresh
frame has arrived for 2.5 seconds, so an old Surface buffer cannot look healthy.
If the external provider is restarted, the app closes the stale Camera2 session
and reopens camera `120` automatically until rendered frames resume.

Build and install locally (no web service is involved):

```bash
./android/vcam-camera2-test/build.sh
adb install -r android/vcam-camera2-test/build/vcam-camera2-test.apk
```

To verify an Xposed camera alias without leaving a camera busy, request its
logical ID and have the activity close itself after a fixed number of frames:

```bash
adb shell am start -W -n dev.vcam.test/.MainActivity \
  --es requested_camera_id 7 --ei stop_after_frames 120
```

The log reports both the requested ID and `CameraDevice.getId()`. For the
MoonPay compatibility route, the expected result is `requested=7; actual=120`.
