#!/system/bin/sh

for pid_file in \
    /data/local/tmp/android-vcam-output-selector.pid \
    /data/local/tmp/android-vcam-output-worker.pid \
    /data/local/tmp/android-vcam-capture.pid \
    /data/local/tmp/android-vcam-return.pid \
    /data/local/tmp/android-vcam-audio.pid \
    /data/local/tmp/android-vcam-sender.pid \
    /data/local/tmp/android-vcam-arch-sender.pid \
    /data/local/tmp/android-vcam-provider-supervisor.pid \
    /data/local/tmp/android-vcam-provider.pid \
    /data/local/tmp/android-vcam-producer.pid; do
    if [ -r "$pid_file" ]; then
        pid=$(cat "$pid_file" 2>/dev/null)
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    fi
done

# Only the module-owned external provider and loopback are stopped. The QCOM
# physical-camera provider is owned by Android init and is never addressed.
am stopservice --user 0 -n dev.vcam.app/.ReturnAudioService >/dev/null 2>&1 || true
sleep 1
rmmod v4l2loopback 2>/dev/null || true
rm -f /data/local/tmp/android-vcam.log \
      /data/local/tmp/android-vcam-return.mjpg \
      /data/local/tmp/android-vcam-return.progress \
      /data/local/tmp/android-vcam-output.state \
      /data/local/tmp/android-vcam-audio.progress \
      /data/local/tmp/android-vcam-output-ffmpeg.log \
      /data/local/tmp/android-vcam-capture-ffmpeg.log \
      /data/local/tmp/android-vcam-return-ffmpeg.log \
      /data/local/tmp/android-vcam-return-ffmpeg.log.current \
      /data/local/tmp/android-vcam-audio-ffmpeg.log \
      /data/local/tmp/android-vcam-audio-ffmpeg.log.current \
      /data/local/tmp/android-vcam-sender-ffmpeg.log \
      /data/local/tmp/android-vcam-sender-ffmpeg.log.current \
      /data/local/tmp/android-vcam-windows-sender.progress \
      /data/local/tmp/android-vcam-windows-sender.state \
      /data/local/tmp/android-vcam-arch-sender.progress \
      /data/local/tmp/android-vcam-arch-sender.state \
      /data/local/tmp/android-vcam-arch-sender-ffmpeg.log \
      /data/local/tmp/android-vcam-arch-sender-ffmpeg.log.current \
      /data/local/tmp/vcam.xml \
      /data/adb/android-vcam-output.conf
