#!/system/bin/sh
MODDIR=${0%/*}
LOG=/data/local/tmp/android-vcam.log
PRODUCER_PID_FILE=/data/local/tmp/android-vcam-producer.pid
PRODUCER2_PID_FILE=/data/local/tmp/android-vcam-producer2.pid
PROVIDER_PID_FILE=/data/local/tmp/android-vcam-provider.pid
PROVIDER_SUPERVISOR_PID_FILE=/data/local/tmp/android-vcam-provider-supervisor.pid
RETURN_PID_FILE=/data/local/tmp/android-vcam-return.pid
SENDER_PID_FILE=/data/local/tmp/android-vcam-sender.pid
CAPTURE_PID_FILE=/data/local/tmp/android-vcam-capture.pid
OUTPUT_PID_FILE=/data/local/tmp/android-vcam-output-selector.pid
VINTF_TARGET=/vendor/etc/vintf/manifest/android.hardware.ir-service.lineage.xml
EXT_IMPL_TARGET=/vendor/lib64/camera.device@3.4-external-impl.so
VCAM_VENDOR="$MODDIR/system/vendor"
RETURN_FIFO=/data/local/tmp/android-vcam-return.mjpg
RETURN_FIFO2=/data/local/tmp/android-vcam-return2.mjpg
. "$MODDIR/bridge.conf"
exec >>"$LOG" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') android-vcam service starting (dual/overlay mode)"
rm -f "$PRODUCER_PID_FILE" "$PRODUCER2_PID_FILE" "$PROVIDER_PID_FILE" \
      "$PROVIDER_SUPERVISOR_PID_FILE" "$RETURN_PID_FILE" "$SENDER_PID_FILE" \
      "$CAPTURE_PID_FILE" "$OUTPUT_PID_FILE"
if ! grep -q " $VINTF_TARGET " /proc/mounts; then
    echo "VINTF fragment is not mounted; refusing late HAL registration"
    exit 1
fi
if ! grep -q " $EXT_IMPL_TARGET " /proc/mounts; then
    echo "external-camera config shim is not mounted; refusing late HAL registration"
    exit 1
fi

# Two processed loopback devices: video20 (back-node feed), video21 (front-node feed).
if ! grep -q '^v4l2loopback ' /proc/modules; then
    insmod "$MODDIR/v4l2loopback.ko" \
        video_nr=20,21 \
        card_label=ProcessedBack,ProcessedFront \
        exclusive_caps=0 max_width=1920 max_height=1080 max_buffers=4 max_openers=8
fi
attempt=0
while ([ ! -e /dev/video20 ] || [ ! -e /dev/video21 ]) && [ "$attempt" -lt 50 ]; do
    sleep 0.1
    attempt=$((attempt + 1))
done
if [ ! -e /dev/video20 ] || [ ! -e /dev/video21 ]; then
    echo "loopback nodes did not appear"; exit 1
fi

cp "$VCAM_VENDOR/etc/external_camera_config.xml" /data/local/tmp/vcam.xml
chmod 0644 /data/local/tmp/vcam.xml
chcon u:object_r:vendor_configs_file:s0 /data/local/tmp/vcam.xml 2>/dev/null

for fifo in "$RETURN_FIFO" "$RETURN_FIFO2"; do
    if [ ! -p "$fifo" ]; then rm -f "$fifo"; mkfifo "$fifo"; fi
    chmod 0660 "$fifo"
done

# Producers: feed each loopback node from its FIFO.
"$VCAM_VENDOR/bin/vcam_mjpeg_fifo" /dev/video20 "$RETURN_FIFO" \
    "$VCAM_VENDOR/etc/android-vcam/vcam-fallback-1280x720.jpg" \
    "$VIDEO_WIDTH" "$VIDEO_HEIGHT" "$VIDEO_FPS" >>"$LOG" 2>&1 &
echo $! >"$PRODUCER_PID_FILE"
"$VCAM_VENDOR/bin/vcam_mjpeg_fifo" /dev/video21 "$RETURN_FIFO2" \
    "$VCAM_VENDOR/etc/android-vcam/vcam-fallback-1280x720.jpg" \
    "$VIDEO_WIDTH" "$VIDEO_HEIGHT" "$VIDEO_FPS" >>"$LOG" 2>&1 &
echo $! >"$PRODUCER2_PID_FILE"
sleep 1

# Cover the default physical camera nodes with the processed loopbacks.
# The legacy QCOM provider is intentionally NOT started: its ISP nodes are
# bind-mounted over, and the external provider re-exposes them as 100/101.
if [ -e /dev/video0 ]; then
    mount --bind /dev/video20 /dev/video0 2>/dev/null && echo "covered /dev/video0"
fi
if [ -e /dev/video1 ]; then
    mount --bind /dev/video21 /dev/video1 2>/dev/null && echo "covered /dev/video1"
fi

# The legacy QCOM provider is auto-started by init and would register the now-
# covered ISP nodes as broken cameras 0-7. An explicit stop prevents init from
# restarting it; its IDs then unregister from the camera service.
for i in 1 2 3 4 5 6 7 8 9 10; do
    LEGACY_PID=$(pidof android.hardware.camera.provider@2.4-service_64)
    [ -n "$LEGACY_PID" ] && break
    sleep 1
done
if [ -n "$LEGACY_PID" ]; then
    stop vendor.camera-provider-2-4
    kill "$LEGACY_PID" 2>/dev/null
    echo "legacy camera provider stopped (pid $LEGACY_PID)"
fi

/system/bin/sh "$MODDIR/provider-supervisor.sh" &
echo $! >"$PROVIDER_SUPERVISOR_PID_FILE"
attempt=0
while [ ! -s "$PROVIDER_PID_FILE" ] && [ "$attempt" -lt 50 ]; do sleep 0.1; attempt=$((attempt + 1)); done

if [ "$BRIDGE_ENABLED" = "1" ]; then
    wait_count=0
    while [ ! -x "$TERMUX_PREFIX/bin/ffmpeg" ]; do
        if [ $((wait_count % 30)) -eq 0 ]; then echo "waiting for user unlock and Termux FFmpeg; cameras on fallback"; fi
        wait_count=$((wait_count + 1)); sleep 2
    done
    echo "Termux FFmpeg available; starting Android-Windows bridge"
    /system/bin/sh "$MODDIR/bridge-capture.sh" >>"$LOG" 2>&1 &
    echo $! >"$CAPTURE_PID_FILE"
    /system/bin/sh "$MODDIR/bridge-output-selector.sh" >>"$LOG" 2>&1 &
    echo $! >"$OUTPUT_PID_FILE"
    /system/bin/sh "$MODDIR/bridge-return.sh" >>"$LOG" 2>&1 &
    echo $! >"$RETURN_PID_FILE"
    /system/bin/sh "$MODDIR/bridge-sender.sh" >>"$LOG" 2>&1 &
    echo $! >"$SENDER_PID_FILE"
else
    echo "network bridge disabled; using fallback frame"
fi
echo "$(date '+%Y-%m-%d %H:%M:%S') p0=$(cat "$PRODUCER_PID_FILE") p1=$(cat "$PRODUCER2_PID_FILE") provider=$(cat "$PROVIDER_PID_FILE" 2>/dev/null)"
