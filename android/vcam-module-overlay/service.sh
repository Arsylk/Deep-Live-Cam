#!/system/bin/sh

MODDIR=${0%/*}
LOG=/data/local/tmp/android-vcam.log
PRODUCER_PID_FILE=/data/local/tmp/android-vcam-producer.pid
PROVIDER_PID_FILE=/data/local/tmp/android-vcam-provider.pid
PROVIDER_SUPERVISOR_PID_FILE=/data/local/tmp/android-vcam-provider-supervisor.pid
RETURN_PID_FILE=/data/local/tmp/android-vcam-return.pid
AUDIO_PID_FILE=/data/local/tmp/android-vcam-audio.pid
SENDER_PID_FILE=/data/local/tmp/android-vcam-sender.pid
ARCH_SENDER_PID_FILE=/data/local/tmp/android-vcam-arch-sender.pid
CAPTURE_PID_FILE=/data/local/tmp/android-vcam-capture.pid
OUTPUT_PID_FILE=/data/local/tmp/android-vcam-output-selector.pid
VINTF_TARGET=/vendor/etc/vintf/manifest/android.hardware.ir-service.lineage.xml
EXT_IMPL_TARGET=/vendor/lib64/camera.device@3.4-external-impl.so
VCAM_VENDOR="$MODDIR/system/vendor"
RETURN_FIFO=/data/local/tmp/android-vcam-return.mjpg
QCOM_PROVIDER=android.hardware.camera.provider@2.4-service_64

. "$MODDIR/bridge.conf"
. "$MODDIR/bridge-app-service-common.sh"

exec >>"$LOG" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') android-vcam single-camera service starting"

validate_port() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
}

private_ports="\
$ANDROID_ENCODER_PORT $ANDROID_RAW_CAMERA_PORT $ANDROID_SENDER_FEED_PORT \
$ANDROID_PROCESSED_FEED_PORT $ANDROID_RETURN_AUDIO_PORT \
$ANDROID_RETURN_AV_PORT $ANDROID_ARCH_SENDER_FEED_PORT"
seen_ports=" "
for private_port in $private_ports; do
    if ! validate_port "$private_port"; then
        echo "invalid private Android transport port: $private_port"
        exit 1
    fi
    case "$seen_ports" in
        *" $private_port "*)
            echo "duplicate private Android transport port: $private_port"
            exit 1
            ;;
    esac
    seen_ports="$seen_ports$private_port "
done

rm -f "$PRODUCER_PID_FILE" "$PROVIDER_PID_FILE" "$PROVIDER_SUPERVISOR_PID_FILE" \
      "$RETURN_PID_FILE" "$SENDER_PID_FILE" "$CAPTURE_PID_FILE" "$OUTPUT_PID_FILE"
rm -f "$AUDIO_PID_FILE" "$ARCH_SENDER_PID_FILE"
rm -f /data/local/tmp/android-vcam-windows-sender.state \
      /data/local/tmp/android-vcam-windows-sender.progress \
      /data/local/tmp/android-vcam-arch-sender.state \
      /data/local/tmp/android-vcam-arch-sender.progress

# Apollo's separate read-only vendor partition is not projected by KernelSU.
# The early hook must install the two known-good vendor-file binds before the
# external provider starts. Fail closed rather than silently losing ID 120.
if ! grep -q " $VINTF_TARGET " /proc/mounts; then
    echo "VINTF fragment is not mounted; refusing late HAL registration"
    exit 1
fi
if ! grep -q " $EXT_IMPL_TARGET " /proc/mounts; then
    echo "external-camera implementation is not mounted; refusing HAL registration"
    exit 1
fi

# This module never replaces /dev/video0 or /dev/video1 and never stops the
# Qualcomm provider. Wait briefly for Android init to bring the physical
# provider up, then leave its lifecycle entirely under init's control.
attempt=0
while [ -z "$(pidof "$QCOM_PROVIDER" 2>/dev/null)" ] && [ "$attempt" -lt 20 ]; do
    sleep 0.5
    attempt=$((attempt + 1))
done
qcom_pid=$(pidof "$QCOM_PROVIDER" 2>/dev/null || true)
if [ -n "$qcom_pid" ]; then
    echo "physical Qualcomm provider remains active (pid $qcom_pid)"
else
    echo "WARNING: Qualcomm provider was not observed; it was not stopped or modified"
fi

# v0.4.9 owns exactly one software node. The AOSP external provider maps the
# V4L2 index 20 to Camera2 external ID 120 (the provider's +100 namespace).
if ! grep -q '^v4l2loopback ' /proc/modules; then
    insmod "$MODDIR/v4l2loopback.ko" \
        video_nr=20 \
        card_label=ProcessedCamera \
        exclusive_caps=0 \
        max_width=1920 \
        max_height=1080 \
        max_buffers=4 \
        max_openers=8
fi

attempt=0
while [ ! -e /dev/video20 ] && [ "$attempt" -lt 50 ]; do
    sleep 0.1
    attempt=$((attempt + 1))
done

if [ ! -e /dev/video20 ]; then
    echo "loopback node /dev/video20 did not appear"
    exit 1
fi

# The packaged external-camera implementation loads this module-owned config.
cp "$VCAM_VENDOR/etc/external_camera_config.xml" /data/local/tmp/vcam.xml
chmod 0644 /data/local/tmp/vcam.xml
chcon u:object_r:vendor_configs_file:s0 /data/local/tmp/vcam.xml 2>/dev/null || true

if [ ! -p "$RETURN_FIFO" ]; then
    rm -f "$RETURN_FIFO"
    mkfifo "$RETURN_FIFO"
fi
chmod 0660 "$RETURN_FIFO"

# The durable output policy is intentionally outside the module directory so
# a module upgrade preserves it. Do not rewrite an existing user selection.
OUTPUT_CONTROL_FILE=${ANDROID_OUTPUT_CONTROL_FILE:-/data/adb/android-vcam-output.conf}
if [ ! -e "$OUTPUT_CONTROL_FILE" ]; then
    output_control_tmp="${OUTPUT_CONTROL_FILE}.tmp.$$"
    umask 077
    {
        printf 'version=1\n'
        printf 'enabled=%s\n' "${ANDROID_OUTPUT_ENABLED:-1}"
        printf 'mirror=%s\n' "${ANDROID_OUTPUT_MIRROR:-0}"
        printf 'rotation=%s\n' "${ANDROID_OUTPUT_ROTATION:-0}"
        printf 'revision=1\n'
    } >"$output_control_tmp" && mv -f "$output_control_tmp" "$OUTPUT_CONTROL_FILE"
fi

"$VCAM_VENDOR/bin/vcam_mjpeg_fifo" \
    /dev/video20 \
    "$RETURN_FIFO" \
    "$VCAM_VENDOR/etc/android-vcam/vcam-fallback-1280x720.jpg" \
    "$VIDEO_WIDTH" "$VIDEO_HEIGHT" "$VIDEO_FPS" >>"$LOG" 2>&1 &
echo $! >"$PRODUCER_PID_FILE"

sleep 1

/system/bin/sh "$MODDIR/provider-supervisor.sh" &
echo $! >"$PROVIDER_SUPERVISOR_PID_FILE"

attempt=0
while [ ! -s "$PROVIDER_PID_FILE" ] && [ "$attempt" -lt 50 ]; do
    sleep 0.1
    attempt=$((attempt + 1))
done

if [ "$BRIDGE_ENABLED" = "1" ]; then
    # Termux is in credential-encrypted app storage and is intentionally absent
    # while user 0 is locked after boot. Camera 120 remains on its local
    # fallback until the first unlock makes FFmpeg available.
    wait_count=0
    while [ ! -x "$TERMUX_PREFIX/bin/ffmpeg" ]; do
        if [ $((wait_count % 30)) -eq 0 ]; then
            echo "waiting for user unlock and Termux FFmpeg; camera 120 is on fallback"
        fi
        wait_count=$((wait_count + 1))
        sleep 2
    done

    echo "Termux FFmpeg available; starting Android-processor bridge"
    if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ]; then
        /system/bin/sh "$MODDIR/bridge-capture.sh" >>"$LOG" 2>&1 &
        echo $! >"$CAPTURE_PID_FILE"
    else
        echo "phone physical-camera capture disabled; using processed webcam return"
    fi
    /system/bin/sh "$MODDIR/bridge-output-selector.sh" >>"$LOG" 2>&1 &
    echo $! >"$OUTPUT_PID_FILE"
    /system/bin/sh "$MODDIR/bridge-return.sh" >>"$LOG" 2>&1 &
    echo $! >"$RETURN_PID_FILE"
    /system/bin/sh "$MODDIR/bridge-audio.sh" >>"$LOG" 2>&1 &
    echo $! >"$AUDIO_PID_FILE"
    if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ]; then
        # Phone-input mode owns the physical front camera and publishes its
        # encoded feed. Camera-busy errors stay inside the app's retry loop.
        start_app_foreground_service "physical front-camera bridge service" \
            -a dev.vcam.app.CONFIGURE \
            -n dev.vcam.app/.CameraBridgeService \
            --es lens_facing front --ez persist true || true
    else
        am stopservice --user 0 \
            -n dev.vcam.app/.CameraBridgeService >/dev/null 2>&1 || true
    fi
    # The app owns the AudioTrack endpoint. Root can start this non-exported
    # service after first unlock; failure is non-fatal so Camera2 stays usable
    # even when the companion APK is absent or being upgraded.
    start_app_foreground_service "return-audio app service" \
        -a dev.vcam.app.RETURN_AUDIO_START \
        -n dev.vcam.app/.ReturnAudioService || true
    if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ]; then
        /system/bin/sh "$MODDIR/bridge-sender.sh" >>"$LOG" 2>&1 &
        echo $! >"$SENDER_PID_FILE"
        if [ "${ARCH_PROCESSOR_ENABLED:-1}" = "1" ]; then
            /system/bin/sh "$MODDIR/bridge-arch-sender.sh" >>"$LOG" 2>&1 &
            echo $! >"$ARCH_SENDER_PID_FILE"
        else
            echo "Arch local-processor sender disabled; Windows slot 0 is unaffected"
        fi
    fi
else
    echo "network bridge disabled; using fallback frame"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') producer=$(cat "$PRODUCER_PID_FILE") provider=$(cat "$PROVIDER_PID_FILE" 2>/dev/null) provider_supervisor=$(cat "$PROVIDER_SUPERVISOR_PID_FILE" 2>/dev/null) capture=$(cat "$CAPTURE_PID_FILE" 2>/dev/null) return=$(cat "$RETURN_PID_FILE" 2>/dev/null) audio=$(cat "$AUDIO_PID_FILE" 2>/dev/null) windows_sender=$(cat "$SENDER_PID_FILE" 2>/dev/null) arch_sender=$(cat "$ARCH_SENDER_PID_FILE" 2>/dev/null) output=$(cat "$OUTPUT_PID_FILE" 2>/dev/null)"
