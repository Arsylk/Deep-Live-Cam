#!/system/bin/sh
set -eu

STAGE=/data/local/tmp/android-vcam-v049
MODDIR=/data/adb/modules/android-vcam
LOG=/data/local/tmp/android-vcam.log
FILES="bridge.conf module.prop README.md bridge-app-service-common.sh bridge-capture.sh deploy-capture-hot.sh bridge-sender.sh bridge-arch-sender.sh bridge-return.sh bridge-audio.sh bridge-output-common.sh bridge-output-selector.sh service.sh"

for name in $FILES; do
    [ -f "$STAGE/$name" ] || {
        echo "missing staged file: $name" >&2
        exit 1
    }
done

# Keep an explicit rollback copy of the previously deployed v0.4.8 files.
for name in $FILES; do
    if [ -e "$MODDIR/$name" ] && [ ! -e "$MODDIR/$name.v0.4.8" ]; then
        cp "$MODDIR/$name" "$MODDIR/$name.v0.4.8"
    fi
done

for name in $FILES; do
    cp "$STAGE/$name" "$MODDIR/$name"
done
chmod 0644 "$MODDIR/bridge.conf" "$MODDIR/module.prop" "$MODDIR/README.md"
chmod 0755 "$MODDIR/bridge-capture.sh" "$MODDIR/bridge-sender.sh" \
    "$MODDIR/deploy-capture-hot.sh" \
    "$MODDIR/bridge-arch-sender.sh" \
    "$MODDIR/bridge-return.sh" "$MODDIR/bridge-audio.sh" \
    "$MODDIR/bridge-app-service-common.sh" \
    "$MODDIR/bridge-output-common.sh" \
    "$MODDIR/bridge-output-selector.sh" "$MODDIR/service.sh"

. "$MODDIR/bridge.conf"
. "$MODDIR/bridge-app-service-common.sh"

stop_worker() {
    pid_file=$1
    [ -s "$pid_file" ] || return 0
    worker_pid=$(cat "$pid_file" 2>/dev/null || true)
    [ -n "$worker_pid" ] || return 0
    child_pids=$(ps -A -o PID,PPID | awk -v parent="$worker_pid" \
        'NR > 1 && $2 == parent { print $1 }')
    for child_pid in $child_pids; do
        kill "$child_pid" 2>/dev/null || true
    done
    kill "$worker_pid" 2>/dev/null || true
    sleep 1
    for child_pid in $child_pids; do
        kill -0 "$child_pid" 2>/dev/null && kill -9 "$child_pid" 2>/dev/null || true
    done
    kill -0 "$worker_pid" 2>/dev/null && kill -9 "$worker_pid" 2>/dev/null || true
}

# The provider and vcam_mjpeg_fifo producer remain alive. Camera2 ID 120 never
# unregisters; it shows its fallback only while these transport workers swap.
stop_worker /data/local/tmp/android-vcam-output-selector.pid
stop_worker /data/local/tmp/android-vcam-return.pid
stop_worker /data/local/tmp/android-vcam-audio.pid
stop_worker /data/local/tmp/android-vcam-sender.pid
stop_worker /data/local/tmp/android-vcam-arch-sender.pid
stop_worker /data/local/tmp/android-vcam-capture.pid
stop_worker /data/local/tmp/android-vcam-preview.pid

# A shell killed during an earlier hot update can leave its direct FFmpeg
# child reparented to init. These endpoint matches belong exclusively to this
# module; remove them before rebinding the same reserved sockets.
stale_ffmpeg_pids=$(ps -A -o PID,ARGS | awk '
    /ffmpeg/ &&
    (/127\.0\.0\.1:10020/ || /127\.0\.0\.1:10022/ ||
     /127\.0\.0\.1:10023/ || /127\.0\.0\.1:10024/ ||
     /127\.0\.0\.1:10025/ || /127\.0\.0\.1:10026/ ||
     /127\.0\.0\.1:10027/ ||
     /0\.0\.0\.0:10001/ || /192\.168\.1\.11:1000[01]/ ||
     /192\.168\.1\.35:10000/) { print $1 }
')
for stale_pid in $stale_ffmpeg_pids; do
    kill "$stale_pid" 2>/dev/null || true
done
sleep 1

if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ]; then
    nohup /system/bin/sh "$MODDIR/bridge-capture.sh" >>"$LOG" 2>&1 </dev/null &
    echo $! >/data/local/tmp/android-vcam-capture.pid
fi
nohup /system/bin/sh "$MODDIR/bridge-output-selector.sh" >>"$LOG" 2>&1 </dev/null &
echo $! >/data/local/tmp/android-vcam-output-selector.pid
nohup /system/bin/sh "$MODDIR/bridge-return.sh" >>"$LOG" 2>&1 </dev/null &
echo $! >/data/local/tmp/android-vcam-return.pid
nohup /system/bin/sh "$MODDIR/bridge-audio.sh" >>"$LOG" 2>&1 </dev/null &
echo $! >/data/local/tmp/android-vcam-audio.pid
if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ]; then
    start_app_foreground_service "physical front-camera bridge service" \
        -a dev.vcam.app.CONFIGURE \
        -n dev.vcam.app/.CameraBridgeService \
        --es lens_facing front --ez persist true || true
else
    am stopservice --user 0 \
        -n dev.vcam.app/.CameraBridgeService >/dev/null 2>&1 || true
fi
start_app_foreground_service "return-audio app service" \
    -a dev.vcam.app.RETURN_AUDIO_START \
    -n dev.vcam.app/.ReturnAudioService || true
if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ]; then
    nohup /system/bin/sh "$MODDIR/bridge-sender.sh" >>"$LOG" 2>&1 </dev/null &
    echo $! >/data/local/tmp/android-vcam-sender.pid
    if [ "${ARCH_PROCESSOR_ENABLED:-1}" = "1" ]; then
        nohup /system/bin/sh "$MODDIR/bridge-arch-sender.sh" >>"$LOG" 2>&1 </dev/null &
        echo $! >/data/local/tmp/android-vcam-arch-sender.pid
    fi
fi
sleep 2

required_pid_files="\
    /data/local/tmp/android-vcam-capture.pid \
    /data/local/tmp/android-vcam-output-selector.pid \
    /data/local/tmp/android-vcam-return.pid \
    /data/local/tmp/android-vcam-audio.pid \
    /data/local/tmp/android-vcam-sender.pid"
if [ "${PHONE_CAPTURE_ENABLED:-1}" = "1" ] && \
   [ "${ARCH_PROCESSOR_ENABLED:-1}" = "1" ]; then
    required_pid_files="$required_pid_files \
        /data/local/tmp/android-vcam-arch-sender.pid"
fi
if [ "${PHONE_CAPTURE_ENABLED:-1}" != "1" ]; then
    required_pid_files="\
        /data/local/tmp/android-vcam-output-selector.pid \
        /data/local/tmp/android-vcam-return.pid \
        /data/local/tmp/android-vcam-audio.pid"
fi
for pid_file in $required_pid_files; do
    worker_pid=$(cat "$pid_file")
    kill -0 "$worker_pid"
done
echo "android-vcam v0.4.9 hot update complete; Camera2 ID 120 stayed registered"
