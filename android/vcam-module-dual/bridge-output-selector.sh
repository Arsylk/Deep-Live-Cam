#!/system/bin/sh
MODDIR=${0%/*}
. "$MODDIR/bridge.conf"
FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
PROGRESS=/data/local/tmp/android-vcam-return.progress
WORKER_PID_FILE=/data/local/tmp/android-vcam-output-worker.pid
WORKER2_PID_FILE=/data/local/tmp/android-vcam-output-worker2.pid
DETAIL_LOG=/data/local/tmp/android-vcam-output-ffmpeg.log
current=""
worker_pid=""
worker2_pid=""
last_progress_size=0
last_progress_at=0
processed_fresh_since=0
stop_workers() {
    for pid in "$worker_pid" "$worker2_pid"; do
        [ -n "$pid" ] || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    worker_pid=""
    worker2_pid=""
    rm -f "$WORKER_PID_FILE" "$WORKER2_PID_FILE"
}
start_workers() {
    source_name=$1
    if [ "$source_name" = "processed" ]; then
        source_port=$ANDROID_PROCESSED_FEED_PORT
    else
        source_port=$ANDROID_RAW_CAMERA_PORT
    fi
    # Write to /dev/video0 FIFO
    "$FFMPEG" -y -hide_banner -loglevel warning -nostdin \
        -fflags nobuffer -flags low_delay \
        -analyzeduration 0 -probesize 32768 \
        -f mpegts -i "udp://127.0.0.1:$source_port?fifo_size=1000000&overrun_nonfatal=1&reuse=1" \
        -map 0:v:0 -an \
        -vf "scale=$VIDEO_WIDTH:$VIDEO_HEIGHT:flags=fast_bilinear:out_range=full,fps=$VIDEO_FPS" \
        -pix_fmt yuvj420p -c:v mjpeg -q:v 2 -f image2pipe "$RETURN_FIFO" \
        >/dev/null 2>"$DETAIL_LOG" &
    worker_pid=$!
    echo "$worker_pid" >"$WORKER_PID_FILE"
    # Write to /dev/video1 FIFO (same source, tee'd)
    "$FFMPEG" -y -hide_banner -loglevel warning -nostdin \
        -fflags nobuffer -flags low_delay \
        -analyzeduration 0 -probesize 32768 \
        -f mpegts -i "udp://127.0.0.1:$source_port?fifo_size=1000000&overrun_nonfatal=1&reuse=1" \
        -map 0:v:0 -an \
        -vf "scale=$VIDEO_WIDTH:$VIDEO_HEIGHT:flags=fast_bilinear:out_range=full,fps=$VIDEO_FPS" \
        -pix_fmt yuvj420p -c:v mjpeg -q:v 2 -f image2pipe "$RETURN_FIFO2" \
        >/dev/null 2>>"$DETAIL_LOG" &
    worker2_pid=$!
    echo "$worker2_pid" >"$WORKER2_PID_FILE"
    current=$source_name
    echo "$(date '+%Y-%m-%d %H:%M:%S') Camera2 source -> $current (cam0=$worker_pid cam1=$worker2_pid)"
}
processed_is_fresh() {
    [ -s "$PROGRESS" ] || return 1
    progress_size=$(stat -c %s "$PROGRESS" 2>/dev/null) || return 1
    now=$(date +%s)
    if [ "$progress_size" -ne "$last_progress_size" ]; then
        last_progress_size=$progress_size
        last_progress_at=$now
    fi
    [ "$last_progress_at" -gt 0 ] && [ $((now - last_progress_at)) -le "$PROCESSED_STALE_SECONDS" ]
}
workers_alive() {
    kill -0 "$worker_pid" 2>/dev/null && kill -0 "$worker2_pid" 2>/dev/null
}
trap 'stop_workers; exit 0' TERM INT EXIT
start_workers raw
while true; do
    desired=raw
    if processed_is_fresh; then
        now=$(date +%s)
        if [ "$current" = "processed" ]; then
            desired=processed
        else
            [ "$processed_fresh_since" -gt 0 ] || processed_fresh_since=$now
            if [ $((now - processed_fresh_since)) -ge "$PROCESSED_RECOVERY_SECONDS" ]; then
                desired=processed
            fi
        fi
    else
        processed_fresh_since=0
    fi
    if [ "$desired" != "$current" ]; then
        stop_workers
        start_workers "$desired"
    elif ! workers_alive; then
        stop_workers
        start_workers "$desired"
    fi
    sleep 0.25
done
