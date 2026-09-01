#!/system/bin/sh

MODDIR=${0%/*}
. "$MODDIR/bridge.conf"
. "$MODDIR/bridge-output-common.sh"

FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
RETURN_FIFO=/data/local/tmp/android-vcam-return.mjpg
PROGRESS=/data/local/tmp/android-vcam-return.progress
WORKER_PID_FILE=/data/local/tmp/android-vcam-output-worker.pid
DETAIL_LOG=/data/local/tmp/android-vcam-output-ffmpeg.log
CONTROL_FILE=${ANDROID_OUTPUT_CONTROL_FILE:-/data/adb/android-vcam-output.conf}
STATE_FILE=${ANDROID_OUTPUT_STATE_FILE:-/data/local/tmp/android-vcam-output.state}
current=""
worker_pid=""
last_progress_frame=-1
last_progress_at=0
processed_fresh_since=0
control_enabled=${ANDROID_OUTPUT_ENABLED:-1}
control_mirror=${ANDROID_OUTPUT_MIRROR:-0}
control_rotation=${ANDROID_OUTPUT_ROTATION:-0}
control_revision=0
last_control_signature=""
last_state_signature=""

read_control_value() {
    control_key=$1
    [ -r "$CONTROL_FILE" ] || return 1
    sed -n "s/^${control_key}=//p" "$CONTROL_FILE" 2>/dev/null | tail -n 1
}

read_control() {
    next_enabled=${ANDROID_OUTPUT_ENABLED:-1}
    next_mirror=${ANDROID_OUTPUT_MIRROR:-0}
    next_rotation=${ANDROID_OUTPUT_ROTATION:-0}
    next_revision=0

    if [ -r "$CONTROL_FILE" ]; then
        candidate=$(read_control_value enabled)
        case "$candidate" in 0|1) next_enabled=$candidate ;; esac
        candidate=$(read_control_value mirror)
        case "$candidate" in 0|1) next_mirror=$candidate ;; esac
        candidate=$(read_control_value rotation)
        case "$candidate" in 0|90|180|270) next_rotation=$candidate ;; esac
        candidate=$(read_control_value revision)
        case "$candidate" in ''|*[!0-9]*) ;; *) next_revision=$candidate ;; esac
    fi

    control_enabled=$next_enabled
    control_mirror=$next_mirror
    control_rotation=$next_rotation
    control_revision=$next_revision
}

write_state() {
    worker_alive=0
    if [ -n "$worker_pid" ] && kill -0 "$worker_pid" 2>/dev/null; then
        worker_alive=1
    fi
    state_signature="$control_enabled:$control_mirror:$control_rotation:$control_revision:$current:$worker_alive"
    [ "$state_signature" != "$last_state_signature" ] || return 0
    state_tmp="${STATE_FILE}.tmp.$$"
    umask 077
    {
        printf 'version=1\n'
        printf 'enabled=%s\n' "$control_enabled"
        printf 'mirror=%s\n' "$control_mirror"
        printf 'rotation=%s\n' "$control_rotation"
        printf 'revision=%s\n' "$control_revision"
        printf 'source=%s\n' "${current:-starting}"
        printf 'worker_alive=%s\n' "$worker_alive"
    } >"$state_tmp" && mv -f "$state_tmp" "$STATE_FILE"
    last_state_signature=$state_signature
}

stop_worker() {
    [ -n "$worker_pid" ] || return 0
    stopped_pid=$worker_pid
    kill "$stopped_pid" 2>/dev/null || true
    stop_attempt=0
    while kill -0 "$stopped_pid" 2>/dev/null && [ "$stop_attempt" -lt 10 ]; do
        sleep 0.1
        stop_attempt=$((stop_attempt + 1))
    done
    if kill -0 "$stopped_pid" 2>/dev/null; then
        kill -9 "$stopped_pid" 2>/dev/null || true
    fi
    wait "$stopped_pid" 2>/dev/null || true
    worker_pid=""
    rm -f "$WORKER_PID_FILE"
}

start_worker() {
    source_name=$1
    if [ "$source_name" = "placeholder" ]; then
        current=placeholder
        rm -f "$WORKER_PID_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Camera2 source -> placeholder"
        return 0
    fi
    if [ "$source_name" = "processed" ]; then
        source_port=$ANDROID_PROCESSED_FEED_PORT
    else
        source_port=$ANDROID_RAW_CAMERA_PORT
    fi
    video_filter=$(build_output_filter \
        "$control_mirror" "$control_rotation" \
        "$VIDEO_WIDTH" "$VIDEO_HEIGHT" "$VIDEO_FPS") || {
        echo "invalid output transform; keeping Camera2 on placeholder"
        current=placeholder
        return 1
    }
    "$FFMPEG" \
        -y -hide_banner -loglevel warning -nostdin \
        -fflags nobuffer -flags low_delay \
        -analyzeduration 0 -probesize 32768 \
        -f mpegts -i "udp://127.0.0.1:$source_port?fifo_size=1000000&overrun_nonfatal=1&reuse=1" \
        -map 0:v:0 -an \
        -vf "$video_filter" \
        -pix_fmt yuvj420p -c:v mjpeg -q:v 2 -f image2pipe "$RETURN_FIFO" \
        >/dev/null 2>"$DETAIL_LOG" &
    worker_pid=$!
    echo "$worker_pid" >"$WORKER_PID_FILE"
    current=$source_name
    echo "$(date '+%Y-%m-%d %H:%M:%S') Camera2 source -> $current (pid $worker_pid)"
}

processed_is_fresh() {
    if [ ! -s "$PROGRESS" ]; then
        last_progress_frame=-1
        last_progress_at=0
        return 1
    fi
    progress_frame=$(
        tail -n 32 "$PROGRESS" 2>/dev/null | awk -F= '
            $1 == "frame" { value = $2 }
            END {
                gsub(/[[:space:]]/, "", value)
                print value
            }
        '
    )
    case "$progress_frame" in
        ''|*[!0-9]*) return 1 ;;
    esac
    now=$(date +%s)
    if [ "$last_progress_frame" -lt 0 ] || [ "$progress_frame" -lt "$last_progress_frame" ]; then
        # A newly observed/recreated file establishes a baseline only. It is
        # not evidence of live video until a later frame counter advances.
        last_progress_frame=$progress_frame
        last_progress_at=0
    elif [ "$progress_frame" -gt "$last_progress_frame" ]; then
        last_progress_frame=$progress_frame
        last_progress_at=$now
    fi
    [ "$last_progress_at" -gt 0 ] && \
        [ $((now - last_progress_at)) -le "$PROCESSED_STALE_SECONDS" ]
}

trap 'stop_worker; exit 0' TERM INT EXIT
read_control
if [ "$control_enabled" = "1" ]; then
    start_worker raw
else
    start_worker placeholder
fi
last_control_signature="$control_enabled:$control_mirror:$control_rotation:$control_revision"
write_state
while true; do
    read_control
    control_signature="$control_enabled:$control_mirror:$control_rotation:$control_revision"
    processed_fresh=0
    if processed_is_fresh; then
        processed_fresh=1
    fi

    desired=placeholder
    if [ "$control_enabled" = "1" ]; then
        desired=raw
    fi
    if [ "$control_enabled" = "1" ] && [ "$processed_fresh" = "1" ]; then
        now=$(date +%s)
        if [ "$current" = "processed" ]; then
            desired=processed
        else
            [ "$processed_fresh_since" -gt 0 ] || processed_fresh_since=$now
            if [ $((now - processed_fresh_since)) -ge "$PROCESSED_RECOVERY_SECONDS" ]; then
                desired=processed
            fi
        fi
    elif [ "$processed_fresh" != "1" ]; then
        processed_fresh_since=0
    fi

    if [ "$desired" != "$current" ] || \
       [ "$control_signature" != "$last_control_signature" ]; then
        stop_worker
        start_worker "$desired"
    elif [ "$desired" != "placeholder" ] && \
         { [ -z "$worker_pid" ] || ! kill -0 "$worker_pid" 2>/dev/null; }; then
        stop_worker
        start_worker "$desired"
    fi
    last_control_signature=$control_signature
    write_state
    sleep 0.25
done
