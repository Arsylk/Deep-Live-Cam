#!/system/bin/sh

MODDIR=${0%/*}
. "$MODDIR/bridge.conf"

FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
DETAIL_LOG=/data/local/tmp/android-vcam-sender-ffmpeg.log
ATTEMPT_LOG=${DETAIL_LOG}.current
PROGRESS=/data/local/tmp/android-vcam-windows-sender.progress
STATE_FILE=/data/local/tmp/android-vcam-windows-sender.state
retry_delay=1
attempt=0
ffmpeg_pid=""

write_state() {
    sender_status=$1
    exit_status=${2:--}
    runtime_seconds=${3:-0}
    state_tmp="${STATE_FILE}.tmp.$$"
    umask 077
    {
        printf 'version=1\n'
        printf 'route=windows-slot-0\n'
        printf 'status=%s\n' "$sender_status"
        printf 'target_host=%s\n' "$WINDOWS_HOST"
        printf 'target_port=%s\n' "$WINDOWS_INPUT_PORT"
        printf 'source_port=%s\n' "$ANDROID_SENDER_FEED_PORT"
        printf 'attempt=%s\n' "$attempt"
        printf 'worker_pid=%s\n' "${ffmpeg_pid:--}"
        printf 'last_exit=%s\n' "$exit_status"
        printf 'last_runtime_seconds=%s\n' "$runtime_seconds"
        printf 'retry_seconds=%s\n' "$retry_delay"
        printf 'progress_file=%s\n' "$PROGRESS"
        printf 'updated_epoch=%s\n' "$(date +%s)"
    } >"$state_tmp" && mv -f "$state_tmp" "$STATE_FILE"
}

stop_sender() {
    trap - TERM INT
    if [ -n "$ffmpeg_pid" ] && kill -0 "$ffmpeg_pid" 2>/dev/null; then
        kill "$ffmpeg_pid" 2>/dev/null || true
        wait "$ffmpeg_pid" 2>/dev/null || true
    fi
    ffmpeg_pid=""
    write_state stopped 0 0
    exit 0
}

trap stop_sender TERM INT

# This worker may fail/retry forever while slot 0 is unselected. The separate
# capture fanout and local Camera2 fallback never stop with it.
while true; do
    attempt=$((attempt + 1))
    rm -f "$PROGRESS"
    started_at=$(date +%s)
    "$FFMPEG" \
        -hide_banner -loglevel warning -nostdin \
        -fflags nobuffer -flags low_delay \
        -analyzeduration 1000000 -probesize 1000000 \
        -f mpegts -i "udp://127.0.0.1:$ANDROID_SENDER_FEED_PORT?fifo_size=1000000&overrun_nonfatal=1" \
        -map 0:v:0 -an -c:v copy \
        -mpegts_flags resend_headers -muxdelay 0 -muxpreload 0 \
        -stats_period 0.5 -progress "$PROGRESS" \
        -f mpegts \
        "srt://$WINDOWS_HOST:$WINDOWS_INPUT_PORT?mode=caller&transtype=live&messageapi=1&latency=$SRT_LATENCY_US&connect_timeout=3000&timeout=5000000&tlpktdrop=1&pkt_size=1316" \
        >/dev/null 2>"$ATTEMPT_LOG" &
    ffmpeg_pid=$!
    write_state running - 0
    wait "$ffmpeg_pid"
    status=$?
    ffmpeg_pid=""
    mv -f "$ATTEMPT_LOG" "$DETAIL_LOG"
    runtime=$(($(date +%s) - started_at))
    [ "$runtime" -lt 30 ] || retry_delay=1
    write_state backoff "$status" "$runtime"
    echo "$(date '+%Y-%m-%d %H:%M:%S') sender exited status=$status after ${runtime}s; retrying in ${retry_delay}s (details: $DETAIL_LOG)"
    sleep "$retry_delay"
    if [ "$runtime" -lt 30 ] && [ "$retry_delay" -lt 10 ]; then
        retry_delay=$((retry_delay * 2))
        [ "$retry_delay" -gt 10 ] && retry_delay=10
    fi
done
