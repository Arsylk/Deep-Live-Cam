#!/system/bin/sh

MODDIR=${0%/*}
. "$MODDIR/bridge.conf"

FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
DETAIL_LOG=/data/local/tmp/android-vcam-return-ffmpeg.log
ATTEMPT_LOG=${DETAIL_LOG}.current
PROGRESS=/data/local/tmp/android-vcam-return.progress
retry_delay=1

# Receive the selected processed return independently of the Camera2 writer.
# The first output remains video-only for Camera2 ID 120.  The second is a
# private A/V copy consumed by bridge-audio.sh; an absent audio track therefore
# cannot take the camera down.
while true; do
    rm -f "$PROGRESS"
    started_at=$(date +%s)
    "$FFMPEG" \
        -hide_banner -loglevel warning -nostdin \
        -fflags nobuffer -flags low_delay \
        -analyzeduration 0 -probesize 32768 \
        -i "srt://0.0.0.0:$ANDROID_RETURN_PORT?mode=listener&transtype=live&messageapi=1&pkt_size=1316&latency=$SRT_LATENCY_US&timeout=5000000&tlpktdrop=1" \
        -map 0:v:0 -an -c:v copy \
        -mpegts_flags resend_headers -muxdelay 0 -muxpreload 0 \
        -stats_period 0.5 -progress "$PROGRESS" \
        -f mpegts "udp://127.0.0.1:$ANDROID_PROCESSED_FEED_PORT?pkt_size=1316" \
        -map 0:v:0 -map 0:a:0? -c copy \
        -mpegts_flags resend_headers -muxdelay 0 -muxpreload 0 \
        -f mpegts "udp://127.0.0.1:$ANDROID_RETURN_AV_PORT?pkt_size=1316" \
        >/dev/null 2>"$ATTEMPT_LOG"
    status=$?
    mv -f "$ATTEMPT_LOG" "$DETAIL_LOG"
    runtime=$(($(date +%s) - started_at))
    [ "$runtime" -lt 30 ] || retry_delay=1
    echo "$(date '+%Y-%m-%d %H:%M:%S') return exited status=$status after ${runtime}s; retrying in ${retry_delay}s (details: $DETAIL_LOG)"
    sleep "$retry_delay"
    if [ "$runtime" -lt 30 ] && [ "$retry_delay" -lt 10 ]; then
        retry_delay=$((retry_delay * 2))
        [ "$retry_delay" -gt 10 ] && retry_delay=10
    fi
done
