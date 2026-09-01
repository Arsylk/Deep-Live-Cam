#!/system/bin/sh

MODDIR=${0%/*}
. "$MODDIR/bridge.conf"

FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
DETAIL_LOG=/data/local/tmp/android-vcam-audio-ffmpeg.log
ATTEMPT_LOG=${DETAIL_LOG}.current
PROGRESS=/data/local/tmp/android-vcam-audio.progress
retry_delay=1

# Decode the optional AAC return track into a process-local, raw microphone
# feed.  Xposed clients consume this only while their redirected front camera
# is active.  This worker is deliberately isolated from the Camera2 writer so
# audio failures never stall or unregister camera 120.
while true; do
    rm -f "$PROGRESS"
    started_at=$(date +%s)
    "$FFMPEG" \
        -hide_banner -loglevel warning -nostdin \
        -fflags nobuffer -flags low_delay \
        -analyzeduration 1000000 -probesize 262144 \
        -f mpegts \
        -i "udp://127.0.0.1:$ANDROID_RETURN_AV_PORT?fifo_size=262144&overrun_nonfatal=1&reuse=1" \
        -map 0:a:0 -vn \
        -af "aresample=$RETURN_AUDIO_SAMPLE_RATE:async=1:first_pts=0" \
        -c:a pcm_s16le -ar "$RETURN_AUDIO_SAMPLE_RATE" \
        -ac "$RETURN_AUDIO_CHANNELS" \
        -flush_packets 1 -stats_period 0.5 -progress "$PROGRESS" \
        -f s16le \
        "udp://127.0.0.1:$ANDROID_RETURN_AUDIO_PORT?pkt_size=960" \
        >/dev/null 2>"$ATTEMPT_LOG"
    status=$?
    mv -f "$ATTEMPT_LOG" "$DETAIL_LOG"
    runtime=$(($(date +%s) - started_at))
    [ "$runtime" -lt 30 ] || retry_delay=1
    echo "$(date '+%Y-%m-%d %H:%M:%S') audio relay exited status=$status after ${runtime}s; retrying in ${retry_delay}s (details: $DETAIL_LOG)"
    sleep "$retry_delay"
    if [ "$runtime" -lt 30 ] && [ "$retry_delay" -lt 10 ]; then
        retry_delay=$((retry_delay * 2))
        [ "$retry_delay" -gt 10 ] && retry_delay=10
    fi
done
