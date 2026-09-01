#!/system/bin/sh

MODDIR=${0%/*}
. "$MODDIR/bridge.conf"

FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
DETAIL_LOG=/data/local/tmp/android-vcam-capture-ffmpeg.log
ATTEMPT_LOG=${DETAIL_LOG}.current
retry_delay=1

# This is the only TCP client of dev.vcam.app's Camera2 encoder. It fans the
# already encoded phone stream to three distinct loopback-only feeds: raw
# Camera2 fallback, Windows slot 0, and the Arch local processor. Every tee
# slave has its own UDP destination, so the network workers can reconnect or
# fail independently without splitting a common datagram stream.
#
# MediaCodec supplies SPS/PPS in-band when this client first connects. Cache
# those parameter sets before repairing timestamps, then prepend the cached
# copy to every keyframe. A UDP/SRT worker which starts between keyframes can
# therefore join at the next IDR instead of decoding slices without PPS 0.
# remove=0 deliberately retains an encoder-provided in-band copy as a safe
# fallback; a duplicate SPS/PPS on an IDR is valid H.264 and negligible here.
while true; do
    started_at=$(date +%s)
    "$FFMPEG" \
        -hide_banner -loglevel warning -nostdin \
        -fflags +genpts+discardcorrupt+nobuffer -flags low_delay \
        -analyzeduration 0 -probesize 65536 \
        -framerate "$VIDEO_FPS" -f h264 \
        -i "tcp://127.0.0.1:$ANDROID_ENCODER_PORT?timeout=5000000" \
        -map 0:v:0 -an -c:v copy \
        -bsf:v "extract_extradata=remove=0,setts=pts=N/($VIDEO_FPS*TB):dts=N/($VIDEO_FPS*TB):duration=1/($VIDEO_FPS*TB),dump_extra=freq=keyframe" \
        -muxdelay 0 -muxpreload 0 \
        -f tee \
        "[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]udp://127.0.0.1:$ANDROID_RAW_CAMERA_PORT?pkt_size=1316|[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]udp://127.0.0.1:$ANDROID_SENDER_FEED_PORT?pkt_size=1316|[f=mpegts:mpegts_flags=resend_headers:onfail=ignore]udp://127.0.0.1:$ANDROID_ARCH_SENDER_FEED_PORT?pkt_size=1316" \
        >/dev/null 2>"$ATTEMPT_LOG"
    status=$?
    mv -f "$ATTEMPT_LOG" "$DETAIL_LOG"
    runtime=$(($(date +%s) - started_at))
    [ "$runtime" -lt 30 ] || retry_delay=1
    echo "$(date '+%Y-%m-%d %H:%M:%S') capture exited status=$status after ${runtime}s; retrying in ${retry_delay}s (details: $DETAIL_LOG)"
    sleep "$retry_delay"
    if [ "$runtime" -lt 30 ] && [ "$retry_delay" -lt 10 ]; then
        retry_delay=$((retry_delay * 2))
        [ "$retry_delay" -gt 10 ] && retry_delay=10
    fi
done
