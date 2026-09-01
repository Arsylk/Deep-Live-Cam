#!/system/bin/sh
set -eu

STAGE=${1:-/data/local/tmp/android-vcam-v049-capture}
MODDIR=/data/adb/modules/android-vcam
LOG=/data/local/tmp/android-vcam.log
CAPTURE_PID_FILE=/data/local/tmp/android-vcam-capture.pid
FILES="bridge-capture.sh module.prop README.md deploy-capture-hot.sh"

[ "$(id -u)" = "0" ] || {
    echo "capture hot update must run as root" >&2
    exit 1
}

for name in $FILES; do
    [ -f "$STAGE/$name" ] || {
        echo "missing staged file: $name" >&2
        exit 1
    }
done

. "$MODDIR/bridge.conf"
FFMPEG="$TERMUX_PREFIX/bin/ffmpeg"
export LD_LIBRARY_PATH="$TERMUX_PREFIX/lib"
for filter in extract_extradata setts dump_extra; do
    "$FFMPEG" -hide_banner -bsfs 2>/dev/null | grep -qx "$filter" || {
        echo "installed Android FFmpeg lacks required bitstream filter: $filter" >&2
        exit 1
    }
done

# Install the validated files atomically while the old capture shell is still
# running, keeping the interruption to one TCP reconnect and one requested IDR.
for name in $FILES; do
    if [ -e "$MODDIR/$name" ] && [ ! -e "$MODDIR/$name.v0.4.8" ]; then
        cp "$MODDIR/$name" "$MODDIR/$name.v0.4.8"
    fi
    next="$MODDIR/$name.new.$$"
    cp "$STAGE/$name" "$next"
    case "$name" in
        *.sh) chmod 0755 "$next" ;;
        *) chmod 0644 "$next" ;;
    esac
    mv -f "$next" "$MODDIR/$name"
done

# Stop only the capture shell and its FFmpeg reader. CameraBridgeService keeps
# Camera2 and MediaCodec running; the provider, producer, output selector,
# return/audio workers, and both UDP/SRT sender workers are deliberately left
# untouched.
capture_pid=""
[ ! -s "$CAPTURE_PID_FILE" ] || capture_pid=$(cat "$CAPTURE_PID_FILE" 2>/dev/null || true)
if [ -n "$capture_pid" ]; then
    child_pids=$(ps -A -o PID,PPID | awk -v parent="$capture_pid" \
        'NR > 1 && $2 == parent { print $1 }')
    for child_pid in $child_pids; do
        kill "$child_pid" 2>/dev/null || true
    done
    kill "$capture_pid" 2>/dev/null || true
    sleep 1
    for child_pid in $child_pids; do
        kill -0 "$child_pid" 2>/dev/null && kill -9 "$child_pid" 2>/dev/null || true
    done
    kill -0 "$capture_pid" 2>/dev/null && kill -9 "$capture_pid" 2>/dev/null || true
fi

# Clean up only an orphaned copy of this module's unique TCP capture reader.
capture_endpoint="tcp://127.0.0.1:$ANDROID_ENCODER_PORT"
stale_capture_pids=$(ps -A -o PID,ARGS | awk -v endpoint="$capture_endpoint" '
    /ffmpeg/ && index($0, endpoint) { print $1 }
')
for stale_pid in $stale_capture_pids; do
    kill "$stale_pid" 2>/dev/null || true
done
sleep 1

nohup /system/bin/sh "$MODDIR/bridge-capture.sh" >>"$LOG" 2>&1 </dev/null &
capture_pid=$!
echo "$capture_pid" >"$CAPTURE_PID_FILE"
sleep 2
kill -0 "$capture_pid"

echo "android-vcam v0.4.9 capture updated; Camera2/provider and downstream workers stayed running"
