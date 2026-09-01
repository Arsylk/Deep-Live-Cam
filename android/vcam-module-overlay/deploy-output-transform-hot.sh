#!/usr/bin/env sh
set -eu

usage() {
    echo "usage: $0 [ADB_SERIAL]" >&2
}

case $# in
    0|1) ;;
    *) usage; exit 2 ;;
esac
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE="$SCRIPT_DIR/bridge-output-common.sh"
ADB_BIN=${ANDROID_ADB_BIN:-adb}

command -v "$ADB_BIN" >/dev/null 2>&1 || {
    echo "adb was not found; install Android platform-tools or set ANDROID_ADB_BIN" >&2
    exit 1
}
[ -f "$SOURCE" ] || {
    echo "missing transform helper: $SOURCE" >&2
    exit 1
}
sh -n "$SOURCE" || {
    echo "refusing to deploy a syntactically invalid bridge-output-common.sh" >&2
    exit 1
}

serial=${1:-${ANDROID_ADB_SERIAL:-}}
if [ -z "$serial" ]; then
    devices_output=$("$ADB_BIN" devices) || {
        echo "unable to query adb devices" >&2
        exit 1
    }
    serial=$(
        printf '%s\n' "$devices_output" |
            awk 'NR > 1 && $2 == "device" { print $1; exit }'
    )
fi
[ -n "$serial" ] || {
    echo "no authorized Android device; connect one or set ANDROID_ADB_SERIAL" >&2
    exit 1
}

device_state=$("$ADB_BIN" -s "$serial" get-state 2>/dev/null || true)
device_state=$(printf '%s' "$device_state" | tr -d '\r\n')
[ "$device_state" = "device" ] || {
    echo "Android device '$serial' is not authorized and online (state: ${device_state:-unavailable})" >&2
    exit 1
}

remote_stage="/data/local/tmp/android-vcam-output-common.hot.$$"
staged=0
cleanup() {
    if [ "$staged" = "1" ]; then
        "$ADB_BIN" -s "$serial" shell rm -f "$remote_stage" >/dev/null 2>&1 || true
        staged=0
    fi
}
trap cleanup 0 1 2 15

# This is intentionally the only adb push in the helper. The installed
# selector is retained and restarted so that it sources the new pure helper.
"$ADB_BIN" -s "$serial" push "$SOURCE" "$remote_stage"
staged=1

remote_body='
set -eu

MODDIR=/data/adb/modules/android-vcam
LOG=/data/local/tmp/android-vcam.log
SELECTOR_PID_FILE=/data/local/tmp/android-vcam-output-selector.pid
WORKER_PID_FILE=/data/local/tmp/android-vcam-output-worker.pid
TARGET="$MODDIR/bridge-output-common.sh"
SELECTOR="$MODDIR/bridge-output-selector.sh"

fail() {
    echo "output-transform hot update failed: $*" >&2
    exit 1
}

[ "$(id -u)" = "0" ] || fail "root access was not granted"
[ -d "$MODDIR" ] || fail "installed module is missing: $MODDIR"
[ -f "$TARGET" ] || fail "installed transform helper is missing: $TARGET"
[ -x "$SELECTOR" ] || fail "installed output selector is missing: $SELECTOR"
[ -f "$STAGE" ] || fail "staged transform helper is missing: $STAGE"
/system/bin/sh -n "$STAGE" || fail "staged transform helper has invalid shell syntax"
grep -q "^build_output_filter()" "$STAGE" || fail "staged file has no build_output_filter helper"

# Resolve and validate the two narrowly scoped process IDs before changing any
# installed file. A stale PID which now belongs to another process must never
# become a kill target.
old_selector=""
old_worker=""
[ ! -s "$SELECTOR_PID_FILE" ] || old_selector=$(cat "$SELECTOR_PID_FILE" 2>/dev/null || true)
[ ! -s "$WORKER_PID_FILE" ] || old_worker=$(cat "$WORKER_PID_FILE" 2>/dev/null || true)
case "$old_selector" in
    "") ;;
    *[!0-9]*|0|1) fail "invalid selector PID file: $old_selector" ;;
esac
case "$old_worker" in
    "") ;;
    *[!0-9]*|0|1) fail "invalid decoder-worker PID file: $old_worker" ;;
esac
if [ -n "$old_selector" ] && kill -0 "$old_selector" 2>/dev/null; then
    tr "\000" " " <"/proc/$old_selector/cmdline" 2>/dev/null |
        grep -Fq "$SELECTOR" || fail "selector PID belongs to another process"
fi
if [ -n "$old_worker" ] && kill -0 "$old_worker" 2>/dev/null; then
    tr "\000" " " <"/proc/$old_worker/cmdline" 2>/dev/null |
        grep -Fq -- "-f image2pipe /data/local/tmp/android-vcam-return.mjpg" ||
        fail "decoder-worker PID belongs to another process"
fi

backup="$TARGET.hot-backup"
backup_next="$backup.new.$$"
next="$TARGET.new.$$"
rm -f "$backup_next" "$next"
cp "$TARGET" "$backup_next"
chmod 0755 "$backup_next"
mv -f "$backup_next" "$backup"
cp "$STAGE" "$next"
chmod 0755 "$next"
mv -f "$next" "$TARGET"

# Signal only the preflight-validated selector PID. Its TERM trap normally
# stops the decoder itself; the explicit worker cleanup only handles a stale
# selector which cannot run that trap.
if [ -n "$old_selector" ] && kill -0 "$old_selector" 2>/dev/null; then
    kill "$old_selector" 2>/dev/null || true
fi

attempt=0
while [ -n "$old_selector" ] && kill -0 "$old_selector" 2>/dev/null && [ "$attempt" -lt 30 ]; do
    sleep 0.1
    attempt=$((attempt + 1))
done
if [ -n "$old_selector" ] && kill -0 "$old_selector" 2>/dev/null; then
    kill -9 "$old_selector" 2>/dev/null || true
fi
if [ -n "$old_worker" ] && kill -0 "$old_worker" 2>/dev/null; then
    kill "$old_worker" 2>/dev/null || true
    sleep 0.2
    kill -0 "$old_worker" 2>/dev/null && kill -9 "$old_worker" 2>/dev/null || true
fi
rm -f "$SELECTOR_PID_FILE" "$WORKER_PID_FILE"

nohup /system/bin/sh "$SELECTOR" >>"$LOG" 2>&1 </dev/null &
new_selector=$!
echo "$new_selector" >"$SELECTOR_PID_FILE"
sleep 1
if ! kill -0 "$new_selector" 2>/dev/null; then
    # Restore the known-running helper and leave Camera2 on a recovered
    # selector, while still returning failure so the bad update is visible.
    rollback="$TARGET.rollback.$$"
    cp "$backup" "$rollback"
    chmod 0755 "$rollback"
    mv -f "$rollback" "$TARGET"
    rm -f "$SELECTOR_PID_FILE" "$WORKER_PID_FILE"
    nohup /system/bin/sh "$SELECTOR" >>"$LOG" 2>&1 </dev/null &
    recovery_selector=$!
    echo "$recovery_selector" >"$SELECTOR_PID_FILE"
    sleep 1
    if kill -0 "$recovery_selector" 2>/dev/null; then
        fail "new selector exited; previous transform helper was restored"
    fi
    fail "new selector and rollback selector both exited; inspect $LOG"
fi

rm -f "$STAGE"
echo "output transform updated; selector pid=$new_selector; Camera2 session was retained"
'

# Feed the privileged script over stdin so no quoting-sensitive temporary
# deployment script is created on the device.
{
    printf 'STAGE="%s"\n' "$remote_stage"
    printf '%s\n' "$remote_body"
} | "$ADB_BIN" -s "$serial" shell "su -c /system/bin/sh"

cleanup
echo "Android output transform hot update succeeded on $serial"
