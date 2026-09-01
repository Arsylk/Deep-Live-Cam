#!/system/bin/sh

MODDIR=${0%/*}
LOG=/data/local/tmp/android-vcam.log
PROVIDER_PID_FILE=/data/local/tmp/android-vcam-provider.pid
PROVIDER="$MODDIR/system/vendor/bin/hw/android.hardware.camera.provider@2.4-external-service_64"
retry_delay=1

# Launch only external/0. This supervisor never signals Android's Qualcomm
# legacy/0 provider and therefore preserves the physical camera IDs.
while true; do
    started_at=$(date +%s)
    LD_LIBRARY_PATH=/vendor/lib64:/vendor/lib64/hw:/system/lib64 \
        "$PROVIDER" >>"$LOG" 2>&1 &
    provider_pid=$!
    echo "$provider_pid" >"$PROVIDER_PID_FILE"
    wait "$provider_pid"
    status=$?
    runtime=$(($(date +%s) - started_at))
    echo "$(date '+%Y-%m-%d %H:%M:%S') external provider exited status=$status after ${runtime}s; retrying in ${retry_delay}s"
    sleep "$retry_delay"
    if [ "$runtime" -ge 30 ]; then
        retry_delay=1
    elif [ "$retry_delay" -lt 10 ]; then
        retry_delay=$((retry_delay * 2))
        [ "$retry_delay" -gt 10 ] && retry_delay=10
    fi
done
