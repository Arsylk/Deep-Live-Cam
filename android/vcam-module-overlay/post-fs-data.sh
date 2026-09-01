#!/system/bin/sh

MODDIR=${0%/*}
LOG=/data/local/tmp/android-vcam.log
VINTF_SOURCE="$MODDIR/android-vcam-vintf.xml"
VINTF_TARGET=/vendor/etc/vintf/manifest/android.hardware.ir-service.lineage.xml
EXT_IMPL_SOURCE="$MODDIR/camera.device@3.4-external-impl.so"
EXT_IMPL_TARGET=/vendor/lib64/camera.device@3.4-external-impl.so

if [ -f "$LOG" ] && [ "$(wc -c <"$LOG")" -gt 1048576 ]; then
    mv -f "$LOG" "$LOG.1"
fi
exec >>"$LOG" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') android-vcam v0.4.2 early external-provider setup"

# Apollo's separate read-only /vendor partition is not projected from a
# KernelSU module. Preserve the original IR declaration while adding
# external/0 before hwservicemanager reads the device manifest.
if ! grep -q " $VINTF_TARGET " /proc/mounts; then
    chcon u:object_r:vendor_configs_file:s0 "$VINTF_SOURCE" 2>/dev/null || true
    mount --bind "$VINTF_SOURCE" "$VINTF_TARGET"
fi

if grep -q " $VINTF_TARGET " /proc/mounts; then
    echo "VINTF fragment installed"
else
    echo "ERROR: VINTF fragment bind failed"
    exit 1
fi

# Bind only the validated external-camera implementation. This does not cover
# /dev/video0 or /dev/video1 and does not replace/stop the Qualcomm provider.
if ! grep -q " $EXT_IMPL_TARGET " /proc/mounts; then
    chcon u:object_r:vendor_file:s0 "$EXT_IMPL_SOURCE" 2>/dev/null || true
    mount --bind "$EXT_IMPL_SOURCE" "$EXT_IMPL_TARGET"
fi

if grep -q " $EXT_IMPL_TARGET " /proc/mounts; then
    echo "external-camera implementation installed"
else
    echo "ERROR: external-camera implementation bind failed"
    exit 1
fi
