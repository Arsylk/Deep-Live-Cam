#!/system/bin/sh
MODDIR=${0%/*}
VCAM_VENDOR="$MODDIR/system/vendor"

# Mount the VINTF fragment that registers the external camera provider.
VINTF_SOURCE="$MODDIR/android-vcam-vintf.xml"
VINTF_TARGET=/vendor/etc/vintf/manifest/android.hardware.ir-service.lineage.xml
if [ -f "$VINTF_SOURCE" ] && [ -f "$VINTF_TARGET" ]; then
    mount -o bind "$VINTF_SOURCE" "$VINTF_TARGET"
fi

# Mount the patched external camera HAL
HAL_SOURCE="$MODDIR/camera.device@3.4-external-impl.so"
HAL_TARGET=/vendor/lib64/camera.device@3.4-external-impl.so
if [ -f "$HAL_SOURCE" ] && [ -f "$HAL_TARGET" ]; then
    mount -o bind "$HAL_SOURCE" "$HAL_TARGET"
fi
