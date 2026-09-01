#!/system/bin/sh

# A previous experimental topology left this persistent device-specific switch
# enabled on Apollo. Its exact vendor implementation is undocumented, so make
# only the known stale-state repair at install time and leave every other value
# untouched. Android init remains responsible for the Qualcomm provider.
camera_disable=$(getprop persist.vendor.camera.disable 2>/dev/null)
if [ "$camera_disable" = "1" ]; then
    setprop persist.vendor.camera.disable 0
    repaired=$(getprop persist.vendor.camera.disable 2>/dev/null)
    if command -v ui_print >/dev/null 2>&1; then
        ui_print "- Restored persist.vendor.camera.disable: 1 -> $repaired"
    else
        echo "Restored persist.vendor.camera.disable: 1 -> $repaired"
    fi
fi
