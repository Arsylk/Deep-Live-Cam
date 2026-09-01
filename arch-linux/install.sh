#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUSTOM_LOOPBACK_VERSION=0.15.4-3
CUSTOM_LOOPBACK_PACKAGE="$SCRIPT_DIR/v4l2loopback-custom/v4l2loopback-custom-${CUSTOM_LOOPBACK_VERSION}-any.pkg.tar.zst"
WINDOWS_HOST=""
CAMERA=""
START_SERVICES=1
LEGACY_SHADOW=0

usage() {
    cat <<'EOF'
Usage: sudo ./arch-linux/install.sh --windows-host IP [options]

Options:
  --windows-host IP   Reserved/static LAN IP of the Windows CUDA machine
  --camera PATH       Stable physical-camera path (the detected Sonix path is default)
  --no-start          Install files but do not enable or start services
  --legacy-shadow     Explicitly retain the old invasive video0/video1 takeover
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --windows-host)
            WINDOWS_HOST=${2:?--windows-host requires a value}
            shift 2
            ;;
        --camera)
            CAMERA=${2:?--camera requires a value}
            shift 2
            ;;
        --no-start)
            START_SERVICES=0
            shift
            ;;
        --legacy-shadow)
            LEGACY_SHADOW=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((EUID != 0)); then
    echo "Run this installer as root (for example, sudo $0 --windows-host 192.168.1.35)." >&2
    exit 1
fi

if [[ ! -f /etc/arch-release ]]; then
    echo "This installer supports Arch Linux and Arch-derived systems." >&2
    exit 1
fi

if [[ ! -e "/usr/lib/modules/$(uname -r)/build" ]]; then
    echo "Install the headers matching the running kernel ($(uname -r)) before continuing." >&2
    exit 1
fi

missing_packages=()
for package in acl ffmpeg v4l-utils pyside6 python-numpy; do
    pacman -Q "$package" >/dev/null 2>&1 || missing_packages+=("$package")
done
if ((${#missing_packages[@]})); then
    pacman -S --needed --noconfirm "${missing_packages[@]}"
fi

# Record the live state before changing packages or configuration.  A loaded
# module is never removed by this installer: a topology change is staged for
# the next controlled reboot so active camera file descriptors remain valid.
MODULE_WAS_LOADED=0
[[ -d /sys/module/v4l2loopback ]] && MODULE_WAS_LOADED=1
MIGRATION_WAS_STAGED=0
[[ -f /etc/deep-live-cam-arch.conf.nextboot ]] && MIGRATION_WAS_STAGED=1
PACKAGE_MIGRATED=0
modprobe_conf=/etc/modprobe.d/deep-live-cam-v4l2loopback.conf
old_options=$(grep '^options v4l2loopback' "$modprobe_conf" 2>/dev/null || true)

installed_custom_version=$(pacman -Q v4l2loopback-custom 2>/dev/null | awk '{print $2}' || true)
if [[ "$installed_custom_version" != "$CUSTOM_LOOPBACK_VERSION" ]]; then
    if [[ ! -f "$CUSTOM_LOOPBACK_PACKAGE" ]]; then
        echo "Missing safe custom loopback package: $CUSTOM_LOOPBACK_PACKAGE" >&2
        echo "Build it with: (cd '$SCRIPT_DIR/v4l2loopback-custom' && makepkg -Csf)" >&2
        exit 1
    fi
    echo "Installing the safe QUERYCAP identity-cloning v4l2loopback module."
    if pacman -Q v4l2loopback-dkms >/dev/null 2>&1; then
        pacman -Rdd --noconfirm v4l2loopback-dkms
    fi
    pacman -U --noconfirm "$CUSTOM_LOOPBACK_PACKAGE"
    PACKAGE_MIGRATED=1
fi

install -d -m 0755 /usr/local/lib/deep-live-cam-arch
install -m 0755 "$SCRIPT_DIR/bin/sender.py" /usr/local/lib/deep-live-cam-arch/sender.py
install -m 0755 "$SCRIPT_DIR/bin/receiver.py" /usr/local/lib/deep-live-cam-arch/receiver.py
if ((LEGACY_SHADOW)); then
    install -m 0755 "$SCRIPT_DIR/bin/shadow.py" /usr/local/lib/deep-live-cam-arch/shadow.py
else
    rm -f /usr/local/lib/deep-live-cam-arch/shadow.py
fi
install -m 0755 "$SCRIPT_DIR/bin/status.py" /usr/local/lib/deep-live-cam-arch/status.py
install -m 0755 "$SCRIPT_DIR/bin/tester.py" /usr/local/lib/deep-live-cam-arch/tester.py
install -m 0644 "$SCRIPT_DIR/bin/source_history.py" /usr/local/lib/deep-live-cam-arch/source_history.py
install -m 0755 "$SCRIPT_DIR/bin/device_control.py" /usr/local/lib/deep-live-cam-arch/device_control.py
install -m 0755 "$SCRIPT_DIR/bin/android_bridge.py" /usr/local/lib/deep-live-cam-arch/android_bridge.py
install -m 0755 "$SCRIPT_DIR/bin/camera_adapters.py" /usr/local/lib/deep-live-cam-arch/camera_adapters.py
install -m 0644 "$SCRIPT_DIR/bin/camera_profiles.py" /usr/local/lib/deep-live-cam-arch/camera_profiles.py
install -m 0755 "$SCRIPT_DIR/bin/select_receiver_source.py" /usr/local/lib/deep-live-cam-arch/select_receiver_source.py
install -m 0755 "$SCRIPT_DIR/bin/configure_receiver_output.py" /usr/local/lib/deep-live-cam-arch/configure_receiver_output.py
install -m 0755 "$SCRIPT_DIR/bin/local_processor_control.py" /usr/local/lib/deep-live-cam-arch/local_processor_control.py
install -m 0755 "$SCRIPT_DIR/bin/windows_phone_relay.py" /usr/local/lib/deep-live-cam-arch/windows_phone_relay.py
install -m 0755 "$SCRIPT_DIR/bin/configure_phone_relay.py" /usr/local/lib/deep-live-cam-arch/configure_phone_relay.py
install -m 0755 "$SCRIPT_DIR/bin/deploy_live_webcam_stack.py" /usr/local/lib/deep-live-cam-arch/deploy_live_webcam_stack.py
install -m 0644 "$SCRIPT_DIR/bin/pipeline_topology.py" /usr/local/lib/deep-live-cam-arch/pipeline_topology.py
install -m 0755 "$SCRIPT_DIR/bin/configure_camera.py" /usr/local/lib/deep-live-cam-arch/configure_camera.py
install -m 0755 "$SCRIPT_DIR/bin/publish_virtual_camera.py" /usr/local/lib/deep-live-cam-arch/publish_virtual_camera.py
install -m 0644 "$SCRIPT_DIR/bin/common.py" /usr/local/lib/deep-live-cam-arch/common.py

# The native manager is a package that sits beside tester.py. Install every
# module in it, and drop modules that no longer exist in the checkout so a
# stale file cannot shadow the current one. The launcher symlink resolves to
# this directory, which puts the package on sys.path automatically.
manager_target=/usr/local/lib/deep-live-cam-arch/dlc_manager
install -d -m 0755 "$manager_target" "$manager_target/pages"
find "$manager_target" -name '__pycache__' -type d -prune -exec rm -rf -- {} +
find "$manager_target" -name '*.py' -type f -delete
while IFS= read -r module; do
    relative=${module#"$SCRIPT_DIR/bin/dlc_manager/"}
    install -D -m 0644 "$module" "$manager_target/$relative"
done < <(find "$SCRIPT_DIR/bin/dlc_manager" -name '*.py' -type f -not -path '*/__pycache__/*')

ln -sfn /usr/local/lib/deep-live-cam-arch/status.py /usr/local/bin/deep-live-cam-status
ln -sfn /usr/local/lib/deep-live-cam-arch/tester.py /usr/local/bin/deep-live-cam-tester

# Remove only compatibility overrides installed by older releases.  Never
# remove a real user/package-owned DroidCam binary or desktop entry.
for managed_link in /usr/local/bin/droidcam /usr/local/bin/droidcam-cli; do
    if [[ -L "$managed_link" ]] \
        && [[ "$(readlink "$managed_link")" == /usr/local/lib/deep-live-cam-arch/droidcam-compat ]]; then
        rm -f -- "$managed_link"
    fi
done
rm -f /usr/local/lib/deep-live-cam-arch/droidcam-compat
rm -f /etc/modprobe.d/deep-live-cam-droidcam-compat.conf
if [[ -f /usr/local/share/applications/droidcam.desktop ]] \
    && grep -q '^Exec=/usr/local/bin/droidcam$' /usr/local/share/applications/droidcam.desktop; then
    rm -f /usr/local/share/applications/droidcam.desktop
fi

install -d -m 0755 /usr/local/share/doc/deep-live-cam-arch
install -m 0644 "$SCRIPT_DIR/README.md" /usr/local/share/doc/deep-live-cam-arch/README.md
install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-sender.service" /etc/systemd/system/deep-live-cam-sender.service
install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-receiver.service" /etc/systemd/system/deep-live-cam-receiver.service
install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-phone-processed.service" /etc/systemd/system/deep-live-cam-phone-processed.service
install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-phone-return-relay.service" /etc/systemd/system/deep-live-cam-phone-return-relay.service
if ((LEGACY_SHADOW)); then
    install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-shadow.service" /etc/systemd/system/deep-live-cam-shadow.service
else
    rm -f /etc/systemd/system/deep-live-cam-shadow.service
fi
install -m 0644 "$SCRIPT_DIR/systemd/deep-live-cam-topology-migration.service" /etc/systemd/system/deep-live-cam-topology-migration.service
install -m 0644 "$SCRIPT_DIR/config/deep-live-cam.modules-load.conf" /etc/modules-load.d/deep-live-cam.conf
# Some DroidCam packages leave a v4l2loopback_dc modules-load request behind.
# Fail that obsolete fork closed so it cannot race the one authoritative
# upstream v4l2loopback instance for /dev/video nodes at the next boot.
install -m 0644 "$SCRIPT_DIR/config/deep-live-cam-droidcam-modprobe.conf" \
    /etc/modprobe.d/deep-live-cam-disable-v4l2loopback-dc.conf
install -m 0644 "$SCRIPT_DIR/config/70-deep-live-cam.rules" /etc/udev/rules.d/70-deep-live-cam.rules
rm -f /etc/udev/rules.d/99-deep-live-cam-camera-access.rules
install -m 0644 "$SCRIPT_DIR/config/72-deep-live-cam-camera-access.rules" \
    /etc/udev/rules.d/72-deep-live-cam-camera-access.rules
install -d -m 0755 /etc/wireplumber/wireplumber.conf.d
install -m 0644 "$SCRIPT_DIR/config/51-deep-live-cam-camera-policy.conf" \
    /etc/wireplumber/wireplumber.conf.d/51-deep-live-cam-camera-policy.conf
install -m 0644 "$SCRIPT_DIR/config/deep-live-cam.sysusers.conf" /usr/lib/sysusers.d/deep-live-cam.conf
install -m 0644 "$SCRIPT_DIR/config/deep-live-cam-tester.desktop" /usr/local/share/applications/deep-live-cam-tester.desktop
install -d -m 0755 /usr/local/share/icons/hicolor/512x512/apps
install -m 0644 "$SCRIPT_DIR/assets/deep-live-cam.png" /usr/local/share/icons/hicolor/512x512/apps/deep-live-cam.png
if command -v update-desktop-database >/dev/null; then
    update-desktop-database /usr/local/share/applications
fi
if command -v gtk-update-icon-cache >/dev/null; then
    gtk-update-icon-cache --force --ignore-theme-index /usr/local/share/icons/hicolor
fi

if [[ ! -e /etc/deep-live-cam-arch.conf ]]; then
    install -m 0644 "$SCRIPT_DIR/config/deep-live-cam-arch.conf" /etc/deep-live-cam-arch.conf
fi
RUNTIME_CONFIG_BACKUP=$(mktemp)
install -m 0600 /etc/deep-live-cam-arch.conf "$RUNTIME_CONFIG_BACKUP"
trap 'rm -f -- "$RUNTIME_CONFIG_BACKUP"' EXIT

set_config() {
    local key=$1 value=$2 temporary
    temporary=$(mktemp)
    awk -v key="$key" -v value="$value" '
        BEGIN { found = 0 }
        index($0, key "=") == 1 { print key "=" value; found = 1; next }
        { print }
        END { if (!found) print key "=" value }
    ' /etc/deep-live-cam-arch.conf >"$temporary"
    install -m 0644 "$temporary" /etc/deep-live-cam-arch.conf
    rm -f -- "$temporary"
}

remove_config() {
    local key=$1 temporary
    temporary=$(mktemp)
    awk -v key="$key" 'index($0, key "=") != 1 { print }' \
        /etc/deep-live-cam-arch.conf >"$temporary"
    install -m 0644 "$temporary" /etc/deep-live-cam-arch.conf
    rm -f -- "$temporary"
}

if [[ -n "$WINDOWS_HOST" ]]; then
    if [[ ! "$WINDOWS_HOST" =~ ^[A-Za-z0-9._:-]+$ ]]; then
        echo "Invalid --windows-host value: $WINDOWS_HOST" >&2
        exit 2
    fi
    set_config WINDOWS_HOST "$WINDOWS_HOST"
fi
if [[ -n "$CAMERA" ]]; then
    set_config PHYSICAL_CAMERA "$CAMERA"
fi
# The optional local processor is a system service, but its content-addressed
# source history and qualified/development model packs are owned by the desktop
# user.  Give that service the same XDG data root without hard-coding a home
# directory into the packaged unit.  Root can read the private source files;
# no camera or model data is copied or made world-readable.
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    processor_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    if [[ -n "$processor_home" \
        && -d "$processor_home/.local/share/deep-live-cam" ]]; then
        # The processor-specific values keep InsightFace's offline cache and
        # the manager's source/model history on one account.  They are mapped
        # to HOME/XDG_DATA_HOME only by the processor unit, so the camera
        # sender and receiver do not inherit an unwritable desktop home.
        set_config PROCESSOR_HOME "$processor_home"
        set_config PROCESSOR_XDG_DATA_HOME "$processor_home/.local/share"
        # Clean up the broad names written by the short-lived 0.5 migration
        # candidate before it reached a release deployment.
        remove_config HOME
        remove_config XDG_DATA_HOME
    fi
fi
if ! grep -q '^LOCAL_PREVIEW_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config LOCAL_PREVIEW_PORT 11000
fi
if ! grep -q '^MANAGER_PREVIEW_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config MANAGER_PREVIEW_PORT 11001
fi
if ! grep -q '^NETWORK_SOURCE_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config NETWORK_SOURCE_PORT 11002
fi
if ! grep -q '^NATIVE_PROCESSOR_SOURCE_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config NATIVE_PROCESSOR_SOURCE_PORT 11005
fi
if ! grep -q '^LOCAL_PROCESSED_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config LOCAL_PROCESSED_PORT 11006
fi
if ! grep -q '^LOCAL_PROCESSED_PREVIEW_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config LOCAL_PROCESSED_PREVIEW_PORT 11007
fi
if ! grep -q '^RECEIVER_SOURCE=' /etc/deep-live-cam-arch.conf; then
    set_config RECEIVER_SOURCE windows
fi
if ! grep -q '^MANAGER_OUTPUT_PREVIEW_PORT=' /etc/deep-live-cam-arch.conf; then
    set_config MANAGER_OUTPUT_PREVIEW_PORT 11003
fi
camera_defaults=(
    CAMERA_WIDTH=1280 CAMERA_HEIGHT=720 CAMERA_FPS=30
    CAMERA_BRIGHTNESS=-12 CAMERA_CONTRAST=12 CAMERA_SATURATION=40 CAMERA_HUE=0
    CAMERA_GAMMA=95 CAMERA_GAIN=0 CAMERA_SHARPNESS=2 CAMERA_BACKLIGHT=0 CAMERA_POWER_LINE=1
    CAMERA_AUTO_EXPOSURE=1 CAMERA_EXPOSURE=157
    CAMERA_EXPOSURE_DYNAMIC_FRAMERATE=0
    CAMERA_AUTO_WHITE_BALANCE=1 CAMERA_WHITE_BALANCE=4600
)
for assignment in "${camera_defaults[@]}"; do
    key=${assignment%%=*}
    value=${assignment#*=}
    if ! grep -q "^${key}=" /etc/deep-live-cam-arch.conf; then
        set_config "$key" "$value"
    fi
done
if ! grep -q '^CAMERA_PROFILE=' /etc/deep-live-cam-arch.conf; then
    # An existing unlabelled configuration is user-tuned until explicitly
    # replaced with a measured named profile. New installs get the profile
    # name from the packaged configuration file.
    set_config CAMERA_PROFILE custom
fi

config_value() {
    awk -F= -v key="$1" '$1 == key { print $2; exit }' /etc/deep-live-cam-arch.conf
}

validate_camera_integer() {
    local key=$1 minimum=$2 maximum=$3 value
    value=$(config_value "$key")
    if [[ ! "$value" =~ ^-?[0-9]+$ ]] || ((value < minimum || value > maximum)); then
        echo "$key must be an integer from $minimum to $maximum (got: $value)." >&2
        exit 2
    fi
}

camera_profile=$(config_value CAMERA_PROFILE)
if [[ "$camera_profile" != "natural-indoor" && "$camera_profile" != "custom" ]]; then
    echo "CAMERA_PROFILE must be natural-indoor or custom (got: $camera_profile)." >&2
    exit 2
fi
for key in CAMERA_AUTO_EXPOSURE CAMERA_EXPOSURE_DYNAMIC_FRAMERATE CAMERA_AUTO_WHITE_BALANCE; do
    value=$(config_value "$key")
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "$key must be 0 or 1 (got: $value)." >&2
        exit 2
    fi
done
validate_camera_integer CAMERA_WIDTH 1 7680
validate_camera_integer CAMERA_HEIGHT 1 4320
validate_camera_integer CAMERA_FPS 1 120
validate_camera_integer CAMERA_BRIGHTNESS -64 64
validate_camera_integer CAMERA_CONTRAST 0 64
validate_camera_integer CAMERA_SATURATION 0 128
validate_camera_integer CAMERA_HUE -40 40
validate_camera_integer CAMERA_GAMMA 72 500
validate_camera_integer CAMERA_GAIN 0 100
validate_camera_integer CAMERA_SHARPNESS 0 6
validate_camera_integer CAMERA_BACKLIGHT 0 2
validate_camera_integer CAMERA_POWER_LINE 0 2
validate_camera_integer CAMERA_EXPOSURE 1 5000
validate_camera_integer CAMERA_WHITE_BALANCE 2800 6500

if [[ "$camera_profile" == "natural-indoor" ]]; then
    natural_indoor_values=(
        CAMERA_INPUT_FORMAT=mjpeg CAMERA_WIDTH=1280 CAMERA_HEIGHT=720 CAMERA_FPS=30
        CAMERA_BRIGHTNESS=-12 CAMERA_CONTRAST=12 CAMERA_SATURATION=40 CAMERA_HUE=0
        CAMERA_GAMMA=95 CAMERA_GAIN=0 CAMERA_SHARPNESS=2 CAMERA_BACKLIGHT=0
        CAMERA_POWER_LINE=1 CAMERA_AUTO_EXPOSURE=1 CAMERA_EXPOSURE=157
        CAMERA_EXPOSURE_DYNAMIC_FRAMERATE=0 CAMERA_AUTO_WHITE_BALANCE=1
        CAMERA_WHITE_BALANCE=4600
    )
    for assignment in "${natural_indoor_values[@]}"; do
        key=${assignment%%=*}
        expected=${assignment#*=}
        actual=$(config_value "$key")
        if [[ "$actual" != "$expected" ]]; then
            echo "CAMERA_PROFILE=natural-indoor requires $key=$expected (got: $actual)." >&2
            exit 2
        fi
    done
fi
topology_defaults=(
    ARCH_HOST=192.168.1.11
    DEVICE_ID=arch-webcam
    DEVICE_SLOT=1
    WINDOWS_INPUT_PORT=10002
    WINDOWS_RETURN_PORT=10003
    WINDOWS_SELECTED_STREAM_PORT=10010
    ANDROID_BRIDGE_ENABLED=1
    ANDROID_HOST=192.168.1.12
    ANDROID_ADB_SERIAL=b83607a5
    ANDROID_MODEL=M2007J3SY
    ANDROID_CAMERA_ID=120
    ANDROID_BRIDGE_PACKAGE=dev.vcam.app
)
for assignment in "${topology_defaults[@]}"; do
    key=${assignment%%=*}
    value=${assignment#*=}
    if ! grep -q "^${key}=" /etc/deep-live-cam-arch.conf; then
        set_config "$key" "$value"
    fi
done
# This package is the Arch slot-1 client. Migrate legacy shared-port installs
# deterministically; leaving 10001/10002 in place would collide with slot 0.
for assignment in \
    DEVICE_ID=arch-webcam DEVICE_SLOT=1 WINDOWS_INPUT_PORT=10002 \
    WINDOWS_RETURN_PORT=10003 WINDOWS_SELECTED_STREAM_PORT=10010 \
    ANDROID_BRIDGE_PACKAGE=dev.vcam.app \
    LOCAL_PREVIEW_PORT=11000 MANAGER_PREVIEW_PORT=11001 \
    NETWORK_SOURCE_PORT=11002 MANAGER_OUTPUT_PREVIEW_PORT=11003 \
    NATIVE_PROCESSOR_SOURCE_PORT=11005 LOCAL_PROCESSED_PORT=11006 \
    LOCAL_PROCESSED_PREVIEW_PORT=11007 \
    WINDOWS_PHONE_RELAY_SOURCE_PORT=11008 LOCAL_PHONE_RELAY_SOURCE_PORT=11009 \
    PHONE_RETURN_RELAY_CONTROL_SOCKET=/run/deep-live-cam/phone-return-relay-control.sock \
    PHONE_RETURN_RELAY_STATE_FILE=/var/lib/deep-live-cam/phone-return-relay.json \
    RECEIVER_SOURCE_STATE_FILE=/var/lib/deep-live-cam/receiver-source.json \
    ANDROID_NATIVE_HEALTH_FILE=/run/deep-live-cam/android-phone-processed-health.json \
    PHONE_RELAY_STALE_SECONDS=3; do
    set_config "${assignment%%=*}" "${assignment#*=}"
done

# The single-node layout is authoritative.  An existing SHADOW_ORIGINAL=1 is
# migrated rather than silently treated as user intent; retaining the old
# takeover requires the explicit --legacy-shadow installer argument.
set_config DROIDCAM_COMPAT 0
if ((LEGACY_SHADOW)); then
    set_config LEGACY_SHADOW 1
    set_config SHADOW_ORIGINAL 1
    set_config VIRTUAL_CAMERAS '"/dev/video0 /dev/video1"'
    set_config VIRTUAL_CAMERA /dev/video0
    set_config LOOPBACK_NODE /dev/video0
    options='devices=2 video_nr=0,1 card_label="Xiaomi Cam,Xiaomi Cam" driver_name="uvcvideo" bus_info="usb-0000:02:00.0-4" exclusive_caps=1,1'
else
    set_config LEGACY_SHADOW 0
    set_config SHADOW_ORIGINAL 0
    set_config VIRTUAL_CAMERAS '"/dev/deep-live-cam"'
    set_config VIRTUAL_CAMERA /dev/deep-live-cam
    set_config LOOPBACK_NODE /dev/video42
    options='devices=1 video_nr=42 card_label="Xiaomi Cam" driver_name="uvcvideo" bus_info="usb-0000:02:00.0-4" exclusive_caps=1'
fi

new_options="options v4l2loopback $options"
{
    echo "# Generated by install.sh. One safe identity-cloning loopback at video42."
    echo "# QUERYCAP is cloned; the unsafe USB sysfs-parent patch remains excluded."
    echo "$new_options"
} > "$modprobe_conf"
if ((LEGACY_SHADOW)); then
    cat > /etc/udev/rules.d/71-deep-live-cam-shadow.rules <<'EOF'
SUBSYSTEM=="video4linux", ACTION=="add", TAG+="systemd", ENV{SYSTEMD_WANTS}+="deep-live-cam-shadow.service"
EOF
else
    rm -f /etc/udev/rules.d/71-deep-live-cam-shadow.rules
fi

# The desktop account needs the video group for the public video42 endpoint.
# The camera-access udev rule moves every non-public camera node to the
# deep-live-cam group and removes their seat ACL; controls are sent through the
# existing service owner rather than opening those nodes from the desktop.
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    if ! id -nG "$SUDO_USER" | grep -qw video; then
        usermod -aG video "$SUDO_USER"
        echo "Added $SUDO_USER to the video group; log out and back in for the public virtual camera."
    fi
fi

systemd-sysusers /usr/lib/sysusers.d/deep-live-cam.conf
systemctl daemon-reload

REBOOT_REQUIRED=0
desired_windows_host=$(config_value WINDOWS_HOST)
if ((MODULE_WAS_LOADED)) && {
    [[ "$old_options" != "$new_options" ]] \
        || ((PACKAGE_MIGRATED)) \
        || ((MIGRATION_WAS_STAGED))
}; then
    REBOOT_REQUIRED=1
    # Keep the currently running services on their matching configuration.
    # The early-boot oneshot commits this complete desired configuration before
    # the clean module and either camera service start.
    install -m 0644 /etc/deep-live-cam-arch.conf /etc/deep-live-cam-arch.conf.nextboot
    install -m 0644 "$RUNTIME_CONFIG_BACKUP" /etc/deep-live-cam-arch.conf
    if [[ "$(config_value SHADOW_ORIGINAL)" == "1" \
        || "$old_options" == *"video_nr=0,1"* ]]; then
        # Mark the still-running old layout explicitly so a userspace service
        # recovery before reboot continues to select its preserved raw node.
        set_config LEGACY_SHADOW 1
        set_config SHADOW_ORIGINAL 1
        set_config VIRTUAL_CAMERAS '"/dev/video0 /dev/video1"'
        set_config VIRTUAL_CAMERA /dev/video0
        set_config LOOPBACK_NODE /dev/video0
    fi
    systemctl enable deep-live-cam-topology-migration.service
else
    rm -f /etc/deep-live-cam-arch.conf.nextboot
    systemctl disable deep-live-cam-topology-migration.service 2>/dev/null || true
    if ((MODULE_WAS_LOADED == 0)) && ! modprobe v4l2loopback; then
        echo "v4l2loopback could not be loaded for $(uname -r). Check the DKMS build output." >&2
        exit 1
    fi
fi

udevadm control --reload
# Do not make udev revisit the live legacy layout.  Its volatile nodes and
# links disappear naturally at the controlled reboot.
if ((REBOOT_REQUIRED == 0)); then
    udevadm trigger --subsystem-match=video4linux
    udevadm settle
fi

configured_windows_host=$desired_windows_host
if command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
    firewall-cmd --permanent --add-rich-rule="rule family=\"ipv4\" source address=\"${configured_windows_host}/32\" port port=\"10003\" protocol=\"udp\" accept"
    firewall-cmd --reload
elif command -v ufw >/dev/null && ufw status | grep -q '^Status: active'; then
    ufw allow in on wlan0 proto udp from "$configured_windows_host" to any port 10003 comment 'Deep Live Cam slot 1 return'
fi

configured_host=$desired_windows_host
if ((START_SERVICES)); then
    if [[ -z "$configured_host" || "$configured_host" == "CHANGE_ME" ]]; then
        echo "--windows-host is required before services can be started." >&2
        exit 2
    fi
    systemctl enable deep-live-cam-receiver.service deep-live-cam-sender.service deep-live-cam-phone-return-relay.service
    if ((LEGACY_SHADOW)); then
        systemctl enable deep-live-cam-shadow.service
    else
        systemctl disable deep-live-cam-shadow.service 2>/dev/null || true
    fi
    if ((REBOOT_REQUIRED)); then
        echo "NOTE: the current camera sessions were left untouched; a controlled reboot is required to commit the new topology." >&2
    else
        # No kernel topology changed under a live process, so normal service
        # deployment can safely pick up the newly installed userspace code.
        if ((LEGACY_SHADOW)); then
            systemctl restart deep-live-cam-shadow.service
        fi
        systemctl restart deep-live-cam-receiver.service deep-live-cam-sender.service deep-live-cam-phone-return-relay.service
    fi
fi

echo
echo "Deep-Live-Cam Arch pipeline installed."
if ((LEGACY_SHADOW)); then
    echo "Legacy shadow:  explicit video0/video1 takeover requested."
    echo "Camera alias:   /dev/video0 (legacy processed feed)"
else
    echo "Virtual camera: /dev/video42, labelled Xiaomi Cam"
    echo "Camera alias:   /dev/deep-live-cam"
    echo "Physical input: reserved for the capture service; its by-id path is unchanged"
    echo "Browser camera: Xiaomi Cam only after 'systemctl --user restart wireplumber'"
fi
if ((REBOOT_REQUIRED)); then
    echo "Migration:      staged; reboot before restarting either camera service"
fi
echo "Local address:  192.168.1.11 (current DHCP address; reserve it in the router)"
echo "Status:         deep-live-cam-status"
echo "Manager:        deep-live-cam-tester"
