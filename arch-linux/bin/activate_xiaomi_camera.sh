#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARCH_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PACKAGE="$ARCH_DIR/v4l2loopback-custom/v4l2loopback-custom-0.15.4-3-any.pkg.tar.zst"
DESKTOP_USER=${DEPLOY_DESKTOP_USER:-arsylk}
RECEIVER_WAS_ACTIVE=0
COMPLETED=0

if ((EUID != 0)); then
    echo "Run this activation helper as root." >&2
    exit 1
fi
if [[ ! -f "$PACKAGE" ]]; then
    echo "Missing custom loopback package: $PACKAGE" >&2
    exit 1
fi
if ! id "$DESKTOP_USER" >/dev/null 2>&1; then
    echo "Desktop user does not exist: $DESKTOP_USER" >&2
    exit 1
fi

if systemctl is-active --quiet deep-live-cam-receiver.service; then
    RECEIVER_WAS_ACTIVE=1
fi

restore_receiver_on_error() {
    local status=$?
    if ((status != 0 && RECEIVER_WAS_ACTIVE && !COMPLETED)); then
        systemctl start deep-live-cam-receiver.service || true
    fi
    exit "$status"
}
trap restore_receiver_on_error EXIT

backup_dir="/var/backups/deep-live-cam-xiaomi-$(date +%Y%m%d-%H%M%S)"
install -d -m 0700 "$backup_dir"
for existing in \
    /etc/modprobe.d/deep-live-cam-v4l2loopback.conf \
    /etc/udev/rules.d/70-deep-live-cam.rules \
    /etc/wireplumber/wireplumber.conf.d/51-deep-live-cam-camera-policy.conf \
    /etc/systemd/system/deep-live-cam-receiver.service \
    /usr/local/lib/deep-live-cam-arch/publish_virtual_camera.py; do
    if [[ -f "$existing" ]]; then
        install -m 0600 "$existing" "$backup_dir/$(basename "$existing")"
    fi
done

if pacman -Q v4l2loopback-dkms >/dev/null 2>&1; then
    pacman -Rdd --noconfirm v4l2loopback-dkms
fi
pacman -U --noconfirm "$PACKAGE"

install -Dm0644 "$ARCH_DIR/config/deep-live-cam-v4l2loopback.conf" \
    /etc/modprobe.d/deep-live-cam-v4l2loopback.conf
install -Dm0644 "$ARCH_DIR/config/70-deep-live-cam.rules" \
    /etc/udev/rules.d/70-deep-live-cam.rules
install -Dm0644 "$ARCH_DIR/config/51-deep-live-cam-camera-policy.conf" \
    /etc/wireplumber/wireplumber.conf.d/51-deep-live-cam-camera-policy.conf
install -Dm0755 "$ARCH_DIR/bin/publish_virtual_camera.py" \
    /usr/local/lib/deep-live-cam-arch/publish_virtual_camera.py
install -Dm0644 "$ARCH_DIR/systemd/deep-live-cam-receiver.service" \
    /etc/systemd/system/deep-live-cam-receiver.service
manager_target=/usr/local/lib/deep-live-cam-arch/dlc_manager
install -d -m 0755 "$manager_target" "$manager_target/pages"
find "$manager_target" -name '__pycache__' -type d -prune -exec rm -rf -- {} +
find "$manager_target" -name '*.py' -type f -delete
while IFS= read -r module; do
    relative=${module#"$ARCH_DIR/bin/dlc_manager/"}
    install -D -m 0644 "$module" "$manager_target/$relative"
done < <(find "$ARCH_DIR/bin/dlc_manager" -name '*.py' -type f -not -path '*/__pycache__/*')
install -Dm0644 "$ARCH_DIR/README.md" \
    /usr/local/share/doc/deep-live-cam-arch/README.md

systemctl daemon-reload
udevadm control --reload

systemctl stop deep-live-cam-receiver.service
if ! modprobe -r v4l2loopback; then
    echo "v4l2loopback is still busy:" >&2
    fuser -v /dev/video42 >&2 || true
    exit 1
fi
modprobe v4l2loopback
udevadm trigger --action=add --subsystem-match=video4linux
udevadm settle
systemctl start deep-live-cam-receiver.service

desktop_uid=$(id -u "$DESKTOP_USER")
runuser -u "$DESKTOP_USER" -- env \
    XDG_RUNTIME_DIR="/run/user/$desktop_uid" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$desktop_uid/bus" \
    systemctl --user restart wireplumber.service

COMPLETED=1
echo "backup=$backup_dir"
echo "xiaomi_camera_activated=1"
