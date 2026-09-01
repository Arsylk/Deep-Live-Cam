#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASE_ZIP="$SCRIPT_DIR/android-vcam-module-v0.3.0.zip"
OVERLAY_DIR="$SCRIPT_DIR/vcam-module-overlay"
OUTPUT_ZIP="$SCRIPT_DIR/android-vcam-module-v0.4.9.zip"
STAGE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/android-vcam-v049.XXXXXX")
TEMP_ZIP="$STAGE_DIR.zip"

cleanup() {
    rm -rf "$STAGE_DIR"
    rm -f "$TEMP_ZIP"
}
trap cleanup EXIT HUP INT TERM

for tool in unzip zip; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "missing packaging tool: $tool" >&2
        exit 1
    }
done

[ -f "$BASE_ZIP" ] || {
    echo "missing v0.3.0 binary base: $BASE_ZIP" >&2
    exit 1
}

unzip -q "$BASE_ZIP" -d "$STAGE_DIR"

# Retain v0.3.0's tested binaries, single-/dev/video20 topology, and the two
# top-level sources used by Apollo's proven vendor-file bind setup. Then apply
# the v0.4.9 lifecycle, reconnect-safe processor transports, and hot
# output-control scripts.

for name in \
    bridge-capture.sh \
    deploy-capture-hot.sh \
    bridge-app-service-common.sh \
    bridge-arch-sender.sh \
    bridge-output-selector.sh \
    bridge-output-common.sh \
    bridge-return.sh \
    bridge-audio.sh \
    bridge-sender.sh \
    bridge.conf \
    customize.sh \
    external_camera_config.xml \
    module.prop \
    post-fs-data.sh \
    provider-supervisor.sh \
    README.md \
    service.sh \
    uninstall.sh; do
    cp "$OVERLAY_DIR/$name" "$STAGE_DIR/$name"
done
cp "$OVERLAY_DIR/external_camera_config.xml" \
   "$STAGE_DIR/system/vendor/etc/external_camera_config.xml"
rm -f "$STAGE_DIR/external_camera_config.xml"

chmod 0755 \
    "$STAGE_DIR/bridge-capture.sh" \
    "$STAGE_DIR/deploy-capture-hot.sh" \
    "$STAGE_DIR/bridge-app-service-common.sh" \
    "$STAGE_DIR/bridge-arch-sender.sh" \
    "$STAGE_DIR/bridge-output-selector.sh" \
    "$STAGE_DIR/bridge-output-common.sh" \
    "$STAGE_DIR/bridge-return.sh" \
    "$STAGE_DIR/bridge-audio.sh" \
    "$STAGE_DIR/bridge-sender.sh" \
    "$STAGE_DIR/customize.sh" \
    "$STAGE_DIR/post-fs-data.sh" \
    "$STAGE_DIR/provider-supervisor.sh" \
    "$STAGE_DIR/service.sh" \
    "$STAGE_DIR/uninstall.sh"
chmod 0644 "$STAGE_DIR/bridge.conf" "$STAGE_DIR/module.prop" "$STAGE_DIR/README.md" \
    "$STAGE_DIR/system/vendor/etc/external_camera_config.xml"

(cd "$STAGE_DIR" && zip -X -q -r "$TEMP_ZIP" .)
unzip -tq "$TEMP_ZIP" >/dev/null
mv -f "$TEMP_ZIP" "$OUTPUT_ZIP"

echo "$OUTPUT_ZIP"
