#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERIAL=${ANDROID_SERIAL:-$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')}
[[ -n "$SERIAL" ]] || { echo "No ADB device is connected" >&2; exit 1; }
temporary=
cleanup() {
    if [[ -n "$temporary" ]]; then
        adb -s "$SERIAL" shell rm -f "$temporary" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

[[ -f "$SCRIPT_DIR/build/vcam-mobile.apk" ]] || "$SCRIPT_DIR/build.sh"
[[ -f "$SCRIPT_DIR/model-pack/manifest.json" ]] || "$SCRIPT_DIR/prepare-models.sh"

adb -s "$SERIAL" install -r "$SCRIPT_DIR/build/vcam-mobile.apk"
adb -s "$SERIAL" shell am force-stop dev.vcam.mobile
adb -s "$SERIAL" shell run-as dev.vcam.mobile rm -rf files/models.staging files/models.previous
adb -s "$SERIAL" shell run-as dev.vcam.mobile mkdir -p files/models.staging

for file in det_10g_640.onnx w600k_r50_112.onnx inswapper_128_mobile.onnx inswapper_128_mobile_fp16.onnx inswapper_emap.f32 qnn_probe.onnx manifest.json; do
    echo "Installing offline model $file"
    temporary="/data/local/tmp/dev.vcam.mobile-$file"
    adb -s "$SERIAL" push "$SCRIPT_DIR/model-pack/$file" "$temporary" >/dev/null
    adb -s "$SERIAL" shell run-as dev.vcam.mobile cp "$temporary" "files/models.staging/$file"
    adb -s "$SERIAL" shell rm -f "$temporary"
    temporary=
done

if adb -s "$SERIAL" shell run-as dev.vcam.mobile test -d files/models; then
    adb -s "$SERIAL" shell run-as dev.vcam.mobile mv files/models files/models.previous
fi
if ! adb -s "$SERIAL" shell run-as dev.vcam.mobile mv files/models.staging files/models; then
    adb -s "$SERIAL" shell run-as dev.vcam.mobile mv files/models.previous files/models || true
    echo "Could not publish the staged model pack" >&2
    exit 1
fi
adb -s "$SERIAL" shell run-as dev.vcam.mobile rm -rf files/models.previous
adb -s "$SERIAL" shell run-as dev.vcam.mobile chmod 700 files/models
adb -s "$SERIAL" shell run-as dev.vcam.mobile chmod 600 \
    files/models/manifest.json \
    files/models/det_10g_640.onnx \
    files/models/w600k_r50_112.onnx \
    files/models/inswapper_128_mobile.onnx \
    files/models/inswapper_128_mobile_fp16.onnx \
    files/models/inswapper_emap.f32 \
    files/models/qnn_probe.onnx

echo "Installed dev.vcam.mobile and its offline model pack on $SERIAL"
echo "The app never uses network access. If another app owns the camera, it waits without disturbing that owner."
