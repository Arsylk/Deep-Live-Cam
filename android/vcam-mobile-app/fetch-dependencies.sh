#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEST="$SCRIPT_DIR/third_party"
mkdir -p "$DEST"

fetch() {
    local name=$1 url=$2 expected=$3 target="$DEST/$name"
    if [[ ! -f "$target" ]] || [[ "$(sha256sum "$target" | awk '{print $1}')" != "$expected" ]]; then
        curl -fL --retry 3 --max-time 300 "$url" -o "$target.tmp"
        printf '%s  %s\n' "$expected" "$target.tmp" | sha256sum -c -
        mv -f -- "$target.tmp" "$target"
    fi
}

fetch \
    onnxruntime-android-qnn-1.29.0.aar \
    https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android-qnn/1.29.0/onnxruntime-android-qnn-1.29.0.aar \
    cc4cfd72a47703aee65d8e6ce7448fa7488e3fff6556f9d6d354ec1687347cfd
fetch \
    qnn-runtime-2.42.0.aar \
    https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/2.42.0/qnn-runtime-2.42.0.aar \
    ecb69bd6a7b1be4717ef10805199fb49ec6b60ca3f481d3068ed9605fba91d5b
fetch \
    opencv-4.12.0.aar \
    https://repo1.maven.org/maven2/org/opencv/opencv/4.12.0/opencv-4.12.0.aar \
    f71846a313388d9da667a59be1e921669fcf792d1ba264b3898253ca789bc3f0

echo "Android dependencies are ready in $DEST"
