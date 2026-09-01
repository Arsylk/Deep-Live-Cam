#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SDK_ROOT=${ANDROID_SDK_ROOT:-/opt/android-sdk}
BUILD_TOOLS=${ANDROID_BUILD_TOOLS:-$(find "$SDK_ROOT/build-tools" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V | tail -n 1)}
PLATFORM=${ANDROID_PLATFORM:-$(find "$SDK_ROOT/platforms" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V | tail -n 1)}
TOOLS="$SDK_ROOT/build-tools/$BUILD_TOOLS"
ANDROID_JAR="$SDK_ROOT/platforms/$PLATFORM/android.jar"
BUILD_DIR="$SCRIPT_DIR/build"
CLASSES_DIR="$BUILD_DIR/classes"
CLASSES_JAR="$BUILD_DIR/classes.jar"
DEX_DIR="$BUILD_DIR/dex"
UNSIGNED_APK="$BUILD_DIR/vcam-camera2-unsigned.apk"
ALIGNED_APK="$BUILD_DIR/vcam-camera2-aligned.apk"
OUTPUT_APK="$BUILD_DIR/vcam-camera2-test.apk"
KEYSTORE=${ANDROID_DEBUG_KEYSTORE:-/home/arsylk/.android/debug.keystore}

rm -rf -- "$BUILD_DIR"
mkdir -p "$CLASSES_DIR" "$DEX_DIR"

javac --release 11 -encoding UTF-8 \
    -classpath "$ANDROID_JAR" \
    -d "$CLASSES_DIR" \
    "$SCRIPT_DIR/src/dev/vcam/test/MainActivity.java"
jar --create --file "$CLASSES_JAR" -C "$CLASSES_DIR" .
"$TOOLS/d8" --lib "$ANDROID_JAR" --output "$DEX_DIR" "$CLASSES_JAR"
"$TOOLS/aapt2" link \
    -I "$ANDROID_JAR" \
    --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
    --min-sdk-version 29 \
    --target-sdk-version 35 \
    --version-code 7 \
    --version-name 0.7 \
    -o "$UNSIGNED_APK"
(cd "$DEX_DIR" && zip -q -j "$UNSIGNED_APK" classes.dex)
"$TOOLS/zipalign" -f 4 "$UNSIGNED_APK" "$ALIGNED_APK"
"$TOOLS/apksigner" sign \
    --ks "$KEYSTORE" \
    --ks-pass pass:android \
    --key-pass pass:android \
    --out "$OUTPUT_APK" \
    "$ALIGNED_APK"
"$TOOLS/apksigner" verify --verbose "$OUTPUT_APK"
echo "$OUTPUT_APK"
