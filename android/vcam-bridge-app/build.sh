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
UNSIGNED_APK="$BUILD_DIR/vcam-app-unsigned.apk"
ALIGNED_APK="$BUILD_DIR/vcam-app-aligned.apk"
OUTPUT_APK="$BUILD_DIR/vcam-app.apk"
KEYSTORE=${ANDROID_DEBUG_KEYSTORE:-/home/arsylk/.android/debug.keystore}

rm -rf -- "$BUILD_DIR"
mkdir -p "$CLASSES_DIR" "$DEX_DIR"

# Compile Kotlin sources (need kotlin-compiler embeddable)
KOTLINC_JAR=$(find /usr/share/kotlin-compiler* /opt -name "kotlin-compiler-embeddable*.jar" 2>/dev/null | head -1)
if [ -z "$KOTLINC_JAR" ]; then
    KOTLINC_JAR=$(find /usr/share -name "kotlin-compiler*.jar" 2>/dev/null | head -1)
fi
if [ -z "$KOTLINC_JAR" ]; then
    echo "ERROR: kotlinc not found. Install kotlin-compiler-embeddable."
    exit 1
fi

java -cp "$KOTLINC_JAR" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
    -classpath "$ANDROID_JAR" \
    -d "$CLASSES_DIR" \
    -no-stdlib -no-reflect \
    "$SCRIPT_DIR/src/dev/vcam/bridge/CameraBridgeService.kt" \
    "$SCRIPT_DIR/src/dev/vcam/bridge/GlCameraRenderer.kt" \
    "$SCRIPT_DIR/src/dev/vcam/bridge/MainActivity.kt" \
    "$SCRIPT_DIR/src/dev/vcam/bridge/PreviewActivity.kt" \
    "$SCRIPT_DIR/src/dev/vcam/bridge/CameraHooks.kt"

# Jar only the app classes (exclude any stubs if present)
jar --create --file "$CLASSES_JAR" -C "$CLASSES_DIR" dev/vcam/bridge

# Dex
"$TOOLS/d8" --lib "$ANDROID_JAR" --output "$DEX_DIR" "$CLASSES_JAR"

# Link APK
"$TOOLS/aapt2" link \
    -I "$ANDROID_JAR" \
    --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
    --min-sdk-version 29 \
    --target-sdk-version 36 \
    --version-code 1 \
    --version-name "1.0" \
    -o "$UNSIGNED_APK"

# Add classes.dex and assets
(cd "$DEX_DIR" && zip -q -j "$UNSIGNED_APK" classes.dex)
if [ -d "$SCRIPT_DIR/assets" ] && [ "$(ls -A "$SCRIPT_DIR/assets")" ]; then
    (cd "$SCRIPT_DIR" && zip -q -r "$UNSIGNED_APK" assets/)
fi

# Align and sign
"$TOOLS/zipalign" -f 4 "$UNSIGNED_APK" "$ALIGNED_APK"
"$TOOLS/apksigner" sign \
    --ks "$KEYSTORE" \
    --ks-pass pass:android \
    --key-pass pass:android \
    --out "$OUTPUT_APK" \
    "$ALIGNED_APK"
"$TOOLS/apksigner" verify --verbose "$OUTPUT_APK"
echo "$OUTPUT_APK"
