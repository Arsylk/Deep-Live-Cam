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
MODULE_CLASSES_JAR="$BUILD_DIR/module-classes.jar"
XPOSED_STUBS_JAR="$BUILD_DIR/xposed-stubs.jar"
DEX_DIR="$BUILD_DIR/dex"
FLAT_DIR="$BUILD_DIR/flat"
UNSIGNED_APK="$BUILD_DIR/camlog-unsigned.apk"
ALIGNED_APK="$BUILD_DIR/camlog-aligned.apk"
OUTPUT_APK="$BUILD_DIR/vcam-camera-logger.apk"
KEYSTORE=${ANDROID_DEBUG_KEYSTORE:-/home/arsylk/.android/debug.keystore}

rm -rf -- "$BUILD_DIR"
mkdir -p "$CLASSES_DIR" "$DEX_DIR" "$FLAT_DIR"

# 1. Compile resources (xposed_scope array referenced by the manifest)
"$TOOLS/aapt2" compile "$SCRIPT_DIR/res/values/arrays.xml" -o "$FLAT_DIR"

# 2. Link manifest + resources
"$TOOLS/aapt2" link \
    -I "$ANDROID_JAR" \
    --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
    --min-sdk-version 29 \
    --target-sdk-version 35 \
    --version-code 11 \
    --version-name 0.4.2 \
    -o "$UNSIGNED_APK" \
    "$FLAT_DIR"/*.flat

# 3. Compile module sources against android.jar + compile-time Xposed stubs.
#    The stubs are NOT packaged; Vector provides the real classes at runtime.
javac --release 11 -encoding UTF-8 \
    -classpath "$ANDROID_JAR" \
    -d "$CLASSES_DIR" \
    "$SCRIPT_DIR"/stubs/de/robv/android/xposed/*.java \
    "$SCRIPT_DIR"/stubs/de/robv/android/xposed/callbacks/*.java \
    "$SCRIPT_DIR"/src/dev/vcam/camlog/CameraRoutingPolicy.java \
    "$SCRIPT_DIR"/src/dev/vcam/camlog/CameraLoggerHooks.java

# 4. Jar ONLY the module classes (exclude stubs)
(cd "$CLASSES_DIR" && jar --create --file "$MODULE_CLASSES_JAR" dev/vcam/camlog)
(cd "$CLASSES_DIR" && jar --create --file "$XPOSED_STUBS_JAR" de/robv/android/xposed)

# 5. Dex the module classes
"$TOOLS/d8" --lib "$ANDROID_JAR" --classpath "$XPOSED_STUBS_JAR" \
    --output "$DEX_DIR" "$MODULE_CLASSES_JAR"

# 6. Add classes.dex and assets/xposed_init into the APK
(cd "$DEX_DIR" && zip -q -j "$UNSIGNED_APK" classes.dex)
(cd "$SCRIPT_DIR" && zip -q -r "$UNSIGNED_APK" assets/xposed_init)

# 7. Align + sign
"$TOOLS/zipalign" -f 4 "$UNSIGNED_APK" "$ALIGNED_APK"
"$TOOLS/apksigner" sign \
    --ks "$KEYSTORE" \
    --ks-pass pass:android \
    --key-pass pass:android \
    --out "$OUTPUT_APK" \
    "$ALIGNED_APK"
"$TOOLS/apksigner" verify --verbose "$OUTPUT_APK"
echo "$OUTPUT_APK"
