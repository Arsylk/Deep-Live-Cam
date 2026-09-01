#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SDK_ROOT=${ANDROID_SDK_ROOT:-/opt/android-sdk}
BUILD_TOOLS=${ANDROID_BUILD_TOOLS:-$(find "$SDK_ROOT/build-tools" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V | tail -n 1)}
PLATFORM=${ANDROID_PLATFORM:-$(find "$SDK_ROOT/platforms" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V | tail -n 1)}
TOOLS="$SDK_ROOT/build-tools/$BUILD_TOOLS"
ANDROID_JAR="$SDK_ROOT/platforms/$PLATFORM/android.jar"
BUILD="$SCRIPT_DIR/build"
CLASSES="$BUILD/classes"
DEX="$BUILD/dex"
LIB="$BUILD/apk/lib/arm64-v8a"
ASSETS="$BUILD/apk/assets/licenses"
RES_COMPILED="$BUILD/resources.zip"
UNSIGNED="$BUILD/vcam-mobile-unsigned.apk"
ALIGNED="$BUILD/vcam-mobile-aligned.apk"
OUTPUT="$BUILD/vcam-mobile.apk"
if [[ -n ${ANDROID_DEBUG_KEYSTORE:-} ]]; then
    KEYSTORE=$ANDROID_DEBUG_KEYSTORE
else
    CURRENT_USER_HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
    KEYSTORE="$CURRENT_USER_HOME/.android/debug.keystore"
fi
ORT="$SCRIPT_DIR/third_party/onnxruntime-android-qnn-1.29.0.aar"
QNN="$SCRIPT_DIR/third_party/qnn-runtime-2.42.0.aar"
OPENCV="$SCRIPT_DIR/third_party/opencv-4.12.0.aar"

for dependency in "$ORT" "$QNN" "$OPENCV"; do
    [[ -f "$dependency" ]] || { echo "Missing $dependency; run ./fetch-dependencies.sh" >&2; exit 1; }
done

if [[ ! -f "$KEYSTORE" ]]; then
    mkdir -p "$(dirname -- "$KEYSTORE")"
    keytool -genkeypair -noprompt -keystore "$KEYSTORE" -storepass android \
        -alias androiddebugkey -keypass android -dname 'CN=Android Debug,O=Android,C=US' \
        -keyalg RSA -keysize 2048 -validity 10000 >/dev/null
fi

rm -rf -- "$BUILD"
mkdir -p "$CLASSES" "$DEX" "$LIB" "$ASSETS" "$BUILD/jars"
unzip -qp "$ORT" classes.jar >"$BUILD/jars/ort.jar"
unzip -qp "$OPENCV" classes.jar >"$BUILD/jars/opencv.jar"

javac --release 11 -encoding UTF-8 \
    -classpath "$ANDROID_JAR:$BUILD/jars/ort.jar:$BUILD/jars/opencv.jar" \
    -d "$CLASSES" \
    $(find "$SCRIPT_DIR/src" -name '*.java' -print | sort)
jar --create --file "$BUILD/jars/app.jar" -C "$CLASSES" .
"$TOOLS/d8" --lib "$ANDROID_JAR" --output "$DEX" \
    "$BUILD/jars/app.jar" "$BUILD/jars/ort.jar" "$BUILD/jars/opencv.jar"

"$TOOLS/aapt2" compile --dir "$SCRIPT_DIR/res" -o "$RES_COMPILED"
"$TOOLS/aapt2" link \
    -I "$ANDROID_JAR" \
    --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
    -R "$RES_COMPILED" \
    --auto-add-overlay \
    --min-sdk-version 29 \
    --target-sdk-version 35 \
    --version-code 1 \
    --version-name 0.1.0 \
    -o "$UNSIGNED"

for specification in \
    "$ORT:jni/arm64-v8a/libonnxruntime.so" \
    "$ORT:jni/arm64-v8a/libonnxruntime4j_jni.so" \
    "$QNN:jni/arm64-v8a/libQnnGpu.so" \
    "$QNN:jni/arm64-v8a/libQnnSystem.so" \
    "$OPENCV:jni/arm64-v8a/libopencv_java4.so" \
    "$OPENCV:jni/arm64-v8a/libc++_shared.so"; do
    archive=${specification%%:*}; member=${specification#*:}
    unzip -qp "$archive" "$member" >"$LIB/${member##*/}"
done
unzip -qp "$QNN" NOTICE.txt >"$ASSETS/QUALCOMM-NOTICE.txt"
unzip -qp "$QNN" LICENSE.pdf >"$ASSETS/QUALCOMM-LICENSE.pdf"

(cd "$DEX" && zip -q -j "$UNSIGNED" classes*.dex)
(cd "$BUILD/apk" && zip -q -r "$UNSIGNED" lib assets)
"$TOOLS/zipalign" -f -P 16 4 "$UNSIGNED" "$ALIGNED"
"$TOOLS/apksigner" sign \
    --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
    --out "$OUTPUT" "$ALIGNED"
"$TOOLS/apksigner" verify --verbose "$OUTPUT" >/dev/null
echo "$OUTPUT"
