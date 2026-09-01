#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TEST_BUILD=$(mktemp -d "${TMPDIR:-/tmp}/vcam-routing-test.XXXXXX")
trap 'rm -rf -- "$TEST_BUILD"' EXIT

javac --release 11 -encoding UTF-8 \
    -d "$TEST_BUILD" \
    "$SCRIPT_DIR/src/dev/vcam/camlog/CameraRoutingPolicy.java" \
    "$SCRIPT_DIR/tests/dev/vcam/camlog/CameraRoutingPolicyTest.java"

java -cp "$TEST_BUILD" dev.vcam.camlog.CameraRoutingPolicyTest
