#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! pkg-config --exists ncnn; then
    echo "ncnn development files are required (Arch/CachyOS package: ncnn)" >&2
    exit 1
fi

c++ -O3 -std=c++17 -fPIC -fvisibility=hidden -fopenmp \
    -Wall -Wextra -Wpedantic \
    $(pkg-config --cflags ncnn) \
    -shared "$script_dir/deep_live_cam_ncnn.cpp" \
    $(pkg-config --libs ncnn) \
    -Wl,-z,defs \
    -o "$script_dir/libdeep_live_cam_ncnn.so"

echo "built $script_dir/libdeep_live_cam_ncnn.so"
