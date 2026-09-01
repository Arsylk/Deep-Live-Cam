# RX 570 ncnn/Vulkan swapper

This backend offloads the selected face-swap generator to ncnn over Mesa RADV.
Face detection, recognition, alignment, tracking, and paste-back remain on the
normal application path. Runtime use is entirely local and performs no
downloads. It supports both legacy INSwapper and the split native-256
conditioner/generator bundle.

## Local preparation

Install the distribution's `ncnn` package, then build the small C ABI bridge:

```bash
./arch-linux/ncnn/build.sh
```

Obtain `pnnx` separately, then convert the already-local ONNX model:

```bash
.venv/bin/python tools/prepare_ncnn_swapper.py --pnnx /path/to/pnnx
```

The conversion script retains only the runtime files under `models/ncnn/`.
It also repairs twelve style-vector reshapes emitted with a redundant batch
axis by pnnx 20260704. Without that correction, ncnn's first channel slice
attempts an invalid allocation and Vulkan reports out-of-device-memory.

Use the backend explicitly with:

```bash
.venv/bin/python run.py --execution-provider cpu --swapper-backend ncnn
```

`--swapper-backend auto` is the default. It selects ncnn when all local assets
exist and otherwise leaves the existing ONNX Runtime path unchanged. Explicit
`ncnn` mode fails clearly instead of silently benchmarking a CPU fallback.
For native-256, automatic model selection additionally enforces the qualified,
auto-eligible, hash-pinned report described below.

## Native-256 bundle

The native model uses a separate, hash-pinned bundle. Follow the export and
offline packaging commands in [../../native256/README.md](../../native256/README.md),
then point the application at its manifest:

```bash
DLC_NATIVE256_MANIFEST=/data/swap256m/ncnn/manifest.json \
python run.py --swapper-model native-256 --swapper-backend ncnn
```

Its identity conditioner runs only when the mapped source changes; the cached
2016-channel style is then passed to the per-frame 256×256 generator. Candidate
RGB and learned semantic alpha are returned separately. Explicit ncnn selection
is fail-closed and never changes models or falls back to ORT.

The development smoke graph was executed through the application adapter on
the local RX 570 at 11.86 ms mean / 15.35 ms p95 over 100 warmed calls, including
Python preprocessing, C-ABI transfer, cached-style generator inference and
output conversion. This is a runtime measurement, not a quality claim: the
repository still has no quality-trained native-256 checkpoint.

## Validated Polaris settings

The RX 570 exposes FP16 storage but not FP16 arithmetic. FP16 storage is still
disabled for this particular network because its variance calculation squares
large activations and overflows FP16 intermediates. Full FP32 produced practical
exact parity with ONNX Runtime and remains comfortably faster.

Measured on the local RX 570 using 100 warmed runs:

- ncnn Vulkan FP32: 158.5 ms median, 160.2 ms p95
- ONNX Runtime CPU: 458.6 ms median, 477.5 ms p95
- speedup: 2.89x for swap inference
- additional VRAM: approximately 3.0 GiB (4.0 GiB total with the desktop)
- raw tensor parity: MAE 5.4e-7, maximum error 6.0e-6
- real aligned crop: 99.9817% of pixels exactly equal, maximum difference 1

The standalone raw-tensor diagnostic is
`tools/ncnn_swapper_benchmark.cpp`.
