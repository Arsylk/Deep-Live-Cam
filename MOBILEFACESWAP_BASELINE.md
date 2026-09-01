# Experimental MobileFaceSwap baseline

This repository includes an offline preparation tool for evaluating the public
MobileFaceSwap checkpoint as a source-conditioned comparison baseline. It does
not install a backend, change application settings, register a model, or make
MobileFaceSwap the default.

The important limitation is that the official public archive contains only
`MobileFaceSwap_224.pdparams`. There is no released native-256 checkpoint. The
tool permits a 256×256 evaluation because the conditioned U-Net is fully
convolutional, but that remains the 224-trained model evaluated at a larger
size and has shown eye/occlusion artifacts in initial testing.

## Offline inputs

Supply every input as a local path:

- The official `checkpoints.tar`, with SHA-256
  `8e45272de4cc55d64294d282bb7605343a406696a05209203065aa1b51aed58e`.
- An official MobileFaceSwap checkout at commit
  `29552474c2f621c27818ff7fc32447e1a60a96fb`.
- A source image containing one clearly visible, authorized face.
- A cache/output directory.
- Optionally, a local pnnx executable. Without it, the tool stops after ONNX
  export.

The script has no downloader and never invokes a package installer. Its
InsightFace detector is loaded by explicit path from the verified local
archive. Archive members are checked before extraction; absolute paths,
traversal paths, links, devices, and missing or incorrectly sized assets are
rejected.

The audited conversion environment is Python 3.11 with:

```text
paddlepaddle==2.6.2
paddle2onnx==1.3.1
onnx==1.17.0
numpy==1.26.4
insightface==0.7.3
opencv-python-headless==4.11.0.86
```

Paddle2ONNX 2.1 is not a substitute in this environment: it requires a newer
Paddle generation and is intentionally rejected by the tool.

## Preparation

ONNX-only example:

```bash
python tools/prepare_mobilefaceswap_baseline.py \
  --checkpoint-tar /local/assets/checkpoints.tar \
  --upstream-dir /local/src/MobileFaceSwap \
  --source-image /local/pictures/authorized-source.jpg \
  --output-dir /local/cache/deep-live-cam/mobilefaceswap \
  --size 224
```

Add an ncnn conversion explicitly:

```bash
python tools/prepare_mobilefaceswap_baseline.py \
  --checkpoint-tar /local/assets/checkpoints.tar \
  --upstream-dir /local/src/MobileFaceSwap \
  --source-image /local/pictures/authorized-source.jpg \
  --output-dir /local/cache/deep-live-cam/mobilefaceswap \
  --size 224 \
  --pnnx /local/tools/pnnx
```

The cache entry is `<first-16-source-sha256>-<size>/`, allowing multiple
source pictures to coexist. It contains the conditioned ONNX model, optional
ncnn param/bin files, and `manifest.json`. The manifest records full source
and model hashes, the audited upstream revision, the checkpoint hash, tensor
contract, training-size warning, licensing warning, and
`quality_status: experimental`.

## Runtime tensor contract

The conditioned per-frame graph is tied to exactly one source identity:

```text
target_rgb   float32 [1, 3, H, H], RGB, range 0..1
    -> candidate_rgb float32 [1, 3, H, H], RGB, range 0..1
    -> alpha         float32 [1, 1, H, H], range 0..1
```

`candidate_rgb` is deliberately uncomposited. The runtime caller may evaluate
the learned alpha independently and, if accepted, composite it as:

```text
candidate_rgb * alpha + target_rgb * (1 - alpha)
```

Selecting another source image requires loading that source's cache entry or
preparing a new one. The public model does not accept the existing INSwapper
512-dimensional embedding directly: its one-time conditioner also requires a
model-specific ArcFace 7×7 feature map.

## Status and licensing

This is for controlled quality and performance comparisons only. It must not
be auto-selected or used as a fallback, and the preparation tool contains no
integration that could do so. Compare identity retention, eyes, occlusions,
profiles, expression, mask seams, and temporal stability against the existing
backend before considering further integration.

The upstream source repository is Apache-2.0. The separately hosted checkpoint
archive does not include an explicit model-weight license and bundles
InsightFace-derived assets. Do not infer commercial redistribution rights for
the weights from the source-code license; obtain author clearance or train
owned weights before redistribution.
