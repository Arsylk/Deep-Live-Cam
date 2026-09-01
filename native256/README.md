# DLC-Swap256-M

This directory contains an isolated training and export scaffold for a native
256×256 face-swap student. It does not change the application's dependencies or
select itself as a runtime backend. Until a checkpoint passes the quality and
runtime gates below, `inswapper_128` remains the production model.

The design is a fixed-shape depthwise-separable U-Net with a learned semantic
fusion mask. Its source contract deliberately remains the existing mapped and
L2-normalized 512-value `buff2fs` latent. A small identity conditioner runs once
when the source changes; the per-frame graph receives its cached 2016-channel
style packet. The singleton spatial axes are part of the deployment contract,
not an implementation detail: they make channel broadcasting unambiguous to
ncnn/pnnx.

## Model contract

Training graph:

```text
target   float32 [B,3,256,256], RGB [0,1]
identity float32 [B,512], mapped and L2-normalized
    -> candidate float32 [B,3,256,256], RGB [0,1]
    -> alpha     float32 [B,1,256,256], [0,1]
```

Deployment graphs:

```text
conditioner: mapped_identity [1,512,1,1] -> style [1,2016,1,1]
generator:   target [1,3,256,256] + style [1,2016,1,1]
             -> candidate [1,3,256,256] + alpha [1,1,256,256]
```

Composite with `alpha*candidate + (1-alpha)*target`. The alpha prediction is
multiplied by a static smooth safety prior whose crop corners are exactly zero.
The mask decoder fuses lightweight projections of all four target-encoder skip
scales, so its boundary can follow eyes, mouth, hair and occlusions rather than
guessing them from the 16×16 bottleneck alone. Foreground and background mask
supervision prevent the area regularizer from learning an all-zero shortcut.
The mapped identity remains the same normalized 512-value vector; deployment
only views it as a singleton NCHW tensor. The exporter represents the trained
Linear conditioner as mathematically identical 1×1 convolutions so conversion
does not introduce an ambiguous vector-to-image reshape.

The fixed medium configuration has approximately 0.571 million trainable
values including BatchNorm state/affines and 0.971 GMAC of per-frame work. The
conditioner is included in the parameter count but excluded from per-frame MACs
because its result is cached.

## Data manifest

Training data is never downloaded. The trainer consumes JSON Lines and rejects
absolute paths, traversal outside `--data-root`, identities shared between
train/validation/test splits, undeclared provenance, or samples that do not
explicitly set `usage_authorized` to `true`.

Each cross-identity row must contain a cached float32 128×128 teacher candidate
and its exact float32 128×128 alpha as NumPy arrays. The contract is deliberately
unambiguous: candidate is NCHW aligned RGB before paste-back, alpha is `[0,1]`, and neither has
later application post-processing. The objective composites the pair over an
area-downsampled copy of the 256 target itself, so teacher background and mask
semantics cannot be accidentally baked into an opaque RGB artifact.

```json
{"sample_id":"train-000001","split":"train","target":"crops/t001.png","target_identity":"target-17","source_identity":"source-03","source_latent":"latents/source-03-mapped.npy","identity_type":"inswapper_mapped_512","identity_map_sha256":"cee626bc81721d71c5d6cb1f76f830b9ae46f595514b0884dd8ae34785576764","source_embedding":"latents/source-03-arcface.npy","teacher_candidate":"teacher/train-000001-candidate.npy","teacher_alpha":"teacher/train-000001-alpha.npy","teacher_type":"uncomposited-candidate-alpha-rgb-128-v1","face_mask":"masks/t001.png","teacher_confidence":0.94,"provenance":"consented internal capture set 2026-08","license":"project training grant v1","usage_authorized":true}
```

For a same-identity reconstruction row, `teacher_candidate`, `teacher_alpha`,
and `teacher_type` may all be null. The loader rejects an incomplete teacher
pair or a pair without the exact contract marker. Latent files are
512-value NumPy arrays. `source_latent` is the mapped model input and every row
must name its identity type and exact map SHA-256; the trainer rejects a hash
that differs from `--identity-map`. `source_embedding` is the raw normalized
ArcFace target used by the differentiable identity loss. Target crops and masks
are 256×256. Teacher crops are 128×128.

Generate each teacher pair from the exact local INSwapper file, aligned target,
mapped latent, and face mask with the offline preparation tool. Supplying the
expected model hash is mandatory; existing outputs are never overwritten.

```bash
python tools/prepare_native256_teacher.py \
  --model models/inswapper_128.onnx \
  --model-sha256 <audited-sha256> \
  --target /data/faces/crops/t001.png \
  --source-latent /data/faces/latents/source-03-mapped.npy \
  --face-mask /data/faces/masks/t001.png \
  --output-dir /data/faces/teacher \
  --sample-id train-000001
```

The resulting sidecar records hashes for the model, every input, and both
float32 outputs. Reference the two `.npy` files from the training manifest.

Temporal metadata is optional but all three fields must appear together:
`next_target`, `flow`, and `flow_valid`. Flow uses pixel displacement in a
`[2,256,256]` NumPy array; validity is `[1,256,256]`. The reusable temporal loss
operates on swap residuals rather than raw output frames so that it does not
reward frozen expressions. Flow points from `target` to `next_target`. Both
temporal weights default to zero; enabling either weight requires at least one
temporal tuple and causes the trainer to execute the paired-frame objective.

## Environment and smoke training

Use a separate training environment; do not install PyTorch into the application
environment merely to run Deep-Live-Cam.

```bash
python -m pip install -r native256/requirements-training.txt
python -m native256.train \
  --manifest /data/faces/train.jsonl \
  --data-root /data/faces \
  --identity-map models/ncnn/inswapper_128_emap.npy \
  --output /data/runs/swap256m
```

The configured identity loss is intentionally fail-closed. Set
`training.identity_model` to a differentiable TorchScript ArcFace model matching
the stored raw embeddings. `--allow-missing-identity-loss` exists only for a
one- or two-step architecture smoke test; such a run cannot produce a valid
face-swap checkpoint.

The current repository supplies neither an authorized training corpus nor a
differentiable ArcFace training checkpoint. The existing ONNX recognizer can
score teacher samples but cannot backpropagate the student identity objective.

## Export

```bash
python -m native256.export_onnx \
  --checkpoint /data/runs/swap256m/checkpoint-00300000.pt \
  --identity-map models/ncnn/inswapper_128_emap.npy \
  --output /data/runs/swap256m/onnx \
  --combined
```

Training binds the checkpoint to the exact 512×512 `buff2fs` `.npy` SHA-256.
Export refuses a different map, copies it into the output bundle as
`identity_map.npy`, and records the contract as
`inswapper-buff2fs-mapped-l2-v1`; raw ArcFace embeddings are not valid model
inputs.

Export is opset 17, batch one, and has no dynamic axes. `manifest.json` is the
runtime-consumable schema-v1 manifest used by `modules.swapper_contract`; the
operator set, parity figures and architecture measurements live separately in
`audit.json`. The exporter checks the
ONNX graph, rejects operators outside a small ncnn/QNN-oriented allowlist,
checks ONNX Runtime parity when available, and emits SHA-256 fingerprints. The
split conditioner/generator files are the intended deployment assets; the
combined graph is a convenience for evaluation. Every exported manifest has
`quality_status: development` and `auto_select_eligible: false`. Exporting a
graph is not qualification; a later, separate quality-gate process must create
an eligible release manifest.

### Offline ncnn/Vulkan package

Build the local C ABI bridge, then convert an exported development bundle with
an explicitly supplied local `pnnx` executable:

```bash
./arch-linux/ncnn/build.sh
python tools/prepare_native256_ncnn.py \
  --manifest /data/runs/swap256m/onnx/manifest.json \
  --pnnx /opt/pnnx/pnnx \
  --output-dir /data/runs/swap256m/ncnn \
  --fp16-storage
```

The packager performs no downloads. It verifies the ONNX source hashes, rejects
external ONNX tensor files and custom or ambiguous ncnn layers, converts both
split graphs in a staging directory, and pins the four resulting ncnn files in
the published manifest. An existing output is never replaced. Convert before
qualification: adding a deployable backend invalidates a prior automatic-
selection qualification, so the tool rejects an already auto-eligible release.

Exercise a local development bundle explicitly with:

```bash
DLC_NATIVE256_MANIFEST=/data/runs/swap256m/ncnn/manifest.json \
python run.py --swapper-model native-256 --swapper-backend ncnn
```

The bridge caches the 2016-channel conditioned source style and returns the
generator candidate and semantic alpha to the same application paste path as
the ONNX adapter. Explicit `ncnn` selection fails closed if any asset, hash,
graph contract, Vulkan device, or output tensor is invalid; it never substitutes
the ORT or legacy model. Backend `auto` may fall back from ncnn to the exact same
native-256 ONNX bundle.

## Loss curriculum

The scaffold implements low-resolution teacher Charbonnier distillation,
same-identity pixel/gradient reconstruction, differentiable ArcFace identity,
known-background preservation, weak semantic mask penalties, PatchGAN hinge
helpers, and optical-flow residual/mask temporal losses. A practical curriculum
is:

1. 25k same-identity reconstruction/mask warm-up steps.
2. 150k steps with a 70/30 cross/self mixture, teacher confidence and identity.
3. 75k hard-pair/adversarial refinement steps.
4. 50k real and synthetic temporal fine-tuning steps.

The entry point trains the deterministic still-image core and, when explicitly
enabled, the paired residual/mask temporal objective. The PatchGAN helper stays
separate because adversarial refinement requires a deliberately balanced real
crop loader and discriminator schedule; silently treating ordinary still rows
as that corpus would train the wrong objective.

## Release gates

A trained model should not enter an application model pack until all of these
are measured on identity-disjoint data:

- source identity similarity no worse than 0.02 cosine below the teacher;
- target landmark/pose error within 5% of the teacher;
- background error and temporal residual flicker at least 20% below the current
  fixed-mask pipeline;
- ncnn/ONNX output difference no more than one 8-bit level;
- strict Android QNN graph admission with CPU fallback disabled;
- warmed swap latency below 20 ms on the RX 570 and below 40 ms on the target
  phone, treated as qualification targets rather than assumed results;
- blind human review prefers or rates the student equal to the teacher on at
  least 60% of hard samples.

Automatic selection additionally requires a local `qualification_report` entry
in `manifest.json`, containing a relative `.json` file and its SHA-256. The
report must identify the same model, bind every deployed ONNX, identity-map and
(when present) ncnn artifact hash, declare a qualified verdict, and mark every
release gate above as passed. `quality_status` and `auto_select_eligible` strings
without that hash-pinned evidence are rejected. Development bundles remain
available only through explicit model selection.

The report schema is intentionally small and auditable:

```json
{
  "schema_version": 1,
  "model_id": "dlc_swap256m",
  "verdict": "qualified",
  "artifacts": {
    "identity_map_sha256": "<64 hex>",
    "identity_conditioner_sha256": "<64 hex>",
    "swapper_sha256": "<64 hex>",
    "ncnn_identity_conditioner_param_sha256": "<64 hex>",
    "ncnn_identity_conditioner_bin_sha256": "<64 hex>",
    "ncnn_swapper_param_sha256": "<64 hex>",
    "ncnn_swapper_bin_sha256": "<64 hex>"
  },
  "gates": {
    "identity_similarity": true,
    "pose_landmarks": true,
    "background_temporal": true,
    "ncnn_parity": true,
    "android_qnn_admission": true,
    "rx570_latency": true,
    "target_phone_latency": true,
    "blind_review": true
  }
}
```

Omit the four `ncnn_*` artifact entries only when the manifest contains no ncnn
representation. The report filename and SHA-256 are then placed under the
manifest's `qualification_report` object.

The runtime program remains completely offline. Training assets and any model
conversion are host-side preparation steps, and their licenses must be audited
before distributing a distilled checkpoint.
