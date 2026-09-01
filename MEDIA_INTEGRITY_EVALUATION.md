# Media quality, forensic exposure, and provenance plan

## What “best possible rating” can defensibly mean

[Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857) evaluates **detectors** on
in-the-wild media. It does not award a naturalness or quality score to a face-swap
generator. Its central result is domain shift: detector performance falls sharply
between older academic sets and contemporary media collected from the web. A clip
that a detector labels authentic is therefore not proved authentic, high quality,
or safe.

This project uses three independent scorecards:

1. **Media quality and temporal stability** — VMAF, SSIM, PSNR, cadence,
   repeated-frame, clipping, detail, and blockiness measurements.
2. **Forensic exposure** — detector AUC, average precision, EER, NIST-style
   CDR at FAR 5%, Brier score, and calibration error. These evaluate detector
   behavior and never add points to the product grade.

The application must not be tuned to reduce manipulated-class detector scores.
That would turn a defensive benchmark into an anti-forensics optimizer. Product
settings may be changed using development-set fidelity, stability, latency, and
blinded human ratings; detector results remain a final, read-only audit.

Everything in the implemented workflow runs locally. It neither uploads private
footage nor downloads models. Detector frameworks may be run separately on a
machine where their official weights are already installed, then exported as the
small JSONL format described below.

## Research basis

- **In-the-wild generalization.** Deepfake-Eval-2024 contains multilingual,
  contemporary video, audio, and images collected from many websites and reports
  large AUC degradation for off-the-shelf detectors. The implication is to use a
  frozen, recent, device-diverse holdout and never claim success from one detector
  or legacy set alone: [paper](https://arxiv.org/abs/2503.02857).
- **Reproducible detector comparison.** DeepfakeBench standardizes preprocessing,
  datasets, implementations, and reports AUC together with AP and EER across a
  broad detector set: [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf).
- **Manipulation and domain coverage.** DF40 includes 40 manipulation methods,
  including InSwap, and separates same-domain, cross-forgery, cross-domain, and
  one-versus-all protocols. Its results demonstrate why method-matched performance
  is not a generalization claim: [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/34239f60eca7ce9bee5280aaf81362d8-Paper-Datasets_and_Benchmarks_Track.pdf).
- **Established forensic operating points.** NIST OpenMFC uses ROC-AUC and Correct
  Detection Rate at a fixed False Alarm Rate; localization tasks additionally use
  MCC. This suite implements ROC-AUC and CDR@FAR 5% for clip-level detector
  exports: [NIST challenge](https://mfc.nist.gov/) and
  [evaluation-plan record](https://www.nist.gov/publications/open-media-forensics-challenge-2022-evaluation-plan).
- **Objective video fidelity.** Netflix VMAF fuses perceptual features and is
  designed to predict subjective video quality; its official implementation also
  exposes PSNR, SSIM, MS-SSIM, and related metrics. The local FFmpeg build has
  `libvmaf`, `ssim`, and `psnr`: [official VMAF repository](https://github.com/Netflix/vmaf).
- **Human ratings.** ITU-T P.910 defines subjective methods for multimedia video;
  P.918 covers dimension-based assessment. The included aggregator supports a
  blinded ACR-style 1–5 export, but protocol compliance also requires controlled
  presentation, randomization, participant handling, and preregistration:
  [P.910](https://www.itu.int/epublications/ar/publication/itu-t-p-910-2023-10-subjective-video-quality-assessment-methods-for-multimedia-applications) and
  [P.918](https://www.itu.int/dms_pubrec/itu-t/rec/p/T-REC-P.918-202001-I%21%21SUM-HTM-E.htm).
- **Authenticity.** C2PA provides signed provenance rather than a visual-quality
  score. The generated manifest uses `c2pa.edited` plus
  `compositeWithTrainedAlgorithmicMedia`:
  [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html),
  [official CLI documentation](https://github.com/contentauth/c2pa-rs/blob/main/cli/docs/usage.md), and
  [IPTC vocabulary entry](https://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia).

## Preregistered evaluation design

Use separate development and frozen-holdout captures. A settings change made after
viewing any holdout or detector result creates a new experiment and requires a new
holdout. Keep the old result; do not silently replace it.

The internal minimum coverage gate is intentionally larger than a demo:

| Factor | Minimum holdout coverage |
|---|---:|
| Paired raw/processed clips | 60 |
| Pseudonymous identities | 30 |
| Physical capture classes | 2 (Android phone and webcam) |
| Lighting bins | 3 (low, ordinary, bright/backlit) |
| Motion bins | 3 (still, speech/expression, head/body motion) |
| Face-size bins | 3 |
| Compression/network conditions | 2 |

Each clip should be 20–30 seconds and contain a synchronized raw camera recording
and the Windows-return recording. Store only a pseudonymous identity ID. Record:

- device and camera/lens selection;
- resolution, nominal FPS, codec, bitrate, and transport route;
- lighting, motion, face-size, occlusion, and compression bins;
- Windows settings JSON and application/model version;
- explicit capture-subject consent and authorization for the source identity;
- manual alignment offsets when the returned stream has transport latency.

The benchmark hashes both clips and canonicalized settings. A report is therefore
tied to exact media and a configuration rather than a mutable preset name.

### Detector protocol

Run at least one spatial baseline, one frequency/source-consistency detector, and
one temporal/video detector from maintained official implementations. DeepfakeBench
and DF40 are suitable reproducibility frameworks when installed locally. Keep each
framework's official preprocessing and weights unchanged.

Export one manipulated-class probability per **clip**, not one row per frame. Include
authentic raw clips and processed clips, preserve the official threshold if one is
published, and report every detector independently. Required JSONL fields are:

```json
{"detector":"name","detector_version":"commit-or-release","detector_source":"official repository","sample_id":"clip-001-raw","label":0,"score":0.08,"subgroups":{"device":"android","lighting":"low"}}
```

`label` is `0` for authentic and `1` for manipulated. `score` is always the
probability of the manipulated class. Do not invert scores to make a result look
better. The tool reports:

- ROC-AUC and average precision;
- equal error rate and its operating threshold;
- Correct Detection Rate at False Alarm Rate no greater than 5%;
- threshold confusion, balanced accuracy, precision, recall, and specificity;
- Brier score and equal-width expected calibration error;
- deterministic stratified 95% bootstrap intervals;
- every viable declared subgroup, with “not available” instead of fabricating a
  metric when a subgroup contains only one class.

### Objective video protocol

FFmpeg resets both timestamps, scales the processed clip to the reference
dimensions, and samples both at the reference cadence before VMAF/SSIM/PSNR. Set
the two offsets so the same physical moment is compared. The report also measures
each decoded stream independently at bounded resolution and checks timestamp
continuity with `ffprobe`.

Full-frame reference metrics have an important limitation: the face is intentionally
different. A high result can be dominated by an unchanged background, while a low
result can penalize the intended identity edit. Treat these metrics as transport
and preservation measurements, not facial realism. The blinded subjective panel is
the authority for naturalness; report face tracking/seam measurements from
`quality-history.jsonl` as supporting engineering diagnostics.

### Subjective protocol

Use randomized, blinded presentation with calibrated playback and no product or
settings labels. Include hidden reference and processed conditions. A useful P.918-
style dimension set is `overall`, `naturalness`, `temporal_stability`, `face_boundary`,
and `color_consistency`, each on the same preregistered 1–5 ACR scale. Collect at
least 24 valid raters per condition for the internal gate; use more when confidence
intervals remain broad.

The JSONL exporter requires `sample_id`, pseudonymous `rater_id`, `condition`
(`reference` or `processed`), a `ratings` object, and an optional
`attention_check_passed`. Failed attention checks are excluded and counted. The
tool reports mean opinion scores, normal-approximation 95% intervals, and paired
degradation scores. This aggregation does not by itself establish ITU compliance.

## Implemented workflow

### Live pipeline baseline

`tools/pipeline_baseline.py` adds a passive, synchronized capture layer around
the existing benchmark. It observes the pre-swap and post-swap arrays from the
same processor invocation and therefore opens no camera, SRT connection, or UDP
preview. It writes lossless paired FFV1 clips, a clean run-scoped quality history,
settings/model/source/code hashes, and an immutable run manifest. Raw spool data
is compressed only after the measurement window so encoder CPU does not bias
pipeline timing.

The current registered baseline and exact hashes are in
`evaluation/baselines/webcam-inswapper128.json`. Local media is intentionally
ignored by Git. `compare` requires the same frozen raw corpus before it can
declare a candidate better, non-inferior, or regressed; unrelated live takes are
reported as diagnostic-only.

Example inputs live in [`evaluation/`](evaluation/). Create real clip paths in a
copy of the study manifest; do not edit the example into a claim of measured
performance.

Run a single offline diagnostic pair:

```bash
uv run --python 3.12 python tools/media_integrity_benchmark.py pair \
  --reference captures/android-raw.mp4 \
  --processed captures/android-return.mp4 \
  --processed-offset 0.35 --duration 30 \
  --settings captures/windows-settings.json \
  --quality-history captures/quality-history.jsonl \
  --device android --intended-use live \
  --capture-subject-consent --source-identity-authorized \
  --output-dir benchmark-results
```

Run the preregistered study and optional detector/subjective exports:

```bash
uv run --python 3.12 python tools/media_integrity_benchmark.py study \
  --manifest evaluation/study-manifest.json \
  --output-dir benchmark-results/holdout
```

Evaluate a detector export by itself:

```bash
uv run --python 3.12 python tools/media_integrity_benchmark.py detectors \
  --input evaluation/detector-results.jsonl \
  --output benchmark-results/detectors.json
```

Every pair produces `report.json`, `report.md`, and an unsigned
`c2pa-manifest.json`. The unsigned file is only a template. It is not a Content
Credential and earns no provenance credit.

For an offline recorded release, install the official `c2patool` separately and
provide a real C2PA signing certificate and key. The command does not configure a
timestamp server and refuses to overwrite an existing output:

```bash
uv run --python 3.12 python tools/media_integrity_benchmark.py sign \
  --asset processed.mp4 --parent raw.mp4 --output processed-signed.mp4 \
  --certificate c2pa-chain.pem --private-key c2pa-private.pem
```

The key must not be committed. A development certificate proves only that the
mechanism works; it does not establish a trusted production identity.
Both signing and verification pass an explicit C2PA settings file with remote
manifest fetching disabled; an asset containing only a remote credential is
reported as unavailable rather than causing a network request.

## Safe improvement loop

1. Capture a development matrix and establish a baseline report.
2. Fix transport faults first: decoder errors, repeated frames, cadence gaps,
   resolution loss, clipping, and bitrate starvation.
3. Change one quality setting at a time. Compare VMAF/SSIM/PSNR, temporal metrics,
   existing face stability diagnostics, latency, and blinded development ratings.
4. Reject changes that improve one clip while regressing device or lighting
   subgroups. Keep the same transport encoding for A/B comparisons.
5. Freeze the selected configuration and its hash before running the holdout.
6. Run detector frameworks once as a forensic audit. Do not feed their outputs back
   into setting selection.
7. Run the blinded panel, sign recorded assets, verify the signed output, and retain
   the raw capture, report, configuration, software versions, and consent record.
8. Publish all evaluated metrics and limitations, including unfavorable subgroups
   and missing tools. Never report an unevaluated item as zero or as a pass.

## Current implementation decisions

- Recorded files require an independently verified signed C2PA manifest for release
  readiness. A template alone is explicitly marked unsigned.
- Detector results are stored under `detector_evaluation` and are excluded from all
  responsible-release score arithmetic.
- No score is claimed for the current product until real synchronized clips,
  detector exports, and subjective ratings are supplied.
