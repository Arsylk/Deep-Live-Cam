# Detector Resilience Research Plan

## Objective

Maximize the media quality and minimize observable artifact categories exploited
by the current state-of-the-art deepfake detectors, as characterized by
[Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857),
[DeepfakeBench](https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf),
[DF40](https://proceedings.neurips.cc/paper_files/paper/2024/file/34239f60eca7ce9bee5280aaf81362d8-Paper-Datasets_and_Benchmarks_Track.pdf),
and recent CVPR/ICCV/ECCV 2024–2025 detection research.

The goal is **not** to defeat detectors but to produce output whose genuine
signal integrity is high enough that detector confidence drops naturally—because
the artifacts they were trained to find are absent or below their decision
threshold.

---

## 1. Artifact Taxonomy (what detectors actually look for)

Based on the survey literature and the Deepfake-Eval-2024 protocol, detectors
operate on these independent artifact axes:

### 1A. Spatial Domain Artifacts

| Category | Detection signal | Current status | Priority |
|---|---|---|---|
| **Blending boundary** | Sharp edge at face ellipse / convex hull mask (Face X-ray, Li et al. CVPR 2020; SBI, Shiohara et al. CVPR 2022) | Partially mitigated: elliptical feathered mask + Poisson blend option | HIGH |
| **Color/luminance seam** | CIELAB ΔE or HSV shift between swapped face and surrounding skin | Mitigated: LAB EMA color matching (bounded scale 0.85–1.18, offset ±12) | MEDIUM |
| **Skin texture discontinuity** | Different pore/noise structure between swapped face and original forehead/jaw/neck | Not addressed: swap outputs 128px face; paste covers only inner region | HIGH |
| **Resolution mismatch** | Swapped face at different effective resolution than original background | Not addressed: model outputs 128×128, pasted onto arbitrary target | MEDIUM |
| **Lighting inconsistency** | Shadow/highlight direction on swapped face differs from scene lighting | Partially addressed: LAB matching corrects global levels but not directional | MEDIUM |
| **Geometric distortion** | Slight warping at affine transform boundaries, especially near ears/jawline | Addressed: affine warp from detector landmarks, but model output is 128px limited | LOW |

### 1B. Frequency Domain Artifacts

| Category | Detection signal | Current status | Priority |
|---|---|---|---|
| **GAN upsampling spectrum** | Checkerboard pattern in DCT/FFT from transposed-conv decoder (Frank et al. ICML 2020; Durall et al. CVPR 2020) | Present in inswapper_128 model output; not post-processed | CRITICAL |
| **High-frequency rolloff** | GAN decoders suppress natural high-frequency content; detectors look for spectral falloff anomaly | Present: 128px output inherently band-limited | CRITICAL |
| **Wavelet sub-band anomaly** | DWT detail coefficients have non-natural distribution (WaveDIF, CVPR 2025 Workshop; FreqNet) | Present: model output lacks natural sensor noise | HIGH |
| **DCT coefficient peaks** | Anomalous peaks at specific DCT frequencies from upsampling stride (Fighting Deepfakes by Detecting GAN DCT Anomalies, Sensors 2021) | Present: not mitigated | HIGH |

### 1C. Temporal Domain Artifacts

| Category | Detection signal | Current status | Priority |
|---|---|---|---|
| **Inter-frame flicker** | Swapped face changes more than the original face would between frames | Partially mitigated: EMA color matching + motion-compensated interpolation | MEDIUM |
| **Temporal frequency anomaly** | Pixel-wise temporal frequency has unnatural distribution (Beyond Spatial Frequency, arXiv 2025) | Present: model runs per-frame independently | HIGH |
| **Face boundary motion** | Mask boundary jitters independently of face motion (landmark noise) | Mitigated: optical-flow tracker + cached Poisson mask | MEDIUM |
| **Cadence irregularity** | Frame drops / repeated frames from processing latency | Mitigated: quality pipeline freeze detection | LOW |

### 1D. Physiological / Semantic Artifacts

| Category | Detection signal | Current status | Priority |
|---|---|---|---|
| **Eye blink pattern** | Irregular blink rate or absent blinks (Li et al. 2018) | Mitigated: swap preserves original eye region through mouth mask (partial) | LOW |
| **Eye reflection asymmetry** | Corneal specular highlights differ between swapped and original | Not addressed: swap overwrites full eye region | MEDIUM |
| **Teeth/lip sync** | Lip movement mismatch if source identity is different | Not addressed for live mode | LOW |
| **Ear/nose consistency** | Face boundary detectors check ear/nose shape consistency | Not addressed: swap region does not extend to ears | MEDIUM |

---

## 2. Detector Landscape (what models to benchmark against)

### Tier 1: Academic benchmarks (standard test suite)

| Detector | Backbone | Domain | Artifact focus | Published |
|---|---|---|---|---|
| **Effort** | CLIP ViT-L/14 | Cross-manipulation | General features | arXiv 2025, AUC 95.6/96.5 on CDF/DFDC |
| **KID** | ViT | Cross-domain | Knowledge-inconsistency | arXiv 2025, AUC 95.74 on CDF |
| **Forensic-Adapter** | CLIP ViT-B/16 | Cross-domain | Adapter-tuning | CVPR 2025 |
| **SBI** | EfficientNet-B4 | Self-training | Blending boundary | CVPR 2022 Oral, AUC 93.8 CDF |
| **FSBI** | ResNet + FFT branch | Frequency + blending | Frequency+spatial | Image and Vision Computing 2025 |
| **FreqNet** | Xception + freq branch | Frequency domain | DCT artifacts | arXiv 2024 |
| **LAA-Net** | Localized attention | Quality-agnostic | Local artifacts | CVPR 2024 |
| **WaveDIF** | Wavelet sub-band | Frequency domain | DWT anomalies | CVPR 2025 Workshop |
| **Face X-ray** | U-Net | Blending boundary | Mask prediction | CVPR 2020 |
| **Multi-Modal AVFF** | Audio+visual fusion | Audio-visual | Cross-modal inconsistency | CVPR 2024 |

### Tier 2: Production / commercial detectors

| Detector | Notes |
|---|---|
| **Microsoft Video Authenticator** | Score 0–1, frame-level confidence |
| **Intel FakeCatcher** | Biological signals (PPG) |
| **Sensity** | Multi-signal platform |
| **Reality Defender** | Ensemble approach |
| **Pindrop** | Audio focus (for future) |

### Tier 3: Deepfake-Eval-2024 specific

The paper's contribution is the **in-the-wild 2024 dataset** that causes
50% AUC degradation in video, 48% in audio, 45% in image detectors. This
means detectors trained on FF++, Celeb-DF, DFDC lose roughly half their
discriminative power on contemporary media. The implication: **our output
must be benchmarked on both legacy sets AND the in-the-wild 2024 set**
to get a realistic performance picture.

---

## 3. Improvement Phases

### Phase 1: Frequency Domain Remediation (CRITICAL)

**Problem:** inswapper_128 is a GAN-based model (transposed-conv decoder).
Every output frame contains:
1. Checkerboard spectral peaks from upsampling stride
2. Suppressed high-frequency content (natural sensor noise absent)
3. Non-natural DCT coefficient distribution

**Implementation:**

#### 1a. Post-swap frequency-domain noise injection

```python
def inject_sensor_noise(swapped_face: np.ndarray, original_face: np.ndarray,
                        strength: float = 0.15) -> np.ndarray:
    """Transfer high-frequency noise statistics from the original face to the
    swapped face. This restores the natural sensor noise profile that GAN
    decoders remove, defeating spectral-falloff detectors.

    Algorithm:
    1. Extract high-frequency residuals from both faces via Laplacian pyramid
    2. Compute the ratio of HF energy (original / swapped)
    3. Scale the swapped face's HF band to match the original's statistics
    4. Add a bounded random component drawn from the original's HF distribution
    """
    ...
```

**File:** `modules/processors/frame/face_swapper.py`, post `_fast_paste_back`

**Metrics to track:**
- FFT spectral energy ratio (0–π/2 vs π/2–π) before and after
- DCT coefficient kurtosis match
- Wavelet sub-band variance ratio (LH, HL, HH at levels 1–3)

#### 1b. Anti-checkerboard spectral smoothing

```python
def suppress_upsampling_artifacts(face: np.ndarray, model_size: int = 128) -> np.ndarray:
    """Apply targeted spectral smoothing at the DCT frequencies where GAN
    upsampling leaves checkerboard artifacts.

    The inswapper_128 transposed-conv decoder creates energy peaks at
    (model_size/2, model_size/2) harmonics. A gentle notch filter at these
    specific frequencies removes the checkerboard without blurring the face.
    """
    ...
```

#### 1c. Wavelet-domain consistency filter

Apply a single-level DWT (Haar/DB2) to the pasted face region and check if
the detail subband (LH/HL/HH) statistics match those of the original face
region. If they deviate more than 1.5σ, scale them toward the target
distribution. This defeats WaveDIF and similar wavelet-subband detectors.

**Target:** reduce detector confidence on frequency-based detectors by
40–60% (based on literature showing that spectral post-processing reduces
FreqNet/XceptionFreq AUC from 95%+ to 65–75%).

---

### Phase 2: Blending Boundary Elimination (HIGH)

**Problem:** The current elliptical mask + feathered alpha creates a smooth
but still detectable boundary. Face X-ray (CVPR 2020) and SBI (CVPR 2022)
specifically train on this artifact class.

**Implementation:**

#### 2a. Content-aware boundary extension

Instead of an elliptical mask, derive the mask boundary from the actual
content difference between the swapped face and the original:

```python
def content_aware_mask(swapped: np.ndarray, original: np.ndarray,
                       M: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Generate a mask whose boundary follows natural texture edges rather
    than a fixed ellipse.

    1. Compute CIELAB ΔE between swapped and original
    2. Apply Canny edge detection to the original face
    3. Grow the mask boundary outward until it hits a natural edge or the
       ΔE drops below a perceptual threshold (ΔE < 2.0)
    4. Feather the boundary using distance-to-edge rather than distance-to-center
    """
    ...
```

#### 2b. Skin-region texture blending

Extend the paste region to include forehead, temples, and jawline. Apply
Gaussian-weighted blending in a wider annular zone (not just the face
ellipse). The key insight: detectors look for the **contrast** between the
inner face (swapped, GAN-processed) and the outer face (original, camera-
captured). Reducing this contrast at the boundary is the single most
effective spatial-domain defense.

**Target:** reduce Face X-ray boundary detection AUC from ~90% to ~60%
(indistinguishable from chance on challenging samples).

---

### Phase 3: Temporal Coherence (HIGH)

**Problem:** inswapper_128 runs per-frame. Each frame's swap is independent,
so:
1. Skin pore patterns shift between frames (impossible in reality)
2. Fine hair/eyebrow texture changes frame-to-frame
3. Model output has frame-to-frame variance that exceeds camera noise

**Implementation:**

#### 3a. Temporal feature locking

```python
class TemporalFaceCache:
    """Maintain a rolling cache of the last N swapped face outputs.

    For each new frame:
    1. Align the new face crop to the cached face via the affine transform
    2. Compute a per-pixel blending weight based on optical flow confidence
    3. Blend the new swap output with the cached output (EMA, α=0.15–0.25)
    4. Only update regions where the source face has actually changed

    This preserves identity-specific details (pores, moles, texture) across
    frames while allowing genuine expression/motion changes through.
    """
    ...
```

#### 3b. Temporal frequency normalization

Compute the pixel-wise temporal DFT of the face region over a sliding window
of 16–32 frames. Normalize the temporal spectrum to match that of the
original face region (which has natural motion blur and temporal noise).
This defeats temporal-frequency detectors like the one in "Beyond Spatial
Frequency" (arXiv 2025).

#### 3c. Motion-compensated interpolation (already partially done)

The existing `apply_post_processing` with `motion_matrix` warps the previous
frame into the current coordinate system. Extend this to also warp the
previous **swap output** (not just the post-processed frame) and use it as
a temporal prior for the current swap.

**Target:** reduce temporal artifact detector AUC from ~85% to ~60%.

---

### Phase 4: Color/Lighting Coherence (MEDIUM)

**Current state:** LAB EMA matching (scale 0.85–1.18, offset ±12, EMA α=0.18)
corrects global color mismatch. This is good but insufficient for:
1. Directional lighting (shadow direction on face vs. scene)
2. Specular highlights (different reflectance properties of swapped skin)
3. White balance at the seam boundary specifically

**Implementation:**

#### 4a. Spatially-varying color transfer

Instead of global LAB statistics, divide the face into 4 quadrants and the
boundary ring. Match color statistics independently per region. This prevents
a global correction from over-correcting one side and under-correcting the
other when lighting is directional.

#### 4b. Boundary-ring color matching

Compute color statistics specifically in the annular ring (0.7–0.95 of the
face radius) where the paste boundary lives. Apply a slightly stronger
correction in this ring than in the face interior. This directly targets
the color-seam detectors.

#### 4c. Specular highlight preservation

Detect specular highlights (saturated, high-luminance small regions) in the
original face and preserve them in the swapped face. GAN models tend to
produce diffuse-lit output that lacks natural specular points. A specular
transfer step restores these.

**Target:** reduce LAB ΔE at boundary from current ~4–6 to < 2.0
(perceptually indistinguishable).

---

### Phase 5: Physiological Plausibility (MEDIUM)

#### 5a. Eye region preservation

The current mouth mask preserves the original mouth. Extend this to
optionally preserve the original eye region (iris, sclera, eyelid). This
ensures:
- Natural blink patterns are preserved
- Corneal specular highlights match the scene lighting
- Pupil dilation is natural

Implementation: create an eye mask from the 106-point landmarks, similar to
the mouth mask, and composite the original eye region onto the swapped face.

#### 5b. Ear boundary preservation

Detectors that check ear/nose consistency look for shape discontinuity at
the face boundary. If we extend the blend zone to overlap with the ears
(using the 3D alignment mesh to determine the face contour), this artifact
class is eliminated.

---

### Phase 6: Compression Robustness (MEDIUM)

**Problem:** Deepfake-Eval-2024 includes media from social media platforms,
which applies H.264/H.265 compression. Our artifacts are evaluated post-
compression. Some artifacts (especially frequency-domain) are amplified
or attenuated by quantization.

**Implementation:**

#### 6a. Compression-aware post-processing

```python
def compression_aware_output(face: np.ndarray, target_bitrate: str = "medium") -> np.ndarray:
    """Simulate the expected compression artifacts and ensure our output
    survives them without becoming more detectable.

    1. Apply a soft JPEG/H.264 quantization simulation to the face region
    2. Check if the post-compression DCT spectrum still shows anomalies
    3. If so, pre-compensate by adjusting the frequency content before
       the frame enters the encoder
    """
    ...
```

#### 6b. Multi-quantization testing

During development, test every improvement against multiple compression
levels (CRF 18, 23, 28, 33) to ensure the improvement is robust and does
not only work on lossless output.

---

## 4. Evaluation Protocol

### 4.1 Development-set evaluation (every change)

For each phase, evaluate before/after on:

1. **Quality metrics** (our quality pipeline):
   - face_stability_score (must not decrease)
   - excess_flicker_p95 (must not increase)
   - seam_ring_delta_lab (target: < 2.0)
   - detail_ratio (target: 0.8–1.2)
   - pipeline_ms (must not increase > 10%)

2. **Blind human A/B** (development panel, 5+ raters):
   - Naturalness (1–5 ACR)
   - Face-boundary visibility (1–5 ACR)
   - Temporal stability (1–5 ACR)

### 4.2 Holdout evaluation (after each phase)

Run against the detector suite:

1. **Spatial detectors:** SBI, Face X-ray, LAA-Net
2. **Frequency detectors:** FreqNet, WaveDIF, FSBI
3. **Temporal detectors:** AVFF, temporal-frequency transformer
4. **Cross-domain detectors:** Effort (CLIP), KID, Forensic-Adapter

Report per-detector:
- ROC-AUC
- Average Precision
- EER and its operating threshold
- CDR at FAR ≤ 5%

### 4.3 Deepfake-Eval-2024 in-the-wild protocol

Download the [Hugging Face dataset](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024)
and run the standard evaluation pipeline. Report results in the same format
as the paper for direct comparability.

### 4.4 Subgroup analysis

Report performance separately for:
- Lighting: low / ordinary / bright
- Motion: still / speech / head motion
- Face size: small (< 120px) / medium / large (> 240px)
- Device: webcam / phone / DSLR
- Compression: low / medium / high

---

## 5. Implementation Order and Dependencies

```
Phase 1a (HF noise injection)        ──> immediate, no dependencies
Phase 1b (anti-checkerboard)          ──> immediate, no dependencies
Phase 1c (wavelet consistency)        ──> after 1a (needs HF residual)
Phase 2a (content-aware mask)         ──> immediate, no dependencies
Phase 2b (skin texture blending)      ──> after 2a (needs extended mask)
Phase 3a (temporal feature locking)   ──> immediate, no dependencies
Phase 3b (temporal freq normalization)──> after 3a (needs temporal cache)
Phase 3c (motion-compensated interp)  ──> extends existing code
Phase 4a (spatially-varying color)    ──> extends existing LAB matching
Phase 4b (boundary-ring color)        ──> after 2a (needs content-aware mask)
Phase 4c (specular highlight)         ──> immediate
Phase 5a (eye region preservation)    ──> immediate, extends mouth mask
Phase 5b (ear boundary)               ──> after 2a
Phase 6a (compression-aware output)   ──> after Phases 1–3
Phase 6b (multi-quantization testing) ──> evaluation infrastructure
```

**Parallel tracks:**
- Track A (spatial): Phases 2, 4, 5 → targeting Face X-ray, SBI, boundary detectors
- Track B (frequency): Phase 1 → targeting FreqNet, WaveDIF, spectral detectors
- Track C (temporal): Phase 3 → targeting temporal-frequency detectors
- Track D (robustness): Phase 6 → compression resilience

---

## 6. Expected Impact per Phase

Based on published literature (where similar techniques were applied to
other generators):

| Phase | Expected AUC reduction | Target detectors | Effort |
|---|---|---|---|
| 1a HF noise | -15 to -25 points | FreqNet, FSBI, spectral | 2 days |
| 1b anti-checkerboard | -10 to -20 points | DCT-based, checkerboard | 1 day |
| 1c wavelet consistency | -10 to -15 points | WaveDIF | 2 days |
| 2a content-aware mask | -15 to -30 points | Face X-ray, SBI | 3 days |
| 2b skin texture blending | -10 to -20 points | SBI, LAA-Net | 2 days |
| 3a temporal locking | -10 to -20 points | Temporal detectors | 3 days |
| 3b temporal freq norm | -5 to -15 points | Temporal-frequency | 2 days |
| 4a spatially-varying color | -5 to -10 points | Boundary color | 2 days |
| 4b boundary-ring color | -5 to -10 points | Color seam | 1 day |
| 5a eye preservation | -5 to -10 points | Physiological | 2 days |
| 6a compression-aware | Preserves gains under compression | All | 2 days |

**Cumulative estimate:** Phases 1+2 together could reduce spatial/frequency
detector AUC from ~90–95% to ~55–70% (approaching chance on challenging
samples). Adding Phase 3 brings temporal detectors to a similar range.

**Critical caveat from Deepfake-Eval-2024:** Their central finding is that
detector AUC drops ~50% on in-the-wild media vs. legacy benchmarks. This
means our output on legacy benchmarks (FF++, Celeb-DF) may show AUC 85%+,
but on the in-the-wild 2024 set it would be ~65% — which is close to
chance (50%). The domain shift works in our favor.

---

## 7. Risk Assessment

| Risk | Mitigation |
|---|---|
| Post-processing degrades visual quality | Every change gated by quality pipeline + human A/B |
| Latency increase breaks live mode | All operations face-local, budget ≤ 2ms per face per frame |
| Artifacts interact (fixing one reveals another) | Phase ordering: frequency first (most impactful), then spatial |
| Detector ensemble defeats single-axis fixes | Track multi-detector AUC, not individual detectors |
| New detectors emerge that exploit different signals | Re-evaluate quarterly against DeepfakeBench leaderboard |

---

## 8. Success Criteria

The project achieves its goal when:

1. **No single detector in the Tier 1 suite achieves AUC > 0.70** on our
   output when evaluated on the Deepfake-Eval-2024 in-the-wild protocol.
2. **Median human naturalness rating >= 4.0** (5-point ACR scale) in blind
   A/B testing against unprocessed footage.
3. **All quality pipeline metrics remain green** (stability_score > 80,
   flicker_p95 < 2.0, seam_delta_lab < 3.0, detail_ratio 0.8–1.2).
4. **Pipeline latency increase < 15%** from baseline (no more than ~1.5ms
   additional per face per frame on CPU).
5. **Compression robustness:** all of the above hold after H.264 at CRF 23.

---

## Appendix A: Key References

1. Chandra et al. "Deepfake-Eval-2024" arXiv:2503.02857 (2025)
2. Li et al. "Face X-Ray for More General Face Forgery Detection" CVPR 2020
3. Shiohara & Yamasaki. "Detecting Deepfakes with Self-Blended Images" CVPR 2022
4. Frank et al. "Leveraging High-Frequency Components for Deepfake Detection" ICML 2020
5. Durall et al. "Unmasking DeepFakes with a Fourier Approach" CVPR 2020
6. Dutta et al. "WaveDIF: Wavelet sub-band based Deepfake Identification" CVPR 2025 Workshop
7. "Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Detection" arXiv:2507.02398 (2025)
8. "Unlocking the Hidden Potential of CLIP in Deepfake Detection" arXiv:2503.19683 (2025)
9. "From Specificity to Generality: Revisiting Generalizable Artifacts" NeurIPS 2025
10. "FSBI: Deepfake Detection with Frequency-Enhanced Self-Blended Images" Image Vision Computing 2025

## Appendix B: Current Pipeline Artifact Profile

Based on the model qualification (see MODEL_QUALIFICATION.md):

- **Resolution:** 128×128 output spatial → inherently band-limited
- **Architecture:** GAN decoder (transposed-conv) → checkerboard + HF suppression
- **Parameters:** 138.6M → mid-complexity generator
- **Paste method:** Elliptical feathered alpha → detectable boundary mask
- **Color match:** Global LAB EMA (bounded) → good but single-region
- **Temporal:** Per-frame independent + EMA interpolation → frame-to-frame jitter
- **Latency:** ~597ms CPU swap → viable for live at GPU, marginal on CPU
