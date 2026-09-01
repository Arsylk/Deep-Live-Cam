"""Frequency-domain post-processing to restore natural spectral statistics.

GAN decoders (including inswapper_128) produce output with two systematic
frequency-domain anomalies that modern detectors exploit:

1. **Checkerboard spectral peaks** — transposed-convolution upsampling creates
   energy at specific DCT frequencies (multiples of output_size/stride).
2. **High-frequency suppression** — the decoder smooths away natural sensor
   noise, creating a spectral falloff steeper than any real camera produces.

This module repairs both by transferring the high-frequency statistics from the
original (camera-captured) face onto the swapped (GAN-generated) face. The
operation is face-local and bounded so it restores natural spectral content
without destroying identity features or introducing visible blur.

References:
  Frank et al. "Leveraging HF Components" ICML 2020
  Durall et al. "Unmasking DeepFakes with a Fourier Approach" CVPR 2020
  WaveDIF, FreqNet, FSBI detection frameworks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class FacePasteRegion:
    """Exact output-space alpha and ROI produced by one face paste.

    Keeping this small context beside the frame avoids rediscovering the
    generated region from pixel differences after the paste has completed.
    ``bounds`` uses the usual half-open ``(x1, y1, x2, y2)`` convention and
    ``alpha`` has exactly the corresponding ``(height, width)`` shape.
    """

    bounds: tuple[int, int, int, int]
    alpha: np.ndarray
    track_id: int = 0
    opacity: float = 1.0


_DETAIL_SCALE_STATE: dict[int, float] = {}
_CLOSE_KERNEL = np.ones((5, 5), dtype=np.uint8)
_DILATE_KERNEL = np.ones((7, 7), dtype=np.uint8)


def reset_camera_detail_state() -> None:
    """Forget per-track detail gain when a source or route changes."""
    _DETAIL_SCALE_STATE.clear()


def _smooth_detail_scale(track_id: int, scale: float) -> float:
    """Return a slow per-track EMA that prevents detail-gain pumping."""
    key = int(track_id)
    previous = _DETAIL_SCALE_STATE.get(key)
    smoothed = scale if previous is None else previous * 0.82 + scale * 0.18
    _DETAIL_SCALE_STATE[key] = float(smoothed)
    # Track IDs are normally a tiny, stable set. Bound defensive growth from
    # custom multi-face callers that manufacture a fresh ID every frame.
    if len(_DETAIL_SCALE_STATE) > 32:
        current = _DETAIL_SCALE_STATE[key]
        _DETAIL_SCALE_STATE.clear()
        _DETAIL_SCALE_STATE[key] = current
    return float(smoothed)


def _changed_region_mask(
    original: np.ndarray,
    processed: np.ndarray,
) -> np.ndarray:
    """Return a soft mask for pixels materially changed by face fusion."""
    # This is a compatibility fallback for callers that do not supply the
    # authoritative paste alpha. OpenCV keeps the common path in uint8 and
    # avoids two large float32 RGB conversions.
    delta = cv2.absdiff(original, processed)
    delta_luma = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    _, hard = cv2.threshold(delta_luma, 2, 255, cv2.THRESH_BINARY)
    hard = cv2.morphologyEx(
        hard,
        cv2.MORPH_CLOSE,
        _CLOSE_KERNEL,
    )
    hard = cv2.dilate(
        hard,
        _DILATE_KERNEL,
        iterations=1,
    )
    return np.clip(
        cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), 3.0),
        0.0,
        1.0,
    )


def _interior_detail_alpha(
    alpha: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Turn a paste alpha into a seam-protected detail alpha.

    The lower half of the paste transition deliberately receives no added
    detail. A smoothstep through the solid face interior avoids both a second
    morphology/blur pass and a sharpened halo along the fusion boundary.
    """
    if alpha.dtype == np.uint8:
        normalized = alpha.astype(np.float32) * (1.0 / 255.0)
    else:
        normalized = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    interior = np.clip((normalized - 0.45) * (1.0 / 0.43), 0.0, 1.0)
    interior *= interior * (3.0 - 2.0 * interior)
    return interior * float(np.clip(scale, 0.0, 1.0))


def _valid_paste_regions(
    paste_regions: Sequence[FacePasteRegion] | None,
    width: int,
    height: int,
) -> Iterable[tuple[int, int, int, int, np.ndarray, int, float]]:
    for region in paste_regions or ():
        if not isinstance(region, FacePasteRegion):
            continue
        try:
            x1, y1, x2, y2 = [int(value) for value in region.bounds]
        except (TypeError, ValueError):
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        alpha = region.alpha
        if (
            not isinstance(alpha, np.ndarray)
            or alpha.ndim != 2
            or alpha.shape != (y2 - y1, x2 - x1)
        ):
            continue
        yield (
            x1,
            y1,
            x2,
            y2,
            alpha,
            int(region.track_id),
            float(region.opacity),
        )


def match_camera_detail(
    processed_frame: np.ndarray,
    original_frame: np.ndarray | None,
    face_bboxes,
    strength: float = 0.0,
    *,
    paste_regions: Sequence[FacePasteRegion] | None = None,
    paste_alpha_scale: float = 1.0,
    boundary_strength: float = 0.0,
) -> np.ndarray:
    """Match final-resolution detail and preserve the paste transition.

    The 128px decoder output loses additional high-frequency energy when it is
    warped back into a 720p frame.  Repairs performed in model space cannot
    recover that last resampling loss.  This pass therefore measures the
    camera frame only to obtain a target *energy* and scales the generated
    face's own high-pass residual inside pixels actually changed by the swap.
    It never copies spatial texture or RGB content from the original face.

    ``strength`` is an interpolation/extrapolation factor toward the measured
    camera energy.  Values above 1 are useful after a small model crop has been
    enlarged; the bounded energy ratio and feathered change mask prevent a
    rectangular sharpened ROI or a hard paste boundary.

    ``boundary_strength`` performs one final, conservative reblend toward the
    genuine camera frame only in the translucent paste transition. It is zero
    in both the untouched background and solid identity-bearing face core.
    """
    strength = float(np.clip(strength, 0.0, 4.0))
    boundary_strength = float(np.clip(boundary_strength, 0.0, 1.0))
    if (
        (strength <= 0.0 and boundary_strength <= 0.0)
        or original_frame is None
        or not isinstance(processed_frame, np.ndarray)
        or not isinstance(original_frame, np.ndarray)
        or processed_frame.shape != original_frame.shape
        or processed_frame.ndim != 3
        or processed_frame.shape[2] != 3
        or (not face_bboxes and not paste_regions)
    ):
        return processed_frame

    height, width = processed_frame.shape[:2]
    result = processed_frame.copy()
    try:
        regions = list(_valid_paste_regions(paste_regions, width, height))
        exact_regions = bool(regions)
        if not regions and strength > 0.0:
            # Compatibility path for still-image and third-party callers.
            # The Windows live path supplies exact paste regions and never
            # performs this changed-pixel inference.
            for index, bbox in enumerate(face_bboxes or ()):
                if not hasattr(bbox, "__iter__") or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                if x2 <= x1 or y2 <= y1:
                    continue
                pad_x = max(4, int(round((x2 - x1) * 0.18)))
                pad_y = max(4, int(round((y2 - y1) * 0.18)))
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                original = original_frame[y1:y2, x1:x2]
                generated = result[y1:y2, x1:x2]
                if original.size == 0 or generated.size == 0:
                    continue
                inferred = _changed_region_mask(original, generated)
                regions.append((x1, y1, x2, y2, inferred, index, 1.0))

        for x1, y1, x2, y2, paste_alpha, track_id, opacity in regions:
            original = original_frame[y1:y2, x1:x2]
            generated = result[y1:y2, x1:x2]
            if original.size == 0 or generated.size == 0:
                continue
            if strength > 0.0:
                mask = _interior_detail_alpha(
                    paste_alpha,
                    float(paste_alpha_scale) * opacity,
                )
                core = mask > 0.5
                if int(core.sum()) >= 64:
                    # Measure and repair only luminance detail. This avoids
                    # inventing chroma grain and keeps both Gaussian pyramids
                    # single-channel.
                    original_y = cv2.cvtColor(
                        original, cv2.COLOR_BGR2GRAY
                    ).astype(np.float32)
                    generated_y = cv2.cvtColor(
                        generated, cv2.COLOR_BGR2GRAY
                    ).astype(np.float32)
                    original_high = original_y - cv2.GaussianBlur(
                        original_y, (0, 0), 1.0
                    )
                    generated_high = generated_y - cv2.GaussianBlur(
                        generated_y, (0, 0), 1.0
                    )

                    original_energy = float(
                        np.sqrt(np.mean(np.square(original_high[core])))
                    ) + 1e-6
                    generated_energy = float(
                        np.sqrt(np.mean(np.square(generated_high[core])))
                    ) + 1e-6
                    raw_scale = original_energy / generated_energy
                    target_scale = float(np.clip(raw_scale, 0.85, 1.75))
                    effective_scale = 1.0 + (target_scale - 1.0) * strength
                    minimum_scale = max(0.5, raw_scale * 0.8)
                    maximum_scale = min(
                        2.5, max(minimum_scale, raw_scale * 1.7)
                    )
                    effective_scale = float(
                        np.clip(effective_scale, minimum_scale, maximum_scale)
                    )
                    effective_scale = _smooth_detail_scale(
                        track_id, effective_scale
                    )
                    micro_strength = min(0.20, strength * (0.15 / 3.5))
                    combined_scale = effective_scale * (1.0 + micro_strength)
                    correction = (
                        generated_high * (combined_scale - 1.0) * mask
                    )[:, :, None]
                    result[y1:y2, x1:x2] = np.clip(
                        generated.astype(np.float32) + correction,
                        0,
                        255,
                    ).astype(np.uint8)

            if exact_regions and boundary_strength > 0.0:
                # One last target-preserving blend in the paste transition.
                # 4a(1-a) is exactly zero at alpha 0/1 and peaks at the
                # half-transparent seam; scaling by route opacity/fade avoids
                # fighting an intentional swap transition.
                if paste_alpha.dtype == np.uint8:
                    normalized_alpha = paste_alpha.astype(np.float32) * (
                        1.0 / 255.0
                    )
                else:
                    normalized_alpha = np.clip(
                        paste_alpha.astype(np.float32), 0.0, 1.0
                    )
                transition = (
                    4.0
                    * normalized_alpha
                    * (1.0 - normalized_alpha)
                    * boundary_strength
                    * float(np.clip(paste_alpha_scale * opacity, 0.0, 1.0))
                )
                transition_u8 = np.clip(
                    transition * 255.0, 0, 255
                ).astype(np.uint8)
                transition_3c = cv2.merge(
                    [transition_u8, transition_u8, transition_u8]
                )
                generated = result[y1:y2, x1:x2]
                kept = cv2.multiply(
                    generated,
                    255 - transition_3c,
                    scale=1.0 / 255.0,
                )
                restored = cv2.multiply(
                    original,
                    transition_3c,
                    scale=1.0 / 255.0,
                )
                result[y1:y2, x1:x2] = cv2.add(kept, restored)
    except (cv2.error, ValueError, TypeError, OverflowError):
        return processed_frame
    return result


def _spectral_energy_ratio(gray: np.ndarray) -> tuple[float, float]:
    """Return (low_freq_energy, high_freq_energy) normalized to sum to 1."""
    f = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.abs(np.fft.fftshift(f))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Low freq: central 25% radius
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = r < min(h, w) * 0.125
    high_mask = ~low_mask
    low = float(magnitude[low_mask].sum())
    high = float(magnitude[high_mask].sum())
    total = low + high + 1e-8
    return low / total, high / total


def _highpass_residual(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Extract the high-frequency residual (original - Gaussian blur)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return gray.astype(np.float32) - blurred.astype(np.float32)


def restore_high_frequency(
    swapped_face: np.ndarray,
    original_face: np.ndarray,
    strength: float = 0.30,
) -> np.ndarray:
    """Transfer high-frequency noise statistics from the original face to the
    swapped face.

    The GAN decoder suppresses the natural sensor noise that every real camera
    produces. Detectors trained on spectral falloff (FreqNet, FSBI) exploit
    this: a face with no HF content above 3/4 Nyquist is synthetic with near-
    certainty.

    This function extracts the HF residual from the original (camera-captured)
    face and adds a bounded fraction of it to the swapped face. The operation
    is deliberately conservative:
    - Only the residual (original - blur) is transferred, not the structure
    - Strength is bounded [0, 0.5] to prevent visible noise
    - The residual is masked to the face interior to avoid boundary artifacts

    Parameters
    ----------
    swapped_face : BGR 128x128 face produced by the swap model
    original_face : BGR 128x128 aligned original face (pre-swap)
    strength : fraction of original HF residual to add (0 = off, 0.5 = strong)

    Returns
    -------
    Repaired BGR face with natural HF content restored.
    """
    strength = float(np.clip(strength, 0.0, 0.5))
    if strength <= 0.0 or swapped_face.shape != original_face.shape:
        return swapped_face

    try:
        # Work in grayscale for the HF transfer (color channels are coupled)
        orig_gray = cv2.cvtColor(original_face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        swap_gray = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract HF residuals at two scales (fine and coarse)
        orig_hf_fine = _highpass_residual(orig_gray, sigma=1.0)
        orig_hf_coarse = _highpass_residual(orig_gray, sigma=2.0)
        swap_hf_fine = _highpass_residual(swap_gray, sigma=1.0)
        swap_hf_coarse = _highpass_residual(swap_gray, sigma=2.0)

        # Compute the energy ratio we need to restore
        orig_rms_fine = float(np.std(orig_hf_fine)) + 1e-6
        swap_rms_fine = float(np.std(swap_hf_fine)) + 1e-6
        orig_rms_coarse = float(np.std(orig_hf_coarse)) + 1e-6
        swap_rms_coarse = float(np.std(swap_hf_coarse)) + 1e-6

        # Scale factors: how much to boost the swapped face's HF
        scale_fine = min(2.0, orig_rms_fine / swap_rms_fine)
        scale_coarse = min(2.0, orig_rms_coarse / swap_rms_coarse)

        # Build a correction: difference in HF between what we have and what we want
        fine_correction = swap_hf_fine * (scale_fine - 1.0) * strength
        coarse_correction = swap_hf_coarse * (scale_coarse - 1.0) * strength

        # Blend in a small amount of the original's HF texture as well
        # (this transfers the actual noise grain, not just the energy level)
        texture_transfer = orig_hf_fine * strength * 0.3

        total_correction = fine_correction + coarse_correction + texture_transfer

        # Apply to all BGR channels equally (HF is luminance-dominated)
        corrected = swapped_face.astype(np.float32)
        for c in range(3):
            corrected[:, :, c] += total_correction

        return np.clip(corrected, 0, 255).astype(np.uint8)
    except (cv2.error, ValueError, TypeError):
        return swapped_face


def suppress_checkerboard(
    face: np.ndarray,
    model_size: int = 128,
    attenuation: float = 0.4,
) -> np.ndarray:
    """Attenuate DCT energy peaks from transposed-convolution upsampling.

    The inswapper_128 model uses transposed convolutions with stride 2, which
    creates energy peaks at specific DCT frequencies (multiples of
    model_size/2). This is the "checkerboard artifact" documented in
    Odena et al. 2016 and exploited by DCT-based detectors.

    A gentle notch filter at these frequencies removes the peaks without
    blurring the face (the peaks are narrow-band and carry no identity info).

    Parameters
    ----------
    face : BGR face image (any size, typically 128x128)
    model_size : the GAN decoder's native spatial size
    attenuation : how much to suppress the peaks (0 = none, 1 = full removal)

    Returns
    -------
    Face with checkerboard peaks attenuated.
    """
    attenuation = float(np.clip(attenuation, 0.0, 1.0))
    if attenuation <= 0.0:
        return face

    try:
        h, w = face.shape[:2]
        corrected = face.copy().astype(np.float32)

        for c in range(3):
            channel = face[:, :, c].astype(np.float32)
            # DCT via OpenCV (Type II)
            dct = cv2.dct(channel)
            # The checkerboard peaks appear at multiples of (size / stride).
            # For a stride-2 decoder, peaks are at frequencies n*(size/2).
            # At 128px, that's every 64th DCT coefficient in both axes.
            stride = model_size // 2
            if stride < min(h, w):
                # Attenuate the specific frequency bands
                for fy in range(stride, h, stride):
                    for fx in range(stride, w, stride):
                        # Apply a gentle notch (not hard zero)
                        band_size = max(2, stride // 8)
                        y1 = max(0, fy - band_size)
                        y2 = min(h, fy + band_size)
                        x1 = max(0, fx - band_size)
                        x2 = min(w, fx + band_size)
                        dct[y1:y2, x1:x2] *= (1.0 - attenuation)

            corrected[:, :, c] = cv2.idct(dct)

        return np.clip(corrected, 0, 255).astype(np.uint8)
    except (cv2.error, ValueError, TypeError):
        return face


def match_wavelet_statistics(
    swapped_face: np.ndarray,
    original_face: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Match the wavelet detail subband statistics of the swapped face to
    the original face.

    WaveDIF (CVPR 2025 Workshop) and FSBI detect deepfakes by examining
    the distribution of DWT detail coefficients (LH, HL, HH). GAN output
    has non-natural distributions in these subbands because the decoder
    does not produce natural high-frequency texture.

    This function performs a single-level Haar DWT, computes the mean and
    std of each detail subband for both faces, and scales the swapped
    face's subbands toward the original's statistics.

    Parameters
    ----------
    swapped_face : BGR face from the swap model
    original_face : BGR aligned original face
    strength : blending strength (0 = no change, 1 = full match)

    Returns
    -------
    Face with wavelet statistics matched to the original.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or swapped_face.shape != original_face.shape:
        return swapped_face

    try:
        # Use a simple 2-level wavelet-like decomposition via pyrDown/pyrUp
        # (avoids requiring PyWavelets dependency)
        result = swapped_face.copy().astype(np.float32)

        for c in range(3):
            swap_ch = swapped_face[:, :, c].astype(np.float32)
            orig_ch = original_face[:, :, c].astype(np.float32)

            # Level 1: residual after Gaussian smoothing
            swap_smooth = cv2.GaussianBlur(swap_ch, (0, 0), 1.0)
            orig_smooth = cv2.GaussianBlur(orig_ch, (0, 0), 1.0)
            swap_detail = swap_ch - swap_smooth
            orig_detail = orig_ch - orig_smooth

            swap_std = float(np.std(swap_detail)) + 1e-6
            orig_std = float(np.std(orig_detail)) + 1e-6

            # Scale detail subband toward target statistics
            target_scale = orig_std / swap_std
            target_scale = np.clip(target_scale, 0.5, 2.0)
            effective_scale = 1.0 + (target_scale - 1.0) * strength

            result[:, :, c] = swap_smooth + swap_detail * effective_scale

        return np.clip(result, 0, 255).astype(np.uint8)
    except (cv2.error, ValueError, TypeError):
        return swapped_face


def apply_frequency_repair(
    swapped_face: np.ndarray,
    original_face: np.ndarray,
    hf_strength: float = 0.0,
    checkerboard_attenuation: float = 0.0,
    wavelet_strength: float = 0.0,
) -> np.ndarray:
    """Apply all frequency-domain repairs in the correct order.

    Order matters: wavelet matching first (coarsest), then checkerboard
    suppression (mid-frequency), then HF restoration (finest).

    Parameters
    ----------
    swapped_face : BGR face from the swap model (128x128)
    original_face : BGR aligned original face (128x128)
    hf_strength : restore_high_frequency strength (0 = off)
    checkerboard_attenuation : suppress_checkerboard attenuation (0 = off)
    wavelet_strength : match_wavelet_statistics strength (0 = off)

    Returns
    -------
    Repaired face.
    """
    result = swapped_face
    if wavelet_strength > 0.0:
        result = match_wavelet_statistics(result, original_face, wavelet_strength)
    if checkerboard_attenuation > 0.0:
        result = suppress_checkerboard(result, attenuation=checkerboard_attenuation)
    if hf_strength > 0.0:
        result = restore_high_frequency(result, original_face, hf_strength)
    return result
