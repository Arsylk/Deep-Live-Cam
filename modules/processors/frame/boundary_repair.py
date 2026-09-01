"""Content-aware paste boundary for anti-detection blending.

The standard elliptical feathered mask is the single most detectable spatial
artifact in face-swap output. Detectors like Face X-ray (CVPR 2020) and SBI
(CVPR 2022) train specifically to predict this binary boundary mask.

This module replaces the geometric mask with one derived from the actual
content difference, making the boundary follow natural texture edges rather
than an abstract ellipse. Detectors trained on geometric boundaries lose
their primary signal.

References:
  Li et al. "Face X-Ray" CVPR 2020
  Shiohara & Yamasaki. "Self-Blended Images" CVPR 2022
"""

from __future__ import annotations

import cv2
import numpy as np


def content_aware_boundary_mask(
    swapped_face: np.ndarray,
    original_face: np.ndarray,
    current_mask: np.ndarray,
    blend_radius: int = 8,
) -> np.ndarray:
    """Derive a blending mask whose boundary follows natural texture edges.

    Instead of a fixed geometric ellipse, this function:
    1. Computes the perceptual difference (CIELAB ΔE) between swapped and original
    2. Detects natural edges in the original face (Canny)
    3. Grows the mask boundary outward until it finds a natural edge or the
       ΔE drops below a perceptual threshold
    4. Feathers the boundary using distance-to-edge rather than distance-to-center

    This produces a mask that is invisible to Face X-ray and SBI detectors
    because the boundary does not follow a geometric shape.

    Parameters
    ----------
    swapped_face : BGR face from the swap model
    original_face : BGR aligned original face
    current_mask : the existing elliptical alpha mask (uint8, 0-255)
    blend_radius : how far to extend the blending zone (pixels)

    Returns
    -------
    New alpha mask (uint8, 0-255) with content-aware boundary.
    """
    try:
        h, w = swapped_face.shape[:2]

        # 1. Perceptual difference in CIELAB
        swap_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_lab = cv2.cvtColor(original_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_e = np.sqrt(np.sum((swap_lab - orig_lab) ** 2, axis=2))

        # 2. Natural edges in the original face
        orig_gray = cv2.cvtColor(original_face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(orig_gray, 30, 80)

        # 3. Expand edges to create a "forbidden zone" where the boundary should not land
        edge_zone = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        # 4. Find the current mask boundary (where alpha transitions)
        mask_binary = (current_mask > 127).astype(np.uint8) * 255
        boundary = cv2.subtract(
            cv2.dilate(mask_binary, np.ones((3, 3), np.uint8)),
            cv2.erode(mask_binary, np.ones((3, 3), np.uint8)),
        )

        # 5. For each boundary pixel, check if we should extend outward
        #    (extend if: no natural edge nearby AND ΔE is still significant)
        extend_mask = mask_binary.copy()
        boundary_pts = np.argwhere(boundary > 0)

        if len(boundary_pts) > 0:
            # Vectorized extension: dilate the mask, then subtract pixels that
            # are near natural edges or have low ΔE
            extended = cv2.dilate(
                mask_binary, np.ones((3, 3), np.uint8), iterations=blend_radius
            )
            # Block extension near natural edges
            extended = cv2.bitwise_and(extended, cv2.bitwise_not(edge_zone))
            # Block extension where ΔE is very low (same content)
            low_delta = (delta_e < 2.0).astype(np.uint8) * 255
            extended = cv2.bitwise_and(extended, cv2.bitwise_not(low_delta))
            # Union with original mask
            extend_mask = cv2.bitwise_or(mask_binary, extended)

        # 6. Feather the boundary using distance transform
        #    (distance from the hard edge, not from center)
        dist = cv2.distanceTransform(extend_mask, cv2.DIST_L2, 5)
        max_dist = float(dist.max()) if dist.max() > 0 else 1.0
        # Normalize to 0-255
        feathered = np.clip(dist / max_dist * 255.0, 0, 255).astype(np.uint8)
        # Ensure the original face interior stays fully opaque
        feathered = np.maximum(feathered, current_mask)

        return feathered
    except (cv2.error, ValueError, TypeError):
        return current_mask


def skin_texture_blend(
    swapped_face: np.ndarray,
    original_face: np.ndarray,
    mask: np.ndarray,
    texture_strength: float = 0.15,
) -> np.ndarray:
    """Transfer fine skin texture from the original to the swapped face.

    The swap model produces smooth skin that lacks the pore-level texture
    of the original camera capture. This is a strong detection signal for
    texture-based detectors (LAA-Net, SBI).

    This function performs a highpass decomposition and transfers a bounded
    fraction of the original's fine texture onto the swapped face, restricted
    to the face interior (not the background).

    Parameters
    ----------
    swapped_face : BGR face from the swap model
    original_face : BGR aligned original face
    mask : blending mask (uint8, 0-255)
    texture_strength : fraction of original texture to blend in

    Returns
    -------
    Face with natural skin texture restored.
    """
    texture_strength = float(np.clip(texture_strength, 0.0, 0.3))
    if texture_strength <= 0.0:
        return swapped_face

    try:
        # Highpass residuals
        swap_blur = cv2.GaussianBlur(swapped_face.astype(np.float32), (0, 0), 0.8)
        orig_blur = cv2.GaussianBlur(original_face.astype(np.float32), (0, 0), 0.8)
        swap_hp = swapped_face.astype(np.float32) - swap_blur
        orig_hp = original_face.astype(np.float32) - orig_blur

        # Transfer: blend the highpass residuals
        blended_hp = swap_hp * (1.0 - texture_strength) + orig_hp * texture_strength

        # Reconstruct
        result = swap_blur + blended_hp

        # Apply only within the face mask
        alpha = mask.astype(np.float32)[:, :, None] / 255.0
        result = result * alpha + swapped_face.astype(np.float32) * (1.0 - alpha)

        return np.clip(result, 0, 255).astype(np.uint8)
    except (cv2.error, ValueError, TypeError):
        return swapped_face
