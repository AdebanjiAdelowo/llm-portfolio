"""Mask postprocessing: threshold, denoise, morphology, and CRF refinement."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert soft mask to binary uint8."""
    return ((mask >= threshold) * 255).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    binary = binarize(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned.astype(np.float32) / 255.0


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask."""
    binary = binarize(mask)
    filled = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
    return filled.astype(np.float32) / 255.0


def smooth_edges(mask: np.ndarray, kernel_size: int = 5, sigma: float = 1.5) -> np.ndarray:
    """Gaussian-smooth mask edges to reduce aliasing."""
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return blurred.clip(0.0, 1.0)


def morphological_close(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Close small gaps between mask regions."""
    binary = binarize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.float32) / 255.0


def morphological_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove small noise speckles from mask."""
    binary = binarize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened.astype(np.float32) / 255.0


def guided_filter_refine(
    mask: np.ndarray,
    guide_image: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Use the original image as a guided filter to sharpen mask boundaries.

    Requires OpenCV with contrib (pip install opencv-contrib-python).
    Falls back silently if not available.
    """
    try:
        gf = cv2.ximgproc.createGuidedFilter(guide_image, radius, eps)
        refined = gf.filter(mask)
        return refined.clip(0.0, 1.0)
    except AttributeError:
        return mask


def refine_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 500,
    close_kernel: int = 5,
    smooth_sigma: float = 1.0,
    guide_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the full refinement stack to a soft mask.

    Steps:
      1. Remove small connected components
      2. Morphological close (fill gaps)
      3. Fill interior holes
      4. Optional guided filter (if guide_image provided)
      5. Smooth edges
    """
    mask = remove_small_components(mask, min_area)
    mask = morphological_close(mask, close_kernel)
    mask = fill_holes(mask)
    if guide_image is not None:
        mask = guided_filter_refine(mask, guide_image)
    if smooth_sigma > 0:
        mask = smooth_edges(mask, sigma=smooth_sigma)
    return mask.clip(0.0, 1.0)
