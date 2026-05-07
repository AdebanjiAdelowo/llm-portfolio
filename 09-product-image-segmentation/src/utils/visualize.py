"""Visualization helpers for masks, composites, and comparison grids."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def make_composite(image: np.ndarray, mask: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Composite foreground onto a solid background using the mask as alpha.

    Returns an RGB uint8 image (useful for JPEG display / model comparison).
    """
    if image.shape[2] == 4:
        image = image[:, :, :3]
    alpha = mask[..., np.newaxis].astype(np.float32)
    bg = np.full_like(image, bg_color, dtype=np.float32)
    fg = image.astype(np.float32)
    out = (fg * alpha + bg * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return out


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw a semi-transparent mask overlay on the original image."""
    overlay = image.copy().astype(np.float32)
    m = (mask >= 0.5).astype(np.float32)[..., np.newaxis]
    color_arr = np.array(color, dtype=np.float32)
    overlay = overlay * (1 - alpha * m) + color_arr * alpha * m
    return overlay.clip(0, 255).astype(np.uint8)


def draw_boundary(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the mask boundary contour on the image."""
    binary = (mask >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = image.copy()
    cv2.drawContours(canvas, contours, -1, color[::-1], thickness)  # RGB→BGR for cv2
    return canvas


def comparison_grid(
    image: np.ndarray,
    predictions: dict[str, np.ndarray],
    gt_mask: Optional[np.ndarray] = None,
    cell_size: int = 256,
) -> np.ndarray:
    """Build a side-by-side comparison grid.

    Columns: original | [gt] | model_1 | model_2 | ...
    """
    def _thumb(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

    def _mask_to_rgb(m: np.ndarray) -> np.ndarray:
        gray = (m * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    cells = [_thumb(image)]
    labels = ["input"]

    if gt_mask is not None:
        cells.append(_thumb(_mask_to_rgb(gt_mask)))
        labels.append("ground truth")

    for name, mask in predictions.items():
        cells.append(_thumb(_mask_to_rgb(mask)))
        labels.append(name)

    row = np.hstack(cells)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    for i, label in enumerate(labels):
        x = i * cell_size + 4
        cv2.putText(row, label, (x, 16), font, scale, (255, 255, 255), thickness + 1)
        cv2.putText(row, label, (x, 16), font, scale, (0, 0, 0), thickness)

    return row


def save_comparison(
    image: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Union[str, Path],
    gt_mask: Optional[np.ndarray] = None,
    cell_size: int = 256,
) -> None:
    """Save a comparison grid to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid = comparison_grid(image, predictions, gt_mask, cell_size)
    # cv2 expects BGR
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def save_side_by_side(
    original: np.ndarray,
    result: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Save original and result as a 2-column image."""
    h = max(original.shape[0], result.shape[0])
    w = original.shape[1] + result.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:original.shape[0], :original.shape[1]] = original[:, :, :3]
    canvas[:result.shape[0], original.shape[1]:] = result[:, :, :3]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
