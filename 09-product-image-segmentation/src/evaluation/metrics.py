"""Segmentation and matting evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import convolve


@dataclass
class SegmentationMetrics:
    iou: float
    dice: float
    precision: float
    recall: float
    f1: float
    boundary_f1: float
    # Matting-specific
    mad: float          # Mean Absolute Difference (lower is better)
    mse: float          # Mean Squared Error
    grad: float         # Gradient error (detail preservation)
    conn: float         # Connectivity error


def _to_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64)


def iou_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Intersection over Union (Jaccard Index)."""
    p = _to_float(pred) >= threshold
    g = _to_float(gt) >= threshold
    intersection = (p & g).sum()
    union = (p | g).sum()
    return float(intersection / (union + 1e-8))


def dice_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Dice / F1 score on binarized masks."""
    p = _to_float(pred) >= threshold
    g = _to_float(gt) >= threshold
    return float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))


def precision_recall(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    p = _to_float(pred) >= threshold
    g = _to_float(gt) >= threshold
    tp = (p & g).sum()
    precision = float(tp / (p.sum() + 1e-8))
    recall = float(tp / (g.sum() + 1e-8))
    return precision, recall


def boundary_f1(
    pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, dilation: int = 3
) -> float:
    """F1 score computed on boundary pixels (within `dilation` px of edge)."""
    def get_boundary(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = convolve(mask.astype(np.uint8), kernel, mode="constant") > 0
        eroded = convolve((~mask.astype(bool)).astype(np.uint8), kernel, mode="constant") > 0
        return dilated & eroded

    p = (_to_float(pred) >= threshold).astype(bool)
    g = (_to_float(gt) >= threshold).astype(bool)
    bp = get_boundary(p)
    bg = get_boundary(g)

    # Dilate gt boundary for tolerance
    from scipy.ndimage import binary_dilation
    bg_d = binary_dilation(bg, iterations=dilation)
    bp_d = binary_dilation(bp, iterations=dilation)

    prec = float((bp & bg_d).sum() / (bp.sum() + 1e-8))
    rec = float((bg & bp_d).sum() / (bg.sum() + 1e-8))
    return float(2 * prec * rec / (prec + rec + 1e-8))


def mad_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Absolute Difference — primary matting metric."""
    return float(np.mean(np.abs(_to_float(pred) - _to_float(gt))))


def mse_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Squared Error between soft alpha maps."""
    return float(np.mean((_to_float(pred) - _to_float(gt)) ** 2))


def gradient_error(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4) -> float:
    """Gradient-domain error measuring detail preservation (matting-specific).

    Uses Gaussian derivative filters to compare gradient magnitude maps.
    """
    from scipy.ndimage import gaussian_filter

    def grad_mag(x: np.ndarray) -> np.ndarray:
        gx = gaussian_filter(x, sigma, order=(0, 1))
        gy = gaussian_filter(x, sigma, order=(1, 0))
        return np.sqrt(gx ** 2 + gy ** 2)

    return float(np.sum((grad_mag(_to_float(pred)) - grad_mag(_to_float(gt))) ** 2) / 1000.0)


def connectivity_error(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    """Connectivity error — penalizes topological mistakes (extra/missing regions).

    Adapted from the Adobe Composition-1K evaluation protocol.
    """
    from skimage.morphology import label

    thresh_range = np.arange(0, 1 + step, step)
    error = 0.0
    for t in thresh_range:
        pb = (_to_float(pred) >= t).astype(np.uint8)
        gb = (_to_float(gt) >= t).astype(np.uint8)
        _, p_labels = np.unique(label(pb), return_counts=True)
        _, g_labels = np.unique(label(gb), return_counts=True)
        # Penalise difference in connected component counts
        error += abs(len(p_labels) - len(g_labels))
    return float(error * step)


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
) -> SegmentationMetrics:
    """Compute the full metrics suite for one (prediction, ground-truth) pair."""
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

    prec, rec = precision_recall(pred_f, gt_f, threshold)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return SegmentationMetrics(
        iou=iou_score(pred_f, gt_f, threshold),
        dice=dice_score(pred_f, gt_f, threshold),
        precision=prec,
        recall=rec,
        f1=f1,
        boundary_f1=boundary_f1(pred_f, gt_f, threshold),
        mad=mad_score(pred_f, gt_f),
        mse=mse_score(pred_f, gt_f),
        grad=gradient_error(pred_f, gt_f),
        conn=connectivity_error(pred_f, gt_f),
    )


def average_metrics(results: list[SegmentationMetrics]) -> Dict[str, float]:
    """Average a list of per-image metric results."""
    if not results:
        return {}
    fields = SegmentationMetrics.__dataclass_fields__.keys()
    return {f: float(np.mean([getattr(r, f) for r in results])) for f in fields}
