"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    iou_score,
    dice_score,
    precision_recall,
    mad_score,
    mse_score,
    compute_all_metrics,
    average_metrics,
)


@pytest.fixture
def perfect_mask():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    return mask


class TestMetrics:
    def test_iou_perfect(self, perfect_mask):
        assert iou_score(perfect_mask, perfect_mask) == pytest.approx(1.0, abs=1e-4)

    def test_iou_empty(self):
        pred = np.zeros((50, 50), dtype=np.float32)
        gt = np.zeros((50, 50), dtype=np.float32)
        # Both empty: intersection=0, union=0 → 0/(0+eps) ≈ 0
        assert iou_score(pred, gt) < 1e-3

    def test_iou_no_overlap(self, perfect_mask):
        inverted = 1.0 - perfect_mask
        assert iou_score(perfect_mask, inverted) < 1e-3

    def test_dice_perfect(self, perfect_mask):
        assert dice_score(perfect_mask, perfect_mask) == pytest.approx(1.0, abs=1e-4)

    def test_precision_recall_perfect(self, perfect_mask):
        p, r = precision_recall(perfect_mask, perfect_mask)
        assert p == pytest.approx(1.0, abs=1e-4)
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_mad_perfect(self, perfect_mask):
        assert mad_score(perfect_mask, perfect_mask) == pytest.approx(0.0, abs=1e-6)

    def test_mad_inverted(self, perfect_mask):
        # MAD of mask vs its inverse = fraction of 1-valued pixels
        inverted = 1.0 - perfect_mask
        foreground_fraction = perfect_mask.mean()
        # Each pixel flipped by 1.0, so MAD = fraction of changed pixels
        assert mad_score(perfect_mask, inverted) == pytest.approx(
            np.mean(np.abs(perfect_mask - inverted)), abs=1e-5
        )

    def test_mse_perfect(self, perfect_mask):
        assert mse_score(perfect_mask, perfect_mask) == pytest.approx(0.0, abs=1e-6)

    def test_compute_all_metrics_returns_dataclass(self, perfect_mask):
        metrics = compute_all_metrics(perfect_mask, perfect_mask)
        assert metrics.iou == pytest.approx(1.0, abs=1e-4)
        assert metrics.dice == pytest.approx(1.0, abs=1e-4)
        assert metrics.mad == pytest.approx(0.0, abs=1e-5)

    def test_average_metrics(self, perfect_mask):
        m1 = compute_all_metrics(perfect_mask, perfect_mask)
        m2 = compute_all_metrics(perfect_mask * 0.5, perfect_mask)
        avg = average_metrics([m1, m2])
        assert "iou" in avg
        assert 0.0 < avg["iou"] < 1.0
