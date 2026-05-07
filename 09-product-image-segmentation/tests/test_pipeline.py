"""Unit tests for the inference pipeline and postprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.io import save_mask, save_rgba, load_mask
from src.postprocess.mask_refine import (
    binarize,
    fill_holes,
    remove_small_components,
    smooth_edges,
    morphological_close,
    refine_mask,
)
from src.utils.visualize import make_composite, overlay_mask
from src.preprocess.image_ops import (
    resize_with_padding,
    undo_padding,
    normalize,
    to_tensor,
    preprocess_for_inference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image() -> np.ndarray:
    """100x100 RGB image with a white square on black."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 200
    return img


@pytest.fixture
def dummy_mask() -> np.ndarray:
    """Float32 mask matching the white square region."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_resize_with_padding_shape(self, dummy_image):
        out, meta = resize_with_padding(dummy_image, (64, 64))
        assert out.shape == (64, 64, 3)

    def test_undo_padding_restores_size(self, dummy_mask):
        resized, meta = resize_with_padding(dummy_mask[:, :, np.newaxis], (64, 64))
        restored = undo_padding(resized[:, :, 0], meta)
        assert restored.shape == dummy_mask.shape

    def test_normalize_range(self, dummy_image):
        norm = normalize(dummy_image)
        assert norm.dtype == np.float32

    def test_to_tensor_shape(self, dummy_image):
        norm = normalize(dummy_image)
        tensor = to_tensor(norm)
        assert tensor.shape == (3, 100, 100)

    def test_full_preprocess_pipeline(self, dummy_image):
        tensor, meta = preprocess_for_inference(dummy_image, input_size=(64, 64))
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == np.float32
        assert "original_size" in meta


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

class TestMaskRefinement:
    def test_binarize(self, dummy_mask):
        binary = binarize(dummy_mask)
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 255})

    def test_remove_small_components_keeps_large(self, dummy_mask):
        result = remove_small_components(dummy_mask, min_area=100)
        assert result.max() > 0

    def test_remove_small_components_removes_tiny(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50, 50] = 1.0  # single-pixel component
        result = remove_small_components(mask, min_area=10)
        assert result.max() == 0.0

    def test_fill_holes(self, dummy_mask):
        mask_with_hole = dummy_mask.copy()
        mask_with_hole[48:52, 48:52] = 0.0
        filled = fill_holes(mask_with_hole)
        assert filled[50, 50] > 0.5

    def test_smooth_edges_range(self, dummy_mask):
        smoothed = smooth_edges(dummy_mask)
        assert smoothed.min() >= 0.0
        assert smoothed.max() <= 1.0

    def test_refine_mask_shape(self, dummy_mask):
        refined = refine_mask(dummy_mask)
        assert refined.shape == dummy_mask.shape
        assert refined.dtype == np.float32


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestVisualize:
    def test_make_composite_shape(self, dummy_image, dummy_mask):
        composite = make_composite(dummy_image, dummy_mask)
        assert composite.shape == (100, 100, 3)
        assert composite.dtype == np.uint8

    def test_overlay_mask_shape(self, dummy_image, dummy_mask):
        overlay = overlay_mask(dummy_image, dummy_mask)
        assert overlay.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------

class TestIO:
    def test_save_load_mask(self, dummy_mask, tmp_path):
        path = tmp_path / "mask.png"
        save_mask(dummy_mask, path)
        loaded = load_mask(path)
        assert loaded.shape == dummy_mask.shape
        np.testing.assert_allclose(loaded, dummy_mask, atol=0.01)

    def test_save_rgba(self, dummy_image, dummy_mask, tmp_path):
        path = tmp_path / "rgba.png"
        save_rgba(dummy_image, dummy_mask, path)
        assert path.exists()
        from PIL import Image
        pil = Image.open(path)
        assert pil.mode == "RGBA"
