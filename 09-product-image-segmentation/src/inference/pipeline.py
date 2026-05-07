"""End-to-end segmentation + matting inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..data.io import load_image, save_mask, save_rgba
from ..models.base import BaseSegmentationModel
from ..postprocess.mask_refine import refine_mask
from ..utils.visualize import make_composite


@dataclass
class PipelineResult:
    """Container for a single inference result."""

    image: np.ndarray           # Original RGB uint8
    raw_mask: np.ndarray        # Float32 [0,1] direct from model
    refined_mask: np.ndarray    # Float32 [0,1] after postprocessing
    composite: np.ndarray       # RGBA uint8 transparent-background image
    model_name: str = ""
    source_path: Optional[Path] = None


class SegmentationPipeline:
    """Runs a model end-to-end: load → preprocess → infer → postprocess → save.

    Example:
        model = BiRefNetModel(device="cpu")
        pipe = SegmentationPipeline(model, output_dir="outputs")
        result = pipe.run("product.jpg")
    """

    def __init__(
        self,
        model: BaseSegmentationModel,
        output_dir: Union[str, Path] = "outputs",
        mask_threshold: float = 0.5,
        refine: bool = True,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.mask_threshold = mask_threshold
        self.refine = refine
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.model.load()
            self._loaded = True

    def run(
        self,
        image_path: Union[str, Path, np.ndarray],
        save_outputs: bool = True,
        stem: Optional[str] = None,
    ) -> PipelineResult:
        """Process a single image and return a PipelineResult.

        Args:
            image_path: Path to input image or pre-loaded RGB array.
            save_outputs: Write mask + RGBA composite to output_dir.
            stem: Output filename stem (default: input file stem).
        """
        self._ensure_loaded()

        if isinstance(image_path, (str, Path)):
            source = Path(image_path)
            image = load_image(source, mode="RGB")
            stem = stem or source.stem
        else:
            image = image_path
            stem = stem or "output"
            source = None

        raw_mask = self.model.predict(image)

        if self.refine:
            refined = refine_mask(raw_mask, threshold=self.mask_threshold)
        else:
            refined = (raw_mask >= self.mask_threshold).astype(np.float32)

        composite = make_composite(image, refined)

        if save_outputs:
            model_tag = self.model.name.lower()
            mask_path = self.output_dir / "masks" / f"{stem}_{model_tag}_mask.png"
            rgba_path = self.output_dir / "composites" / f"{stem}_{model_tag}.png"
            save_mask(refined, mask_path)
            save_rgba(image, refined, rgba_path)

        return PipelineResult(
            image=image,
            raw_mask=raw_mask,
            refined_mask=refined,
            composite=composite,
            model_name=self.model.name,
            source_path=source,
        )

    def run_batch(
        self,
        image_paths: list[Union[str, Path]],
        save_outputs: bool = True,
    ) -> list[PipelineResult]:
        """Process multiple images sequentially."""
        self._ensure_loaded()
        results = []
        for path in image_paths:
            try:
                results.append(self.run(path, save_outputs=save_outputs))
            except Exception as exc:
                print(f"[WARN] Skipping {path}: {exc}")
        return results
