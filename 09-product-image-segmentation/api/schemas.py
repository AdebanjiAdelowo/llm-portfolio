"""Pydantic request/response schemas for the segmentation API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model: Literal["birefnet", "u2net"] = Field(
        default="birefnet",
        description="Which model to use for segmentation.",
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Binarization threshold for the alpha mask.",
    )
    refine: bool = Field(
        default=True,
        description="Apply morphological refinement to the mask.",
    )
    return_mask: bool = Field(
        default=False,
        description="If true, also return the raw mask image.",
    )


class InferenceResponse(BaseModel):
    model: str
    width: int
    height: int
    processing_ms: float
    rgba_url: str               # URL path to the transparent-background PNG
    mask_url: Optional[str]     # URL path to the mask PNG (if requested)
