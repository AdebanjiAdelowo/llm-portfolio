"""FastAPI application for product image background removal.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Example curl:
    curl -X POST http://localhost:8000/segment \
         -F "file=@product.jpg" \
         -F "model=birefnet" \
         --output result.png

Swagger docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.data.io import load_image, save_mask, save_rgba
from src.inference.pipeline import SegmentationPipeline, PipelineResult
from src.models.birefnet import BiRefNetModel
from src.models.u2net import U2NetModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Product Image Segmentation API",
    description="Remove backgrounds from product images using BiRefNet or U2Net.",
    version="0.1.0",
)

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)
(OUTPUTS / "composites").mkdir(exist_ok=True)
(OUTPUTS / "masks").mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")

# ---------------------------------------------------------------------------
# Model cache — lazy-loaded on first request, cached per model name
# ---------------------------------------------------------------------------

_pipelines: dict[str, SegmentationPipeline] = {}


def get_pipeline(model_name: str, threshold: float, refine: bool) -> SegmentationPipeline:
    if model_name not in _pipelines:
        if model_name == "birefnet":
            model = BiRefNetModel(device="cpu")
        elif model_name == "u2net":
            model = U2NetModel(device="cpu")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        _pipelines[model_name] = SegmentationPipeline(model, output_dir=str(OUTPUTS), refine=refine)
    pipe = _pipelines[model_name]
    pipe.mask_threshold = threshold
    pipe.refine = refine
    return pipe


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded_models": list(_pipelines.keys())}


@app.post("/segment")
async def segment(
    file: UploadFile = File(..., description="Product image (JPEG or PNG)"),
    model: str = Form(default="birefnet", description="Model: birefnet | u2net"),
    threshold: float = Form(default=0.5, ge=0.0, le=1.0),
    refine: bool = Form(default=True),
    return_mask: bool = Form(default=False),
):
    """Remove background from a product image and return a transparent PNG.

    Returns the RGBA composite directly as a PNG file download.
    Set `return_mask=true` to also get the raw mask in the JSON response.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        image = np.array(pil_img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    stem = Path(file.filename or "upload").stem
    pipeline = get_pipeline(model, threshold, refine)

    t0 = time.perf_counter()
    try:
        result: PipelineResult = pipeline.run(image, save_outputs=True, stem=stem)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    rgba_path = OUTPUTS / "composites" / f"{stem}_{model.lower()}.png"
    mask_path = OUTPUTS / "masks" / f"{stem}_{model.lower()}_mask.png"

    return FileResponse(
        path=str(rgba_path),
        media_type="image/png",
        filename=f"{stem}_nobg.png",
        headers={
            "X-Model": model,
            "X-Processing-Ms": f"{elapsed_ms:.1f}",
            "X-Image-Width": str(result.image.shape[1]),
            "X-Image-Height": str(result.image.shape[0]),
        },
    )


@app.post("/segment/json")
async def segment_json(
    file: UploadFile = File(...),
    model: str = Form(default="birefnet"),
    threshold: float = Form(default=0.5),
    refine: bool = Form(default=True),
):
    """Same as /segment but returns JSON with URL paths instead of the file directly.

    Useful when the caller wants to display results in a browser without
    downloading the file.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    image = np.array(pil_img, dtype=np.uint8)
    stem = Path(file.filename or "upload").stem

    pipeline = get_pipeline(model, threshold, refine)
    t0 = time.perf_counter()
    result = pipeline.run(image, save_outputs=True, stem=stem)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "model": model,
        "width": result.image.shape[1],
        "height": result.image.shape[0],
        "processing_ms": round(elapsed_ms, 1),
        "rgba_url": f"/outputs/composites/{stem}_{model.lower()}.png",
        "mask_url": f"/outputs/masks/{stem}_{model.lower()}_mask.png",
    })
