# Product Image Segmentation — Dev Notes

## Project goal
Portfolio project comparing deep learning approaches for product background removal.
Target audience: AI Engineer / CV roles in e-commerce, eyewear, or generative AI.

## Running things

```bash
# Install
pip install -r requirements-dev.txt

# Single image
python scripts/run_inference.py --input product.jpg --model birefnet

# Evaluate models
python scripts/evaluate.py --image-dir data/images --mask-dir data/masks --models birefnet u2net

# API
uvicorn api.main:app --reload

# Tests
pytest tests/ -v
```

## Model status
- BiRefNet: loads from HuggingFace — no setup needed
- U2Net: requires weights/u2net.pth (run scripts/download_weights.sh)
- MODNet: requires weights/modnet_photographic_portrait_matting.ckpt (manual download)
- SAM2: stretch goal, not yet wired up

## Architecture decisions
- All models share BaseSegmentationModel interface — predict(image) → float32 mask
- Pipeline handles load-once / call-many lifecycle
- Mask refinement is optional (--no-refine flag) to isolate raw model quality
- FastAPI caches loaded models in a module-level dict to avoid reloading per request

## Data conventions
- Images: RGB uint8 numpy arrays
- Masks: float32 [0, 1] arrays, 1.0 = foreground
- Ground truth masks: white foreground on black background (standard for DIS5K / matting datasets)
- Image/mask pairs matched by filename stem
