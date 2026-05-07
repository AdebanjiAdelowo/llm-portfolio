# Product Image Segmentation and Matting Pipeline

> Automated background removal for e-commerce product photography — comparing BiRefNet, U²-Net, and MODNet with a production-ready FastAPI demo.

[![CI](https://github.com/adebanjiadelowo/ProductImageSegmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/adebanjiadelowo/ProductImageSegmentation/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Motivation

Clean, transparent-background product images are a core requirement in e-commerce, eyewear, and lifestyle photography. Manual masking in Photoshop is slow and expensive at scale. This project evaluates state-of-the-art deep learning approaches for fully automated product segmentation and matting — with emphasis on edge quality, soft alpha accuracy, and deployment readiness.

The pipeline accepts a single product image and produces:
- A binary or soft **segmentation mask**
- A transparent-background **RGBA composite** (PNG with alpha channel)
- Quantitative evaluation metrics against ground-truth annotations

---

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────┐
│   Preprocessing      │  resize + letterbox padding + normalize
└────────┬────────────┘
         │
    ┌────▼────┐
    │  Model   │  BiRefNet | U²-Net | MODNet | SAM2 (stretch)
    └────┬────┘
         │  raw soft mask [0, 1]
    ┌────▼──────────────┐
    │  Mask Refinement  │  remove small components → morphological close
    │                   │  → fill holes → guided filter → smooth edges
    └────┬──────────────┘
         │
┌────────▼──────────────┐
│     Outputs            │
│  • mask.png            │  grayscale alpha mask
│  • product_nobg.png    │  RGBA transparent composite
│  • eval/results.json   │  per-model metrics
└───────────────────────┘
```

---

## Models Compared

| Model | Architecture | Input Size | Strengths | Notes |
|-------|-------------|-----------|-----------|-------|
| **BiRefNet** | Transformer (Swin-B) | 1024×1024 | SOTA accuracy, fine edge detail | Loads from HuggingFace, no manual download |
| **U²-Net** | Nested U-Net CNN | 320×320 | Fast, lightweight, salient object | ~176 MB weights, great baseline |
| **U²-Netp** | Lightweight U²-Net | 320×320 | 4 MB, API-friendly | Some quality trade-off vs U²-Net |
| **MODNet** | Mobile + semantic branches | 512×512 | Soft alpha, fine hair/edge detail | Designed for matting, not just segmentation |
| **SAM 2** *(stretch)* | Transformer + memory | Variable | Best generalization, interactive | Requires prompts or auto-mode |

---

## Dataset

### Option A: DIS5K (Recommended)
[Dichotomous Image Segmentation](https://github.com/xuebinqin/DIS) — 5,470 high-resolution images with pixel-accurate masks across diverse object categories including products, animals, and structures.

```
data/
├── images/   # .jpg product images
└── masks/    # corresponding .png binary masks (white=foreground)
```

### Option B: Custom Product Dataset
Collect 50–100 product images from Unsplash or Pexels (CC0 license) and annotate with:
- **CVAT** (free, browser-based): https://cvat.ai
- **LabelMe** (Python + local): `pip install labelme`
- Export as PNG binary masks (white foreground on black background)

### Naming Convention
Image and mask files must share the same stem: `glasses_01.jpg` ↔ `glasses_01.png`.

---

## Evaluation Metrics

| Metric | Type | Direction | What it measures |
|--------|------|-----------|-----------------|
| **IoU** | Segmentation | ↑ higher | Overlap between predicted and GT mask |
| **Dice** | Segmentation | ↑ higher | Harmonic mean of precision/recall |
| **Boundary F1** | Segmentation | ↑ higher | Accuracy at mask edges specifically |
| **MAD** | Matting | ↓ lower | Mean Absolute Difference of soft alpha |
| **MSE** | Matting | ↓ lower | Mean Squared Error of soft alpha |
| **Gradient Error** | Matting | ↓ lower | Detail preservation in alpha map |
| **Connectivity** | Matting | ↓ lower | Topological correctness (no missing regions) |

---

## Results

> Replace the placeholder values below with your actual evaluation output from `python scripts/evaluate.py`.

| Model | IoU ↑ | Dice ↑ | MAD ↓ | BoundaryF1 ↑ | Avg Speed (ms/img) |
|-------|-------|--------|-------|--------------|-------------------|
| BiRefNet | — | — | — | — | — |
| U²-Net | — | — | — | — | — |
| MODNet | — | — | — | — | — |

*Evaluated on [dataset name], [N] images, CPU / GPU: [hardware].*

### Qualitative Examples

| Input | BiRefNet | U²-Net | Ground Truth |
|-------|----------|--------|--------------|
| *(add your own images in outputs/visuals/)* | | | |

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get model weights

```bash
# BiRefNet — no download needed (fetched from HuggingFace on first run)

# U2Net (optional)
bash scripts/download_weights.sh
```

### 3. Run inference on a single image

```bash
python scripts/run_inference.py \
    --input path/to/product.jpg \
    --model birefnet \
    --output outputs/
```

Results are written to:
- `outputs/masks/<stem>_birefnetmodel_mask.png`
- `outputs/composites/<stem>_birefnetmodel.png` (transparent background)

### 4. Run batch inference

```bash
python scripts/run_inference.py \
    --input data/images/ \
    --model birefnet \
    --device cuda          # or mps (Apple Silicon) or cpu
```

### 5. Evaluate against ground truth

```bash
python scripts/evaluate.py \
    --image-dir data/images \
    --mask-dir  data/masks \
    --models    birefnet u2net \
    --save-outputs \
    --output    outputs/eval/results.json
```

Outputs a Markdown table you can paste directly into this README.

---

## FastAPI Demo

```bash
uvicorn api.main:app --reload
```

**Swagger UI:** http://localhost:8000/docs

### Remove background via curl

```bash
# Returns transparent PNG directly
curl -X POST http://localhost:8000/segment \
     -F "file=@product.jpg" \
     -F "model=birefnet" \
     -F "threshold=0.5" \
     --output result_nobg.png

# Returns JSON with file URLs (useful for browser frontends)
curl -X POST http://localhost:8000/segment/json \
     -F "file=@product.jpg" \
     -F "model=birefnet"
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","loaded_models":["birefnet"]}
```

---

## Project Structure

```
ProductImageSegmentation/
├── src/
│   ├── data/           # I/O: load_image, load_mask, save_rgba, collect_pairs
│   ├── preprocess/     # Resize, padding, normalization, tensor conversion
│   ├── models/         # BiRefNet, U2Net, MODNet, SAM2 wrappers (common interface)
│   ├── inference/      # SegmentationPipeline: single image + batch
│   ├── postprocess/    # Morphology, hole-filling, guided filter, edge smoothing
│   ├── evaluation/     # IoU, Dice, MAD, gradient error, connectivity error
│   └── utils/          # Visualization: overlays, comparison grids, composites
├── api/
│   ├── main.py         # FastAPI app with /segment and /segment/json endpoints
│   └── schemas.py      # Pydantic request/response models
├── scripts/
│   ├── run_inference.py   # CLI: single image or batch
│   ├── evaluate.py        # CLI: model comparison with metrics table
│   └── download_weights.sh
├── configs/
│   ├── default.yaml    # Pipeline and postprocessing config
│   └── models.yaml     # Per-model hyperparameters and weight paths
├── tests/              # pytest suite: preprocessing, metrics, API (mocked)
├── notebooks/          # Exploratory notebooks for data and model analysis
├── outputs/            # masks/, composites/, eval/, visuals/
└── weights/            # Model checkpoints (gitignored)
```

---

## Implementation Plan

| Phase | Tasks | Status |
|-------|-------|--------|
| **1 — Scaffold** | Project structure, base model interface, I/O | ✅ |
| **2 — Baseline** | BiRefNet integration, single-image pipeline | ⬜ |
| **3 — Postprocessing** | Mask refinement, RGBA composite export | ⬜ |
| **4 — Evaluation** | Metrics suite, evaluation script, results table | ⬜ |
| **5 — Model 2** | U²-Net integration, compare vs BiRefNet | ⬜ |
| **6 — Model 3** | MODNet integration, soft alpha comparison | ⬜ |
| **7 — API** | FastAPI endpoint, test with curl | ⬜ |
| **8 — Tests** | pytest suite, CI pipeline | ⬜ |
| **9 — Visuals** | README images, comparison grids | ⬜ |
| **10 — Stretch** | SAM2 auto mode, fine-tuning on eyewear subset | ⬜ |

---

## Limitations

- **CPU inference is slow** — BiRefNet at 1024×1024 takes ~5–15s per image on CPU. Use a GPU or switch to U²-Netp for latency-sensitive use cases.
- **No fine-tuning** — all models are used zero-shot. A domain-specific eyewear dataset would close the gap on glasses reflections and transparent lenses.
- **Transparent objects** — all methods struggle with glass lenses, clear packaging, and reflective surfaces where the foreground and background blend.
- **No video support** — SAM2's video memory module is not wired up in this pipeline.

---

## Future Work

- [ ] Fine-tune BiRefNet on an eyewear/glasses dataset
- [ ] Add SAM2 auto-mode as a fourth comparison model
- [ ] Build a minimal Gradio or Streamlit frontend for visual demos
- [ ] Add ONNX export for portable deployment
- [ ] Benchmark on GPU and add speed vs. quality Pareto chart
- [ ] Explore test-time augmentation (TTA) for mask stability

---

## References

- [BiRefNet](https://github.com/zhengpeng7/BiRefNet) — Zheng et al., 2024
- [U²-Net](https://github.com/xuebinqin/U-2-Net) — Qin et al., 2020
- [MODNet](https://github.com/ZHKKKe/MODNet) — Ke et al., 2022
- [SAM 2](https://github.com/facebookresearch/sam2) — Ravi et al., 2024
- [DIS5K Dataset](https://github.com/xuebinqin/DIS) — Xin et al., 2022

---

## License

MIT License — see [LICENSE](LICENSE) for details.
