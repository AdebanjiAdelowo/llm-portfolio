# Results: Diffusion LoRA Fine-Tuning on Product Images

*Fill this template in after running your experiment. Replace all [PLACEHOLDER] blocks with your actual values.*

---

## 1. Experiment Setup

| Parameter | Value |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Training images | [N] (train) + [M] (val) |
| Training steps | [N] steps |
| Batch size | [N] (effective [N × grad_accum]) |
| Learning rate | [1e-4] with cosine decay |
| Resolution | 1024 × 1024 |
| GPU | [T4 / A100] |
| Training time | [~X hours] |
| LoRA checkpoint size | [~X MB] (vs [~6 GB] full model) |

---

## 2. Prompts Used for Evaluation

All inference runs used the same seed (42) to make base vs. LoRA comparisons valid.

| # | Prompt |
|---|---|
| P1 | `a product photo of sks eyewear on white background, studio lighting, sharp focus` |
| P2 | `a product photo of sks eyewear, side profile, minimal background` |
| P3 | `sks eyewear displayed on a clean surface, editorial photography` |
| P4 | `close-up of sks eyewear temple detail, macro photography` |
| P5 | `sks eyewear on a wooden table, natural light, lifestyle shot` |

Negative prompt used across all runs:  
`blurry, low quality, distorted, cartoon, painting, illustration, bokeh, grainy`

---

## 3. Quantitative Results

### CLIP Score (higher = better prompt-image alignment)

| Model | P1 | P2 | P3 | P4 | P5 | Mean |
|---|---|---|---|---|---|---|
| Base SDXL | [0.28] | [0.26] | [0.27] | [0.24] | [0.29] | [0.27] |
| LoRA fine-tuned | [0.34] | [0.31] | [0.33] | [0.30] | [0.32] | [0.32] |
| Delta | [+0.06] | [+0.05] | [+0.06] | [+0.06] | [+0.03] | [+0.05] |

*CLIP score computed using `open_clip` with ViT-B/32. Higher is better; human-perceived quality does not always track CLIP score linearly.*

### Style Consistency (manual, 1–5 scale)

Rate across 4 generated images per prompt: 1 = no consistency, 5 = identical style.

| Criterion | Base SDXL | LoRA fine-tuned |
|---|---|---|
| Background consistency | [2] | [4] |
| Product silhouette accuracy | [2] | [4] |
| Lighting style match | [3] | [4] |
| Trigger-word adherence | N/A | [4] |
| Overall visual coherence | [2] | [4] |

---

## 4. Qualitative Results — Sample Grid

Place your generated image grid here. In a GitHub README, use:

```markdown
![Comparison grid](results/comparison_grid.png)
```

**Grid layout:**  
- Rows = evaluation prompts (P1–P4)  
- Left 4 columns = base SDXL outputs  
- Right 4 columns = LoRA fine-tuned outputs  

**What to highlight for a recruiter:**
- The base model generates generic eyewear shapes; the LoRA model generates shapes consistent with the training domain.
- Background became consistently white/clean after fine-tuning.
- Trigger word reliably steers generation after [N] steps.

---

## 5. Training Curve

![Training loss](results/training_loss.png)

*[Describe the loss curve: did it converge smoothly? any plateaus or spikes? what step did it stabilize?]*

Key observations:
- Loss plateaued around step [N], suggesting [over/under]-fitting at this dataset size.
- Validation images at step [N] showed first signs of domain adaptation.
- No significant validation degradation observed (no catastrophic forgetting of base capabilities).

---

## 6. Failure Cases

Honest failure analysis demonstrates engineering maturity to recruiters.

| Failure mode | Example prompt | Root cause | Mitigation attempted |
|---|---|---|---|
| Over-fitted silhouette | `eyewear on a person's face` | 90% of training images show product alone | Add lifestyle images to dataset |
| Trigger word bleed | Generic photo prompt without `sks` | Token binding too strong | Lower LoRA alpha or train fewer steps |
| Color drift | `sks eyewear in red frames` | Most training images are black/gold | Diversify color distribution in dataset |
| Blur at edges | Temple arm details in P4 | Low resolution detail in training data | Pre-filter for sharpness score |

---

## 7. Compute & Constraints

| Constraint | Impact |
|---|---|
| Free-tier T4 (16GB VRAM) | Limited batch size to 1; added grad accumulation to compensate |
| Dataset size (~100 images) | Model memorizes rather than generalizes; style is consistent but not diverse |
| No hyperparameter sweep | Rank=16, lr=1e-4 chosen from literature defaults; not tuned for this domain |
| Single run | No confidence intervals; results are directional, not statistically rigorous |

**Honest framing for recruiters:**  
This project demonstrates end-to-end proficiency with diffusion fine-tuning tooling, curation pipelines, and evaluation methodology. The results are limited by compute and data scale, which are documented explicitly. A production deployment would require 500–2000 images, compute budget for a sweep, and a robust perceptual metric (FID).

---

## 8. Potential Next Steps

- [ ] Expand dataset to 300+ images with diverse product angles and colors
- [ ] Sweep LoRA rank (4, 8, 16, 32) and report CLIP score vs. param count trade-off
- [ ] Fine-tune text encoder alongside U-Net for stronger trigger-word binding
- [ ] Compute FID against a held-out reference set using `clean-fid`
- [ ] Serve LoRA via ComfyUI workflow for non-technical product team use
