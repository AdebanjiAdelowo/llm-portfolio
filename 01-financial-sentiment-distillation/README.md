# Financial Sentiment Distillation

**Author:** Adebanji Oluwatimileyin Adelowo  
**Domain:** FinTech / NLP / Model Compression

---

## Overview

This project implements a full model compression pipeline on **FinBERT** (`ProsusAI/finbert`), a BERT-based model pre-trained on financial text for 3-class sentiment classification (positive / negative / neutral).

The goal is to produce a significantly smaller and faster model without sacrificing accuracy, using two complementary techniques:

1. **Structural Pruning** — remove unimportant attention heads and entire encoder layers
2. **Knowledge Distillation** — train the pruned model to mimic the original's soft output distributions

### Why this matters
Deploying 110M-parameter transformer models in production is expensive. This pipeline shows how to compress a model by ~20% in parameter count while **recovering accuracy above the original baseline** through distillation — a technique used by major AI labs (Microsoft, Google, NVIDIA) in building model families.

---

## Pipeline

```
FinBERT (110M params, baseline)
        │
        ▼
  [Notebook 02] Taylor-Gradient Head Pruning
        │   Importance score: |W × ∇W L|
        │   Zeroes output projection of least-important heads
        ▼
  [Notebook 02] Activation-Norm Layer Dropping
        │   Importance score: mean ||hidden state|| per layer
        │   Removes 3 least-important encoder layers
        ▼
  Pruned Student (88M params)
        │
        ▼
  [Notebook 03] Knowledge Distillation
        │   Loss: α·T²·KL(student‖teacher) + (1-α)·CE(student, y)
        │   T=3, α=0.7, teacher frozen
        ▼
  Distilled Student (88M params, accuracy recovered)
```

---

## Results

| Model | Accuracy | Weighted F1 | Params | Size |
|-------|----------|-------------|--------|------|
| Baseline FinBERT | 84.5% | 0.843 | 110M | ~420 MB |
| Head Pruned | ~80% | ~0.79 | 102M | ~390 MB |
| Layer Dropped | ~76% | ~0.75 | 88M | ~337 MB |
| **Distilled Student** | **96.9%** | **0.969** | **88M** | **337 MB** |

> The distilled model **outperforms the original baseline** — soft targets from the teacher provide richer training signal than one-hot labels alone.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Baseline_Evaluation.ipynb` | Load FinBERT, evaluate on `financial_phrasebank` (all_agree split), save baseline metrics and data splits |
| `02_Model_Pruning.ipynb` | Taylor-gradient attention head pruning + activation-norm layer dropping, save pruned student model |
| `03_Knowledge_Distillation.ipynb` | Pre-compute teacher logits, run KL+CE distillation training, benchmark and save distilled model |

**Run in order** — each notebook saves artifacts loaded by the next.

---

## Dataset

**financial_phrasebank** (`sentences_allagree` split)  
- ~2,264 financial sentences labelled by domain experts  
- Labels: `positive` (0), `negative` (1), `neutral` (2)  
- Split: 80% train / 20% test

---

## Key Concepts

**Taylor-Gradient Importance (Head Pruning)**  
Approximates the loss increase caused by removing a parameter:  
`importance(W) = |W × ∇_W L|`  
Heads with the lowest scores have their output projection zeroed.

**Activation-Norm Importance (Layer Dropping)**  
Layers whose hidden states have consistently small norms contribute little to the representation — these are removed entirely.

**Knowledge Distillation Loss**  
`L = α · T² · KL(σ(z_s/T) ‖ σ(z_t/T)) + (1−α) · CE(z_s, y)`  
- T=3 softens the teacher distribution, revealing inter-class relationships  
- α=0.7 weights soft loss higher than hard labels  
- Teacher logits are cached before training to eliminate teacher forward passes from the loop

---

## Requirements

```bash
pip install -r requirements.txt
```

A GPU (e.g. Colab T4) is recommended for Notebook 3. Notebooks 1 and 2 run comfortably on CPU.

---

## Project Structure

```
01-financial-sentiment-distillation/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_Baseline_Evaluation.ipynb
    ├── 02_Model_Pruning.ipynb
    └── 03_Knowledge_Distillation.ipynb
```

Artifacts saved at runtime (gitignored):
- `baseline_results.json`, `pruning_results.json`, `kd_results.json`
- `data_splits.pkl`
- `student_model/`, `distilled_model/`
- PNG charts: confusion matrix, head/layer importance, training curves, pipeline comparison
