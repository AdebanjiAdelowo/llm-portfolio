# 07 — Efficient LLM Fine-Tuning with LoRA & QLoRA

**Author:** Adebanji Oluwatimileyin Adelowo

## Overview
A hands-on implementation of parameter-efficient fine-tuning (PEFT) techniques — LoRA and QLoRA — applied to large language models. Demonstrates how to fine-tune billion-parameter models on consumer hardware by training only a small fraction of parameters, with and without 4-bit quantization.

## Key Skills Demonstrated
- LoRA: low-rank adapter injection, rank selection (`r`), target module configuration
- QLoRA: 4-bit NF4 quantization with double quantization to reduce GPU memory footprint
- PEFT library integration with Hugging Face Transformers
- Before/after inference comparison to validate fine-tuning impact
- Parameter-efficient training: <1% of model weights trained

## Tech Stack
| Component | Library |
|---|---|
| Base models | Llama-3.2-1B (LoRA), Meta-Llama-3-8B (QLoRA) |
| Fine-tuning | PEFT (LoRA adapters) |
| Quantization | BitsAndBytes (4-bit NF4) |
| Training | Hugging Face Trainer / SFTTrainer (trl) |
| Dataset | `fka/awesome-chatgpt-prompts` |

## Notebooks
| Notebook | Description |
|---|---|
| [lora_tuning.ipynb](notebooks/lora_tuning.ipynb) | LoRA fine-tuning of Llama-3.2-1B — full-precision adapter training |
| [qlora_tuning.ipynb](notebooks/qlora_tuning.ipynb) | QLoRA fine-tuning of Llama-3-8B — 4-bit quantized model + LoRA adapters |

## Setup
```bash
pip install -r requirements.txt
```

Requires a Hugging Face account with access to Meta-Llama models. Set your token:
```
HUGGINGFACE_TOKEN=your-hf-token
```

## Key Concepts

### LoRA
Injects trainable low-rank matrices `A` and `B` alongside frozen weight matrix `W`:
```
W' = W + α · (A · B)   where A ∈ R^(d×r), B ∈ R^(r×k), r << d,k
```
Only `A` and `B` are trained — typically <1% of total parameters.

### QLoRA
Extends LoRA by first quantizing `W` to 4-bit NF4 format, reducing GPU memory by ~4×, then applying LoRA adapters in full precision. Enables fine-tuning 7B+ models on a single 16GB GPU.
