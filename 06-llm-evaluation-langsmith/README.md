# 06 — LLM Evaluation with LangSmith & Embedding Distance

**Author:** Adebanji Oluwatimileyin Adelowo

## Overview
A systematic evaluation framework comparing LLM summarization quality using LangSmith tracing and embedding-based distance metrics. Benchmarks a base T5 model, a fine-tuned T5 variant, and OpenAI GPT on the CNN/DailyMail news summarization dataset.

## Key Skills Demonstrated
- LangSmith dataset creation and evaluation pipelines
- Embedding distance as an automated evaluation metric (no human labels needed)
- Comparing open-source (Hugging Face) vs proprietary (OpenAI) model quality
- Tracing and observability for LLM applications
- Cost-quality tradeoff analysis across model tiers

## Tech Stack
| Component | Library |
|---|---|
| Evaluation framework | LangSmith |
| LLM orchestration | LangChain |
| Open-source models | T5-base, T5-CNN-DM (Hugging Face) |
| Proprietary model | OpenAI GPT |
| Dataset | CNN/DailyMail 3.0.0 |
| Metric | Cosine embedding distance |

## Notebooks
| Notebook | Description |
|---|---|
| [llm_evaluation_langsmith.ipynb](notebooks/llm_evaluation_langsmith.ipynb) | Full evaluation pipeline: dataset creation → model inference → embedding distance scoring → comparison |

## Setup
```bash
pip install -r requirements.txt
```

Set your API keys in a `.env` file:
```
OPENAI_API_KEY=your-openai-key
LANGCHAIN_API_KEY=your-langsmith-key
HUGGINGFACEHUB_API_TOKEN=your-hf-key
```

## Results Summary
| Model | Embedding Distance (lower = better) |
|---|---|
| T5-base (zero-shot) | Higher |
| T5-CNN-DM (fine-tuned) | Lower |
| OpenAI GPT | Lowest |

Fine-tuning on the target domain significantly closes the gap with the proprietary model.
