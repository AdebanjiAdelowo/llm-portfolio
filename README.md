# LLM Portfolio Projects

**Author:** Adebanji Oluwatimileyin Adelowo  
**GitHub:** [AdebanjiAdelowo](https://github.com/AdebanjiAdelowo)  
**Email:** adelowooluwatimileyin@gmail.com

---

A collection of end-to-end Large Language Model projects covering model optimization, NL2SQL pipelines, embedding-based systems, LangChain agents, RAG, evaluation frameworks, and parameter-efficient fine-tuning.

## Projects

| # | Project | Domain | Key Techniques |
|---|---------|--------|----------------|
| 01 | [Financial Sentiment Distillation](#01-financial-sentiment-distillation) | FinTech / NLP | Pruning, Knowledge Distillation |
| 02 | [Enterprise NL2SQL Pipeline](#02-enterprise-nl2sql-pipeline) | Data Engineering | Multi-stage LLM routing, Prompt Engineering |
| 03 | [Bank Customer Risk Engine](#03-bank-customer-risk-engine) | FinTech / ML | Embeddings, XGBoost, Multi-output Regression |
| 04 | [LLM-Powered Data Analyst Agent](#04-llm-powered-data-analyst-agent) | Analytics | LangChain Agents, Pandas DataFrame Agent |
| 05 | [Medical RAG Assistant](#05-medical-rag-assistant) | Healthcare / NLP | RAG, ChromaDB, Conversational Memory |
| 06 | [LLM Evaluation with LangSmith](#06-llm-evaluation-with-langsmith) | MLOps / Evaluation | Embedding Distance, LangSmith Tracing |
| 07 | [LoRA & QLoRA Fine-Tuning](#07-lora--qlora-fine-tuning) | Model Training | PEFT, LoRA, 4-bit Quantization |

---

## 01 Financial Sentiment Distillation

**Folder:** `01-financial-sentiment-distillation/`

A full model compression pipeline applied to FinBERT — a BERT-based financial sentiment classifier. Starting from the 110M-parameter baseline, the model is first pruned using Taylor-gradient importance scoring, then a knowledge distillation step recovers the accuracy lost during pruning.

**Pipeline:**
```
FinBERT (baseline) → Attention Head Pruning → Layer Dropping → Knowledge Distillation → Distilled Student
```

**Results:**

| Model | Accuracy | F1 | Params (M) | Size (MB) |
|-------|----------|----|-----------|-----------|
| Baseline FinBERT | 84.5% | 0.843 | 110M | ~420 MB |
| After Pruning | ~76% | ~0.75 | 88M | ~337 MB |
| Distilled Student | **96.9%** | **0.969** | 88M | 337 MB |

**Tech stack:** PyTorch, HuggingFace Transformers, financial_phrasebank dataset, scikit-learn

---

## 02 Enterprise NL2SQL Pipeline

**Folder:** `02-enterprise-nl2sql/`

A two-stage LLM pipeline that translates natural language business questions into SQL queries against an 8-table enterprise HR/company database. A lightweight model first identifies which tables are needed; a stronger model then generates accurate SQL from a focused prompt.

**Architecture:**
```
User Question → [Stage 1: Table Selector (GPT-3.5)] → Selected Tables
             → [Stage 2: SQL Generator (GPT-3.5 / GPT-4o-mini)] → SQL Query
```

**Results:** ~60% token reduction by selecting only relevant tables; automatic model routing.

**Tech stack:** OpenAI API, pandas, matplotlib

---

## 03 Bank Customer Risk Engine

**Folder:** `03-bank-customer-risk-engine/`

An embedding-powered risk assessment engine for a retail bank. Customer profiles and transactions are encoded into dense vectors, then used to train a multi-output regression model predicting loan approval probability, default risk, and interest rate.

**Tech stack:** sentence-transformers, UMAP, XGBoost, scikit-learn

---

## 04 LLM-Powered Data Analyst Agent

**Folder:** `04-data-analyst-agent/`

An autonomous data analyst agent built with LangChain and OpenAI that analyzes datasets through natural language. The agent identifies correlations, generates charts, selects forecasting models, and produces predictions — all from plain English instructions.

**Key techniques:** LangChain `create_pandas_dataframe_agent`, agentic reasoning, multi-step tool use, completion vs. chat model comparison.

**Tech stack:** LangChain, LangChain Experimental, OpenAI, Pandas

---

## 05 Medical RAG Assistant

**Folder:** `05-medical-rag-assistant/`

A conversational medical assistant using Retrieval-Augmented Generation (RAG). Embeds a medical Q&A knowledge base into ChromaDB and routes queries through a LangChain ReAct agent — answering clinical questions with grounded, context-aware responses while maintaining conversation memory.

**Architecture:**
```
User Query → ReAct Agent → ChromaDB retrieval (medical) OR direct LLM (general) → Answer
```

**Tech stack:** LangChain, ChromaDB, OpenAI Embeddings, MedQuad dataset

---

## 06 LLM Evaluation with LangSmith

**Folder:** `06-llm-evaluation-langsmith/`

A systematic evaluation framework comparing LLM summarization quality using LangSmith tracing and cosine embedding distance. Benchmarks T5-base, fine-tuned T5, and OpenAI GPT on the CNN/DailyMail dataset — demonstrating the cost-quality tradeoff between open-source and proprietary models.

**Results:** Fine-tuned T5 significantly outperforms zero-shot T5; OpenAI GPT achieves lowest embedding distance.

**Tech stack:** LangSmith, LangChain, Hugging Face (T5), OpenAI, CNN/DailyMail dataset

---

## 07 LoRA & QLoRA Fine-Tuning

**Folder:** `07-lora-qlora-finetuning/`

Parameter-efficient fine-tuning of large language models using LoRA and QLoRA. Demonstrates how to fine-tune billion-parameter models (Llama-3.2-1B, Llama-3-8B) training fewer than 1% of total parameters, with and without 4-bit quantization to enable training on consumer hardware.

**Key concepts:**
- **LoRA:** injects trainable low-rank matrices alongside frozen weights (`W' = W + α·AB`)
- **QLoRA:** 4-bit NF4 quantization + LoRA — fine-tune 8B models on a single 16GB GPU

**Tech stack:** PEFT, BitsAndBytes, Transformers, trl, Llama 3

---

## Getting Started

### Prerequisites
- Python 3.10+
- GPU recommended (Google Colab T4/L4 works for all projects)

### Installation
```bash
git clone https://github.com/AdebanjiAdelowo/llm-portfolio.git
cd llm-portfolio

# Install dependencies for a specific project
pip install -r 04-data-analyst-agent/requirements.txt
```

### Environment Variables
Create a `.env` file in each project folder as needed:
```
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langsmith_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

---

## Skills Demonstrated

- **Agents & RAG** — LangChain agents, ChromaDB vector stores, ReAct reasoning, tool routing
- **LLM Evaluation** — LangSmith tracing, embedding distance metrics, model benchmarking
- **Fine-tuning & PEFT** — LoRA, QLoRA, 4-bit quantization, SFT training
- **Model compression** — structured pruning, knowledge distillation
- **LLM pipelines** — multi-stage routing, prompt engineering, NL2SQL
- **Embeddings** — semantic similarity, UMAP, nearest-neighbour retrieval, XGBoost

---

## License

MIT License — free to use, adapt, and build upon with attribution.
