# LLM Portfolio Projects

**Author:** Adebanji Oluwatimileyin Adelowo  
**GitHub:** [adebanjiadelowo](https://github.com/adebanjiadelowo)  
**Email:** adelowooluwatimileyin@gmail.com

---

A collection of end-to-end Large Language Model projects covering model optimization, NL2SQL pipelines, and embedding-based financial intelligence systems.

## Projects

| # | Project | Domain | Key Techniques | Results |
|---|---------|--------|----------------|---------|
| 01 | [Financial Sentiment Distillation](#01-financial-sentiment-distillation) | FinTech / NLP | Pruning, Knowledge Distillation | 96.9% accuracy, 62% parameter reduction |
| 02 | [Enterprise NL2SQL Pipeline](#02-enterprise-nl2sql-pipeline) | Data Engineering | Multi-stage LLM routing, Prompt Engineering | ~60% token cost reduction |
| 03 | [Bank Customer Risk Engine](#03-bank-customer-risk-engine) | FinTech / ML | Embeddings, XGBoost, Multi-output Regression | End-to-end risk scoring pipeline |

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

**Notebooks:**
1. `01_Baseline_Evaluation.ipynb` — load FinBERT, benchmark on financial_phrasebank
2. `02_Model_Pruning.ipynb` — Taylor-gradient head pruning + activation-norm layer dropping
3. `03_Knowledge_Distillation.ipynb` — KL divergence soft-loss distillation (T=3, α=0.7)

---

## 02 Enterprise NL2SQL Pipeline

**Folder:** `02-enterprise-nl2sql/`

A two-stage LLM pipeline that translates natural language business questions into SQL queries against an 8-table enterprise HR/company database. A lightweight model first identifies which tables are needed; a stronger model then generates accurate SQL from a focused prompt — dramatically reducing token costs compared to sending the full schema every time.

**Architecture:**
```
User Question → [Stage 1: Table Selector (GPT-3.5)] → Selected Tables
             → [Stage 2: SQL Generator (GPT-3.5 / GPT-4o-mini)] → SQL Query
             → [Complexity Router] → auto-selects fast vs. strong model
```

**Results:**
- Average **60% token reduction** by selecting only relevant tables
- Automatic model routing: simple queries use GPT-3.5, complex ones escalate to GPT-4o-mini
- Full retry logic and SQL injection guard

**Tech stack:** OpenAI API (GPT-3.5-turbo, GPT-4o-mini), pandas, matplotlib

**Notebooks:**
1. `01_Table_Selector.ipynb` — lightweight table-selection model with JSON output
2. `02_SQL_Generator.ipynb` — schema-rich prompt construction + SQL generation
3. `03_NL2SQL_Pipeline.ipynb` — full `obtainSQL()` API with routing, retry, and safety guard

**Requirements:** `OPENAI_API_KEY` in a `.env` file.

---

## 03 Bank Customer Risk Engine

**Folder:** `03-bank-customer-risk-engine/`

An embedding-powered risk assessment and product recommendation engine for a retail bank. Customer profiles and transaction histories are encoded into dense vectors using `all-MiniLM-L6-v2`, then used to train a multi-output regression model that simultaneously predicts loan approval probability, default risk, and adjusted interest rate.

**Architecture:**
```
Customer Text Profile → Sentence Transformer (384-dim) ─┐
                                                         ├─► 768-dim feature vector → XGBoost / Multi-output Regressor
Transaction History  → Sentence Transformer (384-dim) ─┘
```

**Results:**
- End-to-end `risk_decision()` function: text in, risk scores out
- XGBoost classifier for approval decision
- Multi-output regressor: approval probability + default probability + adjusted interest rate
- Cosine similarity search for nearest-neighbour customer profiling

**Tech stack:** sentence-transformers, UMAP, XGBoost, scikit-learn, pandas, matplotlib

**Notebooks:**
1. `01_Client_Embeddings.ipynb` — generate & store 384-dim client embeddings, UMAP visualisation
2. `02_Product_Embeddings.ipynb` — product catalogue + transaction embeddings
3. `03_Risk_Decision_Model.ipynb` — combine embeddings, train models, expose risk API

---

## Getting Started

### Prerequisites
- Python 3.10+
- (Recommended) A GPU runtime — Google Colab T4 works for all projects

### Installation

```bash
git clone https://github.com/adebanjiadelowo/llm-portfolio.git
cd llm-portfolio

# Install dependencies for a specific project
pip install -r 01-financial-sentiment-distillation/requirements.txt
```

### Environment Variables
For the NL2SQL project, create a `.env` file in `02-enterprise-nl2sql/`:
```
OPENAI_API_KEY=your_key_here
```

### Running Order
Each project's notebooks are numbered and must be run in order — later notebooks depend on artifacts saved by earlier ones.

---

## Skills Demonstrated

- **LLM fine-tuning & compression** — pruning, knowledge distillation, model benchmarking
- **LLM application development** — multi-stage pipelines, prompt engineering, API integration
- **Embeddings & vector search** — semantic similarity, UMAP visualisation, nearest-neighbour retrieval
- **ML engineering** — XGBoost, multi-output regression, evaluation metrics (accuracy, F1, BLEU, ROUGE)
- **Production patterns** — retry logic, input validation, model routing, cost optimisation

---

## License

MIT License — free to use, adapt, and build upon with attribution.
