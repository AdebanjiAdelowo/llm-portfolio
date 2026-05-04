# Bank Customer Risk Engine

**Author:** Adebanji Oluwatimileyin Adelowo  
**Domain:** FinTech / Embeddings / Machine Learning

---

## Overview

An embedding-powered **customer risk assessment and product recommendation engine** for retail banking. Instead of storing customer profiles as structured rows, this system encodes them as dense semantic vectors using a sentence transformer model. These embeddings capture nuanced meaning that traditional features miss — enabling similarity search, cluster analysis, and powerful downstream ML models.

The final pipeline takes a plain-text customer description as input and outputs a complete risk decision: loan approval, default probability, and adjusted interest rate.

---

## Architecture

```
Customer Profile (text)
        │
        ▼
  all-MiniLM-L6-v2 Sentence Transformer
        │  384-dimensional embedding
        ▼
        ├──────────────────────────────────┐
        │                                  │
        │                     Transaction History (text)
        │                                  │
        │                     all-MiniLM-L6-v2
        │                                  │  384-dim
        │                                  ▼
        └─────── Concatenate (768-dim feature vector) ──────┐
                                                            │
                              ┌─────────────────────────────┤
                              │                             │
                    XGBoost Classifier            Multi-Output Regressor
                    (approval decision)           ├─ approval_prob
                                                  ├─ default_prob
                                                  └─ adjusted_rate
```

---

## Results

| Capability | Detail |
|-----------|--------|
| Embedding dimension | 384 (client) + 384 (transaction) = 768 combined |
| Customers modelled | 500 synthetic profiles |
| Products catalogue | 10 financial products |
| Transactions | 500 transaction records |
| Classifiers trained | XGBoost + Logistic Regression |
| Regression targets | approval_prob, default_prob, adjusted_rate (simultaneous) |
| Similarity search | Cosine nearest-neighbour over 500 embeddings |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Client_Embeddings.ipynb` | Generate 500 synthetic customer profiles, encode with `all-MiniLM-L6-v2`, save embeddings, UMAP 2D visualisation, cosine similarity search demo |
| `02_Product_Embeddings.ipynb` | 10-product financial catalogue + 500 transaction records, generate product and transaction embeddings, save to disk |
| `03_Risk_Decision_Model.ipynb` | Load and concatenate embeddings (768-dim), train XGBoost classifier + multi-output regressor, expose `risk_decision(customer_text)` end-to-end function |

**Run in order** — Notebook 3 loads pickle files produced by Notebooks 1 and 2.

---

## Sample Output

```python
risk_decision("Sarah is a 34-year-old software engineer with stable employment 
               and no outstanding loans.")

# Returns:
{
  "approval":         "Approved",
  "approval_prob":    0.91,
  "default_prob":     0.04,
  "adjusted_rate":    5.2,
  "similar_customers": [...]
}
```

---

## Key Concepts

**Why embeddings over structured features?**  
Traditional risk models require hand-crafted feature engineering. Sentence embeddings capture semantic relationships directly from text — "software engineer with stable employment" and "senior developer with consistent income" map to nearby points in embedding space without any explicit rules.

**UMAP Visualisation**  
Reduces 384-dim embeddings to 2D for cluster analysis. Reveals natural groupings (e.g., high-income professionals, students, retirees) without any labels.

**Multi-output Regression**  
A single model simultaneously predicts approval probability, default probability, and adjusted interest rate — ensuring internal consistency between outputs that separate models cannot guarantee.

---

## Requirements

```bash
pip install -r requirements.txt
```

No API keys required — all models run locally.

---

## Project Structure

```
03-bank-customer-risk-engine/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_Client_Embeddings.ipynb
    ├── 02_Product_Embeddings.ipynb
    └── 03_Risk_Decision_Model.ipynb
```

Artifacts saved at runtime (gitignored):
- `client_embeddings.pkl`, `transaction_embeddings.pkl`, `product_catalogue_embeddings.pkl`
- UMAP PNG charts, confusion matrices
