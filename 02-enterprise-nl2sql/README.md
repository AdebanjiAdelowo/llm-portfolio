# Enterprise NL2SQL Pipeline

**Author:** Adebanji Oluwatimileyin Adelowo  
**Domain:** Data Engineering / LLM Application Development

---

## Overview

A production-style **Natural Language to SQL** system designed for large enterprise databases. Rather than blindly feeding every table schema to the model on each query (expensive and inaccurate), this pipeline uses a two-stage architecture:

1. **Stage 1 — Table Selector:** A lightweight model reads short table descriptions and returns only the tables needed to answer the question
2. **Stage 2 — SQL Generator:** A capable SQL model receives a focused prompt containing only the relevant schemas, sample rows, and few-shot examples

An additional **complexity router** automatically escalates ambiguous or multi-table queries from GPT-3.5-turbo to GPT-4o-mini.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│  Stage 1: Table Selector        │
│  Model: GPT-3.5-turbo           │
│  Input: question + table list   │
│  Output: JSON {"tables": [...]} │
└─────────────────────────────────┘
      │  selected tables
      ▼
┌─────────────────────────────────┐
│  Prompt Builder                 │
│  Assembles: CREATE TABLE stmts  │
│           + 3 sample rows       │
│           + few-shot SQL        │
└─────────────────────────────────┘
      │  focused prompt
      ▼
┌─────────────────────────────────┐
│  Complexity Router              │
│  Simple  → GPT-3.5-turbo        │
│  Complex → GPT-4o-mini          │
│  (based on question length,     │
│   table count, prompt size)     │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Stage 2: SQL Generator         │
│  Output: valid SQLite query     │
└─────────────────────────────────┘
      │
      ▼
  SQL Query + metadata
```

---

## Results

| Metric | Value |
|--------|-------|
| Average token reduction vs. full schema | ~60% |
| Retry logic | up to 3 attempts with backoff |
| SQL injection guard | blocks destructive keywords |
| Model routing | automatic fast/strong selection |

### Token reduction by query type

| Query | Tables selected | Prompt (words) | Reduction |
|-------|----------------|----------------|-----------|
| Employee headcount by dept | 2 | ~180 | 67% |
| Highest-paid graduates | 3 | ~270 | 51% |
| Most hours worked | 2 | ~200 | 64% |
| London office high performers | 3 | ~280 | 49% |
| Bonus per department | 3 | ~260 | 53% |

---

## Database Schema

An 8-table enterprise HR/company database:

| Table | Description |
|-------|-------------|
| `employees` | Personal data, job title, hire date |
| `salary` | Annual base salary and bonus per employee |
| `studies` | Educational background |
| `departments` | Name, manager, budget |
| `projects` | Company projects with timelines and budgets |
| `employee_projects` | Project assignments, roles, hours worked |
| `performance_reviews` | Annual review scores |
| `offices` | Office locations and capacity |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Table_Selector.ipynb` | Lightweight table-selection model using GPT-3.5; outputs structured JSON; tested against 8-table schema |
| `02_SQL_Generator.ipynb` | Schema-rich prompt construction; Stage 2 SQL generation; prompt size reduction analysis |
| `03_NL2SQL_Pipeline.ipynb` | Full `obtainSQL(question, model)` API; complexity routing; `obtainSQL_with_retry()`; `safe_execute()` guard |

---

## Requirements

```bash
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Key Design Decisions

- **Two-stage over single-stage:** Separating table selection from SQL generation allows each model to be optimised for its specific task and keeps prompts focused
- **Few-shot examples per table:** Tables with complex query patterns (e.g. `salary`) include worked SQL examples directly in the schema block
- **Complexity scoring:** Question word count, number of selected tables, and prompt size together determine which model tier to use
- **Safety guard:** A denylist of destructive SQL keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`) prevents the generated query from being executed if it mutates data

---

## Project Structure

```
02-enterprise-nl2sql/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_Table_Selector.ipynb
    ├── 02_SQL_Generator.ipynb
    └── 03_NL2SQL_Pipeline.ipynb
```
