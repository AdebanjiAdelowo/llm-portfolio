# 04 — LLM-Powered Data Analyst Agent

**Author:** Adebanji Oluwatimileyin Adelowo

## Overview
An autonomous data analyst agent built with LangChain and OpenAI that can analyze CSV/Excel datasets through natural language. The agent identifies relationships between variables, cleans data, selects appropriate models, and generates visualizations — all from plain English instructions.

## Key Skills Demonstrated
- LangChain agent construction with `create_pandas_dataframe_agent`
- Agentic reasoning: multi-step planning, tool selection, code execution
- Comparing completion vs chat model behavior for agent tasks
- Data analysis and forecasting via LLM-driven Python code generation

## Tech Stack
| Component | Library |
|---|---|
| LLM | OpenAI GPT (via LangChain) |
| Agent framework | LangChain Experimental |
| Data handling | Pandas |
| Dataset | Kaggle Climate Insights |

## Notebooks
| Notebook | Description |
|---|---|
| [data_analyst_agent.ipynb](notebooks/data_analyst_agent.ipynb) | Full agent pipeline: load data → ask questions → generate charts → forecast |

## Setup
```bash
pip install -r requirements.txt
```

Set your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## What the Agent Can Do
- Describe and summarize a dataset in natural language
- Identify correlations and patterns
- Generate charts (line graphs, bar charts) on demand
- Select and run forecasting models (e.g. ARIMA) for future predictions
