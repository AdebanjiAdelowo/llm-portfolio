# 05 — Medical RAG Assistant with LangChain & ChromaDB

**Author:** Adebanji Oluwatimileyin Adelowo

## Overview
A conversational medical assistant that uses Retrieval-Augmented Generation (RAG) to answer clinical questions from a structured medical knowledge base. The system retrieves relevant disease information from ChromaDB and passes it to an LLM, enabling accurate, grounded answers with conversation memory.

## Key Skills Demonstrated
- RAG pipeline: document loading → chunking → embedding → vector store → retrieval
- LangChain agent with custom tools and ReAct reasoning
- ChromaDB for persistent vector storage
- Conversation memory with `ConversationBufferWindowMemory`
- Domain-specific tool routing (medical vs non-medical queries)

## Tech Stack
| Component | Library |
|---|---|
| LLM | OpenAI (via LangChain) |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Agent framework | LangChain ReAct |
| Dataset | MedQuad Medical Q&A (Hugging Face) |

## Notebooks
| Notebook | Description |
|---|---|
| [medical_rag_assistant.ipynb](notebooks/medical_rag_assistant.ipynb) | Full RAG pipeline + conversational agent with medical knowledge base |

## Setup
```bash
pip install -r requirements.txt
```

Set your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Architecture
```
User Query
    │
    ▼
LangChain ReAct Agent
    │
    ├─ Medical query? ──► ChromaDB retrieval ──► LLM answer
    │
    └─ General query? ──► LLM direct answer
         │
         ▼
  Conversation Memory (context-aware follow-ups)
```
