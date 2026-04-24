# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based learning and exemplars repository covering AI/ML topics: LangChain, LangGraph, RAG pipelines, Hugging Face, Polars/Pandas, Streamlit, and AWS Bedrock. It is organized as a collection of standalone scripts and Jupyter notebooks, not a single deployable application.

## Environment Setup

Dependencies are managed with pip. The root `requirements.txt` covers most scripts; `src/ai-ml-exemplars/requirements.txt` lists LangChain-specific packages used by that subpackage.

```shell
pip install -r requirements.txt
```

Environment variables are loaded via `python-dotenv`. Set them using the `dotenv` CLI (run once):

```shell
dotenv set dbhost ""
dotenv set dbname ""
dotenv set dbuser ""
dotenv set dbcred ""
dotenv set aws_access_key_id ""
dotenv set aws_secret_access_key ""
```

Each `src/ai-ml-exemplars/*.py` that uses LLMs also reads from a `.env` file at `src/ai-ml-exemplars/.env`.

## Running Scripts

Scripts are standalone — run them directly with Python:

```shell
python src/ai-ml-exemplars/LCOllamaRag.py
python src/polars-exemplars/CricketStats.py
```

Run the Streamlit chatbot:

```shell
streamlit run src/streamlit-exemplars/file_search_chatbot.py
streamlit run src/streamlit-exemplars/claudechatbot.py
```

There is no test suite or build step.

## Code Architecture

### `src/` — Python source organized by domain

- **`src/ai-ml-exemplars/`** — The primary ML/AI exemplar package. Each file is a self-contained script demonstrating one concept:
  - `LC*.py` / `LD*.py` — LangChain integrations (Ollama, OpenAI, Gemini, Dash chatbots)
  - `LCOllamaRag.py`, `LangChainRagExample.py`, `LCOllamaRagEvalLog.py` — RAG chain patterns using FAISS or Chroma vector stores
  - `LanggraphExample.py` — LangGraph `StateGraph` with conditional edges for agentic RAG (retrieve → generate → verify → refine)
  - `HF*.py` — Hugging Face pipelines (NLP, token embeddings, random forest classifiers)
  - `Pinecone*.py` — Pinecone index create/delete helpers
  - `Spam*.py`, `Rule*.py`, `DQDL*.py` — Classical ML (TF-IDF, scikit-learn) and rule-engine utilities
  - `BackPropagationCode.py` — Manual backpropagation implementation

- **`src/polars-exemplars/`** — Polars DataFrame patterns (date ranges, upsampling, price data, reporting engine)

- **`src/python-exemplars/`** — General Python utilities: web scraping (BeautifulSoup), async patterns, markdown generators, PostgreSQL (psycopg2), Google OAuth

- **`src/database_service/`** — `PGDB_Service_Impl`: thin psycopg2 wrapper that reads DB credentials from env vars

- **`src/handlers/`** — `user_handler.py`: request/user handling utilities

- **`src/streamlit-exemplars/`** — Streamlit UIs: file-search chatbot and a Claude-powered chatbot

### `notebooks/` — Jupyter notebooks by topic

Workshop series (agentic AI, RAG, fine-tuning, NLP) and one-off experiments. Key subdirectories:
- `Agentic-AI-Workshop/` — LangGraph agentic RAG, corrective RAG, conditional edges demos
- `Building-Agentic-AI-Systems/` — Multi-agent systems, MCP server/client, memory engineering, Tool-Use ReAct with LangChain/CrewAI, Text2SQL with LangGraph
- `RAG-Workshop/`, `Multi-Rag-System-With-DeepEval/` — RAG pipelines and evaluation
- `LLM_Fine_Tuning.ipynb`, `Anthropic-Courses/` — Fine-tuning and prompt engineering

### `docs/` — Reference materials

Markdown cheat sheets, system design notes, and ML function curve references. Not auto-generated.

## Key Patterns

**LangChain RAG chain pattern** (used across multiple files):
```
embed docs → FAISS/Chroma vectorstore → retriever → ChatPromptTemplate → LLM → StrOutputParser
```

**LangGraph agentic pattern** (see `LanggraphExample.py`, workshop notebooks):
```
StateGraph → add_node (retrieve/generate/verify/refine) → add_conditional_edges → compile → invoke
```

**Ollama local models** — many scripts default to `llama3.2` (LLM) and `nomic-embed-text` (embeddings) running at `http://localhost:11434`. Ensure Ollama is running locally before executing these scripts.
