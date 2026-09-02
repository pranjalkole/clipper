# Clipper: Multimodal Semantic Search & Agentic Intelligence Platform

Enterprise Multimodal Semantic Search & Agentic RAG Platform utilizing Apple's **MobileCLIP2-S0**, **Qdrant**, **LangGraph**, and **Google Gemini 1.5 Flash**.

## Features
- **MobileCLIP2-S0 Inference**: 512-dimensional L2-normalized image and text embeddings.
- **Qdrant Vector DB**: Vector similarity and faceted hybrid metadata filtering.
- **LangGraph Multi-Agent Architecture**: Intent routing, ReAct retrieval, self-reflective critique, and multimodal synthesis.
- **Enterprise Observability**: OpenTelemetry tracing and LLM performance telemetry.

## Setup & Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```
