# AI API Overview

## Service identity

This repository implements the AI API service for the UNDP Sustainable Energy Digital Intelligence Platform. Based on code and metadata in this repository, it supports the Sustainable Energy Academy (SEA) AI experience, Energy AI chat responses, publication-backed retrieval, and Knowledge Graph experiences.

Confirmed repository metadata:

- API title: `UNDP Sustainable Energy Academy AI API`
- Version in `metadata.yaml`: `0.6.1`
- Main FastAPI app: `main.py`
- Local tester app: `frontend/kg_tester_app.py`
- Deployment workflow: `.github/workflows/azure-webapps-python.yml`
- Azure Web App name in workflow: `sea-ai-api`

Unknown / requires confirmation:

- The exact production frontend applications consuming this API beyond the KG tester and platform references in repository documentation.
- Whether `Energy AI`, `Sustainable Energy Academy`, and `Knowledge Graph experience` are separate production applications or different surfaces within one platform frontend.

## What the API does

The API exposes:

- Knowledge graph node search and graph expansion.
- A streamed AI chat/model endpoint that returns answer chunks, graph data, document references, and suggested follow-up ideas.
- Document and source metadata search over a LanceDB-backed corpus.
- Retrieval-debug and table-debug endpoints for development and operations.
- Static documentation pages and changelog pages.
- A separate `/api/moonshot` router for project dashboard support functions, including natural-language filter parsing, project synopsis generation, and Prodoc PDF lookup/download.

## Platform features consuming it

Confirmed from repository code:

- The local KG tester frontend calls `/model` through `frontend/kg_tester_app.py` at `/kg-tester/api/model`.
- The legacy template `templates/kg_tester.html` calls `/graph` or `/graph/v2` directly.
- The main API serves `/kg-tester`, but README states the standalone tester app under `frontend/` is the intended local tester runtime.
- The Moonshot routes are included in the same FastAPI app under `/api/moonshot`.

Likely but requires confirmation:

- The Sustainable Energy Academy frontend consumes `/model` for chat and graph streaming.
- The Knowledge Graph frontend consumes `/graph/v2`, `/graph`, `/nodes`, and possibly `/model` graph chunks.
- Energy AI consumes `/model` and displays streamed `documents[]` separately as references.

## Supported AI and retrieval capabilities

Confirmed capabilities:

- Chat: `POST /model` streams assistant output as NDJSON.
- RAG: `/model` schedules publication retrieval and feeds selected publication chunks into a publication-grounded continuation prompt.
- Semantic search: LanceDB vector search is used for graph central-node matching and chunk retrieval fallback.
- Lexical search: document and chunk retrieval also use regex/lexical matching.
- Document retrieval: `/documents` searches canonical publication records; `/sources` lists source registry rows; `/debug/retrieve` returns selected documents and chunks for a query.
- Citations/source handling: `/model` streams `documents[]` as structured reference chunks separate from the answer body. Prompts instruct the model not to output source lists or raw URLs in answer text.
- Knowledge graph queries: `/graph` and `/graph/v2` return graph nodes and edges.
- Content recommendation: `/model` returns `ideas[]` follow-up query suggestions generated from conversation history.
- Scope guard: deterministic pre-generation scope guard blocks prompt probes, harmful requests, and clearly unrelated prompts.
- Output safety: assistant content, ideas, graph strings, and document text are HTML-encoded before serialization; markdown images are neutralized; document URLs are restricted to HTTP(S).

Unknown / requires confirmation:

- Whether the publication `documents[]` should be considered formal citations, references, or supporting sources in governance language.
- Whether the platform has a separate citation UI policy outside this API.

## Main runtime and framework

Confirmed runtime stack:

- Python 3.12 in Dockerfile and GitHub Actions.
- FastAPI `0.116.2`.
- Uvicorn `0.35.0`.
- Pydantic v2.
- NetworkX for graph assembly.
- LanceDB with Azure storage for vector/database storage.
- LangChain and LangGraph for Azure OpenAI chat, embeddings, and ReAct agent support.
- Pandas and PyArrow/FSS support for parquet and Azure storage access.

## External model providers and services

Confirmed primary services:

- Azure OpenAI chat model configured by:
  - `AZURE_OPENAI_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_VERSION`
  - `AZURE_OPENAI_CHAT_MODEL`
- Azure OpenAI embedding model configured by:
  - `AZURE_OPENAI_EMBED_MODEL`
- Azure Blob Storage for LanceDB, configured by:
  - `STORAGE_SAS_URL`, or
  - `STORAGE_ACCOUNT_NAME` with `STORAGE_ACCOUNT_KEY` or `STORAGE_SAS_TOKEN`

Confirmed auxiliary Moonshot provider logic:

- Moonshot routes can use Azure OpenAI or OpenAI depending on configured environment variables.
- OpenAI direct provider is selected when `OPENAI_API_KEY` is present and Azure config is absent.

Other external services used in code/notebook:

- UNSD SDG API is used by `main.ipynb` to build the `sdg7` table.
- Azure Blob public Prodoc container is used by Moonshot Prodoc lookup/download functions.

Unknown / requires confirmation:

- Exact deployed Azure OpenAI model names.
- Exact Azure Blob account/container ownership and access policy.
- Whether production uses Dockerfile, Azure App Service Oryx build, or another startup command.
