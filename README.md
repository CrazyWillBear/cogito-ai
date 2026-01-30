# Cogito AI

## One-liner

A chatbot-style agent for philosophy research that performs research to gather evidence and synthesize answers.

## Summary

Cogito AI is an AI research assistant designed to help users explore philosophical questions by searching authoritative sources like the Stanford Encyclopedia of Philosophy and a curated set of Project Gutenberg texts. It leverages LLMs to generate context-aware queries, retrieve relevant sources, and synthesize well-supported answers with full citations. The model architecture is fully customizable, including support for Groq, OpenAI, and local LLMs (see more below).

## Realtime Demo

### Defining philosophical concept

<img src="https://mirrors.williamchastain.com/images/Cogito%20CLI%20Demo.gif" alt="Cogito Demo Gif" width="600"/>

### Comparing philosophers' ideas

<img src="https://mirrors.williamchastain.com/images/Cogito%20CLI%20Demo%202.gif" alt="Cogito Demo Gif 2" width="600"/>

## Features

- Search across the Stanford Encyclopedia of Philosophy
- Semantic search across 1000+ select Project Gutenberg philosophy sources
- Conversation-aware query generation with source + author filtering
- Parallel resource retrieval and text-extraction
- Planning and iterative research with chain-of-thought reasoning
- Citation-backed responses with quoted evidence
- Low hallucination rate from prompt/evidence formatting and source grounding

## Prerequisites

- Python 3.10+
- Linux / macOS / Windows
- Docker
- Groq API key (or OpenAI API key / local setup for custom configuration, see below)

## Quick Start

### 1. Set up databases

Pull and run the pre-populated databases:
```bash
# Qdrant vector database (philosophy embeddings, only uses gRPC)
docker run -p 6333:6333 -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=your-secret-key \
  crazywillbear/cogito-vectors:latest

# PostgreSQL filters database (metadata)
docker run -d \
  -p 5432:5432 \
  -e POSTGRES_USER=your-username \
  -e POSTGRES_PASSWORD=your-password \
  -e POSTGRES_DB=cogito \
  crazywillbear/cogito-filters-postgres:latest
```

### 2. Set up Python environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and update with the credentials you defined above:
```bash
cp .env.example .env
```

Edit `.env`:
```
# Qdrant Configuration
COGITO_QDRANT_URL=localhost
COGITO_QDRANT_PORT=6334
COGITO_QDRANT_API_KEY=your-secret-key
COGITO_QDRANT_COLLECTION=philosophy

# PostgreSQL Configuration
COGITO_POSTGRES_HOST=localhost
COGITO_POSTGRES_PORT=5432
COGITO_POSTGRES_DBNAME=cogito
COGITO_POSTGRES_USER=your_user_here
COGITO_POSTGRES_PASSWORD=your_password_here

# OpenAI (or configure local LLM)
GROQ_API_KEY=your-groq-key
```

### 4. Run
```bash
# For terminal interface
python cogito.py

# For gRPC server
python cogito_server.py
```

## Databases

### Qdrant Vector Database
- **212,248 embeddings** from Project Gutenberg philosophy texts
- Chunked with metadata (section, source title, author(s))
- Collection: ``
- [Docker Hub](https://hub.docker.com/repository/docker/crazywillbear/cogito-vectors)

### PostgreSQL Filters Database
- Metadata for filtering by author and source
- Table: `filters`
- [Docker Hub](https://hub.docker.com/repository/docker/crazywillbear/cogito-filters-postgres)

## Architecture & Important Files (quick map)

- `cogito.py` — CLI loop.
- `cogito_server.py` — gRPC server entrypoint.
- `cogito_servicer/` — gRPC server implementation.
- `ai/research_agent/` — research agent graph, nodes, and schemas.
- `ai/model_config.py` — model config.
- `dbs/` — Qdrant and PostgreSQL classes.
- `embed/` — embedding logic.

## Configuration

It's recommended to leave LLM configuration as-is for best results (current models are optimized for speed, cost, and
accuracy.) If you wish to customize, here are the main areas:

- **LLM configuration**: See `ai/models/model_config.py`
  - Create LangChain `ChatModel` instances with different models, temperature, max tokens, etc. (ideally in something
like `ai/models/<your_model>.py`)
  - You can set reasoning levels for newer OpenAI and Anthropic models.
- **Database connections**: Configured via `.env` file

## License

Copyright (c) 2025 William Chastain. All rights reserved.

This software is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). This project is source‑available, but non‑commercial. See [LICENSE](LICENSE.md) for details.
