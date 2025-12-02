# Cogito AI

An agentic Q&A research assistant for philosophy that uses vector search and LLMs to gather, assess, and synthesize answers from primary philosophical sources.

## Features

- Semantic search across 166,480+ philosophy text embeddings from Project Gutenberg
- Conversation-aware query generation with author/source filtering
- Parallel resource retrieval and summarization
- Iterative research with resource sufficiency assessment
- Citation-backed responses with quoted evidence

## Prerequisites

- Python 3.10+
- Linux / macOS / WSL (not Windows)
- Docker (for databases)
- GPU strongly recommended for embeddings (CPU possible but slower)
- OpenAI API key or local LLM setup (configuration required)

## Quick Start

### 1. Set up databases

Pull and run the pre-populated databases:
```bash
# Qdrant vector database (philosophy embeddings, only uses gRPC)
docker run -d \
  -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=your-secret-key \
  --name cogito-qdrant \
  crazywillbear/cogito-qdrant:v1

# PostgreSQL filters database (metadata)
docker run -d \
  -p 5432:5432 \
  --name cogito-postgres \
  crazywillbear/cogito-postgres-filters:v1
```

**Default PostgreSQL credentials** (these are NOT TO BE USED in production):
- Username: `newuser`
- Password: `newpass123`
- Database: `filters`

### 2. Set up Python environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and update with your credentials:
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
COGITO_POSTGRES_DBNAME=filters
COGITO_POSTGRES_USER=newuser
COGITO_POSTGRES_PASSWORD=newpass123

# OpenAI (or configure local LLM)
OPENAI_API_KEY=your-openai-key
```

### 4. Run
```bash
python main.py
```

## Databases

### Qdrant Vector Database
- **166,480 embeddings** from Project Gutenberg philosophy texts
- Chunked with metadata (chapter, section, source, authors)
- Collection: `philosophy`
- [Docker Hub](https://hub.docker.com/repository/docker/crazywillbear/cogito-qdrant)

### PostgreSQL Filters Database
- Metadata for filtering by author and source
- Table: `filters`
- [Docker Hub](https://hub.docker.com/repository/docker/crazywillbear/cogito-postgres-filters)

## Architecture & Important Files (quick map)

- `main.py` — CLI loop and `ResearchAgent` bootstrap.
- `ai/subgraphs/research_agent/` — research orchestration graph, nodes, and model mapping.
  - `model_config.py` — node-to-model mapping.
  - `research_agent.py` — orchestrates graph execution.
- `ai/models/` — model factory helpers (e.g., `gpt.py`, `llama.py`).
- `dbs/` — qdrant and postgres wrappers (`qdrant.py`, `postgres_filters.py`, `query.py`).
- `embed/` — embedding logic (`embed.py`).

## Configuration

- **LLM configuration**: See `ai/subgraphs/research_agent/model_config.py`
- **Database connections**: Configured via `.env` file

## License

Copyright (c) 2025 William Chastain. All rights reserved.

This software is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). This project is source‑available, but non‑commercial. See [LICENSE](LICENSE.md) for details.
