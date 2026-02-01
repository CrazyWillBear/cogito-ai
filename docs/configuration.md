# Configuration

This document describes all configuration options available in Cogito AI.

## Environment Variables

Cogito AI is configured primarily through environment variables. Copy `.env.example` to `.env` and configure the following:

### Qdrant Vector Database

| Variable | Description | Default |
|----------|-------------|---------|
| `COGITO_QDRANT_URL` | Qdrant server hostname | `localhost` |
| `COGITO_QDRANT_PORT` | Qdrant gRPC port | `6334` |
| `COGITO_QDRANT_API_KEY` | API key for Qdrant authentication | Required |
| `COGITO_QDRANT_COLLECTION` | Name of the Qdrant collection | `proj_gutenberg_philosophy` |

### PostgreSQL Database

| Variable | Description | Default |
|----------|-------------|---------|
| `COGITO_POSTGRES_HOST` | PostgreSQL server hostname | `localhost` |
| `COGITO_POSTGRES_PORT` | PostgreSQL port | `5432` |
| `COGITO_POSTGRES_DBNAME` | Database name | `cogito` |
| `COGITO_POSTGRES_USER` | Database username | Required |
| `COGITO_POSTGRES_PASSWORD` | Database password | Required |

### LLM API Keys

| Variable | Description | Required For |
|----------|-------------|--------------|
| `GROQ_API_KEY` | Groq API key | Default configuration |
| `OPENAI_API_KEY` | OpenAI API key | OpenAI models, embeddings |

## LLM Configuration

### Default Configuration (Groq)

The default configuration uses Groq API models optimized for speed, cost, and accuracy. No changes are needed for typical use.

### Custom Model Configuration

To customize LLM models, edit `ai/research_agent/model_config.py`:

```python
from ai.models.groq import llama_8b_instant, llama_4_scout

RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (llama_8b_instant, None),    # Summarization
    "research_classifier": (oss_20b_low_temp, None),    # Classification
    "extract_text": (llama_8b_instant, None),           # Text extraction
    "plan_research": (llama_4_scout, None),             # Research planning
    "write_response_no_research": (oss_20b_high_temp_med_reasoning, None),
    "write_response_research": (oss_120b_med, None)     # Response synthesis
}
```

### Available Model Providers

#### Groq Models (`ai/models/groq.py`)

Fast inference with Llama and other open-source models.

#### OpenAI Models (`ai/models/gpt.py`)

GPT-4 and other OpenAI models. Supports reasoning levels for newer models.

#### Ollama Models (`ai/models/ollama.py`)

Local LLM inference. Requires Ollama to be running locally.

### Using Local Models

For local inference with Ollama:

1. Install and start [Ollama](https://ollama.ai/)
2. Pull your desired model: `ollama pull gemma3:4b`
3. Update `model_config.py` to use Ollama models:

```python
from ai.models.ollama import gemma3_4b

RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (gemma3_4b, None),
    "research_classifier": (gemma3_4b, None),
    "extract_text": (gemma3_4b, None),
    "plan_research": (gemma3_4b, None),
    "write_response_no_research": (gemma3_4b, None),
    "write_response_research": (gemma3_4b, None)
}
```

## Database Configuration

### Pre-built Docker Images

The easiest way to get started is with the pre-built Docker images:

```bash
# Qdrant with philosophy embeddings
docker run -p 6333:6333 -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=your-secret-key \
  crazywillbear/cogito-vectors:latest

# PostgreSQL with metadata
docker run -d \
  -p 5432:5432 \
  -e POSTGRES_USER=your-username \
  -e POSTGRES_PASSWORD=your-password \
  -e POSTGRES_DB=cogito \
  crazywillbear/cogito-filters-postgres:latest
```

### Qdrant Configuration

The Qdrant database contains:

- **212,248 embeddings** from Project Gutenberg philosophy texts
- Chunked text with metadata (section, source title, authors)

Connection parameters:

- Uses gRPC protocol (port 6334)
- API key authentication
- TLS is disabled by default (configure in `dbs/Qdrant.py` for production)

### PostgreSQL Configuration

The PostgreSQL database contains:

- `filters` table: Author and source metadata for filtering
- `conversations` table: Conversation state for gRPC server

Features:

- Real-time updates via LISTEN/NOTIFY
- Auto-commit isolation level

## gRPC Server Configuration

The gRPC server runs on port 50051 by default. To change this, edit `cogito_servicer/Server.py`:

```python
self.server.add_insecure_port("[::]:50051")  # Change port here
```

### Production Considerations

For production deployments:

1. **Enable TLS for Qdrant**: Update `dbs/Qdrant.py`:
   ```python
   self.client = QdrantClient(url=url, grpc_port=port, prefer_grpc=True, https=True, api_key=api_key)
   ```

2. **Enable TLS for gRPC**: Replace `add_insecure_port` with secure credentials

3. **Use environment-specific configuration**: Consider using different `.env` files for development/production

## Embedding Configuration

Embeddings use OpenAI's `text-embedding-3-small` model by default. This requires an `OPENAI_API_KEY` environment variable.

The embedder is configured in `embed/Embedder.py` and is used by the Qdrant client for query embedding.
