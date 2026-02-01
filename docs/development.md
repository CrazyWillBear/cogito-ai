# Development Guide

This document provides guidance for developers who want to contribute to or extend Cogito AI.

## Development Setup

### Prerequisites

- Python 3.10+
- Docker
- Git
- A code editor (VS Code recommended)

### Clone and Install

```bash
git clone https://github.com/CrazyWillBear/cogito-ai.git
cd cogito-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start Development Databases

```bash
# Qdrant (philosophy embeddings)
docker run -p 6333:6333 -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=dev-key \
  crazywillbear/cogito-vectors:latest

# PostgreSQL (metadata and conversations)
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=dev \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=cogito \
  crazywillbear/cogito-filters-postgres:latest
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your development credentials
```

## Project Structure

```
cogito-ai/
├── cogito.py              # CLI entry point
├── cogito_server.py       # gRPC server entry point
├── ai/
│   ├── models/            # LLM provider implementations
│   │   ├── gpt.py         # OpenAI models
│   │   ├── groq.py        # Groq models
│   │   ├── ollama.py      # Local Ollama models
│   │   └── util.py        # Shared utilities
│   └── research_agent/
│       ├── ResearchAgent.py    # Main agent class
│       ├── model_config.py     # Model configuration
│       ├── nodes/              # Agent graph nodes
│       ├── schemas/            # Type definitions
│       └── sources/            # Data source interfaces
├── cli/
│   ├── main.py            # CLI implementation
│   ├── conversations.py   # Conversation logging
│   ├── db_containers.py   # Docker management
│   └── panels.py          # UI components
├── cogito_servicer/
│   ├── Server.py          # gRPC server
│   ├── CogitoServer.py    # Service implementation
│   ├── cogito.proto       # Protocol buffer definitions
│   └── cogito_pb2*.py     # Generated protobuf code
├── dbs/
│   ├── Postgres.py        # PostgreSQL client
│   ├── Qdrant.py          # Qdrant vector DB client
│   └── QueryAndFilterSchemas.py  # Query schemas
├── embed/
│   └── Embedder.py        # OpenAI embedding wrapper
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── .env.example           # Environment template
```

## Key Concepts

### LangGraph State Machine

The research agent is built using LangGraph, a library for building stateful, multi-step LLM applications.

The agent is a directed graph where:
- **Nodes** are functions that transform state
- **Edges** define transitions between nodes
- **State** is a TypedDict shared across nodes

### Agent Nodes

Each node in `ai/research_agent/nodes/` performs one step:

1. **create_conversation**: Summarizes conversation context
2. **classify_research_needed**: Determines research depth
3. **plan_research**: Plans queries and strategy
4. **execute_queries**: Runs queries against data sources
5. **write_response**: Synthesizes final response

### Adding a New Node

```python
# ai/research_agent/nodes/my_new_node.py
from rich.status import Status
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState

def my_new_node(state: ResearchAgentState, status: Status | None):
    """Description of what this node does."""
    
    if status:
        status.update("Processing...")
    
    # Access state
    conversation = state.get("conversation", [])
    
    # Perform operations...
    result = process_something(conversation)
    
    # Return state updates
    return {"my_field": result}
```

Register in `ResearchAgent.py`:

```python
from ai.research_agent.nodes.my_new_node import my_new_node

# In build():
g.add_node("my_new_node", self._wrap(my_new_node))
g.add_edge("previous_node", "my_new_node")
```

### Adding a New Data Source

Create a new source in `ai/research_agent/sources/`:

```python
# ai/research_agent/sources/my_source.py

async def search_my_source(query: str) -> list[dict]:
    """Search a custom data source."""
    # Implement search logic
    results = await fetch_from_source(query)
    return [
        {
            "content": r["text"],
            "citation": {
                "title": r["title"],
                "authors": r["authors"],
                "source": "My Source"
            }
        }
        for r in results
    ]
```

Integrate in `execute_queries.py`.

## Adding LLM Providers

### Creating a New Provider

```python
# ai/models/my_provider.py
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatMyProvider

def my_model():
    """Create a chat model instance."""
    return ChatMyProvider(
        model="model-name",
        temperature=0.7,
        api_key=os.getenv("MY_API_KEY")
    )
```

### Using in Model Config

```python
# ai/research_agent/model_config.py
from ai.models.my_provider import my_model

RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (my_model(), None),
    # ...
}
```

## Working with gRPC

### Modifying the API

1. Edit `cogito_servicer/cogito.proto`
2. Regenerate Python files:
   ```bash
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cogito_servicer/cogito.proto
   ```
3. Implement new methods in `CogitoServer.py`

### Testing gRPC Locally

Use `grpcurl` or write a test client:

```python
import grpc
from cogito_servicer import cogito_pb2, cogito_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = cogito_pb2_grpc.CogitoStub(channel)

response = stub.Complete(cogito_pb2.Conversation(
    user_id="test",
    conversation_id="1"
))
print(response.status)
```

## Debugging

### Enable Verbose Logging

Add print statements or use Python's logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Agent State

Modify `ResearchAgent.run()` to print state:

```python
def run(self, conversation, status=None):
    init_state = {"conversation": conversation}
    self.status = status
    res = self.graph.invoke(init_state)
    print("Final state:", res)  # Debug
    return res
```

### Test Individual Nodes

```python
from ai.research_agent.nodes.classify_research_needed import classify_research_needed
from langchain_core.messages import HumanMessage

state = {
    "conversation": [HumanMessage(content="What is Stoicism?")]
}
result = classify_research_needed(state, None)
print(result)  # {"research_effort": ResearchEffort.SIMPLE}
```

## Best Practices

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for public functions

### Commits

- Use descriptive commit messages
- Keep commits focused on single changes
- Reference issues when applicable

### Testing

- Test new features manually
- Verify existing functionality isn't broken
- Test with different research levels

## Common Tasks

### Update Dependencies

```bash
pip install --upgrade package-name
pip freeze > requirements.txt
```

### Build Docker Image

```bash
docker build -t cogito-ai .
```

### Run with Docker

```bash
docker run --env-file .env -p 50051:50051 cogito-ai
```

## License

Copyright (c) 2025 William Chastain. All rights reserved.

This software is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). See [LICENSE](../LICENSE.md) for details.
