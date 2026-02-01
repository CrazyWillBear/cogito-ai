# Architecture

This document provides an overview of Cogito AI's system architecture, including its core components, data flow, and design principles.

## System Overview

Cogito AI is built as a modular research agent that combines multiple data sources with LLM-powered reasoning to answer philosophical questions. The system follows a graph-based agent architecture using LangGraph.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│                    (CLI: cogito.py / gRPC: cogito_server.py)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Research Agent                                   │
│                        (ai/research_agent/)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         State Graph                                   │   │
│  │                                                                       │   │
│  │  START ──▶ create_conversation ──▶ classify_research_needed          │   │
│  │                                            │                          │   │
│  │                          ┌─────────────────┴─────────────────┐       │   │
│  │                          ▼                                   ▼       │   │
│  │                   plan_research                       write_response │   │
│  │                          │                                   │       │   │
│  │                          ▼                                   │       │   │
│  │                   execute_queries ◀──────────────────────────┤       │   │
│  │                          │                                   │       │   │
│  │                          └───────────────────────────────────┘       │   │
│  │                                            │                          │   │
│  │                                            ▼                          │   │
│  │                                          END                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Sources                                    │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────────┐ │
│  │   Qdrant Vector   │   │    PostgreSQL     │   │ Stanford Encyclopedia │ │
│  │      Database     │   │      Filters      │   │   of Philosophy (SEP) │ │
│  │  (dbs/Qdrant.py)  │   │ (dbs/Postgres.py) │   │  (sources/sep.py)     │ │
│  └───────────────────┘   └───────────────────┘   └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### User Interfaces

#### CLI (`cogito.py`, `cli/`)

The command-line interface provides an interactive terminal experience:

- **`cogito.py`**: Entry point that calls `cli/main.py`
- **`cli/main.py`**: Main CLI loop with Rich-based UI
- **`cli/panels.py`**: UI components for displaying responses
- **`cli/conversations.py`**: Conversation logging utilities
- **`cli/db_containers.py`**: Docker container management

#### gRPC Server (`cogito_server.py`, `cogito_servicer/`)

The gRPC server enables programmatic access:

- **`cogito_server.py`**: Server entry point
- **`cogito_servicer/Server.py`**: gRPC server implementation
- **`cogito_servicer/CogitoServer.py`**: Service method implementations
- **`cogito_servicer/cogito.proto`**: Protocol buffer definitions

### Research Agent (`ai/research_agent/`)

The core intelligence of Cogito AI, built as a LangGraph state machine:

#### Main Components

- **`ResearchAgent.py`**: Orchestrates the research process
- **`model_config.py`**: LLM configuration for each node

#### Nodes (`nodes/`)

Each node performs a specific step in the research pipeline:

| Node | Description |
|------|-------------|
| `create_conversation.py` | Summarizes conversation context |
| `classify_research_needed.py` | Determines research depth (none/simple/deep) |
| `plan_research.py` | Plans queries and research strategy |
| `execute_queries.py` | Executes queries against data sources |
| `write_response.py` | Synthesizes final response with citations |

#### Schemas (`schemas/`)

Type definitions for the agent state:

- **`ResearchAgentState.py`**: Main state schema
- **`ResearchEffort.py`**: Research depth levels (NONE, SIMPLE, DEEP)
- **`QueryResult.py`**: Query result structure
- **`Citation.py`**: Citation format
- **`QueryList.py`**: Query list structure

#### Sources (`sources/`)

Data source interfaces:

- **`sep.py`**: Stanford Encyclopedia of Philosophy scraper
- **`vector_db.py`**: Vector database query utilities
- **`stringify.py`**: Result formatting utilities

### LLM Models (`ai/models/`)

Configurable LLM backends:

- **`groq.py`**: Groq API models (default)
- **`gpt.py`**: OpenAI GPT models
- **`ollama.py`**: Local Ollama models
- **`util.py`**: Shared utilities

### Databases (`dbs/`)

#### Qdrant Vector Database (`Qdrant.py`)

- Stores over 212,000 embeddings from Project Gutenberg philosophy texts
- Supports fuzzy-matched filtering by author and source
- Batch query capabilities for efficient retrieval

#### PostgreSQL (`Postgres.py`)

- Stores metadata for filtering (authors, sources)
- Manages conversation state for gRPC server
- Supports real-time updates via PostgreSQL LISTEN/NOTIFY

### Embedding (`embed/`)

- **`Embedder.py`**: OpenAI text-embedding-3-small wrapper

## Data Flow

### 1. User Input

User submits a philosophical question via CLI or gRPC.

### 2. Conversation Creation

The agent summarizes the conversation context for efficient processing.

### 3. Research Classification

The classifier determines the appropriate research depth:

- **NONE (0)**: No research needed (casual conversation)
- **SIMPLE (1)**: Basic research (up to 5 iterations)
- **DEEP (2)**: Deep research (up to 8 iterations)

### 4. Research Planning

For questions requiring research, the planner:

- Creates a long-term research strategy
- Generates specific queries for each iteration
- Identifies relevant authors and sources

### 5. Query Execution

Queries are executed against multiple sources:

- **Qdrant**: Semantic search over Project Gutenberg texts
- **SEP**: Stanford Encyclopedia of Philosophy

### 6. Iterative Refinement

The agent iterates between planning and execution until:

- Sufficient evidence is gathered
- Maximum iterations reached
- Query results are satisfactory

### 7. Response Synthesis

The writer node synthesizes a final response with:

- Evidence-backed claims
- Proper citations
- Quoted references

## Design Principles

### Modularity

Components are loosely coupled and can be swapped independently (e.g., different LLM providers).

### Transparency

Research process is visible to users (research level, sources consulted, time taken).

### Accuracy

Multiple strategies reduce hallucinations:

- Source grounding
- Citation requirements
- Evidence formatting

### Flexibility

Supports multiple deployment modes:

- Local CLI for personal use
- gRPC server for integration
- Customizable LLM backends
