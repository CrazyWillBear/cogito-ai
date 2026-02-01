# Getting Started

This guide will help you get Cogito AI up and running on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**
- **Docker** (for running the databases)
- **Git** (for cloning the repository)
- **Groq API key** (or OpenAI API key / local setup for custom configuration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CrazyWillBear/cogito-ai.git
cd cogito-ai
```

### 2. Set Up Databases

Pull and run the pre-populated databases using Docker:

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

### 3. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
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

# LLM API Key (Groq recommended)
GROQ_API_KEY=your-groq-key
```

## Running Cogito AI

### Command-Line Interface (CLI)

For an interactive terminal experience:

```bash
python cogito.py
```

This will start an interactive session where you can ask philosophical questions and receive research-backed answers.

### gRPC Server

For programmatic access via gRPC:

```bash
python cogito_server.py
```

The server will start on port 50051.

> **Note**: In production environments, ensure Qdrant TLS is configured in `dbs/Qdrant.py`.

## Your First Query

After starting the CLI, try asking a philosophical question:

```
▸ You: What is Kant's categorical imperative?
```

Cogito will:
1. Classify the research depth needed
2. Plan and execute research queries
3. Search the Stanford Encyclopedia of Philosophy and Project Gutenberg texts
4. Synthesize a response with citations

## Next Steps

- Learn about the [Architecture](architecture.md)
- Explore [Configuration](configuration.md) options
- Read the [CLI Usage](cli-usage.md) guide
- Check out the [API Reference](api-reference.md) for gRPC integration
