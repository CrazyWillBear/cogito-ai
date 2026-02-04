# Cogito AI

## One-liner

A chatbot agent for philosophy that performs research to gather evidence and synthesize answers.

## Summary

Cogito AI is an agentic chatbot for philosophy research. Cogito searches authoritative sources like the Stanford Encyclopedia of Philosophy and a curated set of Project Gutenberg (primary) texts. It uses a ReAct loop to plan and execute its research. In doing so, it generates context-aware queries, retrieves relevant sources, and synthesizes well-supported answers with full citations. The model architecture is optimized for cost, accuracy, and output quality; it's also customizable (see [Model Configuration](#model-configuration) below).

## Realtime Demo

### Defining philosophical concept

<video src="https://github.com/user/repo/raw/main/assets/demo.mp4" controls="controls" style="max-width: 600px;"></video>

## Features

- ReAct loop for planning and executing research.
- Search across the Stanford Encyclopedia of Philosophy.
- Search across 1000+ select Project Gutenberg philosophy sources.
- Conversation-aware query generation with source + author filtering.
- Parallel resource retrieval and text-extraction.
- Multistep reasoning for formulating and writing responses.
- Citation-backed quotes in responses.
- Low hallucination rate from prompt/evidence formatting and source grounding.

## Prerequisites

- Python 3.13
- Linux / macOS / Windows
- Docker (user must also have Docker permissions)
- OpenAI API key
- Groq API key (required for default model config, see [Model Configuration](#model-configuration) below)

To get a Groq API key, go [here](https://console.groq.com/keys). To get an OpenAI API key, go [here](https://platform.openai.com/api-keys).

## Quick Start

### 1. Set up Python environment

#### Linux / macOS
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Windows
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Set environment variables

- A **Groq API key** is required for the default configuration (see [Model Configuration](#model-configuration) below).
- An **OpenAI API key** is needed regardless, as it's used for text embedding generation.

#### Linux / macOS

```bash
# Add this to your .bashrc, .zshrc, .<whatever>rc file or set directly in terminal before running
export GROQ_API_KEY=your_groq_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

#### Windows

You need to use the System Environment Variables settings in Windows to set these. Follow [this tutorial](https://www.elevenforum.com/t/create-new-environment-variables-in-windows-11.22062/) following option 1, creating a new user variable for each of the above keys (where the variable name is `GROQ_API_KEY` and `OPENAI_API_KEY`, respectively, and the variable value is the key).

### 3. Run

```bash
# For terminal interface
python cogito.py
```

## Databases

### Qdrant Vector Database
- **212,248 embeddings** from Project Gutenberg philosophy texts
- Chunked with metadata (section, source title, author(s))
- [Docker Hub](https://hub.docker.com/repository/docker/crazywillbear/cogito-vectors)

### PostgreSQL Filters Database
- Metadata for filtering by author and source
- [Docker Hub](https://hub.docker.com/repository/docker/crazywillbear/cogito-filters-postgres)

## Model Configuration

It's recommended to leave LLM configuration as-is for best results (current models are optimized for speed, cost, and accuracy). If you wish to customize, here's how:

- Create LangChain `ChatModel` instances with different models, temperature, max tokens, etc. (check `ai/models/` for examples).
- In `ai/research_agent/model_config.py`, assign your chosen models to their tasks.

## License

Copyright (c) 2025 William Chastain. All rights reserved.

This software is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). This project is source‑available, but *CANNOT BE USED* for commerical purposes. In other words, don't sell my software or make money off it. See [LICENSE](LICENSE.md) for details.
