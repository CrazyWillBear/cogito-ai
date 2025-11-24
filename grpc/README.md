# Cogito AI gRPC Server

This directory contains the gRPC server implementation for the Cogito AI research assistant.

## Overview

The gRPC server provides a remote procedure call interface to the Cogito AI agent. It maintains a single `ResearchAgent` instance that is reused for all requests, ensuring efficient resource utilization.

## Files

- `cogito.proto` - Protocol Buffer definition for the Cogito service
- `cogito_pb2.py` - Generated Python protobuf code (auto-generated)
- `cogito_pb2_grpc.py` - Generated Python gRPC code (auto-generated)
- `server.py` - gRPC server implementation
- `__init__.py` - Package initialization

## Service Definition

The service provides a single RPC method:

### Completion

Accepts a chat history and returns the agent's response.

**Request:**
```protobuf
message CompletionRequest {
  repeated Message messages = 1;  // Chat history
}

message Message {
  string role = 1;      // "system", "human", or "ai"
  string content = 2;   // Message content
}
```

**Response:**
```protobuf
message CompletionResponse {
  string response = 1;  // Agent's final response
}
```

## Prerequisites

Install gRPC dependencies:
```bash
pip install -r grpc/requirements.txt
```

All other dependencies from the main project's `requirements.txt` must also be installed:
```bash
pip install -r requirements.txt
```

The server requires:
- Running Qdrant vector database (configured in `dbs/qdrant.py`)
- Running PostgreSQL database with filter metadata (configured in `dbs/postgres_filters.py`)
- OpenAI API key or local LLM access (configured in `ai/subgraphs/research_agent/model_config.py`)

See the main [README](../README.md) for detailed setup instructions.

## Running the Server

From the grpc directory:

```bash
cd grpc
python run_server.py
```

Or from the project root directory:

```bash
python grpc/run_server.py
```

By default, the server listens on port `50051`. You can modify the port in the `run_server.py` file if needed.

**Note:** The `grpc` directory is intentionally not a Python package (no `__init__.py` at package level) to avoid naming conflicts with the `grpcio` library.

## Testing the Server

A test client is provided in `test_client.py`. To test the server:

1. Start the server in one terminal:
```bash
cd grpc
python run_server.py
```

2. Run the test client in another terminal:
```bash
cd grpc  
python test_client.py
```

## Client Example

Here's a simple Python client example:

```python
import sys
import os

# Add grpc directory to path
sys.path.append('path/to/cogito-ai/grpc')

import grpc
import cogito_pb2
import cogito_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = cogito_pb2_grpc.CogitoServiceStub(channel)

# Create request
request = cogito_pb2.CompletionRequest(
    messages=[
        cogito_pb2.Message(role="system", content="You are a helpful philosophical research assistant."),
        cogito_pb2.Message(role="human", content="What did Plato say about justice?")
    ]
)

# Make call
response = stub.Completion(request, timeout=300)  # 5 minute timeout
print(f"Response: {response.response}")

# Close channel
channel.close()
```

## Regenerating gRPC Code

If you modify `cogito.proto`, regenerate the Python code with:

```bash
cd grpc
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cogito.proto
```

## Architecture Notes

- The server uses a **single `ResearchAgent` instance** for all requests to optimize resource usage
  - The agent maintains connections to Qdrant vector database and PostgreSQL
  - Reusing the same agent avoids the overhead of repeatedly establishing database connections
  - The agent's state is reset for each request via the conversation parameter
- The agent is properly closed when the server shuts down
- The server uses ThreadPoolExecutor with a configurable number of workers (default: 10)
- Error handling is implemented to return appropriate gRPC status codes on failures
- The `grpc` directory is intentionally not a Python package to avoid naming conflicts with the `grpcio` library
