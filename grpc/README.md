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
pip install grpcio grpcio-tools
```

All other dependencies from the main project's `requirements.txt` must also be installed.

## Running the Server

From the project root directory:

```bash
cd grpc
python server.py
```

By default, the server listens on port `50051`. You can modify the port in the `server.py` file if needed.

## Client Example

Here's a simple Python client example:

```python
import grpc
import sys
sys.path.insert(0, 'grpc')
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
response = stub.Completion(request)
print(f"Response: {response.response}")
```

## Regenerating gRPC Code

If you modify `cogito.proto`, regenerate the Python code with:

```bash
cd grpc
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cogito.proto
```

## Notes

- The server uses a single `ResearchAgent` instance for all requests to optimize resource usage
- The agent is properly closed when the server shuts down
- The server uses ThreadPoolExecutor with a configurable number of workers (default: 10)
- Error handling is implemented to return appropriate gRPC status codes on failures
