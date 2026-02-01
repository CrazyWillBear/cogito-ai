# API Reference

This document describes the gRPC API for programmatic access to Cogito AI.

## Overview

Cogito AI provides a gRPC server for integration with other applications. The server exposes a simple API for processing philosophical questions through stored conversations.

## Starting the Server

```bash
python cogito_server.py
```

The server listens on port `50051` by default.

## Protocol Buffer Definitions

The API is defined in `cogito_servicer/cogito.proto`:

```protobuf
syntax = "proto3";

package cogito;

service Cogito {
    rpc Complete (Conversation) returns (Status);
}

message Conversation {
    string user_id = 1;
    string conversation_id = 2;
}

message Status {
    string status = 1;
}
```

## Service Methods

### Complete

Processes a conversation and generates a response from the research agent.

#### Request: `Conversation`

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Unique identifier for the user |
| `conversation_id` | string | Unique identifier for the conversation |

#### Response: `Status`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Result status ("Success" or error message) |

#### Behavior

1. Retrieves the conversation from the PostgreSQL database using `user_id` and `conversation_id`
2. Converts the conversation to LangChain message format
3. Runs the research agent on the conversation
4. Appends the agent's response to the conversation
5. Stores the updated conversation back in the database
6. Returns a status indicating success or failure

## Conversation Storage

Conversations are stored in the PostgreSQL database in the `conversations` table:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | integer | User identifier |
| `conversation_id` | integer | Conversation identifier |
| `conversation` | JSONB | Serialized conversation messages |

### Message Format

Conversations are stored as a JSON array of messages in LangChain's dict format:

```json
[
    {
        "type": "human",
        "data": {
            "content": "What is Kant's categorical imperative?"
        }
    },
    {
        "type": "ai",
        "data": {
            "content": "Kant's categorical imperative is..."
        }
    }
]
```

## Client Examples

### Python with grpcio

```python
import grpc
from cogito_servicer import cogito_pb2, cogito_pb2_grpc

# Create channel and stub
channel = grpc.insecure_channel('localhost:50051')
stub = cogito_pb2_grpc.CogitoStub(channel)

# Create request
request = cogito_pb2.Conversation(
    user_id="123",
    conversation_id="456"
)

# Make call
response = stub.Complete(request)
print(f"Status: {response.status}")
```

### Node.js with @grpc/grpc-js

```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDef = protoLoader.loadSync('cogito.proto');
const cogito = grpc.loadPackageDefinition(packageDef).cogito;

const client = new cogito.Cogito(
    'localhost:50051',
    grpc.credentials.createInsecure()
);

client.Complete({ user_id: '123', conversation_id: '456' }, (err, response) => {
    if (err) {
        console.error(err);
    } else {
        console.log('Status:', response.status);
    }
});
```

## Error Handling

The API returns errors in the `status` field:

| Status | Description |
|--------|-------------|
| `Success` | Request completed successfully |
| `Error: <message>` | An error occurred (includes error details) |

Common errors:

- **Conversation not found**: The specified user_id/conversation_id doesn't exist
- **Database connection error**: Unable to connect to PostgreSQL
- **Agent error**: Error during research agent execution

## Concurrency

The gRPC server handles multiple concurrent requests:

- Uses a ThreadPoolExecutor with 10 worker threads
- Each request runs the agent in a separate process via ProcessPoolExecutor (4 workers)
- Database connections are managed per-request

## Integration Workflow

A typical integration workflow:

1. **Store conversation**: Insert/update the conversation in the PostgreSQL `conversations` table
2. **Call Complete**: Make gRPC call with user_id and conversation_id
3. **Retrieve result**: Read the updated conversation from the database

```python
# 1. Store the conversation with user's question
postgres.update_conversation(user_id, conversation_id, [
    {"type": "human", "data": {"content": "What is existentialism?"}}
])

# 2. Call the Complete RPC
response = stub.Complete(cogito_pb2.Conversation(
    user_id=user_id,
    conversation_id=conversation_id
))

# 3. Retrieve the updated conversation with AI response
conversation = postgres.get_conversation(user_id, conversation_id)
ai_response = conversation[-1]["data"]["content"]
```

## Regenerating Protocol Buffers

If you modify `cogito.proto`, regenerate the Python files:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cogito_servicer/cogito.proto
```
