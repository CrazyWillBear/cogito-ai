"""
Simple test client for the Cogito gRPC server.

This script demonstrates how to connect to and use the Cogito gRPC service.
"""

import sys
import os

# Add grpc directory to path for proto imports
grpc_dir = os.path.dirname(os.path.abspath(__file__))
if grpc_dir not in sys.path:
    sys.path.append(grpc_dir)

# Import grpcio library
import grpc

# Import the generated proto files
import cogito_pb2
import cogito_pb2_grpc


def run_test():
    """Test the gRPC server with a simple request."""
    # Create channel
    channel = grpc.insecure_channel('localhost:50051')
    stub = cogito_pb2_grpc.CogitoServiceStub(channel)
    
    print("Testing Cogito gRPC Server...")
    print("-" * 50)
    
    # Create test request
    request = cogito_pb2.CompletionRequest(
        messages=[
            cogito_pb2.Message(
                role="system",
                content="You are a helpful philosophical research assistant."
            ),
            cogito_pb2.Message(
                role="human",
                content="What is epistemology?"
            )
        ]
    )
    
    print("Sending request...")
    print(f"Messages: {len(request.messages)}")
    for i, msg in enumerate(request.messages):
        print(f"  {i+1}. [{msg.role}]: {msg.content}")
    
    print("\nWaiting for response...")
    
    try:
        # Make the call (with timeout to avoid hanging indefinitely)
        response = stub.Completion(request, timeout=300)  # 5 minute timeout
        
        print("\nResponse received:")
        print("-" * 50)
        print(response.response)
        print("-" * 50)
        
    except grpc.RpcError as e:
        print(f"\nError: {e.code()}")
        print(f"Details: {e.details()}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    
    # Close channel
    channel.close()


if __name__ == '__main__':
    run_test()
