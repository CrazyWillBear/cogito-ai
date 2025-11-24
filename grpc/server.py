"""
gRPC Server implementation for Cogito AI research assistant.

This module provides a gRPC server that exposes the ResearchAgent
through a gRPC interface. The server accepts chat history and returns
agent responses.
"""

import sys
import os
from concurrent import futures
import grpc

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ai.subgraphs.research_agent.research_agent import ResearchAgent
import cogito_pb2
import cogito_pb2_grpc


class CogitoServicer(cogito_pb2_grpc.CogitoServiceServicer):
    """
    gRPC Servicer implementation for Cogito AI.
    
    This class implements the CogitoService defined in the proto file.
    It maintains a single ResearchAgent instance that is reused for all requests.
    """

    def __init__(self):
        """Initialize the servicer with a single ResearchAgent instance."""
        self.agent = ResearchAgent()
        self.agent.build()

    def Completion(self, request, context):
        """
        Handle completion requests by invoking the research agent.
        
        Args:
            request: CompletionRequest containing chat history messages
            context: gRPC context
            
        Returns:
            CompletionResponse with the agent's final response
        """
        try:
            # Convert gRPC messages to LangChain message format
            messages = []
            for msg in request.messages:
                role = msg.role.lower()
                content = msg.content
                
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))
                else:
                    # Default to HumanMessage for unknown roles
                    messages.append(HumanMessage(content=content))
            
            # Create conversation dict for the agent
            conversation = {"messages": messages}
            
            # Run the agent
            response = self.agent.run(conversation)
            
            # Return the response
            return cogito_pb2.CompletionResponse(response=response)
            
        except Exception as e:
            # Handle errors gracefully
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing request: {str(e)}")
            return cogito_pb2.CompletionResponse(response="")

    def close(self):
        """Close agent resources when server shuts down."""
        if self.agent:
            self.agent.close()


def serve(port=50051, max_workers=10):
    """
    Start the gRPC server.
    
    Args:
        port: Port number to listen on (default: 50051)
        max_workers: Maximum number of worker threads (default: 10)
    """
    # Create server with thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # Create and register servicer
    servicer = CogitoServicer()
    cogito_pb2_grpc.add_CogitoServiceServicer_to_server(servicer, server)
    
    # Bind to port
    server.add_insecure_port(f'[::]:{port}')
    
    # Start server
    server.start()
    print(f"Cogito gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        servicer.close()
        server.stop(0)


if __name__ == '__main__':
    serve()
