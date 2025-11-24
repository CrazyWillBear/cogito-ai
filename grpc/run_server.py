#!/usr/bin/env python
"""
Wrapper script to run the gRPC server with proper import setup.

This script ensures that imports are resolved correctly by running
the server from the parent directory context.
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the absolute path to the project root (parent of grpc directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Get the grpc directory path
grpc_dir = os.path.join(project_root, 'grpc')

# Add project root to sys.path for project modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import grpcio library first before adding grpc directory to path
from concurrent import futures
import grpc

# Now add grpc directory for proto files
if grpc_dir not in sys.path:
    sys.path.append(grpc_dir)

# Now import project and proto modules
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ai.subgraphs.research_agent.research_agent import ResearchAgent

# Import the generated proto files
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
        logger.info("Initializing ResearchAgent...")
        self.agent = ResearchAgent()
        self.agent.build()
        logger.info("ResearchAgent initialized and ready")

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
            logger.info(f"Processing completion request with {len(request.messages)} messages")
            
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
                    # Log warning for unknown roles and default to HumanMessage
                    logger.warning(f"Unknown message role '{role}', defaulting to 'human'")
                    messages.append(HumanMessage(content=content))
            
            # Create conversation dict for the agent
            conversation = {"messages": messages}
            
            # Run the agent
            logger.info("Invoking ResearchAgent...")
            response = self.agent.run(conversation)
            logger.info("Agent completed successfully")
            
            # Return the response
            return cogito_pb2.CompletionResponse(response=response)
            
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing request: {str(e)}")
            return cogito_pb2.CompletionResponse(response="")

    def close(self):
        """Close agent resources when server shuts down."""
        if self.agent:
            logger.info("Closing ResearchAgent resources...")
            self.agent.close()
            logger.info("ResearchAgent resources closed")


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
    logger.info(f"Cogito gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        servicer.close()
        server.stop(0)
        logger.info("Server stopped")


if __name__ == '__main__':
    serve()
