from concurrent import futures

import grpc

from ai.research_agent.ResearchAgent import ResearchAgent
from cogito_servicer import cogito_pb2_grpc
from cogito_servicer.CogitoServer import CogitoServer
from dbs.Postgres import Postgres


class Server:
    """gRPC server for the Cogito service."""

    def __init__(self, agent: ResearchAgent = None, postgres_db: Postgres = None):
        """Initialize and start the gRPC server for the Cogito service."""

        # Create gRPC server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )

        # Build agent
        if agent is None:
            agent = ResearchAgent()
            agent.build()

        # Initialize database
        if postgres_db is None:
            postgres_db = Postgres()

        # Add Cogito servicer to server
        servicer = CogitoServer(agent, postgres_db)
        cogito_pb2_grpc.add_CogitoServicer_to_server(servicer, self.server)

    def start(self):
        """Start the gRPC server and listen for requests."""

        self.server.add_insecure_port("[::]:50051")
        self.server.start()
        self.server.wait_for_termination()
