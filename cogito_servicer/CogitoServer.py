from concurrent.futures import ProcessPoolExecutor

from ai.research_agent.ResearchAgent import ResearchAgent
from cogito_servicer import cogito_pb2_grpc
from dbs.Postgres import Postgres

# Global process pool for handling requests
process_pool = ProcessPoolExecutor(max_workers=4)


def _run_agent_task(agent: ResearchAgent, conversation: dict) -> str:
    """Helper function to run the agent task in a separate process."""

    return agent.run(conversation)


class CogitoServer(cogito_pb2_grpc.CogitoServicer):
    """gRPC servicer for the Cogito AI research assistant."""

    def __init__(self, agent: ResearchAgent, postgres_db: Postgres):
        """Initialize the CogitoServer with a ResearchAgent and Postgres database."""

        self.agent = agent
        self.postgres_db = postgres_db

    def Ask(self, request, context):
        """Handle the Ask gRPC method to process user questions."""

        # Extract parameters from the request
        user_id = request.user_id
        conversation_id = request.conversation_id
        agent_name = request.agent  # Currently unused, reserved for future use

        # Retrieve the conversation from the Postgres database
        conversation = self.postgres_db.get_conversation(user_id, conversation_id)

        # Run the agent in a separate process
        future = process_pool.submit(_run_agent_task, self.agent, conversation)
        output = future.result()

        return output
