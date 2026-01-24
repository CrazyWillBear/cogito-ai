from concurrent.futures import ProcessPoolExecutor

from langchain_core.messages import AnyMessage, messages_from_dict, AIMessage

from ai.research_agent.ResearchAgent import ResearchAgent
from cogito_servicer import cogito_pb2_grpc
from dbs.Postgres import Postgres

# Global process pool for handling requests
process_pool = ProcessPoolExecutor(max_workers=4)


def _run_agent_task(agent: ResearchAgent, conversation: list[AnyMessage]) -> str:
    """Helper function to run the agent task in a separate process."""

    return agent.run(conversation).get("response")

def _convert_conversation(conversation: list[dict]) -> list[AnyMessage]:
    """Convert a conversation from dict format to AnyMessage format."""

    return messages_from_dict(conversation)


class CogitoServer(cogito_pb2_grpc.CogitoServicer):
    """gRPC servicer for the Cogito AI research assistant."""

    def __init__(self, agent: ResearchAgent, postgres_db: Postgres):
        """Initialize the CogitoServer with a ResearchAgent and Postgres database."""


        print("Initializing CogitoServer...")
        self.agent = agent
        self.postgres_db = postgres_db

    def Complete(self, request, context):
        """Handle the Ask gRPC method to process user questions."""

        try:
            # Extract parameters from the request
            user_id = request.user_id
            conversation_id = request.conversation_id

            print("Attempting to complete conversation for user:", user_id, "conversation:", conversation_id)

            # Retrieve the conversation from the Postgres database
            conversation = self.postgres_db.get_conversation(user_id, conversation_id)
            conversation = _convert_conversation(conversation)

            # Run the agent in a separate process
            future = process_pool.submit(_run_agent_task, self.agent, conversation)
            output = future.result()
            conversation.append(AIMessage(content=output))

            print("Completed conversation for user:", user_id, "conversation:", conversation_id)

            # Convert conversation back to dict format and store
            conversation = [msg.to_dict() for msg in conversation]
            self.postgres_db.update_conversation(user_id, conversation_id, conversation)

            return "Success"

        except Exception as e:
            print("Error during Complete:", str(e))

            return f"Error: {str(e)}"
