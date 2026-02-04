from ai.research_agent.ResearchAgent import ResearchAgent
from cogito_servicer.Server import Server
from dbs.Postgres import Postgres

if __name__ == "__main__":

    # Build agent
    agent = ResearchAgent()
    agent.build()

    # Initialize Postgres database
    postgres_db = Postgres()

    # Start gRPC server
    server = Server(agent, postgres_db)
    server.start()
