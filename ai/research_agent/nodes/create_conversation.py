import time

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.util import extract_content, safe_invoke
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
from util.SpinnerController import SpinnerController


def create_conversation(state: ResearchAgentState, spinner_controller: SpinnerController = None):
    """Initialize a new conversation by summarizing prior messages and extracting the last user message."""

    if spinner_controller:
        spinner_controller.start("::Preparing agent")

    # Extract graph state variables
    conversation = state.get("conversation", [])

    token_limit = 10000
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = len(tokenizer.encode(str([msg.content for msg in conversation])))
    if tokens > token_limit:
        model, reasoning = RESEARCH_AGENT_MODEL_CONFIG["create_conversation_summary"]

        # Build prompt (system and user message)
        system_msg = HumanMessage(content=(
            "## YOUR ROLE\n"
            "You are a conversation summarizer. Your job is to summarize the conversation between the user and the AI "
            "assistant up until and excluding this message, focusing on the key points addressed, questions asked, and "
            "any relevant context that would help. Note philosophers, sources, and concepts discussed.\n\n"
            "Your summary should at most half the length of the original conversation.\n"
        ))

        # Invoke model and extract content
        result = safe_invoke(model, [*conversation[:-1], system_msg], reasoning)
        conversation = [SystemMessage(content=f"## CONVERSATION SUMMARY BEFORE THIS POINT\n{extract_content(result)}"), conversation[-1]]

    # Initialize remaining required keys in state
    state.setdefault('response', '')
    state.setdefault('vector_db_queries', [])
    state.setdefault('sep_queries', [])
    state.setdefault('research_iterations', 1)
    state.setdefault('completed', False)
    state.setdefault('research_effort', ResearchEffort.NONE)
    state.setdefault('query_results', [])
    state.setdefault('all_raw_results', set())

    return {
        "conversation": conversation,
        'response': state['response'],
        'vector_db_queries': state['vector_db_queries'],
        'sep_queries': state['sep_queries'],
        'completed': state['completed'],
        'research_effort': state['research_effort'],
        'query_results': state['query_results'],
        'all_raw_results': state['all_raw_results']
    }
