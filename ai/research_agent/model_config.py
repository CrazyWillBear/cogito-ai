from ai.models.groq import oss_120b, llama_8b_instant, oss_20b

# Model configuration for graph agent_assigner (model, reasoning amount)
RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (llama_8b_instant, None),    # Conversation summarization task
    "research_classifier": (oss_20b, None),             # Research need classification task
    "extract_text": (llama_8b_instant, None),           # Text extraction task
    "plan_research": (oss_120b, None),                  # High complexity planning + structured output task
    "write_response": (oss_120b, None)                  # Moderate complexity evidence synthesis task
}
