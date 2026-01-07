from ai.models.groq import oss_120b, llama_8b_instant

# Model configuration for research agent (model, reasoning amount)
RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (llama_8b_instant, None),    # Summarization task
    "research_classifier": (llama_8b_instant, None),    # Classification task
    "extract_text": (llama_8b_instant, None),           # Text extraction task
    "plan_research": (oss_120b, None),                  # High complexity planning + structured output task
    "write_response": (oss_120b, None)                  # Moderate complexity evidence synthesis task
}
