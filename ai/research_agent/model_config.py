from ai.models.groq import llama_8b_instant, oss_20b, oss_120b_low, oss_120b_med

# Model configuration for research agent (model, reasoning amount)
RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (llama_8b_instant, None),    # Summarization task
    "research_classifier": (oss_20b, None),             # Slightly nuanced classification task
    "extract_text": (oss_20b, None),                    # Text extraction task
    "plan_research": (oss_120b_low, None),              # Moderate complexity planning + structured output task
    "write_response_simple": (oss_120b_low, None),      # Moderate complexity evidence synthesis task
    "write_response_deep": (oss_120b_med, None)         # High complexity evidence synthesis task
}
