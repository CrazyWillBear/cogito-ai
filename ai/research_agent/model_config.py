from ai.models.groq import llama_8b_instant, llama_4_scout, oss_120b_low, oss_20b_med_temp_med_reasoning, oss_120b_med

# Model configuration for research agent (model, reasoning amount)
RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": (llama_8b_instant, None),    # Summarization task
    "research_classifier": (llama_8b_instant, None),    # Slightly nuanced classification task
    "extract_text": (llama_8b_instant, None),           # Text extraction task
    "plan_research": (llama_4_scout, None),          # Moderate complexity planning + structured output task
    "write_response_no_research": (oss_20b_med_temp_med_reasoning, None),  # Moderate complexity evidence synthesis task
    "write_response_research": (oss_120b_med, None)     # Moderate complexity evidence synthesis task
}
