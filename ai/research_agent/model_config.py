from ai.models.groq import llama_8b_instant, llama_4_scout, oss_20b_high_temp_med_reasoning, oss_120b_med, \
    oss_20b_low_temp

RESEARCH_AGENT_MODEL_CONFIG = {
    "create_conversation": llama_8b_instant,                        # Summarization task
    "research_classifier": oss_20b_low_temp,                        # Slightly nuanced classification task
    "extract_text": llama_8b_instant,                               # Text extraction task
    "plan_research": llama_4_scout,                                 # Moderate complexity planning + structured output task
    "write_response_no_research": oss_20b_high_temp_med_reasoning,  # Moderate complexity evidence synthesis task
    "write_response_research": oss_120b_med                         # Moderate complexity evidence synthesis task
}
