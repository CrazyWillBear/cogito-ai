from ai.models.gpt import gpt5_nano, gpt5, gpt5_mini

# Model configuration for graph nodes
MODEL_CONFIG = {
    "create_conversation": gpt5_nano,               # Conversation summarization task
    "query_vector_db": gpt5_nano,                   # Text extraction task
    "write_queries": gpt5_mini,                     # Moderate complexity structured output task
    "assess_resources_classifier": gpt5_nano,       # Light reasoning + classification task
    "assess_resources_feedback": gpt5_nano,         # Research feedback task
    "write_response": gpt5_mini                # High complexity response generation task
}
