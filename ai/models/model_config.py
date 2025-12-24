from ai.models.gpt import gpt5_nano, gpt5_mini


# Model configuration for graph agent_assigner (model, reasoning amount)
MODEL_CONFIG = {
    "create_conversation": (gpt5_nano, "minimal"),            # Conversation summarization task
    "research_classifier": (gpt5_nano, "low"),                # Research need classification task
    "extract_text": (gpt5_nano, "minimal"),                   # Text extraction task
    "write_queries": (gpt5_mini, "minimal"),                      # Moderate complexity structured output task
    "assess_resources_classifier": (gpt5_nano, "minimal"),    # Light reasoning + classification task
    "assess_resources_feedback": (gpt5_nano, "minimal"),      # Research feedback task
    "write_response": (gpt5_mini, "minimal")                  # High complexity response generation task
}
