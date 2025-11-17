from ai.models.gpt import gpt5, gpt5_mini, gpt5_nano


# Model config for individual nodes
MODEL_CONFIG = {
    "write_queries": gpt5_nano,
    "assess_resources": gpt5_nano,
    "assess_summary": gpt5_nano,
    "summarize": gpt5_mini
}
