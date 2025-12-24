from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.models.model_config import MODEL_CONFIG


def extract_text(resource_text, user_query, citation=None):
    """Extract relevant text from the resource based on the user's query using the given model."""

    # Get configured model
    model, reasoning = MODEL_CONFIG["extract_text"]

    # Construct prompt (system and user message)
    system_msg = SystemMessage(content=(
        "You are a text extraction agent. Extract text relevant to the user's query given the following guidelines:\n"
        "- Total text extracted can be up to half the length of the source text.\n"
        "- Focus on relevant arguments, concepts, and ideas presented.\n"
        "- Only EXTRACT TEXT, do not SUMMARIZE, do not FORMULATE ARGUMENTS, do not address parts of the question "
        "unrelated to the source you've been given (for example, if the question addresses multiple philosophers, only "
        "extract text relevant to the philosopher from which the source is written by or about).\n\n"
        "User query:\n"
        f"{user_query}\n"
    ))

    user_msg = HumanMessage(content=(
        "Here is the source text:\n"
        f"{resource_text}"
    ))

    # Invoke model and return extracted output
    res = model.invoke([system_msg, user_msg], reasoning={"effort": reasoning})
    citation = '; '.join(resource_text.split('\n')[-2:])  # Extract citation from resource text
    return (gpt_extract_content(res) + '\n' + citation).strip(), citation
