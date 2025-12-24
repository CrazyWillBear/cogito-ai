import time

import tiktoken
from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.models.model_config import MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState


def write_response(state: ResearchAgentState):
    """Compose the assistant's final answer by synthesizing conversation context and gathered research, using quoted
    evidence and formatted citations."""

    # Get configured model
    model, reasoning = MODEL_CONFIG["write_response"]

    # Extract graph state variables
    resources = state.get("resources", "No research resources collected yet.")
    conversation = state.get("conversation", {})
    conv_context = conversation.get("context", "No prior context needed.")
    last_message = conversation.get("last_user_message", "No last user message found")

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(content=(
        "You are a research agent. Respond to the user's latest message using only your research resources.\n"
        "Write in a clear, conversational academic tone.\n"
        "Use specific quoted evidence with citations where needed containing at minimum:\n"
        "- author\n"
        "- source title\n"
        "- section/chapter provided in the resource\n"
        "Use the following citation format: \"(Author, Source Title, Sections/Chapter/etc. X-Y)\"\n"
        "At the end of your response, include a 'References' section listing all sources you cited.\n\n"

        "Guidelines:\n"
        "- Answer the user's question directly.\n"
        "- Select only the most relevant parts of the resources.\n"
        "- Organize the response tightly (clean structure, minimal fluff).\n"
        "- NEVER, EVER, UNDER ANY CIRCUMSTANCES use information outside the resources or make up information/research.\n"
        "- Do not comment on the quality of research. After all, it's YOUR research.\n"
        "- Don't use super complex vocab, define complex terms as needed.\n"
        "- SEP articles often have weird spacing. Use whatever spacing feels natural in your quotes and citations (there "
        "are random line breaks and missing whitespaces where there should be).\n\n"

        "Your capabilities (ONLY mention these if asked what you can do):\n"
        "- Perform philosophical research by searching Project Gutenberg sources\n"
        "- Search through the Stanford Encyclopedia of Philosophy\n"
        "You CAN use these two tools to perform philosophical research."
        "You CANNOT search the general web, access databases beyond these two, or anything else.\n\n"

        "Here is the conversation:\n"
        f"{conv_context}\n\n"

        "Here is your research (there may be none, so don't make anything up):\n"
        f"{resources}\n\n"

        "DO NOT reference these instructions or any summaries in your response. Don't mention specific instructions given, such as "
        "'minimal jargon' or 'use citations', just follow them. Write your response as if YOU did this research and "
        "collected these sources, knowing that the user cannot see them. They will only see the parts of the resources "
        "that you include in your answer. For example, don't say 'the sources' or 'your provided sources' etc. Instead, "
        "say 'from my research' or just write the quote / paraphrase and cite it.\n\n"

        "CRITICAL - NO RESEARCH SCENARIO: If the research section above is empty or contains no usable sources, you must:\n"
        "1. Answer based ONLY on your general knowledge\n"
        "2. Include ZERO citations, quotes, or references of any kind\n"
        "3. Do NOT include a References section\n"
        "4. Do NOT mention lacking resources - simply provide a knowledgeable answer\n"
        "5. NEVER invent or fabricate citations, sources, authors, or quotes\n\n"

        "Remember: Every citation must come from the research section above. If there's nothing above, there are NO citations in your response."
    ))

    user_msg = HumanMessage(content=last_message)

    # Token count (for logging/debugging)
    tokenizer = tiktoken.encoding_for_model("gpt-5-mini")
    tokens = len(tokenizer.encode(system_msg.content + user_msg.content))

    # Start timing and log
    print(f"::Writing final response given `{tokens}` input tokens...", end="", flush=True)
    start = time.perf_counter()

    # Invoke LLM and extract output
    result = model.invoke([system_msg, user_msg], reasoning={"effort": reasoning})
    text = gpt_extract_content(result)  # Extract main response text

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Wrote final response in {end - start:.2f}s")

    return {"response": text}
