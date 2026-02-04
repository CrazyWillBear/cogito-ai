import time

from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ai.research_agent.ResearchAgent import ResearchAgent
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
from cli.conversations.conversations import save_conversation, Conversation, get_new_conversation_id
from cli.output.panels import ai_bubble, system_panel


def conversation_loop(console: Console, conversation: Conversation | None):
    """Main agent conversation loop."""

    console.print("\n::Type 'exit' or 'quit' to end the conversation.\n", style="bold gold3")

    # Build conversation state
    if conversation:
        messages = conversation["conversation"]
        conversation_id = conversation["id"]
        conversation_name = conversation["name"]
    else:
        messages = []
        conversation_id = get_new_conversation_id()
        conversation_name = None

    # Build agent
    agent = ResearchAgent()
    agent.build()

    # ---- Output section -------------------------------------------------
    for msg in messages:
        if isinstance(msg, HumanMessage):
            console.print(f"[bold cyan]▸ You[/bold cyan]: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            console.print(ai_bubble(msg.content))
            console.print()

    try:
        # ---- Main loop -------------------------------------------------------
        while True:
            # Get input
            user_input = Prompt.ask("[bold cyan]▸ You[/bold cyan]")

            if user_input.lower() in {"exit", "quit"}:
                break

            messages.append(HumanMessage(content=user_input))
            console.print()  # New line for spacing

            # Run agent
            start = time.perf_counter()
            with console.status("[dim]thinking…[/dim]", spinner="clock") as status:
                # Allows us to show status updates from within the agent
                output = agent.run(messages, status=status)
            end = time.perf_counter()

            # Handle output
            txt_out = output.get("response", "No response available")
            research_level = output.get("research_effort", "N/A")
            query_results = output.get("query_results", [])

            # Format resources
            sources = set()
            if (research_level == ResearchEffort.DEEP or research_level == ResearchEffort.SIMPLE) and query_results:
                for res in query_results:
                    raw_result = res.get("result")
                    if type(raw_result) == tuple:
                        sources.add(
                            "\t- " + raw_result[1].get("source", "Unknown Source") + ": " + raw_result[1].get("title",
                                                                                                              "Unknown title") + " by " + ", ".join(
                                raw_result[1].get("authors", [])))
            resources = "\n".join(sources) if sources else "No resources"

            # Output response
            console.print(ai_bubble(txt_out))
            text = (
                f"[italic][dim]Time: {end - start:.2f}s - Research Level (0-3): {research_level}\n"
                f"Resources Found:\n{resources}[/dim][/italic]"
            )
            console.print(
                Panel(
                    text,
                    border_style="dim",
                    padding=(0, 0),
                )
            )
            console.print()

            # Update messages
            messages.append(AIMessage(content=txt_out))

    except KeyboardInterrupt:
        console.print("\n::Quitting...", style="dim italic")  # New line for spacing
        return
    except Exception as e:
        print(f"::Error while running agent:\n{e}")

    # ---- Cleanup ------------------------------------------------------
    try:
        agent.close()

        if not conversation_name:
            conversation_name = Prompt.ask("\n::Enter a name for this conversation", default=f"Conversation {conversation_id}")

        messages_dict = messages_to_dict(messages)
        save_conversation(messages_dict, conversation_id, conversation_name)
        console.print(system_panel("Session ended. Logs written to disk."))
    except KeyboardInterrupt:
        console.print("\n::Quitting before cleanup...", style="dim italic")  # New line for spacing
        return
    except Exception as e:
        print(f"::Error during cleanup:\n{e}")
