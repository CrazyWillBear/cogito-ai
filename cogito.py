import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.prompt import Prompt

from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict

from ai.research_agent.ResearchAgent import ResearchAgent
from ai.research_agent.schemas.ResearchEffort import ResearchEffort

START_TEXT = r"""
      # ###
    /  /###  /                            #
   /  /  ###/                            ###        #
  /  ##   ##                              #        ##
 /  ###                                            ##
##   ##           /###        /###      ###      ########    /###
##   ##          / ###  /    /  ###  /   ###    ########    / ###  /
##   ##         /   ###/    /    ###/     ##       ##      /   ###/
##   ##        ##    ##    ##     ##      ##       ##     ##    ##
##   ##        ##    ##    ##     ##      ##       ##     ##    ##
 ##  ##        ##    ##    ##     ##      ##       ##     ##    ##
  ## #      /  ##    ##    ##     ##      ##       ##     ##    ##
   ###     /   ##    ##    ##     ##      ##       ##     ##    ##
    ######/     ######      ########      ### /    ##      ######
      ###        ####         ### ###      ##/      ##      ####
                                   ###
                             ####   ###
                           /######  /#
                          /     ###/

Copyright (c) 2025 William Chastain (williamchastain.com). All rights reserved.
This software is licensed under the PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0)

Cogito AI: An agentic Q&A research assistant for philosophy.
Type 'exit' or 'quit' to end the conversation.
"""

# -----------------------------------------------------------------------------
# Rich helpers
# -----------------------------------------------------------------------------

console = Console()


def ai_bubble(text: str) -> Panel:
    return Panel(
        Align.left(text),
        title="Cogito",
        border_style="white",
        padding=(1, 2),
    )


def system_panel(text: str) -> Panel:
    return Panel(
        text,
        border_style="dim",
        padding=(0, 0),
    )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def write_logs(logs: str):
    p = Path("logs/agent_logs_recent.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(logs, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    console.clear()
    console.print(Panel(START_TEXT, border_style="dim", width=console.width))

    # Conversation state
    conversation = []

    # Build agent
    agent = ResearchAgent()
    agent.build()

    try:
        while True:
            # ---- Input ------------------------------------------------------
            user_input = Prompt.ask("[bold cyan]▸ You[/bold cyan]")

            if user_input.lower() in {"exit", "quit"}:
                break

            conversation.append(HumanMessage(content=user_input))

            console.print()

            # ---- Agent run --------------------------------------------------
            start = time.perf_counter()

            # Use the context manager directly with text and spinner to avoid duplicate spinners
            with console.status("[dim]Thinking…[/dim]", spinner="clock") as status:
                output = agent.run(conversation, status=status)

            end = time.perf_counter()

            # ---- Output -----------------------------------------------------
            txt_out = output.get("response", "No response available")
            research_level = output.get("research_effort", "N/A")
            query_results = output.get("query_results", [])

            sources = set()
            if (research_level == ResearchEffort.DEEP or research_level == ResearchEffort.SIMPLE) and query_results:
                for res in query_results:
                    raw_result = res.get("result")
                    if type(raw_result) == tuple:
                        sources.add("\t- " + raw_result[1].get("source", "Unknown Source") + ": " + raw_result[1].get("title", "Unknown title") + " by " + ", ".join(raw_result[1].get("authors", [])))

            resources = "\n".join(sources) if sources else "No resources"

            console.print(ai_bubble(txt_out))

            text = \
f"""[italic][dim]Time: {end - start:.2f}s - Research Level: {research_level}
Resources Found:\n{resources}[/dim][/italic]"""
            console.print(
                Panel(
                    text,
                    border_style="dim",
                    padding=(0, 0),
                )
            )

            console.print()

            conversation.append(AIMessage(content=txt_out))

    finally:
        # ---- Cleanup ------------------------------------------------------
        agent.close()
        write_logs(json.dumps(messages_to_dict(conversation), indent=4) + str(output.get("query_results", [])))

        console.print(system_panel("Session ended. Logs written to disk."))