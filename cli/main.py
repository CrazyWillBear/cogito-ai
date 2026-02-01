import json
import time

from langchain_core.messages import messages_to_dict, HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ai.research_agent.ResearchAgent import ResearchAgent
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
from cli.conversations import write_logs
from cli.db_containers import manage_containers
from cli.panels import ai_bubble, system_panel

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


def main():
    """Main CLI loop for interacting with the Research Agent."""

    console = Console()
    console.clear()
    console.print(Panel(START_TEXT, border_style="dim", width=console.width))

    # Setup Docker containers
    with console.status("[dim]Starting dbs...[/dim]", spinner="clock") as status:
        manage_containers(status)

    # Conversation state
    conversation = []

    # Build agent
    agent = ResearchAgent()
    agent.build()

    print()
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
            with console.status("[dim]thinking…[/dim]", spinner="clock") as status:
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
                        sources.add(
                            "\t- " + raw_result[1].get("source", "Unknown Source") + ": " + raw_result[1].get("title",
                                                                                                              "Unknown title") + " by " + ", ".join(
                                raw_result[1].get("authors", [])))

            resources = "\n".join(sources) if sources else "No resources"

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

            conversation.append(AIMessage(content=txt_out))

    except Exception as e:
        print(f"::Error while running agent: {e}")

    finally:
        # ---- Cleanup ------------------------------------------------------
        agent.close()
        write_logs(json.dumps(messages_to_dict(conversation), indent=4))

        console.print(system_panel("Session ended. Logs written to disk."))
