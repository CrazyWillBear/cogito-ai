import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, messages_to_dict

from ai.research_agent.ResearchAgent import ResearchAgent
from util.SpinnerController import SpinnerController

START_TEXT = \
r"""
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

-=-=-=-
"""

def write_logs(logs: str):
    """Helper function to write logs to console."""


    p = Path(f"logs/agent_logs_recent.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(logs, encoding="utf-8")

if __name__ == "__main__":
    # Main entry point for running the Cogito AI research assistant in a console loop.

    print(START_TEXT)

    # Conversation setup
    conversation = []

    # Build agent
    spinner_controller = SpinnerController()
    agent = ResearchAgent(spinner_controller=spinner_controller)
    agent.build()

    # Main loop
    while True:
        # Get user input
        user_input = input("[User]: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        conversation.append(HumanMessage(content=user_input))
        print()

        # Run agent with timing
        start = time.perf_counter()       # start timing
        output = agent.run(conversation)  # invoke/run agent
        txt_out = output.get('response', 'No response available')
        effort = output.get('research_effort', 'Unknown')
        spinner_controller.stop_spinner() # stop spinner
        end = time.perf_counter()         # end timing

        # Print output and time taken
        print(f"[AI]:                            "  # to deal with spinner overwrite
              f"\n"
              f"---\n"
              f"{txt_out}\n"
              f"---\n"
              f"Time was {end - start:.2f}s - Research level {effort}\n"
        )

        # Append AI message to conversation
        conversation.append(AIMessage(content=txt_out))

    # Close agent resources
    agent.close()

    # Print final conversation as dict
    write_logs(json.dumps(messages_to_dict(conversation), indent=4))
