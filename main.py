import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ai.subgraphs.research_agent.research_agent import ResearchAgent

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
                          
AI Philosophical Research Assistant
Type 'exit' or 'quit' to end the conversation.

Created by William Chastain (williamchastain.com)

-=-=-=-

"""

if __name__ == "__main__":
    print(START_TEXT)

    # Conversation setup
    conversation = {
        "messages": [
            SystemMessage(content="You are a helpful philosophical research assistant.")
        ]
    }

    # Build agent
    agent = ResearchAgent()
    agent.build()

    while True:
        user_input = input("[User]: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        conversation["messages"].append(HumanMessage(content=user_input))
        print()

        start = time.perf_counter()

        # Run agent
        output = agent.run(conversation)
        end = time.perf_counter()
        print(f"\r\033[K[AI]:\n---\n{output}\n---\nTime was {end - start:.4f}s\n\n")

        conversation["messages"].append(AIMessage(content=output))
