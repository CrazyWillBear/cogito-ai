from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt
from rich.align import Align

from ai.research_agent.ResearchAgent import ResearchAgent

console = Console()

def user_bubble(text):
    return Panel(
        Align.left(text),
        title="You",
        border_style="cyan",
    )

def ai_bubble(text):
    return Panel(
        Align.left(text),
        title="Cogito",
        border_style="white",
    )

def main_loop(agent: ResearchAgent):

    conversation = []

    console.clear()
    console.print(Panel("🧠 Cogito is ready", border_style="dim"))

    while True:
        user_text = Prompt.ask("[cyan]▸ You[/cyan]")

        console.print()
        console.print(user_bubble(user_text))
        console.print()

        with console.status("[dim]TwinGPT thinking…[/dim]", spinner="dots") as status:
            agent.run(conversation, status)
            # fake LLM call
            response = "Kant says causation is a category of understanding."

        console.print(ai_bubble(response))
        console.print()