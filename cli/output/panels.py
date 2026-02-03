from rich.markdown import Markdown
from rich.panel import Panel


def ai_bubble(text: str) -> Panel:
    """Create a chat bubble for AI responses."""

    return Panel(
        Markdown(text),
        title="Cogito",
        border_style="white",
        style="yellow3",
        padding=(1, 2),
    )

def system_panel(text: str) -> Panel:
    """Create a system message panel."""

    return Panel(
        text,
        border_style="dim",
        padding=(0, 0),
    )
