from rich.align import Align
from rich.console import Console
from rich.table import Table

from cli.conversations.conversations import get_conversations


def list_conversations(console: Console):
    """List all conversations with their IDs."""

    console.print("Existing Conversations:", style="bold yellow2")

    table = Table(
        expand=False,
        padding=(0, 0),
        pad_edge=False,
        style="bold yellow3"
    )
    table.add_column("Conversation Name", header_style="bold yellow2 italic", style="yellow2")
    table.add_column("Conversation ID", header_style="bold yellow2 italic", style="yellow2")

    for conversation in get_conversations():
        table.add_row(f"'{conversation['name']}'", str(conversation['id']))

    console.print(table)

    console.print("*Use the ID to resume a conversation via `-c, --conversation`*\n", style="italic yellow2")
