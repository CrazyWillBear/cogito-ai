from rich.console import Console

from cli.conversations.agent_loop import conversation_loop
from cli.conversations.conversations import user_select_conversation, get_conversation_by_id
from cli.db_containers import manage_containers


def _start_docker_containers(console: Console):
    """Setup Docker containers."""

    with console.status("[dim]Starting dbs...[/dim]", spinner="clock") as status:
        manage_containers(status)

def start_new_conversation(console: Console):
    """Start a new conversation with user-selected conversation."""

    _start_docker_containers(console)
    conversation = user_select_conversation(console)
    conversation_loop(console, conversation)

def resume_conversation(console: Console, conversation_id: int):
    """Resume an existing conversation."""

    _start_docker_containers(console)
    conversation = get_conversation_by_id(conversation_id)

    if not conversation:
        console.print(f"[bold red]Error:[/bold red] [gold3]No conversation found with ID[/gold3] {conversation_id}.")
        return

    conversation_loop(console, conversation)