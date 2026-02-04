import os

from rich.console import Console

from cli.conversations.agent_loop import conversation_loop
from cli.conversations.conversations import user_select_conversation, get_conversation_by_id, get_new_conversation_id, \
    Conversation, CONVERSATIONS_DIR
from cli.db_containers import manage_containers


def _start_docker_containers(console: Console):
    """Setup Docker containers."""

    with console.status("[dim]Starting dbs...[/dim]", spinner="clock") as status:
        manage_containers(status)

def user_option_conversation(console: Console):
    """Start a new conversation with user-selected conversation."""

    _start_docker_containers(console)
    conversation = user_select_conversation(console)
    conversation_loop(console, conversation)

def delete_conversation(console: Console, conversation_id: int):
    """Delete an existing conversation."""

    conversation = get_conversation_by_id(conversation_id)

    if not conversation:
        console.print(f"[bold red]Error:[/bold red] [gold3]No conversation found with ID[/gold3] {conversation_id}.")
        return

    # Delete the conversation file
    path = CONVERSATIONS_DIR / f"conversation-{conversation_id}.json"
    try:
        os.remove(path)
        console.print(f"[bold green]Success:[/bold green] [gold3]Deleted conversation with ID[/gold3] {conversation_id}.")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] [gold3]Failed to delete conversation with ID[/gold3] {conversation_id}.")
        console.print(f"[bold red]Exception:[/bold red] {e}")

def start_new_conversation(console: Console, name: str | None):
    """Start a new conversation with user-selected conversation."""

    _start_docker_containers(console)
    if name is not None:
        conversation_id = get_new_conversation_id()
        conversation: Conversation = {
            "id": conversation_id,
            "name": name,
            "conversation": []
        }
        conversation_loop(console, conversation)
    else:
        conversation_loop(console, None)

def resume_conversation(console: Console, conversation_id: int):
    """Resume an existing conversation."""

    _start_docker_containers(console)
    conversation = get_conversation_by_id(conversation_id)

    if not conversation:
        console.print(f"[bold red]Error:[/bold red] [gold3]No conversation found with ID[/gold3] {conversation_id}.")
        return

    conversation_loop(console, conversation)