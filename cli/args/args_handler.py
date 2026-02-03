import argparse

from rich.console import Console

from cli.args.commands.conversation import start_new_conversation, resume_conversation
from cli.args.commands.list_conversations import list_conversations
from cli.output.patch_markdown_tables import patch_markdown_tables


def execute_args(console: Console):
    """Execute command-line argument handling."""

    # Get args
    args = _parse_args()
    patch_markdown_tables()

    if args.list:
        list_conversations(console)
        exit(0)

    if not args.conversation:
        start_new_conversation(console)
    else:
        resume_conversation(console, args.conversation[0])

def _parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Cogito AI: An agentic chatbot for philosophy research.")

    # Arguments / flags
    parser.add_argument(
        "-c",
        "--conversation",
        type=int,
        help="Loads a conversation given its ID",
        nargs=1,
        metavar="CONVERSATION_ID",
        action="store",
        default=None
    )
    parser.add_argument(
        "-l",
        "--list",
        help="Lists conversations and their IDs",
        action="store_true",
        default=False
    )

    return parser.parse_args()
