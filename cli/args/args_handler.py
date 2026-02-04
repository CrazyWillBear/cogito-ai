import argparse

from rich.console import Console

from cli.args.commands.conversation import resume_conversation, user_option_conversation, start_new_conversation, \
    delete_conversation
from cli.args.commands.list_conversations import list_conversations
from cli.output.patch_markdown_tables import patch_markdown_tables


def execute_args(console: Console):
    """Execute command-line argument handling."""

    # Get args
    args = _parse_args()
    patch_markdown_tables()

    # If list flag is set
    if args.list:
        list_conversations(console)
        return

    # If new conversation flag is set
    if args.new is True:
        start_new_conversation(console, name=None)
        return
    elif args.new is not False:  # means they passed a name
        start_new_conversation(console, name=args.new)
        return

    # If resume conversation flag is set
    if args.conversation is not None:  # if they passed a conversation id
        resume_conversation(console, args.conversation)
        return

    # If delete conversation flag is set
    if args.delete is not None:
        delete_conversation(console, args.delete)
        return

    # if they didn't pass anything
    user_option_conversation(console)
    return

def _parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    # Arguments / flags
    parser.add_argument(
        "-c",
        "--conversation",
        type=int,
        help="loads a conversation given its ID",
        metavar="CONVERSATION_ID",
        action="store",
        default=None
    )
    parser.add_argument(
        "-l",
        "--list",
        help="lists conversations and their IDs",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-n",
        "--new",
        help="starts a new conversation, name optional",
        nargs="?",
        const=True,
        metavar="NAME",
        default=False
    )
    parser.add_argument(
        "-d",
        "--delete",
        help="deletes a conversation given its ID",
        type=int,
        metavar="CONVERSATION_ID",
        action="store"
    )

    return parser.parse_args()
