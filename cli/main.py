from rich.console import Console

from cli.args.args_handler import execute_args

START_TEXT = r"""
 .--.              _  .-.
: .--'            :_;.' `.
: :    .--.  .--. .-.`. .'.--.
: :__ ' .; :' .; :: : : :' .; :
`.__.'`.__.'`._. ;:_; :_;`.__.'
             .-. :
             `._.'
Copyright (c) 2025 William Chastain (williamchastain.com). All rights reserved.
Code and license at https://github.com/CrazyWillBear/cogito-ai
-=-=-=-=-=-
"""

def main():
    """Main entry point for the Cogito CLI application."""

    # Execute args
    console = Console()
    console.print(START_TEXT, style="bold gold3")
    execute_args(console)
