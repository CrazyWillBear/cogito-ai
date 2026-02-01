from rich import box
from rich.markdown import Markdown, TableElement
from rich.panel import Panel
from rich.table import Table


# Give markdown tables some breathing room before drawing chat bubbles.
def _patch_markdown_tables() -> None:
    if getattr(TableElement, "_cogito_patched", False):
        return

    def _rich_console(self, console, options):
        table = Table(
            box=box.SIMPLE,
            pad_edge=False,
            padding=(1, 2),
            style="markdown.table.border",
            show_edge=True,
            collapse_padding=True,
        )

        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                table.add_column(heading)

        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)

        yield table

    TableElement.__rich_console__ = _rich_console
    TableElement._cogito_patched = True


_patch_markdown_tables()


def ai_bubble(text: str) -> Panel:
    """Create a chat bubble for AI responses."""

    return Panel(
        Markdown(text),
        title="Cogito",
        border_style="white",
        padding=(1, 2),
    )


def system_panel(text: str) -> Panel:
    """Create a system message panel."""

    return Panel(
        text,
        border_style="dim",
        padding=(0, 0),
    )
