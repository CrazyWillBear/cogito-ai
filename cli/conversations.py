import time
from pathlib import Path


def write_logs(logs: str):
    """Write conversation logs to disk."""

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    p = Path(f"~/.cogito/conversations/conversation-{timestamp}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(logs, encoding="utf-8")
