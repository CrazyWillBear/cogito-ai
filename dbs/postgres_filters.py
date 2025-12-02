import os
import threading

import psycopg2
import select
from dotenv import load_dotenv


class PostgresFilters:
    """Class to manage PostgreSQL filters with real-time updates."""

    # --- Methods ---
    def __init__(self):
        """Initialize the PostgreSQL connection and set up real-time updates."""

        load_dotenv()

        host = os.getenv("COGITO_POSTGRES_HOST")
        port = int(os.getenv("COGITO_POSTGRES_PORT", "5432"))
        dbname = os.getenv("COGITO_POSTGRES_DBNAME")
        user = os.getenv("COGITO_POSTGRES_USER")
        password = os.getenv("COGITO_POSTGRES_PASSWORD")

        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        # Dict: author -> list of sources
        self.author_sources: dict[str, list[str]] = {}

        self._update_filters()

    def close(self):
        """Close the PostgreSQL connection."""

        self.conn.close()

    def listen(self) -> None:
        """Listen for changes in the filters table and update authors and sources accordingly."""

        cur = self.conn.cursor()
        cur.execute("LISTEN filter_changes;")

        while True:
            if select.select([self.conn], [], [], 1) == ([], [], []):
                continue

            self.conn.poll()

            while self.conn.notifies:
                _ = self.conn.notifies.pop(0)
                self._update_filters()

        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()

    def _update_filters(self) -> None:
        """Update the dict of authors to sources from the database."""

        cur = self.conn.cursor()
        cur.execute("SELECT authors, sources FROM filters;")
        rows = cur.fetchall()

        # Build mapping with de-duplication
        tmp: dict[str, set[str]] = {}
        for author, source in rows:
            if author is None or source is None:
                continue
            if author not in tmp:
                tmp[author] = set()
            tmp[author].add(source)

        # Convert sets to sorted lists for stable order
        self.author_sources = {a: sorted(list(sources)) for a, sources in tmp.items()}

    @property
    def all_authors(self) -> list[str]:
        """Backward-compatible list of all authors."""
        return sorted(self.author_sources.keys())

    @property
    def all_sources(self) -> list[str]:
        """Backward-compatible list of all unique sources."""
        seen: set[str] = set()
        for sources in self.author_sources.values():
            seen.update(sources)
        return sorted(seen)
