import os
import threading

import psycopg2
import select
from dotenv import load_dotenv


class Postgres:
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


        self.conversations_table = "conversations"
        self.filters_table = "filters"

        self._conn_params = {"host": host, "port": port, "dbname": dbname, "user": user, "password": password}

        self.conn = psycopg2.connect(**self._conn_params)
        self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        # Dict: author -> list of sources
        self.author_sources: dict[str, list[str]] = {}

        self._update_filters()

        self.listen()

    def close(self):
        """Close the PostgreSQL connection."""

        self.conn.close()

    def listen(self) -> None:
        """Listen for changes in the filters table and update authors and sources accordingly."""

        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

    def _listen_loop(self):
        """Internal loop to listen for PostgreSQL notifications."""

        # Create a separate connection for listening
        listen_conn = psycopg2.connect(**self._conn_params)
        listen_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = listen_conn.cursor()
        cur.execute("LISTEN filters_changes;")

        # Listen for notifications indefinitely
        try:
            while True:
                # Wait for notifications (uses the connection's fileno)
                if select.select([listen_conn], [], [], 1) == ([], [], []):
                    continue

                listen_conn.poll()

                while listen_conn.notifies:
                    _ = listen_conn.notifies.pop(0)
                    self._update_filters()
        finally:
            cur.close()
            listen_conn.close()

    def _update_filters(self) -> None:
        """Update the dict of authors to sources from the database."""

        cur = self.conn.cursor()
        cur.execute(f"SELECT authors, sources FROM {self.filters_table};")
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

        cur.close()

    def get_conversation(self, user_id: str, conversation_id: str):
        """Retrieve conversation data for a given user and conversation ID."""

        cur = self.conn.cursor()
        try:
            cur.execute(
                f"SELECT conversation FROM {self.conversations_table} WHERE user_id = %s AND conversation_id = %s LIMIT 1;",
                (user_id, conversation_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]
        finally:
            cur.close()

    @property
    def all_authors(self) -> list[str]:
        """List of all authors."""

        return sorted(self.author_sources.keys())

    @property
    def all_sources(self) -> list[str]:
        """List of all unique sources."""

        seen: set[str] = set()
        for sources in self.author_sources.values():
            seen.update(sources)
        return sorted(seen)
