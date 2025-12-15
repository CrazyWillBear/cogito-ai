import os

from openai import OpenAI


class Embedder:
    """Embedder using OpenAI's text-embedding-3-small model."""

    # --- Methods ---
    def __init__(self):
        """Initialize OpenAI client."""

        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def embed_batch(self, texts: list[str]):
        """Embed a list of texts into dense vectors using text-embedding-3-small model."""

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        out = [d.embedding for d in response.data]

        return out
