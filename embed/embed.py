from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once globally (don’t re-load inside the function!)
_embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")


def embed(text: str) -> np.ndarray:
    """Embed text into a dense vector using BAAI/bge-large-en-v1.5."""
    # Model expects a list of sentences
    vector = _embed_model.encode([text], normalize_embeddings=True)

    # encode() returns shape (1, dim) — flatten to (dim,)
    return np.array(vector[0], dtype=np.float32)
