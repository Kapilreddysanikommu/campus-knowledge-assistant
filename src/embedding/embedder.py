"""
Generates embeddings for chunk text using a local sentence-transformers model.

The model runs on this machine, so there is no API key and no per-request
cost. The default model, all-MiniLM-L6-v2, is small enough to run on CPU
and produces vectors of size EMBEDDING_DIMENSION (see config.py).
"""

from typing import List

from sentence_transformers import SentenceTransformer

from src.embedding.config import DEFAULT_MODEL_NAME


class ChunkEmbedder:
    """Wraps a sentence-transformers model for embedding chunk text."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding for a single piece of text."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding per input text, computed in a single batch.

        Embeddings are L2-normalized, which is the standard preparation for
        comparing them with cosine similarity.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
