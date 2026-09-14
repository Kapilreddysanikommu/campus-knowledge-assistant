"""
Shared embedding settings.

Kept separate from embedder.py so that other modules (such as the database
schema, which needs the vector dimension to define the embedding column)
can read these values without importing sentence-transformers, which is a
slow, heavy import.
"""

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 always produces 384-dimensional vectors. If the model
# name above is ever changed, this must be updated to match, since the
# database column width is fixed to this value.
EMBEDDING_DIMENSION = 384
