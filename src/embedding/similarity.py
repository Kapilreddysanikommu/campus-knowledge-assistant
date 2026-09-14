"""
Measures how similar two embedding vectors are.
"""

from typing import List

import numpy as np


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """Return the cosine similarity between two vectors, from -1 to 1.

    A value near 1 means the vectors point in nearly the same direction
    (topically similar text). A value near 0 means they are unrelated.
    """
    array_a = np.asarray(vector_a, dtype=np.float32)
    array_b = np.asarray(vector_b, dtype=np.float32)

    dot_product = np.dot(array_a, array_b)
    magnitude_a = np.linalg.norm(array_a)
    magnitude_b = np.linalg.norm(array_b)

    return float(dot_product / (magnitude_a * magnitude_b))
