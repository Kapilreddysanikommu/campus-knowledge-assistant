"""
Shared data structure returned by every retrieval function.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchResult:
    """One retrieved chunk, along with the score and method that found it."""

    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    chunk_text: str
    score: float
    matched_by: List[str] = field(default_factory=list)
