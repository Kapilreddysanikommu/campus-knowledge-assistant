"""
Request and response models for the API.
"""

from typing import List

from pydantic import BaseModel

from src.retrieval.search import DEFAULT_TOP_K


class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class QueryResultItem(BaseModel):
    document_title: str
    chunk_index: int
    chunk_text: str
    score: float
    matched_by: List[str]


class QueryResponse(BaseModel):
    question: str
    role: str
    allowed_access_levels: List[str]
    results: List[QueryResultItem]
    staleness_notes: List[str]
    answer: str
    has_sufficient_information: bool
