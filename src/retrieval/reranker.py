"""
Reranks retrieval results using a cross-encoder model.

The bi-encoder in src/embedding embeds the query and each chunk
separately, which is what makes semantic search fast enough to run over
an entire chunk collection. A cross-encoder instead reads the query and a
single chunk together in one forward pass and outputs one relevance score
for that pair. This is more accurate, because the model can directly
compare specific words and phrases between the two texts, but it is too
slow to run over a whole collection, so it is used here only to re-score
a short list of candidates that retrieval has already narrowed down.
"""

from typing import List

from sentence_transformers import CrossEncoder

from src.retrieval.models import SearchResult

DEFAULT_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ResultReranker:
    """Wraps a cross-encoder model for reranking a list of search results."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        """Return results re-sorted by cross-encoder relevance score, highest first.

        matched_by is carried over unchanged from the input results, so
        callers can still see which search method(s) originally found each
        chunk after reranking has reordered them.
        """
        if not results:
            return []

        query_chunk_pairs = [(query_text, result.chunk_text) for result in results]
        relevance_scores = self.model.predict(query_chunk_pairs)

        reranked_results = [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_title=result.document_title,
                chunk_index=result.chunk_index,
                chunk_text=result.chunk_text,
                score=float(relevance_score),
                matched_by=result.matched_by,
            )
            for result, relevance_score in zip(results, relevance_scores)
        ]

        reranked_results.sort(key=lambda result: result.score, reverse=True)
        return reranked_results
