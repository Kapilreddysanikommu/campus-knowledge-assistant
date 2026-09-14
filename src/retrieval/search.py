"""
Retrieval functions for finding relevant chunks.

semantic_search finds chunks whose meaning is close to the query, using
vector embeddings. full_text_search finds chunks whose words match the
query, using PostgreSQL's built-in text search. combine_search_results
merges the two ranked lists into one, so callers get the benefit of both:
semantic search catches paraphrased or reworded questions, and full-text
search catches exact terms (names, numbers, codes) that an embedding model
can sometimes blur together.
"""

from typing import List

from psycopg2.extensions import connection as PostgresConnection

from src.embedding.embedder import ChunkEmbedder
from src.retrieval.models import SearchResult

DEFAULT_TOP_K = 5

# Constant used by Reciprocal Rank Fusion, the technique combine_search_results
# uses to merge rankings. Larger values shrink the gap in score between a
# rank-1 and a rank-10 result. 60 is the value commonly used in the
# information retrieval literature and is a reasonable default here.
DEFAULT_RRF_K = 60

SEMANTIC_SEARCH_QUERY = """
    SELECT
        chunks.id,
        chunks.document_id,
        documents.title,
        chunks.chunk_index,
        chunks.chunk_text,
        chunks.embedding <=> %s::vector AS distance
    FROM chunks
    JOIN documents ON documents.id = chunks.document_id
    WHERE chunks.embedding IS NOT NULL
    ORDER BY chunks.embedding <=> %s::vector
    LIMIT %s
"""

FULL_TEXT_SEARCH_QUERY = """
    SELECT
        chunks.id,
        chunks.document_id,
        documents.title,
        chunks.chunk_index,
        chunks.chunk_text,
        ts_rank(chunks.search_vector, websearch_to_tsquery('english', %s)) AS rank
    FROM chunks
    JOIN documents ON documents.id = chunks.document_id
    WHERE chunks.search_vector @@ websearch_to_tsquery('english', %s)
    ORDER BY rank DESC
    LIMIT %s
"""


def semantic_search(
    connection: PostgresConnection,
    embedder: ChunkEmbedder,
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[SearchResult]:
    """Find the top_k chunks whose embeddings are closest to the query's embedding.

    Uses pgvector's cosine distance operator (<=>). Since embeddings are
    normalized, distance ranges from 0 (same direction, most similar) to 2
    (opposite direction), so it is converted to a similarity score
    (1 - distance) for easier reading.
    """
    query_embedding = embedder.embed_text(query_text)

    with connection.cursor() as cursor:
        cursor.execute(SEMANTIC_SEARCH_QUERY, (query_embedding, query_embedding, top_k))
        rows = cursor.fetchall()

    return [
        SearchResult(
            chunk_id=row[0],
            document_id=row[1],
            document_title=row[2],
            chunk_index=row[3],
            chunk_text=row[4],
            score=1.0 - row[5],
            matched_by=["semantic"],
        )
        for row in rows
    ]


def full_text_search(
    connection: PostgresConnection,
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[SearchResult]:
    """Find the top_k chunks whose text best matches the query's search terms.

    Uses PostgreSQL full-text search: websearch_to_tsquery parses query_text
    the way a search engine would (splitting into words, dropping common
    stop words like "the" and "is"), and ts_rank scores how well each
    chunk's search_vector matches those words.
    """
    with connection.cursor() as cursor:
        cursor.execute(FULL_TEXT_SEARCH_QUERY, (query_text, query_text, top_k))
        rows = cursor.fetchall()

    return [
        SearchResult(
            chunk_id=row[0],
            document_id=row[1],
            document_title=row[2],
            chunk_index=row[3],
            chunk_text=row[4],
            score=row[5],
            matched_by=["full_text"],
        )
        for row in rows
    ]


def combine_search_results(
    semantic_results: List[SearchResult],
    full_text_results: List[SearchResult],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[SearchResult]:
    """Merge semantic and full-text results into one ranked list, without duplicates.

    Semantic similarity scores and full-text rank scores are on different,
    incomparable scales, so they cannot simply be sorted together. Instead
    this uses Reciprocal Rank Fusion: each chunk's position in a result list
    (1st, 2nd, and so on) contributes 1 / (rrf_k + rank) to its fused score.
    A chunk that appears in both lists accumulates points from both, which
    naturally merges it into a single entry instead of listing it twice.
    """
    fused_scores = {}
    result_by_chunk_id = {}
    matched_by_chunk_id = {}

    for result_list in (semantic_results, full_text_results):
        for rank, result in enumerate(result_list, start=1):
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            result_by_chunk_id.setdefault(result.chunk_id, result)
            matched_by_chunk_id.setdefault(result.chunk_id, [])
            matched_by_chunk_id[result.chunk_id].extend(result.matched_by)

    combined_results = [
        SearchResult(
            chunk_id=result_by_chunk_id[chunk_id].chunk_id,
            document_id=result_by_chunk_id[chunk_id].document_id,
            document_title=result_by_chunk_id[chunk_id].document_title,
            chunk_index=result_by_chunk_id[chunk_id].chunk_index,
            chunk_text=result_by_chunk_id[chunk_id].chunk_text,
            score=fused_score,
            matched_by=matched_by_chunk_id[chunk_id],
        )
        for chunk_id, fused_score in fused_scores.items()
    ]

    combined_results.sort(key=lambda result: result.score, reverse=True)
    return combined_results[:top_k]
