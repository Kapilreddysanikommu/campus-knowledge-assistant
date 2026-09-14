"""
Manual test script for Step 5 reranking.

Runs a question through Step 4's hybrid retrieval, then reranks the
result list with a cross-encoder, printing both orderings so they can be
compared directly.

Example:
    python scripts/test_reranking.py "what does a WU grade mean"
"""

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedder import ChunkEmbedder
from src.retrieval.models import SearchResult
from src.retrieval.reranker import ResultReranker
from src.retrieval.search import DEFAULT_TOP_K, combine_search_results, full_text_search, semantic_search
from src.storage.database import get_connection

DEFAULT_QUESTION = "what does a WU grade mean"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hybrid retrieval order before and after cross-encoder reranking."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to retrieve and rerank (default: {DEFAULT_TOP_K})",
    )
    return parser.parse_args()


def print_results(label: str, results: List[SearchResult]) -> None:
    print(label)
    if not results:
        print("  (no results)\n")
        return

    for rank, result in enumerate(results, start=1):
        matched_by_label = " + ".join(result.matched_by)
        text_preview = " ".join(result.chunk_text.split())[:150]
        print(
            f"  {rank}. score={result.score:.4f} matched_by=[{matched_by_label}] "
            f'document="{result.document_title}" chunk_index={result.chunk_index}'
        )
        print(f"     {text_preview}...")
    print()


def print_rank_changes(before_results: List[SearchResult], after_results: List[SearchResult]) -> None:
    before_rank_by_chunk_id = {result.chunk_id: rank for rank, result in enumerate(before_results, start=1)}

    print("Rank changes (hybrid rank -> reranked rank):")
    for after_rank, result in enumerate(after_results, start=1):
        before_rank = before_rank_by_chunk_id[result.chunk_id]

        if after_rank < before_rank:
            movement = f"moved up {before_rank - after_rank}"
        elif after_rank > before_rank:
            movement = f"moved down {after_rank - before_rank}"
        else:
            movement = "unchanged"

        print(
            f'  document="{result.document_title}" chunk_index={result.chunk_index}: '
            f"{before_rank} -> {after_rank} ({movement})"
        )
    print()


def main() -> None:
    args = parse_arguments()

    connection = get_connection()
    try:
        embedder = ChunkEmbedder()

        print(f'Question: "{args.question}"\n')

        semantic_results = semantic_search(connection, embedder, args.question, top_k=args.top_k)
        full_text_results = full_text_search(connection, args.question, top_k=args.top_k)
        hybrid_results = combine_search_results(semantic_results, full_text_results, top_k=args.top_k)
        print_results("Hybrid retrieval order (before reranking):", hybrid_results)

        reranker = ResultReranker()
        reranked_results = reranker.rerank(args.question, hybrid_results)
        print_results("Cross-encoder reranked order (after reranking):", reranked_results)

        print_rank_changes(hybrid_results, reranked_results)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
