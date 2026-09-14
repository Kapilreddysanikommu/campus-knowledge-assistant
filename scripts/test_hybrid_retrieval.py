"""
Manual test script for Step 4 hybrid retrieval.

Runs one question through semantic search, full-text search, and the
merged hybrid list, printing all three so they can be compared side by
side.

Example:
    python scripts/test_hybrid_retrieval.py "what is the grading breakdown"
"""

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedder import ChunkEmbedder
from src.retrieval.models import SearchResult
from src.retrieval.search import (
    ALL_ACCESS_LEVELS,
    DEFAULT_TOP_K,
    combine_search_results,
    full_text_search,
    semantic_search,
)
from src.storage.database import get_connection

DEFAULT_QUESTION = "what is the grading breakdown"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare semantic search, full-text search, and combined hybrid retrieval for one question."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to show per search type (default: {DEFAULT_TOP_K})",
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


def main() -> None:
    args = parse_arguments()

    connection = get_connection()
    try:
        embedder = ChunkEmbedder()

        print(f'Question: "{args.question}"\n')

        semantic_results = semantic_search(
            connection, embedder, args.question, ALL_ACCESS_LEVELS, top_k=args.top_k
        )
        print_results("Semantic search results:", semantic_results)

        full_text_results = full_text_search(connection, args.question, ALL_ACCESS_LEVELS, top_k=args.top_k)
        print_results("Full-text search results:", full_text_results)

        combined_results = combine_search_results(semantic_results, full_text_results, top_k=args.top_k)
        print_results("Combined hybrid results:", combined_results)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
