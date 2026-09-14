"""
Manual test script for Step 7 staleness detection.

Part 1 runs a question through hybrid retrieval and reranking directly,
then shows the result list before and after staleness detection is
applied, so you can see whether an older document's chunk was moved below
a newer one on the same topic, and read the notes explaining why.

Part 2 sends the same question to the real /query endpoint to confirm the
same behavior is wired into the actual pipeline, not just a standalone
function.

Example:
    python scripts/test_staleness_detection.py "what is SJSU's basic grading system"
"""

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.access_control import get_allowed_access_levels
from src.api.main import app, embedder, reranker
from src.retrieval.models import SearchResult
from src.retrieval.search import DEFAULT_TOP_K, combine_search_results, full_text_search, semantic_search
from src.retrieval.staleness import StalenessNote, detect_and_apply_staleness
from src.storage.database import get_connection

DEFAULT_QUESTION = "what is SJSU's basic grading system"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval results before and after staleness detection."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to search for")
    parser.add_argument(
        "--role",
        default="student",
        choices=["student", "faculty", "admin"],
        help="Role to search as (default: student)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to retrieve (default: {DEFAULT_TOP_K})",
    )
    return parser.parse_args()


def print_results(label: str, results: List[SearchResult]) -> None:
    print(label)
    if not results:
        print("  (no results)\n")
        return

    for rank, result in enumerate(results, start=1):
        text_preview = " ".join(result.chunk_text.split())[:150]
        print(
            f'  {rank}. score={result.score:.4f} document="{result.document_title}" '
            f"academic_year={result.academic_year} chunk_index={result.chunk_index}"
        )
        print(f"     {text_preview}...")
    print()


def print_notes(notes: List[StalenessNote]) -> None:
    if not notes:
        print("Staleness notes: none\n")
        return

    print("Staleness notes:")
    for note in notes:
        print(f"  - {note.message}")
    print()


def main() -> None:
    args = parse_arguments()
    allowed_access_levels = get_allowed_access_levels(args.role)

    print(f'Question: "{args.question}"')
    print(f"Role: {args.role} (allowed access levels: {allowed_access_levels})\n")

    print("=== Part 1: direct pipeline comparison ===\n")
    connection = get_connection()
    try:
        semantic_results = semantic_search(
            connection, embedder, args.question, allowed_access_levels, top_k=args.top_k
        )
        full_text_results = full_text_search(
            connection, args.question, allowed_access_levels, top_k=args.top_k
        )
        hybrid_results = combine_search_results(semantic_results, full_text_results, top_k=args.top_k)
        reranked_results = reranker.rerank(args.question, hybrid_results)
    finally:
        connection.close()

    print_results("Without staleness detection (reranked order):", reranked_results)

    staleness_adjusted_results, notes = detect_and_apply_staleness(reranked_results)
    print_results("With staleness detection (after recency adjustment):", staleness_adjusted_results)
    print_notes(notes)

    print("=== Part 2: same comparison through the real /query endpoint ===\n")
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"question": args.question, "top_k": args.top_k},
        headers={"X-User-Role": args.role},
    )
    response.raise_for_status()
    response_data = response.json()

    print(f"Endpoint results ({len(response_data['results'])} returned):")
    for rank, result in enumerate(response_data["results"], start=1):
        print(f'  {rank}. score={result["score"]:.4f} document="{result["document_title"]}"')
    print()

    print("Endpoint staleness notes:")
    if response_data["staleness_notes"]:
        for note in response_data["staleness_notes"]:
            print(f"  - {note}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
