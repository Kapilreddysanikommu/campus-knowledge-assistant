"""
Manual test script for Step 6 role-based access control.

Part 1 calls the retrieval-layer functions directly with each role's
allowed access levels and lists which documents came back, before
combine_search_results or reranking ever run. This is the proof that the
SQL query itself excludes a document outside the caller's allowed access
levels, rather than retrieving it and filtering it out afterward.

Part 2 sends the same question to the real /query endpoint, once as a
student and once as a faculty member, and prints both full result sets so
the end-to-end behavior can be compared directly.

Example:
    python scripts/test_rbac_query.py "what is the process for a grade appeal"
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.access_control import get_allowed_access_levels
from src.api.main import app, embedder
from src.retrieval.search import full_text_search, semantic_search
from src.storage.database import get_connection

DEFAULT_QUESTION = "what is the process for a grade appeal"
FACULTY_ONLY_DOCUMENT_TITLE = "Grading Symbols, Drop and Withdrawal Policy"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare /query results for a student role versus a faculty role."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to request (default: 5)")
    return parser.parse_args()


def print_raw_candidate_documents(role: str, connection, question: str, top_k: int) -> None:
    allowed_access_levels = get_allowed_access_levels(role)
    semantic_results = semantic_search(connection, embedder, question, allowed_access_levels, top_k=top_k)
    full_text_results = full_text_search(connection, question, allowed_access_levels, top_k=top_k)

    candidate_titles = sorted({result.document_title for result in semantic_results + full_text_results})

    print(f"Role: {role}")
    print(f"  Allowed access levels: {allowed_access_levels}")
    print(f"  Documents returned as raw SQL candidates: {candidate_titles}")
    if FACULTY_ONLY_DOCUMENT_TITLE in candidate_titles:
        print(f'  Includes "{FACULTY_ONLY_DOCUMENT_TITLE}"')
    else:
        print(f'  Does NOT include "{FACULTY_ONLY_DOCUMENT_TITLE}"')
    print()


def print_query_response(role: str, response_data: dict) -> None:
    print(f"Role: {role}")
    print(f"  Allowed access levels: {response_data['allowed_access_levels']}")

    results = response_data["results"]
    if not results:
        print("  (no results)\n")
        return

    for rank, result in enumerate(results, start=1):
        text_preview = " ".join(result["chunk_text"].split())[:150]
        print(
            f'  {rank}. score={result["score"]:.4f} document="{result["document_title"]}" '
            f'chunk_index={result["chunk_index"]}'
        )
        print(f"     {text_preview}...")
    print()


def main() -> None:
    args = parse_arguments()

    print(f'Question: "{args.question}"\n')

    print("=== Part 1: raw retrieval-layer candidates, before combine or rerank ===\n")
    connection = get_connection()
    try:
        print_raw_candidate_documents("student", connection, args.question, args.top_k)
        print_raw_candidate_documents("faculty", connection, args.question, args.top_k)
    finally:
        connection.close()

    print("=== Part 2: full /query endpoint results, after hybrid retrieval and reranking ===\n")
    client = TestClient(app)

    student_response = client.post(
        "/query",
        json={"question": args.question, "top_k": args.top_k},
        headers={"X-User-Role": "student"},
    )
    student_response.raise_for_status()
    print_query_response("student", student_response.json())

    faculty_response = client.post(
        "/query",
        json={"question": args.question, "top_k": args.top_k},
        headers={"X-User-Role": "faculty"},
    )
    faculty_response.raise_for_status()
    print_query_response("faculty", faculty_response.json())


if __name__ == "__main__":
    main()
