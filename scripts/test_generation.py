"""
Manual test script for Step 8 answer generation.

Sends three questions through the real /query endpoint and prints the
generated answer for each:

Case 1: a question with a clear answer in a document the caller can see.
Case 2: a question whose real answer exists, but only in a document
        outside the caller's access level (the Step 6 RBAC case).
Case 3: a question with no answer in any document.

Case 1 should answer correctly with citations. Cases 2 and 3 should both
say the assistant does not have enough information, rather than guess.

Example:
    python scripts/test_generation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app

TEST_CASES = [
    {
        "label": "Case 1: answerable question, permitted document",
        "question": "what GPA is required for advancement to candidacy in the MSADI program",
        "role": "student",
    },
    {
        "label": "Case 2: real answer exists, but outside the caller's access level",
        "question": "what is the process for a grade appeal",
        "role": "student",
    },
    {
        "label": "Case 3: no answer exists in any document",
        "question": "what is the deadline to add a class",
        "role": "student",
    },
]


def run_case(client: TestClient, case: dict) -> None:
    print(case["label"])
    print(f'  Question: "{case["question"]}"')
    print(f"  Role: {case['role']}")

    response = client.post(
        "/query",
        json={"question": case["question"]},
        headers={"X-User-Role": case["role"]},
    )
    response.raise_for_status()
    response_data = response.json()

    print(f"  Has sufficient information: {response_data['has_sufficient_information']}")
    print(f"  Source documents considered: {[r['document_title'] for r in response_data['results']]}")
    print(f"  Answer:\n    {response_data['answer']}")
    print()


def main() -> None:
    client = TestClient(app)
    for case in TEST_CASES:
        run_case(client, case)


if __name__ == "__main__":
    main()
