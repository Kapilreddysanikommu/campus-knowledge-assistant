"""
Manual test script for Step 10 observability.

Sends a handful of real queries through the /query endpoint with tracing
enabled, then reads back what Langfuse actually captured for each
request and prints a summary: per-stage timing, total latency, and
whether the request was answered or refused. This is a printed stand-in
for what a screenshot of the Langfuse dashboard would show.

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to be set in .env.

Example:
    python scripts/test_tracing.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app
from src.observability.tracing import get_tracing_client

TEST_QUERIES = [
    {"question": "What GPA is required for advancement to candidacy in the MSADI program?", "role": "student"},
    {"question": "What is the process for a grade appeal?", "role": "student"},
    {"question": "What is the deadline to add a class each semester?", "role": "student"},
    {"question": "What does the WU grade symbol mean?", "role": "faculty"},
]

PIPELINE_STAGE_NAMES = ["retrieval", "reranking", "staleness_detection", "generation"]

# Langfuse ingests traces asynchronously on its backend after the SDK
# sends them, so this is how long to wait after the requests finish
# before reading the traces back through the API. 8 seconds was not
# reliably enough in testing, so this is generous on purpose.
INGESTION_WAIT_SECONDS = 25


def run_queries(client: TestClient) -> List[dict]:
    responses = []
    for query in TEST_QUERIES:
        response = client.post(
            "/query",
            json={"question": query["question"]},
            headers={"X-User-Role": query["role"]},
        )
        response.raise_for_status()
        responses.append(response.json())
    return responses


def print_captured_trace(tracing_client, response_data: dict) -> None:
    trace_id = response_data["trace_id"]

    print(f'Question: "{response_data["question"]}" (role={response_data["role"]})')
    print(f"  Trace id: {trace_id}")

    if trace_id is None:
        print("  No trace was captured for this request.\n")
        return

    observations = tracing_client.api.observations.get_many(trace_id=trace_id).data
    observations_by_name: Dict[str, object] = {observation.name: observation for observation in observations}

    print(f"  Answered: {response_data['has_sufficient_information']}")

    for stage_name in PIPELINE_STAGE_NAMES:
        observation = observations_by_name.get(stage_name)
        if observation is None:
            print(f"  {stage_name}: not captured")
            continue
        latency_seconds = observation.latency if observation.latency is not None else 0.0
        print(f"  {stage_name}: {latency_seconds:.3f}s")

    root_observation = observations_by_name.get("query_documents")
    if root_observation is not None and root_observation.latency is not None:
        print(f"  Total latency: {root_observation.latency:.3f}s")
    print()


def main() -> None:
    client = TestClient(app)
    tracing_client = get_tracing_client()

    print(f"Sending {len(TEST_QUERIES)} queries through the traced /query endpoint...\n")
    responses = run_queries(client)

    print(f"Waiting {INGESTION_WAIT_SECONDS}s for Langfuse to ingest the traces...\n")
    time.sleep(INGESTION_WAIT_SECONDS)

    print("=== Captured traces ===\n")
    for response_data in responses:
        print_captured_trace(tracing_client, response_data)


if __name__ == "__main__":
    main()
