"""
Deterministic, exact checks computed from a pipeline run, as a fast and
reliable complement to the LLM-judged Ragas metrics. These do not require
an LLM call and are especially important for the RBAC cases, where the
correctness requirement (a specific document must never appear) needs an
exact guarantee rather than a fuzzy judgment.
"""

from dataclasses import dataclass

from src.evaluation.pipeline_runner import PipelineRunResult


@dataclass
class DeterministicCheckResult:
    refusal_correct: bool
    expected_document_retrieved: bool
    forbidden_document_retrieved: bool


def run_deterministic_checks(run_result: PipelineRunResult) -> DeterministicCheckResult:
    test_case = run_result.test_case
    retrieved_titles = set(run_result.retrieved_document_titles)

    refusal_correct = run_result.has_sufficient_information != test_case.should_refuse

    expected_document_retrieved = (
        not test_case.expected_documents
        or bool(retrieved_titles.intersection(test_case.expected_documents))
    )

    forbidden_document_retrieved = (
        test_case.real_answer_document is not None
        and test_case.real_answer_document in retrieved_titles
    )

    return DeterministicCheckResult(
        refusal_correct=refusal_correct,
        expected_document_retrieved=expected_document_retrieved,
        forbidden_document_retrieved=forbidden_document_retrieved,
    )
