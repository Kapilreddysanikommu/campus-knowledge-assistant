"""
Runs the full pipeline (retrieval, reranking, staleness detection,
generation) against the evaluation test set, scores the results, and
prints a summary: overall scores and which test cases scored low.

Two kinds of checks are used together:

Deterministic checks (exact, no LLM call) confirm whether the system
refused when it should have, answered when it should have, retrieved the
expected document for an answerable question, and never retrieved a
document the caller's role is not permitted to see.

Ragas checks (LLM-judged, run only on answerable questions) score
faithfulness (does the generated answer actually match the retrieved
chunks, rather than being hallucinated), context precision (are the
retrieved chunks relevant), and context recall (did retrieval find
everything needed to answer).

Example:
    python scripts/run_evaluation.py
"""

import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedder import ChunkEmbedder
from src.evaluation.checks import DeterministicCheckResult, run_deterministic_checks
from src.evaluation.pipeline_runner import PipelineRunResult, run_pipeline_for_case
from src.evaluation.ragas_evaluation import RagasScores, build_judge_llm, score_pipeline_run
from src.evaluation.test_cases import EvaluationTestCase, load_test_cases
from src.generation.generator import AnswerGenerator
from src.retrieval.reranker import ResultReranker

TEST_SET_PATH = PROJECT_ROOT / "data" / "evaluation_test_set.json"

# A case is called out in the low-scoring summary if any Ragas score
# falls below this, or if either deterministic pass/fail check fails.
LOW_SCORE_THRESHOLD = 0.7


class CaseEvaluation:
    """Everything gathered about one test case's evaluation run."""

    def __init__(
        self,
        test_case: EvaluationTestCase,
        run_result: PipelineRunResult,
        checks: DeterministicCheckResult,
        ragas_scores: Optional[RagasScores],
        ragas_scoring_error: Optional[str] = None,
    ):
        self.test_case = test_case
        self.run_result = run_result
        self.checks = checks
        self.ragas_scores = ragas_scores
        self.ragas_scoring_error = ragas_scoring_error

    @property
    def passed(self) -> bool:
        if not self.checks.refusal_correct:
            return False
        if self.checks.forbidden_document_retrieved:
            return False
        if not self.checks.expected_document_retrieved:
            return False
        if self.ragas_scoring_error is not None:
            return False
        if self.ragas_scores is not None:
            lowest_score = min(
                self.ragas_scores.faithfulness,
                self.ragas_scores.context_precision,
                self.ragas_scores.context_recall,
            )
            if lowest_score < LOW_SCORE_THRESHOLD:
                return False
        return True


def evaluate_case(
    test_case: EvaluationTestCase,
    embedder: ChunkEmbedder,
    reranker: ResultReranker,
    generator: AnswerGenerator,
    judge_llm,
) -> CaseEvaluation:
    run_result = run_pipeline_for_case(test_case, embedder, reranker, generator)
    checks = run_deterministic_checks(run_result)

    ragas_scores = None
    ragas_scoring_error = None
    try:
        ragas_scores = score_pipeline_run(run_result, judge_llm)
    except Exception as error:
        # A single judge-scoring failure (for example, a transient LLM
        # error) should not lose progress on every other test case, but
        # it is still reported as a failure rather than silently skipped.
        ragas_scoring_error = str(error)

    return CaseEvaluation(test_case, run_result, checks, ragas_scores, ragas_scoring_error)


def print_case_progress(evaluation: CaseEvaluation) -> None:
    status = "PASS" if evaluation.passed else "FAIL"
    test_case = evaluation.test_case
    print(f"[{status}] {test_case.id} ({test_case.category}, role={test_case.role}): {test_case.question}")

    if evaluation.checks.forbidden_document_retrieved:
        print(f"    forbidden document was retrieved: {test_case.real_answer_document}")
    if not evaluation.checks.refusal_correct:
        expected = "refuse" if test_case.should_refuse else "answer"
        actual = "refused" if not evaluation.run_result.has_sufficient_information else "answered"
        print(f"    expected the system to {expected}, but it {actual}")
    if not evaluation.checks.expected_document_retrieved:
        print(f"    expected document(s) not retrieved: {test_case.expected_documents}")
    if evaluation.ragas_scoring_error is not None:
        print(f"    Ragas scoring failed: {evaluation.ragas_scoring_error}")
    if evaluation.ragas_scores is not None:
        scores = evaluation.ragas_scores
        print(
            f"    faithfulness={scores.faithfulness:.2f} "
            f"context_precision={scores.context_precision:.2f} "
            f"context_recall={scores.context_recall:.2f}"
        )


def average(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def print_summary(evaluations: List[CaseEvaluation]) -> None:
    total_count = len(evaluations)
    passed_count = sum(1 for evaluation in evaluations if evaluation.passed)

    refusal_correct_count = sum(1 for e in evaluations if e.checks.refusal_correct)
    forbidden_leak_count = sum(1 for e in evaluations if e.checks.forbidden_document_retrieved)
    expected_doc_count = sum(1 for e in evaluations if e.checks.expected_document_retrieved)

    scored_evaluations = [e for e in evaluations if e.ragas_scores is not None]
    average_faithfulness = average([e.ragas_scores.faithfulness for e in scored_evaluations])
    average_context_precision = average([e.ragas_scores.context_precision for e in scored_evaluations])
    average_context_recall = average([e.ragas_scores.context_recall for e in scored_evaluations])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total test cases: {total_count}")
    print(f"Passed overall: {passed_count}/{total_count}")
    print(f"Refusal behavior correct: {refusal_correct_count}/{total_count}")
    print(f"Expected document retrieved (when applicable): {expected_doc_count}/{total_count}")
    print(f"RBAC leaks (forbidden document retrieved): {forbidden_leak_count}/{total_count}")

    print(f"\nRagas scores (answerable cases only, n={len(scored_evaluations)}):")
    if average_faithfulness is not None:
        print(f"  Faithfulness:      {average_faithfulness:.2f}")
        print(f"  Context precision: {average_context_precision:.2f}")
        print(f"  Context recall:    {average_context_recall:.2f}")
    else:
        print("  (no answerable cases were scored)")

    print("\nBy category:")
    categories = sorted({evaluation.test_case.category for evaluation in evaluations})
    for category in categories:
        category_evaluations = [e for e in evaluations if e.test_case.category == category]
        category_passed = sum(1 for e in category_evaluations if e.passed)
        print(f"  {category}: {category_passed}/{len(category_evaluations)} passed")

    failing_evaluations = [evaluation for evaluation in evaluations if not evaluation.passed]
    print(f"\nLow-scoring or failing cases ({len(failing_evaluations)}):")
    if not failing_evaluations:
        print("  none")
    for evaluation in failing_evaluations:
        test_case = evaluation.test_case
        reasons = []
        if not evaluation.checks.refusal_correct:
            reasons.append("wrong refusal behavior")
        if evaluation.checks.forbidden_document_retrieved:
            reasons.append("RBAC leak")
        if not evaluation.checks.expected_document_retrieved:
            reasons.append("expected document missing")
        if evaluation.ragas_scoring_error is not None:
            reasons.append(f"Ragas scoring failed ({evaluation.ragas_scoring_error})")
        if evaluation.ragas_scores is not None:
            scores = evaluation.ragas_scores
            if scores.faithfulness < LOW_SCORE_THRESHOLD:
                reasons.append(f"low faithfulness ({scores.faithfulness:.2f})")
            if scores.context_precision < LOW_SCORE_THRESHOLD:
                reasons.append(f"low context precision ({scores.context_precision:.2f})")
            if scores.context_recall < LOW_SCORE_THRESHOLD:
                reasons.append(f"low context recall ({scores.context_recall:.2f})")
        print(f'  {test_case.id}: "{test_case.question}" -- {", ".join(reasons)}')


def main() -> None:
    test_cases = load_test_cases(TEST_SET_PATH)
    print(f"Loaded {len(test_cases)} test cases from {TEST_SET_PATH}\n")

    embedder = ChunkEmbedder()
    reranker = ResultReranker()
    generator = AnswerGenerator()
    judge_llm = build_judge_llm()

    evaluations = []
    for index, test_case in enumerate(test_cases, start=1):
        print(f"--- Running case {index}/{len(test_cases)} ---")
        evaluation = evaluate_case(test_case, embedder, reranker, generator, judge_llm)
        print_case_progress(evaluation)
        evaluations.append(evaluation)

    print_summary(evaluations)


if __name__ == "__main__":
    main()
