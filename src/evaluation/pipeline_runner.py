"""
Runs the full pipeline (retrieval, reranking, staleness detection,
generation) for one evaluation test case, and collects everything the
evaluation metrics need: the retrieved chunk texts, which documents they
came from, the generated answer, and whether the system judged it had
enough information to answer.
"""

from dataclasses import dataclass
from typing import List

from src.api.access_control import get_allowed_access_levels
from src.embedding.embedder import ChunkEmbedder
from src.evaluation.test_cases import EvaluationTestCase
from src.generation.generator import AnswerGenerator
from src.retrieval.reranker import ResultReranker
from src.retrieval.search import DEFAULT_TOP_K, combine_search_results, full_text_search, semantic_search
from src.retrieval.staleness import detect_and_apply_staleness
from src.storage.database import get_connection


@dataclass
class PipelineRunResult:
    """Everything produced by running one test case through the pipeline."""

    test_case: EvaluationTestCase
    retrieved_document_titles: List[str]
    retrieved_chunk_texts: List[str]
    answer: str
    has_sufficient_information: bool


def run_pipeline_for_case(
    test_case: EvaluationTestCase,
    embedder: ChunkEmbedder,
    reranker: ResultReranker,
    generator: AnswerGenerator,
    top_k: int = DEFAULT_TOP_K,
) -> PipelineRunResult:
    """Run one test case through the same pipeline the /query endpoint uses."""
    allowed_access_levels = get_allowed_access_levels(test_case.role)

    connection = get_connection()
    try:
        semantic_results = semantic_search(
            connection, embedder, test_case.question, allowed_access_levels, top_k=top_k
        )
        full_text_results = full_text_search(
            connection, test_case.question, allowed_access_levels, top_k=top_k
        )
    finally:
        connection.close()

    hybrid_results = combine_search_results(semantic_results, full_text_results, top_k=top_k)
    reranked_results = reranker.rerank(test_case.question, hybrid_results)
    final_results, _staleness_notes = detect_and_apply_staleness(reranked_results)
    generated_answer = generator.generate_answer(test_case.question, final_results)

    return PipelineRunResult(
        test_case=test_case,
        retrieved_document_titles=[result.document_title for result in final_results],
        retrieved_chunk_texts=[result.chunk_text for result in final_results],
        answer=generated_answer.answer,
        has_sufficient_information=generated_answer.has_sufficient_information,
    )
