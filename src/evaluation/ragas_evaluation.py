"""
Scores a pipeline run with Ragas: context precision and context recall
(did retrieval find the right chunks) and answer faithfulness (does the
generated answer actually match what the retrieved chunks say, rather
than being hallucinated).

Ragas 0.4's classic RAG metrics (Faithfulness, LLMContextPrecisionWithReference,
LLMContextRecall) still rely internally on the older langchain-based LLM
wrapper. Importing ragas at all currently fails, because one of its
modules imports a langchain_community submodule (a Vertex AI chat model
shim) that no longer exists in current langchain-community, even though
this project never uses Vertex AI. That import exists only so ragas can
do an isinstance() check internally, so a harmless stand-in class is
registered in sys.modules before ragas is imported, which satisfies the
import without needing the unrelated Vertex AI integration installed.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub_module = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAIStub:
        pass

    _vertexai_stub_module.ChatVertexAI = _ChatVertexAIStub
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub_module

from dataclasses import dataclass
from typing import Optional

from langchain_anthropic import ChatAnthropic
from ragas import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness

from src.evaluation.pipeline_runner import PipelineRunResult

DEFAULT_JUDGE_MODEL_NAME = "claude-sonnet-5"

# Generous headroom: Sonnet 5 uses adaptive thinking by default when the
# thinking parameter is not explicitly set, which consumes part of this
# budget before the actual verdict output, so a low limit can cause the
# response to be cut off mid-generation.
DEFAULT_JUDGE_MAX_TOKENS = 8192


@dataclass
class RagasScores:
    faithfulness: float
    context_precision: float
    context_recall: float


def build_judge_llm(model_name: str = DEFAULT_JUDGE_MODEL_NAME) -> LangchainLLMWrapper:
    """Wrap Claude as the judge model Ragas uses to score each metric.

    bypass_temperature is required because Ragas' wrapper otherwise sets a
    fixed sampling temperature on every call, and current Claude models
    reject the temperature parameter entirely.
    """
    chat_model = ChatAnthropic(model=model_name, max_tokens=DEFAULT_JUDGE_MAX_TOKENS)
    return LangchainLLMWrapper(chat_model, bypass_temperature=True)


def score_pipeline_run(
    run_result: PipelineRunResult, judge_llm: LangchainLLMWrapper
) -> Optional[RagasScores]:
    """Score one pipeline run, or return None if scoring does not apply.

    Ragas scoring is skipped for test cases where the system is expected
    to refuse: there is no meaningful "correct context" to measure
    precision or recall against when no document is supposed to answer
    the question. Whether a refusal case behaved correctly is instead
    checked by the deterministic checks in checks.py.
    """
    if run_result.test_case.should_refuse:
        return None

    sample = SingleTurnSample(
        user_input=run_result.test_case.question,
        response=run_result.answer,
        retrieved_contexts=run_result.retrieved_chunk_texts or [""],
        reference=run_result.test_case.reference_answer,
    )

    faithfulness_metric = Faithfulness(llm=judge_llm)
    context_precision_metric = LLMContextPrecisionWithReference(llm=judge_llm)
    context_recall_metric = LLMContextRecall(llm=judge_llm)

    return RagasScores(
        faithfulness=faithfulness_metric.single_turn_score(sample),
        context_precision=context_precision_metric.single_turn_score(sample),
        context_recall=context_recall_metric.single_turn_score(sample),
    )
