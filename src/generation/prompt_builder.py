"""
Builds the prompt sent to the LLM for generating a grounded answer.

The model is given only the retrieved chunk text, never the full source
documents, and is instructed to answer using only that text, to cite
which document each part came from, and to say so explicitly when the
excerpts do not actually contain an answer, rather than guess.
"""

from typing import List

from src.retrieval.models import SearchResult

REFUSAL_MESSAGE = "I don't have enough information to answer this question based on the available documents."

SYSTEM_PROMPT = (
    "You are a campus knowledge assistant. Answer the user's question using only "
    "the document excerpts provided in the user message. Do not use any outside "
    "knowledge, and do not guess.\n\n"
    "For every part of your answer, cite the source document by its exact title "
    "in the format (Source: <document title>).\n\n"
    "If the excerpts do not actually contain an answer to the question, respond "
    f'with exactly this sentence and nothing else: "{REFUSAL_MESSAGE}"'
)


def build_user_message(question: str, results: List[SearchResult]) -> str:
    """Combine the retrieved excerpts and the question into one user message."""
    excerpt_blocks = [
        f"Excerpt {index} - Document: {result.document_title} "
        f"(Academic year: {result.academic_year})\n{result.chunk_text}"
        for index, result in enumerate(results, start=1)
    ]
    excerpts_text = "\n\n".join(excerpt_blocks)

    return f"Document excerpts:\n\n{excerpts_text}\n\nQuestion: {question}"
