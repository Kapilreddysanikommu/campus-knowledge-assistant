"""
Generates the final answer to a user's question using Claude, grounded
only in the retrieved chunk text passed to it.
"""

import os
from typing import List

import anthropic
from dotenv import load_dotenv

from src.generation.models import GeneratedAnswer
from src.generation.prompt_builder import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_user_message
from src.retrieval.models import SearchResult

load_dotenv()

DEFAULT_MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_MAX_TOKENS = 1024


class AnswerGenerator:
    """Wraps the Claude API call used to produce a grounded, cited answer."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate_answer(self, question: str, results: List[SearchResult]) -> GeneratedAnswer:
        """Ask Claude to answer the question using only the given retrieved chunks.

        If there are no chunks to ground an answer in, the API is not
        called at all: there is nothing for the model to answer from.
        """
        if not results:
            return GeneratedAnswer(
                answer=REFUSAL_MESSAGE,
                has_sufficient_information=False,
                source_document_titles=[],
            )

        user_message = build_user_message(question, results)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = next(block.text for block in response.content if block.type == "text")
        has_sufficient_information = REFUSAL_MESSAGE.lower() not in answer_text.lower()

        return GeneratedAnswer(
            answer=answer_text,
            has_sufficient_information=has_sufficient_information,
            source_document_titles=sorted({result.document_title for result in results}),
        )
