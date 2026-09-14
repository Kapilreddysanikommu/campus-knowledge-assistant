"""
Shared data structure returned by the answer generator.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class GeneratedAnswer:
    """The LLM's answer, along with whether it found enough grounding to answer.

    source_document_titles lists every document whose chunks were given to
    the model as context, regardless of whether the model judged them
    sufficient to answer the question.
    """

    answer: str
    has_sufficient_information: bool
    source_document_titles: List[str]
