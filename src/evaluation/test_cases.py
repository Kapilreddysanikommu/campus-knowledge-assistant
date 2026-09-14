"""
Loads the evaluation test set from data/evaluation_test_set.json.

Each test case describes one question to run through the full pipeline,
the role to search as, and what a correct outcome looks like: either the
document(s) the answer should be grounded in, or that the system should
refuse to answer.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EvaluationTestCase:
    """One question-answer pair from the evaluation test set."""

    id: str
    category: str
    question: str
    role: str
    should_refuse: bool
    expected_documents: List[str]
    reference_answer: str
    real_answer_document: Optional[str] = None
    notes: Optional[str] = None


def load_test_cases(test_set_path: Path) -> List[EvaluationTestCase]:
    with open(test_set_path, "r", encoding="utf-8") as test_set_file:
        raw_cases = json.load(test_set_file)

    return [
        EvaluationTestCase(
            id=case["id"],
            category=case["category"],
            question=case["question"],
            role=case["role"],
            should_refuse=case["should_refuse"],
            expected_documents=case["expected_documents"],
            reference_answer=case["reference_answer"],
            real_answer_document=case.get("real_answer_document"),
            notes=case.get("notes"),
        )
        for case in raw_cases
    ]
