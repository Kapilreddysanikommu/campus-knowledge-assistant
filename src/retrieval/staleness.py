"""
Detects staleness among the top retrieval results: cases where results
from documents of different academic years, but the same department,
both appear in the same result list for a question.

This project does not have a topic classifier, so it reuses a signal the
pipeline already computed instead of building one: if two chunks from
different-year documents both made it into the same reranked top-k list
for the same question, the reranker itself already judged both relevant
to that question. Sharing a department on top of that is a cheap guard
against coincidences (an old, unrelated document happening to share a
result list with another old document purely by chance).

When staleness is detected, two things happen. First, a StalenessNote is
always produced describing which documents and years are involved, so the
conflict is visible rather than silently resolved. Second, if an older
document's chunk scored only about as well as the newest document's chunk
on the same topic (a close call), the newer chunk is moved above it. If
the older chunk scored substantially better, its position is left alone:
a strong score gap means the reranker found it meaningfully more relevant
to this specific question, and recency should not override that.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.retrieval.models import SearchResult

# How close two reranked scores need to be, in the cross-encoder's own
# score units, before an older document's chunk is moved below a newer
# one on the same topic. This project's cross-encoder produces raw,
# unbounded scores that have ranged from roughly -10 (irrelevant) to +5
# (highly relevant) in testing; results that plausibly answer the same
# question tend to land within about a point of each other, while a
# clearly stronger match tends to lead by several points. This is a rough
# calibration for that score range, not a universal constant.
DEFAULT_RECENCY_TIE_BREAK_MARGIN = 1.0

# How far below the newest result's score an older, demoted result is
# placed. Small enough to only affect relative order among close results,
# not push a demoted result below unrelated, lower-scoring ones.
DEMOTION_SORT_OFFSET = 0.01


@dataclass
class StalenessNote:
    """A human-readable flag that two documents of different years overlap."""

    department: str
    newer_title: str
    newer_academic_year: str
    older_title: str
    older_academic_year: str

    @property
    def message(self) -> str:
        return (
            f'"{self.newer_title}" ({self.newer_academic_year}) and "{self.older_title}" '
            f'({self.older_academic_year}) both appear in these results and cover overlapping '
            f"{self.department} topics. The {self.newer_academic_year} document is more recent "
            f"and may supersede the {self.older_academic_year} one where they conflict."
        )


def parse_academic_year(academic_year: Optional[str]) -> Optional[int]:
    """Extract the latest four-digit year mentioned in an academic_year string.

    Handles a single year ("2018") and a range ("2025-2026"), using the
    later year in a range since it best reflects how current the document
    is. Returns None if no year can be found.
    """
    if not academic_year:
        return None

    years_found = [int(match) for match in re.findall(r"\d{4}", academic_year)]
    if not years_found:
        return None

    return max(years_found)


def detect_and_apply_staleness(
    results: List[SearchResult],
    recency_tie_break_margin: float = DEFAULT_RECENCY_TIE_BREAK_MARGIN,
) -> Tuple[List[SearchResult], List[StalenessNote]]:
    """Reorder close ties toward the newer document and report any staleness found.

    Returns the (possibly reordered) result list alongside a list of
    StalenessNote objects. Each SearchResult's own score field is left
    unchanged; only the order of the list may change, so the score a
    caller sees always reflects the reranker's real judgment.
    """
    parsed_years = {result.chunk_id: parse_academic_year(result.academic_year) for result in results}

    department_groups = {}
    for result in results:
        year = parsed_years[result.chunk_id]
        if result.department is None or year is None:
            continue
        department_groups.setdefault(result.department, []).append(result)

    notes: List[StalenessNote] = []
    demoted_sort_key = {}

    for department, group_results in department_groups.items():
        years_in_group = {parsed_years[result.chunk_id] for result in group_results}
        if len(years_in_group) <= 1:
            continue

        newest_year = max(years_in_group)
        newest_result = max(
            (result for result in group_results if parsed_years[result.chunk_id] == newest_year),
            key=lambda result: result.score,
        )

        already_reported_years = set()
        for result in group_results:
            year = parsed_years[result.chunk_id]
            if year == newest_year:
                continue

            if year not in already_reported_years:
                already_reported_years.add(year)
                notes.append(
                    StalenessNote(
                        department=department,
                        newer_title=newest_result.document_title,
                        newer_academic_year=newest_result.academic_year,
                        older_title=result.document_title,
                        older_academic_year=result.academic_year,
                    )
                )

            score_gap = result.score - newest_result.score
            if 0 <= score_gap <= recency_tie_break_margin:
                demoted_sort_key[result.chunk_id] = newest_result.score - DEMOTION_SORT_OFFSET

    reordered_results = sorted(
        results,
        key=lambda result: demoted_sort_key.get(result.chunk_id, result.score),
        reverse=True,
    )

    return reordered_results, notes
