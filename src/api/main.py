"""
FastAPI application exposing a role-filtered retrieval endpoint.

Authentication here is a single request header, X-User-Role, rather than
a full login system with accounts, passwords, and sessions. That is a
reasonable simplification for a demo project: the goal is to prove the
RBAC pattern itself (a role resolves to a set of allowed document access
levels, and that set is enforced inside the SQL query before retrieval
runs), and that pattern does not depend on how the caller's identity was
established. A real deployment would replace only the get_user_role
dependency below (for example, to decode a JWT or session cookie into a
role) with everything downstream of it unchanged.
"""

from fastapi import Depends, FastAPI, Header, HTTPException

from src.api.access_control import VALID_ROLES, get_allowed_access_levels
from src.api.schemas import QueryRequest, QueryResponse, QueryResultItem
from src.embedding.embedder import ChunkEmbedder
from src.generation.generator import AnswerGenerator
from src.retrieval.reranker import ResultReranker
from src.retrieval.search import combine_search_results, full_text_search, semantic_search
from src.retrieval.staleness import detect_and_apply_staleness
from src.storage.database import get_connection

app = FastAPI(title="Campus Knowledge Assistant API")

# Loaded once at startup and reused across requests. These wrap
# transformer models (and, for the generator, the Claude API client),
# which are far too slow or wasteful to set up again on every request.
embedder = ChunkEmbedder()
reranker = ResultReranker()
generator = AnswerGenerator()


def get_user_role(x_user_role: str = Header(...)) -> str:
    """Resolve the caller's role from the X-User-Role header.

    This stands in for a full authentication system. See the module
    docstring for why that is a reasonable simplification here.
    """
    role = x_user_role.strip().lower()
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X-User-Role header: {x_user_role!r}. Must be one of {VALID_ROLES}.",
        )
    return role


def get_db_connection():
    connection = get_connection()
    try:
        yield connection
    finally:
        connection.close()


@app.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    role: str = Depends(get_user_role),
    connection=Depends(get_db_connection),
) -> QueryResponse:
    """Search only the documents the caller's role is allowed to see.

    allowed_access_levels is resolved from the role before any retrieval
    happens, and is passed into semantic_search and full_text_search,
    which include it in their SQL WHERE clause. Chunks belonging to a
    document outside the allowed levels are therefore never fetched from
    the database at all, not fetched and then discarded afterward.

    After reranking, detect_and_apply_staleness checks whether the top
    results span documents from the same department but different
    academic years, moves a close-scoring older result below the newest
    one on the same topic, and returns a plain-language note for every
    such case found.

    Finally, generator.generate_answer sends only the resulting chunk
    text (never full documents, and never content outside the caller's
    allowed access levels) to Claude, which must answer using only that
    text and cite its sources, or say it does not have enough information
    rather than guess.
    """
    allowed_access_levels = get_allowed_access_levels(role)

    semantic_results = semantic_search(
        connection, embedder, request.question, allowed_access_levels, top_k=request.top_k
    )
    full_text_results = full_text_search(
        connection, request.question, allowed_access_levels, top_k=request.top_k
    )
    hybrid_results = combine_search_results(semantic_results, full_text_results, top_k=request.top_k)
    reranked_results = reranker.rerank(request.question, hybrid_results)
    final_results, staleness_notes = detect_and_apply_staleness(reranked_results)
    generated_answer = generator.generate_answer(request.question, final_results)

    return QueryResponse(
        question=request.question,
        role=role,
        allowed_access_levels=allowed_access_levels,
        results=[
            QueryResultItem(
                document_title=result.document_title,
                chunk_index=result.chunk_index,
                chunk_text=result.chunk_text,
                score=result.score,
                matched_by=result.matched_by,
            )
            for result in final_results
        ],
        staleness_notes=[note.message for note in staleness_notes],
        answer=generated_answer.answer,
        has_sufficient_information=generated_answer.has_sufficient_information,
    )
