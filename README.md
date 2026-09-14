# Campus Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions
about campus documents (syllabi, handbooks, policies, course catalogs, and
similar PDFs) by retrieving relevant passages and using a language model to
generate grounded answers.

## Project structure

```
src/
    ingestion/    Extracts text and structure from source documents (PDFs, etc.)
    chunking/     Splits extracted text into overlapping chunks sized for embedding
    embedding/    Generates vector embeddings for chunk text using a local model
    storage/      Stores documents, chunks, and their embeddings in PostgreSQL
    retrieval/    Finds the most relevant chunks for a given user query
    generation/   Generates a final answer from retrieved chunks using an LLM
    evaluation/   Measures retrieval and answer quality
scripts/          Small, runnable scripts for manually testing each stage
data/             Local data, such as sample PDFs (not committed to git)
```

Each subfolder under `src/` corresponds to one stage of the RAG pipeline and
will be built out incrementally.

## Current status

**Step 1: Document ingestion and chunking**

- `src/ingestion/pdf_ingestor.py` uses [Docling](https://github.com/DS4SD/docling)
  to convert a PDF into Markdown text, preserving structure such as headers
  and tables.
- `src/chunking/text_chunker.py` splits that Markdown text into overlapping,
  token-bounded chunks sized for an embedding model's context window. Chunk
  size and overlap are configurable, not hardcoded.
- `scripts/inspect_chunks.py` is a manual test script: point it at a PDF and
  it prints the extracted text length and every resulting chunk, so the
  output can be inspected by eye.

**Step 2: Metadata schema and storage in PostgreSQL**

- `src/storage/schema.py` defines two tables: `documents` (one row per
  source file, with title, department, academic year, and access level)
  and `chunks` (one row per chunk, linked to its document with a foreign
  key, in order via `chunk_index`).
- `src/storage/database.py` opens a connection to PostgreSQL using
  environment variables, so the same code works against a local Docker
  container or a native install.
- `src/storage/repository.py` inserts a document's metadata and its chunks
  into those tables.
- `scripts/run_pipeline.py` runs the full pipeline end to end: ingest a
  PDF, chunk it, then store the document and chunk rows in PostgreSQL. It
  can process a single PDF given as an argument, or a batch of PDFs listed
  in a JSON manifest (`--manifest`), each with its own metadata.

**Step 3: Embeddings and vector storage**

- The [pgvector](https://github.com/pgvector/pgvector) PostgreSQL extension
  is enabled, and `chunks.embedding` stores each chunk's vector.
- `src/embedding/embedder.py` generates embeddings locally using the
  sentence-transformers model `all-MiniLM-L6-v2`. It runs on CPU, needs no
  API key, and produces no ongoing API cost.
- `src/embedding/similarity.py` computes cosine similarity between two
  embedding vectors.
- `scripts/run_pipeline.py` now generates an embedding for every chunk
  before storing it.
- `scripts/test_embedding_similarity.py` is a manual test script: it
  embeds a topically similar pair of texts and a topically different pair,
  then prints the cosine similarity for each so you can confirm the
  similar pair scores higher.

**Step 4: Hybrid retrieval**

- `chunks.search_vector` is a generated `tsvector` column (computed
  automatically from `chunk_text` by PostgreSQL) with a GIN index, used for
  full-text search.
- `src/retrieval/search.py` has three functions: `semantic_search` embeds
  the query and finds the closest chunks by pgvector cosine distance;
  `full_text_search` uses PostgreSQL's `websearch_to_tsquery` and `ts_rank`
  to find chunks matching the query's words; `combine_search_results`
  merges both ranked lists into one using Reciprocal Rank Fusion, with no
  duplicate chunks.
- `scripts/test_hybrid_retrieval.py` runs a question through all three and
  prints each list, so semantic and full-text results can be compared side
  by side against the final merged list.

**Step 5: Cross-encoder reranking**

- `src/retrieval/reranker.py` has `ResultReranker`, which uses the
  cross-encoder model `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score a
  list of results. Unlike the bi-encoder used for embeddings, a
  cross-encoder reads the query and a chunk together in one pass, which is
  more accurate but too slow to run over an entire chunk collection, so it
  is used here to re-score only the short candidate list that hybrid
  retrieval already produced. Its scores are raw, unbounded relevance
  logits (not a 0-1 similarity), meaningful only for sorting, not for
  comparing against semantic or full-text scores.
- `scripts/test_reranking.py` runs a question through Step 4's hybrid
  retrieval, then reranks the result, printing the order before and after
  along with a summary of which chunks moved up or down.

**Step 6: Role-based access control (RBAC) in FastAPI**

- `src/api/access_control.py` maps a role (`student`, `faculty`, `admin`)
  to the list of `documents.access_level` values that role may search.
  `student` sees `public` and `student`; `faculty` additionally sees
  `faculty`; `admin` sees everything.
- `src/api/main.py` is a minimal FastAPI app with one endpoint,
  `POST /query`. Authentication is a single `X-User-Role` request header
  rather than a full login system (see the module docstring for why that
  is a reasonable simplification for a demo). The role resolves to an
  allowed access level list *before* retrieval runs, and that list is
  passed into `semantic_search` and `full_text_search`, both of which now
  require it (there is no default that silently searches everything).
  Each function adds `AND documents.access_level = ANY(%s)` to its SQL
  `WHERE` clause, so a document outside the caller's allowed levels is
  never fetched from the database, not fetched and then hidden from the
  response.
- `scripts/test_rbac_query.py` proves this two ways: first by calling the
  retrieval functions directly for a student and a faculty role and
  listing which documents came back as raw SQL candidates (before
  combining or reranking), then by calling the real `/query` endpoint for
  both roles and printing the full results side by side.

**Step 7: Staleness detection**

- `src/retrieval/staleness.py` catches a specific problem: two documents
  covering the same topic but published in different academic years, one
  possibly superseding the other, both showing up in the same result
  list. There is no topic classifier in this project, so it reuses a
  signal retrieval already produced instead of building one: if two
  chunks from different-year documents both made it into the same
  reranked top-k list for the same question, the reranker already judged
  both relevant to it; requiring them to also share a department is a
  cheap guard against unrelated documents coincidentally both being old.
  When this is detected, a `StalenessNote` explaining it is always
  produced, and if an older document's chunk scored only about as well as
  (within `DEFAULT_RECENCY_TIE_BREAK_MARGIN`) the newest document's chunk
  on the same topic, the newer one is moved above it; a clearly stronger
  older result is left in place, since a wide score gap means the
  reranker found something more directly relevant to this specific
  question. Only the order changes, never the score value a caller sees.
- `src/api/main.py` applies `detect_and_apply_staleness` after reranking,
  and `POST /query` now also returns `staleness_notes`.
- `scripts/test_staleness_detection.py` shows the reranked order with and
  without staleness detection side by side, along with the notes, then
  confirms the same behavior through the real `/query` endpoint.

**Step 8: Grounded answer generation with citations**

- `src/generation/prompt_builder.py` builds the prompt sent to the LLM: a
  system prompt instructing it to answer using only the excerpts it is
  given, to cite the source document for every part of its answer in the
  format `(Source: <document title>)`, and to respond with an exact,
  fixed refusal sentence if the excerpts do not actually answer the
  question, rather than guess or use outside knowledge. The user message
  contains only the retrieved chunk text and its document title and
  academic year, never full source documents.
- `src/generation/generator.py` has `AnswerGenerator`, which sends that
  prompt to Claude (the official `anthropic` Python SDK, model
  configurable via the `ANTHROPIC_MODEL` environment variable, default
  `claude-opus-5`) and returns whether the refusal sentence was used, so
  callers get a plain boolean instead of having to re-parse the answer
  text.
- `src/api/main.py` calls `generate_answer` as the final pipeline step,
  after staleness detection, using only the resulting chunks; `POST
  /query` now also returns `answer` and `has_sufficient_information`.
  Since retrieval was already filtered by the caller's access level
  before this step, the LLM never sees chunk text from a document outside
  what the caller is permitted to read.
- `scripts/test_generation.py` runs three cases through the real
  endpoint: a question answerable from permitted documents (expects a
  cited answer), the Step 6 RBAC case where the real answer exists only
  in a document outside the caller's access level (expects a refusal),
  and a question with no answer in any document (expects a refusal).

**Step 9: Evaluation suite**

- `data/evaluation_test_set.json` has 31 question-answer pairs against
  the real ingested documents: questions with a clear answer, questions
  that should be refused because no document answers them, questions
  that should be refused because the real answer exists only outside the
  caller's access level (the Step 6 RBAC case, extended to several
  different facts and roles), and adversarial phrasings (paraphrases,
  false premises, compound questions, and a genuine staleness case where
  two documents give different figures for the same policy).
- `src/evaluation/pipeline_runner.py` runs the full pipeline (retrieval,
  reranking, staleness detection, generation) for one test case.
- `src/evaluation/checks.py` has fast, exact checks that need no LLM
  call: did the system refuse exactly when it should have, and did a
  document outside the caller's access level ever get retrieved.
- `src/evaluation/ragas_evaluation.py` scores each answerable case with
  the Ragas evaluation library: faithfulness (does the generated answer
  actually match the retrieved chunks, rather than being hallucinated),
  and context precision and recall (did retrieval find the right
  chunks). Ragas's classic RAG metrics still need an older langchain-
  based LLM wrapper internally; Claude is wired in as the judge model
  through that wrapper rather than through OpenAI.
- `scripts/run_evaluation.py` runs every test case through the real
  pipeline, scores it, and prints a summary: overall pass rate, average
  Ragas scores, a per-category breakdown, and every low-scoring or
  failing case by name so weak spots are easy to find.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust it if your database connection
details differ from the defaults:

```
copy .env.example .env
```

Then set `ANTHROPIC_API_KEY` in `.env` to your own Claude API key. This is
required for answer generation (Step 8); everything before that step
works without it.

## Setting up PostgreSQL

The project expects PostgreSQL reachable at the host, port, database name,
user, and password in `.env` (defaults: `localhost:5432`, database
`campus_knowledge_assistant`, user `postgres`, password `postgres`). It
also requires the [pgvector](https://github.com/pgvector/pgvector)
extension, since chunk embeddings are stored as a `VECTOR` column.
`psql`'s default `postgres:16` image does not include pgvector, so this
project uses `pgvector/pgvector:pg16` instead, which is the official
pgvector image built on top of `postgres:16` (same PostgreSQL version,
extension pre-installed). Pick one of the following.

**Option A: Docker (used for this project's local setup)**

```
docker run --name campus-postgres ^
  -e POSTGRES_PASSWORD=postgres ^
  -e POSTGRES_DB=campus_knowledge_assistant ^
  -p 5432:5432 ^
  -v campus-postgres-data:/var/lib/postgresql/data ^
  -d pgvector/pgvector:pg16
```

This requires Docker Desktop to be running first. Check the container is
healthy with:

```
docker exec campus-postgres pg_isready -U postgres
```

To stop and restart it later:

```
docker stop campus-postgres
docker start campus-postgres
```

If you already have a `campus-postgres` container running the plain
`postgres:16` image from an earlier step, switch images without losing
data by stopping and removing the old container, then running the command
above again. The named volume (`campus-postgres-data`) holds the actual
data files and is untouched by removing the container itself:

```
docker stop campus-postgres
docker rm campus-postgres
```

**Option B: Native install via winget (run in an elevated PowerShell)**

```
winget install --id PostgreSQL.PostgreSQL.17 -e --source winget --override "--unattendedmodeui minimal --mode unattended --superpassword postgres --serverport 5432"
```

This installs PostgreSQL as a Windows service (`postgresql-x64-17`) that
starts automatically on boot, using `postgres` as the superuser password to
match `.env.example`. Manage the service with:

```
Get-Service postgresql-x64-17
Start-Service postgresql-x64-17
Stop-Service postgresql-x64-17
```

Then create the project database:

```
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -c "CREATE DATABASE campus_knowledge_assistant;"
```

pgvector is not bundled with the native Windows installer. Building it
from source on Windows is involved enough that Docker (Option A) is the
easier path for local development; if you need a native install with
pgvector, see the
[pgvector Windows build instructions](https://github.com/pgvector/pgvector#windows).

The database tables, the pgvector extension, and the embedding column do
not need to be set up manually. `scripts/run_pipeline.py` calls
`create_tables()` on every run, which enables the extension and creates
anything missing, and is safe to call repeatedly.

## Testing ingestion and chunking

Place a sample PDF in `data/sample_pdfs/`, then run:

```
python scripts/inspect_chunks.py data/sample_pdfs/your_file.pdf
```

Optional flags to override the default chunk size and overlap:

```
python scripts/inspect_chunks.py data/sample_pdfs/your_file.pdf --chunk-size 300 --chunk-overlap 50
```

## Running the full pipeline into PostgreSQL

With PostgreSQL running and `.env` configured, run:

```
python scripts/run_pipeline.py data/sample_pdfs/your_file.pdf --title "CS 101 Syllabus" --department "Computer Science" --academic-year "2025-2026" --access-level student
```

`--department`, `--academic-year`, `--chunk-size`, and `--chunk-overlap`
are optional. `--access-level` defaults to `public` and must be one of
`public`, `student`, `faculty`, or `admin`. Each chunk's embedding is
generated locally before insertion; the first run downloads the
sentence-transformers model (roughly 80MB) and caches it for later runs.

### Processing multiple PDFs in one run

To ingest several PDFs at once, each with its own metadata, use a JSON
manifest instead of the single-file flags:

```
python scripts/run_pipeline.py --manifest data/document_manifest.json
```

The manifest is a list of objects, one per PDF:

```json
[
  {
    "pdf_path": "data/sample_pdfs/example.pdf",
    "title": "Example Document",
    "department": "Registrar",
    "academic_year": "2025-2026",
    "access_level": "student"
  }
]
```

`department`, `academic_year`, and `access_level` are optional in each
entry (`access_level` defaults to `public` if omitted). `--chunk-size` and
`--chunk-overlap` still apply to every document in the manifest. If one
PDF fails to process, the script reports the error and continues with the
rest of the manifest rather than stopping the whole run; the final line
reports how many documents succeeded and how many failed.

## Testing embeddings

To confirm the embedding model produces higher similarity scores for
related text than for unrelated text:

```
python scripts/test_embedding_similarity.py
```

Or supply your own text to compare:

```
python scripts/test_embedding_similarity.py --similar-a "..." --similar-b "..." --different-a "..." --different-b "..."
```

### Verifying the data landed correctly

Connect with `docker exec -it campus-postgres psql -U postgres -d campus_knowledge_assistant`
(or `psql` directly for a native install), then run:

```sql
-- Row counts in each table
SELECT
    (SELECT count(*) FROM documents) AS document_count,
    (SELECT count(*) FROM chunks) AS chunk_count;

-- One sample chunk together with its parent document's metadata
SELECT
    documents.title,
    documents.department,
    documents.access_level,
    chunks.chunk_index,
    chunks.token_count,
    chunks.chunk_text
FROM chunks
JOIN documents ON documents.id = chunks.document_id
ORDER BY chunks.id
LIMIT 1;

-- Title, access level, and chunk count for every document, useful after
-- a batch run to confirm everything from a manifest landed correctly
SELECT
    documents.title,
    documents.access_level,
    count(chunks.id) AS chunk_count
FROM documents
LEFT JOIN chunks ON chunks.document_id = documents.id
GROUP BY documents.id, documents.title, documents.access_level
ORDER BY documents.id;

-- Chunk count and total tokens per document, useful for spotting a document
-- that ingested with far fewer chunks than expected
SELECT
    documents.title,
    count(chunks.id) AS chunk_count,
    sum(chunks.token_count) AS total_tokens
FROM documents
LEFT JOIN chunks ON chunks.document_id = documents.id
GROUP BY documents.id, documents.title
ORDER BY documents.id;

-- Confirm embeddings were stored with the expected dimension
SELECT id, chunk_index, vector_dims(embedding) AS embedding_dimensions
FROM chunks
ORDER BY id;
```

## Testing hybrid retrieval

To compare semantic search, full-text search, and the merged hybrid list
for one question:

```
python scripts/test_hybrid_retrieval.py "what is the grading breakdown"
```

With no argument, it uses that same question as a default. Add `--top-k`
to change how many results are shown per search type (default 5):

```
python scripts/test_hybrid_retrieval.py "when is the midterm exam" --top-k 3
```

The output shows why each search type matters: semantic search ranks
every chunk by meaning, even ones that do not share any words with the
question, while full-text search only returns chunks that literally
contain the query's words, but ranks exact matches precisely. The merged
list promotes chunks found by both to the top.

## Testing reranking

To compare hybrid retrieval's order against the cross-encoder reranked
order for one question:

```
python scripts/test_reranking.py "what does a WU grade mean"
```

With no argument, it uses that same question as a default. Add `--top-k`
to change how many results are retrieved and reranked (default 5):

```
python scripts/test_reranking.py "what courses are offered in Fall 2026" --top-k 8
```

The output prints the hybrid order, the reranked order, and a summary of
which chunks moved up or down, so you can judge whether reranking pushed
the chunk that actually answers the question higher.

## Running the API

Start the FastAPI application with:

```
python -m uvicorn src.api.main:app --reload
```

Then send a request with a role header, for example using curl:

```
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -H "X-User-Role: student" ^
  -d "{\"question\": \"what is the process for a grade appeal\"}"
```

`X-User-Role` must be `student`, `faculty`, or `admin`; any other value
returns a 400 error. Interactive API docs are available at
`http://127.0.0.1:8000/docs` once the server is running.

## Testing role-based access control

To see proof that a student's search never retrieves chunks from a
faculty-only document, compared against a faculty search for the same
question:

```
python scripts/test_rbac_query.py "what is the process for a grade appeal"
```

The first part of the output calls the retrieval functions directly and
lists which documents came back as raw candidates for each role, before
any combining or reranking happens. The second part shows the full
`/query` endpoint results for both roles side by side.

## Testing staleness detection

To compare retrieval results with and without staleness detection for a
question that multiple documents from different years could plausibly
answer:

```
python scripts/test_staleness_detection.py "what is SJSU's basic grading system"
```

Add `--role` (`student`, `faculty`, or `admin`, default `student`) or
`--top-k` to adjust the search:

```
python scripts/test_staleness_detection.py "how many credit/no credit units can I use" --top-k 8
```

The output shows the reranked order before staleness detection, the
reordered result after it runs, and the staleness notes explaining which
documents and years were found to overlap. A close-scoring older result
moves below the newest result on the same topic; a clearly stronger older
result is left where the reranker placed it.

## Testing answer generation

Requires `ANTHROPIC_API_KEY` to be set in `.env`. To run the three cases
described above (answerable, RBAC-blocked, and no answer anywhere) and
print each generated answer:

```
python scripts/test_generation.py
```

Case 1 should return a cited answer with `has_sufficient_information:
true`. Cases 2 and 3 should both return the fixed refusal sentence with
`has_sufficient_information: false`, rather than a guess.

## Running the evaluation suite

Requires `ANTHROPIC_API_KEY` to be set in `.env`. Runs all 31 test cases
in `data/evaluation_test_set.json` through the real pipeline and prints a
scored summary:

```
python scripts/run_evaluation.py
```

This makes one generation call per test case plus, for every case where
the system is expected to answer, three additional Claude calls to score
it with Ragas, so a full run is on the order of 80-90 API calls and takes
several minutes. The summary reports the overall pass rate, average
faithfulness, context precision, and context recall, a per-category
breakdown, and every low-scoring or failing case by id and question, so
weak spots can be looked up directly in the test set file.
