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
