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
    storage/      Stores documents and chunks in PostgreSQL
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
  PDF, chunk it, then store the document and chunk rows in PostgreSQL.

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
`campus_knowledge_assistant`, user `postgres`, password `postgres`). Pick
one of the following.

**Option A: Docker (used for this project's local setup)**

```
docker run --name campus-postgres ^
  -e POSTGRES_PASSWORD=postgres ^
  -e POSTGRES_DB=campus_knowledge_assistant ^
  -p 5432:5432 ^
  -v campus-postgres-data:/var/lib/postgresql/data ^
  -d postgres:16
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

The database tables themselves do not need to be created manually.
`scripts/run_pipeline.py` calls `create_tables()` on every run, which is
safe to call repeatedly since it only creates tables that do not already
exist.

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
`public`, `student`, `faculty`, or `admin`.

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
```
