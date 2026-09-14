"""
Defines and creates the database schema for documents and their chunks.

A document row represents one source file (a syllabus, handbook, policy,
and so on) along with metadata used to control who can see it. Each chunk
row is one piece of that document's text, linked back to its document, in
the order it appeared, along with the embedding vector and the full-text
search vector used for retrieval.
"""

from psycopg2.extensions import connection as PostgresConnection

from src.embedding.config import EMBEDDING_DIMENSION

CREATE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_DOCUMENTS_TABLE = """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        department TEXT,
        academic_year TEXT,
        access_level TEXT NOT NULL CHECK (access_level IN ('public', 'student', 'faculty', 'admin')),
        source_filename TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

CREATE_CHUNKS_TABLE = f"""
    CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        page_number INTEGER,
        chunk_index INTEGER NOT NULL,
        embedding VECTOR({EMBEDDING_DIMENSION}),
        search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index)
    )
"""

# Covers the case where the chunks table was created before the embedding
# column existed. IF NOT EXISTS makes this a harmless no-op otherwise.
ADD_EMBEDDING_COLUMN = f"""
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding VECTOR({EMBEDDING_DIMENSION})
"""

# Same idea for the full-text search column: computed automatically from
# chunk_text by PostgreSQL, so it never needs to be set manually on insert.
ADD_SEARCH_VECTOR_COLUMN = """
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
"""

CREATE_CHUNKS_DOCUMENT_ID_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)
"""

CREATE_CHUNKS_SEARCH_VECTOR_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING GIN (search_vector)
"""


def create_tables(connection: PostgresConnection) -> None:
    """Create the vector extension, tables, columns, and indexes if they do not already exist."""
    with connection.cursor() as cursor:
        cursor.execute(CREATE_VECTOR_EXTENSION)
        cursor.execute(CREATE_DOCUMENTS_TABLE)
        cursor.execute(CREATE_CHUNKS_TABLE)
        cursor.execute(ADD_EMBEDDING_COLUMN)
        cursor.execute(ADD_SEARCH_VECTOR_COLUMN)
        cursor.execute(CREATE_CHUNKS_DOCUMENT_ID_INDEX)
        cursor.execute(CREATE_CHUNKS_SEARCH_VECTOR_INDEX)
    connection.commit()
