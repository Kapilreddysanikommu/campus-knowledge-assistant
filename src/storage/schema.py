"""
Defines and creates the database schema for documents and their chunks.

A document row represents one source file (a syllabus, handbook, policy,
and so on) along with metadata used to control who can see it. Each chunk
row is one piece of that document's text, linked back to its document, in
the order it appeared.
"""

from psycopg2.extensions import connection as PostgresConnection

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

CREATE_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        page_number INTEGER,
        chunk_index INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index)
    )
"""

CREATE_CHUNKS_DOCUMENT_ID_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)
"""


def create_tables(connection: PostgresConnection) -> None:
    """Create the documents and chunks tables if they do not already exist."""
    with connection.cursor() as cursor:
        cursor.execute(CREATE_DOCUMENTS_TABLE)
        cursor.execute(CREATE_CHUNKS_TABLE)
        cursor.execute(CREATE_CHUNKS_DOCUMENT_ID_INDEX)
    connection.commit()
