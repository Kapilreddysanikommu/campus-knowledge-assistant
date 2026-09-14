"""
Inserts ingested documents and their chunks into PostgreSQL.

This module is the bridge between the Step 1 pipeline (ingestion and
chunking, which produce plain Python objects) and the database schema
defined in schema.py.
"""

from dataclasses import dataclass
from typing import List, Optional

from psycopg2.extensions import connection as PostgresConnection

from src.chunking.text_chunker import TextChunk

VALID_ACCESS_LEVELS = {"public", "student", "faculty", "admin"}


@dataclass
class DocumentMetadata:
    """Metadata describing a source document, stored alongside its text."""

    title: str
    source_filename: str
    department: Optional[str] = None
    academic_year: Optional[str] = None
    access_level: str = "public"


INSERT_DOCUMENT_QUERY = """
    INSERT INTO documents (title, department, academic_year, access_level, source_filename)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
"""

INSERT_CHUNK_QUERY = """
    INSERT INTO chunks (document_id, chunk_text, token_count, page_number, chunk_index)
    VALUES (%s, %s, %s, %s, %s)
"""


def insert_document(connection: PostgresConnection, metadata: DocumentMetadata) -> int:
    """Insert one document row and return its generated id."""
    if metadata.access_level not in VALID_ACCESS_LEVELS:
        raise ValueError(
            f"access_level must be one of {sorted(VALID_ACCESS_LEVELS)}, got: {metadata.access_level}"
        )

    with connection.cursor() as cursor:
        cursor.execute(
            INSERT_DOCUMENT_QUERY,
            (
                metadata.title,
                metadata.department,
                metadata.academic_year,
                metadata.access_level,
                metadata.source_filename,
            ),
        )
        document_id = cursor.fetchone()[0]

    connection.commit()
    return document_id


def insert_chunks(connection: PostgresConnection, document_id: int, chunks: List[TextChunk]) -> int:
    """Insert every chunk belonging to a document and return how many were inserted.

    page_number is stored as NULL for now, since the Step 1 chunker works
    from a single flattened Markdown string and does not yet track which
    PDF page each chunk came from.
    """
    rows = [
        (document_id, chunk.text, chunk.token_count, None, chunk.chunk_index)
        for chunk in chunks
    ]

    with connection.cursor() as cursor:
        cursor.executemany(INSERT_CHUNK_QUERY, rows)

    connection.commit()
    return len(rows)
