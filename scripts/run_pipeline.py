"""
Runs the full pipeline: ingest a PDF, chunk it, generate an embedding for
each chunk, and store everything in PostgreSQL.

Example:
    python scripts/run_pipeline.py data/sample_pdfs/sample_syllabus.pdf \\
        --title "CS 101 Syllabus" --department "Computer Science" \\
        --academic-year "2025-2026" --access-level student
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    TextChunker,
)
from src.embedding.embedder import ChunkEmbedder
from src.ingestion.pdf_ingestor import extract_text_from_pdf
from src.storage.database import get_connection
from src.storage.repository import DocumentMetadata, insert_chunks, insert_document
from src.storage.schema import create_tables


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF, chunk it, and store the results in PostgreSQL."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    parser.add_argument("--title", required=True, help="Human-readable document title")
    parser.add_argument("--department", default=None, help="Owning department, e.g. Computer Science")
    parser.add_argument("--academic-year", default=None, help="Academic year, e.g. 2025-2026")
    parser.add_argument(
        "--access-level",
        default="public",
        choices=["public", "student", "faculty", "admin"],
        help="Who is allowed to see this document (default: public)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_TOKENS,
        help=f"Maximum tokens per chunk (default: {DEFAULT_CHUNK_SIZE_TOKENS})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
        help=f"Overlapping tokens between consecutive chunks (default: {DEFAULT_CHUNK_OVERLAP_TOKENS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print(f"Extracting text from: {args.pdf_path}")
    ingested_document = extract_text_from_pdf(args.pdf_path)
    print(f"Extracted {len(ingested_document.markdown_text)} characters of Markdown text")

    chunker = TextChunker(
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
    )
    chunks = chunker.chunk_text(
        ingested_document.markdown_text,
        source_document=ingested_document.source_path,
    )
    print(f"Split document into {len(chunks)} chunks")

    print("Generating embeddings")
    embedder = ChunkEmbedder()
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    print(f"Generated {len(embeddings)} embeddings")

    connection = get_connection()
    try:
        create_tables(connection)

        metadata = DocumentMetadata(
            title=args.title,
            source_filename=Path(args.pdf_path).name,
            department=args.department,
            academic_year=args.academic_year,
            access_level=args.access_level,
        )
        document_id = insert_document(connection, metadata)
        print(f"Inserted document record with id {document_id}")

        inserted_chunk_count = insert_chunks(connection, document_id, chunks, embeddings)
        print(f"Inserted {inserted_chunk_count} chunk records")
    finally:
        connection.close()

    print("Pipeline complete")


if __name__ == "__main__":
    main()
