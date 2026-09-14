"""
Runs the full pipeline: ingest one or more PDFs, chunk them, generate an
embedding for each chunk, and store everything in PostgreSQL.

Single file, metadata given as flags:
    python scripts/run_pipeline.py data/sample_pdfs/sample_syllabus.pdf \\
        --title "CS 101 Syllabus" --department "Computer Science" \\
        --academic-year "2025-2026" --access-level student

Multiple files in one run, metadata given by a manifest file:
    python scripts/run_pipeline.py --manifest data/document_manifest.json

The manifest is a JSON file containing a list of objects, each with
pdf_path, title, and optionally department, academic_year, and
access_level. See data/document_manifest.json for an example.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


@dataclass
class DocumentJob:
    """One PDF to ingest, together with the metadata to store alongside it."""

    pdf_path: Path
    title: str
    department: Optional[str] = None
    academic_year: Optional[str] = None
    access_level: str = "public"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest one or more PDFs, chunk them, embed them, and store the results in PostgreSQL."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=None,
        help="Path to a single PDF file to ingest (omit this and use --manifest for multiple files)",
    )
    parser.add_argument("--title", default=None, help="Human-readable document title (single-file mode)")
    parser.add_argument("--department", default=None, help="Owning department, e.g. Computer Science")
    parser.add_argument("--academic-year", default=None, help="Academic year, e.g. 2025-2026")
    parser.add_argument(
        "--access-level",
        default="public",
        choices=["public", "student", "faculty", "admin"],
        help="Who is allowed to see this document (default: public)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a JSON manifest describing multiple PDFs to ingest in one run",
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


def load_jobs_from_manifest(manifest_path: Path) -> List[DocumentJob]:
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        entries = json.load(manifest_file)

    return [
        DocumentJob(
            pdf_path=Path(entry["pdf_path"]),
            title=entry["title"],
            department=entry.get("department"),
            academic_year=str(entry["academic_year"]) if entry.get("academic_year") is not None else None,
            access_level=entry.get("access_level", "public"),
        )
        for entry in entries
    ]


def build_jobs(args: argparse.Namespace) -> List[DocumentJob]:
    if args.manifest:
        return load_jobs_from_manifest(Path(args.manifest))

    if not args.pdf_path or not args.title:
        raise SystemExit("Provide either --manifest, or a pdf_path together with --title.")

    return [
        DocumentJob(
            pdf_path=Path(args.pdf_path),
            title=args.title,
            department=args.department,
            academic_year=args.academic_year,
            access_level=args.access_level,
        )
    ]


def process_document(connection, embedder: ChunkEmbedder, chunker: TextChunker, job: DocumentJob) -> int:
    """Ingest, chunk, embed, and store one document. Returns its document id."""
    print(f"\nProcessing: {job.pdf_path}")

    ingested_document = extract_text_from_pdf(job.pdf_path)
    print(f"  Extracted {len(ingested_document.markdown_text)} characters of Markdown text")

    chunks = chunker.chunk_text(
        ingested_document.markdown_text,
        source_document=ingested_document.source_path,
    )
    print(f"  Split into {len(chunks)} chunks")

    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    metadata = DocumentMetadata(
        title=job.title,
        source_filename=job.pdf_path.name,
        department=job.department,
        academic_year=job.academic_year,
        access_level=job.access_level,
    )
    document_id = insert_document(connection, metadata)
    inserted_chunk_count = insert_chunks(connection, document_id, chunks, embeddings)
    print(f"  Stored as document id {document_id} with {inserted_chunk_count} chunks")

    return document_id


def main() -> None:
    args = parse_arguments()
    jobs = build_jobs(args)

    chunker = TextChunker(chunk_size_tokens=args.chunk_size, chunk_overlap_tokens=args.chunk_overlap)
    embedder = ChunkEmbedder()

    connection = get_connection()
    succeeded_count = 0
    failed_count = 0
    try:
        create_tables(connection)

        for job in jobs:
            try:
                process_document(connection, embedder, chunker, job)
                succeeded_count += 1
            except Exception as error:
                # One bad PDF should not stop the rest of a multi-document run.
                failed_count += 1
                print(f"  Failed to process {job.pdf_path}: {error}")
    finally:
        connection.close()

    print(f"\nPipeline complete: {succeeded_count} succeeded, {failed_count} failed")


if __name__ == "__main__":
    main()
