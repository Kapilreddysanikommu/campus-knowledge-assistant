"""
Manual test script for ingestion and chunking.

Run this on a sample PDF to see exactly what text gets extracted and how it
gets split into chunks, so the output can be inspected by eye before it ever
reaches an embedding model or a vector store.

Example:
    python scripts/inspect_chunks.py data/sample_pdfs/syllabus.pdf
    python scripts/inspect_chunks.py data/sample_pdfs/syllabus.pdf --chunk-size 300 --chunk-overlap 50
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_ingestor import extract_text_from_pdf
from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    TextChunker,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF with Docling and print the resulting chunks."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to test with")
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

    print(f"Reading PDF: {args.pdf_path}")
    ingested_document = extract_text_from_pdf(args.pdf_path)
    print(f"Extracted {len(ingested_document.markdown_text)} characters of Markdown text\n")

    chunker = TextChunker(
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
    )
    chunks = chunker.chunk_text(
        ingested_document.markdown_text,
        source_document=ingested_document.source_path,
    )

    print(
        f"Produced {len(chunks)} chunks "
        f"(chunk_size={args.chunk_size} tokens, overlap={args.chunk_overlap} tokens)\n"
    )

    for chunk in chunks:
        print("-" * 80)
        print(f"Chunk {chunk.chunk_index} | {chunk.token_count} tokens")
        print("-" * 80)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()
