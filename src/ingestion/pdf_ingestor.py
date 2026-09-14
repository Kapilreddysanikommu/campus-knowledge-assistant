"""
Extracts text from PDF documents using Docling.

Docling parses the visual and logical structure of a PDF (headers, paragraphs,
tables, lists) and converts it into Markdown. Keeping the Markdown structure
matters for later steps: headers give us natural section boundaries for
chunking, and tables stay readable instead of turning into a wall of numbers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from docling.document_converter import DocumentConverter


@dataclass
class IngestedDocument:
    """Holds the text extracted from a single source PDF."""

    source_path: str
    markdown_text: str


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> IngestedDocument:
    """Convert a PDF file into structured Markdown text.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        An IngestedDocument containing the source path and the extracted
        Markdown text, with headers and tables preserved where possible.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

    converter = DocumentConverter()
    conversion_result = converter.convert(str(pdf_path))
    markdown_text = conversion_result.document.export_to_markdown()

    return IngestedDocument(source_path=str(pdf_path), markdown_text=markdown_text)
