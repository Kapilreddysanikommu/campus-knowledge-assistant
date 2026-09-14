"""
Splits extracted document text into overlapping chunks for embedding.

Chunks are sized by token count rather than character count, because
embedding models have a context window measured in tokens. Overlap between
consecutive chunks carries a bit of trailing context forward, so a sentence
or idea that gets cut off at a chunk boundary is not completely lost to
whichever chunk retrieval picks.
"""

from dataclasses import dataclass
from typing import List

import tiktoken

DEFAULT_CHUNK_SIZE_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 75
DEFAULT_ENCODING_NAME = "cl100k_base"


@dataclass
class TextChunk:
    """A single chunk of text ready to be embedded."""

    text: str
    chunk_index: int
    token_count: int
    source_document: str


class TextChunker:
    """Splits text into overlapping, token-bounded chunks.

    Splitting works paragraph by paragraph so chunks tend to break on
    natural boundaries instead of mid-sentence. Paragraphs are packed into a
    chunk until the next paragraph would push it past chunk_size_tokens, at
    which point a new chunk starts, seeded with the last chunk_overlap_tokens
    worth of text from the previous chunk.
    """

    def __init__(
        self,
        chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        encoding_name: str = DEFAULT_ENCODING_NAME,
    ):
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, source_document: str = "") -> List[TextChunk]:
        """Split text into a list of overlapping TextChunk objects."""
        paragraphs = self._split_into_paragraphs(text)

        chunks: List[TextChunk] = []
        current_paragraphs: List[str] = []
        current_token_count = 0

        for paragraph in paragraphs:
            paragraph_token_count = self._count_tokens(paragraph)

            if paragraph_token_count > self.chunk_size_tokens:
                current_paragraphs, current_token_count = self._flush_chunk(
                    chunks, current_paragraphs, source_document
                )
                for piece in self._split_oversized_paragraph(paragraph):
                    chunks.append(
                        TextChunk(
                            text=piece,
                            chunk_index=len(chunks),
                            token_count=self._count_tokens(piece),
                            source_document=source_document,
                        )
                    )
                continue

            would_overflow = current_token_count + paragraph_token_count > self.chunk_size_tokens
            if would_overflow and current_paragraphs:
                current_paragraphs, current_token_count = self._flush_chunk(
                    chunks, current_paragraphs, source_document, keep_overlap=True
                )

            current_paragraphs.append(paragraph)
            current_token_count += paragraph_token_count

        self._flush_chunk(chunks, current_paragraphs, source_document)
        return chunks

    def _flush_chunk(
        self,
        chunks: List[TextChunk],
        current_paragraphs: List[str],
        source_document: str,
        keep_overlap: bool = False,
    ):
        """Finalize the in-progress chunk and return the state for the next one."""
        if not current_paragraphs:
            return [], 0

        chunk_text_value = "\n\n".join(current_paragraphs)
        chunks.append(
            TextChunk(
                text=chunk_text_value,
                chunk_index=len(chunks),
                token_count=self._count_tokens(chunk_text_value),
                source_document=source_document,
            )
        )

        if not keep_overlap:
            return [], 0

        overlap_text = self._get_overlap_text(chunk_text_value)
        if not overlap_text:
            return [], 0
        return [overlap_text], self._count_tokens(overlap_text)

    def _get_overlap_text(self, chunk_text_value: str) -> str:
        """Return the trailing slice of a chunk to seed the next chunk with."""
        tokens = self.encoding.encode(chunk_text_value)
        if len(tokens) <= self.chunk_overlap_tokens:
            return chunk_text_value
        overlap_tokens = tokens[-self.chunk_overlap_tokens :]
        return self.encoding.decode(overlap_tokens)

    def _split_oversized_paragraph(self, paragraph: str) -> List[str]:
        """Split a single paragraph that is larger than chunk_size_tokens on its own."""
        tokens = self.encoding.encode(paragraph)
        step = self.chunk_size_tokens - self.chunk_overlap_tokens

        pieces = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size_tokens, len(tokens))
            pieces.append(self.encoding.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return pieces

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text on blank lines, which is how Markdown separates paragraphs."""
        raw_paragraphs = text.split("\n\n")
        return [paragraph.strip() for paragraph in raw_paragraphs if paragraph.strip()]

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
