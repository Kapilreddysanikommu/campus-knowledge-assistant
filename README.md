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
    storage/      Stores chunks and their embeddings in a vector database
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

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Testing ingestion and chunking

Place a sample PDF in `data/sample_pdfs/`, then run:

```
python scripts/inspect_chunks.py data/sample_pdfs/your_file.pdf
```

Optional flags to override the default chunk size and overlap:

```
python scripts/inspect_chunks.py data/sample_pdfs/your_file.pdf --chunk-size 300 --chunk-overlap 50
```
