"""
Manual test script for the embedding module.

Compares two pairs of text: one pair that is topically similar and one
pair that is topically different. If embeddings are working correctly, the
similar pair should score noticeably higher than the different pair.

Example with the built-in defaults:
    python scripts/test_embedding_similarity.py

Example with your own text:
    python scripts/test_embedding_similarity.py \\
        --similar-a "Homework is due every Friday at midnight." \\
        --similar-b "All assignments must be submitted by 11:59 PM on Fridays." \\
        --different-a "Homework is due every Friday at midnight." \\
        --different-b "Visitor parking permits are available at the campus kiosk."
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedder import ChunkEmbedder
from src.embedding.similarity import cosine_similarity

DEFAULT_SIMILAR_TEXT_A = "Homework assignments make up 30 percent of the final course grade."
DEFAULT_SIMILAR_TEXT_B = "Thirty percent of your grade comes from homework submitted during the semester."
DEFAULT_DIFFERENT_TEXT_A = "Homework assignments make up 30 percent of the final course grade."
DEFAULT_DIFFERENT_TEXT_B = "Students must display a valid parking permit on campus between 7am and 6pm."


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cosine similarity between a similar pair and a different pair of text."
    )
    parser.add_argument("--similar-a", default=DEFAULT_SIMILAR_TEXT_A)
    parser.add_argument("--similar-b", default=DEFAULT_SIMILAR_TEXT_B)
    parser.add_argument("--different-a", default=DEFAULT_DIFFERENT_TEXT_A)
    parser.add_argument("--different-b", default=DEFAULT_DIFFERENT_TEXT_B)
    return parser.parse_args()


def print_pair_result(label: str, text_a: str, text_b: str, similarity_score: float) -> None:
    print(label)
    print(f"  A: {text_a}")
    print(f"  B: {text_b}")
    print(f"  Cosine similarity: {similarity_score:.4f}\n")


def main() -> None:
    args = parse_arguments()
    embedder = ChunkEmbedder()

    similar_embedding_a = embedder.embed_text(args.similar_a)
    similar_embedding_b = embedder.embed_text(args.similar_b)
    different_embedding_a = embedder.embed_text(args.different_a)
    different_embedding_b = embedder.embed_text(args.different_b)

    similar_score = cosine_similarity(similar_embedding_a, similar_embedding_b)
    different_score = cosine_similarity(different_embedding_a, different_embedding_b)

    print_pair_result("Similar pair:", args.similar_a, args.similar_b, similar_score)
    print_pair_result("Different pair:", args.different_a, args.different_b, different_score)

    if similar_score > different_score:
        print("Result: the similar pair scored higher, as expected.")
    else:
        print("Result: the similar pair did NOT score higher than the different pair.")


if __name__ == "__main__":
    main()
