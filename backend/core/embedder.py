# backend/core/embedder.py
# ─────────────────────────────────────────────────────────────────────────────
# Embedding model wrapper.
#
# Uses Hugging Face Sentence Transformers locally.
# Default model: BAAI/bge-small-en-v1.5
#
# Usage:
#     vectors = embed_texts(["hello world"])
#     query_vec = embed_query("what is machine learning?")
#
# Compatible with ChromaDB and LangChain.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "local"
)
# Singleton model instance
_local_model = None


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_model():
    """
    Loads the embedding model once and reuses it.
    """
    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL
        )

        _local_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        logger.info(
            "Embedding model loaded successfully."
        )

        logger.info(
            "Embedding dimension: %d",
            _local_model.get_embedding_dimension()
        )

    return _local_model


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(
    texts: Sequence[str]
) -> list[list[float]]:
    """
    Embeds multiple passages/documents.

    Args:
        texts: List of text strings.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not texts:
        return []

    return _embed_local(texts)


def embed_query(
    query: str
) -> list[float]:
    """
    Embeds a search query.

    BGE performs better when queries are prefixed.

    Args:
        query: User search question.

    Returns:
        Embedding vector.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "embed_query received an empty string."
        )

    prefixed_query = (
        "Represent this question for searching relevant passages: "
        f"{query}"
    )

    return _embed_local(
        [prefixed_query]
    )[0]


def get_embedding_dimension() -> int:
    """
    Returns embedding dimension.
    """
    model = _get_local_model()
    dimension = model.get_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Embedding model did not provide a valid dimension.")
    return dimension


# ─────────────────────────────────────────────────────────────────────────────
# Internal Embedding Logic
# ─────────────────────────────────────────────────────────────────────────────

def _embed_local(
    texts: list[str]
) -> list[list[float]]:
    """
    Batch embedding using Sentence Transformers.
    """
    model = _get_local_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return [
        embedding.tolist()
        for embedding in embeddings
    ]


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-Compatible Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VideoIQEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class.
    """

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(
        self,
        text: str
    ) -> list[float]:
        return embed_query(text)

# ─────────────────────────────────────────────────────────────────────────────
# Standalone Test
# Run:
#     python -m backend.core.embedder
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )

    sample_passages = [
        "The hook in the first five seconds grabbed attention immediately.",
        "Video analytics show higher engagement for shorter content.",
        "The creator used trending audio to boost organic reach.",
    ]

    sample_query = (
        "What makes a good hook in a social media video?"
    )

    print(
        f"\n── Embedder Test "
        f"(model={EMBEDDING_MODEL}) ──"
    )

    # Passage embeddings
    start = time.time()

    vectors = embed_texts(
        sample_passages
    )

    elapsed = time.time() - start

    print("\nPassage Embeddings:")

    for i, (text, vec) in enumerate(
        zip(sample_passages, vectors)
    ):
        print(
            f"[{i}] "
            f"dim={len(vec)} "
            f"first3={[round(v, 4) for v in vec[:3]]}"
        )

    print(
        f"\nTime: {elapsed:.3f}s "
        f"for {len(sample_passages)} passages"
    )

    # Query embedding
    start = time.time()

    query_vector = embed_query(
        sample_query
    )

    elapsed = time.time() - start

    print("\nQuery Embedding:")
    print(sample_query)

    print(
        f"dim={len(query_vector)} "
        f"first3={[round(v, 4) for v in query_vector[:3]]}"
    )

    print(
        f"Time: {elapsed:.3f}s"
    )

    # Dimension verification
    dimension = get_embedding_dimension()

    print(
        f"\nEmbedding Dimension: {dimension}"
    )

    assert (
        len(vectors[0]) == dimension
    ), "Dimension mismatch!"

    print("\n✅ All checks passed.")