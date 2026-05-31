# backend/core/embedder.py
# ─────────────────────────────────────────────────────────────────────────────
# Embedding model wrapper.
#
# Loads BGE (or OpenAI) once at startup. The rest of the codebase
# calls embed_texts() and never touches sentence-transformers directly.
#
# BGE small: 33M params, 384-dim vectors, beats OpenAI Ada on MTEB, zero cost.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import logging
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMB_MODEL   = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model instances
# ─────────────────────────────────────────────────────────────────────────────

_local_model = None          # sentence_transformers.SentenceTransformer
_openai_client = None        # openai.OpenAI


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", EMBEDDING_MODEL)
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(
            "Embedding model loaded. Dimension: %d",
            _local_model.get_sentence_embedding_dimension(),
        )
    return _local_model


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY not set but EMBEDDING_PROVIDER=openai. "
                "Set the key in .env or switch to EMBEDDING_PROVIDER=local."
            )
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI embedding client initialised (model=%s)", OPENAI_EMB_MODEL)
    return _openai_client


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    Embeds a list of text strings into dense vectors.

    Uses whichever provider is set in EMBEDDING_PROVIDER env var.
    - "local"  → BGE via sentence-transformers (default, free, fast)
    - "openai" → OpenAI text-embedding-3-small

    BGE-specific: prepends "Represent this sentence: " prefix for
    retrieval queries (improves accuracy per BGE docs). For passage
    embedding (chunked transcript text) no prefix is needed.

    Args:
        texts: List of strings to embed. Can be a single string in a list.

    Returns:
        List of float vectors, one per input text.
        All vectors have the same dimension (384 for BGE-small, 1536 for OpenAI-small).
    """
    if not texts:
        return []

    # Normalise: strip empty strings that would confuse the model
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    else:
        return _embed_local(texts)


def embed_query(query: str) -> list[float]:
    """
    Embeds a single query string.

    For BGE models, queries get a special prefix ("Represent this question: ")
    that improves retrieval performance. Passages do NOT get this prefix.

    Args:
        query: The user's question or search string.

    Returns:
        Single float vector.
    """
    query = query.strip()
    if not query:
        raise ValueError("embed_query received an empty string.")

    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai([query])[0]

    # BGE query prefix
    prefixed = f"Represent this question for searching relevant passages: {query}"
    return _embed_local([prefixed])[0]


def get_embedding_dimension() -> int:
    """
    Returns the vector dimension for the active embedding model.
    Needed when initialising ChromaDB collection.
    """
    if EMBEDDING_PROVIDER == "openai":
        # text-embedding-3-small → 1536
        return 1536

    model = _get_local_model()
    return model.get_sentence_embedding_dimension()


# ─────────────────────────────────────────────────────────────────────────────
# Private implementations
# ─────────────────────────────────────────────────────────────────────────────

def _embed_local(texts: list[str]) -> list[list[float]]:
    """Batch embed using sentence-transformers (BGE or any SBERT model)."""
    model = _get_local_model()

    # normalize_embeddings=True gives cosine-ready unit vectors
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    # numpy array → list of Python floats
    return [emb.tolist() for emb in embeddings]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Batch embed using OpenAI embeddings API."""
    client = _get_openai_client()

    # OpenAI recommends replacing newlines with spaces for better quality
    cleaned = [t.replace("\n", " ") for t in texts]

    response = client.embeddings.create(
        model=OPENAI_EMB_MODEL,
        input=cleaned,
    )
    # Sort by index to preserve order (API may reorder in batch)
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [item.embedding for item in sorted_data]


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible embedding class
# ─────────────────────────────────────────────────────────────────────────────

class VideoIQEmbeddings:
    """
    Thin LangChain Embeddings-compatible wrapper around embed_texts / embed_query.
    Pass an instance of this to ChromaDB's LangChain integration.

    Usage:
        embeddings = VideoIQEmbeddings()
        vectorstore = Chroma(embedding_function=embeddings, ...)
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.embedder
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sample_passages = [
        "The hook in the first five seconds grabbed attention immediately.",
        "Video analytics show higher engagement for shorter content.",
        "The creator used trending audio to boost organic reach.",
    ]
    sample_query = "What makes a good hook in a social media video?"

    print(f"\n── Embedder test (provider={EMBEDDING_PROVIDER}, model={EMBEDDING_MODEL}) ──")

    # Passage embeddings
    t0 = time.time()
    vectors = embed_texts(sample_passages)
    elapsed = time.time() - t0

    print(f"\nPassage embeddings:")
    for i, (text, vec) in enumerate(zip(sample_passages, vectors)):
        print(f"  [{i}] '{text[:55]}...' → dim={len(vec)}, first3={[round(v,4) for v in vec[:3]]}")
    print(f"  Time: {elapsed:.3f}s for {len(sample_passages)} passages")

    # Query embedding
    t1 = time.time()
    qvec = embed_query(sample_query)
    print(f"\nQuery embedding:")
    print(f"  '{sample_query}'")
    print(f"  dim={len(qvec)}, first3={[round(v,4) for v in qvec[:3]]}")
    print(f"  Time: {time.time()-t1:.3f}s")

    # Dimension check
    dim = get_embedding_dimension()
    print(f"\nEmbedding dimension: {dim}")
    assert len(vectors[0]) == dim, "Dimension mismatch!"
    print("All checks passed.")