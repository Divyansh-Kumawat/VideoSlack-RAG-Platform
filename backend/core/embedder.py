# backend/core/embedder.py
# ─────────────────────────────────────────────────────────────────────────────
# Local embedding model (sentence-transformers). Default: BAAI/bge-small-en-v1.5
# Runs entirely on this machine — transcripts never leave it for embedding.
#
#     vectors   = embed_texts(["hello world"])
#     query_vec = embed_query("what is the hook?")
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import threading
from typing import Sequence

from langchain_core.embeddings import Embeddings

from backend import config

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = config.EMBEDDING_MODEL

# BGE v1.5 retrieves better when queries (not passages) carry this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None
_lock = threading.Lock()


def get_model():
    """Loads the embedding model once (thread-safe) and reuses it."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
                _model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info(
                    "Embedding model ready (dim=%s)",
                    _model.get_sentence_embedding_dimension(),
                )
    return _model


def get_embedding_dimension() -> int:
    dim = get_model().get_sentence_embedding_dimension()
    if not dim:
        raise RuntimeError("Embedding model did not report a dimension.")
    return int(dim)


def _encode(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embeds passages. Output length always equals input length."""
    if not texts:
        return []
    return _encode([(t or " ").strip() or " " for t in texts])


def embed_query(query: str) -> list[float]:
    query = (query or "").strip()
    if not query:
        raise ValueError("embed_query received an empty string.")
    prefix = BGE_QUERY_INSTRUCTION if "bge" in EMBEDDING_MODEL.lower() else ""
    return _encode([prefix + query])[0]


class VideoIQEmbeddings(Embeddings):
    """LangChain-compatible wrapper used by the Chroma vector store."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    vecs = embed_texts(["The hook grabbed attention.", "Shorter videos get more engagement."])
    q = embed_query("What makes a good hook?")
    print(f"dim={len(q)} passages={len(vecs)} sim={sum(a * b for a, b in zip(q, vecs[0])):.3f}")
