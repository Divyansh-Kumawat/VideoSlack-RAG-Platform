# Shared fixtures: isolated ChromaDB + data dir per test, fake embeddings,
# and stubbed network calls — the suite runs offline in a few seconds.

from __future__ import annotations

import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings

from backend import config
from backend.core import vector_store


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (no model download)."""

    dim = 128

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CHROMA_COLLECTION_NAME", "test_chunks")
    vector_store.reset()
    vector_store.set_embeddings(FakeEmbeddings())
    yield
    vector_store.reset()


def make_transcript(topic: str, n: int = 30) -> dict:
    segs = [
        {"text": f"In part {i} we talk about {topic} and why it matters.", "start": float(i * 4)}
        for i in range(n)
    ]
    return {
        "text": " ".join(s["text"] for s in segs),
        "language": "en",
        "source": "captions",
        "segments": segs,
    }


def make_metadata(youtube_id: str, title: str, views: int, likes: int, comments: int) -> dict:
    return {
        "youtube_id": youtube_id,
        "url": f"https://www.youtube.com/watch?v={youtube_id}",
        "title": title,
        "creator": f"{title} Channel",
        "follower_count": 10_000,
        "views": views,
        "likes": likes,
        "comments": comments,
        "upload_date": "2026-01-15",
        "duration_seconds": 120,
        "hashtags": ["shorts"],
        "thumbnail_url": None,
        "description": "A test video.",
    }
