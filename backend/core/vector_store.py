# backend/core/vector_store.py
# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB vector store — chunking, upsert, retrieval.
#
# Every chunk carries video_id ("A"/"B") so retrieval can target one video,
# and start_seconds so answers can cite the exact moment in the video.
# Persists to CHROMA_PERSIST_DIR (default backend/chroma_db).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import config

logger = logging.getLogger(__name__)

NO_TIMESTAMP = -1.0   # Chroma metadata can't hold None


# ─────────────────────────────────────────────────────────────────────────────
# Chunk model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    text:          str
    video_id:      str            # "A" or "B"
    chunk_index:   int
    start_char:    int
    start_seconds: Optional[float]
    creator:       str
    title:         str
    youtube_id:    str
    url:           str

    def metadata(self) -> dict:
        return {
            "video_id":      self.video_id,
            "chunk_index":   self.chunk_index,
            "start_char":    self.start_char,
            "start_seconds": NO_TIMESTAMP if self.start_seconds is None else float(self.start_seconds),
            "creator":       self.creator,
            "title":         self.title,
            "youtube_id":    self.youtube_id,
            "url":           self.url,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[chromadb.api.ClientAPI] = None
_vectorstore: Optional[Chroma] = None
_embeddings: Optional[Embeddings] = None
_lock = threading.Lock()


def set_embeddings(embeddings: Embeddings) -> None:
    """Override the embedding function (used by tests). Call before first use."""
    global _embeddings, _vectorstore
    _embeddings = embeddings
    _vectorstore = None


def reset() -> None:
    """Drop cached client/vectorstore so the next call re-reads config (tests)."""
    global _client, _vectorstore
    _client = None
    _vectorstore = None


def _get_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        logger.info("ChromaDB client at %s", config.CHROMA_PERSIST_DIR)
    return _client


def get_vectorstore() -> Chroma:
    """LangChain Chroma wrapper (singleton). Creates the collection if needed."""
    global _vectorstore, _embeddings
    if _vectorstore is None:
        with _lock:
            if _vectorstore is None:
                if _embeddings is None:
                    from backend.core.embedder import VideoIQEmbeddings

                    _embeddings = VideoIQEmbeddings()
                _vectorstore = Chroma(
                    client=_get_client(),
                    collection_name=config.CHROMA_COLLECTION_NAME,
                    embedding_function=_embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
    return _vectorstore


def _collection():
    return get_vectorstore()._collection


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_transcript(
    transcript: str,
    video_id: str,
    creator: str,
    title: str,
    youtube_id: str,
    url: str,
    segments: Sequence[dict] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TranscriptChunk]:
    """
    Splits a transcript into overlapping chunks of ~chunk_size characters.

    With timestamped `segments` (from transcript.py), chunks are built from
    whole caption segments so each chunk knows the second it starts at.
    The last few segments of a chunk (≤ chunk_overlap chars) are repeated
    at the start of the next one so ideas aren't cut in half.
    Without segments, it falls back to plain character splitting.
    """
    size    = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )

    def make(text: str, idx: int, start_char: int, start_s: Optional[float]) -> TranscriptChunk:
        return TranscriptChunk(
            text=text, video_id=video_id, chunk_index=idx, start_char=start_char,
            start_seconds=start_s, creator=creator, title=title,
            youtube_id=youtube_id, url=url,
        )

    chunks: list[TranscriptChunk] = []

    if not segments:
        offset = 0
        for i, text in enumerate(splitter.split_text(transcript)):
            start = transcript.find(text, offset)
            start = offset if start == -1 else start
            offset = start + len(text) - overlap if len(text) > overlap else start + len(text)
            chunks.append(make(text, i, max(start, 0), None))
        return chunks

    # Expand segments into (text, start_seconds, start_char) units, splitting
    # any single segment that is itself longer than a chunk.
    units: list[tuple[str, float, int]] = []
    char_pos = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        pieces = splitter.split_text(text) if len(text) > size else [text]
        for piece in pieces:
            units.append((piece, float(seg.get("start") or 0.0), char_pos))
        char_pos += len(text) + 1   # segments are joined with a single space

    window: list[tuple[str, float, int]] = []
    window_len = 0

    def flush() -> None:
        if window:
            chunks.append(make(" ".join(u[0] for u in window), len(chunks), window[0][2], window[0][1]))

    for unit in units:
        unit_len = len(unit[0]) + 1
        if window and window_len + unit_len > size:
            flush()
            # carry trailing units forward as overlap
            carried: list[tuple[str, float, int]] = []
            carried_len = 0
            for u in reversed(window):
                if carried_len + len(u[0]) + 1 > overlap:
                    break
                carried.insert(0, u)
                carried_len += len(u[0]) + 1
            window, window_len = carried, carried_len
        window.append(unit)
        window_len += unit_len
    flush()

    logger.info("Chunked Video %s: %d chunks from %d chars", video_id, len(chunks), len(transcript))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_id(video_id: str, chunk_index: int, text: str) -> str:
    return hashlib.md5(f"{video_id}:{chunk_index}:{text[:100]}".encode()).hexdigest()


def upsert_chunks(chunks: list[TranscriptChunk]) -> int:
    """Embeds and stores chunks. Deterministic IDs make re-ingestion idempotent."""
    if not chunks:
        logger.warning("upsert_chunks called with no chunks.")
        return 0
    get_vectorstore().add_texts(
        texts=[c.text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
        ids=[_chunk_id(c.video_id, c.chunk_index, c.text) for c in chunks],
    )
    logger.info("Upserted %d chunks", len(chunks))
    return len(chunks)


def clear_video(video_id: str) -> int:
    """Deletes every chunk belonging to one video. Returns how many were removed."""
    col = _collection()
    ids = col.get(where={"video_id": video_id}, include=[]).get("ids", [])
    if ids:
        col.delete(ids=ids)
    logger.info("Cleared %d chunks for video %s", len(ids), video_id)
    return len(ids)


def clear_all() -> None:
    for vid in ("A", "B"):
        clear_video(vid)


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

def get_chunk_count(video_id: Optional[str] = None) -> int:
    try:
        col = _collection()
        if video_id:
            return len(col.get(where={"video_id": video_id}, include=[]).get("ids", []))
        return col.count()
    except Exception as e:   # never let a health check crash
        logger.warning("Chunk count failed: %s", e)
        return 0


def _search(query: str, k: int, video_id: Optional[str]) -> list[tuple[Document, float]]:
    if k <= 0 or get_chunk_count(video_id) == 0:
        return []
    kwargs = {"filter": {"video_id": video_id}} if video_id else {}
    results = get_vectorstore().similarity_search_with_score(query, k=k, **kwargs)
    # cosine distance → similarity in [0, 1]
    return [(doc, round(max(0.0, 1.0 - float(dist)), 4)) for doc, dist in results]


def retrieve(
    query: str,
    video_id_filter: Optional[str] = None,
    k: Optional[int] = None,
) -> list[tuple[Document, float]]:
    """
    Top-k chunks for a query, as (Document, similarity) pairs.

    - video_id_filter "A"/"B" → only that video.
    - None → k/2 from EACH video, so comparison questions always see both
      sides even when one transcript is much longer than the other.
    """
    k = k or config.RETRIEVER_TOP_K
    if video_id_filter in ("A", "B"):
        return _search(query, k, video_id_filter)

    per_video = max(2, (k + 1) // 2)
    results = _search(query, per_video, "A") + _search(query, per_video, "B")
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    segs = [{"text": f"Sentence number {i} about hooks and pacing.", "start": i * 3.0} for i in range(40)]
    cs = chunk_transcript(" ".join(s["text"] for s in segs), "A", "Me", "Test", "XXXXXXXXXXX", "u", segs)
    for c in cs[:3]:
        print(c.chunk_index, c.start_seconds, c.start_char, c.text[:60])
