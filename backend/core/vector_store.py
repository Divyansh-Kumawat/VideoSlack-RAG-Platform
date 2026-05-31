# backend/core/vector_store.py
# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB vector store — init, upsert, retrieval.
#
# Every chunk is tagged with video_id ("A" or "B") so the retriever
# can be filtered per video or across both.
# Persists to CHROMA_PERSIST_DIR (default: ./chroma_db).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import logging
import hashlib
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv

from backend.core.embedder import VideoIQEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION_NAME", "videoiq_chunks")
CHUNK_SIZE              = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP           = int(os.getenv("CHUNK_OVERLAP", 50))
RETRIEVER_TOP_K         = int(os.getenv("RETRIEVER_TOP_K", 6))


# ─────────────────────────────────────────────────────────────────────────────
# Data class for a single chunk (internal representation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    text:        str
    video_id:    str          # "A" or "B"
    chunk_index: int
    start_char:  int
    creator:     str
    title:       str
    platform:    str
    url:         str


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB client — singleton
# ─────────────────────────────────────────────────────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None
_vectorstore:   Chroma | None = None


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialised at %s", CHROMA_PERSIST_DIR)
    return _chroma_client


def get_vectorstore() -> Chroma:
    """
    Returns the LangChain Chroma vectorstore (singleton).
    Creates the collection if it doesn't exist.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = VideoIQEmbeddings()
        _vectorstore = Chroma(
            client=_get_client(),
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        logger.info("LangChain Chroma vectorstore ready (collection=%s)", CHROMA_COLLECTION_NAME)
    return _vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_transcript(
    transcript: str,
    video_id: str,
    creator: str,
    title: str,
    platform: str,
    url: str,
) -> list[TranscriptChunk]:
    """
    Splits a transcript into overlapping chunks using
    RecursiveCharacterTextSplitter.

    Chunk size of 500 chars with 50 overlap is the sweet spot for
    social media content: short punchy sentences stay together,
    hooks aren't split across chunks.

    Args:
        transcript: Full raw transcript text
        video_id:   "A" or "B"
        creator:    Creator name (stored as metadata on every chunk)
        title:      Video title
        platform:   "youtube" or "instagram"
        url:        Original video URL

    Returns:
        List of TranscriptChunk objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    raw_chunks = splitter.split_text(transcript)
    chunks = []
    char_offset = 0

    for i, text in enumerate(raw_chunks):
        # Find where this chunk starts in the original transcript
        start = transcript.find(text, char_offset)
        if start == -1:
            start = char_offset
        char_offset = start + len(text)

        chunks.append(
            TranscriptChunk(
                text=text,
                video_id=video_id,
                chunk_index=i,
                start_char=start,
                creator=creator,
                title=title,
                platform=platform,
                url=url,
            )
        )

    logger.info(
        "Chunked Video %s transcript: %d chunks from %d chars",
        video_id, len(chunks), len(transcript),
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk_id(video_id: str, chunk_index: int, text: str) -> str:
    """
    Deterministic chunk ID — re-ingesting the same video won't create duplicates.
    Hash of (video_id + chunk_index + first 100 chars of text).
    """
    raw = f"{video_id}:{chunk_index}:{text[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()


def upsert_chunks(chunks: list[TranscriptChunk]) -> int:
    """
    Embeds and upserts chunks into ChromaDB.

    Uses deterministic IDs so re-running ingestion on the same video
    overwrites existing chunks (upsert semantics) rather than duplicating.

    Args:
        chunks: List of TranscriptChunk objects from chunk_transcript()

    Returns:
        Number of chunks upserted.
    """
    if not chunks:
        logger.warning("upsert_chunks called with empty list — nothing to store.")
        return 0

    vs = get_vectorstore()

    texts     = [c.text for c in chunks]
    metadatas = [
        {
            "video_id":    c.video_id,
            "chunk_index": c.chunk_index,
            "start_char":  c.start_char,
            "creator":     c.creator,
            "title":       c.title,
            "platform":    c.platform,
            "url":         c.url,
        }
        for c in chunks
    ]
    ids = [_make_chunk_id(c.video_id, c.chunk_index, c.text) for c in chunks]

    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    logger.info("Upserted %d chunks into ChromaDB", len(chunks))
    return len(chunks)


def clear_video(video_id: str) -> None:
    """
    Deletes all chunks for a given video_id from the collection.
    Call before re-ingesting a video to get a clean slate.
    """
    client = _get_client()
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    results = collection.get(where={"video_id": video_id})
    ids_to_delete = results.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for video_id=%s", len(ids_to_delete), video_id)
    else:
        logger.info("No existing chunks for video_id=%s", video_id)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever(video_id_filter: Optional[str] = None) -> VectorStoreRetriever:
    """
    Returns a LangChain VectorStoreRetriever, optionally filtered by video_id.

    Args:
        video_id_filter:
            - None      → retrieves from both videos (used for comparison questions)
            - "A"       → only YouTube chunks
            - "B"       → only Instagram chunks

    Returns:
        LangChain retriever ready to plug into ConversationalRetrievalChain.
    """
    vs = get_vectorstore()

    search_kwargs: dict = {"k": RETRIEVER_TOP_K}

    if video_id_filter in ("A", "B"):
        search_kwargs["filter"] = {"video_id": video_id_filter}
        logger.debug("Retriever filtering to video_id=%s, k=%d", video_id_filter, RETRIEVER_TOP_K)
    else:
        logger.debug("Retriever across all videos, k=%d", RETRIEVER_TOP_K)

    return vs.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def get_chunk_count(video_id: Optional[str] = None) -> int:
    """
    Returns total number of chunks stored, optionally filtered by video_id.
    Useful for health checks and ingestion confirmation.
    """
    client = _get_client()
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
        if video_id:
            results = collection.get(where={"video_id": video_id})
            return len(results.get("ids", []))
        return collection.count()
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.vector_store
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    SAMPLE_TRANSCRIPT_A = """
    Hey guys welcome back to my channel. Today I'm going to show you three things
    that went viral last week and why they worked. The first one is all about the hook.
    You have five seconds to grab attention or people swipe. The second thing is pacing.
    Fast cuts keep people watching. The third is a strong call to action at the end.
    If you do all three your engagement rate will skyrocket. Like and subscribe for more.
    """

    SAMPLE_TRANSCRIPT_B = """
    Good morning everyone. In today's reel I want to talk about content consistency.
    Posting every day is less important than posting good content. Quality beats quantity.
    Use trending audio and relevant hashtags to boost your organic reach.
    The algorithm rewards saves and shares more than likes. So make content worth saving.
    """

    print("\n── vector_store standalone test ──")

    # 1. Chunk
    chunks_a = chunk_transcript(
        transcript=SAMPLE_TRANSCRIPT_A,
        video_id="A",
        creator="TestCreator",
        title="Test YouTube Video",
        platform="youtube",
        url="https://youtube.com/watch?v=test",
    )
    chunks_b = chunk_transcript(
        transcript=SAMPLE_TRANSCRIPT_B,
        video_id="B",
        creator="@testgram",
        title="Test Instagram Reel",
        platform="instagram",
        url="https://instagram.com/reel/test/",
    )
    print(f"Chunks A: {len(chunks_a)}, Chunks B: {len(chunks_b)}")

    # 2. Upsert
    total = upsert_chunks(chunks_a + chunks_b)
    print(f"Upserted: {total} total chunks")
    print(f"DB count: {get_chunk_count()} total, A={get_chunk_count('A')}, B={get_chunk_count('B')}")

    # 3. Retrieval — all videos
    retriever = get_retriever()
    docs = retriever.invoke("What makes a good hook?")
    print(f"\nRetrieval (all) for 'What makes a good hook?' → {len(docs)} docs")
    for doc in docs:
        print(f"  video_id={doc.metadata['video_id']} chunk={doc.metadata['chunk_index']}: {doc.page_content[:80]}")

    # 4. Retrieval — filtered to B only
    retriever_b = get_retriever(video_id_filter="B")
    docs_b = retriever_b.invoke("How to grow on social media?")
    print(f"\nRetrieval (B only) → {len(docs_b)} docs")
    for doc in docs_b:
        print(f"  video_id={doc.metadata['video_id']} chunk={doc.metadata['chunk_index']}: {doc.page_content[:80]}")

    print("\nAll vector_store tests passed.")