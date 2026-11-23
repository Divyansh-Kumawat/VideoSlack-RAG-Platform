# backend/routers/ingest.py
# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest        — ingest two YouTube videos (A and B)
# GET  /ingest/status — chunk counts
# GET  /videos        — currently ingested videos (restores the UI on reload)
# DELETE /videos      — wipe everything
#
# Pipeline per video: transcript → metadata → engagement → chunk → embed → store
# Both videos are fetched in parallel, and the store is only modified once BOTH
# succeed — a failure on Video B never leaves you with half-replaced data.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException

from backend.core import vector_store, video_store
from backend.core.engagement import compute_engagement_rate_from_metadata
from backend.core.metadata import get_metadata
from backend.core.transcript import TranscriptResult, get_transcript
from backend.models.schemas import (
    ErrorResponse,
    IngestRequest,
    IngestResponse,
    VideoID,
    VideoMetadata,
    VideosResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])

HOOK_SECONDS = 15

# One ingestion at a time — two concurrent requests would interleave writes.
_ingest_lock = asyncio.Lock()


@dataclass
class _Fetched:
    video_id: str
    url: str
    meta: dict
    transcript: TranscriptResult


def opening_text(t: TranscriptResult, seconds: float = HOOK_SECONDS, max_chars: int = 400) -> str:
    """The first ~15 seconds of speech — the video's hook."""
    segs = t.get("segments") or []
    if segs:
        early = [s["text"] for s in segs if s["start"] < seconds] or [segs[0]["text"]]
        text = " ".join(early)
    else:
        text = t["text"][:250]
    return text[:max_chars].strip()


def _fetch(video_id: str, url: str) -> _Fetched:
    """Network-bound half of the pipeline (no writes)."""
    logger.info("[%s] Fetching transcript + metadata: %s", video_id, url)
    transcript = get_transcript(url)
    meta = get_metadata(url)
    return _Fetched(video_id=video_id, url=url, meta=meta, transcript=transcript)


def _store(f: _Fetched) -> VideoMetadata:
    """Write half of the pipeline: replace this video's chunks."""
    m = f.meta
    vector_store.clear_video(f.video_id)
    chunks = vector_store.chunk_transcript(
        transcript=f.transcript["text"],
        video_id=f.video_id,
        creator=m["creator"],
        title=m["title"],
        youtube_id=m["youtube_id"],
        url=f.url,
        segments=f.transcript["segments"],
    )
    vector_store.upsert_chunks(chunks)

    return VideoMetadata(
        video_id=VideoID(f.video_id),
        youtube_id=m["youtube_id"],
        url=f.url,
        title=m["title"],
        creator=m["creator"],
        follower_count=m.get("follower_count"),
        views=m["views"],
        likes=m["likes"],
        comments=m["comments"],
        engagement_rate=compute_engagement_rate_from_metadata(m),
        upload_date=m["upload_date"],
        duration_seconds=m["duration_seconds"],
        hashtags=m.get("hashtags", []),
        thumbnail_url=m.get("thumbnail_url"),
        description=m.get("description"),
        transcript=f.transcript["text"],
        opening_text=opening_text(f.transcript),
        transcript_source=f.transcript["source"],
        language=f.transcript["language"],
        chunk_count=len(chunks),
    )


def _error(video_id: str, exc: BaseException) -> HTTPException:
    logger.error("Video %s failed: %s", video_id, exc, exc_info=exc)
    return HTTPException(
        status_code=422,
        detail=ErrorResponse(
            error="ingestion_failed",
            detail=f"Video {video_id}: {exc}",
            video_id=VideoID(video_id),
        ).model_dump(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Ingest two YouTube videos")
async def ingest_videos(request: IngestRequest) -> IngestResponse:
    if _ingest_lock.locked():
        raise HTTPException(
            status_code=409,
            detail=ErrorResponse(
                error="ingestion_in_progress",
                detail="Another ingestion is already running. Try again when it finishes.",
            ).model_dump(),
        )
    async with _ingest_lock:
        return await _ingest(request)


async def _ingest(request: IngestRequest) -> IngestResponse:
    logger.info("Ingest: A=%s  B=%s", request.youtube_url_a, request.youtube_url_b)

    results = await asyncio.gather(
        asyncio.to_thread(_fetch, "A", request.youtube_url_a),
        asyncio.to_thread(_fetch, "B", request.youtube_url_b),
        return_exceptions=True,
    )
    for vid, res in zip(("A", "B"), results):
        if isinstance(res, BaseException):
            raise _error(vid, res)

    try:
        video_a = await asyncio.to_thread(_store, results[0])
        video_b = await asyncio.to_thread(_store, results[1])
    except Exception as e:
        logger.error("Storing chunks failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(error="storage_failed", detail=str(e)).model_dump(),
        )

    video_store.save_videos([video_a, video_b])
    total = video_a.chunk_count + video_b.chunk_count
    logger.info("Ingestion complete: A=%d chunks, B=%d chunks", video_a.chunk_count, video_b.chunk_count)

    return IngestResponse(
        video_a=video_a,
        video_b=video_b,
        chunks_stored=total,
        message="Both videos ingested. You can now ask questions.",
    )


@router.get("/ingest/status", summary="Chunk counts per video")
async def ingestion_status() -> dict:
    a, b = vector_store.get_chunk_count("A"), vector_store.get_chunk_count("B")
    return {"chunks_video_a": a, "chunks_video_b": b, "total_chunks": a + b, "ready": a > 0 and b > 0}


@router.get("/videos", response_model=VideosResponse, summary="Currently ingested videos")
async def list_videos() -> VideosResponse:
    videos = video_store.get_videos()
    a, b = videos["A"], videos["B"]
    ready = bool(a and b and vector_store.get_chunk_count() > 0)
    return VideosResponse(video_a=a, video_b=b, ready=ready)


@router.delete("/videos", summary="Delete all ingested videos")
async def delete_videos() -> dict:
    await asyncio.to_thread(vector_store.clear_all)
    video_store.clear_videos()
    return {"status": "cleared"}
