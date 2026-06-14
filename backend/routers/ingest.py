# backend/routers/ingest.py
# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest
# Both Video A and Video B are YouTube videos.
# Pipeline: transcript → metadata → engagement → chunk → embed → store
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    IngestRequest,
    IngestResponse,
    VideoMetadata,
    VideoID,
    ErrorResponse,
)
from backend.core.transcript   import get_transcript
from backend.core.metadata     import get_metadata
from backend.core.engagement   import compute_engagement_rate_from_metadata
from backend.core.vector_store import (
    chunk_transcript,
    upsert_chunks,
    clear_video,
    get_chunk_count,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline for a single video
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_pipeline(url: str, video_id: str) -> VideoMetadata:
    """
    Full ingestion pipeline for one YouTube video.
    Platform is always 'youtube' — no Instagram logic needed.
    """
    logger.info("[%s] Starting pipeline: %s", video_id, url)

    # Step 1 — Transcript
    logger.info("[%s] Step 1/5: Extracting transcript...", video_id)
    transcript_result = get_transcript(url=url, platform="youtube")
    transcript_text   = transcript_result["text"]
    language          = transcript_result.get("language", "unknown")
    logger.info("[%s] Transcript: %d chars, lang=%s", video_id, len(transcript_text), language)

    # Step 2 — Metadata
    logger.info("[%s] Step 2/5: Fetching metadata...", video_id)
    meta = get_metadata(url=url, platform="youtube")

    # Step 3 — Engagement rate
    logger.info("[%s] Step 3/5: Computing engagement rate...", video_id)
    engagement_rate = compute_engagement_rate_from_metadata(meta)

    # Step 4 — Clear stale chunks (idempotent re-ingestion)
    logger.info("[%s] Step 4/5: Clearing old chunks...", video_id)
    clear_video(video_id)

    # Step 5 — Chunk + embed + store
    logger.info("[%s] Step 5/5: Chunking and embedding...", video_id)
    chunks = chunk_transcript(
        transcript=transcript_text,
        video_id=video_id,
        creator=meta["creator"],
        title=meta["title"],
        platform="youtube",
        url=url,
    )
    upsert_chunks(chunks)
    logger.info("[%s] Done: %d chunks stored", video_id, len(chunks))

    return VideoMetadata(
        video_id=VideoID(video_id),
        platform="youtube",
        url=url,
        title=meta["title"],
        creator=meta["creator"],
        follower_count=meta.get("follower_count"),
        views=meta["views"],
        likes=meta["likes"],
        comments=meta["comments"],
        engagement_rate=engagement_rate,
        upload_date=meta["upload_date"],
        duration_seconds=meta["duration_seconds"],
        hashtags=meta.get("hashtags", []),
        thumbnail_url=meta.get("thumbnail_url"),
        transcript=transcript_text,
        language=language,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=IngestResponse, summary="Ingest two YouTube videos")
async def ingest_videos(request: IngestRequest) -> IngestResponse:
    """
    Runs the full pipeline for both YouTube URLs sequentially.
    Returns full metadata for both videos ready for the frontend VideoCards.
    """
    logger.info("Ingest: A=%s  B=%s", request.youtube_url_a, request.youtube_url_b)

    # Video A
    try:
        video_a = _run_single_pipeline(url=request.youtube_url_a, video_id="A")
    except Exception as e:
        logger.error("Video A failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(error="ingestion_failed", detail=str(e), video_id=VideoID.A).model_dump(),
        )

    # Video B
    try:
        video_b = _run_single_pipeline(url=request.youtube_url_b, video_id="B")
    except Exception as e:
        logger.error("Video B failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(error="ingestion_failed", detail=str(e), video_id=VideoID.B).model_dump(),
        )

    total_chunks = get_chunk_count()
    logger.info("Ingestion complete. Total chunks: %d", total_chunks)

    return IngestResponse(
        video_a=video_a,
        video_b=video_b,
        chunks_stored=total_chunks,
        message="Both YouTube videos ingested successfully. You can now ask questions.",
    )


@router.get("/status", summary="Check ingestion status")
async def ingestion_status() -> dict:
    return {
        "chunks_video_a": get_chunk_count("A"),
        "chunks_video_b": get_chunk_count("B"),
        "total_chunks":   get_chunk_count(),
        "ready":          get_chunk_count("A") > 0 and get_chunk_count("B") > 0,
    }