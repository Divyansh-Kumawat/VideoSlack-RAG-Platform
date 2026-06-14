# backend/routers/chat.py
# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# Streams RAG responses as Server-Sent Events (SSE).
#
# SSE event types:
#   data: {"type": "token",   "content": "Hello"}   ← one per streamed token
#   data: {"type": "sources", "content": [...]}      ← retrieved chunks
#   data: {"type": "error",   "content": "..."}      ← on failure
#   data: {"type": "done"}                           ← terminal event
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from backend.core.rag_chain import build_rag_stream, run_rag_query
from backend.core.vector_store import get_chunk_count

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ─────────────────────────────────────────────────────────────────────────────
# Guard: reject chat requests if no data is ingested yet
# ─────────────────────────────────────────────────────────────────────────────

def _assert_data_ready() -> None:
    """
    Raises HTTPException if neither video has been ingested yet.
    Prevents the LLM from hallucinating answers against an empty vector store.
    """
    total = get_chunk_count()
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="no_data_ingested",
                detail=(
                    "No video data found in the vector store. "
                    "Please call POST /ingest with two video URLs first."
                ),
            ).model_dump(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Detect which video the question is about (simple heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_video_filter(question: str) -> str | None:
    """
    Heuristic filter: if the question mentions only Video A or only Video B,
    restrict retrieval to that video's chunks.

    Returns "A", "B", or None (both videos — default for comparison questions).

    This keeps retrieved context focused and avoids polluting Video A answers
    with Video B chunks and vice versa.
    """
    q = question.lower()

    mentions_a = any(kw in q for kw in ["video a", "youtube", "video 1"])
    mentions_b = any(kw in q for kw in ["video b", "instagram", "reel", "video 2"])

    if mentions_a and not mentions_b:
        return "A"
    if mentions_b and not mentions_a:
        return "B"
    return None   # comparison / cross-video question → retrieve from both


# ─────────────────────────────────────────────────────────────────────────────
# Streaming SSE wrapper with error handling
# ─────────────────────────────────────────────────────────────────────────────

async def _safe_stream(
    question: str,
    history,
    video_id_filter: str | None,
) -> AsyncGenerator[str, None]:
    """
    Wraps build_rag_stream with error handling.
    Any exception mid-stream is caught and emitted as an SSE error event
    so the frontend doesn't hang waiting for a done event that never comes.
    """
    try:
        async for chunk in build_rag_stream(
            question=question,
            history=history,
            video_id_filter=video_id_filter,
        ):
            yield chunk
    except Exception as e:
        logger.error("RAG stream error: %s", e, exc_info=True)
        error_payload = json.dumps({
            "type":    "error",
            "content": f"An error occurred while generating the answer: {str(e)}",
        })
        yield f"data: {error_payload}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/stream",
    summary="Stream a RAG answer (SSE)",
    description=(
        "Streams the LLM answer token-by-token as Server-Sent Events. "
        "Final events include retrieved source chunks for citation display. "
        "Use this endpoint for the production chat UI."
    ),
    response_class=StreamingResponse,
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Main streaming chat endpoint.

    Returns a StreamingResponse with media_type='text/event-stream'.
    The frontend reads this with the EventSource API or fetch + ReadableStream.

    Event sequence:
      1. N × token events (one per streamed LLM token)
      2. 1 × sources event (retrieved chunks as citation data)
      3. 1 × done event (signals stream end to frontend)
    """
    _assert_data_ready()

    video_filter = _detect_video_filter(request.question)
    logger.info(
        "Chat stream: question='%s...', history_turns=%d, filter=%s",
        request.question[:60], len(request.history), video_filter,
    )

    return StreamingResponse(
        _safe_stream(
            question=request.question,
            history=request.history,
            video_id_filter=video_filter,
        ),
        media_type="text/event-stream",
        headers={
            # Prevent proxies/nginx from buffering the SSE stream
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":      "keep-alive",
        },
    )


@router.post(
    "",
    response_model=ChatResponse,
    summary="Non-streaming RAG answer (for testing)",
    description=(
        "Returns the full answer + sources in one JSON response. "
        "Use for curl testing and CI. Production should use /chat/stream."
    ),
)
async def chat_sync(request: ChatRequest) -> ChatResponse:
    """
    Synchronous (non-streaming) chat endpoint.

    Useful for:
    - curl testing during development
    - CI/integration tests
    - Clients that don't support SSE

    Production chat UI should use /chat/stream for streaming.
    """
    _assert_data_ready()

    video_filter = _detect_video_filter(request.question)
    logger.info(
        "Chat sync: question='%s...', filter=%s",
        request.question[:60], video_filter,
    )

    try:
        result = run_rag_query(
            question=request.question,
            history=request.history,
            video_id_filter=video_filter,
        )
    except Exception as e:
        logger.error("RAG query error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="rag_error",
                detail=str(e),
            ).model_dump(),
        )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )


@router.get(
    "/health",
    summary="Chat health check",
)
async def chat_health() -> dict:
    """Quick check that the chat endpoint is reachable and data is ready."""
    chunks_a = get_chunk_count("A")
    chunks_b = get_chunk_count("B")
    return {
        "status":         "ok",
        "chunks_video_a": chunks_a,
        "chunks_video_b": chunks_b,
        "ready":          chunks_a > 0 and chunks_b > 0,
    }