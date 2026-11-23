# backend/routers/chat.py
# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/stream — streamed RAG answer as Server-Sent Events
# POST /chat        — same answer as one JSON response (curl / tests)
#
# SSE event types (one JSON object per `data:` line):
#   {"type": "status",  "content": "..."}   ← optional progress (e.g. rate-limit retry)
#   {"type": "token",   "content": "Hello"} ← one per streamed token
#   {"type": "sources", "content": [...]}   ← retrieved chunks for citation
#   {"type": "error",   "content": "..."}   ← on failure
#   {"type": "done"}                        ← always the last event
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.core import vector_store
from backend.core.rag_chain import LLMError, answer, detect_video_filter, sse, stream_answer
from backend.models.schemas import ChatRequest, ChatResponse, ChatTurn, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _assert_data_ready() -> None:
    if vector_store.get_chunk_count() == 0:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="no_data_ingested",
                detail="No videos have been ingested yet. Submit two YouTube URLs first.",
            ).model_dump(),
        )


def _resolve_filter(request: ChatRequest) -> Optional[str]:
    if request.video_filter is not None:
        return request.video_filter.value
    return detect_video_filter(request.question)


async def _event_stream(
    question: str,
    history: list[ChatTurn],
    video_filter: Optional[str],
) -> AsyncGenerator[str, None]:
    """Wraps the RAG stream so the client ALWAYS gets a terminal `done` event."""
    try:
        async for event in stream_answer(question, history, video_filter):
            yield sse(event)
    except LLMError as e:
        yield sse({"type": "error", "content": str(e)})
    except Exception as e:
        logger.error("RAG stream error: %s", e, exc_info=True)
        yield sse({"type": "error", "content": f"Something went wrong while answering: {e}"})
    yield sse({"type": "done"})


@router.post(
    "/stream",
    summary="Stream a RAG answer (SSE)",
    response_class=StreamingResponse,
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    _assert_data_ready()
    video_filter = _resolve_filter(request)
    logger.info(
        "Chat stream: q='%s', history=%d, filter=%s",
        request.question[:80], len(request.history), video_filter,
    )
    return StreamingResponse(
        _event_stream(request.question, request.history, video_filter),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # stop nginx/proxies buffering the stream
            "Connection": "keep-alive",
        },
    )


@router.post("", response_model=ChatResponse, summary="Non-streaming RAG answer")
async def chat_sync(request: ChatRequest) -> ChatResponse:
    _assert_data_ready()
    video_filter = _resolve_filter(request)
    try:
        result = await answer(request.question, request.history, video_filter)
    except LLMError as e:
        raise HTTPException(
            status_code=502,
            detail=ErrorResponse(error="llm_error", detail=str(e)).model_dump(),
        )
    return ChatResponse(answer=result["answer"], sources=result["sources"], session_id=request.session_id)
