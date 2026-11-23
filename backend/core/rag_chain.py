# backend/core/rag_chain.py
# ─────────────────────────────────────────────────────────────────────────────
# RAG pipeline — retrieval + prompt assembly + streamed Mistral answer.
#
# Per chat turn:
#   1. Decide which video(s) to search (explicit filter or inferred from text)
#   2. Retrieve top-k transcript chunks from ChromaDB (balanced across A & B)
#   3. Build ONE prompt: instructions + both videos' stats + excerpts + history
#   4. Stream Mistral tokens; emit cited source chunks at the end
#
# Exactly one LLM call per question (no separate "condense question" call) —
# free Mistral tiers allow very few requests per minute.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Optional

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend import config
from backend.core import vector_store, video_store
from backend.core.engagement import compare_engagement
from backend.models.schemas import (
    ChatTurn,
    SourceChunk,
    VideoID,
    VideoMetadata,
    format_timestamp,
)

logger = logging.getLogger(__name__)

RATE_LIMIT_BACKOFF = [12, 30]   # seconds to wait before each retry on HTTP 429


class LLMError(RuntimeError):
    """An LLM failure with a message that is safe to show to the user."""


# ─────────────────────────────────────────────────────────────────────────────
# Which video is the question about?
# ─────────────────────────────────────────────────────────────────────────────

_A_PATTERNS = re.compile(r"\b(video\s*a|video\s*1|first\s+video|1st\s+video)\b", re.I)
_B_PATTERNS = re.compile(r"\b(video\s*b|video\s*2|second\s+video|2nd\s+video)\b", re.I)


def detect_video_filter(question: str) -> Optional[str]:
    """
    "A" if the question mentions only Video A, "B" if only Video B,
    otherwise None (search both — default for comparisons).
    """
    a = bool(_A_PATTERNS.search(question))
    b = bool(_B_PATTERNS.search(question))
    if a and not b:
        return "A"
    if b and not a:
        return "B"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

# Words that usually point back at the previous turn ("why did it work?", "what about them?").
_FOLLOW_UP_RE = re.compile(r"^(why|how come|and|but|so|what about)\b|\b(it|its|that|they|them|he|she)\b", re.I)


def build_retrieval_query(question: str, history: list[ChatTurn]) -> str:
    """
    Follow-ups like "why is that?" retrieve poorly on their own, so short
    or pronoun-heavy questions are combined with the previous user question.
    """
    prev = next((t.content for t in reversed(history) if t.role == "user"), None)
    if prev and (len(question.split()) <= 8 or _FOLLOW_UP_RE.search(question)):
        return f"{prev}\n{question}"
    return question


def to_source_chunks(results: list[tuple[Document, float]]) -> list[SourceChunk]:
    seen: set[tuple[str, int]] = set()
    sources: list[SourceChunk] = []
    for doc, score in results:
        m = doc.metadata or {}
        vid = str(m.get("video_id", "A"))
        idx = int(m.get("chunk_index", 0))
        if (vid, idx) in seen or vid not in ("A", "B"):
            continue
        seen.add((vid, idx))

        start_s = m.get("start_seconds")
        start_s = None if start_s is None or float(start_s) < 0 else float(start_s)
        yt_id = m.get("youtube_id")
        link = None
        if yt_id:
            link = f"https://www.youtube.com/watch?v={yt_id}"
            if start_s is not None:
                link += f"&t={int(start_s)}s"

        sources.append(SourceChunk(
            video_id=VideoID(vid),
            chunk_index=idx,
            text=doc.page_content,
            start_char=int(m.get("start_char", 0)),
            start_seconds=start_s,
            timestamp=format_timestamp(start_s) if start_s is not None else None,
            url=link,
            title=m.get("title"),
            creator=m.get("creator"),
            score=score,
        ))
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VideoSlack, an expert YouTube content strategist. You help creators understand why videos perform the way they do by analysing two YouTube videos: Video A and Video B.

You are given (1) each video's real statistics and (2) transcript excerpts retrieved for the current question.

Rules:
- Ground every claim in the statistics or excerpts below. Never invent numbers, quotes or facts. If the information isn't there, say so plainly.
- Always say which video (A or B) a point refers to.
- When you use a transcript excerpt, cite it inline with its tag exactly as written, e.g. [A 0:42] or [B 1:05].
- Engagement rate = (likes + comments) / views × 100. Use the precomputed values; don't recompute differently.
- For comparisons, structure the answer as: **Video A**, **Video B**, **Verdict** (or a short table when comparing numbers).
- Be concise, specific and actionable — creators want things they can do in their next video.
- Format with Markdown (short paragraphs, bullet lists, bold for key numbers)."""


def _fmt_int(n: Optional[int]) -> str:
    return "unknown" if n is None else f"{n:,}"


def describe_video(v: VideoMetadata) -> str:
    lines = [
        f"## Video {v.video_id.value}: \"{v.title}\"",
        f"- Creator: {v.creator} ({_fmt_int(v.follower_count)} subscribers)",
        f"- URL: {v.url}",
        f"- Uploaded: {v.upload_date or 'unknown'} · Duration: {format_timestamp(v.duration_seconds)}"
        f" ({'Short' if v.duration_seconds and v.duration_seconds <= 180 else 'long-form'})",
        f"- Views: {_fmt_int(v.views)} · Likes: {_fmt_int(v.likes)} · Comments: {_fmt_int(v.comments)}",
        f"- Engagement rate: {v.engagement_rate}%",
    ]
    if v.follower_count:
        lines.append(f"- Views per subscriber: {v.views / v.follower_count:.2f}")
    if v.hashtags:
        lines.append(f"- Tags/hashtags: {', '.join(v.hashtags[:15])}")
    if v.description:
        desc = re.sub(r"\s+", " ", v.description)[:300]
        lines.append(f"- Description (start): {desc}")
    if v.opening_text:
        lines.append(f"- Opening / hook (first ~15s of transcript): \"{v.opening_text}\"")
    lines.append(
        f"- Transcript language: {v.language or 'unknown'} "
        f"(from {'auto-transcribed audio' if v.transcript_source == 'whisper' else 'YouTube captions'})"
    )
    return "\n".join(lines)


def build_stats_block(videos: dict[str, Optional[VideoMetadata]]) -> str:
    parts = [describe_video(v) for v in videos.values() if v]
    a, b = videos.get("A"), videos.get("B")
    if a and b:
        cmp = compare_engagement(a.model_dump(), b.model_dump())
        parts.append(f"## Engagement comparison\n{cmp['summary']}")
    return "\n\n".join(parts) if parts else "(no video statistics available)"


def build_context_block(sources: list[SourceChunk]) -> str:
    if not sources:
        return "(no transcript excerpts matched this question)"
    ordered = sorted(sources, key=lambda s: (s.video_id.value, s.start_seconds or 0, s.chunk_index))
    out = []
    for s in ordered:
        tag = f"[{s.video_id.value} {s.timestamp}]" if s.timestamp else f"[{s.video_id.value} #{s.chunk_index}]"
        out.append(f"{tag} {s.text}")
    return "\n\n".join(out)


def build_messages(
    question: str,
    history: list[ChatTurn],
    sources: list[SourceChunk],
    videos: dict[str, Optional[VideoMetadata]],
) -> list[BaseMessage]:
    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"# Video statistics\n{build_stats_block(videos)}\n\n"
        f"# Transcript excerpts retrieved for this question\n{build_context_block(sources)}"
    )
    messages: list[BaseMessage] = [SystemMessage(content=system)]
    for turn in history[-(config.MEMORY_MAX_TURNS * 2):]:
        cls = HumanMessage if turn.role == "user" else AIMessage
        if turn.content.strip():
            messages.append(cls(content=turn.content))
    messages.append(HumanMessage(content=question))
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

def get_llm():
    if not config.MISTRAL_API_KEY:
        raise LLMError(
            "MISTRAL_API_KEY is not set in backend/.env. Get a key at https://console.mistral.ai"
        )
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(  # type: ignore[call-arg]  (pydantic aliases confuse Pylance)
        api_key=config.MISTRAL_API_KEY,
        model=config.MISTRAL_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_retries=2,
        timeout=120,
    )


def _explain_http_error(e: httpx.HTTPStatusError) -> str:
    code = e.response.status_code
    if code == 401:
        return "Mistral rejected the API key (401). Check MISTRAL_API_KEY in backend/.env."
    if code == 403:
        return (
            f"Your Mistral plan doesn't allow model '{config.MISTRAL_MODEL}' (403). "
            "Set MISTRAL_MODEL in backend/.env to a model your tier allows, e.g. ministral-14b-latest."
        )
    if code == 429:
        return (
            "Mistral rate limit reached — your plan allows only a few requests per minute "
            f"for '{config.MISTRAL_MODEL}'. Wait about a minute and ask again."
        )
    return f"Mistral API error {code}."


async def _stream_llm(messages: list[BaseMessage]) -> AsyncGenerator[dict[str, Any], None]:
    """Streams tokens, retrying on 429 only if nothing has been streamed yet."""
    llm = get_llm()
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        started = False
        try:
            async for chunk in llm.astream(messages):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    started = True
                    yield {"type": "token", "content": text}
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning("Mistral HTTP %s (attempt %d): %s", status, attempt + 1, e)
            if status == 429 and not started and attempt < config.LLM_MAX_RETRIES:
                delay = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                yield {
                    "type": "status",
                    "content": f"Mistral rate limit hit — retrying in {delay}s…",
                }
                await asyncio.sleep(delay)
                continue
            raise LLMError(_explain_http_error(e)) from e
        except httpx.TransportError as e:
            raise LLMError(f"Could not reach Mistral: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def stream_answer(
    question: str,
    history: list[ChatTurn],
    video_id_filter: Optional[str] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Yields event dicts:
      {"type": "status",  "content": "..."}      — optional progress notes
      {"type": "token",   "content": "..."}      — streamed answer text
      {"type": "sources", "content": [...]}      — cited chunks
    Raises LLMError for user-facing LLM problems.
    """
    query = build_retrieval_query(question, history)
    results = await asyncio.to_thread(vector_store.retrieve, query, video_id_filter)
    sources = to_source_chunks(results)
    videos = video_store.get_videos()
    messages = build_messages(question, history, sources, videos)

    logger.info(
        "RAG: filter=%s, %d sources, %d history turns, model=%s",
        video_id_filter, len(sources), len(history), config.MISTRAL_MODEL,
    )

    async for event in _stream_llm(messages):
        yield event

    yield {"type": "sources", "content": [s.model_dump(mode="json") for s in sources]}


async def answer(
    question: str,
    history: list[ChatTurn],
    video_id_filter: Optional[str] = None,
) -> dict[str, Any]:
    """Non-streaming variant: returns {"answer": str, "sources": [...]}."""
    parts: list[str] = []
    sources: list = []
    async for event in stream_answer(question, history, video_id_filter):
        if event["type"] == "token":
            parts.append(event["content"])
        elif event["type"] == "sources":
            sources = event["content"]
    return {"answer": "".join(parts), "sources": sources}


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
