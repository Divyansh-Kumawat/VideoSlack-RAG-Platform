# backend/models/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models — single source of truth for every request/response contract.
# Every router, core module, and frontend API call speaks these types.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, HttpUrl, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class VideoID(str, Enum):
    """Identifier tag attached to every chunk in the vector store."""
    A = "A"   # YouTube video
    B = "B"   # Instagram reel


class EmbeddingProvider(str, Enum):
    local = "local"
    openai = "openai"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Sent by the frontend when the user submits two video URLs.
    Kicks off the full pipeline: transcript → metadata → embed → store.
    """
    youtube_url: str = Field(
        ...,
        description="Full YouTube video URL (e.g. https://www.youtube.com/watch?v=xxx or https://youtu.be/xxx)",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    instagram_url: str = Field(
        ...,
        description="Full Instagram Reel URL (e.g. https://www.instagram.com/reel/xxx/)",
        examples=["https://www.instagram.com/reel/C-xxxxxxxx/"],
    )

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube(cls, v: str) -> str:
        v = v.strip()
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError("youtube_url must be a valid YouTube link")
        return v

    @field_validator("instagram_url")
    @classmethod
    def validate_instagram(cls, v: str) -> str:
        v = v.strip()
        if "instagram.com" not in v:
            raise ValueError("instagram_url must be a valid Instagram link")
        return v


class ChatRequest(BaseModel):
    """
    Sent on every chat turn. History is the full prior conversation
    so the backend can reconstruct LangChain memory from scratch
    (stateless backend design — no server-side session needed).
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The creator's question about the two videos",
        examples=["Why did Video A get more engagement than Video B?"],
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="All prior turns in this session, oldest first",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional client-generated session UUID for future caching",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class VideoMetadata(BaseModel):
    """
    All metadata extracted for one video.
    Returned by the ingestion pipeline and displayed in the frontend VideoCard.
    Also stored as chunk-level metadata in ChromaDB.
    """
    video_id: VideoID = Field(
        ...,
        description="A = YouTube, B = Instagram",
    )
    platform: str = Field(
        ...,
        description="'youtube' or 'instagram'",
        examples=["youtube", "instagram"],
    )
    url: str = Field(..., description="Original URL as submitted")
    title: str = Field(..., description="Video title")
    creator: str = Field(..., description="Channel / account name")
    follower_count: Optional[int] = Field(
        default=None,
        description="Subscriber count (YouTube) or follower count (Instagram)",
    )
    views: int = Field(..., ge=0, description="Total view count")
    likes: int = Field(..., ge=0, description="Total like count")
    comments: int = Field(..., ge=0, description="Total comment count")
    engagement_rate: float = Field(
        ...,
        description="(likes + comments) / views × 100, rounded to 2 dp",
        examples=[4.73],
    )
    upload_date: str = Field(
        ...,
        description="Upload date as ISO 8601 string (YYYY-MM-DD)",
        examples=["2024-03-15"],
    )
    duration_seconds: int = Field(
        ...,
        ge=0,
        description="Video duration in seconds",
    )
    hashtags: list[str] = Field(
        default_factory=list,
        description="List of hashtags found in title/description/caption",
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="URL of the video thumbnail image",
    )
    transcript: str = Field(
        ...,
        description="Full transcript text (raw, pre-chunking)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected transcript language code (e.g. 'en', 'hi')",
    )

    @property
    def duration_formatted(self) -> str:
        """Returns duration as MM:SS or HH:MM:SS string."""
        h, rem = divmod(self.duration_seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


class SourceChunk(BaseModel):
    """
    A single retrieved chunk returned alongside an LLM answer.
    Rendered as a citation badge in the frontend (e.g. 'Video A · chunk 3').
    """
    video_id: VideoID = Field(..., description="Which video this chunk came from")
    chunk_index: int = Field(..., ge=0, description="Position of chunk in the video transcript")
    text: str = Field(..., description="The actual chunk text shown as citation context")
    start_char: int = Field(..., ge=0, description="Character offset in the full transcript")
    score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score from vector retrieval (0–1)",
    )
    creator: Optional[str] = Field(
        default=None,
        description="Creator name — surfaced for 'who made Video B?' queries",
    )


class ChatTurn(BaseModel):
    """One question-answer pair in the conversation history."""
    role: str = Field(
        ...,
        description="'user' or 'assistant'",
        pattern="^(user|assistant)$",
    )
    content: str = Field(..., description="Message text")


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """
    Returned after both videos have been ingested successfully.
    Frontend uses this to populate the two VideoCard components.
    """
    video_a: VideoMetadata = Field(..., description="YouTube video metadata")
    video_b: VideoMetadata = Field(..., description="Instagram reel metadata")
    chunks_stored: int = Field(
        ...,
        description="Total number of chunks embedded and stored in ChromaDB",
    )
    message: str = Field(
        default="Ingestion complete",
        description="Human-readable status message",
    )


class ChatResponse(BaseModel):
    """
    Returned for non-streaming chat responses.
    For streaming, individual tokens are sent as SSE text/event-stream,
    with this model sent as the final 'done' event payload.
    """
    answer: str = Field(..., description="Full LLM-generated answer")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Retrieved chunks that grounded this answer, deduplicated",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Echo of the session_id from the request",
    )


class ErrorResponse(BaseModel):
    """Standard error envelope used across all 4xx / 5xx responses."""
    error: str = Field(..., description="Short error code, e.g. 'transcript_unavailable'")
    detail: str = Field(..., description="Human-readable explanation")
    video_id: Optional[VideoID] = Field(
        default=None,
        description="Set when the error is specific to one video",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Update forward references (needed because ChatRequest references ChatTurn
# which is defined below it in the file)
# ─────────────────────────────────────────────────────────────────────────────
ChatRequest.model_rebuild()