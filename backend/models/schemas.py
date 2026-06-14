# backend/models/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models — single source of truth for every request/response contract.
# Both Video A and Video B are YouTube videos.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class VideoID(str, Enum):
    A = "A"   # YouTube video 1
    B = "B"   # YouTube video 2


class EmbeddingProvider(str, Enum):
    local  = "local"
    openai = "openai"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Sent by the frontend when the user submits two YouTube video URLs.
    Both are YouTube — Video A and Video B.
    """
    youtube_url_a: str = Field(
        ...,
        description="First YouTube video URL (Video A)",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    youtube_url_b: str = Field(
        ...,
        description="Second YouTube video URL (Video B)",
        examples=["https://www.youtube.com/watch?v=9bZkp7q19f0"],
    )

    @field_validator("youtube_url_a", "youtube_url_b")
    @classmethod
    def validate_youtube(cls, v: str) -> str:
        v = v.strip()
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError(f"Must be a valid YouTube URL, got: {v}")
        return v


class ChatRequest(BaseModel):
    """
    Sent on every chat turn. History is the full prior conversation
    so the backend can reconstruct LangChain memory (stateless backend).
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
        description="Optional client-generated session UUID",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class VideoMetadata(BaseModel):
    """All metadata for one video. Returned by ingestion, shown in VideoCard."""
    video_id:         VideoID
    platform:         str = "youtube"
    url:              str
    title:            str
    creator:          str
    follower_count:   Optional[int]  = None
    views:            int
    likes:            int
    comments:         int
    engagement_rate:  float
    upload_date:      str
    duration_seconds: int
    hashtags:         list[str]      = []
    thumbnail_url:    Optional[str]  = None
    transcript:       str
    language:         Optional[str]  = None

    @property
    def duration_formatted(self) -> str:
        h, rem = divmod(self.duration_seconds, 3600)
        m, s   = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class SourceChunk(BaseModel):
    """A retrieved chunk returned as a citation alongside an LLM answer."""
    video_id:    VideoID
    chunk_index: int
    text:        str
    start_char:  int
    score:       Optional[float] = None
    creator:     Optional[str]   = None


class ChatTurn(BaseModel):
    """One question-answer pair in conversation history."""
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Returned after both videos are ingested. Populates both VideoCards."""
    video_a:       VideoMetadata
    video_b:       VideoMetadata
    chunks_stored: int
    message:       str = "Ingestion complete"


class ChatResponse(BaseModel):
    """Full answer + sources. Used by the sync /chat endpoint."""
    answer:     str
    sources:    list[SourceChunk] = []
    session_id: Optional[str]    = None


class ErrorResponse(BaseModel):
    """Standard error envelope for all 4xx/5xx responses."""
    error:    str
    detail:   str
    video_id: Optional[VideoID] = None


# Rebuild forward references
ChatRequest.model_rebuild()