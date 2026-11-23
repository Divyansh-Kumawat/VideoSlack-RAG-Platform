# backend/models/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models — single source of truth for every request/response contract.
# The app compares two YouTube videos: Video A and Video B.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Helpers ───────────────────────────────────────────────────────────────────

_YT_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/|/v/)([A-Za-z0-9_-]{11})"
)


def extract_youtube_id(url: str) -> str:
    """
    Extracts the 11-character video ID from any common YouTube URL format:
      https://www.youtube.com/watch?v=ID · https://youtu.be/ID
      https://youtube.com/shorts/ID     · https://youtube.com/embed/ID
      https://youtube.com/live/ID
    Raises ValueError if no ID can be found.
    """
    match = _YT_ID_RE.search(url or "")
    if not match:
        raise ValueError(f"Could not find a YouTube video ID in: {url}")
    return match.group(1)


def format_timestamp(seconds: float | int | None) -> str:
    s = int(seconds or 0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── Enums ─────────────────────────────────────────────────────────────────────

class VideoID(str, Enum):
    A = "A"   # first video
    B = "B"   # second video


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Sent by the frontend when the user submits two YouTube video URLs."""
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
            raise ValueError(f"Must be a YouTube URL, got: {v}")
        extract_youtube_id(v)   # raises ValueError → 422 with a clear message
        return v

    @model_validator(mode="after")
    def different_videos(self) -> "IngestRequest":
        if extract_youtube_id(self.youtube_url_a) == extract_youtube_id(self.youtube_url_b):
            raise ValueError("Video A and Video B must be two different videos.")
        return self


class ChatTurn(BaseModel):
    """One message in the conversation history."""
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """
    Sent on every chat turn. History is the full prior conversation —
    the backend is stateless and rebuilds context from it each time.
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
    video_filter: Optional[VideoID] = Field(
        default=None,
        description=(
            "Restrict retrieval to one video. If omitted, the backend infers it "
            "from the question ('Video A …') and otherwise searches both."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional client-generated session UUID",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class VideoMetadata(BaseModel):
    """All metadata for one video. Returned by ingestion, shown in a VideoCard."""
    video_id:          VideoID
    platform:          str = "youtube"
    youtube_id:        str
    url:               str
    title:             str
    creator:           str
    follower_count:    Optional[int] = None
    views:             int
    likes:             int
    comments:          int
    engagement_rate:   float
    upload_date:       str
    duration_seconds:  int
    hashtags:          list[str] = []
    thumbnail_url:     Optional[str] = None
    description:       Optional[str] = None
    transcript:        str
    opening_text:      Optional[str] = None  # first ~15s of speech — the "hook"
    transcript_source: str = "captions"      # "captions" | "whisper"
    language:          Optional[str] = None
    chunk_count:       int = 0

    @property
    def duration_formatted(self) -> str:
        return format_timestamp(self.duration_seconds)


class SourceChunk(BaseModel):
    """A retrieved chunk returned as a citation alongside an LLM answer."""
    video_id:      VideoID
    chunk_index:   int
    text:          str
    start_char:    int = 0
    start_seconds: Optional[float] = None
    timestamp:     Optional[str]   = None     # "1:23"
    url:           Optional[str]   = None     # deep link to that moment
    title:         Optional[str]   = None
    creator:       Optional[str]   = None
    score:         Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Returned after both videos are ingested. Populates both VideoCards."""
    video_a:       VideoMetadata
    video_b:       VideoMetadata
    chunks_stored: int
    message:       str = "Ingestion complete"


class VideosResponse(BaseModel):
    """Currently ingested videos — lets the frontend restore state on reload."""
    video_a: Optional[VideoMetadata] = None
    video_b: Optional[VideoMetadata] = None
    ready:   bool = False


class ChatResponse(BaseModel):
    """Full answer + sources. Used by the non-streaming /chat endpoint."""
    answer:     str
    sources:    list[SourceChunk] = []
    session_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error envelope for all 4xx/5xx responses."""
    error:    str
    detail:   str
    video_id: Optional[VideoID] = None
