# backend/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration. Every module reads settings from here instead of
# calling os.getenv() directly, so defaults live in one place.
#
# .env is always loaded from the backend/ directory, and relative paths
# (ChromaDB dir, temp audio dir, data dir) are resolved against backend/ —
# so the server behaves the same no matter which directory you start it from.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'").strip()


def _int(name: str, default: int) -> int:
    raw = _str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _path(name: str, default: str) -> Path:
    p = Path(_str(name, default) or default)
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


# ── Server ────────────────────────────────────────────────────────────────────
ENVIRONMENT  = _str("ENVIRONMENT", "development") or "development"
API_HOST     = _str("API_HOST", "0.0.0.0") or "0.0.0.0"
API_PORT     = _int("API_PORT", 8000)
CORS_ORIGINS = [
    o.strip()
    for o in _str("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

# ── LLM (Mistral) ─────────────────────────────────────────────────────────────
MISTRAL_API_KEY    = _str("MISTRAL_API_KEY")
MISTRAL_MODEL      = _str("MISTRAL_MODEL", "ministral-14b-latest") or "ministral-14b-latest"
LLM_TEMPERATURE    = float(_str("LLM_TEMPERATURE", "0.3") or 0.3)
LLM_MAX_RETRIES    = _int("LLM_MAX_RETRIES", 2)      # retries on 429 rate-limit
MEMORY_MAX_TURNS   = _int("MEMORY_MAX_TURNS", 10)    # Q/A pairs of history sent to the LLM

# ── YouTube ───────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = _str("YOUTUBE_API_KEY")            # optional — yt-dlp fallback if unset

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5"

# ── Vector DB / RAG ───────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR     = _path("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = _str("CHROMA_COLLECTION_NAME", "videoiq_chunks") or "videoiq_chunks"
CHUNK_SIZE             = _int("CHUNK_SIZE", 500)
CHUNK_OVERLAP          = _int("CHUNK_OVERLAP", 50)
RETRIEVER_TOP_K        = _int("RETRIEVER_TOP_K", 6)

# ── Persistent app data (ingested video metadata) ─────────────────────────────
DATA_DIR = _path("DATA_DIR", "./data")

# ── Whisper fallback (videos without captions) ────────────────────────────────
WHISPER_ENABLED  = _str("WHISPER_ENABLED", "true").lower() in ("1", "true", "yes")
WHISPER_MODEL    = _str("WHISPER_MODEL", "small") or "small"
WHISPER_TEMP_DIR = _path("WHISPER_TEMP_DIR", "./tmp_audio")
