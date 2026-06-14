# backend/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application entrypoint.
#
# Startup sequence:
#   1. Load .env
#   2. Pre-load BGE embedding model (so first request isn't slow)
#   3. Initialise ChromaDB client
#   4. Mount routers
#
# Run: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = logging.DEBUG if os.getenv("ENVIRONMENT", "development") == "development" else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", 8000))
ENVIRONMENT  = os.getenv("ENVIRONMENT", "development")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — runs on startup and shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-warm expensive singletons so the first request isn't slow.
    - BGE embedding model (~90MB, takes 2-4s to load)
    - ChromaDB client (opens the persistent directory)

    Shutdown: nothing to clean up (ChromaDB flushes automatically).
    """
    logger.info("=" * 60)
    logger.info("VideoIQ API starting up (env=%s)", ENVIRONMENT)
    logger.info("=" * 60)

    # Pre-load embedding model
    from backend.core.embedder import _get_local_model, EMBEDDING_PROVIDER
    if EMBEDDING_PROVIDER == "local":
        logger.info("Pre-loading BGE embedding model...")
        _get_local_model()
        logger.info("Embedding model ready.")
    else:
        logger.info("Using OpenAI embeddings (no pre-load needed).")

    # Initialise ChromaDB (opens the persistent directory)
    from backend.core.vector_store import get_vectorstore, get_chunk_count
    get_vectorstore()
    total = get_chunk_count()
    logger.info("ChromaDB ready. Chunks in store: %d", total)

    logger.info("VideoIQ API ready. Listening on %s:%d", API_HOST, API_PORT)
    logger.info("Docs: http://%s:%d/docs", API_HOST, API_PORT)
    logger.info("=" * 60)

    yield   # Application runs here

    # Shutdown
    logger.info("VideoIQ API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VideoIQ API",
    description=(
        "Full-stack RAG chatbot for social media video analysis. "
        "Ingests YouTube + Instagram Reels, embeds transcripts, "
        "and answers creator questions via streaming LangChain RAG."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],    # Needed for SSE streams in some browsers
)


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception and returns a consistent JSON error
    instead of a raw 500 Internal Server Error.
    """
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error":  "internal_server_error",
            "detail": str(exc),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

from backend.routers.ingest import router as ingest_router
from backend.routers.chat   import router as chat_router

app.include_router(ingest_router)
app.include_router(chat_router)


# ─────────────────────────────────────────────────────────────────────────────
# Root & health endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "service": "VideoIQ API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    """
    Full health check — confirms embedding model, ChromaDB, and env vars are ready.
    Used by Docker healthcheck and deployment platforms (Render, Railway).
    """
    from backend.core.vector_store import get_chunk_count
    from backend.core.embedder import EMBEDDING_PROVIDER, EMBEDDING_MODEL

    issues: list[str] = []

    if not os.getenv("MISTRAL_API_KEY"):
        issues.append("MISTRAL_API_KEY not set")
    if not os.getenv("YOUTUBE_API_KEY"):
        issues.append("YOUTUBE_API_KEY not set")

    total_chunks = get_chunk_count()

    return {
        "status":             "ok" if not issues else "degraded",
        "issues":             issues,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model":    EMBEDDING_MODEL,
        "chunks_in_store":    total_chunks,
        "environment":        ENVIRONMENT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dev runner — python backend/main.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=ENVIRONMENT == "development",
        log_level="info",
    )