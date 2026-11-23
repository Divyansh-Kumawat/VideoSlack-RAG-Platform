# backend/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application entrypoint.
#
# Startup: load config → pre-load the embedding model → open ChromaDB → mount routers.
#
# Run from the repository root:
#   uvicorn backend.main:app --reload --reload-dir backend --port 8000
# or:
#   python -m backend.main
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config

logging.basicConfig(
    level=logging.DEBUG if config.ENVIRONMENT == "development" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Third-party libraries are very chatty at DEBUG.
for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
              "filelock", "multipart", "watchfiles", "asyncio", "googleapiclient"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
# chromadb 0.5.x logs a harmless posthog version-mismatch error on every call.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

logger = logging.getLogger("backend")

APP_NAME = "VideoSlack API"
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.core import vector_store
    from backend.core.embedder import get_model

    logger.info("%s starting (env=%s, model=%s)", APP_NAME, config.ENVIRONMENT, config.MISTRAL_MODEL)
    logger.info("Loading embedding model %s (first run downloads it)…", config.EMBEDDING_MODEL)
    await asyncio.to_thread(get_model)
    vector_store.get_vectorstore()
    logger.info("ChromaDB ready at %s: %d chunks stored.",
                config.CHROMA_PERSIST_DIR, vector_store.get_chunk_count())
    if not config.MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY is not set — chat will fail until it is.")
    logger.info("Ready. Docs at http://localhost:%d/docs", config.API_PORT)
    yield
    logger.info("%s shutting down.", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    description=(
        "RAG chatbot that ingests two YouTube videos (transcripts + stats) and "
        "answers creator questions about them with streamed, cited answers."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": {"error": "internal_server_error", "detail": str(exc)}},
    )


from backend.routers.chat import router as chat_router      # noqa: E402
from backend.routers.ingest import router as ingest_router  # noqa: E402

app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/", tags=["health"])
async def root():
    return {"service": APP_NAME, "version": APP_VERSION, "status": "running", "docs": "/docs"}


@app.get("/health", tags=["health"])
async def health():
    from backend.core import vector_store

    issues: list[str] = []
    if not config.MISTRAL_API_KEY:
        issues.append("MISTRAL_API_KEY not set — chat is unavailable")
    notes: list[str] = []
    if not config.YOUTUBE_API_KEY:
        notes.append("YOUTUBE_API_KEY not set — using yt-dlp for metadata")

    return {
        "status":          "ok" if not issues else "degraded",
        "issues":          issues,
        "notes":           notes,
        "llm_model":       config.MISTRAL_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunks_in_store": vector_store.get_chunk_count(),
        "environment":     config.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.ENVIRONMENT == "development",
        reload_dirs=["backend"],
    )
