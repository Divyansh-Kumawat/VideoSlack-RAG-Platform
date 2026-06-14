

# test.py
# ─────────────────────────────────────────────────────────────────────────────
# VideoIQ Backend — End-to-End Test Script (Two YouTube Videos)
#
# Paste your two YouTube URLs at the top and run:
#   python test.py
#
# Tests everything directly against Python modules — no server needed.
# All results print to console.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# ✏️  PASTE YOUR TWO YOUTUBE URLs HERE
# ─────────────────────────────────────────────────────────────────────────────
YOUTUBE_URL_A = "https://youtube.com/shorts/XiqalmRZUdg?si=sbcWb0Ts0LhNym-3"   # ← Video A
YOUTUBE_URL_B = "https://youtube.com/shorts/kiMpfSFsOfg?si=QVQpeD-cCaBk7XoK"   # ← Video B

# ─────────────────────────────────────────────────────────────────────────────
# Test questions for RAG chain
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "What is the engagement rate of Video A?",
    "What is the engagement rate of Video B?",
    "Why did one video get more engagement than the other?",
    "Compare the hooks in the first 5 seconds of both videos.",
    "Who is the creator of Video B and what is their subscriber count?",
    "Suggest improvements for Video B based on what worked in Video A.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Suppress noisy library logs
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SEP  = "─" * 60

def header(title: str):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")

def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def result(label: str, status: str, detail: str = ""):
    print(f"  {status}  {label}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")

def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight check
# ─────────────────────────────────────────────────────────────────────────────

def preflight_check():
    issues = []
    if "XXXXXXXXXXX" in YOUTUBE_URL_A:
        issues.append("YOUTUBE_URL_A is still a placeholder — replace it at the top of test.py")
    if "XXXXXXXXXXX" in YOUTUBE_URL_B:
        issues.append("YOUTUBE_URL_B is still a placeholder — replace it at the top of test.py")
    if YOUTUBE_URL_A == YOUTUBE_URL_B:
        issues.append("Both URLs are identical — use two different YouTube videos")
    if not os.getenv("YOUTUBE_API_KEY"):
        issues.append("YOUTUBE_API_KEY is not set in .env")
    if not os.getenv("MISTRAL_API_KEY"):
        issues.append("MISTRAL_API_KEY is not set in .env")

    if issues:
        print("\n  ⚠️  Fix these before running:")
        for issue in issues:
            print(f"     • {issue}")
        print()
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Transcript Extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_transcripts() -> tuple[str, str]:
    section("1 / 5 · Transcript Extraction (youtube-transcript-api)")

    from backend.core.transcript import get_youtube_transcript

    transcript_a = transcript_b = ""

    for label, url, var in [("Video A", YOUTUBE_URL_A, "a"), ("Video B", YOUTUBE_URL_B, "b")]:
        t0 = time.time()
        try:
            res  = get_youtube_transcript(url)
            text = res["text"]
            result(
                f"{label} transcript",
                PASS,
                f"lang={res['language']}  chars={len(text)}  time={elapsed(t0)}\n"
                f"preview: {text[:150]}...",
            )
            if var == "a":
                transcript_a = text
            else:
                transcript_b = text
        except Exception as e:
            result(f"{label} transcript", FAIL, str(e))
            raise

    return transcript_a, transcript_b


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Metadata Fetching
# ─────────────────────────────────────────────────────────────────────────────

def test_metadata() -> tuple[dict, dict]:
    section("2 / 5 · Metadata Fetching (YouTube Data API v3)")

    from backend.core.metadata import get_youtube_metadata

    meta_a = meta_b = {}

    for label, url, var in [("Video A", YOUTUBE_URL_A, "a"), ("Video B", YOUTUBE_URL_B, "b")]:
        t0 = time.time()
        try:
            meta = get_youtube_metadata(url)
            result(
                f"{label} metadata",
                PASS,
                f"title    : {meta['title']}\n"
                f"creator  : {meta['creator']}\n"
                f"subs     : {meta.get('follower_count', 'N/A'):,}\n"
                f"views    : {meta['views']:,}\n"
                f"likes    : {meta['likes']:,}\n"
                f"comments : {meta['comments']:,}\n"
                f"duration : {meta['duration_seconds']}s\n"
                f"hashtags : {meta.get('hashtags', [])[:5]}\n"
                f"uploaded : {meta['upload_date']}  time={elapsed(t0)}",
            )
            if var == "a":
                meta_a = meta
            else:
                meta_b = meta
        except Exception as e:
            result(f"{label} metadata", FAIL, str(e))
            raise

    return meta_a, meta_b


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Engagement Rate
# ─────────────────────────────────────────────────────────────────────────────

def test_engagement(meta_a: dict, meta_b: dict):
    section("3 / 5 · Engagement Rate Computation")

    from backend.core.engagement import compute_engagement_rate_from_metadata, compare_engagement

    rate_a     = compute_engagement_rate_from_metadata(meta_a)
    rate_b     = compute_engagement_rate_from_metadata(meta_b)
    comparison = compare_engagement(meta_a, meta_b)

    result(
        "Video A engagement rate",
        PASS if rate_a >= 0 else FAIL,
        f"{rate_a}%  "
        f"({meta_a['likes']:,} likes + {meta_a['comments']:,} comments) "
        f"/ {meta_a['views']:,} views",
    )
    result(
        "Video B engagement rate",
        PASS if rate_b >= 0 else FAIL,
        f"{rate_b}%  "
        f"({meta_b['likes']:,} likes + {meta_b['comments']:,} comments) "
        f"/ {meta_b['views']:,} views",
    )
    result("Comparison", PASS, comparison["summary"])


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Vector Store
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_store(
    transcript_a: str, meta_a: dict,
    transcript_b: str, meta_b: dict,
):
    section("4 / 5 · Chunking + Embedding + ChromaDB")

    from backend.core.vector_store import (
        chunk_transcript, upsert_chunks, clear_video,
        get_chunk_count, get_retriever,
    )

    # Clear stale data
    clear_video("A")
    clear_video("B")

    # Chunk
    t0 = time.time()
    chunks_a = chunk_transcript(transcript_a, "A", meta_a["creator"], meta_a["title"], "youtube", YOUTUBE_URL_A)
    chunks_b = chunk_transcript(transcript_b, "B", meta_b["creator"], meta_b["title"], "youtube", YOUTUBE_URL_B)
    result(
        "Chunking",
        PASS,
        f"Video A → {len(chunks_a)} chunks  |  Video B → {len(chunks_b)} chunks  |  time={elapsed(t0)}",
    )

    # Embed + upsert
    t0 = time.time()
    try:
        total = upsert_chunks(chunks_a + chunks_b)
        result("Embed + upsert", PASS, f"{total} chunks stored in ChromaDB  time={elapsed(t0)}")
    except Exception as e:
        result("Embed + upsert", FAIL, str(e))
        raise

    # Verify counts
    ca, cb = get_chunk_count("A"), get_chunk_count("B")
    result(
        "DB verification",
        PASS if ca > 0 and cb > 0 else FAIL,
        f"Video A = {ca} chunks  |  Video B = {cb} chunks  |  total = {get_chunk_count()}",
    )

    # Retrieval — both videos
    t0 = time.time()
    docs = get_retriever().invoke("What is the main topic of this video?")
    result(
        "Retrieval (both videos)",
        PASS if docs else FAIL,
        f"{len(docs)} docs returned  time={elapsed(t0)}\n" +
        "\n".join(
            f"Video {d.metadata['video_id']} · chunk {d.metadata['chunk_index']}: "
            f"{d.page_content[:80]}..."
            for d in docs[:4]
        ),
    )

    # Retrieval — A only
    docs_a = get_retriever(video_id_filter="A").invoke("hook opening")
    result(
        "Retrieval (filter: A only)",
        PASS if all(d.metadata["video_id"] == "A" for d in docs_a) else FAIL,
        f"{len(docs_a)} docs, all from A: {all(d.metadata['video_id'] == 'A' for d in docs_a)}",
    )

    # Retrieval — B only
    docs_b = get_retriever(video_id_filter="B").invoke("engagement content")
    result(
        "Retrieval (filter: B only)",
        PASS if all(d.metadata["video_id"] == "B" for d in docs_b) else FAIL,
        f"{len(docs_b)} docs, all from B: {all(d.metadata['video_id'] == 'B' for d in docs_b)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — RAG Chain
# ─────────────────────────────────────────────────────────────────────────────

def test_rag_chain():
    section("5 / 5 · RAG Chain — Questions & Answers")

    from backend.core.rag_chain import run_rag_query
    from backend.models.schemas import ChatTurn

    history = []

    for i, question in enumerate(TEST_QUESTIONS):
        t0 = time.time()
        try:
            res     = run_rag_query(question=question, history=history)
            answer  = res["answer"]
            sources = res["sources"]

            history.append(ChatTurn(role="user",      content=question))
            history.append(ChatTurn(role="assistant", content=answer))

            source_tags = ", ".join(
                f"Video {s['video_id']} chunk {s['chunk_index']}"
                for s in sources[:3]
            ) or "none"

            result(
                f"Q{i+1}: {question}",
                PASS,
                f"time={elapsed(t0)}  sources=[{source_tags}]\n"
                f"answer: {answer}",
            )
        except Exception as e:
            result(f"Q{i+1}: {question}", FAIL, str(e))

        print()   # blank line between questions for readability

    result(
        "Multi-turn memory",
        PASS if len(history) == len(TEST_QUESTIONS) * 2 else FAIL,
        f"{len(TEST_QUESTIONS)} turns accumulated correctly in history",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    header("VideoIQ — Backend End-to-End Test (Two YouTube Videos)")

    print(f"\n  Video A : {YOUTUBE_URL_A}")
    print(f"  Video B : {YOUTUBE_URL_B}\n")

    preflight_check()

    overall = time.time()

    try:
        transcript_a, transcript_b = test_transcripts()
    except Exception:
        print("\n  ⛔ Transcript extraction failed — cannot continue.")
        sys.exit(1)

    try:
        meta_a, meta_b = test_metadata()
    except Exception:
        print("\n  ⛔ Metadata fetch failed — cannot continue.")
        sys.exit(1)

    test_engagement(meta_a, meta_b)

    try:
        test_vector_store(transcript_a, meta_a, transcript_b, meta_b)
    except Exception:
        print("\n  ⛔ Vector store failed — cannot test RAG chain.")
        sys.exit(1)

    test_rag_chain()

    header(f"All tests complete — total time: {elapsed(overall)}")
    print()


if __name__ == "__main__":
    main()