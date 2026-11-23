from __future__ import annotations

import pytest
from conftest import make_metadata, make_transcript
from pydantic import ValidationError

from backend.core import rag_chain, vector_store
from backend.core.engagement import compare_engagement, compute_engagement_rate
from backend.core.metadata import extract_hashtags, parse_iso8601_duration
from backend.core.transcript import clean_text
from backend.models.schemas import (
    ChatTurn,
    IngestRequest,
    VideoID,
    VideoMetadata,
    extract_youtube_id,
    format_timestamp,
)
from backend.routers.ingest import opening_text

# ── engagement ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "likes,comments,views,expected",
    [(1000, 200, 50_000, 2.4), (5000, 800, 100_000, 5.8), (0, 0, 10_000, 0.0), (100, 10, 0, 0.0)],
)
def test_engagement_rate(likes, comments, views, expected):
    assert compute_engagement_rate(likes, comments, views) == expected


def test_compare_engagement_winner():
    cmp = compare_engagement({"likes": 50, "comments": 0, "views": 1000},
                             {"likes": 10, "comments": 0, "views": 1000})
    assert cmp["winner"] == "A" and cmp["difference"] == 4.0


# ── parsing helpers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?si=abc",
    "https://youtube.com/shorts/dQw4w9WgXcQ?si=sbcWb0Ts0LhNym-3",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
])
def test_extract_youtube_id(url):
    assert extract_youtube_id(url) == "dQw4w9WgXcQ"


def test_extract_youtube_id_rejects_garbage():
    with pytest.raises(ValueError):
        extract_youtube_id("https://youtube.com/@channel")


def test_ingest_request_validation():
    with pytest.raises(ValidationError):
        IngestRequest(youtube_url_a="https://vimeo.com/1", youtube_url_b="https://youtu.be/dQw4w9WgXcQ")
    with pytest.raises(ValidationError, match="different"):
        IngestRequest(youtube_url_a="https://youtu.be/dQw4w9WgXcQ",
                      youtube_url_b="https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_duration_and_timestamp():
    assert parse_iso8601_duration("PT4M13S") == 253
    assert parse_iso8601_duration("PT1H2M30S") == 3750
    assert parse_iso8601_duration("P0D") == 0
    assert format_timestamp(75) == "1:15"
    assert format_timestamp(3750) == "1:02:30"


def test_hashtags_and_cleaning():
    assert extract_hashtags("Go #aws #cloud #aws") == ["aws", "cloud"]
    assert clean_text("[Music] hello  (applause)  world >> ok") == "hello world ok"


def test_opening_text_uses_first_15_seconds():
    t = make_transcript("hooks")
    hook = opening_text(t)
    assert "part 0" in hook and "part 3" in hook and "part 4" not in hook


# ── chunking + retrieval ──────────────────────────────────────────────────────

def test_chunking_keeps_timestamps_and_overlap():
    t = make_transcript("pacing", n=40)
    chunks = vector_store.chunk_transcript(
        t["text"], "A", "c", "t", "dQw4w9WgXcQ", "u", segments=t["segments"],
        chunk_size=200, chunk_overlap=60,
    )
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].start_seconds == 0.0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    starts = [c.start_seconds for c in chunks]
    assert starts == sorted(starts)
    # overlap: next chunk begins with the last segment of the previous one
    last_seg_of_first = chunks[0].text.split(". ")[-1]
    assert chunks[1].text.startswith(last_seg_of_first.rstrip("."))
    # start_char points into the full transcript
    assert t["text"][chunks[2].start_char:].startswith(chunks[2].text[:30])


def test_chunking_without_segments():
    text = "word " * 400
    chunks = vector_store.chunk_transcript(text, "B", "c", "t", "x" * 11, "u", segments=None)
    assert chunks and all(c.start_seconds is None for c in chunks)


def _ingest_pair():
    for vid, topic in (("A", "cloud certification exam tips"), ("B", "cooking pasta recipe")):
        t = make_transcript(topic)
        vector_store.upsert_chunks(vector_store.chunk_transcript(
            t["text"], vid, "c", topic, "dQw4w9WgXcQ", "u", segments=t["segments"],
        ))


def test_retrieve_is_balanced_and_filterable():
    _ingest_pair()
    both = vector_store.retrieve("cloud certification", None, k=6)
    assert {d.metadata["video_id"] for d, _ in both} == {"A", "B"}
    only_b = vector_store.retrieve("cloud certification", "B", k=4)
    assert only_b and all(d.metadata["video_id"] == "B" for d, _ in only_b)
    assert all(0.0 <= s <= 1.0 for _, s in both)


def test_clear_video():
    _ingest_pair()
    assert vector_store.get_chunk_count("A") > 0
    vector_store.clear_video("A")
    assert vector_store.get_chunk_count("A") == 0
    assert vector_store.get_chunk_count("B") > 0


# ── RAG helpers ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q,expected", [
    ("What is the engagement rate of Video A?", "A"),
    ("Tell me about the second video", "B"),
    ("How does video 2 open?", "B"),
    ("Compare Video A and Video B", None),
    ("Which one did better?", None),
    ("I watched a youtube video about this", None),
])
def test_detect_video_filter(q, expected):
    assert rag_chain.detect_video_filter(q) == expected


def test_retrieval_query_includes_previous_question_for_follow_ups():
    history = [ChatTurn(role="user", content="What is the hook of Video A?"),
               ChatTurn(role="assistant", content="It opens with a question.")]
    assert "hook of Video A" in rag_chain.build_retrieval_query("Why does that work?", history)
    standalone = "Summarise the main teaching points covered across both of these videos in detail"
    assert rag_chain.build_retrieval_query(standalone, history) == standalone


def test_prompt_contains_stats_sources_and_history():
    _ingest_pair()
    sources = rag_chain.to_source_chunks(vector_store.retrieve("cloud", None, k=4))
    assert sources[0].url and "&t=" in sources[0].url and sources[0].timestamp
    meta = make_metadata("dQw4w9WgXcQ", "Cloud Tips", 1000, 90, 10)
    video = VideoMetadata(video_id=VideoID.A, engagement_rate=10.0, transcript="x",
                          opening_text="Hey there", **meta)
    history = [ChatTurn(role="user", content="hi"), ChatTurn(role="assistant", content="hello")]
    msgs = rag_chain.build_messages("Which is better?", history, sources, {"A": video, "B": None})
    system = msgs[0].content
    assert "Engagement rate: 10.0%" in system and "Views: 1,000" in system
    assert "Hey there" in system
    assert "[A 0:00]" in system or "[B 0:00]" in system
    assert [m.type for m in msgs[1:]] == ["human", "ai", "human"]
