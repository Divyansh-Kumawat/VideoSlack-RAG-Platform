from __future__ import annotations

import json

import httpx
import pytest
from conftest import make_metadata, make_transcript
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from backend.core import rag_chain
from backend.main import app
from backend.routers import ingest as ingest_router

URL_A = "https://youtube.com/shorts/XiqalmRZUdg"
URL_B = "https://www.youtube.com/watch?v=kiMpfSFsOfg"

# TestClient without a `with` block skips the lifespan (no real model load).
client = TestClient(app)


@pytest.fixture
def fake_sources(monkeypatch):
    """Stub transcript + metadata fetching for both videos."""
    data = {
        "XiqalmRZUdg": (make_transcript("aws data engineer certification"),
                        make_metadata("XiqalmRZUdg", "AWS Data Engineer", 54_000, 368, 13)),
        "kiMpfSFsOfg": (make_transcript("aws cloud practitioner basics"),
                        make_metadata("kiMpfSFsOfg", "Cloud Practitioner", 20_000, 500, 20)),
    }

    def fake_transcript(url):
        return data["XiqalmRZUdg" if "Xiqalm" in url else "kiMpfSFsOfg"][0]

    def fake_metadata(url):
        return data["XiqalmRZUdg" if "Xiqalm" in url else "kiMpfSFsOfg"][1]

    monkeypatch.setattr(ingest_router, "get_transcript", fake_transcript)
    monkeypatch.setattr(ingest_router, "get_metadata", fake_metadata)
    return data


class FakeLLM:
    def __init__(self, tokens=("Video A ", "wins [A 0:04]."), fail_times=0, status=429):
        self.tokens, self.fail_times, self.status, self.calls = tokens, fail_times, status, 0
        self.last_messages = None

    async def astream(self, messages):
        self.calls += 1
        self.last_messages = messages
        if self.calls <= self.fail_times:
            req = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
            raise httpx.HTTPStatusError("err", request=req, response=httpx.Response(self.status, request=req))
        for t in self.tokens:
            yield AIMessageChunk(content=t)


def _events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def _ingest():
    r = client.post("/ingest", json={"youtube_url_a": URL_A, "youtube_url_b": URL_B})
    assert r.status_code == 200, r.text
    return r.json()


# ── ingestion ─────────────────────────────────────────────────────────────────

def test_health():
    assert client.get("/health").json()["status"] in ("ok", "degraded")


def test_ingest_and_videos(fake_sources):
    body = _ingest()
    a, b = body["video_a"], body["video_b"]
    assert a["video_id"] == "A" and a["engagement_rate"] == 0.71
    assert b["engagement_rate"] == 2.6
    assert a["chunk_count"] > 0 and body["chunks_stored"] == a["chunk_count"] + b["chunk_count"]
    assert a["opening_text"].startswith("In part 0")

    v = client.get("/videos").json()
    assert v["ready"] and v["video_b"]["title"] == "Cloud Practitioner"
    status = client.get("/ingest/status").json()
    assert status["ready"] and status["chunks_video_a"] == a["chunk_count"]


def test_failed_video_b_keeps_previous_data(fake_sources, monkeypatch):
    _ingest()
    before = client.get("/ingest/status").json()

    def boom(url):
        if "kiMpfSFsOfg" in url:
            raise RuntimeError("no captions")
        return fake_sources["XiqalmRZUdg"][0]

    monkeypatch.setattr(ingest_router, "get_transcript", boom)
    r = client.post("/ingest", json={"youtube_url_a": URL_A, "youtube_url_b": URL_B})
    assert r.status_code == 422
    assert r.json()["detail"]["video_id"] == "B"
    assert "no captions" in r.json()["detail"]["detail"]
    assert client.get("/ingest/status").json() == before


def test_ingest_rejects_same_video():
    r = client.post("/ingest", json={"youtube_url_a": URL_A, "youtube_url_b": URL_A})
    assert r.status_code == 422


def test_delete_videos(fake_sources):
    _ingest()
    assert client.delete("/videos").status_code == 200
    v = client.get("/videos").json()
    assert v == {"video_a": None, "video_b": None, "ready": False}


# ── chat ──────────────────────────────────────────────────────────────────────

def test_chat_requires_ingestion():
    r = client.post("/chat/stream", json={"question": "hi"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "no_data_ingested"


def test_chat_stream_events(fake_sources, monkeypatch):
    _ingest()
    llm = FakeLLM()
    monkeypatch.setattr(rag_chain, "get_llm", lambda: llm)

    r = client.post("/chat/stream", json={"question": "Which video has better engagement?", "history": []})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    types = [e["type"] for e in events]
    assert types == ["token", "token", "sources", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Video A wins [A 0:04]."
    sources = events[2]["content"]
    assert {s["video_id"] for s in sources} == {"A", "B"}
    assert all(s["url"].startswith("https://www.youtube.com/watch?v=") for s in sources)
    # the prompt carries real stats for both videos
    system = llm.last_messages[0].content
    assert "Engagement rate: 0.71%" in system and "Engagement rate: 2.6%" in system


def test_chat_explicit_filter(fake_sources, monkeypatch):
    _ingest()
    monkeypatch.setattr(rag_chain, "get_llm", lambda: FakeLLM())
    r = client.post("/chat/stream", json={"question": "What is it about?", "video_filter": "B"})
    sources = next(e for e in _events(r) if e["type"] == "sources")["content"]
    assert sources and all(s["video_id"] == "B" for s in sources)


def test_chat_retries_on_rate_limit(fake_sources, monkeypatch):
    _ingest()
    llm = FakeLLM(fail_times=1)
    monkeypatch.setattr(rag_chain, "get_llm", lambda: llm)
    monkeypatch.setattr(rag_chain, "RATE_LIMIT_BACKOFF", [0])
    events = _events(client.post("/chat/stream", json={"question": "Compare them"}))
    assert [e["type"] for e in events][:2] == ["status", "token"]
    assert events[-1]["type"] == "done" and llm.calls == 2


def test_chat_reports_auth_error(fake_sources, monkeypatch):
    _ingest()
    monkeypatch.setattr(rag_chain, "get_llm", lambda: FakeLLM(fail_times=9, status=401))
    events = _events(client.post("/chat/stream", json={"question": "Compare them"}))
    assert [e["type"] for e in events] == ["error", "done"]
    assert "MISTRAL_API_KEY" in events[0]["content"]


def test_chat_sync(fake_sources, monkeypatch):
    _ingest()
    monkeypatch.setattr(rag_chain, "get_llm", lambda: FakeLLM())
    r = client.post("/chat", json={"question": "Compare the hooks", "session_id": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Video A wins [A 0:04]." and body["sources"] and body["session_id"] == "s1"
