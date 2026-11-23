# backend/scripts/smoke_test.py
# ─────────────────────────────────────────────────────────────────────────────
# LIVE end-to-end check against a running backend — real YouTube, real Mistral.
# (The offline unit tests live in backend/tests and need none of this.)
#
#   1. start the API:  uvicorn backend.main:app --port 8000
#   2. run:            python backend/scripts/smoke_test.py [URL_A URL_B]
#
# Note: free Mistral tiers allow ~2 requests/minute, so this script asks only
# two questions and leaves a pause between them.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sys
import time

import httpx

API = os.getenv("API_URL", "http://localhost:8000")
URL_A = sys.argv[1] if len(sys.argv) > 2 else "https://youtube.com/shorts/XiqalmRZUdg"
URL_B = sys.argv[2] if len(sys.argv) > 2 else "https://youtube.com/shorts/kiMpfSFsOfg"

QUESTIONS = [
    "What is the engagement rate of each video, and which one wins?",
    "Compare the hooks in the first few seconds of both videos.",
]


def step(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def main() -> int:
    client = httpx.Client(base_url=API, timeout=httpx.Timeout(600, connect=5))

    step("Health")
    try:
        health = client.get("/health").json()
    except httpx.ConnectError:
        print(f"Backend not reachable at {API}. Start it first.")
        return 1
    print(json.dumps(health, indent=2))

    step("Ingest")
    t0 = time.time()
    r = client.post("/ingest", json={"youtube_url_a": URL_A, "youtube_url_b": URL_B})
    if r.status_code != 200:
        print(f"FAILED ({r.status_code}): {r.text}")
        return 1
    body = r.json()
    for key in ("video_a", "video_b"):
        v = body[key]
        print(
            f"Video {v['video_id']}: {v['title'][:60]!r} by {v['creator']} — "
            f"{v['views']:,} views, ER {v['engagement_rate']}%, "
            f"{v['chunk_count']} chunks ({v['transcript_source']})"
        )
    print(f"took {time.time() - t0:.1f}s")

    for i, q in enumerate(QUESTIONS):
        if i:
            time.sleep(31)   # stay under the free-tier rate limit
        step(f"Q{i + 1}: {q}")
        answer, sources = "", []
        with client.stream("POST", "/chat/stream", json={"question": q, "history": []}) as resp:
            if resp.status_code != 200:
                print(f"FAILED ({resp.status_code}): {resp.read().decode()}")
                return 1
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["type"] == "token":
                    answer += ev["content"]
                elif ev["type"] == "sources":
                    sources = ev["content"]
                elif ev["type"] in ("status", "error"):
                    print(f"[{ev['type']}] {ev['content']}")
        print(answer.strip() or "(no answer)")
        print("sources:", ", ".join(f"{s['video_id']}@{s['timestamp']}" for s in sources) or "none")

    print("\nSmoke test finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
