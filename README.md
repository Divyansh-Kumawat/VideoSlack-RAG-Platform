# VideoSlack

Paste two YouTube videos and ask an AI analyst why one outperformed the other.

VideoSlack pulls each video's transcript and stats (views, likes, comments, subscribers), computes engagement rates, and indexes the transcripts in a local vector store. You then chat with a RAG assistant. Its answers are grounded in the real numbers and cite the exact transcript moment they rely on, e.g. `[A 0:42]`, which links to that second of the video.

Built with FastAPI, LangChain, ChromaDB, local BGE embeddings, Mistral, and Next.js.

## Feature

- **Two-video comparison.** Video A and Video B are ingested together. Shows side-by-side cards, a head-to-head metrics panel, and the opening "hook" of each video.
- **Transcripts that just work.** Uses YouTube captions first. When a video has none (common for Shorts), it downloads the audio and transcribes it locally with Whisper.
- **Metadata with or without a key.** Uses the YouTube Data API v3 when `YOUTUBE_API_KEY` is set, otherwise falls back to yt-dlp.
- **Timestamped citations.** Chunks are built from caption segments, so every source knows its start time. Answers link to the moment in the video.
- **Balanced retrieval.** Comparison questions retrieve from both videos equally. Questions about a single video ("What's the hook of Video B?") are scoped to it automatically, or explicitly from the UI.
- **Streaming chat.** Tokens stream over Server-Sent Events. The UI supports stop, retry, and per-pair chat history.
- **One LLM call per question.** Free Mistral tiers allow ~2 requests/minute, so there is no separate "rephrase the question" call. Rate-limit (429) responses are retried automatically, with progress shown in the UI.
- **Local embeddings.** `BAAI/bge-small-en-v1.5` runs on your machine; only the final prompt is sent to Mistral.

## Tech stack

| Layer | Tech |
|---|---|
| API | FastAPI, Uvicorn |
| RAG | LangChain core, ChromaDB (cosine), sentence-transformers (BGE) |
| LLM | Mistral (`ministral-14b-latest` by default) |
| YouTube | youtube-transcript-api, yt-dlp, YouTube Data API v3, OpenAI Whisper (local) |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, react-markdown |

## Quick start

### Prerequisites

- Python 3.10+
- Node.js 20+
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (only needed for videos without captions)
- A [Mistral API key](https://console.mistral.ai) (free tier works)
- Optional: a [YouTube Data API v3 key](https://console.cloud.google.com)

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then put your MISTRAL_API_KEY (and optional YOUTUBE_API_KEY) in it
cd ..
```

Run the API **from the repository root** (the code imports the `backend` package):

```bash
uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- The first start downloads the embedding model (~130 MB).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. If the API isn't on `http://localhost:8000`, copy `frontend/.env.example` to `frontend/.env.local` and set `NEXT_PUBLIC_API_URL`.

## How it works

### Ingestion: `POST /ingest`

```
for Video A and Video B, in parallel:
  transcript  → captions (youtube-transcript-api)  ─┐ fallback: yt-dlp audio + Whisper
  metadata    → YouTube Data API v3                ─┘ fallback: yt-dlp
then, only if BOTH succeeded:
  engagement  → (likes + comments) / views × 100
  chunk       → ~500-char chunks built from caption segments (keeps start time)
  embed+store → BGE embeddings → ChromaDB (tagged video_id A/B)
  save stats  → backend/data/videos.json
```

If either video fails, the previously ingested pair is left untouched.

### Chat: `POST /chat/stream`

1. Decide the scope: an explicit `video_filter`, or inferred from the question ("Video A", "the second video"…), or both.
2. Retrieve the top-k chunks (k/2 per video for comparisons). Short follow-ups ("why is that?") are expanded with the previous question.
3. Build one prompt: instructions, both videos' stats and hooks, timestamped excerpts, and recent chat history.
4. Stream Mistral's answer, then send the source chunks.

The backend is stateless for chat: the client sends the history with every request.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Ingest two YouTube videos |
| `GET` | `/videos` | Currently ingested videos (used to restore the UI) |
| `DELETE` | `/videos` | Remove all ingested data |
| `GET` | `/ingest/status` | Chunk counts per video |
| `POST` | `/chat/stream` | Streamed answer (SSE) |
| `POST` | `/chat` | Same answer as one JSON response |
| `GET` | `/health` | Config and readiness check |

### `POST /ingest`

```json
{
  "youtube_url_a": "https://youtube.com/shorts/XiqalmRZUdg",
  "youtube_url_b": "https://www.youtube.com/watch?v=kiMpfSFsOfg"
}
```

Returns `{ video_a, video_b, chunks_stored, message }`. Each video includes `title`, `creator`, `follower_count`, `views`, `likes`, `comments`, `engagement_rate`, `duration_seconds`, `upload_date`, `hashtags`, `thumbnail_url`, `opening_text` (the first ~15 s), `transcript`, `transcript_source` (`captions` | `whisper`) and `chunk_count`.

Errors return `422` with `{"detail": {"error": "ingestion_failed", "detail": "...", "video_id": "B"}}`. A second ingestion started while one is running returns `409`.

### `POST /chat/stream`

```json
{
  "question": "Why did Video B get more engagement?",
  "history": [
    { "role": "user", "content": "Which video has more views?" },
    { "role": "assistant", "content": "Video B, with 352K views…" }
  ],
  "video_filter": null
}
```

`video_filter` is `"A"`, `"B"` or `null` (infer from the question / search both).

The response is `text/event-stream`, one JSON object per `data:` line:

```
data: {"type": "status",  "content": "Mistral rate limit hit — retrying in 12s…"}   (optional)
data: {"type": "token",   "content": "Video B"}
data: {"type": "token",   "content": " opens with…"}
data: {"type": "sources", "content": [{"video_id": "B", "timestamp": "0:00", "url": "https://www.youtube.com/watch?v=…&t=0s", "text": "…", "score": 0.41, …}]}
data: {"type": "done"}
```

On failure an `{"type": "error", "content": "…"}` event is sent. `done` is always the last event.

## Configuration

All settings live in `backend/.env`; see [`backend/.env.example`](backend/.env.example) for the full list. The important ones:

| Variable | Default | Notes |
|---|---|---|
| `MISTRAL_API_KEY` | — | **Required** for chat |
| `MISTRAL_MODEL` | `ministral-14b-latest` | Must be allowed by your Mistral plan |
| `YOUTUBE_API_KEY` | — | Optional; yt-dlp is used without it |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Re-ingest after changing |
| `RETRIEVER_TOP_K` | `6` | Chunks per answer (split across videos for comparisons) |
| `WHISPER_ENABLED` / `WHISPER_MODEL` | `true` / `small` | Fallback for videos without captions |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Frontend origins |

## Development

```bash
# Backend unit + API tests: offline, fake embeddings & LLM, ~10 s (run from the repo root)
python -m pytest

# Live end-to-end check against a running API (real YouTube + Mistral)
python backend/scripts/smoke_test.py [URL_A URL_B]

# Frontend
cd frontend
npm run lint
npm run build
```

## Project structure

```
videoSlack/
├── backend/
│   ├── main.py              # FastAPI app, startup, health
│   ├── config.py            # all settings (.env loaded from backend/)
│   ├── core/
│   │   ├── transcript.py    # captions → Whisper fallback, timestamped segments
│   │   ├── metadata.py      # YouTube Data API → yt-dlp fallback
│   │   ├── engagement.py    # engagement-rate maths
│   │   ├── embedder.py      # local BGE embeddings
│   │   ├── vector_store.py  # chunking, ChromaDB upsert/retrieval
│   │   ├── video_store.py   # persisted video stats (data/videos.json)
│   │   └── rag_chain.py     # prompt building + streamed Mistral answer
│   ├── routers/             # /ingest, /videos, /chat
│   ├── models/schemas.py    # request/response models
│   ├── tests/               # pytest suite (offline)
│   └── scripts/smoke_test.py
├── frontend/
│   ├── app/                 # layout, page, global styles
│   ├── components/          # IngestForm, VideoCard, ComparisonPanel, ChatPanel, Markdown
│   └── lib/                 # API client (SSE parser), types, formatters
└── pytest.ini
```

## Troubleshooting

- **Chat says "rate limit reached".** Free Mistral tiers allow about 2 requests per minute per model. Wait a minute, or use a paid tier.
- **403 / "model not available in your subscription tier".** Set `MISTRAL_MODEL` to a model your plan allows (e.g. `ministral-14b-latest`).
- **Transcript errors on every video.** YouTube changes often; update the scrapers with `pip install -U youtube-transcript-api yt-dlp`.
- **"Whisper needs ffmpeg".** Install ffmpeg and make sure `ffmpeg -version` works in the same terminal.
- **Frontend shows "Can't reach the backend".** Start the API from the repo root, and check `NEXT_PUBLIC_API_URL` / `CORS_ORIGINS`.

## License

MIT
