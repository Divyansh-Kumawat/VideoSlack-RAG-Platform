# VideoSlack

A RAG (Retrieval Augmented Generation) system that ingests YouTube video transcripts and enables intelligent, context-aware conversations with source citation. Built with FastAPI, LangChain, ChromaDB, and Next.js.

## 🎯 Features

- **Video Ingestion Pipeline**: Automatically extract transcripts from YouTube videos, compute engagement metrics, and generate embeddings
- **Local Embeddings**: Uses BGE embeddings via `sentence-transformers` for privacy-first embedding generation
- **Persistent Vector Store**: ChromaDB for efficient semantic search across video transcripts
- **Streaming Chat**: Real-time token-by-token responses via Server-Sent Events (SSE)
- **Source Citation**: Automatic retrieval and citation of source chunks from ingested videos
- **Conversational Memory**: Maintains chat history for contextual multi-turn conversations
- **LLM Integration**: Powered by Mistral AI with OpenAI fallback support
- **CORS-Enabled API**: Ready for frontend integration

## 📦 Tech Stack

### Backend
- **Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (BGE)
- **LLM Provider**: Mistral AI
- **Server**: Uvicorn

### Frontend
- **Framework**: Next.js 16.2.6
- **Language**: TypeScript
- **UI**: React 19 + Tailwind CSS 4

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- pip and npm/yarn

### Backend Setup

1. **Install Python dependencies**:
```bash
cd backend
python -m venv .venv
# On Windows
.venv\Scripts\Activate.ps1
# On macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Configure environment variables**:
Create a `.env` file in the `backend` directory:
```env
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Mistral AI
MISTRAL_API_KEY=your_mistral_api_key

# Optional: OpenAI fallback
OPENAI_API_KEY=your_openai_api_key
```

3. **Run the backend**:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Docs: `http://localhost:8000/docs` (Swagger UI)

### Frontend Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Run development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📚 API Endpoints

### Ingest Video
**POST** `/ingest`

Ingests a YouTube video transcript, extracts metadata, and stores embeddings in the vector database.

Request:
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=..."
}
```

Response:
```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Video Title",
  "duration_seconds": 600,
  "engagement_rate": 0.85,
  "chunk_count": 42,
  "status": "success"
}
```

### Chat
**POST** `/chat`

Streams a RAG response with source citations as Server-Sent Events.

Request:
```json
{
  "query": "What is the main topic of the video?",
  "chat_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ],
  "k": 5,
  "video_ids": null
}
```

Response (Server-Sent Events):
```
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " main"}
data: {"type": "sources", "content": [{"video_id": "...", "text": "..."}]}
data: {"type": "done"}
```

Event Types:
- `token`: Streamed response token
- `sources`: Retrieved source chunks with metadata
- `error`: Error message
- `done`: End of stream

## 📁 Project Structure

```
videoSlack/
├── backend/
│   ├── main.py                 # FastAPI entrypoint
│   ├── requirements.txt         # Python dependencies
│   ├── test.py                 # Tests
│   ├── core/
│   │   ├── embedder.py         # Embedding model management
│   │   ├── engagement.py       # Video engagement metrics
│   │   ├── metadata.py         # Metadata extraction
│   │   ├── rag_chain.py        # RAG chain & streaming logic
│   │   ├── transcript.py       # Transcript extraction
│   │   └── vector_store.py     # ChromaDB integration
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   └── routers/
│       ├── chat.py             # Chat endpoint
│       └── ingest.py           # Ingest endpoint
├── frontend/
│   ├── package.json            # Node.js dependencies
│   ├── next.config.ts          # Next.js configuration
│   ├── tsconfig.json           # TypeScript configuration
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page
│   │   └── globals.css         # Global styles
│   └── public/                 # Static assets
└── README.md                   # This file
```

## 🔄 Data Pipeline

1. **Transcript Extraction** → Extract text from YouTube video
2. **Metadata** → Extract title, duration, uploader info
3. **Engagement** → Compute engagement metrics from metadata
4. **Chunking** → Split transcript into semantic chunks
5. **Embedding** → Generate embeddings using BGE model
6. **Storage** → Store in ChromaDB with metadata

## 💬 Chat Flow

1. User sends query + optional chat history
2. RAG chain retrieves top-k relevant chunks from ChromaDB
3. Mistral AI generates response using retrieved context
4. Response is streamed token-by-token via SSE
5. Source chunks are sent as final event for citation

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` or `production` |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated CORS origins |
| `MISTRAL_API_KEY` | Required | Mistral AI API key |
| `OPENAI_API_KEY` | Optional | OpenAI API key for fallback |

## 🛠️ Development

### Running Tests
```bash
cd backend
python -m pytest test.py -v
```

### Linting (Frontend)
```bash
cd frontend
npm run lint
```

## 📝 Notes

- The backend uses a **stateless design** — chat history is managed client-side and sent with each request
- Embeddings are generated **locally** using BGE for privacy and cost-efficiency
- Vector store is **persistent** and stored locally in ChromaDB
- First request may be slower due to model loading; subsequent requests are optimized
- `tmp_audio/` directory is used for temporary audio processing during transcript extraction

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the project.

## 📄 License

MIT

## 📧 Support

For questions or issues, please open an issue on the repository.
