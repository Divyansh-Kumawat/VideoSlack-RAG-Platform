# backend/core/rag_chain.py
# ─────────────────────────────────────────────────────────────────────────────
# RAG chain — ConversationalRetrievalChain + streaming + source citation.
#
# Architecture:
#   1. Rebuild ConversationBufferMemory from client-sent history (stateless backend)
#   2. Retrieve top-k chunks from ChromaDB (filtered or cross-video)
#   3. Stream Mistral tokens via SSE
#   4. Collect source chunks and emit them as a final SSE event
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import json
import logging
from typing import AsyncGenerator, Any, Sequence

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

from backend.core.vector_store import get_retriever
from backend.models.schemas import ChatTurn, SourceChunk, VideoID

load_dotenv()

logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 6))
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", 10))

# ─────────────────────────────────────────────────────────────────────────────
# System prompt injected before every answer
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VideoIQ, an expert social media analyst assistant.
You have access to transcripts and metadata from two videos:
- Video A: a YouTube video
- Video B: an Instagram Reel

Use ONLY the retrieved context to answer. Never hallucinate statistics.
When citing engagement rates, hooks, or creator details — state which video
(A or B) the information comes from.

If asked to compare the two videos, structure your answer clearly:
  Video A: ...
  Video B: ...
  Conclusion: ...

Be concise, specific, and actionable. Creators want insights they can act on.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Streaming callback handler
# ─────────────────────────────────────────────────────────────────────────────

class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Collects streamed tokens into a queue-like list.
    The async generator in build_rag_stream() reads from this.
    """

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.source_documents: list[Document] = []
        self.done: bool = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)

    from typing import Sequence, Any

def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> None:
    """Captures retrieved docs the moment retrieval finishes."""
    self.source_documents = list(documents)
    logger.debug("Retriever returned %d documents", len(documents))

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self.done = True

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        self.done = True


# ─────────────────────────────────────────────────────────────────────────────
# Memory reconstruction (stateless backend)
# ─────────────────────────────────────────────────────────────────────────────

def _build_memory(history: list[ChatTurn]) -> ConversationBufferMemory:
    """
    Reconstructs LangChain ConversationBufferMemory from client-sent history.

    The backend is fully stateless — no Redis, no server sessions.
    The client sends the full conversation history on every request,
    and we rebuild memory from scratch each turn.

    Truncates to the last MEMORY_MAX_TURNS turns to avoid context overflow.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Truncate to last N turns
    recent = history[-(MEMORY_MAX_TURNS * 2):]

    for turn in recent:
        if turn.role == "user":
            memory.chat_memory.add_message(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            memory.chat_memory.add_message(AIMessage(content=turn.content))

    return memory


# ─────────────────────────────────────────────────────────────────────────────
# Source chunk extraction
# ─────────────────────────────────────────────────────────────────────────────

def _docs_to_source_chunks(docs: list[Document]) -> list[SourceChunk]:
    """
    Converts LangChain Documents into SourceChunk schema objects.
    Deduplicates by (video_id, chunk_index).
    """
    seen: set[tuple[str, int]] = set()
    sources: list[SourceChunk] = []

    for doc in docs:
        meta = doc.metadata or {}
        vid  = meta.get("video_id", "A")
        cidx = int(meta.get("chunk_index", 0))
        key  = (vid, cidx)

        if key in seen:
            continue
        seen.add(key)

        sources.append(
            SourceChunk(
                video_id=VideoID(vid),
                chunk_index=cidx,
                text=doc.page_content,
                start_char=int(meta.get("start_char", 0)),
                score=meta.get("score"),
                creator=meta.get("creator"),
            )
        )

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangChain chain
# ─────────────────────────────────────────────────────────────────────────────

def _build_chain(
    memory: ConversationBufferMemory,
    callback_handler: StreamingCallbackHandler,
    video_id_filter: str | None = None,
) -> ConversationalRetrievalChain:
    """
    Assembles ConversationalRetrievalChain with:
    - Mistral streaming LLM
    - ChromaDB retriever (optionally filtered)
    - Buffer memory
    - Custom streaming callback
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY not set in .env. "
            "Get a free key at https://console.mistral.ai"
        )

    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MISTRAL_MODEL,
        streaming=True,
        temperature=0.3,       # Low temp for factual accuracy on analytics questions
        callbacks=[callback_handler],
    )

    retriever = get_retriever(video_id_filter=video_id_filter)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": _build_prompt_template(),
        },
    )

    return chain


def _build_prompt_template():
    """Builds the PromptTemplate with system context injected."""
    from langchain.prompts import PromptTemplate

    template = (
        SYSTEM_PROMPT
        + """
\nContext from video transcripts:
{context}

Question: {question}

Answer:"""
    )

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: async streaming generator
# ─────────────────────────────────────────────────────────────────────────────

async def build_rag_stream(
    question: str,
    history: list[ChatTurn],
    video_id_filter: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Main entry point for the chat router.

    Yields SSE-formatted strings:
      data: {"type": "token",   "content": "Hello"}
      data: {"type": "sources", "content": [...]}
      data: {"type": "done"}

    Args:
        question:         The user's question
        history:          Full conversation history (client-managed)
        video_id_filter:  "A", "B", or None (both videos)

    Usage in FastAPI:
        return StreamingResponse(
            build_rag_stream(question, history),
            media_type="text/event-stream"
        )
    """
    import asyncio

    handler = StreamingCallbackHandler()
    memory  = _build_memory(history)
    chain   = _build_chain(memory, handler, video_id_filter)

    # Run the chain in a thread (LangChain sync chains block the event loop)
    async def _run_chain():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: chain.invoke({"question": question}),
        )

    # Start chain execution in background
    chain_task = asyncio.create_task(_run_chain())

    # Stream tokens as they arrive
    token_index = 0
    while not handler.done or token_index < len(handler.tokens):
        # Yield any new tokens
        while token_index < len(handler.tokens):
            token = handler.tokens[token_index]
            token_index += 1
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"

        # Brief sleep to avoid busy-waiting
        if not handler.done:
            await asyncio.sleep(0.01)

    # Drain any remaining tokens after done flag
    while token_index < len(handler.tokens):
        token = handler.tokens[token_index]
        token_index += 1
        payload = json.dumps({"type": "token", "content": token})
        yield f"data: {payload}\n\n"

    # Wait for chain to fully complete
    await chain_task

    # Emit source chunks
    sources = _docs_to_source_chunks(handler.source_documents)
    sources_payload = json.dumps({
        "type": "sources",
        "content": [s.model_dump() for s in sources],
    })
    yield f"data: {sources_payload}\n\n"

    # Terminal event
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    logger.info(
        "Stream complete: %d tokens, %d sources",
        len(handler.tokens), len(sources),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: non-streaming (for testing / fallback)
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_query(
    question: str,
    history: list[ChatTurn],
    video_id_filter: str | None = None,
) -> dict[str, Any]:
    """
    Synchronous RAG query — returns full answer + sources at once.
    Use for curl testing. Production uses build_rag_stream().
    """
    handler = StreamingCallbackHandler()
    memory  = _build_memory(history)
    chain   = _build_chain(memory, handler, video_id_filter)

    result  = chain.invoke({"question": question})
    answer  = result.get("answer", "")
    sources = _docs_to_source_chunks(result.get("source_documents", []))

    return {
        "answer":  answer,
        "sources": [s.model_dump() for s in sources],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.rag_chain
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Requires vector store to have data — run vector_store.py test first
    print("\n── rag_chain standalone test ──")
    print("NOTE: Run vector_store.py test first to seed ChromaDB with sample data.\n")

    test_questions = [
        "What makes a good hook in a social media video?",
        "Compare the engagement strategies in both videos.",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        try:
            result = run_rag_query(question=q, history=[])
            print(f"A: {result['answer'][:300]}")
            print(f"Sources: {len(result['sources'])} chunks")
            for s in result["sources"]:
                print(f"  Video {s['video_id']} · chunk {s['chunk_index']}: {s['text'][:60]}...")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
        print()