"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, streamChat } from "@/lib/api";
import { uid } from "@/lib/format";
import type { ChatMessage, ChatTurn, SourceChunk, VideoMetadata, VideoSlot } from "@/lib/types";
import { SlotBadge, Spinner } from "./IngestForm";
import { Markdown } from "./Markdown";

const SUGGESTIONS = [
  "Why did one video get more engagement than the other?",
  "Compare the hooks in the first few seconds of both videos.",
  "Who is each creator and how big is their audience?",
  "Suggest 3 concrete improvements for the weaker video.",
];

type Scope = "both" | VideoSlot;

function loadHistory(key: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(key);
    const parsed = raw ? (JSON.parse(raw) as ChatMessage[]) : [];
    return parsed.map((m) => ({ ...m, streaming: false, status: undefined }));
  } catch {
    return [];
  }
}

export function ChatPanel({ videoA, videoB }: { videoA: VideoMetadata; videoB: VideoMetadata }) {
  const storageKey = `videoslack:chat:${videoA.youtube_id}:${videoB.youtube_id}`;
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadHistory(storageKey));
  const [input, setInput] = useState("");
  const [scope, setScope] = useState<Scope>("both");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const videos = { A: videoA, B: videoB };

  // persist finished conversations
  useEffect(() => {
    if (busy) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(messages.slice(-60)));
    } catch {
      /* storage full or blocked — chat still works */
    }
  }, [messages, busy, storageKey]);

  // keep the newest message in view while streaming, unless the user scrolled up
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const patch = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  async function send(question: string) {
    const q = question.trim();
    if (!q || busy) return;

    const history: ChatTurn[] = messages
      .filter((m) => !m.error && m.content.trim())
      .map(({ role, content }) => ({ role, content }));
    const userMsg: ChatMessage = { id: uid(), role: "user", content: q };
    const botId = uid();
    setMessages((ms) => [...ms, userMsg, { id: botId, role: "assistant", content: "", streaming: true }]);
    setInput("");
    setBusy(true);
    stickToBottom.current = true;

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      for await (const ev of streamChat(q, history, scope === "both" ? null : scope, ctrl.signal)) {
        if (ev.type === "token") patch(botId, (m) => ({ ...m, content: m.content + ev.content, status: undefined }));
        else if (ev.type === "status") patch(botId, (m) => ({ ...m, status: ev.content }));
        else if (ev.type === "sources") patch(botId, (m) => ({ ...m, sources: ev.content }));
        else if (ev.type === "error") patch(botId, (m) => ({ ...m, error: ev.content, status: undefined }));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        patch(botId, (m) => ({ ...m, error: (e as ApiError).message || "Something went wrong." }));
      }
    } finally {
      patch(botId, (m) => ({
        ...m,
        streaming: false,
        status: undefined,
        error: m.error ?? (m.content ? undefined : ctrl.signal.aborted ? "Stopped." : "No answer was returned."),
      }));
      setBusy(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  }

  function retry(index: number) {
    const question = messages[index - 1]?.content;
    if (!question) return;
    setMessages((ms) => ms.slice(0, index - 1));
    setTimeout(() => send(question), 0);
  }

  return (
    <section
      className="flex h-full min-h-[560px] flex-col overflow-hidden rounded-2xl border border-line bg-surface"
      aria-label="Chat about the videos"
    >
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Ask about these videos</h2>
          <p className="text-xs text-muted">Answers cite transcript moments — click a tag to jump there.</p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={() => {
              abortRef.current?.abort();
              setMessages([]);
            }}
            className="shrink-0 rounded-lg px-2.5 py-1.5 text-xs text-muted hover:bg-surface-2 hover:text-ink"
          >
            Clear chat
          </button>
        )}
      </header>

      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
        className="flex-1 space-y-5 overflow-y-auto px-4 py-5"
        aria-live="polite"
      >
        {messages.length === 0 ? (
          <div className="mx-auto max-w-md py-6 text-center">
            <p className="mb-4 text-sm text-muted">Try one of these, or ask your own question.</p>
            <div className="grid gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-xl border border-line px-3.5 py-2.5 text-left text-sm transition hover:border-faint hover:bg-surface-2"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <p className="max-w-[85%] rounded-2xl rounded-br-md bg-ink px-3.5 py-2 text-sm whitespace-pre-wrap text-bg">
                  {m.content}
                </p>
              </div>
            ) : (
              <AssistantMessage key={m.id} m={m} videos={videos} onRetry={() => retry(i)} canRetry={!busy} />
            ),
          )
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="border-t border-line p-3"
      >
        <div className="mb-2 flex items-center gap-2">
          <span className="text-xs text-muted" id="scope-label">
            Search
          </span>
          <div role="radiogroup" aria-labelledby="scope-label" className="inline-flex rounded-lg bg-surface-2 p-0.5">
            {(["both", "A", "B"] as Scope[]).map((s) => (
              <button
                key={s}
                type="button"
                role="radio"
                aria-checked={scope === s}
                onClick={() => setScope(s)}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                  scope === s
                    ? `bg-surface shadow-sm ${s === "A" ? "text-a" : s === "B" ? "text-b" : "text-ink"}`
                    : "text-muted hover:text-ink"
                }`}
              >
                {s === "both" ? "Both videos" : `Video ${s}`}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-end gap-2">
          <label htmlFor="chat-input" className="sr-only">
            Your question
          </label>
          <textarea
            id="chat-input"
            ref={inputRef}
            rows={1}
            value={input}
            maxLength={2000}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder="e.g. Why did Video A get fewer comments?"
            className="max-h-40 min-h-11 w-full min-w-0 flex-1 resize-none rounded-xl border border-line bg-bg px-3.5 py-2.5 text-sm outline-none field-sizing-content placeholder:text-faint focus:border-faint"
          />
          {busy ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="h-11 rounded-xl border border-line px-4 text-sm font-medium hover:bg-surface-2"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="h-11 rounded-xl bg-ink px-4 text-sm font-semibold text-bg transition hover:opacity-90 disabled:opacity-40"
            >
              Send
            </button>
          )}
        </div>
      </form>
    </section>
  );
}

function AssistantMessage({
  m,
  videos,
  onRetry,
  canRetry,
}: {
  m: ChatMessage;
  videos: Record<VideoSlot, VideoMetadata>;
  onRetry: () => void;
  canRetry: boolean;
}) {
  const waiting = m.streaming && !m.content;
  return (
    <div className="max-w-full">
      {waiting && (
        <p className="flex items-center gap-2 text-sm text-muted">
          <Spinner className="size-3.5" />
          {m.status ?? "Searching transcripts and thinking…"}
        </p>
      )}
      {m.content && (
        <div className={`prose-chat ${m.streaming ? "caret" : ""}`}>
          <Markdown text={m.content} videos={videos} />
        </div>
      )}
      {m.error && (
        <div role="alert" className="mt-2 flex flex-wrap items-center gap-2 rounded-lg bg-bad-soft px-3 py-2 text-sm text-bad">
          <span className="flex-1">{m.error}</span>
          {canRetry && m.error !== "Stopped." && (
            <button onClick={onRetry} className="font-semibold underline underline-offset-2">
              Retry
            </button>
          )}
        </div>
      )}
      {!m.streaming && m.sources && m.sources.length > 0 && <Sources sources={m.sources} />}
    </div>
  );
}

function Sources({ sources }: { sources: SourceChunk[] }) {
  return (
    <details className="group mt-3">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-muted hover:text-ink">
        <span className="transition group-open:rotate-90" aria-hidden>
          ▸
        </span>
        {sources.length} transcript source{sources.length === 1 ? "" : "s"}
      </summary>
      <ul className="mt-2 space-y-2">
        {sources.map((s) => (
          <li key={`${s.video_id}-${s.chunk_index}`} className="rounded-xl border border-line bg-bg p-3">
            <div className="mb-1 flex items-center gap-2 text-xs">
              <SlotBadge slot={s.video_id} />
              {s.url ? (
                <a
                  href={s.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`font-semibold tabular-nums hover:underline ${s.video_id === "A" ? "text-a" : "text-b"}`}
                >
                  ▶ {s.timestamp ?? "Open video"}
                </a>
              ) : (
                <span className="text-muted">chunk {s.chunk_index}</span>
              )}
              {s.score != null && <span className="ml-auto text-faint">{Math.round(s.score * 100)}% match</span>}
            </div>
            <p className="line-clamp-4 text-xs leading-relaxed text-muted">{s.text}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}
