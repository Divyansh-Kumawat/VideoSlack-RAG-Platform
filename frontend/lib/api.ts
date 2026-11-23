import type {
  ChatTurn,
  Health,
  IngestResponse,
  StreamEvent,
  VideoSlot,
  VideosResponse,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  videoId: VideoSlot | null;

  constructor(message: string, status: number, videoId: VideoSlot | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.videoId = videoId;
  }
}

/** Turns FastAPI's various error shapes into one readable message. */
async function toApiError(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  let videoId: VideoSlot | null = null;
  try {
    const body = await res.json();
    const d = body?.detail;
    if (typeof d === "string") message = d;
    else if (Array.isArray(d)) {
      // pydantic validation errors
      message = d.map((e) => String(e?.msg ?? e).replace(/^Value error, /, "")).join(" · ");
    } else if (d && typeof d === "object") {
      message = d.detail || d.error || message;
      videoId = d.video_id ?? null;
    }
  } catch {
    /* non-JSON body */
  }
  return new ApiError(message, res.status, videoId);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
  } catch {
    throw new ApiError(
      `Can't reach the backend at ${API_URL}. Is it running? (uvicorn backend.main:app --port 8000)`,
      0,
    );
  }
  if (!res.ok) throw await toApiError(res);
  return res.json() as Promise<T>;
}

export const getHealth = () => request<Health>("/health");
export const getVideos = () => request<VideosResponse>("/videos");
export const deleteVideos = () => request<{ status: string }>("/videos", { method: "DELETE" });

export const ingestVideos = (youtube_url_a: string, youtube_url_b: string, signal?: AbortSignal) =>
  request<IngestResponse>("/ingest", {
    method: "POST",
    body: JSON.stringify({ youtube_url_a, youtube_url_b }),
    signal,
  });

/**
 * POST /chat/stream and yield parsed SSE events.
 * Uses fetch + ReadableStream (EventSource can't send a POST body).
 */
export async function* streamChat(
  question: string,
  history: ChatTurn[],
  videoFilter: VideoSlot | null,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question, history, video_filter: videoFilter }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(`Can't reach the backend at ${API_URL}.`, 0);
  }
  if (!res.ok || !res.body) throw await toApiError(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const data = raw
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      try {
        const event = JSON.parse(data) as StreamEvent;
        yield event;
        if (event.type === "done") return;
      } catch {
        /* ignore malformed chunk */
      }
    }
  }
}
