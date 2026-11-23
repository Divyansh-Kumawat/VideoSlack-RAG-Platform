// Mirrors backend/models/schemas.py

export type VideoSlot = "A" | "B";

export interface VideoMetadata {
  video_id: VideoSlot;
  platform: string;
  youtube_id: string;
  url: string;
  title: string;
  creator: string;
  follower_count: number | null;
  views: number;
  likes: number;
  comments: number;
  engagement_rate: number;
  upload_date: string;
  duration_seconds: number;
  hashtags: string[];
  thumbnail_url: string | null;
  description: string | null;
  transcript: string;
  opening_text: string | null;
  transcript_source: "captions" | "whisper" | string;
  language: string | null;
  chunk_count: number;
}

export interface IngestResponse {
  video_a: VideoMetadata;
  video_b: VideoMetadata;
  chunks_stored: number;
  message: string;
}

export interface VideosResponse {
  video_a: VideoMetadata | null;
  video_b: VideoMetadata | null;
  ready: boolean;
}

export interface SourceChunk {
  video_id: VideoSlot;
  chunk_index: number;
  text: string;
  start_char: number;
  start_seconds: number | null;
  timestamp: string | null;
  url: string | null;
  title: string | null;
  creator: string | null;
  score: number | null;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatMessage extends ChatTurn {
  id: string;
  sources?: SourceChunk[];
  status?: string;          // transient progress note, e.g. rate-limit retry
  error?: string;
  streaming?: boolean;
}

export type StreamEvent =
  | { type: "token"; content: string }
  | { type: "status"; content: string }
  | { type: "sources"; content: SourceChunk[] }
  | { type: "error"; content: string }
  | { type: "done" };

export interface Health {
  status: "ok" | "degraded";
  issues: string[];
  notes: string[];
  llm_model: string;
  embedding_model: string;
  chunks_in_store: number;
}
