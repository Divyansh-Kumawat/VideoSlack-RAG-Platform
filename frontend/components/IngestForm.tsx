"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, ingestVideos } from "@/lib/api";
import type { IngestResponse, VideoSlot } from "@/lib/types";

const YT_ID_RE = /(?:v=|youtu\.be\/|\/shorts\/|\/embed\/|\/live\/)([A-Za-z0-9_-]{11})/;

const STEPS = [
  "Fetching transcripts…",
  "Reading views, likes & comments…",
  "Computing engagement rates…",
  "Embedding transcript chunks…",
  "Almost there — no captions means transcribing audio, which can take a minute…",
];

const EXAMPLES = {
  a: "https://youtube.com/shorts/XiqalmRZUdg",
  b: "https://youtube.com/shorts/kiMpfSFsOfg",
};

function validate(url: string): string | null {
  if (!url.trim()) return "Paste a YouTube link";
  if (!/youtube\.com|youtu\.be/.test(url)) return "That isn't a YouTube link";
  if (!YT_ID_RE.test(url)) return "Couldn't find a video ID in that link";
  return null;
}

export function IngestForm({ onIngested }: { onIngested: (r: IngestResponse) => void }) {
  const [urlA, setUrlA] = useState("");
  const [urlB, setUrlB] = useState("");
  const [touched, setTouched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  const [error, setError] = useState<{ message: string; video?: VideoSlot | null } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 3500);
    return () => clearInterval(t);
  }, [loading]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const errA = touched ? validate(urlA) : null;
  const sameVideo =
    touched && !errA && !validate(urlB) && urlA.match(YT_ID_RE)?.[1] === urlB.match(YT_ID_RE)?.[1];
  const errB = touched ? validate(urlB) || (sameVideo ? "Video B must be a different video" : null) : null;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setTouched(true);
    if (validate(urlA) || validate(urlB) || urlA.match(YT_ID_RE)?.[1] === urlB.match(YT_ID_RE)?.[1]) return;

    setError(null);
    setStep(0);
    setLoading(true);
    abortRef.current = new AbortController();
    try {
      const res = await ingestVideos(urlA.trim(), urlB.trim(), abortRef.current.signal);
      onIngested(res);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      const apiErr = err as ApiError;
      setError({ message: apiErr.message, video: apiErr.videoId });
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={submit} className="w-full space-y-4" noValidate>
      <UrlField
        slot="A"
        value={urlA}
        onChange={setUrlA}
        error={errA || (error?.video === "A" ? error.message : null)}
        disabled={loading}
        placeholder={EXAMPLES.a}
      />
      <UrlField
        slot="B"
        value={urlB}
        onChange={setUrlB}
        error={errB || (error?.video === "B" ? error.message : null)}
        disabled={loading}
        placeholder={EXAMPLES.b}
      />

      {error && !error.video && (
        <p role="alert" className="rounded-lg bg-bad-soft px-3 py-2 text-sm text-bad">
          {error.message}
        </p>
      )}

      <div className="flex flex-col-reverse items-stretch gap-3 pt-1 sm:flex-row sm:items-center sm:justify-between">
        <button
          type="button"
          disabled={loading}
          onClick={() => {
            setUrlA(EXAMPLES.a);
            setUrlB(EXAMPLES.b);
            setError(null);
          }}
          className="text-sm text-muted underline-offset-4 hover:text-ink hover:underline disabled:opacity-50"
        >
          Try an example pair
        </button>
        <button
          type="submit"
          disabled={loading}
          className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-ink px-6 text-sm font-semibold text-bg transition hover:opacity-90 disabled:opacity-70"
        >
          {loading && <Spinner />}
          {loading ? "Analysing…" : "Analyse videos"}
        </button>
      </div>

      <p aria-live="polite" className="min-h-5 text-center text-sm text-muted sm:text-right">
        {loading ? STEPS[step] : ""}
      </p>
    </form>
  );
}

function UrlField({
  slot,
  value,
  onChange,
  error,
  disabled,
  placeholder,
}: {
  slot: VideoSlot;
  value: string;
  onChange: (v: string) => void;
  error: string | null;
  disabled: boolean;
  placeholder: string;
}) {
  const id = `url-${slot}`;
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 flex items-center gap-2 text-sm font-medium">
        <SlotBadge slot={slot} />
        Video {slot}
      </label>
      <input
        id={id}
        type="url"
        inputMode="url"
        autoComplete="off"
        spellCheck={false}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-invalid={!!error}
        aria-describedby={error ? `${id}-err` : undefined}
        className={`h-11 w-full rounded-xl border bg-surface px-3.5 text-sm outline-none transition placeholder:text-faint focus:ring-2 disabled:opacity-60 ${
          error
            ? "border-bad focus:ring-bad/25"
            : slot === "A"
              ? "border-line focus:border-a focus:ring-a/20"
              : "border-line focus:border-b focus:ring-b/20"
        }`}
      />
      {error && (
        <p id={`${id}-err`} className="mt-1.5 text-sm text-bad">
          {error}
        </p>
      )}
    </div>
  );
}

export function SlotBadge({ slot, size = "sm" }: { slot: VideoSlot; size?: "sm" | "md" }) {
  return (
    <span
      className={`inline-grid shrink-0 place-items-center rounded-md font-bold text-white ${
        slot === "A" ? "bg-a" : "bg-b"
      } ${size === "sm" ? "size-5 text-[0.7rem]" : "size-6 text-xs"}`}
      aria-hidden
    >
      {slot}
    </span>
  );
}

export function Spinner({ className = "size-4" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity=".25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
