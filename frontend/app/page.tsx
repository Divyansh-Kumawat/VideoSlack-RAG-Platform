"use client";

import { useCallback, useEffect, useState } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { ComparisonPanel } from "@/components/ComparisonPanel";
import { IngestForm, Spinner } from "@/components/IngestForm";
import { VideoCard } from "@/components/VideoCard";
import { API_URL, getHealth, getVideos } from "@/lib/api";
import type { Health, VideoMetadata } from "@/lib/types";

type Pair = { a: VideoMetadata; b: VideoMetadata };
type View = { kind: "loading" } | { kind: "offline"; message: string } | { kind: "form" } | { kind: "ready" };

export default function Home() {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [pair, setPair] = useState<Pair | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  const load = useCallback(async () => {
    setView({ kind: "loading" });
    try {
      const [videos, h] = await Promise.all([getVideos(), getHealth()]);
      setHealth(h);
      if (videos.ready && videos.video_a && videos.video_b) {
        setPair({ a: videos.video_a, b: videos.video_b });
        setView({ kind: "ready" });
      } else {
        setView({ kind: "form" });
      }
    } catch (e) {
      setView({ kind: "offline", message: (e as Error).message });
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data fetch on mount
    load();
  }, [load]);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-4">
          <button onClick={() => pair && setView({ kind: "ready" })} className="flex items-center gap-2">
            <Logo />
            <span className="text-[0.95rem] font-semibold tracking-tight">VideoSlack</span>
          </button>
          {view.kind === "ready" && (
            <button
              onClick={() => setView({ kind: "form" })}
              className="rounded-lg border border-line px-3 py-1.5 text-sm font-medium hover:bg-surface"
            >
              New comparison
            </button>
          )}
          {view.kind === "form" && pair && (
            <button
              onClick={() => setView({ kind: "ready" })}
              className="rounded-lg px-3 py-1.5 text-sm text-muted hover:text-ink"
            >
              Back to current
            </button>
          )}
        </div>
      </header>

      {health && health.issues.length > 0 && (
        <div className="border-b border-line bg-warn-soft">
          <p className="mx-auto max-w-7xl px-4 py-2 text-sm">
            <strong>Backend needs attention:</strong> {health.issues.join(" · ")}
          </p>
        </div>
      )}

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        {view.kind === "loading" && (
          <div className="grid place-items-center py-32 text-muted">
            <Spinner className="size-6" />
          </div>
        )}

        {view.kind === "offline" && (
          <div className="mx-auto max-w-lg py-24 text-center">
            <h1 className="mb-2 text-lg font-semibold">Can&apos;t reach the backend</h1>
            <p className="mb-1 text-sm text-muted">{view.message}</p>
            <p className="mb-6 text-sm text-muted">
              Start it from the repo root with{" "}
              <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs">
                uvicorn backend.main:app --port 8000
              </code>{" "}
              (expected at {API_URL}).
            </p>
            <button
              onClick={load}
              className="h-10 rounded-xl bg-ink px-5 text-sm font-semibold text-bg hover:opacity-90"
            >
              Try again
            </button>
          </div>
        )}

        {view.kind === "form" && (
          <div className="mx-auto max-w-xl py-10 sm:py-16">
            <h1 className="mb-3 text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
              Why did one video beat the other?
            </h1>
            <p className="mb-8 text-muted text-pretty">
              Paste two YouTube videos. VideoSlack pulls their transcripts and stats, then lets you ask an AI analyst
              about hooks, engagement and what to change — with answers that cite the exact moment in each video.
            </p>
            <div className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
              <IngestForm
                onIngested={(r) => {
                  setPair({ a: r.video_a, b: r.video_b });
                  setView({ kind: "ready" });
                }}
              />
            </div>
          </div>
        )}

        {view.kind === "ready" && pair && (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-5 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
            <div className="space-y-4">
              <VideoCard video={pair.a} />
              <VideoCard video={pair.b} />
              <ComparisonPanel a={pair.a} b={pair.b} />
            </div>
            <div className="lg:sticky lg:top-20 lg:h-[calc(100vh-6.5rem)]">
              <ChatPanel key={`${pair.a.youtube_id}-${pair.b.youtube_id}`} videoA={pair.a} videoB={pair.b} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function Logo() {
  return (
    <svg viewBox="0 0 28 28" className="size-7" aria-hidden>
      <rect x="1" y="5" width="15" height="18" rx="4" fill="var(--a)" />
      <rect x="12" y="5" width="15" height="18" rx="4" fill="var(--b)" fillOpacity=".9" />
      <path d="M11.5 10.5v7l5.5-3.5z" fill="#fff" />
    </svg>
  );
}
