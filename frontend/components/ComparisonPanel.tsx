"use client";

import { fmtCompact, fmtDuration } from "@/lib/format";
import type { VideoMetadata } from "@/lib/types";

interface Metric {
  label: string;
  a: number | null;
  b: number | null;
  format: (n: number) => string;
  /** Some metrics (duration) have no "better" direction. */
  neutral?: boolean;
}

export function ComparisonPanel({ a, b }: { a: VideoMetadata; b: VideoMetadata }) {
  const vps = (v: VideoMetadata) => (v.follower_count ? v.views / v.follower_count : null);
  const metrics: Metric[] = [
    { label: "Engagement rate", a: a.engagement_rate, b: b.engagement_rate, format: (n) => `${n.toFixed(2)}%` },
    { label: "Views", a: a.views, b: b.views, format: fmtCompact },
    { label: "Likes", a: a.likes, b: b.likes, format: fmtCompact },
    { label: "Comments", a: a.comments, b: b.comments, format: fmtCompact },
    { label: "Subscribers", a: a.follower_count, b: b.follower_count, format: fmtCompact },
    { label: "Views / subscriber", a: vps(a), b: vps(b), format: (n) => n.toFixed(2) },
    { label: "Duration", a: a.duration_seconds, b: b.duration_seconds, format: fmtDuration, neutral: true },
  ];

  const diff = Math.abs(a.engagement_rate - b.engagement_rate);
  const winner = a.engagement_rate === b.engagement_rate ? null : a.engagement_rate > b.engagement_rate ? "A" : "B";

  return (
    <section className="rounded-2xl border border-line bg-surface p-4" aria-labelledby="cmp-title">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h2 id="cmp-title" className="text-sm font-semibold">
          Head to head
        </h2>
        <div className="flex items-center gap-3 text-[0.7rem] text-muted">
          <Legend slot="A" />
          <Legend slot="B" />
        </div>
      </div>

      <p className="mb-4 text-sm text-muted">
        {winner ? (
          <>
            <strong className={winner === "A" ? "text-a" : "text-b"}>Video {winner}</strong> engages better by{" "}
            <strong className="text-ink">{diff.toFixed(2)} pts</strong> (
            {fmtX(winner === "A" ? a.engagement_rate : b.engagement_rate, winner === "A" ? b.engagement_rate : a.engagement_rate)}
            ).
          </>
        ) : (
          "Both videos have the same engagement rate."
        )}
      </p>

      <ul className="space-y-3">
        {metrics.map((m) => (
          <MetricRow key={m.label} m={m} />
        ))}
      </ul>
    </section>
  );
}

function fmtX(hi: number, lo: number) {
  return lo > 0 ? `${(hi / lo).toFixed(1)}× higher` : "the other has none";
}

function Legend({ slot }: { slot: "A" | "B" }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className={`size-2 rounded-full ${slot === "A" ? "bg-a" : "bg-b"}`} aria-hidden />
      {slot}
    </span>
  );
}

function MetricRow({ m }: { m: Metric }) {
  if (m.a == null && m.b == null) return null;
  const max = Math.max(m.a ?? 0, m.b ?? 0) || 1;
  const better = m.neutral || m.a === m.b || m.a == null || m.b == null ? null : m.a > m.b ? "A" : "B";

  return (
    <li>
      <div className="mb-1 text-xs text-muted">{m.label}</div>
      {(["A", "B"] as const).map((slot) => {
        const v = slot === "A" ? m.a : m.b;
        const pct = v == null ? 0 : Math.max(2, (v / max) * 100);
        return (
          <div key={slot} className="flex items-center gap-2">
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
              <div
                className={`h-full rounded-full ${slot === "A" ? "bg-a" : "bg-b"} ${
                  better && better !== slot ? "opacity-45" : ""
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span
              className={`w-16 text-right text-xs tabular-nums ${better === slot ? "font-semibold text-ink" : "text-muted"}`}
            >
              <span className="sr-only">Video {slot}: </span>
              {v == null ? "—" : m.format(v)}
            </span>
          </div>
        );
      })}
    </li>
  );
}
