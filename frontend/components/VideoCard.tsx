"use client";

import Image from "next/image";
import { fmtCompact, fmtDate, fmtDuration, fmtFull } from "@/lib/format";
import type { VideoMetadata } from "@/lib/types";
import { SlotBadge } from "./IngestForm";

export function VideoCard({ video }: { video: VideoMetadata }) {
  const slot = video.video_id;
  const accent = slot === "A" ? "text-a" : "text-b";

  return (
    <article className="overflow-hidden rounded-2xl border border-line bg-surface">
      <div className="flex gap-3 p-3">
        <a
          href={video.url}
          target="_blank"
          rel="noopener noreferrer"
          className="relative block aspect-video w-32 shrink-0 overflow-hidden rounded-lg bg-surface-2 sm:w-36"
          aria-label={`Open “${video.title}” on YouTube`}
        >
          {video.thumbnail_url ? (
            <Image src={video.thumbnail_url} alt="" fill sizes="144px" className="object-cover" />
          ) : null}
          <span className="absolute right-1 bottom-1 rounded bg-black/75 px-1 text-[0.68rem] font-medium text-white tabular-nums">
            {fmtDuration(video.duration_seconds)}
          </span>
        </a>

        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-1.5">
            <SlotBadge slot={slot} />
            <span className={`text-xs font-semibold tracking-wide uppercase ${accent}`}>Video {slot}</span>
          </div>
          <h3 className="line-clamp-2 text-sm leading-snug font-semibold" title={video.title}>
            {video.title}
          </h3>
          <p className="mt-0.5 truncate text-xs text-muted">
            {video.creator}
            {video.follower_count != null && ` · ${fmtCompact(video.follower_count)} subs`}
            {` · ${fmtDate(video.upload_date)}`}
          </p>
        </div>
      </div>

      <dl className="grid grid-cols-4 border-t border-line text-center">
        <Stat label="Views" value={fmtCompact(video.views)} title={fmtFull(video.views)} />
        <Stat label="Likes" value={fmtCompact(video.likes)} title={fmtFull(video.likes)} />
        <Stat label="Comments" value={fmtCompact(video.comments)} title={fmtFull(video.comments)} />
        <Stat label="Engagement" value={`${video.engagement_rate.toFixed(2)}%`} strong accent={accent} />
      </dl>

      {(video.opening_text || video.hashtags.length > 0) && (
        <div className="space-y-2 border-t border-line px-3 py-2.5">
          {video.opening_text && (
            <p className="text-xs leading-relaxed text-muted">
              <span className="font-semibold text-ink">Hook · </span>“{video.opening_text}”
            </p>
          )}
          {video.hashtags.length > 0 && (
            <ul className="flex flex-wrap gap-1" aria-label="Tags">
              {video.hashtags.slice(0, 6).map((t) => (
                <li key={t} className="rounded-full bg-surface-2 px-2 py-0.5 text-[0.7rem] text-muted">
                  #{t}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <details className="group border-t border-line">
        <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2 text-xs font-medium text-muted hover:text-ink">
          <span>
            Transcript · {video.chunk_count} chunk{video.chunk_count === 1 ? "" : "s"}
            {video.transcript_source === "whisper" ? " · auto-transcribed" : " · captions"}
          </span>
          <span className="transition group-open:rotate-180" aria-hidden>
            ▾
          </span>
        </summary>
        <p className="max-h-48 overflow-y-auto px-3 pb-3 text-xs leading-relaxed whitespace-pre-line text-muted">
          {video.transcript}
        </p>
      </details>
    </article>
  );
}

function Stat({
  label,
  value,
  title,
  strong,
  accent,
}: {
  label: string;
  value: string;
  title?: string;
  strong?: boolean;
  accent?: string;
}) {
  return (
    <div className="border-l border-line px-1 py-2 first:border-l-0" title={title}>
      <dt className="text-[0.68rem] text-muted">{label}</dt>
      <dd className={`text-sm tabular-nums ${strong ? `font-bold ${accent}` : "font-semibold"}`}>{value}</dd>
    </div>
  );
}
