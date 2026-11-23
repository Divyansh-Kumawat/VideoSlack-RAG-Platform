"use client";

import { memo, useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { parseTimestamp } from "@/lib/format";
import type { VideoMetadata, VideoSlot } from "@/lib/types";

// Citations the model writes inline: [A 0:42], [B 1:05], **[A 0:00]**, [A @ 0:42]
const CITATION_RE = /\[(A|B)\s*@?\s*(\d{1,2}:\d{2}(?::\d{2})?)\]/g;

/** Rewrites [A 0:42] into a markdown link to that moment of the video. */
function linkCitations(text: string, videos: Partial<Record<VideoSlot, VideoMetadata | null>>) {
  return text.replace(CITATION_RE, (match, slot: VideoSlot, ts: string) => {
    const v = videos[slot];
    if (!v) return match;
    const url = `https://www.youtube.com/watch?v=${v.youtube_id}&t=${parseTimestamp(ts)}s`;
    return `[${slot} · ${ts}](${url} "cite")`;
  });
}

const components: Components = {
  a({ href, title, children }) {
    if (title === "cite") {
      const slot = String(children).trim().startsWith("B") ? "B" : "A";
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          title={`Open Video ${slot} at this moment`}
          className={`inline-flex items-center rounded-md px-1.5 py-px text-[0.75rem] font-medium no-underline! tabular-nums align-[1px] whitespace-nowrap ${
            slot === "A" ? "bg-a-soft text-a" : "bg-b-soft text-b"
          } hover:brightness-95`}
        >
          ▶ {children}
        </a>
      );
    }
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="text-a">
        {children}
      </a>
    );
  },
};

function MarkdownImpl({
  text,
  videos,
}: {
  text: string;
  videos: Partial<Record<VideoSlot, VideoMetadata | null>>;
}) {
  const processed = useMemo(() => linkCitations(text, videos), [text, videos]);
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {processed}
    </ReactMarkdown>
  );
}

export const Markdown = memo(MarkdownImpl);
