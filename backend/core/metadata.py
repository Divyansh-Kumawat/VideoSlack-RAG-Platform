# backend/core/metadata.py
# ─────────────────────────────────────────────────────────────────────────────
# YouTube metadata extraction.
#
# Primary:  YouTube Data API v3 (needs YOUTUBE_API_KEY, ~2 quota units/video).
# Fallback: yt-dlp page scrape (no key needed) — used when the key is missing
#           or the API call fails, so ingestion never depends on the key.
#
# Returns a dict whose keys map directly onto the VideoMetadata schema.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import re
from typing import Any

from backend import config
from backend.models.schemas import extract_youtube_id

logger = logging.getLogger(__name__)


class MetadataError(RuntimeError):
    """Raised with a human-readable reason when metadata cannot be fetched."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_iso8601_duration(duration: str) -> int:
    """'PT4M13S' → 253, 'PT1H2M30S' → 3750, 'P1DT1S' → 86401."""
    match = re.fullmatch(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", duration or ""
    )
    if not match:
        return 0
    d, h, m, s = (int(g or 0) for g in match.groups())
    return d * 86400 + h * 3600 + m * 60 + s


def extract_hashtags(text: str) -> list[str]:
    """Hashtags (without '#') found in a text blob, in order, de-duplicated."""
    seen: dict[str, None] = {}
    for tag in re.findall(r"#(\w+)", text or ""):
        seen.setdefault(tag, None)
    return list(seen)


def _merge_tags(*groups: list[str], limit: int = 20) -> list[str]:
    out: dict[str, None] = {}
    for group in groups:
        for tag in group or []:
            tag = tag.strip().lstrip("#")
            if tag and tag.lower() not in {t.lower() for t in out}:
                out[tag] = None
    return list(out)[:limit]


def _best_thumbnail(thumbnails: dict[str, Any]) -> str | None:
    for key in ("maxres", "standard", "high", "medium", "default"):
        if thumbnails.get(key, {}).get("url"):
            return thumbnails[key]["url"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Data API v3
# ─────────────────────────────────────────────────────────────────────────────

def get_metadata_from_api(youtube_id: str, url: str) -> dict[str, Any]:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY, cache_discovery=False)
        response = (
            youtube.videos()
            .list(part="snippet,statistics,contentDetails", id=youtube_id)
            .execute()
        )
    except HttpError as e:
        raise MetadataError(f"YouTube Data API error: {getattr(e, 'reason', e)}") from e

    items = response.get("items", [])
    if not items:
        raise MetadataError(
            f"No YouTube video found for ID '{youtube_id}' — it may be private or deleted."
        )

    item    = items[0]
    snippet = item.get("snippet", {})
    stats   = item.get("statistics", {})
    content = item.get("contentDetails", {})
    description = snippet.get("description", "")

    follower_count = None
    channel_id = snippet.get("channelId")
    if channel_id:
        try:
            ch = youtube.channels().list(part="statistics", id=channel_id).execute()
            ch_items = ch.get("items", [])
            if ch_items and not ch_items[0].get("statistics", {}).get("hiddenSubscriberCount"):
                follower_count = int(ch_items[0]["statistics"].get("subscriberCount", 0))
        except HttpError:
            logger.warning("Could not fetch subscriber count for channel %s", channel_id)

    return {
        "youtube_id":       youtube_id,
        "url":              url,
        "title":            snippet.get("title", "Untitled video"),
        "creator":          snippet.get("channelTitle", "Unknown creator"),
        "follower_count":   follower_count,
        "views":            int(stats.get("viewCount", 0)),
        "likes":            int(stats.get("likeCount", 0)),       # 0 if likes are hidden
        "comments":         int(stats.get("commentCount", 0)),    # 0 if comments are off
        "upload_date":      (snippet.get("publishedAt") or "")[:10],
        "duration_seconds": parse_iso8601_duration(content.get("duration", "")),
        "hashtags":         _merge_tags(extract_hashtags(description), snippet.get("tags", [])),
        "thumbnail_url":    _best_thumbnail(snippet.get("thumbnails", {})),
        "description":      description[:2000],
    }


# ─────────────────────────────────────────────────────────────────────────────
# yt-dlp fallback
# ─────────────────────────────────────────────────────────────────────────────

def get_metadata_from_ytdlp(youtube_id: str, url: str) -> dict[str, Any]:
    import yt_dlp
    from yt_dlp.utils import DownloadError

    opts: Any = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=False)
    except DownloadError as e:
        raise MetadataError(f"Could not read video page: {e}") from e
    if not info:
        raise MetadataError("yt-dlp returned no information for this video.")

    raw_date = info.get("upload_date") or ""          # "20240315"
    upload_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else ""
    description = info.get("description") or ""

    return {
        "youtube_id":       youtube_id,
        "url":              url,
        "title":            info.get("title") or "Untitled video",
        "creator":          info.get("channel") or info.get("uploader") or "Unknown creator",
        "follower_count":   info.get("channel_follower_count"),
        "views":            int(info.get("view_count") or 0),
        "likes":            int(info.get("like_count") or 0),
        "comments":         int(info.get("comment_count") or 0),
        "upload_date":      upload_date,
        "duration_seconds": int(info.get("duration") or 0),
        "hashtags":         _merge_tags(extract_hashtags(description), info.get("tags") or []),
        "thumbnail_url":    info.get("thumbnail"),
        "description":      description[:2000],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_metadata(url: str) -> dict[str, Any]:
    youtube_id = extract_youtube_id(url)

    if config.YOUTUBE_API_KEY:
        try:
            meta = get_metadata_from_api(youtube_id, url)
            logger.info("Metadata via YouTube API: '%s' views=%d", meta["title"], meta["views"])
            return meta
        except MetadataError as e:
            if "No YouTube video found" in str(e):
                raise
            logger.warning("YouTube API failed (%s) — falling back to yt-dlp.", e)
    else:
        logger.info("YOUTUBE_API_KEY not set — using yt-dlp for metadata.")

    meta = get_metadata_from_ytdlp(youtube_id, url)
    logger.info("Metadata via yt-dlp: '%s' views=%d", meta["title"], meta["views"])
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.metadata <url>
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    m = get_metadata(test_url)
    m.pop("description", None)
    print(json.dumps(m, indent=2))
