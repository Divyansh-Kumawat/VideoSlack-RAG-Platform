# backend/core/metadata.py
# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction for YouTube (via YouTube Data API v3)
# and Instagram (via instaloader).
#
# Returns a dict that maps directly to VideoMetadata schema.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from typing import Any

import instaloader
from instaloader.instaloader import Instaloader
from instaloader.structures import Post, Profile
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_youtube_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Cannot extract YouTube video ID from: {url}")
    return match.group(1)


def _extract_instagram_shortcode(url: str) -> str:
    """Extracts shortcode from URLs like /reel/CODE/ or /p/CODE/"""
    match = re.search(r"/(?:reel|p)/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError(f"Cannot extract Instagram shortcode from: {url}")
    return match.group(1)


def _parse_iso8601_duration(duration: str) -> int:
    """
    Converts YouTube ISO 8601 duration string to total seconds.
    e.g. 'PT4M13S' → 253,  'PT1H2M30S' → 3750
    """
    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match = re.match(pattern, duration)
    if not match:
        return 0
    hours   = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _extract_hashtags(text: str) -> list[str]:
    """Returns list of hashtags (without #) found in any text blob."""
    return re.findall(r"#(\w+)", text or "")


# ─────────────────────────────────────────────────────────────────────────────
# YouTube metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_youtube_metadata(url: str) -> dict[str, Any]:
    """
    Fetches video metadata using the YouTube Data API v3.

    Costs ~3 quota units (snippet + statistics + contentDetails).
    Free tier: 10,000 units/day → supports ~3,333 video lookups/day.

    Returns dict matching VideoMetadata fields (except video_id, platform,
    transcript — those are added by the ingestion pipeline).
    """
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY is not set in .env. "
            "Get a free key at https://console.cloud.google.com"
        )

    video_id = _extract_youtube_video_id(url)
    logger.info("Fetching YouTube metadata for video_id=%s", video_id)

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Single API call — fetch all three parts at once to save quota
        response = (
            youtube.videos()
            .list(
                part="snippet,statistics,contentDetails",
                id=video_id,
            )
            .execute()
        )
    except HttpError as e:
        raise RuntimeError(
            f"YouTube Data API error: {e.reason}. "
            "Check your API key and quota at console.cloud.google.com."
        ) from e

    items = response.get("items", [])
    if not items:
        raise RuntimeError(
            f"No YouTube video found for ID '{video_id}'. "
            "The video may be private, deleted, or the ID is wrong."
        )

    item       = items[0]
    snippet    = item.get("snippet", {})
    stats      = item.get("statistics", {})
    content    = item.get("contentDetails", {})

    # --- Extract fields ---
    title         = snippet.get("title", "Unknown Title")
    creator       = snippet.get("channelTitle", "Unknown Creator")
    upload_date   = snippet.get("publishedAt", "")[:10]   # "2024-03-15T..."  → "2024-03-15"
    description   = snippet.get("description", "")
    tags          = snippet.get("tags", [])
    thumbnail_url = (
        snippet.get("thumbnails", {})
        .get("maxres", snippet.get("thumbnails", {}).get("high", {}))
        .get("url")
    )

    views     = int(stats.get("viewCount",    0))
    likes     = int(stats.get("likeCount",    0))
    comments  = int(stats.get("commentCount", 0))

    duration_seconds = _parse_iso8601_duration(content.get("duration", "PT0S"))

    # Combine API tags + hashtags from description
    hashtags = list(set(tags + _extract_hashtags(description)))

    # Fetch subscriber count (separate channels call)
    channel_id = snippet.get("channelId", "")
    follower_count = None
    if channel_id:
        try:
            ch_response = (
                youtube.channels()
                .list(part="statistics", id=channel_id)
                .execute()
            )
            ch_items = ch_response.get("items", [])
            if ch_items:
                follower_count = int(
                    ch_items[0].get("statistics", {}).get("subscriberCount", 0)
                )
        except HttpError:
            logger.warning("Could not fetch subscriber count for channel %s", channel_id)

    logger.info(
        "YouTube metadata: title='%s', views=%d, likes=%d, comments=%d",
        title, views, likes, comments,
    )

    return {
        "url":              url,
        "title":            title,
        "creator":          creator,
        "follower_count":   follower_count,
        "views":            views,
        "likes":            likes,
        "comments":         comments,
        "upload_date":      upload_date,
        "duration_seconds": duration_seconds,
        "hashtags":         hashtags,
        "thumbnail_url":    thumbnail_url,
        "platform":         "youtube",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Instagram metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_instagram_metadata(url: str) -> dict[str, Any]:
    """
    Scrapes Instagram Reel metadata using instaloader.

    Works anonymously for public profiles. Set INSTAGRAM_USERNAME +
    INSTAGRAM_PASSWORD in .env to increase rate limits.

    Note: Instagram does not expose likes publicly via API; instaloader
    scrapes what is visible on the public page. Likes may show as 0
    if Instagram hides them (they started hiding counts in 2019).
    """
    shortcode = _extract_instagram_shortcode(url)
    logger.info("Fetching Instagram metadata for shortcode=%s", shortcode)

    username = os.getenv("INSTAGRAM_USERNAME", "").strip()
    password = os.getenv("INSTAGRAM_PASSWORD", "").strip()

    L = instaloader.Instaloader(
        quiet=True,
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
    )

    # Login if credentials provided (better rate limits)
    if username and password:
        try:
            L.login(username, password)
            logger.debug("Logged into Instagram as %s", username)
        except instaloader.exceptions.BadCredentialsException:
            logger.warning("Instagram login failed — proceeding anonymously")
        except Exception as e:
            logger.warning("Instagram login error: %s — proceeding anonymously", e)

    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except instaloader.exceptions.InstaloaderException as e:
        raise RuntimeError(
            f"Could not fetch Instagram Reel '{shortcode}': {e}. "
            "The post may be private, deleted, or geo-restricted."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching Instagram post: {e}") from e

    # --- Extract fields ---
    title       = (post.caption or "")[:200].strip()   # Use caption as title (truncated)
    creator     = post.owner_username
    views       = post.video_view_count or 0
    likes       = post.likes or 0
    comments    = post.comments or 0
    upload_date = post.date_utc.strftime("%Y-%m-%d") if post.date_utc else ""
    hashtags    = _extract_hashtags(post.caption or "")

    # Duration: instaloader exposes it for videos/reels
    duration_seconds = int(getattr(post, "video_duration", 0) or 0)

    # Thumbnail
    thumbnail_url = post.url   # direct CDN URL of the video thumbnail

    # Follower count from owner profile
    follower_count = None
    try:
        profile = instaloader.Profile.from_username(L.context, creator)
        follower_count = profile.followers
    except Exception as e:
        logger.warning("Could not fetch follower count for @%s: %s", creator, e)

    logger.info(
        "Instagram metadata: creator=@%s, views=%d, likes=%d, comments=%d",
        creator, views, likes, comments,
    )

    return {
        "url":              url,
        "title":            title or f"Reel by @{creator}",
        "creator":          f"@{creator}",
        "follower_count":   follower_count,
        "views":            views,
        "likes":            likes,
        "comments":         comments,
        "upload_date":      upload_date,
        "duration_seconds": duration_seconds,
        "hashtags":         hashtags,
        "thumbnail_url":    thumbnail_url,
        "platform":         "instagram",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def get_metadata(url: str, platform: str) -> dict[str, Any]:
    """
    Unified entry point for the ingestion pipeline.

    Args:
        url:      Video URL
        platform: "youtube" or "instagram"

    Returns:
        Dict with all metadata fields (maps to VideoMetadata schema).
    """
    platform = platform.lower().strip()
    if platform == "youtube":
        return get_youtube_metadata(url)
    elif platform in ("instagram", "instagram_reel"):
        return get_instagram_metadata(url)
    else:
        raise ValueError(f"Unknown platform: '{platform}'")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.metadata
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    TEST_YT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("\n── YouTube metadata test ──")
    try:
        meta = get_youtube_metadata(TEST_YT_URL)
        meta_display = {k: v for k, v in meta.items() if k != "thumbnail_url"}
        print(json.dumps(meta_display, indent=2))
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)

    # Uncomment to test Instagram:
    # TEST_IG_URL = "https://www.instagram.com/reel/XXXXXXXXXXXX/"
    # print("\n── Instagram metadata test ──")
    # try:
    #     meta = get_instagram_metadata(TEST_IG_URL)
    #     print(json.dumps(meta, indent=2))
    # except Exception as e:
    #     print(f"FAILED: {e}", file=sys.stderr)