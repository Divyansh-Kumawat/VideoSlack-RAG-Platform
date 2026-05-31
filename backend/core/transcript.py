# backend/core/transcript.py
# ─────────────────────────────────────────────────────────────────────────────
# Transcript extraction for YouTube and Instagram Reels.
#
# YouTube  → youtube-transcript-api (no download, instant, no API key)
# Instagram → yt-dlp downloads audio → Whisper transcribes locally
#
# Both paths return: { "text": str, "language": str }
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import whisper
import os
import re
import uuid
import logging
import tempfile
from pathlib import Path
from typing import TypedDict


import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from youtube_transcript_api.formatters import TextFormatter


load_dotenv()

logger = logging.getLogger(__name__)

# ── Config from .env ──────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
WHISPER_TEMP_DIR   = Path(os.getenv("WHISPER_TEMP_DIR", "./tmp_audio"))
WHISPER_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Whisper model — loaded once at module level (expensive) ───────────────────
_whisper_model: whisper.Whisper | None = None

def _get_whisper_model() -> whisper.Whisper:
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", WHISPER_MODEL_SIZE)
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptResult(TypedDict):
    text: str        # clean plain-text transcript
    language: str    # ISO 639-1 language code e.g. "en", "hi", "es"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_youtube_video_id(url: str) -> str:
    """
    Extracts the 11-character video ID from any YouTube URL format:
      - https://www.youtube.com/watch?v=XXXXXXXXXXX
      - https://youtu.be/XXXXXXXXXXX
      - https://youtube.com/shorts/XXXXXXXXXXX
    """
    patterns = [
        r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract YouTube video ID from URL: {url}")


def _clean_transcript_text(text: str) -> str:
    """
    Removes filler artifacts left by auto-generated captions:
    - [Music], [Applause], (inaudible) etc.
    - Excess whitespace and newlines.
    """
    # Remove caption annotations like [Music], [Laughter], (inaudible)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# YouTube transcript extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_youtube_transcript(url: str) -> TranscriptResult:
    """
    Fetches transcript for a YouTube video using youtube-transcript-api.

    Strategy:
    1. Try to get a manually created transcript in English.
    2. Fall back to any manually created transcript (any language).
    3. Fall back to auto-generated transcript.
    4. Raise RuntimeError if nothing is available.

    No API key required. No download needed.
    """
    video_id = _extract_youtube_video_id(url)
    logger.info("Fetching YouTube transcript for video_id=%s", video_id)

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except VideoUnavailable:
        raise RuntimeError(f"YouTube video is unavailable or private: {url}")
    except TranscriptsDisabled:
        raise RuntimeError(f"Transcripts are disabled for this YouTube video: {url}")

    # Priority order: manual EN → any manual → any generated
    transcript = None
    language_code = "en"

    try:
        transcript = transcript_list.find_manually_created_transcript(["en"])
        language_code = "en"
        logger.debug("Found manual English transcript")
    except NoTranscriptFound:
        pass

    if transcript is None:
        try:
            # Get any manually created transcript and translate if needed
            transcript = transcript_list.find_manually_created_transcript(
                [t.language_code for t in transcript_list]
            )
            language_code = transcript.language_code
            logger.debug("Found manual transcript in language: %s", language_code)
        except NoTranscriptFound:
            pass

    if transcript is None:
        try:
            # Last resort: auto-generated
            transcript = transcript_list.find_generated_transcript(
                [t.language_code for t in transcript_list]
            )
            language_code = transcript.language_code
            logger.debug("Found auto-generated transcript in language: %s", language_code)
        except NoTranscriptFound:
            raise RuntimeError(
                f"No transcript available for YouTube video: {url}. "
                "The video may have no captions enabled."
            )

    # Fetch and format
    fetched = transcript.fetch()
    formatter = TextFormatter()
    raw_text = formatter.format_transcript(fetched)
    clean_text = _clean_transcript_text(raw_text)

    if not clean_text:
        raise RuntimeError(f"Transcript fetched but was empty after cleaning: {url}")

    logger.info(
        "YouTube transcript ready: %d characters, language=%s",
        len(clean_text), language_code
    )
    return TranscriptResult(text=clean_text, language=language_code)


# ─────────────────────────────────────────────────────────────────────────────
# Instagram Reels transcript extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_instagram_transcript(url: str) -> TranscriptResult:
    """
    Downloads the audio from an Instagram Reel using yt-dlp,
    then transcribes it locally using OpenAI Whisper.

    Steps:
    1. yt-dlp downloads the reel audio as .mp3 to WHISPER_TEMP_DIR
    2. Whisper transcribes the .mp3 (model loaded once, reused)
    3. Temp file is deleted regardless of success/failure
    4. Returns clean text + detected language

    Requires: ffmpeg on system PATH (for yt-dlp audio conversion)
    """
    logger.info("Downloading Instagram Reel audio: %s", url)

    # Unique filename to avoid collisions in concurrent requests
    tmp_filename = WHISPER_TEMP_DIR / f"reel_{uuid.uuid4().hex}"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp_filename),           # yt-dlp adds extension automatically
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",       # 128kbps — sufficient for speech
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "cookiefile": None,                      # Add path to cookies.txt if needed for private reels
    }

    audio_path = Path(str(tmp_filename) + ".mp3")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("yt-dlp returned no info for this URL")

        if not audio_path.exists():
            raise RuntimeError(
                f"yt-dlp did not produce an audio file at {audio_path}. "
                "Ensure ffmpeg is installed and on PATH."
            )

        logger.info("Audio downloaded to %s (%.1f KB)", audio_path, audio_path.stat().st_size / 1024)

        # Transcribe with Whisper
        model = _get_whisper_model()
        logger.info("Transcribing with Whisper model=%s ...", WHISPER_MODEL_SIZE)

        result = model.transcribe(
            str(audio_path),
            verbose=False,
            fp16=False,                          # fp16=False for CPU safety; set True on GPU
        )

        raw_text: str = result.get("text", "")
        language_code: str = result.get("language", "unknown")

        clean_text = _clean_transcript_text(raw_text)

        if not clean_text:
            raise RuntimeError(
                "Whisper produced an empty transcript. "
                "The reel may be music-only or have no speech."
            )

        logger.info(
            "Instagram transcript ready: %d characters, language=%s",
            len(clean_text), language_code
        )
        return TranscriptResult(text=clean_text, language=language_code)

    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(
            f"yt-dlp failed to download the Instagram Reel: {e}. "
            "The reel may be private, deleted, or geo-restricted."
        ) from e

    finally:
        # Always clean up temp audio file
        if audio_path.exists():
            audio_path.unlink()
            logger.debug("Cleaned up temp audio file: %s", audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_transcript(url: str, platform: str) -> TranscriptResult:
    """
    Unified dispatcher. Call this from the ingestion pipeline.

    Args:
        url:      The video URL (YouTube or Instagram)
        platform: "youtube" or "instagram"

    Returns:
        TranscriptResult with keys: text (str), language (str)

    Raises:
        ValueError:   Unknown platform
        RuntimeError: Extraction failed (with human-readable reason)
    """
    platform = platform.lower().strip()

    if platform == "youtube":
        return get_youtube_transcript(url)
    elif platform in ("instagram", "instagram_reel"):
        return get_instagram_transcript(url)
    else:
        raise ValueError(
            f"Unknown platform '{platform}'. Expected 'youtube' or 'instagram'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — run: python -m backend.core.transcript
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Quick smoke test with a known public video
    TEST_YT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    print("\n── YouTube transcript test ──")
    try:
        result = get_youtube_transcript(TEST_YT_URL)
        print(f"Language : {result['language']}")
        print(f"Length   : {len(result['text'])} chars")
        print(f"Preview  : {result['text'][:300]}")
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)

    # Uncomment and replace with a real public reel URL to test Instagram
    # TEST_IG_URL = "https://www.instagram.com/reel/XXXXXXXXXXXX/"
    # print("\n── Instagram transcript test ──")
    # try:
    #     result = get_instagram_transcript(TEST_IG_URL)
    #     print(f"Language : {result['language']}")
    #     print(f"Length   : {len(result['text'])} chars")
    #     print(f"Preview  : {result['text'][:300]}")
    # except Exception as e:
    #     print(f"FAILED: {e}", file=sys.stderr)