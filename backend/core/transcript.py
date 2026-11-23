# backend/core/transcript.py
# ─────────────────────────────────────────────────────────────────────────────
# Transcript extraction for YouTube videos.
#
# Strategy:
#   1. Captions via youtube-transcript-api (fast, no download, no API key).
#      Preference: manual English → auto English → any manual → any auto.
#   2. Fallback when no captions exist (common for Shorts): download the audio
#      with yt-dlp and transcribe it locally with Whisper. Needs ffmpeg on PATH.
#
# Returns timestamped segments, so retrieved chunks can cite "Video A @ 0:42".
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, TypedDict, cast

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from backend import config
from backend.models.schemas import extract_youtube_id

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

class Segment(TypedDict):
    text:  str
    start: float   # seconds from video start


class TranscriptResult(TypedDict):
    text:     str             # full clean transcript
    language: str             # ISO 639-1 code, e.g. "en"
    source:   str             # "captions" | "whisper"
    segments: list[Segment]   # timestamped pieces of `text`


class TranscriptError(RuntimeError):
    """Raised with a human-readable reason when no transcript can be produced."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_ANNOTATION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")   # [Music], (applause)
_WS_RE         = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Removes caption annotations like [Music] / (inaudible) and collapses whitespace."""
    text = _ANNOTATION_RE.sub(" ", text or "")
    text = text.replace(">>", " ")
    return _WS_RE.sub(" ", text).strip()


def _build_result(raw_segments: list[tuple[str, float]], language: str, source: str) -> TranscriptResult:
    segments: list[Segment] = []
    for text, start in raw_segments:
        t = clean_text(text)
        if t:
            segments.append(Segment(text=t, start=round(float(start), 2)))
    full = " ".join(s["text"] for s in segments)
    return TranscriptResult(text=full, language=language, source=source, segments=segments)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Captions
# ─────────────────────────────────────────────────────────────────────────────

def _pick_transcript(transcript_list):
    """Manual English → generated English → any manual → any generated."""
    try:
        return transcript_list.find_transcript(["en", "en-US", "en-GB", "en-IN"])
    except NoTranscriptFound:
        pass
    available = list(transcript_list)
    manual = [t for t in available if not t.is_generated]
    generated = [t for t in available if t.is_generated]
    if manual or generated:
        return (manual or generated)[0]
    raise NoTranscriptFound(transcript_list.video_id, ["en"], transcript_list)


def get_caption_transcript(youtube_id: str) -> TranscriptResult:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(youtube_id)
    transcript = _pick_transcript(transcript_list)
    fetched = transcript.fetch()
    result = _build_result(
        [(s.text, s.start) for s in fetched.snippets],
        language=transcript.language_code.split("-")[0],
        source="captions",
    )
    logger.info(
        "Captions for %s: %d segments, %d chars, lang=%s, generated=%s",
        youtube_id, len(result["segments"]), len(result["text"]),
        result["language"], transcript.is_generated,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Whisper fallback
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper   # heavy import — only when actually needed

        logger.info("Loading Whisper model: %s", config.WHISPER_MODEL)
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
    return _whisper_model


def get_whisper_transcript(url: str) -> TranscriptResult:
    """Downloads the audio track with yt-dlp and transcribes it with Whisper."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    config.WHISPER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"yt_{uuid.uuid4().hex}"
    ydl_opts: Any = {
        "format": "bestaudio/best",
        "outtmpl": str(config.WHISPER_TEMP_DIR / f"{stem}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
    }

    try:
        logger.info("Downloading audio for Whisper: %s", url)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
        except DownloadError as e:
            raise TranscriptError(f"Could not download the video's audio: {e}") from e

        files = list(config.WHISPER_TEMP_DIR.glob(f"{stem}.*"))
        if not files:
            raise TranscriptError("Audio download produced no file.")

        logger.info("Transcribing %s with Whisper (%s)…", files[0].name, config.WHISPER_MODEL)
        try:
            result = _get_whisper_model().transcribe(str(files[0]), fp16=False, verbose=None)
        except FileNotFoundError as e:
            raise TranscriptError(
                "Whisper needs ffmpeg installed and on PATH to decode audio."
            ) from e

        result = cast(dict[str, Any], result)
        language = result.get("language") or "unknown"
        if isinstance(language, list):
            language = language[0] if language else "unknown"
        transcript = _build_result(
            [(str(seg["text"]), float(seg["start"])) for seg in result.get("segments", [])],
            language=str(language),
            source="whisper",
        )
        logger.info("Whisper transcript: %d chars, lang=%s", len(transcript["text"]), language)
        return transcript
    finally:
        for f in config.WHISPER_TEMP_DIR.glob(f"{stem}.*"):
            try:
                f.unlink()
            except OSError:
                logger.warning("Could not delete temp file %s", f)


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_transcript(url: str) -> TranscriptResult:
    """
    Returns the transcript of a YouTube video.

    Raises:
        TranscriptError with a human-readable reason if nothing works.
    """
    youtube_id = extract_youtube_id(url)
    logger.info("Fetching transcript for %s", youtube_id)

    caption_problem: str
    try:
        result = get_caption_transcript(youtube_id)
        if result["text"]:
            return result
        caption_problem = "captions were empty"
    except VideoUnavailable as e:
        raise TranscriptError(f"YouTube video {youtube_id} is unavailable or private.") from e
    except (TranscriptsDisabled, NoTranscriptFound):
        caption_problem = "the video has no captions"
    except CouldNotRetrieveTranscript as e:
        caption_problem = f"captions could not be retrieved ({type(e).__name__})"

    if not config.WHISPER_ENABLED:
        raise TranscriptError(
            f"No transcript for {youtube_id}: {caption_problem}. "
            "Enable WHISPER_ENABLED=true to transcribe the audio instead."
        )

    logger.info("No usable captions for %s (%s) — falling back to Whisper.", youtube_id, caption_problem)
    result = get_whisper_transcript(url)
    if not result["text"]:
        raise TranscriptError(
            f"No transcript for {youtube_id}: {caption_problem}, and Whisper heard no speech "
            "(the video may be music-only)."
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.transcript <url>
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    r = get_transcript(test_url)
    print(f"source={r['source']} lang={r['language']} chars={len(r['text'])} segments={len(r['segments'])}")
    print(r["text"][:300])
