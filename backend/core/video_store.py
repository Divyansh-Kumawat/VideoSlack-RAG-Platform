# backend/core/video_store.py
# ─────────────────────────────────────────────────────────────────────────────
# Persists the metadata of the currently ingested videos (A and B) as JSON.
#
# The vector store only holds transcript chunks. The stats (views, likes,
# engagement rate, subscribers…) live here so that:
#   - the RAG prompt can always include hard numbers, and
#   - the frontend can restore the video cards after a page reload.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from backend import config
from backend.models.schemas import VideoMetadata

logger = logging.getLogger(__name__)
_lock = threading.Lock()


def _path() -> Path:
    return config.DATA_DIR / "videos.json"


def _read() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", p, e)
        return {}


def save_videos(videos: list[VideoMetadata]) -> None:
    with _lock:
        data = _read()
        for v in videos:
            data[v.video_id.value] = v.model_dump(mode="json")
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


def get_video(video_id: str) -> Optional[VideoMetadata]:
    raw = _read().get(video_id)
    if not raw:
        return None
    try:
        return VideoMetadata.model_validate(raw)
    except Exception as e:
        logger.warning("Stored metadata for video %s is invalid: %s", video_id, e)
        return None


def get_videos() -> dict[str, Optional[VideoMetadata]]:
    return {"A": get_video("A"), "B": get_video("B")}


def clear_videos() -> None:
    with _lock:
        p = _path()
        if p.exists():
            p.unlink()
