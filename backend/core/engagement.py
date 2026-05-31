# backend/core/engagement.py
# ─────────────────────────────────────────────────────────────────────────────
# Engagement rate computation.
#
# Formula: (likes + comments) / views × 100
# Isolated as a pure function — zero dependencies, always testable.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_engagement_rate(
    likes: int,
    comments: int,
    views: int,
    precision: int = 2,
) -> float:
    """
    Computes engagement rate as a percentage.

    Formula: (likes + comments) / views × 100

    Args:
        likes:     Total like count
        comments:  Total comment count
        views:     Total view count
        precision: Decimal places to round to (default 2)

    Returns:
        Engagement rate as float (e.g. 4.73 means 4.73%).
        Returns 0.0 if views is 0 to avoid ZeroDivisionError.

    Examples:
        >>> compute_engagement_rate(1000, 200, 50000)
        2.4
        >>> compute_engagement_rate(0, 0, 0)
        0.0
    """
    if views <= 0:
        logger.warning(
            "Views is %d — cannot compute engagement rate. Returning 0.0.", views
        )
        return 0.0

    rate = (likes + comments) / views * 100
    return round(rate, precision)


def compute_engagement_rate_from_metadata(metadata: dict[str, Any]) -> float:
    """
    Convenience wrapper — pass a raw metadata dict directly.

    Expects keys: 'likes', 'comments', 'views' (all int).
    Safe to call with partial dicts — missing keys default to 0.

    Args:
        metadata: Dict returned by metadata.py (or VideoMetadata.dict())

    Returns:
        Engagement rate float.
    """
    likes    = int(metadata.get("likes",    0))
    comments = int(metadata.get("comments", 0))
    views    = int(metadata.get("views",    0))

    rate = compute_engagement_rate(likes, comments, views)

    logger.debug(
        "Engagement rate: (likes=%d + comments=%d) / views=%d × 100 = %.2f%%",
        likes, comments, views, rate,
    )
    return rate


def compare_engagement(
    metadata_a: dict[str, Any],
    metadata_b: dict[str, Any],
) -> dict[str, Any]:
    """
    Compares engagement rates of two videos and returns a structured summary.
    Used by the RAG chain to ground comparative questions with hard numbers.

    Returns:
        {
            "rate_a": float,
            "rate_b": float,
            "winner": "A" | "B" | "tie",
            "difference": float,        # absolute percentage point difference
            "summary": str              # human-readable one-liner
        }
    """
    rate_a = compute_engagement_rate_from_metadata(metadata_a)
    rate_b = compute_engagement_rate_from_metadata(metadata_b)
    diff   = round(abs(rate_a - rate_b), 2)

    if rate_a > rate_b:
        winner = "A"
        summary = (
            f"Video A has a higher engagement rate ({rate_a}%) "
            f"than Video B ({rate_b}%) by {diff} percentage points."
        )
    elif rate_b > rate_a:
        winner = "B"
        summary = (
            f"Video B has a higher engagement rate ({rate_b}%) "
            f"than Video A ({rate_a}%) by {diff} percentage points."
        )
    else:
        winner = "tie"
        summary = f"Both videos have the same engagement rate ({rate_a}%)."

    return {
        "rate_a":     rate_a,
        "rate_b":     rate_b,
        "winner":     winner,
        "difference": diff,
        "summary":    summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — python -m backend.core.engagement
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    # Unit tests
    cases = [
        # (likes, comments, views, expected)
        (1000,  200,  50_000,  2.4),
        (5000,  800, 100_000,  5.8),
        (0,       0,  10_000,  0.0),
        (100,    10,       0,  0.0),   # edge: zero views
    ]

    print("── compute_engagement_rate unit tests ──")
    all_pass = True
    for likes, comments, views, expected in cases:
        result = compute_engagement_rate(likes, comments, views)
        status = "PASS" if result == expected else f"FAIL (got {result}, expected {expected})"
        print(f"  likes={likes}, comments={comments}, views={views} → {result}%  [{status}]")
        if "FAIL" in status:
            all_pass = False

    print(f"\nAll tests passed: {all_pass}")

    # Comparison test
    print("\n── compare_engagement test ──")
    meta_a = {"likes": 5000, "comments": 800, "views": 100_000}
    meta_b = {"likes": 1000, "comments": 200, "views":  50_000}
    comparison = compare_engagement(meta_a, meta_b)
    print(json.dumps(comparison, indent=2))