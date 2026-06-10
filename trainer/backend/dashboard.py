from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean
from typing import Any

from .parser import QuestionNote


def build_dashboard(notes: list[QuestionNote], progress: dict[str, Any]) -> dict[str, Any]:
    question_progress = progress.get("questions", {})
    merged = [_merge_progress(note, question_progress.get(str(note.number), {})) for note in notes]

    rated = [item for item in merged if item["progress"].get("rating")]
    ratings = [item["progress"]["rating"] for item in rated]
    topics = _topic_coverage(merged)
    weakest = sorted(merged, key=_priority_score, reverse=True)[:12]
    suggested = _balanced_queue(merged, limit=12)

    return {
        "summary": {
            "total_questions": len(merged),
            "rated_questions": len(rated),
            "unrated_questions": len(merged) - len(rated),
            "average_rating": round(mean(ratings), 2) if ratings else None,
            "updated_at": progress.get("updated_at"),
        },
        "weakest_questions": [_compact_question(item) for item in weakest],
        "topic_coverage": topics,
        "suggested_queue": [_compact_question(item) for item in suggested],
    }


def merge_note_progress(note: QuestionNote, progress: dict[str, Any]) -> dict[str, Any]:
    return _merge_progress(note, progress.get("questions", {}).get(str(note.number), {}))


def _merge_progress(note: QuestionNote, progress: dict[str, Any]) -> dict[str, Any]:
    data = note.summary()
    data["progress"] = {
        "rating": progress.get("rating"),
        "review_count": progress.get("review_count", 0),
        "last_reviewed": progress.get("last_reviewed"),
        "note": progress.get("note", ""),
    }
    data["priority_score"] = round(_priority_score(data), 3)
    return data


def _compact_question(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": item["number"],
        "title": item["title"],
        "topic": item["topic"],
        "original_question": item["original_question"],
        "progress": item["progress"],
        "priority_score": item["priority_score"],
    }


def _topic_coverage(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["topic"]].append(item)

    coverage = []
    for topic, topic_items in groups.items():
        ratings = [
            item["progress"]["rating"]
            for item in topic_items
            if item["progress"].get("rating")
        ]
        weak_count = sum(
            1
            for item in topic_items
            if not item["progress"].get("rating") or item["progress"]["rating"] <= 2
        )
        rated_count = len(ratings)
        total = len(topic_items)
        coverage.append(
            {
                "topic": topic,
                "total": total,
                "rated": rated_count,
                "unrated": total - rated_count,
                "weak_count": weak_count,
                "average_rating": round(mean(ratings), 2) if ratings else None,
                "coverage_percent": round((rated_count / total) * 100),
            }
        )
    return sorted(coverage, key=lambda row: row["topic"])


def _balanced_queue(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=_priority_score, reverse=True):
        by_topic[item["topic"]].append(item)

    topic_order = sorted(
        by_topic,
        key=lambda topic: (
            _topic_need_score(by_topic[topic]),
            by_topic[topic][0]["topic"],
        ),
        reverse=True,
    )

    queue: list[dict[str, Any]] = []
    while len(queue) < limit:
        added = False
        for topic in topic_order:
            if by_topic[topic]:
                queue.append(by_topic[topic].pop(0))
                added = True
                if len(queue) == limit:
                    break
        if not added:
            break
    return queue


def _topic_need_score(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return mean(_priority_score(item) for item in items)


def _priority_score(item: dict[str, Any]) -> float:
    progress = item.get("progress", {})
    rating = progress.get("rating")
    review_count = progress.get("review_count") or 0
    last_reviewed = progress.get("last_reviewed")

    if rating is None:
        score = 7.0
    else:
        score = float(6 - rating)
        if rating <= 2:
            score += 1.5

    score += _staleness_score(last_reviewed)
    if review_count == 0:
        score += 1.0
    return score


def _staleness_score(last_reviewed: str | None) -> float:
    if not last_reviewed:
        return 0.0
    try:
        reviewed_at = datetime.fromisoformat(last_reviewed)
    except ValueError:
        return 0.0
    if reviewed_at.tzinfo is None:
        reviewed_at = reviewed_at.replace(tzinfo=UTC)
    days = max((datetime.now(UTC) - reviewed_at).days, 0)
    return min(days / 14, 2.5)
