from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


EMPTY_PROGRESS = {"version": 1, "updated_at": None, "questions": {}}


class ProgressStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return dict(EMPTY_PROGRESS)
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("version", 1)
        data.setdefault("updated_at", None)
        data.setdefault("questions", {})
        return data

    def question_progress(self, number: int) -> dict[str, Any]:
        data = self.read()
        return data["questions"].get(str(number), {})

    def update_rating(self, number: int, rating: int, note: str | None = None) -> dict[str, Any]:
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")

        data = self.read()
        now = datetime.now(UTC).isoformat()
        key = str(number)
        current = data["questions"].setdefault(
            key,
            {
                "rating": None,
                "review_count": 0,
                "last_reviewed": None,
                "history": [],
                "note": "",
            },
        )

        current["rating"] = rating
        current["review_count"] = int(current.get("review_count") or 0) + 1
        current["last_reviewed"] = now
        if note is not None:
            current["note"] = note
        current.setdefault("history", []).append({"rating": rating, "reviewed_at": now})
        data["updated_at"] = now
        self.write(data)
        return current

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(self.path)
