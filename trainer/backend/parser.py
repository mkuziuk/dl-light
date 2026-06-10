from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


QUESTION_FILE_RE = re.compile(r"^(?P<number>\d{2}) - (?P<title>.+)\.md$")
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
MERMAID_RE = re.compile(r"```mermaid\s*\n.*?\n```", re.DOTALL)
IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")

REQUIRED_SECTIONS = (
    "Главная идея",
    "Минимум для ответа",
    "Формулы / схема",
    "Диаграмма",
    "Уточнения экзаменатора",
    "Частые ошибки",
)


@dataclass(frozen=True)
class QuestionNote:
    number: int
    title: str
    topic: str
    path: str
    source: str | None
    original_question: str
    sections: dict[str, str]
    section_order: list[str]
    mermaid_count: int
    assets: list[str]

    def summary(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "topic": self.topic,
            "path": self.path,
            "source": self.source,
            "original_question": self.original_question,
            "mermaid_count": self.mermaid_count,
            "assets": self.assets,
        }

    def full(self) -> dict:
        data = self.summary()
        data["sections"] = self.sections
        data["section_order"] = self.section_order
        return data


@dataclass(frozen=True)
class QuestionRecord:
    number: int
    title: str
    topic: str
    path: str
    source: str | None
    original_question: str
    versions: dict[str, QuestionNote]

    def summary(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "topic": self.topic,
            "path": self.path,
            "source": self.source,
            "original_question": self.original_question,
            "answer_versions": list(self.versions),
        }

    def full(self) -> dict:
        data = self.summary()
        data["versions"] = {
            version: note.full() for version, note in self.versions.items()
        }
        return data


ANSWER_ROOTS = {
    "light": "light-answers",
    "complete": "complete-answers",
}


def find_question_files(answer_root: Path) -> list[Path]:
    files: list[Path] = []
    for topic_dir in answer_root.glob("Topic * - *"):
        if not topic_dir.is_dir():
            continue
        for path in topic_dir.glob("*.md"):
            if QUESTION_FILE_RE.match(path.name):
                files.append(path)
    return sorted(files, key=lambda path: int(path.name[:2]))


def parse_all_notes(vault_root: Path) -> list[QuestionRecord]:
    notes_by_version: dict[str, dict[int, QuestionNote]] = {}
    for version, folder_name in ANSWER_ROOTS.items():
        answer_root = vault_root / folder_name
        notes_by_version[version] = {
            note.number: note
            for note in (
                parse_note(path, vault_root)
                for path in find_question_files(answer_root)
            )
        }

    light_notes = notes_by_version.get("light", {})
    records: list[QuestionRecord] = []
    for number in sorted(light_notes):
        light_note = light_notes[number]
        versions = {
            version: notes[number]
            for version, notes in notes_by_version.items()
            if number in notes
        }
        records.append(
            QuestionRecord(
                number=number,
                title=light_note.title,
                topic=light_note.topic,
                path=light_note.path,
                source=light_note.source,
                original_question=light_note.original_question,
                versions=versions,
            )
        )
    return records


def parse_note(path: Path, vault_root: Path) -> QuestionNote:
    match = QUESTION_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a question note: {path}")

    text = path.read_text(encoding="utf-8")
    number = int(match.group("number"))
    title = _extract_title(text) or match.group("title")
    sections = _extract_sections(text)
    source = _extract_source(text)
    original_question = _extract_original_question(text)
    assets = _extract_assets(text)

    return QuestionNote(
        number=number,
        title=title,
        topic=path.parent.name,
        path=str(path.relative_to(vault_root)),
        source=source,
        original_question=original_question,
        sections=sections,
        section_order=list(sections),
        mermaid_count=len(MERMAID_RE.findall(text)),
        assets=assets,
    )


def validate_notes(notes: list[QuestionRecord]) -> list[str]:
    errors: list[str] = []
    numbers = [note.number for note in notes]
    if len(notes) != 62:
        errors.append(f"Expected 62 question notes, found {len(notes)}")
    if numbers != sorted(numbers):
        errors.append("Question notes are not sorted by number")
    if len(set(numbers)) != len(numbers):
        errors.append("Duplicate question numbers found")

    for note in notes:
        missing_versions = [
            version for version in ANSWER_ROOTS if version not in note.versions
        ]
        if missing_versions:
            errors.append(
                f"Question {note.number:02d} missing versions: {', '.join(missing_versions)}"
            )
            continue

        light_note = note.versions["light"]
        complete_note = note.versions["complete"]
        if light_note.topic != complete_note.topic:
            errors.append(
                f"Question {note.number:02d} topic mismatch between answer versions"
            )
        if Path(light_note.path).name != Path(complete_note.path).name:
            errors.append(
                f"Question {note.number:02d} filename mismatch between answer versions"
            )

        missing = [
            section for section in REQUIRED_SECTIONS if section not in light_note.sections
        ]
        if missing:
            errors.append(f"Question {note.number:02d} missing sections: {', '.join(missing)}")
        if not light_note.original_question:
            errors.append(f"Question {note.number:02d} has no original question")
        if light_note.mermaid_count != 1:
            errors.append(
                f"Question {note.number:02d} has {light_note.mermaid_count} Mermaid blocks"
            )
        if not complete_note.original_question:
            errors.append(f"Question {note.number:02d} complete answer has no original question")
        if not complete_note.sections:
            errors.append(f"Question {note.number:02d} complete answer has no sections")
    return errors


def _extract_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _extract_source(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("Source:"):
            return line.removeprefix("Source:").strip()
    return None


def _extract_original_question(text: str) -> str:
    marker = "Original question:"
    start = text.find(marker)
    if start == -1:
        return ""
    rest = text[start + len(marker) :]
    next_heading = re.search(r"^##\s+", rest, re.MULTILINE)
    block = rest[: next_heading.start()] if next_heading else rest
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            stripped = stripped[1:].strip()
        lines.append(stripped)
    return " ".join(lines).strip()


def _extract_sections(text: str) -> dict[str, str]:
    matches = list(HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        section_name = match.group(1).strip()
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[section_name] = text[section_start:section_end].strip()
    return sections


def _extract_assets(text: str) -> list[str]:
    assets: list[str] = []
    for match in IMAGE_RE.finditer(text):
        target = match.group(1).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        assets.append(target)
    return assets
