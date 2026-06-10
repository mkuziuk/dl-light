from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from .dashboard import build_dashboard
from .parser import parse_all_notes, validate_notes
from .progress import ProgressStore


def main() -> int:
    vault_root = Path(__file__).resolve().parents[2]
    notes = parse_all_notes(vault_root)
    errors = validate_notes(notes)
    progress = ProgressStore(vault_root / "trainer" / "data" / "progress.json").read()
    dashboard = build_dashboard(notes, progress)
    vendor_files = [
        vault_root / "trainer" / "frontend" / "vendor" / "mathjax" / "tex-svg-nofont.js",
        vault_root / "trainer" / "frontend" / "vendor" / "mermaid" / "mermaid.min.js",
        vault_root / "trainer" / "data" / "progress.example.json",
    ]

    if dashboard["summary"]["total_questions"] != 62:
        errors.append("Dashboard question count is not 62")
    if len(dashboard["topic_coverage"]) != 18:
        errors.append("Dashboard topic count is not 18")
    if not dashboard["suggested_queue"]:
        errors.append("Dashboard suggested queue is empty")
    for path in vendor_files:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"Missing required app asset: {path.relative_to(vault_root)}")
    with TemporaryDirectory() as temp_dir:
        temp_progress = ProgressStore(Path(temp_dir) / "progress.json")
        updated = temp_progress.update_rating(32, 4)
        reread = temp_progress.question_progress(32)
        if updated["rating"] != 4 or reread.get("review_count") != 1:
            errors.append("Progress rating write/read check failed")

    if errors:
        print("Checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Checks passed:")
    print(f"- question notes: {dashboard['summary']['total_questions']}")
    print(f"- topics: {len(dashboard['topic_coverage'])}")
    print(f"- suggested queue: {len(dashboard['suggested_queue'])}")
    print("- progress write/read: ok")
    print("- required app assets: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
