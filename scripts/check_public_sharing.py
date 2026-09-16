#!/usr/bin/env python3
"""Fail fast if public-sharing privacy invariants are violated.

This check is intentionally independent of the experiment. It only inspects the
shareable project tree and provenance CSV schemas; it does not alter labels,
methodology, source experimental logic, or reported results.

When the project is a Git checkout, tracked files are inspected with ``git
ls-files``. When the project is an exported/downloaded ZIP with no Git metadata
(or Git is unavailable), the extracted filesystem tree is inspected instead.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
PROVENANCE = ROOT / "provenance"

FORBIDDEN_COLUMN_FRAGMENTS = (
    "user_text",
    "assistant_text",
    "text",
    "evidence",
    "note",
    "prompt",
    "response",
    "conversation",
    "message",
)

ALLOWED_PROVENANCE_COLUMNS = {
    "audit_number",
    "source_index",
    "pair_index",
    "weak_labels",
    "manual_audit_valid",
    "manual_missed_positive",
    "manual_missed_labels",
    "manual_relevant",
    "manual_category",
}

EXPECTED_PROVENANCE_FILES = {
    "manual_relevance_decisions_200.csv",
    "scaled_positive_audit_decisions_34.csv",
    "full_positive_audit_decisions_56.csv",
    "full_negative_audit_decisions_100.csv",
}


def shareable_paths() -> tuple[list[str], str]:
    """Return project paths using Git when available, otherwise the ZIP tree."""
    git_dir = ROOT / ".git"
    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            paths = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            return paths, "tracked Git files"
        except (FileNotFoundError, subprocess.CalledProcessError):
            # A reviewer may have a Git checkout but no Git executable available.
            # Fall back to the actual shareable filesystem tree.
            pass

    paths = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".venv" not in path.parts
        and ".cleanroom_runs" not in path.parts
        and "__pycache__" not in path.parts
    ]
    return sorted(paths), "extracted ZIP/project files"


def fail(message: str) -> None:
    raise SystemExit(f"PUBLIC-SHARING PRIVACY CHECK FAILED: {message}")


def check_samples(paths: list[str]) -> None:
    samples = [p for p in paths if p.startswith("data/samples/")]
    unexpected = [p for p in samples if p != "data/samples/.gitkeep"]
    if unexpected:
        fail("data/samples files are not allowed: " + ", ".join(unexpected))


def check_provenance(paths: list[str]) -> None:
    provenance_files = {
        Path(p).name
        for p in paths
        if p.startswith("provenance/") and p.endswith(".csv")
    }
    missing = EXPECTED_PROVENANCE_FILES - provenance_files
    extra = provenance_files - EXPECTED_PROVENANCE_FILES
    if missing:
        fail(
            "missing expected decision-only provenance files: "
            + ", ".join(sorted(missing))
        )
    if extra:
        fail("unexpected provenance CSVs present: " + ", ".join(sorted(extra)))

    for name in sorted(EXPECTED_PROVENANCE_FILES):
        path = PROVENANCE / name
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                fail(f"{name} is empty")

        normalized = [column.strip() for column in header]
        lowered = [column.lower() for column in normalized]

        forbidden = [
            column
            for column in lowered
            if any(fragment in column for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
        ]
        if forbidden:
            fail(
                f"{name} contains forbidden text/evidence/note columns: {forbidden}"
            )

        unknown = [
            column
            for column in normalized
            if column not in ALLOWED_PROVENANCE_COLUMNS
        ]
        if unknown:
            fail(
                f"{name} contains non identifier/category/decision columns: {unknown}"
            )


if __name__ == "__main__":
    paths, mode = shareable_paths()
    check_samples(paths)
    check_provenance(paths)
    print("PUBLIC-SHARING PRIVACY CHECK PASSED")
    print("Inspection mode:", mode)
    print("data/samples contains only .gitkeep in the shareable tree.")
    print("Provenance CSVs contain only identifier/category/decision columns.")
