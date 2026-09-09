#!/usr/bin/env python3
"""Portable clean-room runner for the frozen IRP notebooks.

The historical notebooks are never edited.  This runner executes temporary,
in-memory copies, replays decision-only human-audit provenance, skips historical
environment-repair cells, and rewrites only historical absolute Codespaces paths
inside those temporary copies so the artefact can run from any extracted project
directory on Windows, macOS, Linux, Codespaces, or GitHub Actions.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
RUN_DIR = PROJECT_ROOT / ".cleanroom_runs"
KERNEL_NAME = "irp-cleanroom"
HISTORICAL_PROJECT_ROOTS = ("/workspaces/irp-disempowerment-nlp",)

_LEGACY_PATH = Path(__file__).with_name("_run_cleanroom_legacy.py")
_spec = importlib.util.spec_from_file_location("_irp_cleanroom_legacy", _LEGACY_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load clean-room compatibility helper: {_LEGACY_PATH}")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)

NOTEBOOKS = legacy.NOTEBOOKS


def rewrite_historical_paths(notebook) -> int:
    """Rewrite historical absolute project paths only in the temporary copy."""
    replacement = PROJECT_ROOT.as_posix()
    changed = 0
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        updated = source
        for historical_root in HISTORICAL_PROJECT_ROOTS:
            updated = updated.replace(historical_root, replacement)
        if updated != source:
            cell["source"] = updated
            changed += 1
    return changed


def repair_generated_compatibility_source(notebook) -> int:
    """Repair runner-generated source only; frozen notebook source is untouched."""
    changed = 0
    malformed = 'print("\nAudit files verified.")'
    replacement = 'print()\nprint("Audit files verified.")'
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if malformed in source:
            cell["source"] = source.replace(malformed, replacement)
            changed += 1
    return changed


def check_environment() -> None:
    print("[setup] Checking installed package consistency...")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def register_kernel() -> None:
    print("\n[setup] Registering the current virtual environment as a Jupyter kernel...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            "Python (IRP clean-room)",
        ],
        check=True,
    )


def execute_notebook(name: str) -> None:
    source_path = NOTEBOOK_DIR / name
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    audit_adjustments = 0
    if name == "02_filtering_and_sampling.ipynb":
        audit_adjustments = legacy.prepare_notebook02(notebook)
    elif name == "06_scaled_experiment.ipynb":
        audit_adjustments = legacy.prepare_notebook06(notebook)

    generated_repairs = repair_generated_compatibility_source(notebook)
    path_rewrites = rewrite_historical_paths(notebook)

    original_count = len(notebook.cells)
    notebook.cells = [
        cell for cell in notebook.cells if not legacy.is_environment_mutation_cell(cell)
    ]
    environment_cells_skipped = original_count - len(notebook.cells)

    print("\n" + "=" * 78)
    print(f"Running {name}")
    print("=" * 78)
    if audit_adjustments:
        print(
            f"Applied {audit_adjustments} clean-room audit-provenance adjustment(s); "
            "decision-only human audit data is used."
        )
    if path_rewrites:
        print(
            f"Rewrote historical absolute project paths in {path_rewrites} temporary "
            "notebook cell(s)."
        )
    if generated_repairs:
        print(f"Applied {generated_repairs} runner compatibility repair(s).")
    if environment_cells_skipped:
        print(
            f"Skipped {environment_cells_skipped} historical environment-repair "
            "cell(s). The locked environment is used instead."
        )

    executor = ExecutePreprocessor(
        timeout=None,
        kernel_name=KERNEL_NAME,
        allow_errors=False,
    )
    executor.preprocess(notebook, {"metadata": {"path": str(PROJECT_ROOT)}})

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUN_DIR / name
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(f"PASS: {name}")
    print(f"Executed copy: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute notebooks 01-07 in documented clean-room order."
    )
    parser.add_argument("--start", type=int, default=1, choices=range(1, 8), metavar="N")
    parser.add_argument("--end", type=int, default=7, choices=range(1, 8), metavar="N")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    check_environment()
    register_kernel()

    for notebook_name in NOTEBOOKS[args.start - 1 : args.end]:
        execute_notebook(notebook_name)

    if not args.no_verify and args.end == 7:
        print("\n" + "=" * 78)
        print("Running full reproduction verification")
        print("=" * 78)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "verify_reproduction.py"),
                "--full",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print("\nClean-room notebook execution completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
