#!/usr/bin/env python3
"""
Execute the IRP notebooks in clean-room order in GitHub Codespaces or Actions.

The original notebooks preserve historical exploratory/environment-repair cells and
manual-audit provenance. This runner executes temporary copies, skips only cells
that must not be replayed in a clean-room environment, and never edits the source
notebooks or frozen experimental logic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
RUN_DIR = PROJECT_ROOT / ".cleanroom_runs"
KERNEL_NAME = "irp-cleanroom"

NOTEBOOKS = [
    "01_dataset_inspection.ipynb",
    "02_filtering_and_sampling.ipynb",
    "03_labelling_pilot.ipynb",
    "04_noise_and_preprocessing.ipynb",
    "05_model_training_evaluation.ipynb",
    "06_scaled_experiment.ipynb",
    "07_final_synthesis.ipynb",
]

ENVIRONMENT_MUTATION_MARKERS = (
    "Clean-reinstall current scikit-learn version",
    "pip uninstall",
    "pip install",
    "%pip install",
    "!pip install",
    "Restart the Jupyter kernel now",
)

NOTEBOOK02_RECOVERY_MARKER = "Successfully recovered labels:"
NOTEBOOK02_INTERACTIVE_CALLS = (
    "show_next_unreviewed()",
    "label_validation_example(",
    "show_validation_example(",
)
NOTEBOOK02_MANUAL_AUDIT_FILENAME = "lmsys_relevance_manual_validation_200.csv"


def is_environment_mutation_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False

    source = "".join(cell.get("source", []))
    lower_source = source.lower()

    if any(marker.lower() in lower_source for marker in ENVIRONMENT_MUTATION_MARKERS):
        return True

    return (
        "subprocess.check_call" in source
        and '"-m"' in source
        and '"pip"' in source
    )


def prepare_notebook02(notebook) -> int:
    """Prepare Notebook 02 for faithful sequential clean-room execution.

    Notebook 02 records an interrupted manual topical-relevance audit. The notebook
    later reconstructs successfully saved historical decisions from recorded outputs.
    A complete original audit CSV is now committed as provenance for clean-room use.
    The temporary runner therefore prevents the notebook's initial blank audit export
    from overwriting that committed provenance, while leaving filtering, recovery,
    sampling, and all frozen experimental logic unchanged.
    """
    recovery_index = None
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if NOTEBOOK02_RECOVERY_MARKER in source:
            recovery_index = index
            break

    if recovery_index is None:
        raise RuntimeError("Notebook 02 historical audit recovery block was not found.")

    manual_audit_path = (
        PROJECT_ROOT / "data" / "samples" / NOTEBOOK02_MANUAL_AUDIT_FILENAME
    )
    use_committed_audit = manual_audit_path.exists()

    kept = []
    skipped = 0

    for index, cell in enumerate(notebook.cells):
        if index >= recovery_index or cell.get("cell_type") != "code":
            kept.append(cell)
            continue

        source = "".join(cell.get("source", []))

        # The notebook originally creates a blank manual-audit CSV before the
        # human review. In clean-room execution, preserve the committed completed
        # audit instead of clobbering it with blank annotation columns.
        is_blank_audit_export = (
            use_committed_audit
            and NOTEBOOK02_MANUAL_AUDIT_FILENAME in source
            and "validation_df.to_csv" in source
        )
        if is_blank_audit_export:
            skipped += 1
            continue

        # Keep helper definitions; skip only calls that historically required a
        # human-in-the-loop decision before the notebook's own recovery checkpoint.
        is_definition = (
            "def show_next_unreviewed" in source
            or "def label_validation_example" in source
            or "def show_validation_example" in source
        )
        is_interactive_call = any(marker in source for marker in NOTEBOOK02_INTERACTIVE_CALLS)

        if is_interactive_call and not is_definition:
            skipped += 1
            continue

        kept.append(cell)

    notebook.cells = kept
    return skipped


def ensure_supported_layout() -> None:
    """Allow the documented Codespaces layout or a GitHub Actions runner."""
    expected_codespaces_root = Path("/workspaces/irp-disempowerment-nlp")
    running_in_actions = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"

    if PROJECT_ROOT == expected_codespaces_root or running_in_actions:
        return

    raise SystemExit(
        "\nThis automated clean-room runner supports GitHub Codespaces and "
        "GitHub Actions.\n"
        f"Expected Codespaces root: {expected_codespaces_root}\n"
        f"Current repository root:  {PROJECT_ROOT}\n\n"
        "For another environment, follow the manual notebook instructions "
        "in README.md."
    )


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


def check_environment() -> None:
    print("[setup] Checking installed package consistency...")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def execute_notebook(name: str) -> None:
    source_path = NOTEBOOK_DIR / name
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    audit_cells_skipped = 0
    if name == "02_filtering_and_sampling.ipynb":
        audit_cells_skipped = prepare_notebook02(notebook)

    original_count = len(notebook.cells)
    notebook.cells = [
        cell for cell in notebook.cells if not is_environment_mutation_cell(cell)
    ]
    environment_cells_skipped = original_count - len(notebook.cells)

    print("\n" + "=" * 78)
    print(f"Running {name}")
    print("=" * 78)
    if audit_cells_skipped:
        print(
            f"Skipped {audit_cells_skipped} Notebook 02 clean-room-only audit "
            "replay cell(s); committed manual-audit provenance is preserved."
        )
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
    executor.preprocess(
        notebook,
        {"metadata": {"path": str(PROJECT_ROOT)}},
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUN_DIR / name
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(f"PASS: {name}")
    print(f"Executed copy: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute notebooks 01-07 in the documented clean-room order."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        choices=range(1, 8),
        metavar="N",
        help="Start at notebook N (1-7). Default: 1.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=7,
        choices=range(1, 8),
        metavar="N",
        help="Stop after notebook N (1-7). Default: 7.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not run the full reproduction verifier at the end.",
    )
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    ensure_supported_layout()
    check_environment()
    register_kernel()

    selected = NOTEBOOKS[args.start - 1 : args.end]
    for notebook_name in selected:
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
