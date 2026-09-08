#!/usr/bin/env python3
"""
Execute the IRP notebooks in clean-room order inside GitHub Codespaces.

The original notebooks preserve historical exploratory/environment-repair cells.
This runner executes temporary copies and skips only cells that mutate the Python
environment. It never edits the source notebooks.
"""

from __future__ import annotations

import argparse
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


def ensure_codespaces_layout() -> None:
    expected = Path("/workspaces/irp-disempowerment-nlp")
    if PROJECT_ROOT != expected:
        raise SystemExit(
            "\nThis automated clean-room runner supports the documented GitHub "
            "Codespaces layout only.\n"
            f"Expected repository root: {expected}\n"
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

    original_count = len(notebook.cells)
    notebook.cells = [
        cell for cell in notebook.cells if not is_environment_mutation_cell(cell)
    ]
    skipped = original_count - len(notebook.cells)

    print("\n" + "=" * 78)
    print(f"Running {name}")
    print("=" * 78)
    if skipped:
        print(
            f"Skipped {skipped} historical environment-repair cell(s). "
            "The locked environment is used instead."
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

    ensure_codespaces_layout()
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
