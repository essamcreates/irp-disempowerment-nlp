#!/usr/bin/env python3
"""
Execute the IRP notebooks in clean-room order in GitHub Codespaces or Actions.

The original notebooks preserve historical exploratory/environment-repair cells.
This runner executes temporary copies and skips only cells that mutate the Python
environment. It never edits the source notebooks.
"""

from __future__ import annotations

import argparse
import ast
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

NOTEBOOK02_HELPERS = (
    "show_validation_example",
    "show_next_unreviewed",
    "label_validation_example",
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


def inject_notebook02_helpers(notebook) -> int:
    """
    Make Notebook 02 executable top-to-bottom without changing its source file.

    The historical notebook contains manual-review helper functions that were
    defined later during the interactive annotation session, while earlier cells
    call those helpers. For a clean-room sequential run, copy just those function
    definitions into a temporary synthetic cell immediately before their first
    use. The original notebook remains untouched and its recorded decisions are
    still executed exactly as stored.
    """
    definitions: dict[str, str] = {}

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in NOTEBOOK02_HELPERS and node.name not in definitions:
                    definitions[node.name] = ast.unparse(node)

    missing = [name for name in NOTEBOOK02_HELPERS if name not in definitions]
    if missing:
        raise RuntimeError(
            "Notebook 02 clean-room helper definitions could not be found: "
            + ", ".join(missing)
        )

    first_use_index = None
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "show_next_unreviewed()" in source and "def show_next_unreviewed" not in source:
            first_use_index = index
            break

    if first_use_index is None:
        raise RuntimeError("Notebook 02 first manual-review helper use was not found.")

    helper_source = (
        "# Clean-room execution shim: hoist historical manual-review helpers.\n"
        "# The source notebook is not modified.\n\n"
        + "\n\n".join(definitions[name] for name in NOTEBOOK02_HELPERS)
    )
    notebook.cells.insert(first_use_index, nbformat.v4.new_code_cell(helper_source))
    return 1


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

    injected = 0
    if name == "02_filtering_and_sampling.ipynb":
        injected = inject_notebook02_helpers(notebook)

    original_count = len(notebook.cells)
    notebook.cells = [
        cell for cell in notebook.cells if not is_environment_mutation_cell(cell)
    ]
    skipped = original_count - len(notebook.cells)

    print("\n" + "=" * 78)
    print(f"Running {name}")
    print("=" * 78)
    if injected:
        print(
            "Inserted a temporary Notebook 02 helper-definition shim for "
            "top-to-bottom execution."
        )
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
