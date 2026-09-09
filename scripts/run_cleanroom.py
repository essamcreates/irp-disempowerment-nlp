#!/usr/bin/env python3
"""Execute the IRP notebooks in clean-room order without changing frozen logic."""
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
PROVENANCE_DIR = PROJECT_ROOT / "provenance"
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

# Decision-only provenance committed to the repository. These files contain no
# LMSYS user/assistant text or quoted weak-label evidence.
NOTEBOOK02_DECISIONS_FILENAME = "manual_relevance_decisions_200.csv"
NOTEBOOK06_SCALED_DECISIONS_FILENAME = "scaled_positive_audit_decisions_34.csv"
NOTEBOOK06_FULL_POSITIVE_DECISIONS_FILENAME = "full_positive_audit_decisions_56.csv"
NOTEBOOK06_NEGATIVE_DECISIONS_FILENAME = "full_negative_audit_decisions_100.csv"

NOTEBOOK02_AUDIT_LOAD_MARKER = "validation_df = pd.read_csv("
NOTEBOOK02_DOWNSTREAM_MARKER = "validated_relevant_df = validation_df["
NOTEBOOK02_INTERACTIVE_CALLS = (
    "show_next_unreviewed()",
    "label_validation_example(",
    "show_validation_example(",
)

NOTEBOOK06_SCALED_VERIFY_MARKER = "# Verify completed scaled positive audit"
NOTEBOOK06_FULL_POSITIVE_COMBINE_MARKER = (
    "# Combine and save the complete 56-example positive audit"
)
NOTEBOOK06_NEGATIVE_CHECK_MARKER = "negative_audit_check_df = pd.read_csv("


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


def require_provenance_file(filename: str) -> Path:
    path = PROVENANCE_DIR / filename
    if not path.is_file():
        raise RuntimeError(
            "Decision-only clean-room provenance is missing: " + str(path)
        )
    return path


def prepare_notebook02(notebook) -> int:
    """Apply recorded relevance decisions to freshly reconstructed validation rows."""
    require_provenance_file(NOTEBOOK02_DECISIONS_FILENAME)

    load_index = None
    downstream_index = None
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if load_index is None and NOTEBOOK02_AUDIT_LOAD_MARKER in source:
            load_index = index
        if NOTEBOOK02_DOWNSTREAM_MARKER in source:
            downstream_index = index
            break

    if (
        load_index is None
        or downstream_index is None
        or load_index >= downstream_index
    ):
        raise RuntimeError(
            "Notebook 02 clean-room audit checkpoint markers were not found "
            "in the expected order."
        )

    kept = []
    adjusted = 0

    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            kept.append(cell)
            continue

        source = "".join(cell.get("source", []))

        if index == load_index:
            cell["source"] = f'''
manual_decision_path = PROJECT_ROOT / "provenance" / "{NOTEBOOK02_DECISIONS_FILENAME}"
manual_decisions_df = pd.read_csv(
    manual_decision_path,
    dtype={{
        "manual_relevant": "string",
        "manual_category": "string",
    }},
    keep_default_na=False,
)

assert len(manual_decisions_df) == 200
assert manual_decisions_df[["source_index", "pair_index"]].duplicated().sum() == 0
assert manual_decisions_df["manual_relevant"].isin(["0", "1"]).all()

validation_df = validation_df.drop(
    columns=["manual_relevant", "manual_category", "manual_notes"],
    errors="ignore",
)
validation_df = validation_df.merge(
    manual_decisions_df,
    on=["source_index", "pair_index"],
    how="left",
    validate="one_to_one",
)
assert len(validation_df) == 200
assert validation_df["manual_relevant"].notna().all()
validation_df["manual_relevant"] = validation_df["manual_relevant"].astype("string")
validation_df["manual_category"] = validation_df["manual_category"].fillna("").astype("string")
validation_df["manual_notes"] = ""

validation_path = SAMPLES_DIR / "lmsys_relevance_manual_validation_200.csv"
validation_df.to_csv(validation_path, index=False)

completed = validation_df[
    validation_df["manual_relevant"].isin(["0", "1"])
]
print("Using decision-only manual relevance provenance:", manual_decision_path)
print("Total rows:", len(validation_df))
print("Reviewed:", len(completed))
print("Remaining:", len(validation_df) - len(completed))
print("Current decisions:")
print(completed["manual_relevant"].value_counts().sort_index())
'''
            kept.append(cell)
            adjusted += 1
            continue

        is_definition = any(
            marker in source
            for marker in (
                "def show_next_unreviewed",
                "def label_validation_example",
                "def show_validation_example",
            )
        )
        is_interactive_call = any(
            marker in source for marker in NOTEBOOK02_INTERACTIVE_CALLS
        )

        if is_interactive_call and not is_definition:
            adjusted += 1
            continue

        # The historical interactive annotation transcript sits between the load
        # checkpoint and the first downstream construction step. The decisions are
        # already represented in the decision-only provenance above.
        if load_index < index < downstream_index:
            adjusted += 1
            continue

        kept.append(cell)

    notebook.cells = kept
    return adjusted


def prepare_notebook06(notebook) -> int:
    """Apply recorded human audit decisions to freshly reconstructed audit rows."""
    for filename in (
        NOTEBOOK06_SCALED_DECISIONS_FILENAME,
        NOTEBOOK06_FULL_POSITIVE_DECISIONS_FILENAME,
        NOTEBOOK06_NEGATIVE_DECISIONS_FILENAME,
    ):
        require_provenance_file(filename)

    adjusted = 0
    found_scaled = False
    found_full_positive = False
    found_negative = False

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))

        if NOTEBOOK06_SCALED_VERIFY_MARKER in source:
            cell["source"] = f'''
# ------------------------------------------------------------
# Clean-room replay of completed scaled positive audit
# ------------------------------------------------------------
scaled_decision_path = PROJECT_ROOT / "provenance" / "{NOTEBOOK06_SCALED_DECISIONS_FILENAME}"
scaled_decisions_df = pd.read_csv(scaled_decision_path, keep_default_na=False)

assert len(scaled_decisions_df) == 34
assert scaled_decisions_df[["audit_number", "source_index", "pair_index"]].duplicated().sum() == 0

scaled_positive_audit_df = scaled_positive_audit_df.drop(
    columns=["manual_audit_valid", "manual_audit_note"],
    errors="ignore",
)
scaled_positive_audit_df = scaled_positive_audit_df.merge(
    scaled_decisions_df,
    on=["audit_number", "source_index", "pair_index", "weak_labels"],
    how="left",
    validate="one_to_one",
)
scaled_positive_audit_df["manual_audit_valid"] = pd.to_numeric(
    scaled_positive_audit_df["manual_audit_valid"],
    errors="raise",
)
scaled_positive_audit_df["manual_audit_note"] = ""

assert len(scaled_positive_audit_df) == 34
assert scaled_positive_audit_df["manual_audit_valid"].notna().all()
assert scaled_positive_audit_df["manual_audit_valid"].isin([0, 1]).all()

scaled_positive_audit_path = SCALED_DATA_DIR / "lmsys_scaled_weak_v5_positive_audit_34.csv"
scaled_positive_audit_df.to_csv(scaled_positive_audit_path, index=False)

print("Using decision-only scaled positive audit provenance:", scaled_decision_path)
print("Reviewed:", len(scaled_positive_audit_df), "/ 34")
print("Valid:", int(scaled_positive_audit_df["manual_audit_valid"].sum()))
print("False positives:", int((scaled_positive_audit_df["manual_audit_valid"] == 0).sum()))
'''
            adjusted += 1
            found_scaled = True
            continue

        if NOTEBOOK06_FULL_POSITIVE_COMBINE_MARKER in source:
            cell["source"] = f'''
# ------------------------------------------------------------
# Combine freshly reconstructed positive audit rows and apply
# decision-only completed 56-row provenance.
# ------------------------------------------------------------
complete_positive_audit_df = pd.concat(
    [
        saved_positive_audit_df.copy(),
        expanded_positive_audit_df.copy(),
    ],
    ignore_index=True,
)
complete_positive_audit_df = (
    complete_positive_audit_df
    .sort_values("audit_number")
    .reset_index(drop=True)
)

full_positive_decision_path = PROJECT_ROOT / "provenance" / "{NOTEBOOK06_FULL_POSITIVE_DECISIONS_FILENAME}"
full_positive_decisions_df = pd.read_csv(
    full_positive_decision_path,
    keep_default_na=False,
)
assert len(full_positive_decisions_df) == 56
assert full_positive_decisions_df[["audit_number", "source_index", "pair_index"]].duplicated().sum() == 0

complete_positive_audit_df = complete_positive_audit_df.drop(
    columns=["manual_audit_valid", "manual_audit_note"],
    errors="ignore",
)
complete_positive_audit_df = complete_positive_audit_df.merge(
    full_positive_decisions_df,
    on=["audit_number", "source_index", "pair_index", "weak_labels"],
    how="left",
    validate="one_to_one",
)
complete_positive_audit_df["manual_audit_valid"] = pd.to_numeric(
    complete_positive_audit_df["manual_audit_valid"],
    errors="coerce",
)
complete_positive_audit_df["manual_audit_note"] = ""

assert len(complete_positive_audit_df) == 56
assert complete_positive_audit_df["audit_number"].tolist() == list(range(1, 57))
assert complete_positive_audit_df[["source_index", "pair_index"]].duplicated().sum() == 0
assert complete_positive_audit_df["manual_audit_valid"].notna().all()

total_valid = int(complete_positive_audit_df["manual_audit_valid"].sum())
total_false_positive = len(complete_positive_audit_df) - total_valid
assert total_valid == 16
assert total_false_positive == 40

final_audit_summary = complete_positive_audit_df.groupby("weak_labels").agg(
    detected=("audit_number", "size"),
    valid=("manual_audit_valid", "sum"),
)
final_audit_summary["valid"] = final_audit_summary["valid"].astype(int)
final_audit_summary["false_positive"] = (
    final_audit_summary["detected"] - final_audit_summary["valid"]
)
final_audit_summary["audit_precision_pct"] = (
    final_audit_summary["valid"] / final_audit_summary["detected"] * 100
).round(1)
overall_audit_precision = total_valid / len(complete_positive_audit_df) * 100

complete_positive_audit_path = SCALED_DATA_DIR / "lmsys_full_weak_v5_positive_audit_56.csv"
complete_positive_audit_df.to_csv(complete_positive_audit_path, index=False)

print("Using decision-only complete positive audit provenance:", full_positive_decision_path)
print("Reviewed:", len(complete_positive_audit_df), "/ 56")
print("Valid:", total_valid)
print("False positives:", total_false_positive)
display(final_audit_summary)
'''
            adjusted += 1
            found_full_positive = True
            continue

        if NOTEBOOK06_NEGATIVE_CHECK_MARKER in source:
            cell["source"] = f'''
negative_decision_path = PROJECT_ROOT / "provenance" / "{NOTEBOOK06_NEGATIVE_DECISIONS_FILENAME}"
negative_decisions_df = pd.read_csv(
    negative_decision_path,
    keep_default_na=False,
)
assert len(negative_decisions_df) == 100
assert negative_decisions_df[["audit_number", "source_index", "pair_index"]].duplicated().sum() == 0

negative_audit_100_df = negative_audit_100_df.drop(
    columns=["manual_missed_positive", "manual_missed_labels", "manual_audit_note"],
    errors="ignore",
)
negative_audit_100_df = negative_audit_100_df.merge(
    negative_decisions_df,
    on=["audit_number", "source_index", "pair_index"],
    how="left",
    validate="one_to_one",
)
negative_audit_100_df["manual_missed_positive"] = pd.to_numeric(
    negative_audit_100_df["manual_missed_positive"],
    errors="coerce",
)
negative_audit_100_df["manual_missed_labels"] = (
    negative_audit_100_df["manual_missed_labels"].fillna("")
)
negative_audit_100_df["manual_audit_note"] = ""

assert len(negative_audit_100_df) == 100
assert negative_audit_100_df["manual_missed_positive"].notna().all()
assert negative_audit_100_df["manual_missed_positive"].isin([0, 1]).all()

negative_audit_100_path = SCALED_DATA_DIR / "lmsys_full_weak_v5_negative_audit_100.csv"
negative_audit_100_df.to_csv(negative_audit_100_path, index=False)

negative_audit_check_df = pd.read_csv(negative_audit_100_path)
print("Using decision-only negative audit provenance:", negative_decision_path)
print("Negative audit reviewed:")
print(len(negative_audit_check_df))
print(
    "Negative-audit missed positives:",
    int(pd.to_numeric(negative_audit_check_df["manual_missed_positive"]).sum()),
)
print("\nAudit files verified.")
'''
            adjusted += 1
            found_negative = True
            continue

    if not found_scaled:
        raise RuntimeError(
            "Notebook 06 scaled positive audit verification marker was not found."
        )
    if not found_full_positive:
        raise RuntimeError(
            "Notebook 06 complete positive audit checkpoint marker was not found."
        )
    if not found_negative:
        raise RuntimeError(
            "Notebook 06 negative audit verification marker was not found."
        )

    return adjusted


def ensure_supported_layout() -> None:
    expected = Path("/workspaces/irp-disempowerment-nlp")
    if (
        PROJECT_ROOT == expected
        or os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    ):
        return
    raise SystemExit(
        "Automated clean-room runner supports GitHub Codespaces/Actions. "
        f"Expected {expected}; got {PROJECT_ROOT}."
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

    audit_adjustments = 0
    if name == "02_filtering_and_sampling.ipynb":
        audit_adjustments = prepare_notebook02(notebook)
    elif name == "06_scaled_experiment.ipynb":
        audit_adjustments = prepare_notebook06(notebook)

    original_count = len(notebook.cells)
    notebook.cells = [
        cell for cell in notebook.cells if not is_environment_mutation_cell(cell)
    ]
    environment_cells_skipped = original_count - len(notebook.cells)

    print("\n" + "=" * 78)
    print(f"Running {name}")
    print("=" * 78)
    if audit_adjustments:
        print(
            f"Applied {audit_adjustments} clean-room audit-provenance "
            "adjustment(s); decision-only human audit data is used."
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
        description="Execute notebooks 01-07 in documented clean-room order."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        choices=range(1, 8),
        metavar="N",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=7,
        choices=range(1, 8),
        metavar="N",
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    ensure_supported_layout()
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
