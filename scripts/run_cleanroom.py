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
KERNEL_NAME = "irp-cleanroom"
NOTEBOOKS = ["01_dataset_inspection.ipynb","02_filtering_and_sampling.ipynb","03_labelling_pilot.ipynb","04_noise_and_preprocessing.ipynb","05_model_training_evaluation.ipynb","06_scaled_experiment.ipynb","07_final_synthesis.ipynb"]
ENVIRONMENT_MUTATION_MARKERS = ("Clean-reinstall current scikit-learn version","pip uninstall","pip install","%pip install","!pip install","Restart the Jupyter kernel now")
NOTEBOOK02_MANUAL_AUDIT_FILENAME = "lmsys_relevance_manual_validation_200.csv"
NOTEBOOK02_AUDIT_LOAD_MARKER = "validation_df = pd.read_csv("
NOTEBOOK02_DOWNSTREAM_MARKER = "validated_relevant_df = validation_df["
NOTEBOOK02_INTERACTIVE_CALLS = ("show_next_unreviewed()","label_validation_example(","show_validation_example(")
NOTEBOOK06_SCALED_AUDIT_FILENAME = "lmsys_scaled_weak_v5_positive_audit_34.csv"
NOTEBOOK06_SCALED_VERIFY_MARKER = "# Verify completed scaled positive audit"

def is_environment_mutation_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code": return False
    source = "".join(cell.get("source", [])); lower_source = source.lower()
    if any(marker.lower() in lower_source for marker in ENVIRONMENT_MUTATION_MARKERS): return True
    return "subprocess.check_call" in source and '"-m"' in source and '"pip"' in source

def prepare_notebook02(notebook) -> int:
    manual_audit_path = PROJECT_ROOT / "data" / "samples" / NOTEBOOK02_MANUAL_AUDIT_FILENAME
    if not manual_audit_path.exists(): return 0
    load_index = downstream_index = None
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code": continue
        source = "".join(cell.get("source", []))
        if load_index is None and NOTEBOOK02_AUDIT_LOAD_MARKER in source: load_index = index
        if NOTEBOOK02_DOWNSTREAM_MARKER in source: downstream_index = index; break
    if load_index is None or downstream_index is None or load_index >= downstream_index:
        raise RuntimeError("Notebook 02 clean-room audit checkpoint markers were not found in the expected order.")
    kept=[]; adjusted=0
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code": kept.append(cell); continue
        source = "".join(cell.get("source", []))
        if NOTEBOOK02_MANUAL_AUDIT_FILENAME in source and "validation_df.to_csv" in source:
            cell["source"] = f'validation_path = SAMPLES_DIR / "{NOTEBOOK02_MANUAL_AUDIT_FILENAME}"\nprint("Using committed manual audit:", validation_path)\n'
            kept.append(cell); adjusted += 1; continue
        is_definition = any(x in source for x in ("def show_next_unreviewed","def label_validation_example","def show_validation_example"))
        is_interactive_call = any(marker in source for marker in NOTEBOOK02_INTERACTIVE_CALLS)
        if is_interactive_call and not is_definition: adjusted += 1; continue
        if load_index < index < downstream_index: adjusted += 1; continue
        kept.append(cell)
    notebook.cells=kept
    return adjusted

def prepare_notebook06(notebook) -> int:
    """Restore the committed 34-row human audit immediately before its verification.

    Notebook 06 preserves the historical interactive audit transcript. Sequential
    replay recreates a blank audit and only the first historical decision before the
    checkpoint. The original completed audit is committed as provenance, so the
    temporary clean-room copy reloads it at the checkpoint. No labels or source
    experimental logic are changed.
    """
    audit_path = PROJECT_ROOT / "data" / "samples" / NOTEBOOK06_SCALED_AUDIT_FILENAME
    if not audit_path.exists(): return 0
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code": continue
        source = "".join(cell.get("source", []))
        if NOTEBOOK06_SCALED_VERIFY_MARKER in source:
            loader = nbformat.v4.new_code_cell(
                f'scaled_positive_audit_path = PROJECT_ROOT / "data" / "samples" / "{NOTEBOOK06_SCALED_AUDIT_FILENAME}"\n'
                'scaled_positive_audit_df = pd.read_csv(scaled_positive_audit_path)\n'
                'print("Using committed scaled positive audit:", scaled_positive_audit_path)\n'
            )
            notebook.cells.insert(index, loader)
            return 1
    raise RuntimeError("Notebook 06 scaled positive audit verification marker was not found.")

def ensure_supported_layout() -> None:
    expected = Path("/workspaces/irp-disempowerment-nlp")
    if PROJECT_ROOT == expected or os.environ.get("GITHUB_ACTIONS", "").lower() == "true": return
    raise SystemExit(f"Automated clean-room runner supports GitHub Codespaces/Actions. Expected {expected}; got {PROJECT_ROOT}.")

def register_kernel() -> None:
    print("\n[setup] Registering the current virtual environment as a Jupyter kernel...")
    subprocess.run([sys.executable,"-m","ipykernel","install","--user","--name",KERNEL_NAME,"--display-name","Python (IRP clean-room)"],check=True)

def check_environment() -> None:
    print("[setup] Checking installed package consistency..."); subprocess.run([sys.executable,"-m","pip","check"],check=True)

def execute_notebook(name: str) -> None:
    source_path=NOTEBOOK_DIR/name
    if not source_path.exists(): raise FileNotFoundError(source_path)
    with source_path.open("r",encoding="utf-8") as handle: notebook=nbformat.read(handle,as_version=4)
    audit_adjustments=0
    if name=="02_filtering_and_sampling.ipynb": audit_adjustments=prepare_notebook02(notebook)
    elif name=="06_scaled_experiment.ipynb": audit_adjustments=prepare_notebook06(notebook)
    original_count=len(notebook.cells)
    notebook.cells=[cell for cell in notebook.cells if not is_environment_mutation_cell(cell)]
    environment_cells_skipped=original_count-len(notebook.cells)
    print("\n"+"="*78); print(f"Running {name}"); print("="*78)
    if audit_adjustments: print(f"Applied {audit_adjustments} clean-room audit-provenance adjustment(s); committed human audit data is used.")
    if environment_cells_skipped: print(f"Skipped {environment_cells_skipped} historical environment-repair cell(s). The locked environment is used instead.")
    executor=ExecutePreprocessor(timeout=None,kernel_name=KERNEL_NAME,allow_errors=False)
    executor.preprocess(notebook,{"metadata":{"path":str(PROJECT_ROOT)}})
    RUN_DIR.mkdir(parents=True,exist_ok=True); output_path=RUN_DIR/name
    with output_path.open("w",encoding="utf-8") as handle: nbformat.write(notebook,handle)
    print(f"PASS: {name}"); print(f"Executed copy: {output_path}")

def main() -> int:
    parser=argparse.ArgumentParser(description="Execute notebooks 01-07 in documented clean-room order.")
    parser.add_argument("--start",type=int,default=1,choices=range(1,8),metavar="N")
    parser.add_argument("--end",type=int,default=7,choices=range(1,8),metavar="N")
    parser.add_argument("--no-verify",action="store_true")
    args=parser.parse_args()
    if args.start>args.end: parser.error("--start must be less than or equal to --end")
    ensure_supported_layout(); check_environment(); register_kernel()
    for notebook_name in NOTEBOOKS[args.start-1:args.end]: execute_notebook(notebook_name)
    if not args.no_verify and args.end==7:
        print("\n"+"="*78); print("Running full reproduction verification"); print("="*78)
        subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts"/"verify_reproduction.py"),"--full"],cwd=PROJECT_ROOT,check=True)
    print("\nClean-room notebook execution completed successfully."); return 0
if __name__=="__main__": raise SystemExit(main())
