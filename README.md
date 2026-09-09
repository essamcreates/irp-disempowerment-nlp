# Data-Centric AI Safety

## Clean-room reproduction guide
### Evaluating Text Cleaning Techniques for Robust Detection of Disempowerment Patterns in Human-AI Conversations

This repository contains the completed technical artefact for an MSc Independent Research Project investigating how text cleaning and preprocessing affect robustness of an NLP classifier for disempowerment-associated patterns in human-AI conversations.

The frozen experiment is preserved. The clean-room tooling does **not** change methodology, weak labels, source experimental logic, random seed, or reported results. It exists only to make reproduction safer, more portable, and easier to verify.

> This is a research experiment, not a production harm-detection system. Weak labels are heuristic linguistic indicators and should not be treated as definitive diagnoses of harm or disempowerment.

## Running from the submitted ZIP

This is the recommended route for an examiner or third-party reviewer. **GitHub Codespaces is optional, not required.** A fresh extracted ZIP can be run from any ordinary local project directory on Windows, macOS, or Linux.

### Prerequisites

You need:

- Python **3.12**;
- an internet connection for dependency installation and LMSYS streaming;
- a Hugging Face account with access to `lmsys/lmsys-chat-1m`;
- enough disk space for generated intermediate files and results;
- patience for the scaled LMSYS scan and model evaluation.

The dataset revision used by the frozen experiment is:

```text
200748d9d3cddcc9d782887541057aca0b18c5da
```

### 1. Extract the ZIP

Extract it anywhere convenient, for example:

```text
C:\Users\you\Documents\irp-disempowerment-nlp
/Users/you/Documents/irp-disempowerment-nlp
/home/you/irp-disempowerment-nlp
```

The clean-room runner detects the project directory automatically. Historical absolute Codespaces paths are rewritten **only in temporary notebook copies**. The frozen source notebooks under `notebooks/` are never edited.

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
python -m pip check
```

If PowerShell blocks activation, use a normal user-scoped execution policy or run the equivalent commands from Command Prompt. Do not modify the project notebooks to work around environment setup.

#### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
python -m pip check
```

A healthy environment should end with:

```text
No broken requirements found.
```

### 3. Authenticate with Hugging Face

First make sure your Hugging Face account has accepted any access conditions for LMSYS-Chat-1M.

Then run:

```bash
hf auth login
```

Use a read token. Do not paste tokens into notebooks, scripts, or committed files.

You can confirm the login with:

```bash
hf auth whoami
```

If you receive `401`, `403`, gated-dataset, or authentication errors, confirm that the logged-in Hugging Face account has access to `lmsys/lmsys-chat-1m`.

### 4. Run the complete artefact

From the extracted project directory, run:

```bash
python scripts/run_cleanroom.py
```

That single command:

- checks the installed environment;
- registers the current Python environment as the temporary Jupyter kernel;
- runs notebooks **01 through 07 in order**;
- replays the original human decisions from decision-only provenance under `provenance/`;
- skips historical environment-repair cells that would mutate the locked environment;
- rewrites historical absolute Codespaces paths only in temporary notebook copies;
- stores executed temporary copies under `.cleanroom_runs/`;
- runs `scripts/verify_reproduction.py --full` at the end.

`.cleanroom_runs/` is local-only and must not be committed or shared as a dataset-derived artefact.

### Expected runtime

Runtime depends heavily on internet throughput and compute. Notebook 06 is the longest stage because it performs the scaled LMSYS scan and evaluation. A complete clean-room run can take roughly **45–120 minutes**, and slower connections may take longer.

Long streaming/model cells can remain quiet for several minutes without being stuck.

## Reproduction checkpoints

A successful full reproduction should reach these frozen checkpoints:

```text
Scaled sources:                      1,633
Positive target examples:               17
Negative target examples:            1,616
Positive prevalence:                 ~1.04%
Noise conditions:                         8
Preprocessing configurations:             4
Total experimental conditions:           32
OOF evaluation rows:                 52,256
```

Clean + no preprocessing:

```text
Pooled ROC-AUC:                0.8628
Pooled Average Precision:      0.1233
Mean-fold ROC-AUC:             0.8729
Mean-fold Average Precision:   0.1974
Predicted positives at default
LinearSVC threshold:                0
```

Notebook 07 should finish with:

```text
NOTEBOOK 07 FINAL INTEGRITY CHECK PASSED.
```

The full verifier should also complete successfully.

## Privacy-safe decision provenance

The public/shareable branch intentionally does **not** track LMSYS-derived text samples or raw audit CSVs under `data/samples/`.

Only this placeholder is tracked there:

```text
data/samples/.gitkeep
```

The original manual decisions required for exact software reproduction are stored under `provenance/` as compact decision-only CSVs:

```text
manual_relevance_decisions_200.csv
scaled_positive_audit_decisions_34.csv
full_positive_audit_decisions_56.csv
full_negative_audit_decisions_100.csv
```

These files contain only identifiers, weak-label/category fields where required for deterministic joins, and manual decision fields. They do **not** contain `user_text`, `assistant_text`, quoted evidence, free-text audit notes, prompts, responses, or conversation text.

Run the privacy regression check at any time with:

```bash
python scripts/check_public_sharing.py
```

It fails if:

- tracked `data/samples/` contains anything other than `.gitkeep`;
- an expected provenance file is missing;
- an unexpected provenance CSV is present;
- provenance headers contain text/evidence/note-style columns;
- provenance contains columns outside the approved identifier/category/decision schema.

## What the project does

The pipeline:

1. streams LMSYS-Chat-1M;
2. extracts eligible English user-assistant exchanges;
3. applies the frozen relevance filter;
4. creates deterministic candidate pools and a pilot sample;
5. applies frozen weak-labelling rules for sycophantic validation, overconfident judgement, and directive advice;
6. replays the original manual relevance/audit decisions for exact reproduction;
7. injects controlled synthetic text noise;
8. applies four preprocessing configurations;
9. trains the TF-IDF + LinearSVC baseline;
10. evaluates held-out sources across 32 noise × preprocessing conditions;
11. analyses score robustness and cue retention;
12. compares pilot and scaled results.

The final 1,633-row target is an **audit-corrected weak-label target**, not exhaustive human-annotated ground truth.

## Notebook order

The clean-room runner executes:

```text
01_dataset_inspection.ipynb
02_filtering_and_sampling.ipynb
03_labelling_pilot.ipynb
04_noise_and_preprocessing.ipynb
05_model_training_evaluation.ipynb
06_scaled_experiment.ipynb
07_final_synthesis.ipynb
```

Partial execution is available for debugging:

```bash
python scripts/run_cleanroom.py --start 1 --end 5
python scripts/run_cleanroom.py --start 6 --end 7
```

If starting later than Notebook 01, required intermediate files from earlier notebooks must already exist locally.

## Key frozen configuration

```text
Random seed:              42
Weak-label version:       disempowerment_weak_v5
Noise version:            noise_v1
Preprocessing version:    preprocess_v1
Model:                    TF-IDF + LinearSVC
LinearSVC C:              1.0
Class weight:             balanced
Cross-validation:         3-fold source-level StratifiedKFold
```

Noise conditions:

```text
clean
casing
punctuation
whitespace
typo
word_deletion
filler
mixed
```

Preprocessing configurations:

```text
none
minimal
noise_aware
aggressive
```

## Important audit checkpoints

Positive audit:

```text
56 reviewed
16 supported positives
40 rejected false positives
```

Negative audit:

```text
100 reviewed
99 confirmed negative
1 missed positive
```

Final target:

```text
1,633 total sources
17 positive
1,616 negative
```

## Quick verification without rerunning LMSYS

To inspect the committed result package without rebuilding the dataset pipeline:

```bash
python scripts/verify_reproduction.py --quick
```

This checks the frozen configuration and committed headline result files.

## Troubleshooting

### `python` is not Python 3.12

Check:

```bash
python --version
```

On Windows, `py -3.12` may be the correct launcher. On macOS/Linux, use `python3.12` when creating the environment.

### `No module named ...`

Activate `.venv`, reinstall the locked environment, then run:

```bash
python -m pip check
```

### Hugging Face authentication or gated-dataset errors

Run:

```bash
hf auth whoami
```

Confirm the account has accepted LMSYS-Chat-1M access conditions. Re-run `hf auth login` if necessary.

### A notebook appears frozen

Notebook 06 contains long network and modelling stages. Check CPU/network activity and allow time before interrupting it.

### Re-running after a failed attempt

Generated intermediates are local. For the cleanest restart, use a fresh extracted ZIP or remove generated local data/results that are not part of the submitted repository before starting again. Do not alter the frozen source notebooks or provenance decisions.

### Historical `/workspaces/irp-disempowerment-nlp` paths

You should **not** create a compatibility symlink manually. `scripts/run_cleanroom.py` rewrites that historical path only in the temporary in-memory/executed notebook copies. This is what makes the same ZIP runnable from Windows, macOS, Linux, Codespaces, and GitHub Actions.

## GitHub Codespaces route

Codespaces remains a convenient option but is not required.

Inside a Codespace:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
hf auth login
python scripts/run_cleanroom.py
```

The same runner is used locally and in CI.

## Automated clean-room testing

The `Clean-room reproduction` GitHub Actions workflow runs from the ordinary GitHub Actions checkout directory, not from `/workspaces/irp-disempowerment-nlp` and without a compatibility symlink. It performs the privacy regression check, installs the locked Python 3.12 environment, authenticates to Hugging Face via the configured repository secret, runs notebooks 01–07, and executes the full verifier.

The workflow is read-only with respect to repository contents and does not upload `.cleanroom_runs/` or LMSYS-derived text as artifacts.

## Results to inspect

Completed headline outputs are under:

```text
results/tables/
results/figures/
```

Useful files include:

```text
results/tables/final_pilot_vs_scaled_comparison.csv
results/tables/final_research_question_synthesis.csv
results/figures/final_pilot_vs_scaled_preprocessing_retention.png
```

## Historical Git limitation

The current branch tip is sanitized for public/ZIP sharing, but files removed from the branch tip can remain recoverable from earlier Git commits, pull-request refs, Actions logs/caches, or repository history until that history and any retained caches are explicitly rewritten/purged. A ZIP exported from the sanitized branch tip contains only the current sanitized tree; publishing the full Git repository/history is a separate privacy decision.

Do not merge this branch to `main` until the final sharing decision is made.
