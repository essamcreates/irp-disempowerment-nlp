# Data-Centric AI Safety

## Clean-room reproduction guide
### Evaluating Text Cleaning Techniques for Robust Detection of Disempowerment Patterns in Human-AI Conversations

This repository contains the completed technical artefact for an MSc Independent Research Project investigating how text cleaning and preprocessing choices affect the robustness of an NLP classifier for detecting disempowerment-associated patterns in human-AI conversations.

This README is written for a new user who has never seen the project before. It supports two use cases:

1. **Inspect the completed artefact** without rerunning the dataset pipeline.
2. **Reproduce the experiment from a fresh GitHub Codespace** using the same pinned dataset revision, locked environment, random seed, frozen methodology, and recorded manual audit decisions.

> **Important:** this is a research experiment, not a production harm-detection system. Weak labels are heuristic linguistic indicators and must not be treated as definitive evidence that a conversation is harmful or disempowering.

---

## 1. What the project does

The pipeline:

1. streams the LMSYS-Chat-1M dataset;
2. extracts eligible English user-assistant exchanges;
3. applies a frozen relevance filter;
4. creates deterministic candidate pools and a pilot sample;
5. applies weak-labelling rules for three disempowerment-associated patterns;
6. preserves the original manual relevance and audit decisions used in the study;
7. injects controlled synthetic text noise;
8. applies four preprocessing configurations;
9. trains a TF-IDF + LinearSVC baseline;
10. evaluates held-out sources across 32 noise × preprocessing conditions;
11. analyses ranking robustness, decision-score shifts, and linguistic cue retention;
12. compares the pilot and scaled experiments.

The three target patterns are:

- **sycophantic validation**
- **overconfident judgement**
- **directive advice**

---

## 2. Research questions

**RQ1.** How do text cleaning and preprocessing strategies affect the robustness of NLP classifiers for detecting disempowerment patterns in noisy human-AI conversations?

**RQ2.** Which textual noise types and preprocessing configurations produce the greatest changes in classifier performance across clean, noisy, and cleaned-noisy conversation data?

**RQ3.** How do preprocessing choices influence the preservation or loss of linguistic cues associated with disempowerment, including sycophantic validation, overconfident judgement, and directive advice?

---

## 3. Choose your route

### Route A — Inspect the completed results

Use this if you only want to review the finished artefact.

Open:

```text
notebooks/07_final_synthesis.ipynb
```

and inspect:

```text
results/tables/
results/figures/
```

Useful final files include:

```text
results/tables/final_pilot_vs_scaled_comparison.csv
results/tables/final_research_question_synthesis.csv
results/figures/final_pilot_vs_scaled_preprocessing_retention.png
```

You can also run the quick verifier:

```bash
python scripts/verify_reproduction.py --quick
```

This checks the frozen configuration and committed headline result files without requiring the LMSYS dataset.

### Route B — Full reproduction from scratch

Use this if you want to recreate the experiment.

The supported clean-room environment is **GitHub Codespaces**, because several historical notebooks use the original Codespaces project path:

```text
/workspaces/irp-disempowerment-nlp
```

The automated runner executes temporary copies of the notebooks and does not edit the source notebooks.

---

## 4. Expected final reproduction checkpoints

A successful reproduction should end with:

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
Predicted positives at the
default LinearSVC threshold:        0
```

Directive cue-retention checkpoint:

```text
Aggressive preprocessing: 0.0%
```

Notebook 07 should finish with:

```text
NOTEBOOK 07 FINAL INTEGRITY CHECK PASSED.
```

---

## 5. Beginner setup with GitHub Codespaces

### Step 1 — Create a Codespace

On the GitHub repository page:

1. Click the green **Code** button.
2. Open the **Codespaces** tab.
3. Click **Create codespace on main**.
4. Wait for the browser-based VS Code window to open.

You should see folders such as:

```text
notebooks/
src/
results/
data/
scripts/
```

### Step 2 — Open a terminal

In VS Code:

1. Click **Terminal**.
2. Click **New Terminal**.

Check your location:

```bash
pwd
```

It should be:

```text
/workspaces/irp-disempowerment-nlp
```

### Step 3 — Create and activate the Python environment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now begin with:

```text
(.venv)
```

### Step 4 — Install the exact locked environment

Run:

```bash
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
python -m pip check
```

Expected final line:

```text
No broken requirements found.
```

The original final environment used:

```text
Python          3.12.1
NumPy           2.5.2
pandas          2.3.3
SciPy           1.18.0
scikit-learn    1.9.0
matplotlib      3.11.1
```

---

## 6. Hugging Face dataset access

Primary dataset:

```text
lmsys/lmsys-chat-1m
```

Pinned revision:

```text
200748d9d3cddcc9d782887541057aca0b18c5da
```

Before reproduction:

1. Sign in to Hugging Face.
2. Open the LMSYS-Chat-1M dataset page.
3. Accept the dataset access/licence conditions if prompted.
4. In the Codespaces terminal, run:

```bash
hf auth login
```

Do not paste access tokens into notebooks or source files.

If you see a `401`, `403`, gated-dataset, or authentication error, confirm that the account used by `hf auth login` has access to LMSYS-Chat-1M.

---

## 7. Recommended full reproduction command

After setup and Hugging Face authentication, run:

```bash
python scripts/run_cleanroom.py
```

The runner:

- uses notebooks 01–07 in order;
- executes temporary copies;
- skips only historical environment-repair cells that would uninstall/reinstall packages;
- keeps the locked environment unchanged;
- leaves the original notebooks untouched;
- runs the full verifier at the end.

Executed copies are written to:

```text
.cleanroom_runs/
```

This folder is local-only and should not be committed.

### Running only part of the sequence

Examples:

```bash
python scripts/run_cleanroom.py --start 1 --end 5
python scripts/run_cleanroom.py --start 6 --end 7
```

If restarting at a later notebook, the local intermediate files produced by the earlier notebooks must already exist.

---

## 8. Manual notebook route

If you prefer to click through the notebooks yourself, run them in exactly this order:

```text
01_dataset_inspection.ipynb
02_filtering_and_sampling.ipynb
03_labelling_pilot.ipynb
04_noise_and_preprocessing.ipynb
05_model_training_evaluation.ipynb
06_scaled_experiment.ipynb
07_final_synthesis.ipynb
```

For each notebook:

1. open it in VS Code;
2. select the `.venv` Python kernel;
3. run the cells in order;
4. wait for each long streaming/modelling cell to finish before continuing.

Do not skip ahead during a full reproduction because later notebooks depend on local files produced by earlier notebooks.

---

## 9. Notebook-by-notebook checkpoints

### Notebook 01 — Dataset inspection

```text
notebooks/01_dataset_inspection.ipynb
```

Purpose:

- verifies LMSYS access;
- inspects the runtime schema;
- streams 25 examples;
- creates a small local inspection sample.

Expected local output:

```text
data/samples/lmsys_inspection_sample.csv
```

Expected rows:

```text
25
```

These inspection counts are diagnostic only and are not research findings.

### Notebook 02 — Filtering and sampling

```text
notebooks/02_filtering_and_sampling.ipynb
```

Frozen relevance-filter version:

```text
relevance_v1
```

Expected checkpoints:

```text
Conversation scan limit:              150,000
Raw candidate pairs:                      250
Unique-source candidates:                 229
Manual validation pool:                   200
Manually relevant:                        108
Manually not relevant:                     92
Final deterministic pilot:                100
Random seed:                                42
```

Important local outputs include:

```text
data/samples/lmsys_relevance_manual_validation_200.csv
data/samples/lmsys_relevance_pilot_100.csv
```

#### Manual relevance decisions

The study originally reviewed all 200 validation examples manually.

For **exact reproduction**, use the recorded decisions preserved in Notebook 02. Do not reinterpret or alter them.

For an **independent replication**, you may repeat the human review yourself. If your decisions differ, your downstream target may differ and should be reported as a new replication rather than a failure of the original software reproduction.

### Notebook 03 — Weak-labelling pilot

```text
notebooks/03_labelling_pilot.ipynb
```

Frozen weak-label version:

```text
disempowerment_weak_v5
```

Expected:

```text
Pilot rows:                    100
Positive weak labels:            6
Negative weak labels:           94

Sycophantic validation:           0
Overconfident judgement:          0
Directive advice:                 6
```

Expected local output:

```text
data/samples/lmsys_disempowerment_weak_v5_pilot_100.csv
```

### Notebook 04 — Noise and preprocessing

```text
notebooks/04_noise_and_preprocessing.ipynb
```

Frozen noise version:

```text
noise_v1
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

Frozen preprocessing version:

```text
preprocess_v1
```

Preprocessing configurations:

```text
none
minimal
noise_aware
aggressive
```

Expected pilot matrix:

```text
100 × 8 × 4 = 3,200 rows
```

Expected local output:

```text
data/samples/lmsys_noise_preprocess_v1_matrix_3200.csv
```

### Notebook 05 — Pilot model evaluation

```text
notebooks/05_model_training_evaluation.ipynb
```

Baseline:

```text
TF-IDF word unigrams + bigrams
LinearSVC(C=1.0, class_weight="balanced", random_state=42)
3-fold source-level StratifiedKFold
```

Models are trained only on clean + no-preprocessing training text.

Expected clean pilot result:

```text
Accuracy:                  0.94
Balanced accuracy:         0.50
Precision:                 0
Recall:                    0
F1:                        0

TN:                        94
FP:                         0
FN:                         6
TP:                         0

Clean mean-fold ROC-AUC:   0.867
Clean mean-fold AP:        0.537
```

### Notebook 06 — Scaled experiment

```text
notebooks/06_scaled_experiment.ipynb
```

This is the longest stage.

The original full 1-million-conversation scan took about 25 minutes in GitHub Codespaces. Earlier exploratory/scaled scans in the historical notebook can add more time. Allow approximately **45–90 minutes** for the whole notebook depending on network and compute.

Do not assume a streaming cell has frozen simply because no output appears for several minutes.

Expected full-scan checkpoint:

```text
Conversations scanned:         1,000,000
User-assistant pairs:          2,015,645
Eligible pairs:                1,265,057
Raw relevance candidates:          1,802
Unique candidate sources:          1,633
```

Frozen weak_v5 detects:

```text
56 weak-positive examples
```

Positive manual audit:

```text
Reviewed:                    56
Supported positives:         16
Rejected false positives:    40
Audited positive precision: ~28.6%
```

Category audit:

```text
Directive advice:          49 detected / 15 valid
Overconfident judgement:    6 detected /  1 valid
Sycophantic validation:     1 detected /  0 valid
```

Negative audit:

```text
Deterministic sample constructed: 200
First 100 reviewed:                100
Confirmed negative:                99
Missed positive:                    1
```

The 100-row negative audit is a diagnostic check, not an exhaustive recall estimate.

#### Exact reproduction and the manual audits

For exact reproduction, preserve the recorded audit judgements in Notebook 06.

The notebook contains the original positive-audit and negative-audit decisions used to construct the final target. Running the notebook reconstructs the local audit files from those recorded decisions.

If you independently change those judgements, you are performing an independent replication and should expect the downstream target and metrics potentially to change.

Expected final target:

```text
1,633 total sources
17 positive
1,616 negative
Positive prevalence: ~1.04%

Sycophantic validation:      0
Overconfident judgement:     1
Directive advice:           16
```

This must be described as an:

```text
audit-corrected weak-label target
```

It is not exhaustive human-annotated ground truth.

Expected local files include:

```text
data/samples/lmsys_full_weak_v5_positive_audit_56.csv
data/samples/lmsys_full_weak_v5_negative_audit_100.csv
data/samples/lmsys_full_audit_corrected_target_1633.csv
data/samples/lmsys_scaled_noise_preprocess_v1_matrix_52256.csv
```

Scaled matrix checkpoint:

```text
1,633 × 8 × 4 = 52,256 rows
17 × 32 = 544 positive-condition rows
```

Expected clean scaled pooled metrics:

```text
ROC-AUC:             0.8628
Average Precision:   0.1233
```

Expected clean mean-fold metrics:

```text
ROC-AUC:             0.8729
Average Precision:   0.1974
```

Expected default-threshold confusion totals:

```text
TN: 1616
FP:    0
FN:   17
TP:    0
```

Mean pooled metrics across noise conditions:

```text
none:         ROC-AUC 0.8606   AP 0.1234
minimal:      ROC-AUC 0.8606   AP 0.1234
noise_aware:  ROC-AUC 0.8631   AP 0.1238
aggressive:   ROC-AUC 0.6677   AP 0.0175
```

The defensible conclusion is that `none`, `minimal`, and `noise_aware` perform broadly similarly, while `aggressive` preprocessing causes substantially greater degradation.

Do not interpret the tiny numerical advantage of `noise_aware` as proof of improvement.

Worst noise-only condition:

```text
mixed
ROC-AUC 0.8520
AP      0.1119
```

Worst overall condition:

```text
whitespace + aggressive
ROC-AUC 0.6467
AP      0.0163
```

Positive score sensitivity:

```text
Mean non-aggressive absolute shift: ~0.0098
Mean aggressive absolute shift:     ~0.1973
Aggressive/non-aggressive ratio:     ~20.17×
```

Directive cue retention:

```text
none:         109 / 112 = 97.3%
minimal:      109 / 112 = 97.3%
noise_aware:  110 / 112 = 98.2%
aggressive:     0 / 112 = 0.0%
```

### Notebook 07 — Final synthesis

```text
notebooks/07_final_synthesis.ipynb
```

This notebook does not train a new model. It consolidates the committed pilot and scaled result tables.

Expected final integrity output:

```text
Core tables present: 11/11
Core figures present: 8/8
Scaled OOF rows: 52256
Scaled unique sources: 1633
Scaled conditions: 32
Scaled positives: 17
Clean pooled ROC-AUC: 0.8628
Clean pooled AP: 0.1233
Clean mean-fold ROC-AUC: 0.8729
Aggressive cue retention: 0.0%

NOTEBOOK 07 FINAL INTEGRITY CHECK PASSED.
```

---

## 10. Automated verification

### Quick verification

This requires only the repository's committed files:

```bash
python scripts/verify_reproduction.py --quick
```

It checks:

- pinned dataset revision;
- random seed;
- frozen filter/labelling/noise/preprocessing versions;
- pilot clean ranking metrics;
- 32-condition scaled summary;
- scaled source and positive counts;
- clean scaled ranking metrics;
- 52,256 OOF rows;
- directive cue-retention result;
- final synthesis table sizes.

Successful output ends with:

```text
REPRODUCTION VERIFICATION PASSED
```

### Full verification

After notebooks 01–07 have been reproduced locally:

```bash
python scripts/verify_reproduction.py --full
```

Full mode additionally checks:

- `pip check`;
- all 125 exact package versions in `requirements-lock.txt`;
- all required Git-ignored intermediate CSVs;
- manual validation totals;
- positive and negative audit totals;
- final audit-corrected target composition;
- scaled matrix dimensions and positive-condition count.

---

## 11. Frozen experimental configuration

Do not change these values when reproducing the reported experiment:

```text
Dataset revision:
200748d9d3cddcc9d782887541057aca0b18c5da

Relevance filtering:    relevance_v1
Weak labelling:         disempowerment_weak_v5
Noise generation:       noise_v1
Preprocessing:          preprocess_v1
Random seed:            42
Classifier:             LinearSVC
Features:               TF-IDF word unigrams + bigrams
Cross-validation:       3-fold source-level StratifiedKFold
```

Changing these creates a new experiment rather than an exact reproduction.

---

## 12. Local files and Git policy

Dataset-derived conversation data and local intermediate files are intentionally not committed.

Local files are written under:

```text
data/samples/
```

Committed research outputs are stored under:

```text
results/tables/
results/figures/
```

Do not commit:

- Hugging Face access tokens;
- `.venv/`;
- `.cleanroom_runs/`;
- raw LMSYS conversation data;
- local dataset samples;
- temporary logs containing conversation text.

The repository commits metrics, identifiers, scores, figures, and diagnostic summaries rather than the raw LMSYS dataset.

---

## 13. Troubleshooting

### `ModuleNotFoundError`

If a notebook says a package such as `pandas`, `sklearn`, or `scipy` is missing:

1. confirm the terminal prompt begins with `(.venv)`;
2. confirm the notebook kernel is `.venv`;
3. rerun:

```bash
pip install -r requirements-lock.txt
```

### Packages work in the terminal but not the notebook

The wrong Jupyter kernel is probably selected.

Choose the `.venv` interpreter in the notebook's **Select Kernel** menu and restart the kernel.

### Hugging Face authentication error

Run:

```bash
hf auth login
```

and confirm that the same account has accepted the LMSYS-Chat-1M access terms.

### A streaming cell appears stuck

The full dataset scan can run for many minutes without intermediate output.

The original 1M scan took about 25 minutes in Codespaces.

### `FileNotFoundError`

Later notebooks depend on local CSVs generated by earlier notebooks.

Confirm that notebooks were run in numerical order and that the preceding export cells completed successfully.

### Numbers do not match

Check, in this order:

1. dataset revision;
2. locked Python environment;
3. random seed `42`;
4. frozen filter/labelling/noise/preprocessing versions;
5. recorded manual decisions;
6. source-level 3-fold cross-validation;
7. notebook execution order.

If row counts differ before modelling, resolve that discrepancy before interpreting later metrics.

---

## 14. Exact reproduction vs independent replication

### Exact reproduction

Use the manual relevance and audit decisions recorded in the notebooks.

Goal: reconstruct the original experimental target and results.

### Independent replication

Repeat the human judgements yourself.

Goal: test whether an independent reviewer reaches similar decisions.

A different human judgement can legitimately produce a different target and therefore different downstream results. That should be reported as a replication difference, not automatically as a software failure.

---

## 15. Interpretation boundaries

The final scaled target is highly imbalanced:

```text
17 positive / 1,633 total
```

At the default LinearSVC decision threshold, the scaled model predicts no positive examples.

Therefore:

- accuracy alone is misleading;
- thresholded precision, recall, and F1 are not the primary robustness evidence;
- ROC-AUC, Average Precision, decision-score shifts, and cue-retention diagnostics are more informative here.

The final target is an **audit-corrected weak-label target**, not exhaustive human ground truth.

Final positive-category composition:

```text
Sycophantic validation:      0
Overconfident judgement:     1
Directive advice:           16
```

The strongest RQ3 linguistic-cue evidence therefore concerns **directive advice**.

The experiment provides evidence that aggressive preprocessing is associated with loss of tracked linguistic cues and reduced classifier discrimination under this pipeline. The cue-retention analysis provides a plausible observable mechanism, but does not establish general causal effects across all NLP systems, datasets, or forms of disempowerment.

---

## 16. Main finding

The central result is not simply that “cleaning helps” or “cleaning hurts”.

> **Preprocessing severity matters.**

Minimal and noise-aware preprocessing largely preserve classifier ranking signal under the tested synthetic noise conditions.

Aggressive normalization removes tracked directive constructions and is associated with much larger negative decision-score shifts and substantial degradation in ranking performance.

This supports treating preprocessing as an experimental design choice rather than a neutral preparation step.

---

## 17. Repository structure

```text
irp-disempowerment-nlp/
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_filtering_and_sampling.ipynb
│   ├── 03_labelling_pilot.ipynb
│   ├── 04_noise_and_preprocessing.ipynb
│   ├── 05_model_training_evaluation.ipynb
│   ├── 06_scaled_experiment.ipynb
│   └── 07_final_synthesis.ipynb
│
├── scripts/
│   ├── run_cleanroom.py
│   └── verify_reproduction.py
│
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── filtering.py
│   ├── labelling.py
│   ├── noise.py
│   └── preprocessing.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── data/
│   └── samples/       # local and Git-ignored
│
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

## 18. Final reproduction checklist

Before declaring success, confirm:

```text
[ ] GitHub Codespace created
[ ] Python 3.12 virtual environment created
[ ] requirements-lock.txt installed
[ ] pip check reports no broken requirements
[ ] Hugging Face authentication succeeds
[ ] pinned LMSYS revision is unchanged
[ ] notebooks 01–07 complete in order
[ ] frozen methodology is unchanged
[ ] random seed is 42
[ ] manual relevance total is 108 / 200
[ ] positive audit is 16 valid / 56
[ ] negative audit is 1 missed positive / 100
[ ] final scaled target has 1,633 sources
[ ] final target has 17 positives
[ ] scaled matrix has 52,256 rows
[ ] clean pooled ROC-AUC rounds to 0.8628
[ ] clean pooled AP rounds to 0.1233
[ ] clean mean-fold ROC-AUC rounds to 0.8729
[ ] aggressive tracked directive cue retention is 0.0%
[ ] Notebook 07 integrity check passes
[ ] scripts/verify_reproduction.py --full passes
```

If all checks pass, the artefact has been successfully reproduced under the documented clean-room workflow.
