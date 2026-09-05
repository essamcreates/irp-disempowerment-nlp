# Data-Centric AI Safety

## Evaluating Text Cleaning Techniques for Robust Detection of Disempowerment Patterns in Human-AI Conversations

This repository contains the technical artefact developed for an MSc Independent Research Project investigating how text cleaning and preprocessing choices affect the robustness of NLP classifiers for detecting disempowerment-associated patterns in human-AI conversations.

The artefact implements a reproducible Python-based experimental pipeline covering:

- streamed dataset inspection and filtering;
- deterministic sampling;
- weak labelling of disempowerment-associated patterns;
- synthetic textual noise generation;
- multiple preprocessing configurations;
- TF-IDF feature extraction;
- LinearSVC classification;
- source-level cross-validation;
- robustness evaluation;
- decision-score sensitivity analysis;
- linguistic cue-retention analysis;
- pilot-to-scaled experimental synthesis.

The repository supports reproducibility of the technical artefact. Classifier outputs are not presented as diagnoses of harm or as exhaustive ground-truth annotations.

---

## Research questions

**RQ1.** How do text cleaning and preprocessing strategies affect the robustness of NLP classifiers for detecting disempowerment patterns in noisy human-AI conversations?

**RQ2.** Which textual noise types and preprocessing configurations produce the greatest changes in classifier performance across clean, noisy, and cleaned-noisy conversation data?

**RQ3.** How do preprocessing choices influence the preservation or loss of linguistic cues associated with disempowerment, including sycophantic validation, overconfident judgement, and directive advice?

---

## Dataset

The primary dataset is:

```text
lmsys/lmsys-chat-1m
```

The experiments use the pinned dataset revision:

```text
200748d9d3cddcc9d782887541057aca0b18c5da
```

Dataset access requires a Hugging Face account with access to LMSYS-Chat-1M.

Authenticate locally before running the dataset notebooks:

```bash
hf auth login
```

Do not commit Hugging Face access tokens, raw LMSYS conversation data, or local dataset samples.

---

## Environment

The artefact was developed using Python 3.12.

Install the declared dependency ranges with:

```bash
pip install -r requirements.txt
```

For exact reproduction of the final development environment:

```bash
pip install -r requirements-lock.txt
```

The final environment passed:

```bash
python -m pip check
```

All 125 packages in `requirements-lock.txt` matched the working environment.

Key package versions:

```text
Python          3.12.1
NumPy           2.5.2
pandas          2.3.3
SciPy           1.18.0
scikit-learn    1.9.0
matplotlib      3.11.1
```

---

## Notebook execution order

Run the notebooks in numerical order.

### 01 — Dataset inspection

```text
notebooks/01_dataset_inspection.ipynb
```

Inspects streamed LMSYS-Chat-1M examples and validates the conversation structure.

### 02 — Filtering and sampling

```text
notebooks/02_filtering_and_sampling.ipynb
```

Applies the frozen relevance filter, constructs deterministic candidate pools, and creates the pilot sample.

### 03 — Weak-labelling pilot

```text
notebooks/03_labelling_pilot.ipynb
```

Applies weak-labelling rules for:

- sycophantic validation;
- overconfident judgement;
- directive advice.

Frozen weak-label version:

```text
disempowerment_weak_v5
```

### 04 — Noise and preprocessing

```text
notebooks/04_noise_and_preprocessing.ipynb
```

Applies the frozen synthetic noise and preprocessing configurations.

```text
Noise version:          noise_v1
Preprocessing version: preprocess_v1
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

### 05 — Pilot model training and evaluation

```text
notebooks/05_model_training_evaluation.ipynb
```

Runs the pilot modelling experiment using TF-IDF word unigrams/bigrams and `LinearSVC` with deterministic 3-fold source-level cross-validation.

The model is trained on clean, unprocessed training text and evaluated across all noise × preprocessing conditions.

### 06 — Scaled experiment

```text
notebooks/06_scaled_experiment.ipynb
```

Runs the scaled robustness experiment.

Final scaled experimental target:

```text
1,633 sources
17 positive examples
1,616 negative examples
32 noise × preprocessing conditions
52,256 out-of-fold evaluated rows
```

The target is an **audit-corrected weak-label target**, not exhaustive human-annotated ground truth.

The scaled experiment includes:

- source-level cross-validation;
- pooled and mean-fold ranking metrics;
- noise sensitivity analysis;
- preprocessing sensitivity analysis;
- positive decision-score analysis;
- directive linguistic cue-retention analysis.

### 07 — Final synthesis

```text
notebooks/07_final_synthesis.ipynb
```

Consolidates the pilot and scaled experiments without introducing new modelling, labelling, noise, or preprocessing procedures.

It contains:

- pilot-to-scaled comparison;
- RQ1-RQ3 technical synthesis;
- final synthesis figure;
- final artefact integrity checks.

---

## Frozen experimental configuration

```text
Relevance filtering:    relevance_v1
Weak labelling:         disempowerment_weak_v5
Noise generation:       noise_v1
Preprocessing:          preprocess_v1
Random seed:            42
Classifier:             LinearSVC
Features:               TF-IDF word unigrams + bigrams
Cross-validation:       3-fold source-level StratifiedKFold
```

These components should not be silently changed when reproducing the final experiment.

---

## Results

Generated outputs are stored under:

```text
results/tables/
results/figures/
```

The final clean scaled baseline produced:

```text
Pooled ROC-AUC:             0.8628
Pooled Average Precision:   0.1233
Mean-fold ROC-AUC:          0.8729
```

At the default `LinearSVC` decision threshold, the scaled model predicted no positive examples. Hard-threshold precision, recall, and F1 are therefore not used as the primary robustness evidence.

Ranking metrics, decision-score sensitivity, and linguistic cue-retention diagnostics provide the main experimental evidence.

Non-aggressive preprocessing configurations preserved broadly similar ranking performance, whereas aggressive preprocessing caused substantially larger degradation.

For tracked directive constructions, non-aggressive preprocessing retained approximately 97-98% of cue-bearing rows, whereas aggressive preprocessing retained 0%.

These findings should be interpreted in the context of the sparse and highly imbalanced audit-corrected weak-label target.

---

## Local data and Git policy

Dataset-derived conversation text and local sample files are intentionally excluded from version control.

Local samples are stored under:

```text
data/samples/
```

Do not commit:

- Hugging Face access tokens;
- `.venv/`;
- raw LMSYS conversation data;
- local dataset samples;
- temporary logs containing conversation text.

Committed result tables contain experimental metrics, identifiers, scores, and diagnostic summaries rather than redistributed raw conversation datasets.

---

## Repository structure

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
├── src/
│   ├── filtering.py
│   ├── labelling.py
│   ├── noise.py
│   ├── preprocessing.py
│   └── modelling.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── data/
│   └── samples/
│
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

## Reproducibility

For reproduction:

1. install `requirements-lock.txt`;
2. authenticate with Hugging Face;
3. use the pinned LMSYS dataset revision;
4. execute notebooks in numerical order;
5. preserve the frozen experimental components;
6. preserve random seed `42` and the source-level cross-validation procedure.

Some dataset-derived intermediate files are intentionally not committed and must be regenerated by the preceding notebooks.

---

## Scope and limitations

The artefact evaluates the robustness of a baseline NLP classifier under controlled preprocessing and synthetic noise conditions.

Weak labels are heuristic linguistic indicators and should not be interpreted as definitive evidence that a conversation is harmful or disempowering.

The final scaled target is highly imbalanced and contains few positive examples. Category-specific evidence is particularly sparse for sycophantic validation and overconfident judgement. The strongest linguistic cue-retention evidence therefore concerns directive advice.

The purpose of the artefact is to evaluate how data-processing choices affect classifier behaviour, rather than to deploy a production disempowerment-detection system.