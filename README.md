# IRP Disempowerment NLP — Technical Artefact

This repository contains the reproducible implementation for the MSc Independent Research Project:

**Data-Centric AI Safety: Evaluating Text Cleaning Techniques for Robust Detection of Disempowerment Patterns in Human-AI Conversations**

## Current implementation status

Implemented:
- project structure;
- central configuration;
- gated Hugging Face streaming loader;
- safe schema inspection;
- extraction of the first user/assistant message pair;
- local small-sample export;
- metadata-only experiment logging.

Not yet implemented:
- relevance filtering;
- weak labelling;
- manual validation;
- noise injection;
- preprocessing variants;
- TF-IDF + SVM;
- statistical tests;
- error analysis;
- DistilBERT extension.

No experimental results are claimed by this repository at this stage.

## Dataset access

The primary dataset is `lmsys/lmsys-chat-1m`.

Before running the loader:
1. Create/sign in to a Hugging Face account.
2. Open the LMSYS-Chat-1M dataset page.
3. Review and accept the dataset licence/access conditions.
4. Authenticate locally with `hf auth login`.

Do not paste or commit a Hugging Face token into source code or notebooks.

## Run

Create a virtual environment, install `requirements.txt`, then either:

```bash
python inspect_dataset.py
```

or open:

```text
notebooks/01_dataset_inspection.ipynb
```

The first inspection requests only 25 streamed examples and saves a flattened local sample to:

```text
data/samples/lmsys_inspection_sample.csv
```

Dataset content is ignored by Git to avoid accidental redistribution.
