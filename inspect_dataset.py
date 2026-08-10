"""Small, safe LMSYS-Chat-1M inspection run.

Run from the project root:
    python inspect_dataset.py
"""

import time

from src.config import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    INSPECTION_SAMPLE_SIZE,
    SAMPLES_DIR,
)
from src.data_loading import (
    append_experiment_log,
    build_safe_flat_sample,
    load_lmsys_stream,
    preview_text,
    schema_summary,
    take_examples,
)


def main() -> None:
    print(f"Dataset: {DATASET_ID}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Revision: {DATASET_REVISION}")
    print(f"Inspection sample size: {INSPECTION_SAMPLE_SIZE}")

    stream = load_lmsys_stream()
    examples = take_examples(stream, INSPECTION_SAMPLE_SIZE)

    print("\nDiscovered top-level schema:")
    print(schema_summary(examples).to_string(index=False))

    safe_df = build_safe_flat_sample(examples)

    if safe_df.empty:
        raise RuntimeError("No examples were returned from the dataset stream.")

    print("\nConversation field candidates found:")
    print(safe_df["conversation_field"].value_counts(dropna=False).to_string())

    print("\nFirst 3 local previews (truncated):")
    for idx, row in safe_df.head(3).iterrows():
        print(f"\nExample {idx + 1}")
        print("User:", preview_text(row["user_text"]))
        print("Assistant:", preview_text(row["assistant_text"]))

    output_path = SAMPLES_DIR / "lmsys_inspection_sample.csv"
    safe_df.to_csv(output_path, index=False)

    log_path = append_experiment_log(
        event="dataset_inspection",
        details={
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "sample_size_requested": INSPECTION_SAMPLE_SIZE,
            "sample_size_loaded": len(examples),
            "conversation_fields_found": sorted(
                {
                    str(v)
                    for v in safe_df["conversation_field"].dropna().unique().tolist()
                }
            ),
            "saved_sample": str(output_path),
            "results_claimed": False,
        },
    )

    print(f"\nLocal sample saved to: {output_path}")
    print(f"Experiment log updated: {log_path}")
    print("\nNo modelling or performance experiment has been run.")


    time.sleep(1)


if __name__ == "__main__":
    main()

