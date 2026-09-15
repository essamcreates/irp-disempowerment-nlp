#!/usr/bin/env python3
"""Create decision-only audit provenance and remove tracked LMSYS text files.

This script is intentionally narrow: it strips dataset-derived conversation text and
other textual evidence from the four audit/validation CSVs that were temporarily
committed while hardening clean-room reproducibility. The resulting provenance files
contain only non-text identifiers and human decisions required for exact software
reproduction.

It is idempotent. Once the repository is sanitized, rerunning it only validates the
safe provenance files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "samples"
PROVENANCE_DIR = ROOT / "provenance"

SPECS = {
    "lmsys_relevance_manual_validation_200.csv": {
        "safe_name": "manual_relevance_decisions_200.csv",
        "columns": [
            "source_index",
            "pair_index",
            "manual_relevant",
            "manual_category",
        ],
        "rows": 200,
    },
    "lmsys_scaled_weak_v5_positive_audit_34.csv": {
        "safe_name": "scaled_positive_audit_decisions_34.csv",
        "columns": [
            "audit_number",
            "source_index",
            "pair_index",
            "weak_labels",
            "manual_audit_valid",
        ],
        "rows": 34,
    },
    "lmsys_full_weak_v5_positive_audit_56.csv": {
        "safe_name": "full_positive_audit_decisions_56.csv",
        "columns": [
            "audit_number",
            "source_index",
            "pair_index",
            "weak_labels",
            "manual_audit_valid",
        ],
        "rows": 56,
    },
    "lmsys_full_weak_v5_negative_audit_100.csv": {
        "safe_name": "full_negative_audit_decisions_100.csv",
        "columns": [
            "audit_number",
            "source_index",
            "pair_index",
            "manual_missed_positive",
            "manual_missed_labels",
        ],
        "rows": 100,
    },
}

FORBIDDEN_SAFE_COLUMNS = {
    "user_text",
    "assistant_text",
    "sycophantic_evidence",
    "overconfident_evidence",
    "directive_evidence",
    "manual_notes",
    "manual_audit_note",
}


def validate_safe(path: Path, spec: dict) -> None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_columns = spec["columns"]
    if list(df.columns) != expected_columns:
        raise RuntimeError(
            f"Unsafe/unexpected provenance schema for {path}: {list(df.columns)}"
        )
    if len(df) != spec["rows"]:
        raise RuntimeError(
            f"Unexpected row count for {path}: {len(df)} != {spec['rows']}"
        )
    forbidden = FORBIDDEN_SAFE_COLUMNS.intersection(df.columns)
    if forbidden:
        raise RuntimeError(f"Forbidden text columns in {path}: {sorted(forbidden)}")
    key_columns = [c for c in ("source_index", "pair_index") if c in df.columns]
    if key_columns and df[key_columns].duplicated().any():
        raise RuntimeError(f"Duplicate provenance identifiers in {path}")


def main() -> int:
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    raw_present = [name for name in SPECS if (DATA_DIR / name).exists()]
    safe_present = [
        spec["safe_name"]
        for spec in SPECS.values()
        if (PROVENANCE_DIR / spec["safe_name"]).exists()
    ]

    if not raw_present:
        if len(safe_present) != len(SPECS):
            raise RuntimeError(
                "Raw audit files are absent but the complete decision-only provenance "
                "set is not present."
            )
        for spec in SPECS.values():
            validate_safe(PROVENANCE_DIR / spec["safe_name"], spec)
        print("Audit provenance is already sanitized and validated.")
        return 0

    missing_raw = [name for name in SPECS if not (DATA_DIR / name).exists()]
    if missing_raw:
        raise RuntimeError(
            "Refusing partial sanitization; missing raw files: " + ", ".join(missing_raw)
        )

    for raw_name, spec in SPECS.items():
        raw_path = DATA_DIR / raw_name
        raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        missing_columns = [c for c in spec["columns"] if c not in raw_df.columns]
        if missing_columns:
            raise RuntimeError(
                f"Required decision columns missing from {raw_path}: {missing_columns}"
            )
        if len(raw_df) != spec["rows"]:
            raise RuntimeError(
                f"Unexpected row count for {raw_path}: {len(raw_df)} != {spec['rows']}"
            )

        safe_df = raw_df[spec["columns"]].copy()
        safe_path = PROVENANCE_DIR / spec["safe_name"]
        safe_df.to_csv(safe_path, index=False)
        validate_safe(safe_path, spec)
        raw_path.unlink()
        print(f"Sanitized: {raw_name} -> provenance/{spec['safe_name']}")

    print("All tracked dataset-derived audit CSVs were removed from the working tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
