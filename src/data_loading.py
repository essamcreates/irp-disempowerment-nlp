"""Dataset loading and safe inspection utilities for LMSYS-Chat-1M."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset

from .config import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    LOGS_DIR,
    PREVIEW_CHARS,
)


def load_lmsys_stream(
    dataset_id: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    revision: str = DATASET_REVISION,
):
    """
    Load LMSYS-Chat-1M as a Hugging Face IterableDataset.

    Authentication is expected to be configured outside the source code
    (for example with `hf auth login`). Do not hard-code access tokens.
    """
    return load_dataset(
        dataset_id,
        split=split,
        streaming=True,
        revision=revision,
    )


def take_examples(dataset: Iterable[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Materialise only the first n streamed examples."""
    if n <= 0:
        raise ValueError("n must be greater than 0")
    return list(islice(dataset, n))


def schema_summary(examples: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Summarise discovered top-level fields without displaying their values.
    This avoids unnecessarily exposing conversation content during schema inspection.
    """
    if not examples:
        return pd.DataFrame(columns=["field", "python_types", "non_null_count"])

    fields = sorted({key for row in examples for key in row.keys()})
    rows = []

    for field in fields:
        values = [row.get(field) for row in examples]
        non_null = [v for v in values if v is not None]
        types = sorted({type(v).__name__ for v in non_null})
        rows.append(
            {
                "field": field,
                "python_types": ", ".join(types) if types else "None",
                "non_null_count": len(non_null),
            }
        )

    return pd.DataFrame(rows)


def _message_role(message: dict[str, Any]) -> str | None:
    """Support common conversational schemas without assuming one in advance."""
    role = message.get("role")
    if role is None:
        role = message.get("from")
    if role is None:
        return None

    role = str(role).strip().lower()
    aliases = {
        "human": "user",
        "prompter": "user",
        "gpt": "assistant",
        "bot": "assistant",
    }
    return aliases.get(role, role)


def _message_text(message: dict[str, Any]) -> str:
    """Extract message text from common field names."""
    for key in ("content", "value", "text"):
        value = message.get(key)
        if value is not None:
            return str(value)
    return ""


def find_conversation_field(example: dict[str, Any]) -> str | None:
    """
    Find a top-level field that looks like a list of message dictionaries.
    Returns None if no suitable field is found.
    """
    preferred = ("conversation", "conversations", "messages", "chat")

    for field in preferred:
        value = example.get(field)
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value[:3])
        ):
            return field

    for field, value in example.items():
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value[:3])
        ):
            roles = {_message_role(item) for item in value[:5]}
            if roles & {"user", "assistant"}:
                return field

    return None


def extract_first_user_assistant_pair(
    example: dict[str, Any],
    conversation_field: str | None = None,
) -> dict[str, Any]:
    """
    Extract the first user message and first assistant response after it.

    The output deliberately excludes conversation IDs and user-identifying metadata.
    """
    field = conversation_field or find_conversation_field(example)
    messages = example.get(field, []) if field else []

    user_text = ""
    assistant_text = ""
    seen_user = False

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue

            role = _message_role(message)
            text = _message_text(message)

            if role == "user" and not seen_user:
                user_text = text
                seen_user = True
            elif role == "assistant" and seen_user:
                assistant_text = text
                break

    return {
        "conversation_field": field,
        "model": example.get("model"),
        "language": example.get("language"),
        "redacted": example.get("redacted"),
        "n_messages": len(messages) if isinstance(messages, list) else None,
        "user_text": user_text,
        "assistant_text": assistant_text,
    }


def build_safe_flat_sample(examples: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Create a local analysis table with only fields needed for the research pipeline.
    Full raw rows and possible user identifiers are not retained here.
    """
    rows = [extract_first_user_assistant_pair(example) for example in examples]
    return pd.DataFrame(rows)


def preview_text(text: Any, max_chars: int = PREVIEW_CHARS) -> str:
    """Return a one-line, truncated preview for interactive inspection."""
    if text is None:
        return ""
    compact = " ".join(str(text).split())
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")


def append_experiment_log(
    event: str,
    details: dict[str, Any],
    log_path: Path | None = None,
) -> Path:
    """Append a metadata-only JSONL record to the experiment log."""
    path = log_path or (LOGS_DIR / "experiment_log.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "details": details,
    }

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return path
