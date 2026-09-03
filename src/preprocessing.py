"""Deterministic text preprocessing for robustness experiments."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PREPROCESS_VERSION = "preprocess_v1"

PREPROCESS_CONFIGS = (
    "none",
    "minimal",
    "noise_aware",
    "aggressive",
)


def _coerce_text(text: object) -> str:
    """Safely convert an input value to text."""

    if text is None:
        return ""

    return str(text)


def _normalise_whitespace(text: str) -> str:
    """
    Normalize tabs, line breaks, repeated whitespace, and outer whitespace.
    """

    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------
# Preprocessing configurations
# ---------------------------------------------------------------------


def preprocess_none(
    text: object,
) -> str:
    """Return the input text without preprocessing."""

    return _coerce_text(text)


def preprocess_minimal(
    text: object,
) -> str:
    """
    Apply low-risk formatting normalization.

    Operations:
    - Unicode NFKC normalization;
    - line-break and tab normalization;
    - repeated-whitespace collapse;
    - leading/trailing whitespace removal.
    """

    source = _coerce_text(text)

    source = unicodedata.normalize(
        "NFKC",
        source,
    )

    source = _normalise_whitespace(
        source
    )

    return source


def _normalise_repeated_punctuation(
    text: str,
) -> str:
    """
    Reduce repeated sentence-ending punctuation while preserving
    the punctuation category.

    Examples:
    - "Really???" -> "Really?"
    - "Stop!!!" -> "Stop!"
    - "Okay..." -> "Okay."
    """

    text = re.sub(
        r"\.{2,}",
        ".",
        text,
    )

    text = re.sub(
        r"!{2,}",
        "!",
        text,
    )

    text = re.sub(
        r"\?{2,}",
        "?",
        text,
    )

    return text


def _normalise_punctuation_spacing(
    text: str,
) -> str:
    """Normalize common spacing irregularities around punctuation."""

    # Remove whitespace immediately before punctuation.
    text = re.sub(
        r"\s+([,;:.!?])",
        r"\1",
        text,
    )

    # Add a missing space after comma/semicolon/colon/question/exclamation
    # when alphabetic text immediately follows.
    text = re.sub(
        r"([,;:!?])(?=[A-Za-z])",
        r"\1 ",
        text,
    )

    # Add a missing space after a period only when it appears to separate
    # alphabetic text rather than a decimal number.
    text = re.sub(
        r"(?<=[A-Za-z])\.(?=[A-Za-z])",
        ". ",
        text,
    )

    return text


def _remove_unambiguous_fillers(
    text: str,
) -> str:
    """
    Remove the explicitly defined filler tokens 'um' and 'uh'.

    Adjacent filler punctuation is also removed where appropriate so that
    constructions such as "should, um, speak" become "should speak".

    'well' and 'you know' are deliberately retained because they may be
    legitimate discourse content.
    """

    # Filler surrounded by commas:
    # "should, um, speak" -> "should speak"
    text = re.sub(
        r"(?i),\s*\b(?:um|uh)\b\s*,?",
        " ",
        text,
    )

    # Filler at the beginning or elsewhere with an optional following comma:
    # "Uh, you should..." -> "you should..."
    text = re.sub(
        r"(?i)\b(?:um|uh)\b\s*,?",
        " ",
        text,
    )

    return text


def preprocess_noise_aware(
    text: object,
) -> str:
    """
    Apply preprocessing targeted at several synthetic noise families.

    Operations:
    - minimal preprocessing;
    - lowercasing;
    - repeated-punctuation normalization;
    - punctuation-spacing normalization;
    - removal of unambiguous 'um'/'uh' fillers;
    - final whitespace normalization.

    Automatic spelling correction and recovery of whitespace-joined words
    are intentionally excluded because both can introduce ambiguous edits.
    """

    source = preprocess_minimal(
        text
    )

    source = source.lower()

    source = _normalise_repeated_punctuation(
        source
    )

    source = _normalise_punctuation_spacing(
        source
    )

    source = _remove_unambiguous_fillers(
        source
    )

    source = _normalise_whitespace(
        source
    )

    return source


def preprocess_aggressive(
    text: object,
) -> str:
    """
    Apply stronger traditional NLP cleaning.

    Operations:
    - noise-aware preprocessing;
    - remove punctuation and non-alphanumeric symbols;
    - remove standard English stop words;
    - final whitespace normalization.

    This configuration intentionally risks removing task-relevant linguistic
    cues and is included as an experimental comparison rather than as an
    assumed best preprocessing strategy.
    """

    source = preprocess_noise_aware(
        text
    )

    # Replace punctuation/symbols with spaces while preserving
    # alphanumeric characters and whitespace.
    source = re.sub(
        r"[^a-z0-9\s]",
        " ",
        source,
    )

    source = _normalise_whitespace(
        source
    )

    tokens = re.findall(
        r"\b[a-z0-9]+\b",
        source,
    )

    filtered_tokens = [
        token
        for token in tokens
        if token not in ENGLISH_STOP_WORDS
    ]

    return " ".join(
        filtered_tokens
    )


# ---------------------------------------------------------------------
# Configuration-level interface
# ---------------------------------------------------------------------


_PREPROCESS_FUNCTIONS: Dict[
    str,
    Callable[[object], str],
] = {
    "none": preprocess_none,
    "minimal": preprocess_minimal,
    "noise_aware": preprocess_noise_aware,
    "aggressive": preprocess_aggressive,
}


def apply_preprocessing(
    text: object,
    config: str,
) -> str:
    """Apply one named deterministic preprocessing configuration."""

    if config not in PREPROCESS_CONFIGS:
        raise ValueError(
            f"Unknown preprocessing configuration: {config}"
        )

    return _PREPROCESS_FUNCTIONS[
        config
    ](
        text
    )


def generate_preprocessing_variants(
    text: object,
) -> Dict[str, str]:
    """Generate all preprocessing variants for one input text."""

    return {
        config: apply_preprocessing(
            text,
            config,
        )
        for config in PREPROCESS_CONFIGS
    }